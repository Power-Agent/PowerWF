"""The interface observes existing work and keeps personal mode changes isolated."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import threading

from fastapi.testclient import TestClient
from b2g_agent.agents.llm import StructuredLLMClient
from b2g_agent.collaboration import SessionStore
from b2g_agent.observability.journal import current_journal, emit
from b2g_agent.orchestration import B2GPipeline
from b2g_agent.personal_workspace import PersonalSettings, PersonalWorkspace
from b2g_agent.web.app import create_app
from test_web_app import REPO_ROOT, _offline_asgi_loop, make_client


def test_ui_document_versions_its_assets_and_is_not_cached(tmp_path):
    api = make_client(tmp_path)
    response = api.get("/")
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    static = REPO_ROOT / "src" / "b2g_agent" / "web" / "static"
    for filename in ("styles.css", "app.js"):
        digest = hashlib.sha256((static / filename).read_bytes()).hexdigest()[:16]
        assert f"/static/{filename}?v={digest}" in response.text
    assert "__B2G_STYLE_VERSION__" not in response.text
    assert "__B2G_SCRIPT_VERSION__" not in response.text


def test_ui_static_assets_revalidate_and_versioned_urls_work(tmp_path):
    api = make_client(tmp_path)
    for filename in ("styles.css", "app.js"):
        url = re.search(rf'/static/{re.escape(filename)}\?v=[a-f0-9]+', api.get("/").text).group()
        response = api.get(url)
        assert response.status_code == 200
        assert response.headers["cache-control"] == "no-cache"
        cached = api.get(url, headers={"If-None-Match": response.headers["etag"]})
        assert cached.status_code == 304
        assert cached.headers["cache-control"] == "no-cache"


def test_ui_asset_versions_follow_content_without_restarting_server(tmp_path, monkeypatch):
    module = importlib.import_module("b2g_agent.web.app")
    static = tmp_path / "static"
    static.mkdir()
    (static / "index.html").write_bytes((module.STATIC_DIR / "index.html").read_bytes())
    css = static / "styles.css"
    script = static / "app.js"
    css.write_text(":root {color: red}", encoding="utf-8")
    script.write_text("window.uiVersion = 1;", encoding="utf-8")
    monkeypatch.setattr(module, "STATIC_DIR", static)
    api = make_client(tmp_path)
    before = api.get("/").text
    for asset, updated in ((css, ":root {color: tan}"), (script, "window.uiVersion = 2;")):
        original = asset.stat()
        asset.write_text(updated, encoding="utf-8")
        os.utime(asset, ns=(original.st_atime_ns, original.st_mtime_ns))
        assert asset.stat().st_size == original.st_size
        after = api.get("/").text
        extract = lambda html: re.search(rf'/static/{re.escape(asset.name)}\?v=([a-f0-9]+)', html).group(1)
        assert extract(before) != extract(after)
        assert api.get(f"/static/{asset.name}?v={extract(after)}").text == updated


def personal_client(tmp_path: Path):
    private = tmp_path / "personal"
    private.mkdir()
    settings = PersonalSettings(online_memory_enabled=False, maximum_agent_revisions=2)
    (private / "settings.json").write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    client = StructuredLLMClient.from_env(repo_root=REPO_ROOT, mode="offline", settings=settings.llm, record_root=private / "lr")
    pipeline = B2GPipeline.default(repo_root=REPO_ROOT, llm_client=client, runtime_root=private)
    store = SessionStore(pipeline=pipeline, output_root=private / "sessions", repo_root=REPO_ROOT)
    store.workspace = PersonalWorkspace(private, settings)
    app = create_app(store)
    return TestClient(app, backend_options={"loop_factory": _offline_asgi_loop}), store, private


def test_mode_switch_only_affects_new_sessions_and_preserves_other_settings(tmp_path, monkeypatch):
    api, store, private = personal_client(tmp_path)
    old = api.post("/api/sessions", json={"role": "distribution_grid_operator"}).json()
    old_pipeline = store.get(old["session_id"]).pipeline
    monkeypatch.setenv("OPENAI_API_KEY", "sk-ui-test-secret")
    before = json.loads((private / "settings.json").read_text(encoding="utf-8"))

    switched = api.put("/api/ui/settings", json={"llm_mode": "live"})
    assert switched.status_code == 200
    assert switched.json()["llm_mode"] == "live"
    assert "sk-ui-test-secret" not in switched.text
    fresh = api.post("/api/sessions", json={"role": "building_energy_manager"}).json()
    assert fresh["ui"]["llm_mode"] == "live"
    assert api.get(f"/api/sessions/{old['session_id']}").json()["ui"]["llm_mode"] == "offline"
    assert store.get(old["session_id"]).pipeline is old_pipeline
    after = json.loads((private / "settings.json").read_text(encoding="utf-8"))
    assert {k: v for k, v in before.items() if k != "llm_mode"} == {k: v for k, v in after.items() if k != "llm_mode"}

    assert api.put("/api/ui/settings", json={"llm_mode": "offline"}).status_code == 200
    assert store.get(fresh["session_id"]).pipeline.translator.llm.mode == "live"


def test_missing_credential_or_invalid_mode_keeps_settings_and_runtime(tmp_path, monkeypatch):
    api, store, private = personal_client(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    before = (private / "settings.json").read_bytes()
    pipeline = store.pipeline
    assert api.put("/api/ui/settings", json={"llm_mode": "live"}).status_code == 422
    assert api.put("/api/ui/settings", json={"llm_mode": "unexpected"}).status_code == 422
    assert (private / "settings.json").read_bytes() == before
    assert store.pipeline is pipeline


def test_settings_write_failure_keeps_active_runtime(tmp_path, monkeypatch):
    api, store, private = personal_client(tmp_path)
    before = (private / "settings.json").read_bytes()
    pipeline = store.pipeline

    def fail_write(*args, **kwargs):
        raise OSError("read-only test configuration")

    monkeypatch.setattr("b2g_agent.web.interface.durable_write", fail_write)
    assert api.put("/api/ui/settings", json={"llm_mode": "replay"}).status_code == 422
    assert store.pipeline is pipeline
    assert (private / "settings.json").read_bytes() == before


def test_injected_store_has_read_only_interface_settings(tmp_path):
    api = make_client(tmp_path)
    response = api.get("/api/ui/settings")
    assert response.status_code == 200
    assert response.json()["editable"] is False
    assert api.put("/api/ui/settings", json={"llm_mode": "live"}).status_code == 409


def test_progress_is_live_scoped_and_excludes_journal_payloads(tmp_path, monkeypatch):
    api = make_client(tmp_path)
    episode = next(item for item in api.get("/api/episodes?role=distribution_grid_operator").json()["episodes"] if item["difficulty"]["communication"] == "direct")
    created = api.post("/api/sessions", json={"role": episode["start_role"], "mode": "benchmark", "episode_id": episode["episode_id"]}).json()
    session_id = created["session_id"]
    other = api.post("/api/sessions", json={"role": "building_energy_manager"}).json()["session_id"]
    session = api.app.state.sessions.get(session_id)
    original = session.handle_message
    started, release = threading.Event(), threading.Event()

    def slow_message(*args, **kwargs):
        emit("stage.started", stage="translator.translate", inputs={"text": "private-source-sentinel"}, api_key="secret-sentinel")
        started.set()
        assert release.wait(20)
        return original(*args, **kwargs)

    monkeypatch.setattr(session, "handle_message", slow_message)
    url = f"/api/sessions/{session_id}/messages"
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(api.post, url, json={"text": episode["sender_message"]}, headers={"X-UI-Operation-ID": "ui-first"})
        try:
            assert started.wait(20)
            progress = api.get(f"/api/sessions/{session_id}/progress?operation_id=ui-first")
            assert progress.json()["status"] == "running"
            assert any(item["stage"] == "translator.translate" for item in progress.json()["events"])
            assert "private-source-sentinel" not in progress.text
            assert "secret-sentinel" not in progress.text
            assert "journal_path" not in progress.text
            assert api.get(f"/api/sessions/{other}/progress").json()["events"] == []
            assert api.get(f"/api/sessions/{session_id}/progress?operation_id=ui-other").json()["status"] == "waiting"
            assert api.post(url, json={"text": "duplicate"}).status_code == 409
        finally:
            release.set()
        result = future.result(timeout=20)
    assert result.status_code == 200
    progress = api.get(f"/api/sessions/{session_id}/progress").json()
    assert progress["status"] == "completed"
    assert progress["workflow_status"] == result.json()["result"]["workflow_status"]
    assert progress["elapsed_seconds"] >= 0
    assert len(result.json()["snapshot"]["results"]) == 1


def test_progress_failure_and_identifier_validation(tmp_path, monkeypatch):
    api = make_client(tmp_path)
    session_id = api.post("/api/sessions", json={"role": "distribution_grid_operator"}).json()["session_id"]
    session = api.app.state.sessions.get(session_id)

    def fail_message(*args, **kwargs):
        emit("stage.failed", stage="translator.translate", traceback="private-stack-sentinel")
        raise ValueError("Controlled request failure")

    monkeypatch.setattr(session, "handle_message", fail_message)
    assert api.post(f"/api/sessions/{session_id}/messages", json={"text": "request"}).status_code == 422
    progress = api.get(f"/api/sessions/{session_id}/progress")
    assert progress.json()["status"] == "failed"
    assert "private-stack-sentinel" not in progress.text
    assert api.get(f"/api/sessions/{session_id}/progress?operation_id=../invalid").status_code == 422
    assert api.get("/api/sessions/unknown/progress").status_code == 404
    assert api.post(f"/api/sessions/{session_id}/messages", json={"text": "request"}, headers={"X-UI-Operation-ID": "../invalid"}).status_code == 422


def test_progress_waits_for_complete_json_lines(tmp_path):
    api = make_client(tmp_path)
    session_id = api.post("/api/sessions", json={"role": "distribution_grid_operator"}).json()["session_id"]
    session = api.app.state.sessions.get(session_id)
    interface = api.app.state.interface
    with interface.operation(session, "message", "partial-line"):
        journal = current_journal()
        emit("stage.started", stage="actor.plan")
        partial = json.dumps({"event": "stage.completed", "payload": {"stage": "actor.plan", "elapsed_seconds": 1.5}})
        with journal.path.open("a", encoding="utf-8") as stream:
            stream.write(partial)
        first = interface.progress(session_id)
        assert len(first["events"]) == 1
        with journal.path.open("a", encoding="utf-8") as stream:
            stream.write("\n")
        second = interface.progress(session_id)
        assert len(second["events"]) == 2
        assert second["events"][-1]["elapsed_seconds"] == 1.5
