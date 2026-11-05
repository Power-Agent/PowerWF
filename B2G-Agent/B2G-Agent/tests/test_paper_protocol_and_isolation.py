"""Offline checks for the 18-metric protocol and private/formal separation."""
import csv
import importlib.util
import json
import os
from pathlib import Path

import pytest

from b2g_agent.agents.llm import StructuredLLMClient
from b2g_agent.agents.translator import TranslatedFeedback
from b2g_agent.case_io import CaseBundle
from b2g_agent.collaboration import SessionStore
from b2g_agent.contracts import AgentRole
from b2g_agent.evaluation.agent_metrics import NINE_METRICS
from b2g_agent.evaluation.computation_cost import computation_cost
from b2g_agent.evaluation.paper_metrics import QUALITY, AGENTS, COST, write_paper_metrics
from b2g_agent.observability import LLMSettings
from b2g_agent.orchestration import B2GPipeline
from b2g_agent.personal_workspace import PersonalWorkspace
from test_llm_records import _fake_client

ROOT = Path(__file__).resolve().parents[1]


def suite_module():
    spec = importlib.util.spec_from_file_location("suite_for_isolation", ROOT / "experiments/run_proposed_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_eighteen_metrics_and_whole_framework_cost(tmp_path):
    records = [{"row": {"total_tokens": tokens, "llm_latency_seconds": latency},
                "diagnostics": {"translator_retrieval_seconds": retrieval, "reviewer_retrieval_seconds": .25}}
               for tokens, latency, retrieval in [(120, 3, .1), (180, 7, .4)]]
    cost = computation_cost(records)
    assert cost["framework"]["total_token_consumption"]["value"] == 300
    assert cost["framework"]["llm_inference_latency"]["value"] == 5
    assert cost["framework"]["retrieval_time"]["value"] == .5
    summary = {"results": {"all_attempts": {"attempts": 2, "Y": .5,
        "all_attempt_metrics_zero_penalty": {field: {"mean": .5, "std": 0} for _, field in QUALITY.values() if field}},
        "efficiency": cost}, "agent_level": {"metrics": {name: {"value": .5, "denominator": 2} for name in NINE_METRICS}}}
    rows = write_paper_metrics(tmp_path, summary)
    assert [row["metric"] for row in rows] == [*QUALITY, *AGENTS, *COST]
    assert len(list(csv.DictReader((tmp_path / "metrics_18.csv").open(encoding="utf-8")))) == 18
    assert all(row["direction"] == "higher" for row in rows if row["layer"] == "agent_performance")
    records[0]["diagnostics"]["reviewer_retrieval_seconds"] = None
    incomplete = computation_cost(records)["framework"]["retrieval_time"]
    assert incomplete["value"] is None and incomplete["missing_attempts"] == 1


def test_personal_store_owns_outputs_and_ignores_shared_runtime_paths(tmp_path, monkeypatch):
    private = tmp_path / "personal"
    monkeypatch.setenv("B2G_PERSONAL_ROOT", str(private))
    monkeypatch.setenv("B2G_LLM_RECORD_ROOT", str(tmp_path / "do-not-write-formal-lr"))
    monkeypatch.setenv("B2G_SESSION_OUTPUT_ROOT", str(tmp_path / "do-not-write-formal-sessions"))
    before = dict(os.environ)
    seed = ROOT / "cases/ieee33_10bem_15min/reviewer_memory/verified_seed.jsonl"
    original_seed = seed.read_bytes()
    store = SessionStore(repo_root=ROOT)
    assert store.output_root == private / "sessions"
    assert store.pipeline.translator.llm.record_root == private / "lr"
    assert store.pipeline.journal_root == private / "interactions"
    assert store.pipeline.reviewer.online_memory_path == private / "memory/verified_online.jsonl"
    session = store.create(AgentRole.DISTRIBUTION_GRID_OPERATOR)
    result = session.handle_message("I need a little more flexibility tomorrow, but I have not decided the building or quantity yet.")
    assert Path(result.interaction_journal).is_relative_to(private)
    assert (private / "sessions" / session.session_id / "session.json").is_file()
    memory = {"failure_signature": "S-UNIT-MISMATCH", "failure_channel": "semantic",
        "responsible_agent": "translator", "direction": "grid_to_building", "task_families": ["peak_shaving"],
        "affected_fields": ["target_power"], "trigger": "unit mismatch", "root_cause": "controlled fixture",
        "wrong_output": "MW", "correction": "kW", "generalized_rule": "preserve units", "applicability_boundary": "unit-bearing quantities"}
    path = store.pipeline.reviewer.record_verified_failure(memory, confirmation_source="program", source_split="interactive")
    assert path.is_relative_to(private) and seed.read_bytes() == original_seed
    formal = B2GPipeline.default(repo_root=ROOT, formal_evaluation=True)
    assert path not in formal.reviewer._memory_files()
    with pytest.raises(ValueError, match="disabled"):
        formal.reviewer.record_verified_failure(memory, confirmation_source="program", source_split="development")
    assert dict(os.environ) == before
    assert not (tmp_path / "do-not-write-formal-lr").exists()
    assert not (tmp_path / "do-not-write-formal-sessions").exists()
    with pytest.raises(ValueError, match="inside its private workspace"):
        SessionStore(repo_root=ROOT, output_root=tmp_path / "formal")


@pytest.mark.parametrize("suffix", ["outputs/benchmark", "cases", "knowledge", ".local-data", "."])
def test_personal_root_cannot_overlap_formal_inputs_or_results(monkeypatch, suffix):
    monkeypatch.setenv("B2G_PERSONAL_ROOT", str(ROOT / suffix))
    with pytest.raises(ValueError, match="Personal storage"):
        PersonalWorkspace.open(ROOT)


def test_formal_protocol_overrides_personal_environment_without_mutating_it(tmp_path, monkeypatch):
    module = suite_module()
    protocol = module.load_protocol(CaseBundle.default(ROOT))
    monkeypatch.setenv("B2G_PERSONAL_ROOT", str(tmp_path / "private"))
    monkeypatch.setenv("B2G_LLM_MODE", "offline")
    monkeypatch.setenv("B2G_MODEL", "personal-model")
    monkeypatch.setenv("B2G_EVENT_JOURNAL", str(tmp_path / "private.jsonl"))
    before = dict(os.environ)
    suite_root = tmp_path / "formal"
    env = module.protocol_environment(protocol, suite_root, 0)
    assert env["B2G_MODEL"] == "gpt-6-luna" and env["B2G_MAX_OUTPUT_TOKENS"] == "9000"
    assert env["B2G_MAX_TOTAL_TOKENS_PER_TASK"] == "" and env["B2G_ONLINE_MEMORY_ENABLED"] == "false"
    assert "B2G_PERSONAL_ROOT" not in env and "B2G_EVENT_JOURNAL" not in env
    assert Path(env["B2G_LLM_RECORD_ROOT"]).is_relative_to(suite_root)
    cache = Path(env["B2G_SIMULATION_OUTPUT_ROOT"])
    assert cache == module.physics_cache_root(suite_root)
    assert cache != module.physics_cache_root(tmp_path / "private")
    assert not cache.is_relative_to(tmp_path / "private")
    assert dict(os.environ) == before and protocol["episodes"] * protocol["repeats"] == 600
    with pytest.raises(ValueError, match="personal"):
        B2GPipeline.default(repo_root=ROOT, formal_evaluation=True, runtime_root=tmp_path)


def test_llm_latency_measures_request_not_persistence_or_parsing(tmp_path, monkeypatch):
    import b2g_agent.agents.llm as module
    client, completions = _fake_client(json.dumps({"sender_message": "I accept the request."}))
    llm = StructuredLLMClient(mode="live", settings=LLMSettings(model="offline-fixture"),
        record_root=tmp_path / "lr", client=client)
    clock = [10.0]
    create = completions.create
    def timed_create(**kwargs):
        clock[0] += 2.5
        return create(**kwargs)
    monkeypatch.setattr(completions, "create", timed_create)
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock[0])
    llm.call(TranslatedFeedback, system="Offline fixture", user_payload={}, agent_name="translator_feedback")
    manifest = json.loads(next((tmp_path / "lr").glob("*/i/*/manifest.json")).read_text())
    assert manifest["latency_seconds"] == 2.5 and manifest["latency_scope"] == "api_request_wall"
    assert completions.calls[0]["max_completion_tokens"] == 9000
