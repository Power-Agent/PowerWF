from __future__ import annotations

import hashlib
import json
import asyncio
import selectors
import threading
from pathlib import Path

from fastapi.testclient import TestClient

from b2g_agent.collaboration import SessionStore
from b2g_agent.contracts import PhysicsEvidence, ReceiverFeedback
from b2g_agent.orchestration import B2GPipeline
from b2g_agent.web.app import create_app


REPO_ROOT = Path(__file__).resolve().parents[1]


def _offline_asgi_loop():
    """ASGITransport uses no network; wake its loop without Windows TCP socketpair.

    Only the in-process test transport uses this loop. Real file-descriptor I/O
    is rejected, so tests cannot silently reach an API or start an HTTP server.
    """
    wake = threading.Event()

    class MemorySelector(selectors.SelectSelector):
        def select(self, timeout=None):
            if self.get_map():
                raise AssertionError("Offline ASGI tests must not register network I/O")
            wake.wait(timeout)
            wake.clear()
            return []

    class MemoryLoop(asyncio.SelectorEventLoop):
        def _make_self_pipe(self):
            pass

        def _close_self_pipe(self):
            pass

        def _write_to_self(self):
            wake.set()

    return MemoryLoop(selector=MemorySelector())


def make_client(tmp_path: Path) -> TestClient:
    pipeline = B2GPipeline.default(repo_root=REPO_ROOT)
    store = SessionStore(pipeline=pipeline, output_root=tmp_path / "sessions", repo_root=REPO_ROOT)
    return TestClient(create_app(store), backend_options={"loop_factory": _offline_asgi_loop})


class _FakeTaskPhysicsExecutor:
    def run(self, specification, plan, *, run_root):  # type: ignore[no-untyped-def]
        requested = float(specification.fields["target_power"].value)
        evidence_path = run_root / specification.episode_id / "evidence_package.json"
        evidence_path.parent.mkdir(parents=True, exist_ok=True)
        evidence_path.write_text('{"status":"validated_full"}\n', encoding="utf-8")
        plan_hash = hashlib.sha256(
            json.dumps(
                plan.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return (
            ReceiverFeedback(
                decision="accepted",
                text="I accept the request after the local physical checks passed.",
                committed_power_kw=requested,
                completion_percentage=100.0,
                completion_basis="The validated service equals the requested service.",
            ),
            PhysicsEvidence(
                workflow="closed_loop",
                status="validated_full",
                evidence_path=str(evidence_path),
                evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
                receiver_decision="accepted",
                committed_power_kw=requested,
                completion_percentage=100.0,
                replan_count=1,
                converged=True,
                hard_constraints_passed=True,
                business_metrics={
                    "delivery_ratio": 1.0,
                    "requested_output_values": {
                        name: 0.0 for name in specification.requested_outputs
                    },
                },
                input_hashes={"task_specification": "b" * 64},
                execution_plan_id=plan.plan_id,
                execution_plan_sha256=plan_hash,
                execution_log={
                    "time_alignment_passed": True,
                    "bem_bus_mapping_passed": True,
                },
            ),
        )


def test_case_and_episode_catalog(tmp_path: Path) -> None:
    api = make_client(tmp_path)
    case = api.get("/api/case")
    episodes = api.get("/api/episodes?role=distribution_grid_operator&limit=200")
    assert case.status_code == 200
    assert case.json()["episode_count"] == 200
    assert episodes.status_code == 200
    assert episodes.json()["count"] > 0
    assert "ground_truth" not in episodes.json()["episodes"][0]


def test_benchmark_session_exposes_four_states_and_allows_only_one_turn(tmp_path: Path) -> None:
    api = make_client(tmp_path)
    episodes = api.get("/api/episodes?role=distribution_grid_operator&limit=200").json()["episodes"]
    episode = next(item for item in episodes if item["difficulty"]["communication"] == "direct")
    created = api.post(
        "/api/sessions",
        json={
            "role": episode["start_role"],
            "mode": "benchmark",
            "episode_id": episode["episode_id"],
        },
    )
    assert created.status_code == 200
    session_id = created.json()["session_id"]
    response = api.post(f"/api/sessions/{session_id}/messages", json={"text": episode["sender_message"]})
    assert response.status_code == 200
    payload = response.json()
    assert payload["result"]["task_specification"]["episode_id"] == episode["episode_id"]
    assert payload["result"]["reviewer_decision"]["verdict"] in {"pass", "revise", "abstain"}
    exchange = payload["result"]["information_exchange"]
    assert exchange["sender_sent"] == episode["sender_message"]
    assert exchange["receiver_received"]
    assert exchange["receiver_sent"] is None
    assert exchange["sender_received"] is None
    assert {message["speaker"] for message in payload["snapshot"]["messages"]} >= {
        "sender", "agent_to_receiver"
    }

    feedback = api.post(
        f"/api/sessions/{session_id}/feedback",
        json={
            "decision": "accepted",
            "text": "I accept every requested term after checking the available evidence.",
            "completion_percentage": 100,
            "completion_basis": "Every requested term was accepted without modification.",
        },
    )
    assert feedback.status_code == 200
    feedback_result = feedback.json()["result"]
    assert feedback_result["information_exchange"]["decision"] == "pending_physics"
    assert feedback_result["information_exchange"]["completion_percentage"] is None
    assert feedback_result["proposed_receiver_feedback"]["completion_percentage"] == 100

    duplicate = api.post(
        f"/api/sessions/{session_id}/messages",
        json={"text": episode["sender_message"]},
    )
    assert duplicate.status_code == 422


def test_interactive_session_accepts_free_form_multi_turn_messages(tmp_path: Path) -> None:
    pipeline = B2GPipeline.default(repo_root=REPO_ROOT)
    pipeline.physics_executor = _FakeTaskPhysicsExecutor()  # type: ignore[assignment]
    store = SessionStore(
        pipeline=pipeline,
        output_root=tmp_path / "sessions",
        repo_root=REPO_ROOT,
    )
    api = TestClient(create_app(store), backend_options={"loop_factory": _offline_asgi_loop})
    created = api.post(
        "/api/sessions",
        json={
            "role": "distribution_grid_operator",
            "mode": "interactive",
        },
    )
    assert created.status_code == 200
    assert created.json()["episode"] is None
    session_id = created.json()["session_id"]

    first = api.post(
        f"/api/sessions/{session_id}/messages",
        json={
            "sender_role": "distribution_grid_operator",
            "text": (
                "On the winter peak 01 operating day, ask BEM 01 at bus 3 to provide 80 kW of peak "
                "shaving during 16:00-18:00."
            ),
        },
    )
    assert first.status_code == 200
    assert first.json()["result"]["turn_index"] == 1
    assert first.json()["result"]["information_exchange"]["decision"] == "pending_physics"

    closed = api.post(
        f"/api/sessions/{session_id}/feedback",
        json={
            "decision": "accepted",
            "text": "I can deliver the full request while respecting comfort limits.",
            "completion_percentage": 100,
            "completion_basis": "All requested service is committed.",
        },
    )
    assert closed.status_code == 200
    assert closed.json()["result"]["information_exchange"]["decision"] == "pending_physics"
    assert closed.json()["result"]["proposed_receiver_feedback"] is not None

    premature = api.post(
        f"/api/sessions/{session_id}/messages",
        json={
            "sender_role": "building_energy_manager",
            "text": "For the next turn, reduce it to 60 kW during 17:00-18:00.",
        },
    )
    assert premature.status_code == 422

    executed = api.post(f"/api/sessions/{session_id}/execute", json={})
    assert executed.status_code == 200
    assert executed.json()["result"]["information_exchange"]["decision"] == "accepted"

    second = api.post(
        f"/api/sessions/{session_id}/messages",
        json={
            "sender_role": "building_energy_manager",
            "text": "For the next turn, reduce it to 60 kW during 17:00-18:00.",
        },
    )
    assert second.status_code == 200
    payload = second.json()
    assert payload["result"]["turn_index"] == 2
    assert payload["result"]["task_specification"]["sender_role"] == "building_energy_manager"
    assert payload["result"]["task_specification"]["target_bems"] == ["bem_01"]
    assert payload["result"]["task_specification"]["target_buses"] == [3]
    assert len(payload["snapshot"]["results"]) == 2
    assert payload["snapshot"]["results"][0]["episode_id"] != payload["snapshot"]["results"][1]["episode_id"]

    final = api.post(f"/api/sessions/{session_id}/finalize")
    assert final.status_code == 200
    assert final.json()["turn_count"] == 2
    assert "## Turn 1" in final.json()["report_markdown"]
    assert "## Turn 2" in final.json()["report_markdown"]


def test_pending_turn_can_execute_physics_and_store_evidence(tmp_path: Path) -> None:
    pipeline = B2GPipeline.default(repo_root=REPO_ROOT)
    pipeline.physics_executor = _FakeTaskPhysicsExecutor()  # type: ignore[assignment]
    store = SessionStore(
        pipeline=pipeline,
        output_root=tmp_path / "sessions",
        repo_root=REPO_ROOT,
    )
    api = TestClient(create_app(store), backend_options={"loop_factory": _offline_asgi_loop})
    episode = next(
        item
        for item in api.get(
            "/api/episodes?role=distribution_grid_operator&limit=200"
        ).json()["episodes"]
        if item["difficulty"]["communication"] == "direct"
    )
    created = api.post(
        "/api/sessions",
        json={
            "role": episode["start_role"],
            "mode": "benchmark",
            "episode_id": episode["episode_id"],
        },
    ).json()
    session_id = created["session_id"]
    pending = api.post(
        f"/api/sessions/{session_id}/messages",
        json={"text": episode["sender_message"]},
    )
    assert pending.status_code == 200

    executed = api.post(
        f"/api/sessions/{session_id}/execute",
        json={},
    )
    assert executed.status_code == 200
    result = executed.json()["result"]
    assert result["information_exchange"]["decision"] == "accepted"
    assert result["information_exchange"]["completion_percentage"] == 100.0
    assert result["physics_evidence"]["status"] == "validated_full"
    assert result["reviewer_decision"]["routing_decision"] == "release"
    assert {item["speaker"] for item in executed.json()["snapshot"]["messages"]} >= {
        "receiver",
        "agent_to_sender",
    }
