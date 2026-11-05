import json
from pathlib import Path

import pytest

from b2g_agent.case_io import CaseBundle
from b2g_agent.contracts import PhysicsEvidence, ReceiverFeedback
from b2g_agent.evaluation import (
    physics_cache_key,
    physics_input_bundle_sha256,
    summarize_evaluation_runs,
    task_specification_sha256,
    write_evaluation_summary,
)
from b2g_agent.orchestration import B2GPipeline


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_physics_cache_key_covers_task_code_and_inputs() -> None:
    specification = CaseBundle.default(REPO_ROOT).get_groundtruth(
        "ep-001"
    ).task_specification
    input_bundle = physics_input_bundle_sha256({"case:a": "1" * 64})
    baseline = physics_cache_key(
        specification,
        physics_implementation_bundle="2" * 64,
        physics_input_bundle=input_bundle,
    )
    assert len(baseline) == 64
    assert baseline == physics_cache_key(
        specification,
        physics_implementation_bundle="2" * 64,
        physics_input_bundle=input_bundle,
    )
    assert baseline != physics_cache_key(
        specification,
        physics_implementation_bundle="3" * 64,
        physics_input_bundle=input_bundle,
    )
    changed = specification.model_copy(update={"source_text": specification.source_text + " "})
    assert task_specification_sha256(changed) == task_specification_sha256(specification)
    assert baseline == physics_cache_key(
        changed,
        physics_implementation_bundle="2" * 64,
        physics_input_bundle=input_bundle,
    )
    physical_change = specification.model_copy(
        update={"target_buses": [33, *specification.target_buses[1:]]}
    )
    assert task_specification_sha256(physical_change) != task_specification_sha256(
        specification
    )
    assert baseline != physics_cache_key(
        physical_change,
        physics_implementation_bundle="2" * 64,
        physics_input_bundle=input_bundle,
    )


def test_pipeline_can_complete_from_cached_physics_without_rerunning() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = case.get_episode("ep-001")
    pipeline = B2GPipeline.default(repo_root=REPO_ROOT)
    pending = pipeline.run_contract_stage(
        episode.sender_message,
        episode=episode,
        sender=episode.start_role,
    )
    feedback = ReceiverFeedback(
        decision="accepted",
        text="I accept the request after the frozen physics checks passed.",
        committed_power_kw=60.0,
        completion_percentage=100.0,
        completion_basis="Validated delivered-versus-requested service.",
    )
    evidence = PhysicsEvidence(
        workflow="closed_loop",
        status="validated_full",
        evidence_path="test/evidence_package.json",
        evidence_sha256="a" * 64,
        receiver_decision="accepted",
        committed_power_kw=60.0,
        completion_percentage=100.0,
        replan_count=0,
        converged=True,
        hard_constraints_passed=True,
        business_metrics={},
        input_hashes={},
    )
    completed = pipeline.complete_with_physics(
        pending,
        feedback=feedback,
        evidence=evidence,
    )
    assert completed.information_exchange.decision == "accepted"
    assert completed.physics_evidence == evidence
    assert completed.information_exchange.sender_received


def test_evaluation_summary_aggregates_repeats_failures_and_usage(
    tmp_path: Path,
) -> None:
    run_root = tmp_path / "run"
    (run_root / "scores").mkdir(parents=True)
    rows = [
        {
            "run_id": "ep-001-r01",
            "status": "completed",
            "method_id": "full_multi_agent",
            "episode_id": "ep-001",
            "repeat": 1,
            "elapsed_seconds": 2.0,
            "llm_call_count": 3,
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "physics_executed": True,
            "physics_cache_status": "miss",
        },
        {
            "run_id": "ep-001-r02",
            "status": "completed",
            "method_id": "full_multi_agent",
            "episode_id": "ep-001",
            "repeat": 2,
            "elapsed_seconds": 1.0,
            "llm_call_count": 3,
            "prompt_tokens": 90,
            "completion_tokens": 10,
            "total_tokens": 100,
            "physics_executed": True,
            "physics_cache_status": "hit",
        },
        {
            "run_id": "ep-002-r01",
            "status": "failed",
            "method_id": "full_multi_agent",
            "episode_id": "ep-002",
            "repeat": 1,
            "elapsed_seconds": 0.5,
            "llm_call_count": 1,
            "prompt_tokens": 5,
            "completion_tokens": 0,
            "total_tokens": 5,
            "physics_executed": False,
            "physics_cache_status": "not_required",
        },
    ]
    (run_root / "index.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    for run_id, quality, success in (
        ("ep-001-r01", 90.0, True),
        ("ep-001-r02", 100.0, True),
    ):
        (run_root / "scores" / f"{run_id}.json").write_text(
            json.dumps(
                {
                    "semantic_fidelity": quality / 100.0,
                    "actionability": 1.0,
                    "physical_feasibility": 1.0,
                    "goal_achievement": 1.0,
                    "interaction_quality": quality,
                    "task_success": success,
                    "agent_metrics": {"actor_plan_legal": True},
                }
            ),
            encoding="utf-8",
        )

    summary = summarize_evaluation_runs([run_root])
    method = summary["methods"]["full_multi_agent"]
    assert method["run_count"] == 3
    assert method["failed_count"] == 1
    assert method["Y"] == pytest.approx(2 / 3)
    assert method["Y_among_scored"] == 1.0
    assert method["interaction_quality_all_attempts_mean"] == pytest.approx(190 / 3)
    assert method["score_metrics"]["interaction_quality"]["mean"] == 95.0
    assert method["score_metrics"]["interaction_quality"][
        "within_episode_sample_std_mean"
    ] == pytest.approx(7.0710678119)
    assert method["llm_usage"]["total_tokens"] == 225
    assert method["physics"]["cache_hit_count"] == 1
    json_path, csv_path = write_evaluation_summary(summary, tmp_path / "summary")
    assert json_path.is_file()
    assert csv_path.is_file()
