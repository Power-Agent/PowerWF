from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from b2g_agent.agents import ActorAgent, ReviewerAgent, TranslatorAgent
from b2g_agent.agents.reviewer_validation import ALL_SIGNATURES, validate_controlled_failure
from b2g_agent.case_io import CaseBundle
from b2g_agent.contracts import PhysicsEvidence, ReceiverFeedback
from b2g_agent.orchestration import B2GPipeline


REPO_ROOT = Path(__file__).resolve().parents[1]
MEMORY_ROOT = REPO_ROOT / "cases" / "ieee33_10bem_15min" / "reviewer_memory"


def test_failure_memory_is_rich_balanced_and_test_clean() -> None:
    records = [
        json.loads(line)
        for line in (MEMORY_ROOT / "verified_seed.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 150
    assert {item["failure_signature"] for item in records} == set(ALL_SIGNATURES)
    assert Counter(item["failure_channel"] for item in records) == {
        "semantic": 70,
        "evidence": 80,
    }
    assert {item["source_split"] for item in records} == {"development", "memory_seed"}
    assert all(item["status"] == "programmatically_verified_seed" for item in records)
    assert all(item["verification_evidence"]["detected_signature"] == item["failure_signature"] for item in records)


def test_independent_failure_validator_rejects_a_clean_artifact() -> None:
    assert validate_controlled_failure(
        "E-NONCONVERGENCE", {"converged": True}
    ) is False


def test_reviewer_checks_physics_evidence_channel() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = next(item for item in case.iter_episodes() if item.difficulty.communication == "direct")
    specification = TranslatorAgent(REPO_ROOT / "knowledge").translate(
        episode.sender_message, episode=episode, sender=episode.start_role
    )
    plan = ActorAgent().plan(specification)
    assert plan is not None
    evidence = PhysicsEvidence(
        workflow="closed_loop",
        status="validated_partial",
        evidence_path="missing-controlled-evidence.json",
        evidence_sha256="0" * 64,
        receiver_decision="modified",
        committed_power_kw=1.0,
        completion_percentage=1.0,
        replan_count=1,
        converged=False,
        hard_constraints_passed=False,
        business_metrics={},
        input_hashes={},
    )
    decision = ReviewerAgent(MEMORY_ROOT / "verified_seed.jsonl").review(
        specification,
        plan,
        simulation_status="completed",
        physics_evidence=evidence,
        require_physics_evidence=True,
    )
    codes = {issue.code for issue in decision.issues}
    assert {"E-NONCONVERGENCE", "E-HARD-CONSTRAINT-VIOLATION", "E-MISSING-OUTPUT"} <= codes
    assert decision.routing_decision == "actor"


def test_online_reflexion_keeps_test_split_locked(tmp_path: Path) -> None:
    reviewer = ReviewerAgent(tmp_path)
    record = {
        "failure_signature": "S-UNIT-MISMATCH",
        "failure_channel": "semantic",
        "responsible_agent": "translator",
        "direction": "grid_to_building",
        "task_families": ["peak_shaving"],
        "affected_fields": ["target_power"],
        "trigger": "unit changed",
        "root_cause": "normalization failure",
        "wrong_output": "kW became MW",
        "correction": "restore kW",
        "generalized_rule": "preserve units",
        "applicability_boundary": "unit-bearing values",
    }
    with pytest.raises(ValueError, match="evaluation closes"):
        reviewer.record_verified_failure(
            record,
            confirmation_source="program",
            source_split="test",
        )
    output = reviewer.record_verified_failure(
        record,
        confirmation_source="expert",
        source_split="development",
    )
    assert output.is_file()


def test_manual_feedback_uses_reverse_translator_and_reviewer() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = next(item for item in case.iter_episodes() if item.difficulty.communication == "direct")
    pipeline = B2GPipeline.default(repo_root=REPO_ROOT)
    pending = pipeline.run_contract_stage(
        episode.sender_message,
        episode=episode,
        sender=episode.start_role,
    )
    proposed = pipeline.complete_with_receiver_feedback(
        pending,
        feedback=ReceiverFeedback(
            decision="modified",
            text="We can deliver 75 percent while keeping comfort within limits.",
            committed_power_kw=(pending.information_exchange.requested_power_kw or 1.0) * 0.75,
            completion_percentage=75.0,
            completion_basis="Validated deliverable quantity divided by requested quantity.",
        ),
    )
    assert proposed.information_exchange.decision == "pending_physics"
    assert proposed.proposed_receiver_feedback is not None
    assert proposed.proposed_receiver_feedback.completion_percentage == 75.0
    assert proposed.reverse_transfer is not None
    assert "75%" in proposed.reverse_transfer.translated_message
    assert proposed.reviewer_decision.routing_decision == "actor"
    assert proposed.reviewer_decision.checked_memory_ids


def test_physics_completion_backfills_outputs_and_reviews_reverse_transfer(
    tmp_path: Path,
) -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = next(item for item in case.iter_episodes() if item.difficulty.communication == "direct")
    pipeline = B2GPipeline.default(repo_root=REPO_ROOT)
    pending = pipeline.run_contract_stage(
        episode.sender_message,
        episode=episode,
        sender=episode.start_role,
    )
    requested = pending.information_exchange.requested_power_kw or 1.0
    feedback = ReceiverFeedback(
        decision="accepted",
        text="We accept the full request after validating every physical constraint.",
        committed_power_kw=requested,
        completion_percentage=100.0,
        completion_basis="Validated service equals requested service.",
    )
    values = {name: 0.0 for name in pending.task_specification.requested_outputs}
    evidence_path = tmp_path / "controlled-evidence.json"
    evidence_path.write_text('{"status":"controlled"}\n', encoding="utf-8")
    assert pending.execution_plan is not None
    plan_hash = hashlib.sha256(
        json.dumps(
            pending.execution_plan.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    evidence = PhysicsEvidence(
        workflow="closed_loop",
        status="validated_full",
        evidence_path=str(evidence_path),
        evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
        receiver_decision="accepted",
        committed_power_kw=requested,
        completion_percentage=100.0,
        replan_count=0,
        converged=True,
        hard_constraints_passed=True,
        business_metrics={"requested_output_values": values},
        input_hashes={},
        execution_plan_id=pending.execution_plan.plan_id,
        execution_plan_sha256=plan_hash,
        execution_log={
            "time_alignment_passed": True,
            "bem_bus_mapping_passed": True,
        },
    )
    completed = pipeline.complete_with_physics(
        pending,
        feedback=feedback,
        evidence=evidence,
    )
    assert completed.reviewer_decision.routing_decision == "release"
    assert completed.reviewer_decision.issues == []
    assert all(
        completed.task_specification.fields[name].status.value == "resolved"
        for name in pending.task_specification.requested_outputs
    )
    assert "100% complete" in (completed.information_exchange.sender_received or "")
