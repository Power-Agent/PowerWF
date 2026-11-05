"""Offline regressions from the failed qualification; never benchmark results."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from b2g_agent.agents.actor import PlanCompiler
from b2g_agent.agents.response_validation import validate_response
from b2g_agent.case_io import CaseBundle
from b2g_agent.contracts import ReceiverFeedback, SemanticReviewResult
from b2g_agent.contracts.readiness import execution_readiness
from b2g_agent.evaluation.physics_cache import task_specification_sha256
from test_correction_workflow import assessment, events, setup_pipeline

ROOT = Path(__file__).resolve().parents[1]
CAPTURED = json.loads((Path(__file__).parent / "fixtures/qualification_reviewer_20261030.json").read_text(encoding="utf-8"))["records"]


def uncertain(count):
    result = assessment()
    result["status"] = "uncertain"
    for check in result["checks"][:count]:
        check.update(score=0.5, responsible_agent="translator", source_excerpt="provide 60 kW",
            structured_field="fields.target_power", basis_locator="fixture:controlled_uncertainty",
            uncertainty_reason="Controlled ambiguity for testing the uncertainty threshold.",
            message="This fixture leaves the cited term uncertain, not proven wrong.")
    return result


@pytest.mark.parametrize("index", [0, 1])
def test_captured_uncertainty_cannot_be_silently_labeled_failure(index):
    raw = deepcopy(CAPTURED[index]["response"])
    with pytest.raises(ValueError, match="status must be uncertain"):
        validate_response(SemanticReviewResult, raw)
    assert raw == CAPTURED[index]["response"]


def test_captured_invalid_review_repairs_reviewer_without_rewriting_contract(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "reviewer" and n == 1:
            return deepcopy(CAPTURED[0]["response"])
        if agent == "reviewer" and n == 2:
            assert "response_validation_error" in user
            # Explicit offline repair double, not a claim about a new model reply.
            return uncertain(2)
    _, pending, llm = setup_pipeline(tmp_path, hook)
    assert llm.counts == {"translator": 1, "actor": 1, "reviewer": 2}
    assert pending.reviewer_decision.routing_decision == "actor"
    assert not pending.reviewer_decision.issues
    assert pending.reviewer_decision.semantic_risk_score == 0.125
    assert any(e["event"] == "reviewer.response_invalid" for e in events(pending))


@pytest.mark.parametrize("count,expected_translations", [(2, 1), (4, 1), (5, 2)])
def test_uncertainty_uses_existing_threshold_and_keeps_findings(tmp_path, count, expected_translations):
    def hook(agent, n, user, payload):
        if agent == "reviewer" and n == 1:
            return uncertain(count)
    _, pending, llm = setup_pipeline(tmp_path, hook)
    assert llm.counts["translator"] == expected_translations
    first = pending.review_history[0]["decision"]
    assert first["semantic_risk_score"] == count * 0.5 / 8
    assert first["semantic_risk_threshold"] == 0.25
    assert sum(c["score"] > 0 for c in first["semantic_checks"]) == count
    assert bool(first["issues"]) == (count > 4)


def test_one_explicit_failure_blocks_despite_low_aggregate_risk(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "reviewer" and n == 1:
            return assessment(True)
    _, pending, llm = setup_pipeline(tmp_path, hook)
    assert llm.counts["translator"] == 2
    assert pending.review_history[0]["decision"]["semantic_risk_score"] == 0.125
    reviews = [u for a, u in llm.calls if a == "reviewer"]
    assert not reviews[0]["review_history"]
    assert reviews[1]["review_history"][0]["decision"]["issues"]
    assert reviews[1]["revision_changes"] is not None
    recorded = [e["payload"]["inputs"] for e in events(pending)
                if e["event"] == "stage.started" and e["payload"].get("stage") == "reviewer.review"]
    assert recorded[1]["review_history"][0]["decision"]["issues"]


@pytest.mark.parametrize("fault", ["owner", "citation", "uncertainty", "issue_check_conflict"])
def test_inconsistent_review_is_rejected_instead_of_inventing_a_correction(fault):
    raw = assessment(True)
    if fault == "owner":
        raw["checks"][0]["responsible_agent"] = "none"
    elif fault == "citation":
        raw["checks"][0]["source_excerpt"] = None
    elif fault == "uncertainty":
        raw = uncertain(1)
        raw["checks"][0]["uncertainty_reason"] = None
    else:
        raw["issues"] = [{"code": raw["checks"][0]["check_id"], "message": "Conflicting owner",
            "responsible_agent": "actor", "score": 1, "correction_required": "Fix"}]
    with pytest.raises(ValueError):
        SemanticReviewResult.model_validate(raw)


def unsupported_clarification():
    raw = deepcopy(CAPTURED[2]["response"])
    raw.update(clarification_verdict="unsupported",
        clarification_basis="The original source specifies 60 kW for peak shaving; the extra question was introduced by Translator.")
    check = next(c for c in raw["checks"] if c["check_id"] == "semantic_decision_completion")
    check.update(score=1, responsible_agent="translator", message="The unnecessary clarification prevents the requested commitment.")
    return raw


def test_captured_clarification_conflict_returns_translator_then_actor(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "translator" and n == 1:
            payload.update(clarification_required=True,
                clarification_question="Does provide 60 kW mean a reduction or another action?")
        if agent == "reviewer" and n == 1:
            return deepcopy(CAPTURED[2]["response"])
        if agent == "reviewer" and n == 2:
            assert "Adjudicate" in user["response_validation_error"]
            return unsupported_clarification()
    pipeline, pending, llm = setup_pipeline(tmp_path, hook)
    assert llm.counts == {"translator": 2, "actor": 1, "reviewer": 3}
    assert pending.workflow_status == "awaiting_physics"
    assert not pending.task_specification.clarification_required
    assert pending.review_history[0]["decision"]["clarification_verdict"] == "unsupported"
    assert "clarification_required" in [u for a, u in llm.calls if a == "reviewer"][-1]["revision_changes"]["changed_fields"]
    # Test double verifies the corrected gate, not real simulator performance.
    result = pipeline.execute_physics(pending, run_root=tmp_path / "physics")
    assert pipeline.physics_executor.calls == 1 and result.workflow_status == "released"


def test_genuine_clarification_still_stops_before_tools(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "translator":
            payload["fields"]["target_power"].update(value=None, status="missing")
            payload.update(clarification_required=True, clarification_question="What power target is required?")
        if agent == "reviewer":
            payload.update(clarification_verdict="required",
                clarification_basis="Controlled missing-target fixture requires sender input.")
    pipeline, pending, llm = setup_pipeline(tmp_path, hook)
    assert pending.workflow_status == "needs_clarification"
    assert pending.reviewer_decision.clarification_verdict == "required"
    assert pipeline.physics_executor.calls == 0 and llm.counts["translator"] == 1


def test_explicit_need_for_sender_input_blocks_even_below_uncertainty_threshold(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "reviewer":
            result = uncertain(1)
            result["checks"][0]["responsible_agent"] = "human"
            return result
    pipeline, pending, _ = setup_pipeline(tmp_path, hook)
    assert pending.reviewer_decision.semantic_risk_score < 0.25
    assert pending.workflow_status == "needs_clarification"
    assert pipeline.physics_executor.calls == 0


def test_unsupported_clarification_exhaustion_is_blocked_not_released(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "translator":
            payload.update(clarification_required=True, clarification_question="Controlled unnecessary question")
        if agent == "reviewer":
            return unsupported_clarification()
    pipeline, pending, _ = setup_pipeline(tmp_path, hook, revisions=4)
    assert pending.workflow_status == "blocked"
    assert pending.reviewer_decision.routing_decision != "clarify_sender"
    assert pending.reviewer_decision.revision_count == 4
    assert pipeline.physics_executor.calls == 0 and len(pending.review_history) == 4


def test_history_survives_physical_and_reverse_revisions_without_cross_task_leak(tmp_path):
    reverse = 0
    def hook(agent, n, user, payload):
        nonlocal reverse
        if agent == "reviewer" and user["simulation_status"] == "completed":
            reverse += 1
            if reverse == 1:
                return assessment(True)
    pipeline, pending, llm = setup_pipeline(tmp_path, hook, failures=1)
    result = pipeline.execute_physics(pending, run_root=tmp_path / "physics")
    reviews = [u for a, u in llm.calls if a == "reviewer"]
    assert all(u["review_history"] and u["revision_changes"] for u in reviews[1:])
    assert result.workflow_status == "released" and len(result.review_history) == 4
    assert any(u["review_history"][-1]["decision"]["routing_decision"] == "actor"
               for u in reviews if u["review_history"] and u["simulation_status"] == "not_run")
    episode = pipeline.case.get_episode("ep-001")
    pipeline.run_contract_stage(episode.sender_message, episode=episode, sender=episode.start_role)
    assert [u for a, u in llm.calls if a == "reviewer"][-1]["review_history"] == []


def test_followup_review_receives_prior_user_context(tmp_path):
    pipeline, pending, llm = setup_pipeline(tmp_path)
    prior = [{"speaker": "sender", "text": pending.task_specification.source_text}]
    pipeline.run_message("Please retain the same requirements and use 50 kW instead.",
        sender=pending.task_specification.sender_role, context=pending.interaction_context,
        prior_specification=pending.task_specification, conversation_history=prior, turn_index=2)
    review = [u for a, u in llm.calls if a == "reviewer"][-1]
    assert review["conversation_history"] == prior
    assert review["review_history"] == []  # A new turn, not another task's audit.


def test_manual_receiver_feedback_also_supplies_review_history(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "reviewer" and n == 2:
            return assessment(True)
    pipeline, pending, llm = setup_pipeline(tmp_path, hook)
    result = pipeline.complete_with_receiver_feedback(pending, feedback=ReceiverFeedback(
        decision="modified", text="I can provide 30 kW, which is half of the request.",
        committed_power_kw=30, completion_percentage=50, completion_basis="Half the service."))
    reviews = [u for a, u in llm.calls if a == "reviewer"]
    assert len(reviews) == 3
    assert reviews[-1]["review_history"][-1]["reviewed_source_text"] == "I can provide 30 kW, which is half of the request."
    assert reviews[-1]["review_history"][-1]["decision"]["issues"]
    assert result.information_exchange.decision == "pending_physics"


def test_provenance_alone_cannot_change_physical_inputs_or_plan_identity(tmp_path):
    _, pending, _ = setup_pipeline(tmp_path)
    with_provenance = pending.task_specification
    legacy = with_provenance.model_copy(deep=True)
    for bound in legacy.constraints:
        bound.origin = None
        bound.evidence = []
    assert task_specification_sha256(with_provenance) == task_specification_sha256(legacy)
    assert PlanCompiler().compile(with_provenance) == PlanCompiler().compile(legacy)


def test_case_policy_provenance_is_public_and_does_not_read_oracles(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail("Agent policy must not read evaluation truth")
    monkeypatch.setattr(CaseBundle, "get_groundtruth", forbidden)
    monkeypatch.setattr(CaseBundle, "iter_groundtruth", forbidden)
    _, pending, llm = setup_pipeline(tmp_path)
    policy = next(u for a, u in llm.calls if a == "reviewer")["public_operating_constraints"]
    assert len(policy) == 4
    for constraint in pending.task_specification.constraints:
        assert constraint.origin == "case_policy"
        evidence = constraint.evidence[0]
        assert "groundtruth" not in evidence.locator
        assert "optimization/joint_dispatch.json" in evidence.locator and len(evidence.sha256) == 64
    comfort = next(c for c in policy if c["metric"] == "comfort_violation_minutes")
    assert comfort["value"] == 0 and "baseline" in comfort["evidence"][0]["excerpt"]


@pytest.mark.parametrize("change", ["drop", "relax", "soften"])
def test_missing_or_relaxed_case_bound_is_returned_and_preflight_blocks(change, tmp_path):
    def corrupt(spec):
        if change == "drop":
            spec.constraints = [c for c in spec.constraints if c.metric != "comfort_violation_minutes"]
        else:
            comfort = next(c for c in spec.constraints if c.metric == "comfort_violation_minutes")
            if change == "relax":
                comfort.value = 15
            else:
                comfort.hard = False
    def hook(agent, n, user, payload):
        if agent == "translator" and n == 1:
            from b2g_agent.contracts import TaskSpecification
            spec = TaskSpecification.model_validate(payload)
            corrupt(spec)
            return spec.model_dump(mode="json")
    pipeline, pending, llm = setup_pipeline(tmp_path, hook)
    assert llm.counts["translator"] == 2
    assert any(i["code"] == "T-CASE-POLICY" for i in pending.review_history[0]["decision"]["issues"])
    corrupt(pending.task_specification)
    assert any(i.code == "T-CASE-POLICY" for i in execution_readiness(pending.task_specification, case=pipeline.case))
    result = pipeline.execute_physics(pending, run_root=tmp_path / "physics")
    assert pipeline.physics_executor.calls == 0 and result.workflow_status != "released"
