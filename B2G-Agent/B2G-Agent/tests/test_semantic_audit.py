"""Offline contract/route tests; captured inputs and synthetic responses are distinguished."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from b2g_agent.agents import ActorAgent, ReviewerAgent, TranslatorAgent
from b2g_agent.agents.llm import _strict_json_schema
from b2g_agent.agents.translator import _conversation_payload
from b2g_agent.case_io import CaseBundle
from b2g_agent.case_io.operating_policy import public_review_context
from b2g_agent.contracts import SemanticReviewResult
from b2g_agent.contracts.semantic_audit import SemanticAuditResult
from b2g_agent.observability import LLMSettings, physics_implementation_bundle_sha256
from b2g_agent.orchestration import B2GPipeline
from reviewer_fixtures import categorical_fixture
from test_correction_workflow import ScriptedLLM, assessment, events, setup_pipeline

ROOT = Path(__file__).resolve().parents[1]
CAPTURED = json.loads((Path(__file__).parent / "fixtures/qualification_reviewer_20261030_second.json").read_text(encoding="utf-8"))["records"]


def clean_audit(phase="sender_instruction"):
    # An explicit test double, not a predicted or observed model response.
    return categorical_fixture(assessment(), phase)


class LiteralReplyClient:
    available = True
    settings = SimpleNamespace(maximum_schema_repairs=2)

    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def call(self, model, *, user_payload, **kwargs):
        self.calls.append(deepcopy(user_payload))
        raw = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        return model.model_validate(deepcopy(raw))


@pytest.mark.parametrize("index", range(5))
def test_captured_numeric_replies_remain_immutable_and_are_not_autoconverted(index):
    raw = deepcopy(CAPTURED[index]["response"])
    historical = SemanticReviewResult.model_validate(raw)
    expected_risk = 0.25 if index == 1 else 0.5
    assert sum(c.score * c.weight for c in historical.checks) / sum(c.weight for c in historical.checks) == expected_risk
    with pytest.raises(ValueError):
        SemanticAuditResult.model_validate(raw)
    assert raw == CAPTURED[index]["response"]


def test_captured_numeric_response_never_becomes_translator_feedback(tmp_path):
    scripted = ScriptedLLM()
    pipeline = B2GPipeline.default(repo_root=ROOT, llm_client=scripted, formal_evaluation=True)
    raw = LiteralReplyClient([CAPTURED[-1]["response"]])
    pipeline.reviewer.llm = raw
    episode = pipeline.case.get_episode("ep-001")
    with pytest.raises(RuntimeError, match="format-repair"):
        pipeline.run_contract_stage(episode.sender_message, episode=episode, sender=episode.start_role)
    assert len(raw.calls) == 3
    assert scripted.counts == {"translator": 1, "actor": 1}
    assert "response_validation_error" in raw.calls[-1]


def test_new_wire_schema_has_one_categorical_judgement_and_no_free_numeric_scores():
    schema = _strict_json_schema(SemanticAuditResult)
    assert not {"status", "issues", "score", "weight"} & schema["properties"].keys()
    assert not {"score", "weight"} & schema["$defs"]["SemanticFinding"]["properties"].keys()


@pytest.mark.parametrize("finding", ["consistent", "context_resolved"])
def test_supported_meaning_has_zero_risk_but_still_requires_physics(tmp_path, finding):
    def hook(agent, n, user, payload):
        if agent == "reviewer":
            result = clean_audit()
            for check in result["checks"]:
                check["finding"] = finding
            return result
    _, pending, llm = setup_pipeline(tmp_path, hook)
    assert pending.reviewer_decision.semantic_risk_score == 0
    assert pending.reviewer_decision.evidence_risk_score > 0
    assert pending.reviewer_decision.routing_decision == "actor"
    assert pending.workflow_status == "awaiting_physics"
    assert llm.counts["translator"] == 1
    derived = [e for e in events(pending) if e["event"] == "reviewer.semantic_scoring"]
    assert derived and derived[0]["payload"]["audit"]["checks"][0]["finding"] == finding


@pytest.mark.parametrize("fault", ["numeric_score", "resolved_ambiguity", "identical_alternatives", "no_material_difference", "pass_with_correction"])
def test_contradictory_structured_findings_are_rejected_before_routing(fault):
    payload = clean_audit()
    check = payload["checks"][0]
    if fault == "numeric_score":
        check["score"] = 0.5
    elif fault == "pass_with_correction":
        check.update(finding="context_resolved", correction_required="Change the already resolved field")
    else:
        check.update(finding="material_ambiguity", responsible_agent="translator", correction_required="Resolve the differing actions",
            observed_meaning="Unresolved reference", ambiguity={
                "alternatives": ["Reduce demand by 35 kW", "Set absolute demand to 35 kW"],
                "execution_difference": "Different load schedules", "unresolved_after_context": True})
        if fault == "resolved_ambiguity":
            check["ambiguity"]["unresolved_after_context"] = False
        elif fault == "identical_alternatives":
            check["ambiguity"]["alternatives"] = ["Same schedule", "Same schedule"]
        elif fault == "no_material_difference":
            check["observed_meaning"] = check["expected_meaning"]
    with pytest.raises(ValueError):
        SemanticAuditResult.model_validate(payload)


def test_review_phase_mismatch_repairs_reviewer_only(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "reviewer" and n == 1:
            return clean_audit("receiver_feedback")
    _, _, llm = setup_pipeline(tmp_path, hook)
    assert llm.counts == {"translator": 1, "actor": 1, "reviewer": 2}
    assert "Review phase" in [u for a, u in llm.calls if a == "reviewer"][-1]["response_validation_error"]


def test_true_missing_unit_still_requires_sender_input():
    case = CaseBundle.default(ROOT)
    episode = case.get_episode("ep-001")
    text = episode.sender_message.replace("60 kW", "60")
    spec = TranslatorAgent(ROOT / "knowledge", case=case).translate(text, episode=episode, sender=episode.start_role)
    assert spec.clarification_required
    payload = clean_audit()
    payload.update(clarification_verdict="required", clarification_basis="No unit is established by the request or context.")
    check = next(c for c in payload["checks"] if c["check_id"] == "semantic_quantities_units")
    check.update(finding="material_ambiguity", expected_meaning="A power quantity with a confirmed unit",
        observed_meaning="60 without a unit", source_excerpt="provide 60", responsible_agent="human",
        correction_required="Ask the sender for the power unit.", ambiguity={
            "alternatives": ["60 kW", "60 MW"], "execution_difference": "A factor of 1000 in requested power.",
            "unresolved_after_context": True})
    decision = ReviewerAgent(case=case, llm=LiteralReplyClient([payload])).review(spec, ActorAgent().plan(spec), source_text=text)
    assert decision.routing_decision == "clarify_sender"
    assert decision.semantic_risk_score < decision.semantic_risk_threshold


def test_explicit_semantic_conflict_still_blocks_even_with_public_context(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "reviewer" and n == 1:
            result = clean_audit()
            check = next(c for c in result["checks"] if c["check_id"] == "semantic_constraints")
            check.update(finding="contradiction", expected_meaning="Sender's explicit constraint",
                observed_meaning="Different bound silently attributed to the sender", responsible_agent="translator",
                correction_required="Preserve the explicit sender requirement and clarify its conflict with public policy.")
            return result
    _, pending, llm = setup_pipeline(tmp_path, hook)
    assert llm.counts["translator"] == 2
    assert pending.review_history[0]["decision"]["semantic_risk_score"] == 0.125


def test_public_context_is_complete_and_does_not_read_evaluation_truth(monkeypatch):
    monkeypatch.setattr(CaseBundle, "get_groundtruth", lambda *a: pytest.fail("Private oracle access"))
    monkeypatch.setattr(CaseBundle, "iter_groundtruth", lambda *a: pytest.fail("Private oracle access"))
    case = CaseBundle.default(ROOT)
    context = public_review_context(case)
    assert context["timezone"] == "America/New_York"
    assert context["time_resolution_minutes"] == 15 and len(context["bem_to_bus"]) == 10
    assert "net-load reduction" in context["service_sign_convention"]
    assert all("groundtruth" not in s["locator"] for s in context["sources"])
    # Protect the kernel boundary that was missed in the preceding repair.
    current = physics_implementation_bundle_sha256(ROOT)
    matches = []
    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "physics_implementation_bundle": matches.append(child)
                else: walk(child)
        elif isinstance(value, list):
            for child in value: walk(child)
    for path in (case.case_dir / "groundtruth/numeric_oracles").glob("ep-*.json"):
        walk(json.loads(path.read_bytes()))  # Test-only reference check, outside agent inputs.
    assert len(matches) == 121 and set(matches) == {current}


def test_translator_separates_latest_feedback_from_participant_facts():
    history = [{"speaker": "sender", "text": "Use bus 9."},
               {"speaker": "reviewer", "text": "Withdrawn concern."},
               {"speaker": "reviewer", "text": "Current correction."}]
    before = deepcopy(history)
    payload = _conversation_payload(history)
    assert payload["conversation_history"] == history[:1]
    assert payload["active_reviewer_feedback"] == "Current correction."
    assert history == before


def test_exhausted_review_is_persisted_as_blocked_without_tool_dispatch(tmp_path, monkeypatch):
    from experiments import run_contract_evaluation as runner
    pipeline, blocked, _ = setup_pipeline(tmp_path, lambda a, n, u, p: assessment(True) if a == "reviewer" else None, revisions=1)
    assert blocked.workflow_status == "blocked"
    returned = pipeline.execute_physics(blocked, run_root=tmp_path / "must-not-run")
    assert returned.workflow_status == "blocked" and pipeline.physics_executor.calls == 0
    assert not (tmp_path / "must-not-run").exists()
    monkeypatch.setattr(pipeline, "execute_physics", lambda *a, **k: pytest.fail("Blocked task must not dispatch physics"))
    monkeypatch.setattr(pipeline, "run_contract_stage", lambda *a, **k: blocked)
    blocked.llm_request_hashes = []
    blocked.llm_invocation_ids = []
    llm = pipeline.translator.llm
    llm.request_hashes, llm.invocation_ids = [], []
    llm.settings = LLMSettings(model="offline-test-fixture")
    llm.record_root = tmp_path / "lr"
    monkeypatch.setattr(runner.B2GPipeline, "default", lambda **k: pipeline)
    monkeypatch.setattr(runner, "git_state", lambda: ("0" * 40, False))
    monkeypatch.setattr(runner, "runtime_tool_versions", lambda: {})
    monkeypatch.setattr(runner, "parse_args", lambda: SimpleNamespace(mode="offline", method="full_multi_agent", split="all",
        repeats=1, episode_id=["ep-001"], execute_physics=True, score=True, continue_on_error=True,
        output_root=tmp_path / "run", physics_cache_root=tmp_path / "pc", no_physics_cache=True))
    assert runner.main() == 0
    index = next((tmp_path / "run").glob("*/index.jsonl"))
    row = json.loads(index.read_text(encoding="utf-8").splitlines()[0])
    assert row["status"] == "completed" and row["workflow_status"] == "blocked"
    assert not row["task_success"] and row["physics_attempt_count"] == 0 and row["error"] is None
    result = json.loads((index.parent / "results/ep-001-r01.json").read_bytes())
    assert result["information_exchange"]["sender_received"] is None
    from experiments.report_proposed_suite import cohort
    from b2g_agent.evaluation.summary import summarize_evaluation_runs
    operational = summarize_evaluation_runs([index.parent])["methods"]["full_multi_agent"]
    assert operational["workflow_outcomes"] == {"blocked": 1}
    assert operational["failed_count"] == 0 and operational["Y"] == 0
    paper = cohort([{"row": row, "result": result, "score": None}])
    assert paper["workflow_outcomes"] == {"blocked": 1}
    assert paper["runtime_failures"] == 0 and paper["Y"] == 0
