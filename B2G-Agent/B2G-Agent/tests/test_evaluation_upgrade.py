"""Offline controls for the 600-attempt protocol; never benchmark results."""
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from b2g_agent.agents.actor import PlanCompiler
from b2g_agent.agents.llm import StructuredLLMClient
from b2g_agent.agents.reviewer import ReviewerAgent
from b2g_agent.case_io import CaseBundle
from b2g_agent.agents.translator import TranslatedFeedback
from b2g_agent.contracts.outcomes import validated_non_execution_report, rejection_causes, annotate_non_execution_feedback
from b2g_agent.evaluation.agent_metrics import (NINE_METRICS, capture_diagnostics,
    evaluate_agents, mapping_fields, reference_review)
from b2g_agent.evaluation.computation_cost import computation_cost
from b2g_agent.evaluation.physics_cache import execution_plan_sha256
from b2g_agent.evaluation.scoring import InteractionScorer
from b2g_agent.observability import LLMSettings
from b2g_agent.observability.journal import EventJournal, durable_write
from test_correction_workflow import setup_pipeline, assessment

ROOT = Path(__file__).resolve().parents[1]
CASE = CaseBundle.default(ROOT)
SCORER = InteractionScorer(CASE.case_dir / "evaluation/scoring.json")


def rejection(tmp_path, bidirectional=False):
    final = {"receiver_decision": "rejected", "committed_power_kw": 0.0,
             "completion_percentage": 0.0, "release_eligible": False,
             "business_metrics": {"requested_output_values": {"reduction_kw": 0}}}
    validation = tmp_path / "validation.json"
    validation.write_text(json.dumps({"grid_violations": {"line_overload_intervals": 2}}), encoding="utf-8")
    package = {"status": "validated_partial", "final": final, "iterations": [{"coupled_validation": {
        "path": validation.name, "sha256": hashlib.sha256(validation.read_bytes()).hexdigest()}}]}
    if bidirectional:
        directions = {}
        for name in ("reduction", "absorption"):
            child = tmp_path / (name + ".json")
            child.write_text(json.dumps(package), encoding="utf-8")
            directions[name] = {**final, "evidence_package": child.name,
                "sha256": hashlib.sha256(child.read_bytes()).hexdigest()}
        package = {"status": "validated_partial", "directions": directions, **final}
    path = tmp_path / "package.json"
    path.write_text(json.dumps(package), encoding="utf-8")
    return {"evidence_path": str(path), "evidence_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "workflow": "bidirectional_flexibility_envelope" if bidirectional else "closed_loop",
            "execution_log": {"rejection_causes": {key: ["line_overload_intervals"]
                for key in (("reduction", "absorption") if bidirectional else ("event",))}},
            "converged": True, "hard_constraints_passed": False, **final}


@pytest.mark.parametrize("bidirectional", [False, True])
def test_verified_rejection_is_distinct_from_an_unsafe_commitment(tmp_path, bidirectional):
    evidence = rejection(tmp_path, bidirectional)
    assert validated_non_execution_report(evidence)
    spec = CASE.get_groundtruth("ep-001").task_specification.model_copy(deep=True)
    spec.requested_outputs = ["reduction_kw"]
    validate = lambda ev: ReviewerAgent()._validate_evidence(spec, None, ev, execution_log=None, required=True)
    assert not validate(evidence)
    assert not validated_non_execution_report({**evidence, "committed_power_kw": 1})
    assert not validated_non_execution_report({**evidence, "converged": False})
    assert not validated_non_execution_report({**evidence, "evidence_sha256": "0" * 64})
    assert not validated_non_execution_report({**evidence, "business_metrics": {}})
    assert any(i.code == "E-HARD-CONSTRAINT-VIOLATION" for i in validate({**evidence, "receiver_decision": "accepted"}))
    candidate = SimpleNamespace(physics_evidence=SimpleNamespace(**evidence),
        information_exchange=SimpleNamespace(decision="rejected"))
    # The production gate accepts a dict or Pydantic model; use a dict-backed model here.
    candidate.physics_evidence.model_dump = lambda **kw: evidence
    truth = CASE.get_groundtruth("ep-001")
    numeric = {"final_decision": {"receiver_decision": "rejected"},
               "grid_violations": {"line_overload_intervals": 2}}
    if bidirectional:
        numeric["validated_metrics_by_direction"] = {direction: {"grid_violations": {"line_overload_intervals": 2}}
            for direction in ("reduction", "absorption")}
    assert SCORER._physical_feasibility(CASE.get_episode("ep-001"), truth, candidate, numeric) == 1
    wrong_cause = {"final_decision": {"receiver_decision": "rejected"},
                   "grid_violations": {"low_voltage_intervals": 2}}
    assert SCORER._physical_feasibility(CASE.get_episode("ep-001"), truth, candidate, wrong_cause) == 0
    assert SCORER._physical_feasibility(CASE.get_episode("ep-001"), truth, candidate,
        {"final_decision": {"receiver_decision": "accepted"}}) == 0
    if bidirectional:
        (tmp_path / "absorption.json").write_text("{}", encoding="utf-8")
        assert not validated_non_execution_report(evidence)
    else:
        evidence["execution_log"] = {}
        candidate.physics_evidence.execution_log = {}
        assert any(i.code == "E-REJECTION-CAUSE-MISSING" for i in validate(evidence))
        assert SCORER._physical_feasibility(CASE.get_episode("ep-001"), truth, candidate, numeric) == 0


def test_cached_rejection_feedback_cites_verified_constraint_causes(tmp_path):
    from b2g_agent.contracts import PhysicsEvidence, ReceiverFeedback
    evidence = rejection(tmp_path)
    evidence.pop("release_eligible")
    evidence.update(status="validated_partial", replan_count=0, input_hashes={})
    evidence["execution_log"] = {}
    model = PhysicsEvidence.model_validate(evidence)
    feedback = ReceiverFeedback(decision="rejected", text="I reject the request.",
        committed_power_kw=0, completion_percentage=0, completion_basis="Stored validation.")
    revised, annotated = annotate_non_execution_feedback(feedback, model)
    assert "line thermal limit" in revised.text and "No schedule is released" in revised.text
    assert annotated.execution_log["rejection_causes"] == rejection_causes(annotated)
    assert annotated.evidence_sha256 == model.evidence_sha256
    assert not model.execution_log and feedback.text == "I reject the request."


def test_symmetric_envelope_rejects_when_only_one_direction_is_unsafe(tmp_path):
    evidence = rejection(tmp_path, bidirectional=True)
    child_path = tmp_path / "reduction.json"
    child = json.loads(child_path.read_text())
    child["final"].update(receiver_decision="modified", committed_power_kw=50,
                          completion_percentage=50, release_eligible=True)
    safe_validation = tmp_path / "safe-validation.json"
    safe_validation.write_text(json.dumps({"grid_violations": {"line_overload_intervals": 0}}))
    child["iterations"][-1]["coupled_validation"] = {"path": safe_validation.name,
        "sha256": hashlib.sha256(safe_validation.read_bytes()).hexdigest()}
    child_path.write_text(json.dumps(child))
    parent_path = tmp_path / "package.json"
    parent = json.loads(parent_path.read_text())
    parent["directions"]["reduction"].update(child["final"], sha256=hashlib.sha256(child_path.read_bytes()).hexdigest())
    parent_path.write_text(json.dumps(parent))
    evidence["evidence_sha256"] = hashlib.sha256(parent_path.read_bytes()).hexdigest()
    assert validated_non_execution_report(evidence)
    assert rejection_causes(evidence) == {"absorption": ["line_overload_intervals"]}
    assert evidence["committed_power_kw"] == evidence["completion_percentage"] == 0


def test_four_revisions_recover_after_the_old_limit(tmp_path):
    def hook(agent, n, user, payload):
        return assessment(True) if agent == "reviewer" and n <= 3 else None
    pipeline, pending, llm = setup_pipeline(tmp_path, hook, revisions=4)
    assert pending.execution_plan.valid and not pending.reviewer_decision.issues
    assert llm.counts["reviewer"] == 4 and pipeline.physics_executor.calls == 0


def test_format_repairs_keep_every_response_and_per_call_limit(tmp_path, monkeypatch):
    contents = iter(["{truncated", "{}", json.dumps({"sender_message": "I accept 60 kW during the requested window."})])
    calls, response_writes = [], []
    def create(**kwargs):
        calls.append(kwargs)
        content = next(contents)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            model_dump=lambda **kw: {"choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    import b2g_agent.agents.llm as module
    original_write = module.durable_write
    def write(path, text):
        if path.name == "response.json":
            response_writes.append(path)
        original_write(path, text)
    monkeypatch.setattr(module, "durable_write", write)
    client = StructuredLLMClient(mode="live", settings=LLMSettings(model="offline-double"),
        record_root=tmp_path / "lr", client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    journal = EventJournal(tmp_path / "journal.jsonl", "fixture")
    with journal.activate():
        reply = client.call(TranslatedFeedback, system="Offline fixture", user_payload={}, agent_name="translator_feedback")
    assert "60 kW" in reply.sender_message
    assert len(calls) == len(set(response_writes)) == len(response_writes) == 3
    assert all(call["max_completion_tokens"] == 9000 for call in calls)
    manifests = [json.loads(p.read_text()) for p in (tmp_path / "lr").glob("*/i/*/manifest.json")]
    assert sorted(m["status"] for m in manifests) == ["completed", "failed", "failed"]
    responses = [json.loads(p.read_text()) for p in response_writes]
    assert sum(r["usage"]["total_tokens"] for r in responses) == 45
    assert len([line for line in journal.path.read_text().splitlines() if json.loads(line)["event"] == "llm.format_repair"]) == 2


def test_atomic_write_retries_only_temporary_sharing_failure(tmp_path, monkeypatch):
    import b2g_agent.observability.journal as module
    original, calls = module.os.replace, []
    def replace(source, target):
        calls.append(target)
        if len(calls) <= 2:
            raise PermissionError("temporary scanner lock")
        original(source, target)
    monkeypatch.setattr(module.os, "replace", replace)
    monkeypatch.setattr(module.time, "sleep", lambda delay: None)
    path = tmp_path / "response.json"
    durable_write(path, "immutable response")
    assert path.read_text() == "immutable response" and len(calls) == 3


def test_journal_retains_raw_reply_when_response_file_cannot_be_written(tmp_path, monkeypatch):
    import b2g_agent.agents.llm as module
    content = json.dumps({"sender_message": "I accept the requested service."})
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        model_dump=lambda **kw: {"choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    client = StructuredLLMClient(mode="live", settings=LLMSettings(model="offline-double"),
        record_root=tmp_path / "lr", client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))))
    original = module.durable_write
    def fail_response(path, text):
        if path.name == "response.json":
            raise PermissionError("controlled persistent file lock")
        return original(path, text)
    monkeypatch.setattr(module, "durable_write", fail_response)
    journal = EventJournal(tmp_path / "journal.jsonl", "fixture")
    with journal.activate(), pytest.raises(PermissionError):
        client.call(TranslatedFeedback, system="Offline fixture", user_payload={}, agent_name="translator_feedback")
    saved = [json.loads(line) for line in journal.path.read_text().splitlines()]
    replies = [event["payload"]["response"] for event in saved if event["event"] == "llm.response"]
    assert len(replies) == 1 and replies[0]["usage"]["total_tokens"] == 15
    report = load_report_module()
    info = report.inspect_journal(journal.path, tmp_path / "transcript.md", "fixture")
    errors, warnings = [], []
    records = report.verify_invocations({"run_id": "fixture", "status": "failed",
        "llm_request_hashes": client.request_hashes, "llm_invocation_ids": client.invocation_ids},
        tmp_path / "lr", errors, warnings, info["llm_responses"])
    assert not errors and warnings and records[0]["usage"]["total_tokens"] == 15


def test_mapping_requires_all_five_dimensions_per_field():
    reference = CASE.get_groundtruth("ep-001").task_specification
    predicted = reference.model_copy(deep=True)
    predicted.fields["time_resolution"].confidence = 0.5
    assert all(row["correct"] for row in mapping_fields(reference, predicted, SCORER))
    predicted.fields["target_power"].unit = "MW"
    rows = mapping_fields(reference, predicted, SCORER)
    assert all(not row["correct"] for row in rows if row["variable"] == "target_power")
    predicted = reference.model_copy(deep=True)
    predicted.target_buses[0] += 1
    assert all(not row["correct"] for row in mapping_fields(reference, predicted, SCORER) if row["object"] == reference.target_bems[0])


def test_reviewer_feedback_metrics_keep_all_failures_and_executable_denominators():
    from b2g_agent.evaluation.reviewer_feedback_metrics import REVIEWER_METRICS, reviewer_feedback_metrics
    rows = [
        dict(source_executable=False, appropriate_handling=True, reached_reverse_review=False, task_success=True),
        dict(source_executable=True, appropriate_handling=True, reached_reverse_review=True, task_success=False),
        dict(source_executable=True, appropriate_handling=False, reached_reverse_review=False, task_success=False),
    ]
    metrics = reviewer_feedback_metrics(rows)
    assert [metrics[name]["denominator"] for name in REVIEWER_METRICS] == [3, 2, 2]
    assert [metrics[name]["value"] for name in REVIEWER_METRICS] == [2/3, .5, 0]
    assert reviewer_feedback_metrics(rows[:1])[REVIEWER_METRICS[1]]["value"] is None


def test_memory_excludes_same_episode_and_formal_updates_are_frozen(tmp_path):
    pipeline, pending, _ = setup_pipeline(tmp_path)
    memory = json.loads((CASE.case_dir / "reviewer_memory/verified_seed.jsonl").read_text().splitlines()[0])
    memory.update(source_episode_id=pending.episode_id, direction=f"{pending.task_specification.sender_role.value}->{pending.task_specification.receiver_role.value}")
    other = {**memory, "memory_id": "different-source", "source_episode_id": "different-episode"}
    path = tmp_path / "memory.jsonl"
    path.write_text(json.dumps(memory) + "\n" + json.dumps(other) + "\n")
    found = ReviewerAgent(path)._retrieve_memories(pending.task_specification, set())
    assert [m["memory_id"] for m in found] == ["different-source"]
    assert pipeline.reviewer.online_memory_path is None


def load_report_module():
    spec = importlib.util.spec_from_file_location("report_fixture", ROOT / "experiments/report_proposed_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_same_attempt_metrics_cost_and_report(tmp_path):
    pipeline, pending, _ = setup_pipeline(tmp_path)
    journal_path = Path(pending.interaction_journal)
    report = load_report_module()
    journal = report.inspect_journal(journal_path, tmp_path / "transcript.md", "ep-001-r01")
    record = {"row": {"run_id": "ep-001-r01", "episode_id": "ep-001", "repeat": 1,
        "interaction_journal": str(journal_path), "status": "failed", "error": "Controlled fixture",
        "elapsed_seconds": 1.0, "total_tokens": 45, "per_agent_llm_usage": {
            "translator": {"total_tokens": 15}, "translator_feedback": {"total_tokens": 30}}},
        "journal": journal, "score": None, "result": None}
    agents = evaluate_agents([record], CASE, SCORER)
    assert set(agents["metrics"]) == set(NINE_METRICS)
    assert agents["metrics"]["actor_tool_chain_success"]["value"] == 0
    assert agents["review_samples_missing"] == 1
    assert agents["reviewer_samples"][0]["reference_label"] == "evidence"
    costs = computation_cost([record])
    assert costs["per_agent_usage"]["translator"]["total_tokens"] == 45
    assert costs["per_agent_retrieval_seconds"]["reviewer"]["count"] == 1
    assert costs["per_agent_retrieval_seconds"]["actor"]["applicable"] is False
    suite = {"started_at": "2026-10-29T00:00:00+00:00", "completed_at": "2026-10-29T00:00:02+00:00"}
    summary = {"agent_level": agents, "results": report.aggregate([record], CASE, suite),
        "expected_attempts": 600, "status": "partial", "generated_at": "offline fixture",
        "integrity": {k: 0 for k in ("journals_verified", "four_message_records", "llm_invocations_verified",
            "unique_physics_packages_verified", "physics_referenced_artifacts_verified")}}
    summary["integrity"].update(frozen_files_unchanged=True, errors=[], recording_warnings=[], usage_adjustments=[])
    report.write_report(tmp_path / "RESULTS.md", summary)
    text = (tmp_path / "RESULTS.md").read_text(encoding="utf-8")
    assert "1 / 600" in text and r"Q^{\mathrm{tsr}}" in text
    assert all(name in text for name in NINE_METRICS)
    # The label comes from the reviewed input, not a chosen Reviewer verdict.
    sample = agents["reviewer_samples"][0]
    inputs = sample["reviewed_input"]
    changed = json.loads(json.dumps(inputs))
    changed["specification"]["fields"]["target_power"]["value"] += 5
    assert reference_review(changed, CASE.get_groundtruth("ep-001"), None, SCORER)[0] == "semantic"
    with pytest.raises(ValueError, match="Invalid independently"):
        evaluate_agents([record], CASE, SCORER, {sample["sample_id"]: {
            "reference_label": "evidence", "reference_route": "Actor", "annotator": "offline-fixture",
            "reviewed_input_sha256": "stale-input-hash"}})
    with pytest.raises(ValueError, match="do not match"):
        evaluate_agents([record], CASE, SCORER, {"unknown-sample": {}})


def test_runtime_timing_survives_failed_output_and_later_repairs_do_not_replace_initial_output(tmp_path):
    path = tmp_path / "trace.jsonl"
    events = []
    def event(kind, span, seconds, **payload):
        events.append({"event": kind, "span_id": span, "timestamp": f"2026-10-29T00:00:{seconds:02d}+00:00",
                       "event_id": str(len(events)), "payload": payload})
    event("stage.started", "first", 0, stage="translator.translate", inputs={})
    event("retrieval.completed", "first", 0, agent="translator", elapsed_seconds=0.25)
    event("stage.completed", "first", 1, stage="translator.translate", output={"retrieval_trace": {"elapsed_seconds": 999}})
    event("stage.started", "revision", 2, stage="translator.translate", inputs={})
    event("retrieval.completed", "revision", 2, agent="translator", elapsed_seconds=0.5)
    event("llm.format_repair", "revision", 3, agent="translator")
    event("stage.failed", "revision", 4, stage="translator.translate")
    path.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    diagnostic = capture_diagnostics(path)
    assert diagnostic["translator_retrieval_seconds"] == 0.75
    assert not diagnostic["invalid_schema_agents"]
    assert diagnostic["reviewer_retrieval_seconds"] == 0


def test_suite_without_execute_prepares_600_without_creating_outputs(monkeypatch, capsys):
    spec = importlib.util.spec_from_file_location("suite_fixture", ROOT / "experiments/run_proposed_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr("sys.argv", ["run_proposed_suite.py"])
    monkeypatch.setattr(module, "freeze_sources", lambda root: pytest.fail("Preparation must not write a run"))
    monkeypatch.setattr(module.subprocess, "run", lambda *a, **k: pytest.fail("Preparation must not launch workers"))
    assert module.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["expected_attempts"] == 600 and result["api_calls"] == result["physics_runs"] == 0
