"""Offline fault injection; doubles are test fixtures, never case results."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from b2g_agent.agents.actor import PlanCompiler
from b2g_agent.agents.llm import StructuredLLMClient
from b2g_agent.agents.response_validation import validate_response
from b2g_agent.case_io import CaseBundle
from b2g_agent.contracts import (ActionType, PhysicsEvidence, PlanProposal, ReceiverFeedback,
                                 SemanticReviewResult, TaskSpecification)
from b2g_agent.contracts.models import SEMANTIC_REVIEW_CHECK_IDS
from b2g_agent.contracts.semantic_audit import SemanticAuditResult
from reviewer_fixtures import categorical_fixture
from b2g_agent.contracts.readiness import execution_readiness
from b2g_agent.evaluation.physics_cache import PhysicsResultCache, execution_plan_sha256
from b2g_agent.observability import LLMSettings
from b2g_agent.observability.journal import EventJournal, traced_stage
from b2g_agent.observability.journal import subprocess_environment
from b2g_agent.orchestration import B2GPipeline
from b2g_agent.orchestration.task_physics import TaskPhysicsExecutor, _numerically_converged

ROOT = Path(__file__).resolve().parents[1]


def assessment(fail=False, responsible="translator", passed_with_findings=False):
    checks = [{"check_id": k, "channel": "semantic", "score": 0.0,
               "message": "Consistent with the supplied source."} for k in sorted(SEMANTIC_REVIEW_CHECK_IDS)]
    if fail:
        checks[0].update(score=1.0, responsible_agent=responsible, message="Correct the cited field.",
                         structured_field="task_specification.fields.target_power", source_excerpt="60 kW",
                         basis_locator="test:controlled_fault")
    return {"status": "failed" if fail and not passed_with_findings else "passed", "issues": [], "checks": checks}


class ScriptedLLM:
    available = True

    def __init__(self, hook=None, unique_plans=False):
        self.request_hashes, self.invocation_ids, self.calls = [], [], []
        self.counts = Counter()
        self.hook, self.unique_plans = hook, unique_plans

    def call(self, response_model, *, system, user_payload, agent_name):
        self.counts[agent_name] += 1
        number = self.counts[agent_name]
        self.calls.append((agent_name, user_payload))
        self.request_hashes.append(f"fixture-request-{len(self.calls)}")
        self.invocation_ids.append(f"fixture-invocation-{len(self.calls)}")
        if agent_name in {"translator", "translator_feedback"}:
            payload = json.loads(json.dumps(user_payload["deterministic_draft"]))
        elif agent_name == "actor":
            spec = TaskSpecification.model_validate(user_payload["task_specification"])
            actions = PlanCompiler().compile(spec).model_dump(mode="json")["actions"]
            if self.unique_plans:
                for a in actions:
                    a["action_id"] += f"_{number}"
                    a["depends_on"] = [d + f"_{number}" for d in a["depends_on"]]
            payload = {"actions": actions}
        else:
            payload = assessment()
        if self.hook:
            replacement = self.hook(agent_name, number, user_payload, payload)
            if replacement is not None:
                payload = replacement
        if response_model is SemanticAuditResult:
            payload = categorical_fixture(payload, user_payload["review_phase"])
        return response_model.model_validate(payload)


class PhysicsDouble:
    def __init__(self, root, failures=0, exceptions=0):
        self.root, self.failures, self.exceptions, self.calls = root, failures, exceptions, 0

    @traced_stage("physics.executor")
    def run(self, spec, plan, *, run_root):
        self.calls += 1
        if self.calls <= self.exceptions:
            raise RuntimeError("Controlled offline tool failure")
        path = self.root / f"fixture-evidence-{self.calls}.json"
        path.write_text('{"test_fixture_only": true}', encoding="utf-8")
        feedback = ReceiverFeedback(decision="accepted", text="I accept this controlled test request.",
            committed_power_kw=float(spec.fields["target_power"].value), completion_percentage=100,
            completion_basis="Offline control-flow fixture, not a physical measurement.")
        evidence = PhysicsEvidence(workflow="closed_loop", status="validated_full",
            evidence_path=str(path), evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            receiver_decision="accepted", committed_power_kw=feedback.committed_power_kw,
            completion_percentage=100, replan_count=0, converged=True,
            hard_constraints_passed=self.calls > self.failures,
            business_metrics={name: 1.0 for name in spec.requested_outputs}, input_hashes={},
            execution_plan_id=plan.plan_id, execution_plan_sha256=execution_plan_sha256(plan),
            execution_log={"time_alignment_passed": True, "bem_bus_mapping_passed": True})
        return feedback, evidence


def setup_pipeline(tmp_path, hook=None, **kwargs):
    llm = ScriptedLLM(hook, unique_plans=kwargs.pop("unique_plans", False))
    pipeline = B2GPipeline.default(repo_root=ROOT, llm_client=llm, formal_evaluation=True)
    pipeline.maximum_agent_revisions = kwargs.pop("revisions", 2)
    pipeline.physics_executor = PhysicsDouble(tmp_path, **kwargs)
    episode = pipeline.case.get_episode("ep-001")
    pending = pipeline.run_contract_stage(episode.sender_message, episode=episode, sender=episode.start_role)
    return pipeline, pending, llm


def events(result):
    return [json.loads(line) for line in Path(result.interaction_journal).read_text(encoding="utf-8").splitlines()]


def test_review_status_normalization_retains_every_finding():
    raw = assessment(True, passed_with_findings=True)
    parsed, changes = validate_response(SemanticReviewResult, raw)
    assert parsed.status == "failed" and parsed.checks[0].score == 1
    assert raw["status"] == "passed" and changes[0]["original"] == "passed"
    with pytest.raises(ValueError):
        validate_response(SemanticReviewResult, {**raw, "checks": raw["checks"][:-1]})
    with pytest.raises(ValueError):
        validate_response(SemanticReviewResult, {**assessment(), "status": "failed"})


def test_raw_reviewer_response_is_durable_before_parser_and_has_normalization(tmp_path, monkeypatch):
    raw = assessment(True, passed_with_findings=True)
    content = json.dumps(raw)
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
                               model_dump=lambda **kw: {"choices": [{"message": {"content": content}}],
                                   "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}})
    client = StructuredLLMClient(mode="live", settings=LLMSettings(model="offline-double"),
        record_root=tmp_path / "lr", client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))))
    import b2g_agent.agents.llm as module
    real_validate = module.validate_response

    def assert_saved(model, value):
        paths = list((tmp_path / "lr").glob("*/i/*/response.json"))
        assert len(paths) == 1 and json.loads(paths[0].read_text(encoding="utf-8"))["choices"][0]["message"]["content"] == content
        return real_validate(model, value)

    monkeypatch.setattr(module, "validate_response", assert_saved)
    result = client.call(SemanticReviewResult, system="Offline fixture", user_payload={}, agent_name="reviewer")
    assert result.status == "failed"
    metadata = json.loads(next((tmp_path / "lr").glob("*/i/*/manifest.json")).read_text(encoding="utf-8"))
    assert metadata["status"] == "completed" and metadata["normalizations"]


def test_forward_semantic_and_evidence_failure_revises_translator_before_actor(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "actor" and n == 1:
            payload["actions"][-1]["depends_on"] = []
        if agent == "reviewer" and n == 1:
            return assessment(True)
    pipeline, pending, llm = setup_pipeline(tmp_path, hook)
    assert [a for a, _ in llm.calls] == ["translator", "actor", "reviewer", "translator", "actor", "reviewer"]
    assert pending.execution_plan.valid and not pending.reviewer_decision.issues
    routed = [e["payload"] for e in events(pending) if e["event"] == "reviewer.return"]
    assert routed[0]["route"] == "translator_then_actor"
    assert llm.calls[3][1]["active_reviewer_feedback"] and llm.calls[4][1]["reviewer_feedback"]


def test_missing_target_key_is_repaired_before_execution(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "translator" and n == 1:
            payload["fields"].pop("target_power")
    pipeline, pending, llm = setup_pipeline(tmp_path, hook)
    assert llm.counts["translator"] == 2
    assert not execution_readiness(pending.task_specification, case=pipeline.case)
    assert pipeline.physics_executor.calls == 0


def test_human_route_creates_recorded_clarification_even_if_initial_contract_did_not(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "reviewer":
            return assessment(True, responsible="human")
    _, result, _ = setup_pipeline(tmp_path, hook)
    assert result.information_exchange.decision == "needs_clarification"
    assert result.information_exchange.sender_received and result.information_exchange.completion_percentage == 0
    assert any(e["event"] == "exchange.clarification" for e in events(result))


@pytest.mark.parametrize("cache_enabled", [False, True])
def test_physical_failure_returns_actor_and_revalidates_new_plan(tmp_path, monkeypatch, cache_enabled):
    pipeline, pending, llm = setup_pipeline(tmp_path, failures=1, unique_plans=True)
    cache = None
    if cache_enabled:
        monkeypatch.setattr("b2g_agent.evaluation.physics_cache.physics_input_hashes", lambda case: {"fixture": "f" * 64})
        cache = PhysicsResultCache(root=tmp_path / "cache", repo_root=ROOT, case=pipeline.case)
    result = pipeline.execute_physics(pending, run_root=tmp_path / "physics", cache=cache)
    assert pipeline.physics_executor.calls == 2 and result.reviewer_decision.routing_decision == "release"
    assert result.llm_invocation_ids == llm.invocation_ids  # Includes the intermediate Actor/review calls.
    assert llm.counts["actor"] == 2 and len(result.physics_attempts) == 2
    if cache_enabled:
        assert result.physics_attempts[0]["cache_key"] != result.physics_attempts[1]["cache_key"]
    assert any(e["event"] == "reviewer.return" and e["payload"]["direction"] == "physics" for e in events(result))


def test_cache_hit_cannot_bypass_failed_evidence_or_revision_limit(tmp_path, monkeypatch):
    pipeline, pending, llm = setup_pipeline(tmp_path, failures=999)
    monkeypatch.setattr("b2g_agent.evaluation.physics_cache.physics_input_hashes", lambda case: {"fixture": "f" * 64})
    cache = PhysicsResultCache(root=tmp_path / "cache", repo_root=ROOT, case=pipeline.case)
    f, e = pipeline.physics_executor.run(pending.task_specification, pending.execution_plan, run_root=tmp_path)
    cache.store(pending.task_specification, execution_plan=pending.execution_plan, feedback=f, evidence=e)
    pipeline.physics_executor.calls = 0
    result = pipeline.execute_physics(pending, run_root=tmp_path / "physics", cache=cache)
    assert pipeline.physics_executor.calls == 0
    assert [a["cache_status"] for a in result.physics_attempts] == ["hit"] * 3
    assert llm.counts["actor"] == 3 and result.reviewer_decision.routing_decision == "actor"
    assert not any(e["event"] == "exchange.released" for e in events(result))


def test_reverse_semantics_are_repaired_before_actor_physical_retry(tmp_path):
    reverse_reviews = 0
    def hook(agent, n, user, payload):
        nonlocal reverse_reviews
        if agent == "reviewer" and user["simulation_status"] == "completed":
            reverse_reviews += 1
            if reverse_reviews == 1:
                return assessment(True)
    pipeline, pending, llm = setup_pipeline(tmp_path, hook, failures=1)
    result = pipeline.execute_physics(pending, run_root=tmp_path / "physics")
    assert result.reviewer_decision.routing_decision == "release"
    sequence = [a for a, _ in llm.calls]
    second_feedback = [i for i, a in enumerate(sequence) if a == "translator_feedback"][1]
    second_actor = [i for i, a in enumerate(sequence) if a == "actor"][1]
    assert second_feedback < second_actor


def test_reverse_failure_exhaustion_does_not_skip_to_actor_or_release(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "reviewer" and user["simulation_status"] == "completed":
            return assessment(True)
    pipeline, pending, llm = setup_pipeline(tmp_path, hook, failures=999)
    result = pipeline.execute_physics(pending, run_root=tmp_path / "physics")
    assert llm.counts["translator_feedback"] == 3 and llm.counts["actor"] == 1
    assert result.reviewer_decision.routing_decision == "translator_then_actor"


def test_tool_exception_is_logged_and_returned_to_actor(tmp_path):
    pipeline, pending, llm = setup_pipeline(tmp_path, exceptions=1)
    result = pipeline.execute_physics(pending, run_root=tmp_path / "physics")
    assert result.reviewer_decision.routing_decision == "release" and llm.counts["actor"] == 2
    assert result.physics_attempts[0]["status"] == "failed"
    assert any(e["event"] == "stage.failed" and "Controlled offline" in e["payload"]["error"] for e in events(result))


@pytest.mark.parametrize("value,unit", [(None, "kW"), (True, "kW"), (float("nan"), "kW"), (-1, "kW"), (60, "MW")])
def test_execution_precheck_stops_invalid_quantity_without_calling_tools(tmp_path, value, unit):
    pipeline, pending, _ = setup_pipeline(tmp_path)
    pending.task_specification.fields["target_power"].value = value
    pending.task_specification.fields["target_power"].unit = unit
    result = pipeline.execute_physics(pending, run_root=tmp_path / "physics")
    assert result.information_exchange.decision == "needs_clarification"
    assert pipeline.physics_executor.calls == 0


def test_stale_plan_valid_flag_cannot_bypass_input_binding_check(tmp_path):
    pipeline, pending, _ = setup_pipeline(tmp_path)
    action = next(a for a in pending.execution_plan.actions if a.action_type == ActionType.MAP_LOADS_TO_BUSES)
    action.inputs["target_buses"] = [33]
    assert pending.execution_plan.valid
    with pytest.raises(ValueError, match="freshly validated"):
        pipeline.execute_physics(pending, run_root=tmp_path / "physics")
    assert pipeline.physics_executor.calls == 0


def test_redundant_transitive_edge_is_audited_but_missing_edges_are_not_repaired():
    case = CaseBundle.default(ROOT)
    spec = case.get_groundtruth("ep-135").task_specification
    compiler = PlanCompiler()
    proposal = PlanProposal(actions=compiler.compile(spec).actions)
    mapping = next(a for a in proposal.actions if a.action_type == ActionType.MAP_LOADS_TO_BUSES)
    scale = next(a for a in proposal.actions if a.action_type == ActionType.SCALE_BEM_PORTFOLIOS)
    mapping.depends_on.append(scale.action_id)
    normalized = compiler.compile_proposal(spec, proposal)
    assert normalized.valid and normalized.normalization_log
    assert scale.action_id in mapping.depends_on  # The original proposal is immutable.
    mapping.depends_on = [scale.action_id]
    invalid = compiler.compile_proposal(spec, proposal)
    assert not invalid.valid
    details = compiler.diagnostics(invalid, spec)
    assert "apply_der_dispatch" in next(d for d in details if d.get("action") == "map_loads_to_buses")["missing_dependencies"]


def test_convergence_and_feasibility_are_separate():
    assert _numerically_converged({"nonlinear_grid_status": "fail", "grid_violations": {"nonconverged_intervals": 0, "line_overload_intervals": 2}})
    assert not _numerically_converged({"grid_violations": {"nonconverged_intervals": 1}})
    assert not _numerically_converged({})


def test_physics_executor_checks_before_creating_outputs_or_subprocesses(tmp_path, monkeypatch):
    spec = CaseBundle.default(ROOT).get_groundtruth("ep-001").task_specification
    plan = PlanCompiler().compile(spec)
    spec.fields.pop("target_power")
    monkeypatch.setattr("b2g_agent.orchestration.task_physics.subprocess.run", lambda *a, **k: pytest.fail("Subprocess must not start"))
    with pytest.raises(ValueError, match="Execution inputs"):
        TaskPhysicsExecutor(ROOT).run(spec, plan, run_root=tmp_path / "must-not-exist")
    assert not (tmp_path / "must-not-exist").exists()


def test_journal_is_visible_before_next_stage_and_preserves_exception(tmp_path):
    journal = EventJournal(tmp_path / "events.jsonl", "offline-journal")
    @traced_stage("controlled_stage")
    def fail():
        records = [json.loads(line) for line in journal.path.read_text(encoding="utf-8").splitlines()]
        assert records[-1]["event"] == "stage.started"
        raise RuntimeError("Controlled failure after durable start")
    with journal.activate(), pytest.raises(RuntimeError):
        fail()
    records = [json.loads(line) for line in journal.path.read_text(encoding="utf-8").splitlines()]
    assert [r["event"] for r in records] == ["stage.started", "stage.failed"]
    assert records[-1]["payload"]["traceback"]


def test_bad_reviewer_shape_retries_reviewer_only_then_stops(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "reviewer":
            return {**assessment(), "checks": []}
    llm = ScriptedLLM(hook)
    pipeline = B2GPipeline.default(repo_root=ROOT, llm_client=llm, formal_evaluation=True)
    episode = pipeline.case.get_episode("ep-001")
    with pytest.raises(RuntimeError, match="format-repair"):
        pipeline.run_contract_stage(episode.sender_message, episode=episode, sender=episode.start_role)
    assert llm.counts == Counter(translator=1, actor=1, reviewer=3)
    records = [json.loads(line) for line in next(pipeline.journal_root.glob("*.jsonl")).read_text(encoding="utf-8").splitlines()]
    assert records[-1]["event"] == "stage.failed"


def test_subprocess_worker_appends_to_same_journal_without_logging_environment_secrets(tmp_path):
    import subprocess
    import sys
    script = tmp_path / "worker_fixture.py"
    script.write_text("import os, subprocess, sys\nsubprocess.run([sys.executable, '-c', \"print('offline-tool-output')\"], capture_output=True, text=True, env=dict(os.environ))\nraise SystemExit(0)\n", encoding="utf-8")
    journal = EventJournal(tmp_path / "subprocess.jsonl", "worker-test")
    with journal.activate():
        environment = subprocess_environment()
        environment["OPENAI_API_KEY"] = "offline-secret-sentinel"
        process = subprocess.run([sys.executable, str(ROOT / "src/b2g_agent/observability/physics_worker.py"), str(script)],
            env=environment, capture_output=True, text=True, timeout=30)
    assert process.returncode == 0, process.stderr
    text = journal.path.read_text(encoding="utf-8")
    assert "offline-secret-sentinel" not in text
    records = [json.loads(line) for line in text.splitlines()]
    assert {r["trace_id"] for r in records} == {"worker-test"}
    assert any(r["event"] == "tool.output" and "offline-tool-output" in r["payload"]["stdout"] for r in records)
    assert any(r["event"] == "physics.script_completed" for r in records)
    assert not any(r["event"] == "stage.failed" for r in records)


def test_failed_journal_write_prevents_the_next_stage(tmp_path, monkeypatch):
    journal = EventJournal(tmp_path / "events.jsonl", "write-failure")
    entered = []
    @traced_stage("must_not_run")
    def work():
        entered.append(True)
    monkeypatch.setattr(journal, "emit", lambda *a, **k: (_ for _ in ()).throw(OSError("disk unavailable")))
    with journal.activate(), pytest.raises(OSError):
        work()
    assert not entered


def test_all_executable_groundtruth_contracts_pass_input_gate():
    case = CaseBundle.default(ROOT)
    checked = 0
    for episode in case.iter_episodes():
        spec = case.get_groundtruth(episode.episode_id).task_specification
        if not spec.clarification_required:
            assert not execution_readiness(spec, case=case), episode.episode_id
            checked += 1
    assert checked == 100


def test_interactive_feedback_semantic_revision_is_retained_without_premature_release(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "reviewer" and user["simulation_status"] == "declared_by_receiver" and n == 2:
            return assessment(True)
    pipeline, pending, llm = setup_pipeline(tmp_path, hook)
    feedback = ReceiverFeedback(decision="modified", text="I can offer 30 kW instead of 60 kW.",
        committed_power_kw=30, completion_percentage=50, completion_basis="Half of the requested service.")
    result = pipeline.complete_with_receiver_feedback(pending, feedback=feedback)
    assert llm.counts["translator_feedback"] == 2
    assert result.proposed_receiver_feedback == feedback
    assert result.information_exchange.decision == "pending_physics"
    assert result.reviewer_decision.routing_decision == "actor"
    assert result.llm_invocation_ids == llm.invocation_ids


def test_revised_actor_plan_is_itself_corrected_before_second_physics_run(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "actor" and n == 2:
            payload["actions"][-1]["depends_on"] = []
    pipeline, pending, llm = setup_pipeline(tmp_path, hook, failures=1)
    result = pipeline.execute_physics(pending, run_root=tmp_path / "physics")
    assert result.reviewer_decision.routing_decision == "release"
    assert llm.counts["actor"] == 3 and pipeline.physics_executor.calls == 2
    assert result.llm_invocation_ids == llm.invocation_ids


def test_translator_can_repair_receiver_wording_without_changing_physical_contract(tmp_path):
    def hook(agent, n, user, payload):
        if agent == "translator":
            payload["receiver_message"] = "Draft missing the requested decision." if n == 1 else "Please assess 60 kW and return acceptance, rejection, or a quantified counteroffer while preserving all constraints."
        if agent == "reviewer" and n == 1:
            return assessment(True)
    _, result, llm = setup_pipeline(tmp_path, hook)
    assert llm.counts["translator"] == 2
    assert "quantified counteroffer" in result.information_exchange.receiver_received
    assert result.task_specification.fields["target_power"].value == 60
    restored = TaskSpecification.model_validate_json(result.task_specification.model_dump_json())
    assert restored.receiver_message == result.information_exchange.receiver_received
    from b2g_agent.evaluation.physics_cache import task_specification_sha256
    changed_wording = restored.model_copy(update={"receiver_message": "Another equivalent wording."})
    assert task_specification_sha256(changed_wording) == task_specification_sha256(restored)


def test_unprepared_day_is_clarified_without_physics(tmp_path):
    pipeline, pending, _ = setup_pipeline(tmp_path)
    pending.task_specification.operating_day = "2026-07-15"
    result = pipeline.execute_physics(pending, run_root=tmp_path / "physics")
    assert result.workflow_status == "needs_clarification" and pipeline.physics_executor.calls == 0
    assert "operating day" in result.information_exchange.sender_received


def test_blocked_interactive_turn_allows_a_new_sender_message(tmp_path):
    from b2g_agent.collaboration import SessionStore
    def hook(agent, n, user, payload):
        if agent == "reviewer" and n <= 5:
            return assessment(True)
    llm = ScriptedLLM(hook)
    pipeline = B2GPipeline.default(repo_root=ROOT, llm_client=llm, formal_evaluation=True)
    store = SessionStore(pipeline=pipeline, output_root=tmp_path / "sessions", repo_root=ROOT)
    from b2g_agent.contracts import AgentRole
    session = store.create(role=AgentRole.DISTRIBUTION_GRID_OPERATOR, mode="interactive")
    first = session.handle_message("On the winter peak 01 operating day, please reduce bem_01 at bus 3 by 60 kW during 16:00-18:00 for peak shaving.")
    assert first.workflow_status == "blocked"
    second = session.handle_message("Please retain the same requirements and use 50 kW instead.")
    assert second.turn_index == 2 and second.interaction_journal != first.interaction_journal
