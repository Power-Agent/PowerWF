"""Terminal sampling must reflect recovery without selecting earlier successes."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from b2g_agent.agents.actor import PlanCompiler
from b2g_agent.case_io import CaseBundle
from b2g_agent.evaluation import InteractionScorer
from b2g_agent.evaluation.agent_metrics import (
    actor_terminal_diagnostic, aggregate_actor_terminal, capture_diagnostics,
    evaluate_agents, read_terminal_actor_result,
)
from b2g_agent.evaluation.physics_cache import execution_plan_sha256

ROOT = Path(__file__).resolve().parents[1]
CASE = CaseBundle.default(ROOT)
SCORER = InteractionScorer(CASE.case_dir / "evaluation/scoring.json")


@pytest.fixture
def physical():
    truth = CASE.get_groundtruth("ep-001")
    numeric = json.loads((CASE.case_dir / "groundtruth/numeric_oracles/ep-001.json").read_text(encoding="utf-8"))
    plan = PlanCompiler().compile(truth.task_specification)
    ev = {"execution_plan_sha256": execution_plan_sha256(plan),
          "business_metrics": {"requested_output_values": numeric["business_metrics"]["requested_output_values"]},
          "hard_constraints_passed": numeric["final_decision"]["release_eligible"],
          "execution_log": {"actions": [{"action_id": a.action_id, "status": "completed"} for a in plan.actions]}}
    terminal = {"task_specification": truth.task_specification.model_dump(mode="json"),
                "execution_plan": plan.model_dump(mode="json"), "physics_evidence": ev,
                "workflow_status": "blocked"}
    return truth, numeric, terminal


def journal(tmp_path, events):
    path = tmp_path / "events.jsonl"
    rows = [{"event": kind, "payload": payload, "event_id": str(i),
             "timestamp": f"2026-10-31T00:00:{i:02d}+00:00", "span_id": str(i)}
            for i, (kind, payload) in enumerate(events)]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_terminal_recovery_ignores_initial_repair_and_return_flags(tmp_path, physical):
    truth, numeric, terminal = physical
    path = journal(tmp_path, [
        ("llm.format_repair", {"agent": "actor"}),
        ("reviewer.return", {"route": "translator_then_actor"}),
        ("attempt.completed", {"result": terminal}),
    ])
    assert capture_diagnostics(path)["invalid_schema_agents"] == ["actor"]
    record = {"row": {"run_id": "ep-001-r01", "episode_id": "ep-001", "interaction_journal": str(path)},
              "journal": {"first_specification": terminal["task_specification"], "first_plan": None,
                          "first_physics_evidence": None, "first_physics_after_review_return": True}}
    metrics = evaluate_agents([record], CASE, SCORER)["metrics"]
    assert all(metrics[k]["value"] == 1 for k in (
        "actor_plan_legality", "actor_tool_chain_success", "actor_evidence_consistency"))
    assert terminal["workflow_status"] == "blocked"  # Diagnostics do not release the task.


@pytest.mark.parametrize("payload_key", ["result", "candidate"])
def test_terminal_candidate_uses_its_own_specification(tmp_path, physical, payload_key):
    truth, numeric, terminal = physical
    path = journal(tmp_path, [("attempt.failed", {payload_key: terminal})])
    selected = read_terminal_actor_result(path)
    assert selected["event"] == "attempt.failed" and selected["event_id"] == "0"
    sample = actor_terminal_diagnostic(selected["result"], truth, numeric, SCORER)
    assert sample["plan_legal"] and sample["tool_chain_success"]
    assert sample["evidence_correct"] == sample["evidence_fields"]
    stale = deepcopy(terminal)
    stale["task_specification"]["target_buses"] = [33]
    assert not actor_terminal_diagnostic(stale, truth, numeric, SCORER)["plan_legal"]


@pytest.mark.parametrize("final_payload", [{}, {"candidate": None}, {"result": None}])
def test_last_missing_terminal_never_falls_back_to_earlier_success(tmp_path, physical, final_payload):
    truth, numeric, terminal = physical
    path = journal(tmp_path, [("attempt.completed", {"result": terminal}),
                              ("attempt.failed", final_payload)])
    selected = read_terminal_actor_result(path)
    assert selected["result"] is None and selected["event_id"] == "1"
    assert capture_diagnostics(path)["terminal_actor_result"] is None
    sample = actor_terminal_diagnostic(selected["result"], truth, numeric, SCORER)
    assert not sample["plan_legal"] and not sample["tool_chain_success"]
    assert sample["evidence_correct"] == 0 and sample["evidence_fields"] > 0


def test_terminal_failure_is_used_even_when_an_earlier_plan_was_valid(tmp_path, physical):
    truth, numeric, terminal = physical
    failed = deepcopy(terminal)
    failed["execution_plan"]["valid"] = False
    failed["physics_evidence"] = None
    path = journal(tmp_path, [("attempt.completed", {"result": terminal}),
                              ("attempt.failed", {"candidate": failed})])
    sample = actor_terminal_diagnostic(read_terminal_actor_result(path)["result"], truth, numeric, SCORER)
    assert not sample["plan_legal"] and not sample["tool_chain_success"] and sample["evidence_correct"] == 0


def test_hash_binding_and_numeric_tolerances_remain_enforced(physical):
    truth, numeric, terminal = physical
    changed = deepcopy(terminal)
    changed["physics_evidence"]["execution_plan_sha256"] = "unbound"
    sample = actor_terminal_diagnostic(changed, truth, numeric, SCORER)
    assert sample["plan_legal"] and not sample["tool_chain_success"]
    changed = deepcopy(terminal)
    values = changed["physics_evidence"]["business_metrics"]["requested_output_values"]
    key = next(iter(values))
    values[key] += 10
    changed["physics_evidence"]["hard_constraints_passed"] = not numeric["final_decision"]["release_eligible"]
    sample = actor_terminal_diagnostic(changed, truth, numeric, SCORER)
    assert sample["evidence_correct"] == sample["evidence_fields"] - 2
    changed = deepcopy(terminal)
    changed["physics_evidence"]["business_metrics"]["requested_output_values"][key] += 1e-9
    sample = actor_terminal_diagnostic(changed, truth, numeric, SCORER)
    assert sample["evidence_correct"] == sample["evidence_fields"]


def test_missing_outcomes_keep_denominators_and_clarifications_are_excluded(physical):
    truth, numeric, terminal = physical
    good = dict(run_id="good", **actor_terminal_diagnostic(terminal, truth, numeric, SCORER))
    missing = dict(run_id="missing", **actor_terminal_diagnostic(None, truth, numeric, SCORER))
    clarify = next(t for t in CASE.iter_groundtruth() if t.perfect_execution.route == "clarify_sender")
    excluded = dict(run_id="clarify", **actor_terminal_diagnostic(terminal, clarify, None, SCORER))
    metrics = aggregate_actor_terminal([good, missing, excluded])
    assert all(v["value"] == .5 for v in metrics.values())
    assert metrics["actor_plan_legality"]["denominator"] == 2
    assert metrics["actor_evidence_consistency"]["denominator"] == 2 * good["evidence_fields"]
    no_reference = dict(run_id="no-reference", **actor_terminal_diagnostic(terminal, truth, None, SCORER))
    unavailable = aggregate_actor_terminal([good, no_reference])["actor_evidence_consistency"]
    assert unavailable["value"] is None
    assert unavailable["missing_reference_runs"] == ["no-reference"]
