"""Post-run agent diagnostics from the same benchmark interactions.

Ground truth is read here only, never by runtime agents. Programmatic labels
describe contract/evidence agreement; they are not human judgments of every
natural-language nuance. Translator retains first-draft sampling; Reviewer measures feedback impact; Actor
uses the terminal result/candidate after the closed loop, including blocked runs.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
from statistics import mean
from types import SimpleNamespace

from b2g_agent.agents.actor import PlanCompiler
from b2g_agent.contracts import ExecutionPlan, TaskSpecification
from b2g_agent.contracts.outcomes import validated_non_execution_report, rejection_causes
from b2g_agent.evaluation.physics_cache import execution_plan_sha256
from b2g_agent.publication import recorded_hash_matches
from b2g_agent.artifact_paths import resolve_recorded_path
from b2g_agent.evaluation.reviewer_feedback_metrics import (
    REVIEWER_METRICS, feedback_outcome, reviewer_feedback_metrics,
)


NINE_METRICS = (
    "translator_required_field_f1", "translator_mapping_accuracy", "translator_clarification_f1",
    "actor_plan_legality", "actor_tool_chain_success", "actor_evidence_consistency",
    *REVIEWER_METRICS,
)

AGENT_SAMPLING_SCOPE = (
    "Translator: first forward contract; Reviewer: appropriate handling and source-executable outcomes; "
    "Actor: terminal result/candidate after closed-loop revisions, never best-of-revisions; "
    "all scheduled applicable attempts retained, including blocked and missing outcomes."
)


def read_terminal_actor_result(path: Path) -> dict:
    """Select only the last attempt terminal event; never borrow an earlier success."""
    selected = {"result": None, "event_id": None, "event": None}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            event = json.loads(line)
            if event["event"] in {"attempt.completed", "attempt.failed"}:
                payload = event["payload"]
                selected = {"result": payload.get("result") or payload.get("candidate"),
                            "event_id": event.get("event_id"), "event": event["event"]}
    return selected


def actor_terminal_diagnostic(terminal, truth, numeric, scorer):
    """Same legality/execution/tolerance checks, applied to one terminal snapshot.

    A blocked candidate can have valid physical evidence without being an
    externally released task. Interaction-quality success remains separate.
    """
    if truth.perfect_execution.route == "clarify_sender":
        return {"applicable": False}
    terminal = terminal or {}
    errors = []
    try:
        plan = ExecutionPlan.model_validate(terminal.get("execution_plan"))
        specification = TaskSpecification.model_validate(terminal.get("task_specification"))
        errors = PlanCompiler().validate(plan, specification)
        plan_ok = plan.valid and not errors
    except (ValueError, TypeError):
        plan, plan_ok = None, False
        errors = ["missing_or_invalid_terminal_plan_or_specification"]
    ev = terminal.get("physics_evidence") or {}
    outputs = ev.get("business_metrics", {}).get("requested_output_values", {})
    actions = ev.get("execution_log", {}).get("actions", [])
    completed_ids = {a.get("action_id") for a in actions if a.get("status") == "completed"}
    executed = bool(plan_ok and ev and actions and plan
                    and {a.action_id for a in plan.actions} <= completed_ids
                    and ev.get("execution_plan_sha256") == execution_plan_sha256(plan)
                    and all(a.get("status") == "completed" for a in actions)
                    and all(outputs.get(k) is not None for k in truth.task_specification.requested_outputs))
    expected = numeric["business_metrics"].get("requested_output_values", {}) if numeric else {}
    checks = {k: bool(ev) and scorer._equivalent(value, outputs.get(k)) for k, value in expected.items()}
    checks["hard_constraints_passed"] = bool(ev and numeric is not None
        and bool(ev.get("hard_constraints_passed")) == bool(numeric["final_decision"]["release_eligible"]))
    return {"applicable": True, "terminal_result_present": bool(terminal),
            "terminal_evidence_present": bool(ev), "plan_legal": bool(plan_ok),
            "plan_validation_errors": errors, "tool_chain_success": executed,
            "evidence_correct": sum(checks.values()), "evidence_fields": len(expected) + 1,
            "evidence_field_checks": checks, "numeric_reference_available": numeric is not None}


def aggregate_actor_terminal(samples):
    applicable = [r for r in samples if r["applicable"]]
    metrics = {
        "actor_plan_legality": ratio(sum(r["plan_legal"] for r in applicable), len(applicable)),
        "actor_tool_chain_success": ratio(sum(r["tool_chain_success"] for r in applicable), len(applicable)),
        "actor_evidence_consistency": ratio(sum(r["evidence_correct"] for r in applicable),
                                             sum(r["evidence_fields"] for r in applicable)),
    }
    missing = [r["run_id"] for r in applicable if not r["numeric_reference_available"]]
    if missing:
        metrics["actor_evidence_consistency"].update(value=None, status="incomplete_numeric_reference",
                                                     missing_reference_runs=missing)
    return metrics


def ratio(numerator, denominator, **extra):
    return {"value": numerator / denominator if denominator else None,
            "numerator": numerator, "denominator": denominator, **extra}


def capture_diagnostics(path: Path) -> dict:
    pending, reviews, durations = {}, {}, defaultdict(float)
    first_agent_done = set()
    observed_stages, invalid_agents = set(), set()
    started_stages = set()
    invalid_review_spans = set()
    translator_retrieval = 0.0
    measured_retrieval, retrieval_events = 0.0, 0
    first_actor_input = None
    terminal_actor_result = None
    for line in path.open(encoding="utf-8"):
        if not line.strip():
            continue
        event = json.loads(line)
        kind, data, span = event["event"], event["payload"], event.get("span_id")
        if kind in {"attempt.completed", "attempt.failed"}:
            terminal_actor_result = data.get("result") or data.get("candidate")
        if kind == "retrieval.completed" and data.get("agent") in {"translator", "single_agent"}:
            measured_retrieval += float(data["elapsed_seconds"])
            retrieval_events += 1
        if kind == "llm.format_repair" and data["agent"] not in first_agent_done:
            invalid_agents.add(data["agent"])
        if kind == "reviewer.response_invalid":
            invalid_review_spans.update(k for k, v in pending.items() if v["stage"] == "reviewer.review")
        if kind == "stage.started":
            started_stages.add(data["stage"])
            pending[span] = {**data, "timestamp": event["timestamp"]}
            if data["stage"] in {"actor.plan", "actor.free_plan"} and first_actor_input is None:
                first_actor_input = data["inputs"].get("specification")
        elif kind in {"stage.completed", "stage.failed"}:
            started = pending.pop(span, None)
            if started is None:
                continue
            stage = data["stage"]
            elapsed = max(0, float(data["elapsed_seconds"]) if "elapsed_seconds" in data else
                (datetime.fromisoformat(event["timestamp"]) - datetime.fromisoformat(started["timestamp"])).total_seconds())
            durations[stage] += elapsed
            observed_stages.add(stage)
            output = data.get("output")
            if kind == "stage.completed":
                stage_agent = {"translator.translate": "translator", "actor.plan": "actor",
                               "actor.free_plan": "free_plan_actor", "translator.translate_feedback": "translator_feedback",
                               "single_agent.contract": "same_llm_single_agent_contract",
                               "single_agent.feedback": "same_llm_single_agent_result"}.get(stage)
                if stage_agent:
                    first_agent_done.add(stage_agent)
            if stage in {"translator.translate", "translator.translate_feedback"} and output:
                translator_retrieval += float((output.get("retrieval_trace") or {}).get("elapsed_seconds") or 0)
            if stage == "reviewer.review":
                inputs = started["inputs"]
                phase = "reverse" if inputs.get("semantic_state") or inputs.get("physics_evidence") else "forward"
                reviews.setdefault(phase, {"phase": phase, "inputs": inputs,
                    "prediction": None if span in invalid_review_spans else output,
                    "event_id": event["event_id"], "format_repaired": span in invalid_review_spans})
    return {"review_samples": list(reviews.values()), "invalid_schema_agents": sorted(invalid_agents),
            "stage_seconds": dict(durations),
            "first_actor_specification": first_actor_input,
            "terminal_actor_result": terminal_actor_result,
            "translator_retrieval_seconds": (durations["retrieval.translator"] if "retrieval.translator" in observed_stages
                else measured_retrieval if retrieval_events else translator_retrieval),
            "translator_retrieval_timing_source": ("runtime_spans" if "retrieval.translator" in observed_stages
                else "runtime_events" if retrieval_events else "legacy_output_trace"),
            "reviewer_retrieval_seconds": (durations["retrieval.reviewer_memory"] if "retrieval.reviewer_memory" in observed_stages
                else 0.0 if "reviewer.review" not in started_stages else None)}


def mapping_fields(reference, predicted, scorer):
    """Per object-variable mapping: all FIVE dimensions must agree together."""
    variables = [name for name, field in reference.fields.items()
                 if field.unit is not None or name == "target_power"]
    directions = lambda spec: (
        spec.sender_role, spec.receiver_role, spec.task_families,
        sorted((o.metric, o.operator) for o in spec.objectives),
    )
    expected_pairs = list(zip(reference.target_bems, reference.target_buses))
    observed_pairs = dict(zip(predicted.target_bems, predicted.target_buses)) if predicted else {}
    def resolution(spec):
        field = spec.fields.get("time_resolution")
        return (field.value, field.unit) if field else None
    rows = []
    for bem, bus in expected_pairs:
        for variable in variables:
            expected = reference.fields[variable]
            observed = predicted.fields.get(variable) if predicted else None
            checks = {
                "object": observed_pairs.get(bem) == bus,
                "variable": observed is not None,
                "direction": predicted is not None and scorer._equivalent(directions(reference), directions(predicted)),
                "unit": observed is not None and observed.unit == expected.unit,
                "time_scale": predicted is not None and scorer._equivalent(reference.windows, predicted.windows)
                    and scorer._equivalent(reference.operating_day, predicted.operating_day)
                    and scorer._equivalent(resolution(reference), resolution(predicted)),
            }
            rows.append({"object": bem, "target_bus": bus, "variable": variable,
                         "checks": checks, "correct": all(checks.values())})
    return rows


def predicted_diagnosis(decision):
    if not decision:
        return "missing", "missing"
    semantic = any(i.get("responsible_agent") in {"translator", "human"} for i in decision.get("issues", []))
    semantic |= float(decision.get("semantic_risk_score", 0)) > float(decision.get("semantic_risk_threshold", 0.25))
    evidence = any(i.get("responsible_agent") == "actor" for i in decision.get("issues", []))
    evidence |= float(decision.get("evidence_risk_score", 0)) > float(decision.get("evidence_risk_threshold", 0))
    label = "semantic" if semantic else "evidence" if evidence else "pass"
    route = decision.get("routing_decision")
    grouped_route = "Translator" if route in {"translator", "translator_then_actor", "clarify_sender"} else "Actor" if route == "actor" else "release" if route == "release" else "missing"
    return label, grouped_route


def reference_review(inputs, truth, numeric, scorer):
    """Label reviewed input independently of the Reviewer's predicted findings."""
    reasons = []
    try:
        specification = TaskSpecification.model_validate(inputs["specification"])
    except (ValueError, KeyError, TypeError):
        return "semantic", "Translator", ["invalid_task_schema"]
    if truth.perfect_execution.route == "clarify_sender":
        return "semantic", "Translator", ["source_requires_clarification"]
    fidelity = scorer._first_transfer(truth, SimpleNamespace(task_specification=specification))
    if fidelity["f1"] < 1:
        reasons.append("contract_differs_from_reference")
    state, evidence = inputs.get("semantic_state") or {}, inputs.get("physics_evidence")
    if state and evidence:
        for name, expected in (("decision", evidence.get("receiver_decision")),
                               ("committed_power", evidence.get("committed_power_kw")),
                               ("completion_percentage", evidence.get("completion_percentage"))):
            observed = state.get("decision") if name == "decision" else state.get("fields", {}).get(name, {}).get("value")
            if not scorer._equivalent(expected, observed):
                reasons.append("reverse_" + name)
    if reasons:
        return "semantic", "Translator", reasons
    try:
        plan = ExecutionPlan.model_validate(inputs.get("plan"))
        if not plan.valid or PlanCompiler().validate(plan, specification):
            reasons.append("invalid_plan")
    except (ValueError, TypeError):
        plan = None
        reasons.append("missing_or_invalid_plan")
    if not evidence:
        reasons.append("physical_evidence_not_yet_available")
    else:
        if not evidence.get("converged"):
            reasons.append("nonconvergence")
        if not evidence.get("hard_constraints_passed") and not validated_non_execution_report(evidence):
            reasons.append("unsafe_action")
        if evidence.get("receiver_decision") == "rejected":
            causes = rejection_causes(evidence)
            if not causes or evidence.get("execution_log", {}).get("rejection_causes") != causes:
                reasons.append("missing_verified_rejection_causes")
        if plan is not None and (evidence.get("execution_plan_id") != plan.plan_id
                or evidence.get("execution_plan_sha256") != execution_plan_sha256(plan)):
            reasons.append("execution_plan_not_bound")
        try:
            if not recorded_hash_matches(evidence["evidence_path"], evidence.get("evidence_sha256")):
                reasons.append("provenance_mismatch")
        except (OSError, KeyError, TypeError):
            reasons.append("missing_provenance")
        if numeric is None:
            return None, None, ["numeric_reference_unavailable"]
        actual = evidence.get("business_metrics", {}).get("requested_output_values", {})
        expected = numeric["business_metrics"].get("requested_output_values", {})
        if any(not scorer._equivalent(value, actual.get(name)) for name, value in expected.items()):
            reasons.append("numeric_evidence_disagrees_with_oracle")
        if any(evidence.get("execution_log", {}).get(k) is False for k in ("time_alignment_passed", "bem_bus_mapping_passed")):
            reasons.append("invalid_execution_alignment")
    return ("evidence", "Actor", reasons) if reasons else ("pass", "release", [])


def evaluate_agents(records, case, scorer, label_overrides=None, *, method_id="full_multi_agent"):
    single = method_id == "same_llm_single_agent"
    field_f1, mappings, actor_samples = [], [], []
    clarity = Counter()
    samples, mapping_audit, costs = [], [], []
    feedback_rows = []
    overrides = label_overrides or {}
    used_overrides = set()
    for record in records:
        row, journal = record["row"], record["journal"]
        truth = case.get_groundtruth(row["episode_id"])
        must_clarify = truth.perfect_execution.route == "clarify_sender"
        diagnostic = capture_diagnostics(resolve_recorded_path(row["interaction_journal"]))
        record["diagnostics"] = diagnostic
        first = journal.get("first_specification")
        invalid = diagnostic["invalid_schema_agents"]
        if single and "same_llm_single_agent_contract" in invalid:
            invalid = [*invalid, "translator", "actor"]
        actual_spec = TaskSpecification.model_validate(first) if first else None
        spec = actual_spec if "translator" not in invalid else None
        field_f1.append(scorer._first_transfer(truth, SimpleNamespace(task_specification=spec))["f1"] if spec else 0)
        mapping = mapping_fields(truth.task_specification, spec, scorer)
        mappings.extend(mapping)
        mapping_audit.append({"run_id": row["run_id"], "fields": mapping})
        actual = spec.clarification_required if spec else not must_clarify
        clarity["tp" if must_clarify and actual else "fn" if must_clarify else "fp" if actual else "tn"] += 1
        clarity["missing"] += spec is None
        numeric_path = case.case_dir / "groundtruth/numeric_oracles" / f"{row['episode_id']}.json"
        numeric = json.loads(numeric_path.read_text(encoding="utf-8")) if numeric_path.is_file() else None
        if not single:
            events = [json.loads(line) for line in resolve_recorded_path(row["interaction_journal"]).read_text(encoding="utf-8").splitlines() if line.strip()]
            feedback_rows.append(feedback_outcome(
                events, truth, numeric, scorer, record.get("score"),
                run_id=row["run_id"], workflow_status=row.get("workflow_status")))
        actor_samples.append(dict(run_id=row["run_id"], **actor_terminal_diagnostic(
            diagnostic["terminal_actor_result"], truth, numeric, scorer)))
        for sample in ([] if single else diagnostic["review_samples"]):
            label, route, reasons = reference_review(sample["inputs"], truth, numeric, scorer)
            sample_id = f"{row['run_id']}:{sample['phase']}"
            input_hash = hashlib.sha256(json.dumps(sample["inputs"], sort_keys=True,
                ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
            origin = "independent_programmatic_reference"
            if sample_id in overrides:
                override = overrides[sample_id]
                expected_routes = {"semantic": "Translator", "evidence": "Actor", "pass": "release"}
                if (override.get("reference_label") not in expected_routes
                        or override.get("reference_route") != expected_routes.get(override.get("reference_label"))
                        or not override.get("annotator") or override.get("reviewed_input_sha256") != input_hash):
                    raise ValueError(f"Invalid independently annotated review label: {sample_id}")
                used_overrides.add(sample_id)
                label, route, origin = override["reference_label"], override["reference_route"], "annotator:" + override["annotator"]
            prediction, prediction_route = predicted_diagnosis(sample["prediction"])
            samples.append({"sample_id": sample_id, "run_id": row["run_id"], "phase": sample["phase"],
                "reference_label": label, "reference_route": route, "reference_basis": reasons,
                "label_origin": origin, "predicted_label": prediction, "predicted_route": prediction_route,
                "reviewed_input_sha256": input_hash,
                "source_event_id": sample["event_id"], "reviewed_input": sample["inputs"], "reviewer_output": sample["prediction"]})
        costs.append({"run_id": row["run_id"], "per_agent_usage": row.get("per_agent_llm_usage", {}),
            "translator_retrieval_seconds": diagnostic["translator_retrieval_seconds"],
            "reviewer_retrieval_seconds": diagnostic["reviewer_retrieval_seconds"],
            "elapsed_seconds": row.get("elapsed_seconds"), "cache_status": row.get("physics_cache_status"),
            "actual_tool_seconds": {k: v for k, v in diagnostic["stage_seconds"].items() if k.startswith("tool.")}})
    if set(overrides) != used_overrides:
        raise ValueError("Diagnostic overrides do not match recorded review samples")
    metrics = {
        "translator_required_field_f1": ratio(sum(field_f1), len(records)),
        "translator_mapping_accuracy": ratio(sum(m["correct"] for m in mappings), len(mappings)),
        "translator_clarification_f1": ratio(2 * clarity["tp"], 2 * clarity["tp"] + clarity["fp"] + clarity["fn"], confusion=dict(clarity)),
        **aggregate_actor_terminal(actor_samples),
        **reviewer_feedback_metrics(feedback_rows),
    }
    assert set(metrics) == set(NINE_METRICS)
    if single:
        for name in NINE_METRICS[-3:]:
            metrics[name] = dict(value=None, numerator=None, denominator=0,
                status="not_applicable_no_reviewer",
                reason="Single-agent baseline has no LLM Reviewer; deterministic guards are not agent predictions.")
    expected_reviews = 0 if single else len(records) + sum(r["applicable"] for r in actor_samples)
    return {"metrics": metrics, "attempts": len(records), "review_samples_observed": len(samples),
            "review_samples_expected": expected_reviews, "review_samples_missing": max(0, expected_reviews - len(samples)),
            "scope": AGENT_SAMPLING_SCOPE,
            "label_limit": "Programmatic contract/evidence reference; free-text meaning is not independently human-annotated unless overridden.",
            "reviewer_samples": samples, "mapping_audit": mapping_audit, "computation_cost_records": costs}
