"""Offline Reviewer diagnosis, actual dispatch and verified repair metrics.

Independent reference checks and event extraction shared by the final Reviewer
feedback indicators. Formal reports use reviewer_feedback_metrics. Reference checks must not read
Reviewer findings, and this module never changes runtime agents or raw records.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

from b2g_agent.agents.actor import PlanCompiler, SUPPORTED_REQUESTED_OUTPUTS
from b2g_agent.contracts import ExecutionPlan, TaskSpecification
from b2g_agent.contracts.outcomes import rejection_causes, validated_non_execution_report
from b2g_agent.evaluation.physics_cache import execution_plan_sha256
from b2g_agent.publication import recorded_hash_matches
from b2g_agent.artifact_paths import resolve_recorded_path


PROTOCOL_ID = "reviewer-process-offline-v1.1"


def phase_of(inputs):
    post_physics = inputs.get("simulation_status") in {"failed", "non_converged", "severe_errors", "completed", "converged"}
    return "reverse" if (inputs.get("semantic_state") or inputs.get("physics_evidence")
        or inputs.get("require_physics_evidence") or post_physics
        or inputs.get("review_direction") == "receiver_feedback") else "forward"


def extract_reviews(events):
    pending, reviews = {}, []
    for index, event in enumerate(events):
        data, kind = event["payload"], event["event"]
        if kind == "stage.started" and data.get("stage") == "reviewer.review":
            pending[event["span_id"]] = dict(start=index, inputs=data["inputs"],
                start_event_id=event["event_id"], span_id=event["span_id"])
        elif kind in {"stage.completed", "stage.failed"} and data.get("stage") == "reviewer.review":
            review = pending.pop(event["span_id"], None)
            if review is not None:
                review.update(end=index, output=data.get("output"), phase=phase_of(review["inputs"]),
                              end_event_id=event["event_id"], failed=kind == "stage.failed")
                reviews.append(review)
    for review in pending.values():
        review.update(end=len(events), output=None, phase=phase_of(review["inputs"]),
                      end_event_id=None, failed=True)
        reviews.append(review)
    return sorted(reviews, key=lambda r: r["start"])


def independent_reference(inputs, truth, numeric, scorer, *, output_check="required-supported"):
    """Recheck actual contract/plan/evidence, without using model diagnoses."""
    if truth.perfect_execution.route == "clarify_sender":
        return dict(label="clarification", semantic=[], evidence=[], unknown=[],
                    required_targets=[], source_requires_clarification=True)
    semantic, evidence, unknown = [], [], []
    try:
        spec = TaskSpecification.model_validate(inputs.get("specification"))
    except (ValueError, TypeError):
        return dict(label="semantic", semantic=["contract.invalid_schema"], evidence=[], unknown=[],
                    required_targets=["translator", "actor"], source_requires_clarification=False)
    fidelity = scorer._first_transfer(truth, SimpleNamespace(task_specification=spec))
    checks = dict(fidelity["field_checks"])
    if output_check not in {"exact", "required-supported"}:
        raise ValueError("Unknown output reference policy")
    if output_check == "required-supported" and "requested_outputs" in checks:
        # Review of evidence coverage should not treat additional supported
        # evidence as an omitted or fabricated required output. Unsupported
        # additions remain errors below. This does not change quality scoring.
        checks["requested_outputs"] = set(truth.task_specification.requested_outputs) <= set(spec.requested_outputs)
    semantic.extend("contract." + name for name, correct in checks.items() if not correct)
    if spec.clarification_required:
        semantic.append("contract.unnecessary_clarification")
    semantic.extend("contract.unsupported_output." + name
                    for name in sorted(set(spec.requested_outputs) - SUPPORTED_REQUESTED_OUTPUTS))
    try:
        plan = ExecutionPlan.model_validate(inputs.get("plan"))
        errors = PlanCompiler().validate(plan, spec)
        if not plan.valid or errors:
            evidence.append("plan.invalid")
    except (ValueError, TypeError):
        plan = None
        evidence.append("plan.missing_or_invalid_schema")
    physical = inputs.get("physics_evidence") or {}
    state = inputs.get("semantic_state") or {}
    if physical and state:
        for name, expected in (("decision", physical.get("receiver_decision")),
                ("committed_power", physical.get("committed_power_kw")),
                ("completion_percentage", physical.get("completion_percentage"))):
            observed = state.get("decision") if name == "decision" else state.get("fields", {}).get(name, {}).get("value")
            if not scorer._equivalent(expected, observed):
                semantic.append("reverse." + name)
    if physical:
        if not physical.get("converged"):
            evidence.append("physics.nonconvergence")
        if not physical.get("hard_constraints_passed") and not validated_non_execution_report(physical):
            evidence.append("physics.unsafe_action")
        if physical.get("receiver_decision") == "rejected":
            causes = rejection_causes(physical)
            if not causes or physical.get("execution_log", {}).get("rejection_causes") != causes:
                evidence.append("physics.missing_verified_rejection_causes")
        if plan is not None and (physical.get("execution_plan_id") != plan.plan_id
                or physical.get("execution_plan_sha256") != execution_plan_sha256(plan)):
            evidence.append("physics.execution_plan_not_bound")
        try:
            if not recorded_hash_matches(physical["evidence_path"], physical.get("evidence_sha256")):
                evidence.append("physics.provenance_mismatch")
        except (OSError, KeyError, TypeError):
            unknown.append("physics.provenance_file_unavailable")
        if numeric is None:
            unknown.append("physics.numeric_reference_unavailable")
        else:
            observed = physical.get("business_metrics", {}).get("requested_output_values", {})
            expected = numeric["business_metrics"].get("requested_output_values", {})
            evidence.extend("physics.numeric_output." + name for name, value in expected.items()
                            if not scorer._equivalent(value, observed.get(name)))
        evidence.extend("physics." + name for name in ("time_alignment_passed", "bem_bus_mapping_passed")
                        if physical.get("execution_log", {}).get(name) is False)
    elif phase_of(inputs) == "reverse":
        evidence.append("physics.missing_required_evidence")
    # Absence of physics is expected during the initial forward stage.
    targets = []
    if any(k.startswith("contract.") for k in semantic):
        targets.append("translator")
    if any(k.startswith("reverse.") for k in semantic):
        targets.append("translator_feedback")
    if evidence or ("translator" in targets and phase_of(inputs) == "forward"):
        targets.append("actor")
    label = "semantic" if semantic else "evidence" if evidence else "unknown" if unknown else "clean"
    return dict(label=label, semantic=sorted(set(semantic)), evidence=sorted(set(evidence)),
                unknown=unknown, required_targets=targets, source_requires_clarification=False)


def predicted_label(decision, phase):
    if not decision:
        return "missing"
    if decision.get("routing_decision") == "clarify_sender":
        return "clarification"
    issues = decision.get("issues", [])
    semantic = any(i.get("responsible_agent") in {"translator", "human"} for i in issues)
    semantic |= float(decision.get("semantic_risk_score", 0)) > float(decision.get("semantic_risk_threshold", .25))
    evidence = any(i.get("responsible_agent") == "actor" and
                   not (phase == "forward" and i.get("code") == "E-MISSING-EVIDENCE") for i in issues)
    if phase == "reverse":
        evidence |= float(decision.get("evidence_risk_score", 0)) > float(decision.get("evidence_risk_threshold", 0))
    return "semantic" if semantic else "evidence" if evidence else "clean"


def feedback_matches(inputs, stage, decision):
    """Require the actual recipient inputs to contain the triggering findings."""
    issues = decision.get("issues", [])
    if not issues:
        return False
    if stage == "actor.plan":
        feedback = inputs.get("reviewer_feedback") or []
        return all(issue in feedback for issue in issues)
    messages = [m for m in inputs.get("conversation_history") or [] if m.get("speaker") == "reviewer"]
    if not messages:
        return False
    return all(issue.get("code", "") in messages[-1].get("text", "") for issue in issues)


def extract_dispatches(events):
    dispatches = []
    returns = [i for i, e in enumerate(events) if e["event"] == "reviewer.return"]
    for offset, index in enumerate(returns):
        payload = events[index]["payload"]
        end = returns[offset + 1] if offset + 1 < len(returns) else len(events)
        decision = payload.get("decision") or {}
        calls = []
        for position in range(index + 1, end):
            event = events[position]
            data = event["payload"]
            if event["event"] != "stage.started":
                continue
            stage = data.get("stage")
            target = {"translator.translate": "translator", "translator.translate_feedback": "translator_feedback",
                      "actor.plan": "actor"}.get(stage)
            if target and feedback_matches(data.get("inputs", {}), stage, decision):
                calls.append(dict(index=position, event_id=event["event_id"], target=target,
                                  stage=stage, span_id=event.get("span_id")))
        dispatches.append(dict(index=index, event_id=events[index]["event_id"],
            direction=payload.get("direction"), decision=decision, calls=calls))
    return dispatches


def audit_opportunity(initial, reviews, dispatches):
    """Count one first-draft opportunity per run/phase; never best-of-revisions."""
    ref, decision = initial["reference"], initial.get("output") or {}
    required = ref["required_targets"]
    route = decision.get("routing_decision")
    expected_routes = {"translator", "translator_then_actor"} if ref["semantic"] else {"actor"}
    correct_diagnosis = predicted_label(decision, initial["phase"]) == ref["label"] and route in expected_routes
    next_review = min((r["start"] for r in reviews if r["start"] > initial["end"]), default=float("inf"))
    matching = [d for d in dispatches if initial["end"] < d["index"] < next_review and d["decision"] == decision]
    # Only the dispatch belonging to the initial diagnosis can start this chain.
    first = min(matching, key=lambda d: d["index"]) if matching else None
    call_evidence, dispatch_evidence = [], []
    if first is not None and correct_diagnosis:
        allowed = {"forward"} if initial["phase"] == "forward" else {"reverse", "physics", "receiver_feedback"}
        chain = [first]
        # A reverse Translator-then-Actor route can require a new review between
        # the Translator call and Actor handoff; preserve that legitimate order.
        if initial["phase"] == "reverse" and route == "translator_then_actor":
            chain += [d for d in dispatches if d["index"] > first["index"]
                      and d["direction"] in allowed and d["decision"].get("routing_decision") in {"actor", "translator_then_actor"}]
        cursor = first["index"]
        for target in required:
            candidates = [(d, c) for d in chain for c in d["calls"]
                          if d["direction"] in allowed and c["target"] == target and c["index"] > cursor]
            if not candidates:
                break
            dispatch, call = min(candidates, key=lambda pair: pair[1]["index"])
            call_evidence.append(call)
            dispatch_evidence.append(dispatch["event_id"])
            cursor = call["index"]
    dispatched = bool(required) and correct_diagnosis and len(call_evidence) == len(required)
    closure, closure_review, unknown_reviews = False, None, []
    if dispatched:
        last_call = call_evidence[-1]["index"]
        for review in reviews:
            if review["start"] <= last_call or review["phase"] != initial["phase"]:
                continue
            candidate = review["reference"]
            if candidate["unknown"]:
                unknown_reviews.append(review["end_event_id"])
                continue
            if candidate["semantic"] or candidate["evidence"] or candidate["label"] != "clean":
                continue
            output = review.get("output") or {}
            expected = {"actor"} if initial["phase"] == "forward" else {"release"}
            if predicted_label(output, review["phase"]) == "clean" and output.get("routing_decision") in expected:
                closure, closure_review = True, review["end_event_id"]
                break
    return dict(phase=initial["phase"], initial_review_event_id=initial["end_event_id"],
        reference=ref, predicted_label=predicted_label(decision, initial["phase"]),
        predicted_route=route, correct_diagnosis=correct_diagnosis,
        dispatched=dispatched, closed=closure, recipient_calls=call_evidence,
        dispatch_event_ids=sorted(set(dispatch_evidence)), closure_review_event_id=closure_review,
        uncertain_closure_review_event_ids=unknown_reviews,
        reason="closed" if closure else "repair_not_independently_verified" if dispatched else
               "no_matching_actual_return" if first is None else "wrong_diagnosis_or_incomplete_recipient_chain")
