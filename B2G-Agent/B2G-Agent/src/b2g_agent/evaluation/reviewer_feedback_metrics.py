"""Final Reviewer feedback-impact indicators, evaluated on recorded interactions."""
from b2g_agent.evaluation.reviewer_process_metrics import (
    audit_opportunity, extract_dispatches, extract_reviews, independent_reference,
)

REVIEWER_METRICS = (
    "reviewer_first_review_appropriate_handling",
    "reviewer_executable_reverse_review_reach",
    "reviewer_executable_task_success",
)

def first_reviews(events: list[dict], truth, numeric, scorer, policy: str):
    reviews = extract_reviews(events)
    for review in reviews:
        review["reference"] = independent_reference(
            review["inputs"], truth, numeric, scorer, output_check=policy
        )
    first = {}
    for review in reviews:
        first.setdefault(review["phase"], review)
    if "forward" not in first:
        drafts = [e["payload"].get("output") for e in events
                  if e["event"] == "stage.completed"
                  and e["payload"].get("stage") == "translator.translate"]
        if drafts:
            plans = [e["payload"].get("output") for e in events
                     if e["event"] == "stage.completed"
                     and e["payload"].get("stage") == "actor.plan"]
            inferred = {"start": -1, "end": len(events),
                        "inputs": {"specification": drafts[0],
                                   "plan": plans[0] if plans else None,
                                   "simulation_status": "not_run"},
                        "output": None, "phase": "forward", "end_event_id": None}
            inferred["reference"] = independent_reference(
                inferred["inputs"], truth, numeric, scorer, output_check=policy
            )
            first["forward"] = inferred
    return reviews, first

def feedback_outcome(events, truth, numeric, scorer, score, *, run_id="", workflow_status=None):
    score = score or {}
    reviews, first = first_reviews(events, truth, numeric, scorer, "required-supported")
    initial = first.get("forward")
    affected = bool(initial and (initial["reference"]["semantic"] or initial["reference"]["evidence"]))
    audit = audit_opportunity(initial, reviews, extract_dispatches(events)) if affected else None
    later_forward = ([review for review in reviews
                      if affected and review["phase"] == "forward"
                      and review["start"] > initial["end"]])
    final_forward = later_forward[-1] if later_forward else None
    error_eliminated = bool(final_forward
        and final_forward["reference"]["label"] == "clean"
        and not final_forward["reference"]["semantic"]
        and not final_forward["reference"]["evidence"]
        and not final_forward["reference"]["unknown"])
    first_route = ((initial.get("output") or {}).get("routing_decision")
                   if initial else None)
    reached_reverse = any(review["phase"] == "reverse" for review in reviews)
    task_success = bool(score.get("task_success"))
    reference_label = initial["reference"]["label"] if initial else None
    appropriate_handling = bool(
        (reference_label == "clarification" and first_route == "clarify_sender" and task_success)
        or (reference_label == "clean" and first_route == "actor" and reached_reverse)
        or (reference_label in {"semantic", "evidence"} and error_eliminated)
    )
    return {
        "run_id": run_id,
        "affected": affected,
        "source_executable": truth.perfect_execution.route != "clarify_sender",
        "reference_label": reference_label,
        "first_forward_route": first_route,
        "appropriate_handling": appropriate_handling,
        "error_eliminated": error_eliminated,
        "repaired": bool(audit and audit["closed"]),
        "correct_feedback_delivered": bool(audit and audit["dispatched"]),
        "reached_reverse_review": reached_reverse,
        "task_success": task_success,
        "final_quality": float(score.get("interaction_quality") or 0),
        "workflow_status": workflow_status,
    }


def reviewer_feedback_metrics(rows):
    """All attempts for handling; source-executable attempts for reach/success."""
    executable = [row for row in rows if row["source_executable"]]
    values = (
        (rows, "appropriate_handling"),
        (executable, "reached_reverse_review"),
        (executable, "task_success"),
    )
    result = {}
    for name, (cohort, field) in zip(REVIEWER_METRICS, values, strict=True):
        count = sum(bool(row[field]) for row in cohort)
        result[name] = {"value": count / len(cohort) if cohort else None,
                        "numerator": count, "denominator": len(cohort),
                        "status": "measured" if cohort else "no_eligible_attempts"}
    return result
