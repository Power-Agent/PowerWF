"""The final 18-indicator vocabulary and method-specific applicable metrics."""
from copy import deepcopy
import csv
import json

AGENT_METRICS_PROTOCOL = "selected-18-feedback-impact-20261103"

QUALITY = {
    "Q_qua": (r"Q^{\mathrm{qua}}", "interaction_quality"),
    "Q_tsr": (r"Q^{\mathrm{tsr}}", None),
    **{f"Q_{short}": (rf"Q^{{\mathrm{{{short}}}}}", field) for short, field in (
        ("F", "semantic_fidelity"), ("A", "actionability"),
        ("V", "physical_feasibility"), ("G", "goal_achievement"))},
}
AGENTS = {
    f"F_{role}_{index}": (rf"F^{{\mathrm{{{role}}}}}_{{{index}}}", field)
    for role, fields in (
        ("tra", ("translator_required_field_f1", "translator_mapping_accuracy", "translator_clarification_f1")),
        ("act", ("actor_plan_legality", "actor_tool_chain_success", "actor_evidence_consistency")),
        ("rew", ("reviewer_first_review_appropriate_handling", "reviewer_executable_reverse_review_reach", "reviewer_executable_task_success")),
    ) for index, field in enumerate(fields, 1)
}
COST = ("total_token_consumption", "llm_inference_latency", "retrieval_time")
MECHANISM_ROLES = {"plain_translator": "tra", "unstructured_actor": "act", "reviewer_no_feedback": "rew"}
MECHANISM_VERSIONS = {method: 4 if method == "reviewer_no_feedback" else 2 for method in MECHANISM_ROLES}
ACTIVE_MECHANISMS = ("plain_translator", "unstructured_actor", "reviewer_no_feedback")


def agent_metric_keys(method="full_multi_agent"):
    if method == "same_llm_single_agent":
        return []
    role = MECHANISM_ROLES.get(method)
    return [name for name in AGENTS if role is None or name.startswith(f"F_{role}_")]


def paper_metric_count(method="full_multi_agent"):
    return len(QUALITY) + len(agent_metric_keys(method)) + len(COST)


def paper_metrics(summary):
    cohort = summary["results"]["all_attempts"]
    rows = []
    for name, (symbol, field) in QUALITY.items():
        values = cohort["all_attempt_metrics_zero_penalty"][field] if field else {}
        rows.append(dict(metric=name, symbol=symbol, layer="interaction_quality",
            value=values.get("mean") if field else cohort["Y"],
            std=values.get("std"), denominator=cohort["attempts"],
            unit="points / 100" if name == "Q_qua" else "ratio",
            aggregation="mean over all attempts" if field else "successful / all attempts",
            direction="higher"))
    for name in agent_metric_keys(summary.get("method", "full_multi_agent")):
        symbol, field = AGENTS[name]
        value = summary["agent_level"]["metrics"][field]
        rows.append(dict(metric=name, symbol=symbol, layer="agent_performance",
            value=value["value"], denominator=value["denominator"], unit="ratio",
            aggregation=field, direction="higher",
            status=value.get("status", "measured" if value["value"] is not None else "unavailable")))
    for name in COST:
        data = summary["results"]["efficiency"]["framework"][name]
        rows.append(dict(metric=name, symbol=name.replace("_", " "), layer="computation_cost",
            value=data["value"], denominator=data["scheduled_attempts"], unit=data["unit"],
            aggregation=data["aggregation"], direction="lower", statistics=deepcopy(data)))
    assert len(rows) == len({r["metric"] for r in rows}) == paper_metric_count(summary.get("method", "full_multi_agent"))
    return rows


def write_paper_metrics(directory, summary):
    summary["agent_metrics_protocol_id"] = AGENT_METRICS_PROTOCOL
    rows = paper_metrics(summary)
    count = len(rows)
    (directory / f"metrics_{count}.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    columns = ("metric", "symbol", "layer", "value", "unit", "denominator", "aggregation", "direction")
    with (directory / f"metrics_{count}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return rows
