"""Aggregate measured runtime costs, retaining failures and timing coverage."""
from collections import Counter
from statistics import mean, stdev


def statistics(values):
    values = [float(v) for v in values if v is not None]
    return {"count": len(values), "mean": mean(values) if values else None,
            "std": stdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else None, "max": max(values) if values else None,
            "total": sum(values) if values else None}


def computation_cost(records):
    single = bool(records) and all(r["row"].get("method_id") == "same_llm_single_agent" for r in records)
    roles = ("single_agent",) if single else ("translator", "actor", "reviewer")
    usage, samples = {role: Counter() for role in roles}, {role: [] for role in roles}
    retrieval = {role: [] for role in roles}
    tools = {}
    framework_values = {key: [] for key in ("total_token_consumption", "llm_inference_latency", "retrieval_time")}
    for record in records:
        per_attempt = {role: Counter() for role in roles}
        for name, values in record["row"].get("per_agent_llm_usage", {}).items():
            role = {"translator_feedback": "translator", "free_plan_actor": "actor",
                    "same_llm_single_agent_contract": "single_agent", "same_llm_single_agent_result": "single_agent"}.get(name, name)
            per_attempt.setdefault(role, Counter()).update(values)
        for role in roles:
            usage[role].update(per_attempt[role])
            samples[role].append(per_attempt[role])
        diagnostic = record.get("diagnostics", {})
        framework_values["total_token_consumption"].append(record["row"].get("total_tokens"))
        framework_values["llm_inference_latency"].append(record["row"].get("llm_latency_seconds"))
        times = [diagnostic.get("translator_retrieval_seconds"), diagnostic.get("reviewer_retrieval_seconds")]
        framework_values["retrieval_time"].append(sum(times) if all(v is not None for v in times) else None)
        if single:
            retrieval["single_agent"].append(framework_values["retrieval_time"][-1])
        else:
            retrieval["translator"].append(diagnostic.get("translator_retrieval_seconds"))
            retrieval["reviewer"].append(diagnostic.get("reviewer_retrieval_seconds"))
            retrieval["actor"].append(0.0)  # Actor has no separate knowledge retrieval stage.
        for stage, seconds in diagnostic.get("stage_seconds", {}).items():
            if stage.startswith("tool."):
                tools.setdefault(stage, []).append(seconds)
    return {
        "framework": {name: {**statistics(values),
            "value": (statistics(values)["total" if name == "total_token_consumption" else "mean"]
                      if all(v is not None for v in values) else None),
            "unit": "tokens" if name == "total_token_consumption" else "seconds",
            "aggregation": "sum over all attempts" if name == "total_token_consumption" else "mean per scheduled attempt (all framework calls)",
            "scheduled_attempts": len(records), "missing_attempts": sum(v is None for v in values)}
            for name, values in framework_values.items()},
        "per_agent_usage": {role: dict(values) for role, values in usage.items()},
        "per_agent_per_attempt": {role: {key: statistics(row.get(key, 0) for row in rows)
            for key in ("call_count", "prompt_tokens", "completion_tokens", "total_tokens", "latency_seconds")}
            for role, rows in samples.items()},
        "per_agent_retrieval_seconds": {role: {
            **statistics(values), "scheduled_attempts": len(records),
            "missing_timing_attempts": sum(v is None for v in values),
            "applicable": role != "actor"} for role, values in retrieval.items()},
        "actual_tool_stage_seconds": {name: statistics(values) for name, values in tools.items()},
        "cost_scope": "All recorded attempts, revisions and failed calls; Translator includes reverse translation.",
        "tool_timing_scope": "Recorded tool spans only; a cache hit does not count as a new simulation. Nested spans must not be summed as wall time.",
    }
