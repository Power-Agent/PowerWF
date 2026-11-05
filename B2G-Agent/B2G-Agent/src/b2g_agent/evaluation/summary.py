from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from b2g_agent.artifact_paths import resolve_recorded_path
from typing import Any, Iterable


SCORE_METRICS = (
    "semantic_fidelity",
    "actionability",
    "physical_feasibility",
    "goal_achievement",
    "interaction_quality",
)


def summarize_evaluation_runs(run_roots: Iterable[Path]) -> dict[str, Any]:
    roots = [resolve_recorded_path(path).resolve() for path in run_roots]
    grouped: dict[str, list[tuple[Path, dict[str, Any], dict[str, Any] | None]]] = (
        defaultdict(list)
    )
    for root in roots:
        index_path = root / "index.jsonl"
        if not index_path.is_file():
            raise FileNotFoundError(f"Evaluation index is missing: {index_path}")
        for line_number, raw_line in enumerate(
            index_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {index_path}:{line_number}") from exc
            method_id = str(row["method_id"])
            score_path = root / "scores" / f"{row['run_id']}.json"
            score = (
                json.loads(score_path.read_text(encoding="utf-8"))
                if score_path.is_file()
                else None
            )
            grouped[method_id].append((root, row, score))

    methods = {
        method_id: _summarize_method(method_id, records)
        for method_id, records in sorted(grouped.items())
    }
    return {
        "schema_version": "2.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_roots": [str(path) for path in roots],
        "method_count": len(methods),
        "methods": methods,
    }


def write_evaluation_summary(summary: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "evaluation_summary.json"
    csv_path = output_dir / "evaluation_summary.csv"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    rows = [_flatten_method(value) for value in summary["methods"].values()]
    fieldnames = sorted({key for row in rows for key in row}) if rows else ["method_id"]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path


def _summarize_method(
    method_id: str,
    records: list[tuple[Path, dict[str, Any], dict[str, Any] | None]],
) -> dict[str, Any]:
    rows = [row for _, row, _ in records]
    scores = [score for _, _, score in records if score is not None]
    completed = [row for row in rows if row.get("status", "completed") == "completed"]
    failed = [row for row in rows if row.get("status") == "failed"]
    metric_summary = {
        name: _metric_summary(
            [float(score[name]) for score in scores if score.get(name) is not None],
            records,
            name,
        )
        for name in SCORE_METRICS
    }
    agent_metric_names = sorted(
        {
            name
            for score in scores
            for name in score.get("agent_metrics", {})
        }
    )
    agent_metrics = {
        name: _numeric_summary(
            [
                float(score["agent_metrics"][name])
                for score in scores
                if name in score.get("agent_metrics", {})
            ]
        )
        for name in agent_metric_names
    }
    elapsed = [
        float(row["elapsed_seconds"])
        for row in rows
        if row.get("elapsed_seconds") is not None
    ]
    retrieval = [
        float(row["retrieval_seconds"])
        for row in rows
        if row.get("retrieval_seconds") is not None
    ]
    per_agent_usage: dict[str, dict[str, float | int]] = {}
    for row in rows:
        for agent, usage in (row.get("per_agent_llm_usage") or {}).items():
            target = per_agent_usage.setdefault(
                str(agent),
                {
                    "call_count": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "latency_seconds": 0.0,
                },
            )
            for name in ("call_count", "prompt_tokens", "completion_tokens", "total_tokens"):
                target[name] = int(target[name]) + int(usage.get(name) or 0)
            target["latency_seconds"] = float(target["latency_seconds"]) + float(
                usage.get("latency_seconds") or 0.0
            )
    return {
        "method_id": method_id,
        "run_roots": sorted({str(root) for root, _, _ in records}),
        "run_count": len(rows),
        "completed_count": len(completed),
        "completed_scope": "Processing ended normally; includes controlled Reviewer blocks.",
        "workflow_outcomes": dict(Counter(row.get("workflow_status") or "unavailable" for row in rows)),
        "failed_count": len(failed),
        "failure_rate": len(failed) / len(rows) if rows else None,
        "scored_count": len(scores),
        "unique_episode_count": len({row["episode_id"] for row in rows}),
        "Y": (
            sum(bool(score["task_success"]) for score in scores) / len(rows)
            if rows
            else None
        ),
        "Y_among_scored": (
            sum(bool(score["task_success"]) for score in scores) / len(scores)
            if scores
            else None
        ),
        "interaction_quality_all_attempts_mean": (
            sum(float(score["interaction_quality"]) for score in scores) / len(rows)
            if rows else None
        ),
        "unscored_attempt_policy": "Unscored failures count as unsuccessful with Q=0; component metrics summarize scored runs only.",
        "score_metrics": metric_summary,
        "final_state_diagnostics": agent_metrics,
        "agent_metric_scope": "Final corrected state only; selected paper indicators are in FINAL_METRICS.json and the five-method comparison.",
        "elapsed_seconds": {
            **_numeric_summary(elapsed),
            "total": sum(elapsed),
        },
        "retrieval_seconds": {
            **_numeric_summary(retrieval),
            "total": sum(retrieval),
        },
        "llm_usage": {
            "call_count": sum(int(row.get("llm_call_count") or 0) for row in rows),
            "prompt_tokens": sum(int(row.get("prompt_tokens") or 0) for row in rows),
            "completion_tokens": sum(
                int(row.get("completion_tokens") or 0) for row in rows
            ),
            "total_tokens": sum(int(row.get("total_tokens") or 0) for row in rows),
            "latency_seconds": sum(
                float(row.get("llm_latency_seconds") or 0.0) for row in rows
            ),
        },
        "per_agent_llm_usage": dict(sorted(per_agent_usage.items())),
        "physics": {
            "executed_count": sum(bool(row.get("physics_executed")) for row in rows),
            "cache_hit_count": sum(row.get("physics_cache_status") == "hit" for row in rows),
            "cache_miss_count": sum(row.get("physics_cache_status") == "miss" for row in rows),
            "not_required_count": sum(
                row.get("physics_cache_status") == "not_required" for row in rows
            ),
        },
    }


def _metric_summary(
    values: list[float],
    records: list[tuple[Path, dict[str, Any], dict[str, Any] | None]],
    metric: str,
) -> dict[str, float | int | None]:
    by_episode: dict[str, list[float]] = defaultdict(list)
    for _, row, score in records:
        if score is not None and score.get(metric) is not None:
            by_episode[str(row["episode_id"])].append(float(score[metric]))
    within_episode_stds = [
        statistics.stdev(episode_values)
        for episode_values in by_episode.values()
        if len(episode_values) > 1
    ]
    return {
        **_numeric_summary(values),
        "within_episode_sample_std_mean": (
            statistics.fmean(within_episode_stds) if within_episode_stds else None
        ),
        "episodes_with_repeats": len(within_episode_stds),
    }


def _numeric_summary(values: list[float]) -> dict[str, float | int | None]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def _flatten_method(method: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        key: value
        for key, value in method.items()
        if key not in {
            "run_roots", "score_metrics", "final_state_diagnostics", "elapsed_seconds",
            "retrieval_seconds", "llm_usage", "per_agent_llm_usage", "physics",
        }
    }
    for metric, values in method["score_metrics"].items():
        for name, value in values.items():
            row[f"{metric}_{name}"] = value
    for metric, values in method["final_state_diagnostics"].items():
        row[f"final_state_{metric}_mean"] = values["mean"]
        row[f"final_state_{metric}_sample_std"] = values["sample_std"]
    for agent, values in method["per_agent_llm_usage"].items():
        for name, value in values.items():
            row[f"agent_usage_{agent}_{name}"] = value
    for group in ("elapsed_seconds", "retrieval_seconds", "llm_usage", "physics"):
        for name, value in method[group].items():
            row[f"{group}_{name}"] = value
    return row
