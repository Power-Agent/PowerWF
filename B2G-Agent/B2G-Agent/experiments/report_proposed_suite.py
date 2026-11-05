"""Audit a formal method suite and produce local interaction readers without API calls.

The frozen runner's scores are retained. Post-processing measures first-draft
Translator/Reviewer outputs and terminal closed-loop Actor outputs from journals.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from b2g_agent.case_io import CaseBundle
from b2g_agent.contracts import PipelineResult
from b2g_agent.evaluation import InteractionScorer
from b2g_agent.evaluation.agent_metrics import evaluate_agents, NINE_METRICS, AGENT_SAMPLING_SCOPE
from b2g_agent.evaluation.computation_cost import computation_cost
from b2g_agent.evaluation.paper_metrics import (
    QUALITY, AGENTS, MECHANISM_ROLES, AGENT_METRICS_PROTOCOL, agent_metric_keys,
    paper_metrics, write_paper_metrics,
)
from b2g_agent.evaluation.protocols import load_formal_protocol
from b2g_agent.observability import request_hash


METRICS = ("semantic_fidelity", "actionability", "physical_feasibility",
           "goal_achievement", "interaction_quality")
LABELS = dict(zip(METRICS, ("F", "A", "V", "G", "Q")))
MESSAGE_NAMES = {"sender_sent": "m1 — Sender sent",
                 "receiver_received": "m2 — Receiver received",
                 "receiver_sent": "m3 — Receiver sent",
                 "sender_received": "m4 — Sender received"}


from b2g_agent.artifact_paths import resolve_recorded_path

def load(path):
    return json.loads(resolve_recorded_path(path).read_text(encoding="utf-8"))


def digest(path):
    h = hashlib.sha256()
    with resolve_recorded_path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode("utf-8")).hexdigest()


def artifact_references(value):
    """Return explicitly hashed file references in an evidence package."""
    if isinstance(value, dict):
        if isinstance(value.get("path"), str) and isinstance(value.get("sha256"), str):
            yield value["path"], value["sha256"]
        for nested in value.values():
            yield from artifact_references(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from artifact_references(nested)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relative(path, start):
    import os
    return Path(os.path.relpath(path, start)).as_posix()


def stats(values):
    values = list(values)
    return {"count": len(values), "mean": statistics.mean(values) if values else None,
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values) if values else None, "max": max(values) if values else None}


def cohort(records):
    return {"attempts": len(records),
            "completed": sum(r["row"]["status"] == "completed" for r in records),
            "completed_scope": "Processing ended normally; includes controlled Reviewer blocks, not only released exchanges.",
            "workflow_outcomes": dict(Counter(
                r["row"].get("workflow_status") or (r.get("result") or {}).get("workflow_status") or "unavailable"
                for r in records)),
            "runtime_failures": sum(r["row"]["status"] == "failed" for r in records),
            "scored": sum(r["score"] is not None for r in records),
            "successes": sum(bool((r["score"] or {}).get("task_success")) for r in records),
            "Y": (sum(bool((r["score"] or {}).get("task_success")) for r in records)
                    / len(records)) if records else None,
            "all_attempt_metrics_zero_penalty": {
                key: stats(float((r["score"] or {}).get(key, 0.0)) for r in records)
                for key in METRICS},
            "scored_only_metrics": {
                key: stats(float(r["score"][key]) for r in records if r["score"] is not None)
                for key in METRICS}}


def failure_category(error, journal=None):
    if not error:
        return None
    if "PermissionError" in error or "OSError" in error:
        return "local_record_storage"
    if any(s in error for s in ("APIConnectionError", "APITimeoutError", "RateLimitError", "AuthenticationError")):
        return "api_transport_or_service"
    if "ResponseValidationError" in error:
        if (journal or {}).get("llm_finish_reasons", {}).get("length"):
            return "model_output_truncated"
        translator_failed = "translator.translate" in (journal or {}).get("failed_stages", {})
        return "translator_output_schema" if "TaskSpecification" in error or translator_failed else "agent_output_schema"
    if "Reviewer response remained invalid" in error:
        return "reviewer_output_schema_after_format_retries"
    if any(s in error for s in ("clean Reviewer actor route", "freshly validated Actor plan", "did not reach an externally releasable", "Reviewer blocked the completed exchange")):
        return "bounded_review_or_plan_guard_block"
    if "Closed-loop physics failed" in error:
        return "physical_execution_error"
    return "other_runtime_error"


def inspect_journal(path, reader, run_id):
    """Stream the original journal; never reconstruct an absent model response."""
    first_spec = first_plan = first_physics = None
    first_execution_spec = None
    single_agent = False
    first_physics_seen = False
    actor_returned_before_first_physics = False
    latest_exchange = None
    counts, stage_counts, return_routes, issue_codes = Counter(), Counter(), Counter(), Counter()
    finish_reasons, failed_stages = Counter(), Counter()
    events, retrieval_seconds = 0, 0.0
    last_terminal = None
    response_records = {}
    physics_files = {}
    lines = [f"# {run_id}", "",
             "All messages below come from recorded events. Missing messages are not reconstructed.", "",
             "Intermediate states are drafts until Reviewer release. A blocked candidate was not delivered externally.", "",
             f"[Full event journal](<{relative(path, reader.parent)}>).", ""]
    lines += ["[LLM requests, raw replies and parsed artifacts](calls.md).", ""]
    for raw in path.open(encoding="utf-8"):
        if not raw.strip():
            continue
        event = json.loads(raw)
        events += 1
        kind, payload, stamp = event["event"], event["payload"], event["timestamp"]
        counts[kind] += 1
        if kind == "single_agent.candidate":
            single_agent = True
            if first_spec is None:
                first_spec = payload["specification"]
            if first_plan is None:
                first_plan = payload.get("plan")
        if kind == "retrieval.completed" and payload.get("agent") == "single_agent":
            retrieval_seconds += float(payload["elapsed_seconds"])
        if kind == "llm.response":
            finish_reasons.update(choice.get("finish_reason") or "unknown"
                                  for choice in payload["response"].get("choices", []))
            response_records[payload["invocation_id"]] = {
                "canonical_sha256": canonical_digest(payload["response"]),
                "usage": payload["response"].get("usage") or {},
                "latency_seconds": payload.get("latency_seconds"),
                "timestamp": stamp}
        if kind == "stage.failed":
            failed_stages[payload["stage"]] += 1
        if kind == "attempt.started":
            lines += ["## Original sender input", "", payload["episode"]["sender_message"], ""]
        if kind == "stage.completed":
            stage, output = payload["stage"], payload["output"]
            stage_counts[stage] += 1
            if stage == "translator.translate":
                if first_spec is None:
                    first_spec = output
                retrieval_seconds += float((output.get("retrieval_trace") or {}).get("elapsed_seconds") or 0)
            if stage == "translator.translate_feedback":
                retrieval_seconds += float((output.get("retrieval_trace") or {}).get("elapsed_seconds") or 0)
            if stage in {"actor.plan", "actor.free_plan"} and first_plan is None and output:
                first_plan = output
            if stage in {"exchange.start", "exchange.feedback"}:
                latest_exchange = output
                lines += [f"## {stamp} — {stage}", ""]
                for key, title in MESSAGE_NAMES.items():
                    if output.get(key):
                        lines += [f"### {title}", "", output[key], ""]
            if stage == "reviewer.review":
                label = "Deterministic validation (no LLM Reviewer)" if single_agent else "Reviewer"
                lines += [f"## {stamp} — {label}", "",
                          f"Route: {output['routing_decision']}; verdict: {output['verdict']}.", ""]
                for issue in output.get("issues", []):
                    issue_codes[issue["code"]] += 1
                    lines += [f"- {issue['code']}: {issue['message']}"]
                lines += [""]
        if kind in {"reviewer.return", "single_agent.revision"}:
            # The shared physics adapter emits a guard finding before the
            # baseline emits its own revision event. Count that revision once.
            if single_agent and kind == "reviewer.return":
                continue
            if first_plan is not None and not first_physics_seen:
                actor_returned_before_first_physics = True
            route = "single_agent" if kind == "single_agent.revision" else payload.get("route", "")
            return_routes[route] += 1
            lines += [f"## {stamp} — Return for revision", "",
                      f"Direction: {payload.get('direction')}; route: {route}; revision: {payload.get('revision')}.", ""]
            for issue in payload.get("decision", {}).get("issues", []):
                lines += [f"- {issue['code']}: {issue['message']}",
                          f"  Required correction: {issue.get('correction_required', '')}"]
            lines += [""]
        if kind in {"physics.completed", "physics.failed"} and not first_physics_seen:
            first_physics_seen = True
            first_physics = payload.get("evidence")
        if kind in {"physics.started", "physics.completed", "physics.failed"}:
            if kind == "physics.started" and first_execution_spec is None:
                first_execution_spec = payload.get("specification")
            attempt = payload.get("attempt", {})
            lines += [f"## {stamp} — {kind}", "",
                      f"Cache: {attempt.get('cache_status')}; key: {attempt.get('cache_key')}.", ""]
            if payload.get("evidence"):
                ev = payload["evidence"]
                physics_files[ev["evidence_path"]] = ev["evidence_sha256"]
                lines += [f"Converged: {ev.get('converged')}; hard constraints passed: {ev.get('hard_constraints_passed')}.",
                          f"Evidence: {ev.get('evidence_path')}", ""]
            if payload.get("feedback"):
                lines += ["### Receiver feedback produced by the physical workflow (pending review)", "",
                          payload["feedback"]["text"], ""]
            if attempt.get("error"):
                lines += [str(attempt["error"]), ""]
        if kind == "exchange.clarification":
            latest_exchange = payload["exchange"]
            lines += [f"## {stamp} — Clarification", ""]
            for key, title in MESSAGE_NAMES.items():
                if latest_exchange.get(key):
                    lines += [f"### {title}", "", latest_exchange[key], ""]
        if kind in {"attempt.completed", "attempt.failed"}:
            last_terminal = kind
            result = payload.get("result") or payload.get("candidate")
            if result:
                latest_exchange = result.get("information_exchange", latest_exchange)
            lines += [f"## {stamp} — {kind}", ""]
            if payload.get("error"):
                lines += [str(payload["error"]), ""]
    if latest_exchange:
        lines += ["## Final recorded message states", ""]
        for key, title in MESSAGE_NAMES.items():
            lines += [f"### {title}", "", latest_exchange.get(key) or "(Not generated.)", ""]
    reader.parent.mkdir(parents=True, exist_ok=True)
    reader.write_text("\n".join(lines), encoding="utf-8")
    return {"events": events, "event_counts": dict(counts), "completed_stage_counts": dict(stage_counts),
            "return_routes": dict(return_routes), "issue_codes": dict(issue_codes),
            "first_specification": first_spec, "first_plan": first_plan,
            "first_physics_evidence": first_physics, "first_physics_seen": first_physics_seen,
            "first_execution_specification": first_execution_spec,
            "first_physics_after_review_return": actor_returned_before_first_physics,
            "retrieval_seconds_all_revisions": retrieval_seconds, "terminal": last_terminal,
            "llm_responses": response_records,
            "llm_finish_reasons": dict(finish_reasons), "failed_stages": dict(failed_stages),
            "physics_evidence_files": physics_files,
            "four_messages_present": bool(latest_exchange and all(latest_exchange.get(k) for k in MESSAGE_NAMES))}


def verify_invocations(manifest, record_root, errors, warnings, response_records):
    records = []
    hashes, ids = manifest["llm_request_hashes"], manifest["llm_invocation_ids"]
    if len(hashes) != len(ids):
        errors.append(f"{manifest['run_id']}: invocation/hash length mismatch")
        return records
    for sha, invocation in zip(hashes, ids):
        directory = record_root / sha / "i" / invocation
        request_path = record_root / sha / "request.json"
        try:
            meta = load(directory / "manifest.json")
            if meta["request_sha256"] != sha or request_hash(load(request_path)) != sha:
                raise ValueError("request hash mismatch")
            for key, filename in [("response_sha256", "response.json"),
                                  ("parsed_sha256", meta.get("parsed_artifact", "parsed.json"))]:
                if key in meta and digest(directory / filename) != meta[key]:
                    raise ValueError(f"{filename} hash mismatch")
            usage = meta.get("usage") or {}
            latency = float(meta.get("latency_seconds") or 0)
            original_status = meta["status"]
            status = original_status
            if (directory / "response.json").is_file():
                raw_response = load(directory / "response.json")
                recorded = response_records.get(invocation)
                if not recorded or canonical_digest(raw_response) != recorded["canonical_sha256"]:
                    raise ValueError("raw response differs from the durable journal")
                usage = raw_response.get("usage") or usage
            elif invocation in response_records and manifest["status"] == "failed":
                usage = response_records[invocation]["usage"] or usage
                warnings.append(f"{manifest['run_id']} / {invocation}: response file unavailable; raw reply and usage remain in the hash-verified durable journal.")
            if original_status not in {"completed", "failed"}:
                if manifest["status"] != "failed" or invocation not in response_records:
                    raise ValueError("invocation is not terminal and its response cannot be verified")
                status = "recording_interrupted_after_response"
                warnings.append(f"{manifest['run_id']} / {invocation}: original invocation manifest is {original_status}; response verified against journal; original run remains failed.")
                measured = response_records[invocation].get("latency_seconds")
                latency = measured if measured is not None else max(0.0,
                    datetime.fromisoformat(response_records[invocation]["timestamp"]).timestamp()
                    - float(meta["started_epoch_seconds"]))
            records.append({"request_sha256": sha, "invocation_id": invocation,
                            "request_path": str(request_path), "directory": str(directory),
                            "parsed_artifact": meta.get("parsed_artifact", "parsed.json"),
                            "status": status, "original_manifest_status": original_status,
                            "agent": meta.get("agent"), "usage": usage, "latency_seconds": latency,
                            "api_request_attempted": meta.get("api_request_attempted", True)})
        except Exception as exc:
            errors.append(f"{manifest['run_id']} / {invocation}: {exc}")
    return records


def aggregate(records, case, suite):
    scoring = load(case.case_dir / "evaluation/scoring.json")
    overall = cohort(records)
    by_family, by_route, by_episode = defaultdict(list), defaultdict(list), defaultdict(list)
    route_counts, issues, stage_counts, failure_errors = Counter(), Counter(), Counter(), Counter()
    event_counts = Counter()
    returned = []
    outcome_counts, failed_conditions, semantic_field_errors = Counter(), Counter(), Counter()
    flagged_releases = 0
    oracle_rejections_blocked = []
    for r in records:
        episode = case.get_episode(r["row"]["episode_id"])
        truth = case.get_groundtruth(episode.episode_id)
        candidate = r["result"]
        if (r["row"]["status"] == "failed" and candidate
                and candidate["information_exchange"]["decision"] == "rejected"
                and any(i["code"] == "E-HARD-CONSTRAINT-VIOLATION" for i in candidate["reviewer_decision"]["issues"])):
            numeric_path = case.case_dir / "groundtruth/numeric_oracles" / f"{episode.episode_id}.json"
            if numeric_path.is_file() and load(numeric_path)["final_decision"]["receiver_decision"] == "rejected":
                oracle_rejections_blocked.append(r["row"]["run_id"])
        by_family[episode.primary_family.value].append(r)
        by_route[truth.perfect_execution.route].append(r)
        by_episode[episode.episode_id].append(r)
        route_counts.update(r["journal"]["return_routes"])
        issues.update(r["journal"]["issue_codes"])
        stage_counts.update(r["journal"]["completed_stage_counts"])
        event_counts.update(r["journal"]["event_counts"])
        outcome_counts[r["row"].get("decision") or "no_final_decision"] += 1
        score = r["score"]
        if score is None:
            failed_conditions["no_score_after_runtime_failure"] += 1
        else:
            for key, threshold in [("semantic_fidelity", scoring["semantic_success_threshold"]), ("actionability", 1.0),
                                   ("physical_feasibility", 1.0), ("goal_achievement", scoring["goal_success_threshold"])]:
                if float(score[key]) < threshold - 1e-12:
                    failed_conditions[key] += 1
            for transfer in ("first_transfer", "second_transfer"):
                for field, passed in score[transfer].get("field_checks", {}).items():
                    if not passed:
                        semantic_field_errors[f"{transfer}:{field}"] += 1
            flagged_releases += bool(score.get("agent_metrics", {}).get("reviewer_erroneous_release"))
        if r["journal"]["return_routes"]:
            returned.append(r)
        if r["row"]["error"]:
            failure_errors[r["row"]["error"].split("\n")[0]] += 1
    per_episode = []
    for episode_id, group in sorted(by_episode.items()):
        per_episode.append({"episode_id": episode_id, "attempts": len(group),
                            "successes": sum(bool((r["score"] or {}).get("task_success")) for r in group),
                            "Q": stats(float((r["score"] or {}).get("interaction_quality", 0)) for r in group)})
    token_total = sum(r["row"].get("total_tokens", 0) for r in records)
    started = datetime.fromisoformat(suite["started_at"])
    ended = datetime.fromisoformat(suite["completed_at"]) if suite.get("completed_at") else datetime.now(timezone.utc)
    return {"all_attempts": overall, "by_groundtruth_route": {k: cohort(v) for k, v in by_route.items()},
            "by_family": {k: cohort(v) for k, v in by_family.items()},
            "per_episode": per_episode,
            "repeat_success_distribution": dict(Counter(str(x["successes"]) for x in per_episode)),
            "mean_within_episode_Q_std": statistics.mean(x["Q"]["std"] for x in per_episode) if per_episode else None,
            "reviewer_returns": {"events_by_route": dict(route_counts), "attempts_with_returns": len(returned),
                                 "task_success_after_return": sum(bool((r["score"] or {}).get("task_success")) for r in returned),
                                 "completed_after_return": sum(r["row"]["status"] == "completed" for r in returned),
                                 "issue_counts_across_reviews": dict(issues)},
            "terminal_errors": dict(failure_errors), "completed_stage_counts": dict(stage_counts),
            "event_counts": dict(event_counts),
            "failure_categories": dict(Counter(failure_category(r["row"]["error"], r["journal"]) for r in records if r["row"]["error"])),
            "outcome_counts": dict(outcome_counts),
            "failed_success_conditions_nonexclusive": dict(failed_conditions),
            "semantic_field_error_counts": dict(semantic_field_errors),
            "final_release_with_programmatically_flagged_error": flagged_releases,
            "oracle_agreeing_rejection_blocked_by_hard_gate": oracle_rejections_blocked,
            "efficiency": {**computation_cost(records),
                           "total_tokens": token_total,
                           "tokens_per_successful_task": token_total / overall["successes"] if overall["successes"] else None,
                           "latency_all_llm_calls_seconds": sum(r["row"].get("llm_latency_seconds", 0) for r in records),
                           "end_to_end_seconds": stats(r["row"]["elapsed_seconds"] for r in records),
                           "suite_wall_seconds": (ended - started).total_seconds(),
                           "retrieval_seconds_including_all_revisions": sum(
                               (r.get("diagnostics", {}).get("translator_retrieval_seconds") or 0)
                               + (r.get("diagnostics", {}).get("reviewer_retrieval_seconds") or 0) for r in records),
                           "retrieval_timing_scope": "Runtime Translator retrieval events and Reviewer memory spans, including revisions. Missing Reviewer timings are unavailable, not zero. Actor does not run separate retrieval.",
                           "physics_attempt_cache_statuses": dict(Counter(r["row"].get("physics_cache_status") for r in records))}}


def write_report(path, summary):
    rows = paper_metrics(summary)
    scoring = summary.get("scoring_configuration") or load(CaseBundle.default(REPO).case_dir / "evaluation/scoring.json")
    cohort, audit = summary["results"]["all_attempts"], summary["integrity"]
    def num(value):
        return "N/A" if value is None else f"{value:.6g}"
    method = summary.get("method", "full_multi_agent")
    role = MECHANISM_ROLES.get(method)
    lines = [f"# {method}: {len(rows)} paper metrics", "",
        f"Recorded attempts: {cohort['attempts']} / {summary['expected_attempts']}; status: {summary['status']}.",
        "All attempts, including failed ones, remain in this method's benchmark. No pilot, personal session or other method is included.",
        (f"Ablation report: 6 interaction-quality metrics + 3 {role} agent metrics + 3 whole-framework cost metrics. Other role diagnostics are not primary metrics." if role else
         "Baseline reports six quality metrics and three computation-cost metrics; role-specific agent metrics do not apply." if method == "same_llm_single_agent" else "Three role-specific agents are evaluated."),
        f"Workflow outcomes: {json.dumps(cohort.get('workflow_outcomes', {}), sort_keys=True)}; runtime failures: {cohort.get('runtime_failures', 0)}.",
        "Completed processing includes controlled Reviewer blocks. A blocked exchange was not released and is not a successful task.",
        "", "## Interaction quality (6)", "",
        f"Success requires Q^F >= {scoring['semantic_success_threshold']:g}, Q^A = 1, Q^V = 1 and Q^G >= {scoring['goal_success_threshold']:g}. Q^qua = 25(Q^F+Q^A+Q^V+Q^G). Q^tsr is the successful-attempt fraction.",
        "Unscored attempts receive an explicit zero aggregation penalty; absent fields are not claimed to be observed zeros.",
        "", "| Symbol | Value | Standard deviation | Attempts |", "| --- | ---: | ---: | ---: |"]
    for row in rows:
        if row["layer"] == "interaction_quality":
            lines.append(f"| ${row['symbol']}$ | {num(row['value'])} | {num(row.get('std'))} | {row['denominator']} |")
    lines += ["", f"## Agent performance ({3 if role else 0 if method == 'same_llm_single_agent' else 9})", "",
        AGENT_SAMPLING_SCOPE,
        "Applicable units are tasks, fields or reviewed drafts, not always 600. Actor uses the terminal result/candidate, including blocked candidates; its physical success does not imply external release or task success.",
        "", "| Symbol | Definition | Value | Denominator |", "| --- | --- | ---: | ---: |"]
    for row in rows:
        if row["layer"] == "agent_performance":
            lines.append(f"| ${row['symbol']}$ | {row['aggregation']} | {num(row['value'])} | {row['denominator']} |")
    modules = summary["agent_level"]
    if method == "reviewer_no_feedback":
        lines += ["", "Reviewer v4 retains full diagnosis and its routing recommendations, but does not execute corrective feedback. The three Reviewer metrics measure appropriate first handling, reverse review reach and final success on the prescribed cohorts. Blocked outcomes and unchanged primary quality/cost metrics are retained."]
    if method != "same_llm_single_agent" and (not role or role == "rew"):
        lines += ["", f"Reviewer draft coverage: {modules['review_samples_observed']}/{modules['review_samples_expected']}; missing: {modules['review_samples_missing']}.",
            modules["label_limit"],
            "Reviewer indicators use appropriate first handling over all attempts and reverse-review reach/final success over source-executable attempts. They describe workflow outcomes; all are higher-is-better. Missing handling/reach/success receives no credit, while an empty eligible cohort is N/A.",
            "[Reviewed drafts and reference labels](reviewer_diagnostic_labels.jsonl); [mapping audit](mapping_field_audit.jsonl)."]
    lines += [
        "", "## Computation cost: whole framework (3)", "",
        "Translator forward/reverse calls, Actor, Reviewer, revisions and failed calls are included together.",
        "", "| Metric | Primary value | Unit | Total | Mean per attempt | Timing/usage coverage |", "| --- | ---: | --- | ---: | ---: | ---: |"]
    for row in rows:
        if row["layer"] == "computation_cost":
            values = row["statistics"]
            lines.append(f"| {row['symbol']} | {num(row['value'])} | {row['unit']} | {num(values['total'])} | {num(values['mean'])} | {values['count']}/{values['scheduled_attempts']} |")
    lines += ["", "Total token consumption is the sum of input and output tokens over all attempts. Latency and retrieval primary values are per-attempt means of all framework calls, with totals also shown.",
        "LLM inference latency is measured at the client request boundary (including network/service time), excluding parsing, local file writes and physical simulation. Concurrent workers' accumulated time is not batch wall time.",
        "", "## Record integrity", "",
        f"Verified journals: {audit['journals_verified']}; complete four-message states: {audit['four_message_records']}; verified LLM invocations: {audit['llm_invocations_verified']}.",
        f"Verified physical packages: {audit['unique_physics_packages_verified']}; referenced artifacts: {audit['physics_referenced_artifacts_verified']}.",
        f"Source hashes unchanged: {audit['frozen_files_unchanged']}; integrity errors: {len(audit['errors'])}; recording warnings: {len(audit['recording_warnings'])}.",
        f"[Interaction index](interaction_records/INDEX.md); [exact {len(rows)}-metric table](metrics_{len(rows)}.csv); [complete machine-readable report](paper_metrics.json).",
        "", "## Interpretation", "",
        f"The {len(summary.get('protocol', {}).get('episode_ids', [])) or summary.get('protocol', {}).get('episodes', 200)} episodes repeated three times are a fixed-corpus descriptive benchmark, not independent repetitions or an untouched holdout. Original split labels are retained. Online memory is frozen and the current source episode is excluded.",
        "No comparisons or causal improvement claims follow without matched baseline/ablation results. Correct Reviewer returns are internal corrections, not automatically terminal task failures.",
        "", "## Terminal failures", ""]
    lines += [f"- {count}: {error}" for error, count in summary["results"]["terminal_errors"].items()] or ["No terminal failures."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--reviewer-labels", type=Path, help="Optional independently annotated labels for the same recorded review drafts")
    args = parser.parse_args()
    root = resolve_recorded_path(args.suite).resolve()
    if (root / "FINAL_METRICS.json").is_file() or (root.parent / "FINAL_METRICS.json").is_file():
        from finalize_selected_reports import main as finalize
        return finalize()
    suite = load(root / "suite_manifest.json")
    if not args.allow_partial and not suite.get("coverage_verified"):
        raise ValueError("Formal report requires verified episode x repetition coverage")
    case = CaseBundle.default(REPO)
    method = suite.get("method", "full_multi_agent")
    current_protocol = load_formal_protocol(case, REPO, method)
    if suite.get("protocol") != current_protocol:
        raise ValueError("Only a suite acquired under the current frozen protocol can enter this report")
    if method in MECHANISM_ROLES and suite.get("agent_metrics_protocol_id") != AGENT_METRICS_PROTOCOL:
        raise ValueError("Ablation suite does not use the current agent metric definitions")
    scorer = InteractionScorer(case.case_dir / "evaluation/scoring.json")
    workers = {str(x["worker"]): resolve_recorded_path(x["llm_record_root"]) for x in suite["workers_config"]}
    errors, warnings, records = [], [], []
    journals_verified = invocations_verified = four_messages = 0
    physics_verified = {}
    physics_artifacts_verified = {}
    output = root / "interaction_records"
    index_lines = ["# Interaction records", "", "Each reader links to the original durable journal and invocation records. Unscored failures show Q=0 under the explicit failure penalty; this is not an observed component score. Blocked drafts were not delivered externally.",
                   "", "| Run | Status | Task success | Q | Reviewer returns |", "| --- | --- | --- | ---: | ---: |"]
    for index in sorted(root.glob("w*/*/index.jsonl")):
        run_root = index.parent
        config = load(run_root / "run_config.json")
        if config.get("llm_mode") != "live" or config.get("method_id") != method:
            errors.append(f"{run_root}: suite method must match the live worker")
        if config.get("protocol_id") != current_protocol["protocol_id"]:
            errors.append(f"{run_root}: worker protocol mismatch")
        for field, expected_setting in (("model", current_protocol["model"]),
                ("max_output_tokens", current_protocol["max_output_tokens_per_call"]),
                ("temperature", current_protocol["temperature"]), ("top_p", current_protocol["top_p"]),
                ("max_total_tokens_per_task", None), ("seed", current_protocol["seed"]),
                ("reasoning_effort", current_protocol["reasoning_effort"]),
                ("maximum_schema_repairs", current_protocol["maximum_schema_repairs"])):
            if config.get("llm_settings", {}).get(field) != expected_setting:
                errors.append(f"{run_root}: worker setting mismatch: {field}")
        if config.get("maximum_agent_revisions") != current_protocol["maximum_agent_revisions"]:
            errors.append(f"{run_root}: worker revision limit mismatch")
        record_root = workers[run_root.parent.name[1:]]
        for raw in index.read_text(encoding="utf-8").splitlines():
            if not raw.strip():
                continue
            row = json.loads(raw)
            run_id = row["run_id"]
            directory = output / run_id
            journal_path = resolve_recorded_path(row["interaction_journal"])
            manifest = load(run_root / "manifests" / f"{run_id}.json")
            if digest(journal_path) != manifest["output_hashes"]["interaction_journal"]:
                errors.append(f"{run_id}: journal hash mismatch")
            else:
                journals_verified += 1
            journal = inspect_journal(journal_path, directory / "transcript.md", run_id)
            for evidence_path, expected_hash in journal["physics_evidence_files"].items():
                if evidence_path not in physics_verified:
                    physics_verified[evidence_path] = digest(resolve_recorded_path(evidence_path))
                    for name, sha in artifact_references(load(evidence_path)):
                        referenced = (resolve_recorded_path(evidence_path).parent / name).resolve()
                        if str(referenced) not in physics_artifacts_verified:
                            physics_artifacts_verified[str(referenced)] = digest(referenced)
                        if physics_artifacts_verified[str(referenced)] != sha:
                            errors.append(f"{run_id}: referenced physical artifact hash mismatch: {referenced}")
                if physics_verified[evidence_path] != expected_hash:
                    errors.append(f"{run_id}: physical evidence package hash mismatch: {evidence_path}")
            if journal["terminal"] != ("attempt.completed" if row["status"] == "completed" else "attempt.failed"):
                errors.append(f"{run_id}: terminal journal event mismatch")
            four_messages += journal["four_messages_present"]
            calls = verify_invocations(manifest, record_root, errors, warnings, journal["llm_responses"])
            if method == "same_llm_single_agent" and any(c["agent"] not in {
                    "same_llm_single_agent_contract", "same_llm_single_agent_result"} for c in calls):
                errors.append(f"{run_id}: baseline invoked a role-specific LLM")
            call_lines = [f"# {run_id} — original LLM records", "",
                          "These links point to the original files; no model reply is regenerated.", "",
                          "| Call | Agent | Status | Request | Raw response | Parsed output | Manifest |",
                          "| ---: | --- | --- | --- | --- | --- | --- |"]
            for number, call in enumerate(calls, 1):
                source = resolve_recorded_path(call["directory"])
                paths = [resolve_recorded_path(call["request_path"]), source / "response.json",
                         source / call["parsed_artifact"], source / "manifest.json"]
                links = [f"[{label}](<{relative(path, directory)}>)" if path.is_file() else "Not produced"
                         for label, path in zip(("request", "response", "parsed", "manifest"), paths)]
                call_lines.append(f"| {number} | {call['agent']} | {call['status']} | " + " | ".join(links) + " |")
            (directory / "calls.md").write_text("\n".join(call_lines) + "\n", encoding="utf-8")
            invocations_verified += len(calls)
            result_path = run_root / "results" / f"{run_id}.json"
            score_path = run_root / "scores" / f"{run_id}.json"
            result = load(result_path) if result_path.is_file() else None
            candidate_path = run_root / "errors" / f"{run_id}.candidate.json"
            if result is None and candidate_path.is_file():
                result = load(candidate_path)
            score = load(score_path) if score_path.is_file() else None
            for key, path in [("result", result_path), ("score", score_path)]:
                if key in manifest["output_hashes"] and digest(path) != manifest["output_hashes"][key]:
                    errors.append(f"{run_id}: {key} hash mismatch")
            # Acquired scores remain immutable. Report the current, explicitly
            # recorded scoring configuration instead of a stale cached verdict.
            if row["status"] == "completed" and result is not None:
                numeric_path = case.case_dir / "groundtruth/numeric_oracles" / f"{row['episode_id']}.json"
                score = scorer.score(episode=case.get_episode(row["episode_id"]),
                    groundtruth=case.get_groundtruth(row["episode_id"]),
                    candidate=PipelineResult.model_validate(result),
                    numeric_oracle=load(numeric_path) if numeric_path.exists() else None).model_dump(mode="json")
            record = {"row": row, "score": score, "journal": journal, "result": result}
            observed_tokens = sum(int(c["usage"].get("total_tokens") or 0) for c in calls)
            if observed_tokens != int(row.get("total_tokens") or 0):
                record["original_index_usage"] = {k: row.get(k) for k in ("total_tokens", "prompt_tokens", "completion_tokens", "llm_latency_seconds", "per_agent_llm_usage")}
                per_agent = {}
                for call in calls:
                    usage = call["usage"]
                    target = per_agent.setdefault(call["agent"], Counter())
                    target.update(call_count=int(call["api_request_attempted"]),
                                  local_request_blocks=int(not call["api_request_attempted"]),
                                  prompt_tokens=int(usage.get("prompt_tokens") or 0),
                                  completion_tokens=int(usage.get("completion_tokens") or 0),
                                  total_tokens=int(usage.get("total_tokens") or 0),
                                  latency_seconds=call["latency_seconds"])
                record["row"] = dict(row, total_tokens=observed_tokens,
                                     prompt_tokens=sum(v["prompt_tokens"] for v in per_agent.values()),
                                     completion_tokens=sum(v["completion_tokens"] for v in per_agent.values()),
                                     llm_latency_seconds=sum(v["latency_seconds"] for v in per_agent.values()),
                                     per_agent_llm_usage={k: dict(v) for k, v in per_agent.items()})
            records.append(record)
            write_json(directory / "record.json", {
                "run_id": run_id, "index_row": row, "score": score,
                "audited_usage_row": record["row"] if "original_index_usage" in record else None,
                "source_journal": str(journal_path), "source_journal_sha256": digest(journal_path),
                "manifest": str(run_root / "manifests" / f"{run_id}.json"),
                "llm_invocations": calls,
                "events": {k: v for k, v in journal.items() if not k.startswith("first_")},
                "information_exchange": result.get("information_exchange") if result else None})
            index_lines.append(f"| [{run_id}]({run_id}/transcript.md) | {row['status']} | {(score or {}).get('task_success', False)} | {(score or {}).get('interaction_quality', 0):.2f} | {sum(journal['return_routes'].values())} |")
    coverage = Counter((r["row"]["episode_id"], r["row"]["repeat"]) for r in records)
    expected = {(ep, repeat) for ep in suite["episode_ids"] for repeat in range(1, suite["repeats"] + 1)}
    if not args.allow_partial and (set(coverage) != expected or any(v != 1 for v in coverage.values())):
        errors.append(f"{len(suite['episode_ids'])} x {suite['repeats']} coverage is incomplete or duplicated")
    frozen = load(root / "frozen_file_hashes.json")
    changed = [name for name, sha in frozen.items() if digest(REPO / name) != sha]
    if changed:
        errors.append(f"Frozen files changed during execution: {changed}")
    overrides = {}
    if args.reviewer_labels:
        for line in args.reviewer_labels.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                if item["sample_id"] in overrides:
                    raise ValueError("Duplicate diagnostic label sample_id")
                overrides[item["sample_id"]] = item
    agent_report = evaluate_agents(records, case, scorer, overrides, method_id=method)
    for filename, key in (("reviewer_diagnostic_labels.jsonl", "reviewer_samples"),
                          ("mapping_field_audit.jsonl", "mapping_audit"),
                          ("computation_cost_records.jsonl", "computation_cost_records")):
        (root / filename).write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in agent_report[key]), encoding="utf-8")
    with (root / "agent_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value", "denominator"])
        writer.writeheader()
        for name in (AGENTS[key][1] for key in agent_metric_keys(method)):
            value = agent_report["metrics"][name]
            writer.writerow({"metric": name, "value": value["value"], "denominator": value["denominator"]})
    summary = {"schema_version": "3.0", "method": method, "generated_at": datetime.now(timezone.utc).isoformat(),
               "status": "invalid" if errors else "partial" if args.allow_partial else "completed",
               "source_suite": str(root), "postprocessor_sha256": digest(Path(__file__)),
               "api_calls_by_postprocessor": 0,
               "pilot_validation": load(root / "pilot_validation.json") if (root / "pilot_validation.json").is_file() else None,
               "locked_replay_validation": load(root / "locked_replay_validation.json") if (root / "locked_replay_validation.json").is_file() else None,
               "results": aggregate(records, case, suite),
               "agent_level": {k: v for k, v in agent_report.items() if k not in {"reviewer_samples", "mapping_audit", "computation_cost_records"}},
               "evaluation_scope": suite.get("evaluation_scope", "historical_frozen_scope"),
               "expected_attempts": len(expected),
               "protocol": suite.get("protocol"),
               "integrity": {"journals_verified": journals_verified,
                             "llm_invocations_verified": invocations_verified,
                             "unique_physics_packages_verified": len(physics_verified),
                             "physics_referenced_artifacts_verified": len(physics_artifacts_verified),
                             "four_message_records": four_messages,
                             "frozen_files_unchanged": not changed, "errors": errors,
                             "recording_warnings": warnings,
                             "usage_adjustments": [{"run_id": r["row"]["run_id"], "original": r["original_index_usage"],
                                                    "audited_total_tokens": r["row"]["total_tokens"]}
                                                   for r in records if "original_index_usage" in r]}}
    output.mkdir(parents=True, exist_ok=True)
    (output / "INDEX.md").write_text("\n".join(index_lines[:6] + sorted(index_lines[6:])) + "\n", encoding="utf-8")
    metric_rows = write_paper_metrics(root, summary)
    summary[f"metrics_{len(metric_rows)}"] = metric_rows
    summary["scoring_configuration"] = scorer.config
    summary["computation_cost"] = summary["results"]["efficiency"]["framework"]
    summary["interaction_quality"] = {row["metric"]: row for row in metric_rows if row["layer"] == "interaction_quality"}
    summary["agent_performance"] = {row["metric"]: row for row in metric_rows if row["layer"] == "agent_performance"}
    write_json(root / "computation_cost.json", summary["computation_cost"])
    write_json(root / "paper_metrics.json", summary)
    write_report(root / "RESULTS.md", summary)
    print(json.dumps({"attempts": len(records), "Y": summary["results"]["all_attempts"]["Y"],
                      "integrity": summary["integrity"]}, ensure_ascii=False))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())

