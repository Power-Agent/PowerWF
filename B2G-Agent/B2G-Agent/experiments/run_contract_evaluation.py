"""Run frozen contract-only or end-to-end benchmark evaluations.

With ``--execute-physics``, each executable m1 request is solved from its
translated TaskSpecification using the local EnergyPlus/pandapower workflow.
Ground truth is loaded only after inference, for optional scoring.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.case_io import CaseBundle  # noqa: E402
from b2g_agent.contracts import EpisodeGroundTruth  # noqa: E402
from b2g_agent.evaluation import (  # noqa: E402
    InteractionScorer,
    PhysicsResultCache,
    task_specification_sha256,
)
from b2g_agent.observability import (  # noqa: E402
    RunManifest,
    physics_implementation_bundle_sha256,
    runtime_tool_versions,
)
from b2g_agent.observability.journal import EventJournal
from b2g_agent.orchestration import B2GPipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["offline", "live", "replay"], default="replay")
    parser.add_argument(
        "--method",
        choices=[
            "full_multi_agent",
            "same_llm_single_agent",
            "plain_translator",
            "unstructured_actor",
            "reviewer_no_feedback",
        ],
        default="full_multi_agent",
    )
    parser.add_argument("--split", choices=["all", "development", "memory_seed", "test"], default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--episode-id", action="append", default=[])
    parser.add_argument("--stop-on-api-service-error", action="store_true",
                        help="Stop this worker on an API outage and retain the failed slot for recovery.")
    parser.add_argument(
        "--execute-physics",
        action="store_true",
        help="Run local EnergyPlus and pandapower for executable tasks.",
    )
    parser.add_argument(
        "--score",
        action="store_true",
        help="Score completed results with the paper-aligned F/A/V/G/Q protocol.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / ".local-runs" / "contract_evaluations",
    )
    parser.add_argument(
        "--physics-cache-root",
        type=Path,
        default=REPO_ROOT / ".local-runs" / "physics_cache",
        help="Content-addressed cache shared across methods and repetitions.",
    )
    parser.add_argument(
        "--no-physics-cache",
        action="store_true",
        help="Force a fresh simulator run for every executable task.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record failed runs and continue with the remaining benchmark.",
    )
    return parser.parse_args()


def git_state() -> tuple[str, bool]:
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout.strip() or "unknown"
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout.strip())
    return commit, dirty


def llm_usage(
    record_root: Path,
    request_hashes: list[str],
    invocation_ids: list[str],
) -> dict[str, object]:
    totals: dict[str, object] = {
        "llm_call_count": 0,
        "local_request_blocks": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_latency_seconds": 0.0,
        "per_agent_llm_usage": {},
    }
    if invocation_ids and len(invocation_ids) != len(request_hashes):
        raise ValueError("LLM request hashes and invocation IDs must be one-to-one")
    for index, digest in enumerate(request_hashes):
        manifest_path: Path | None = None
        if invocation_ids:
            manifest_path = (
                record_root
                / digest
                / "i"
                / invocation_ids[index]
                / "manifest.json"
            )
            if not manifest_path.is_file():
                manifest_path = (
                    record_root
                    / digest
                    / "invocations"
                    / invocation_ids[index]
                    / "manifest.json"
                )
        if manifest_path is not None and manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("request_sha256") != digest:
                raise ValueError(f"LLM invocation/request mismatch: {manifest_path}")
            if manifest.get("api_request_attempted", True) is False:
                totals["local_request_blocks"] = int(totals["local_request_blocks"]) + 1
                continue
            usage = manifest.get("usage") or {}
            agent = str(manifest.get("agent") or "unknown")
            latency = float(manifest.get("latency_seconds") or 0.0)
        else:
            # Read-only compatibility for records written before immutable
            # per-invocation artifacts were introduced.
            path = record_root / digest / "response.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            usage = payload.get("usage") or {}
            agent = "legacy"
            latency = 0.0
        totals["llm_call_count"] = int(totals["llm_call_count"]) + 1
        prompt = int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0)
        completion = int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
        total = int(usage.get("total_tokens", prompt + completion) or 0)
        totals["prompt_tokens"] = int(totals["prompt_tokens"]) + prompt
        totals["completion_tokens"] = int(totals["completion_tokens"]) + completion
        totals["total_tokens"] = int(totals["total_tokens"]) + total
        totals["llm_latency_seconds"] = float(totals["llm_latency_seconds"]) + latency
        by_agent = totals["per_agent_llm_usage"]
        if not isinstance(by_agent, dict):
            continue
        agent_usage = by_agent.setdefault(
            agent,
            {
                "call_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "latency_seconds": 0.0,
            },
        )
        agent_usage["call_count"] += 1
        agent_usage["prompt_tokens"] += prompt
        agent_usage["completion_tokens"] += completion
        agent_usage["total_tokens"] += total
        agent_usage["latency_seconds"] += latency
    return totals


def retrieval_seconds(result: object | None) -> float:
    if result is None:
        return 0.0
    specification = getattr(result, "task_specification", None)
    reverse = getattr(result, "reverse_transfer", None)
    forward_trace = getattr(specification, "retrieval_trace", {}) or {}
    reverse_trace = getattr(reverse, "retrieval_trace", {}) or {}
    return float(forward_trace.get("elapsed_seconds") or 0.0) + float(
        reverse_trace.get("elapsed_seconds") or 0.0
    )


def main() -> int:
    args = parse_args()
    if args.repeats < 1:
        raise ValueError("--repeats must be positive")
    load_dotenv(REPO_ROOT / ".env", override=False)
    os.environ["B2G_LLM_MODE"] = args.mode
    case = CaseBundle.default(REPO_ROOT)
    episodes = case.list_episodes() if args.split == "all" else case.list_episodes(split=args.split)
    if args.episode_id:
        requested = set(args.episode_id)
        episodes = [episode for episode in episodes if episode.episode_id in requested]
        missing = sorted(requested - {episode.episode_id for episode in episodes})
        if missing:
            raise ValueError(
                f"Requested episodes are not in split {args.split}: {', '.join(missing)}"
            )
    if not episodes:
        raise ValueError(f"No episodes selected for split {args.split}")
    pipeline = B2GPipeline.default(
        repo_root=REPO_ROOT,
        method_id=args.method,
        formal_evaluation=True,
    )
    contract_path = case.case_dir / "groundtruth" / "contract_oracles.jsonl"
    truths = {
        truth.episode_id: truth
        for truth in (
            EpisodeGroundTruth.model_validate(json.loads(line))
            for line in contract_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    scoring_path = case.case_dir / "evaluation" / "scoring.json"
    scorer = InteractionScorer(scoring_path) if args.score else None
    commit, dirty = git_state()
    tool_versions = runtime_tool_versions()
    physics_bundle = physics_implementation_bundle_sha256(REPO_ROOT)
    physics_cache = (
        PhysicsResultCache(
            root=args.physics_cache_root,
            repo_root=REPO_ROOT,
            case=case,
        )
        if args.execute_physics and not args.no_physics_cache
        else None
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = args.output_root / f"{stamp}-{args.method}-{args.mode}-{args.split}"
    run_root.mkdir(parents=True, exist_ok=False)
    common_input_hashes = {
        "episodes.jsonl": hashlib.sha256(case.episodes_path.read_bytes()).hexdigest(),
        "contract_oracles.jsonl": hashlib.sha256(contract_path.read_bytes()).hexdigest(),
        "scoring.json": hashlib.sha256(scoring_path.read_bytes()).hexdigest(),
        "methods.json": hashlib.sha256(
            (case.case_dir / "evaluation" / "methods.json").read_bytes()
        ).hexdigest(),
        "physics_implementation_bundle": physics_bundle,
    }
    if physics_cache is not None:
        common_input_hashes["physics_input_bundle"] = physics_cache.input_bundle
    started_at = datetime.now(timezone.utc)
    run_config = {
        "schema_version": "1.0",
        "status": "running",
        "started_at": started_at.isoformat(),
        "method_id": args.method,
        "llm_mode": args.mode,
        "protocol_id": os.getenv("B2G_PROTOCOL_ID"),
        "split": args.split,
        "repeats": args.repeats,
        "episode_ids": [episode.episode_id for episode in episodes],
        "execute_physics": args.execute_physics,
        "score": args.score,
        "continue_on_error": args.continue_on_error,
        "physics_cache_enabled": physics_cache is not None,
        "physics_cache_root": str(args.physics_cache_root.resolve()),
        "input_hashes": common_input_hashes,
        "llm_settings": pipeline.translator.llm.settings.model_dump(mode="json"),
        "maximum_agent_revisions": pipeline.maximum_agent_revisions,
        "evaluation_scope": "fixed_corpus_descriptive_benchmark" if args.split == "all" else "selected_split",
        "reviewer_memory_policy": "frozen_seed_excluding_current_source_episode; no online updates",
        "tool_versions": tool_versions,
    }
    config_path = run_root / "run_config.json"
    config_path.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rows = []
    fatal_error = False
    for episode in episodes:
        for repeat in range(args.repeats):
            run_id = f"{episode.episode_id}-r{repeat + 1:02d}"
            started = perf_counter()
            request_offset = len(pipeline.translator.llm.request_hashes)
            invocation_offset = len(pipeline.translator.llm.invocation_ids)
            output_hashes: dict[str, str] = {}
            result = None
            score_payload = None
            physics_cache_status = "not_required"
            physics_cache_key_value = None
            physics_attempt_log = []
            journal = EventJournal(run_root / "interactions" / f"{run_id}.jsonl", run_id)
            journal_context = journal.activate()
            journal_context.__enter__()
            try:
                journal.emit("attempt.started", run_id=run_id, episode=episode, repeat=repeat + 1)
                source_text = episode.sender_message
                result = pipeline.run_contract_stage(
                    source_text,
                    episode=episode,
                    sender=episode.start_role,
                )
                if (
                    args.execute_physics
                    and result.information_exchange.decision == "pending_physics"
                    and result.workflow_status != "blocked"
                ):
                    result = pipeline.execute_physics(
                        result, run_root=run_root / "physics" / run_id,
                        cache=physics_cache, attempt_log=physics_attempt_log,
                    )
                    if physics_attempt_log:
                        physics_cache_status = physics_attempt_log[-1]["cache_status"]
                        physics_cache_key_value = physics_attempt_log[-1]["cache_key"]
                    if result.physics_evidence is not None:
                        pointer_path = run_root / "physics" / f"{run_id}.json"
                        pointer_path.parent.mkdir(parents=True, exist_ok=True)
                        pointer_path.write_text(json.dumps({
                            "cache_status": physics_cache_status, "cache_key": physics_cache_key_value,
                            "evidence_path": result.physics_evidence.evidence_path,
                            "evidence_sha256": result.physics_evidence.evidence_sha256,
                            "attempts": physics_attempt_log,
                        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                        output_hashes["physics_pointer"] = hashlib.sha256(pointer_path.read_bytes()).hexdigest()
                # Exhausted Reviewer correction is a controlled blocked outcome.
                # Persist and score it as an unsuccessful task; do not dispatch
                # tools or disguise it as a simulator/API exception.
                blocked = result.workflow_status == "blocked"
                if args.execute_physics and result.information_exchange.decision == "pending_physics" and not blocked:
                    raise RuntimeError("Bounded correction workflow did not reach an externally releasable result")
                if result.information_exchange.decision in {
                    "accepted",
                    "modified",
                    "rejected",
                } and result.reviewer_decision.routing_decision != "release" and not blocked:
                    raise RuntimeError(
                        "Reviewer blocked the completed exchange; no external result was released"
                    )
                if (
                    result.information_exchange.decision == "needs_clarification"
                    and result.reviewer_decision.routing_decision != "clarify_sender"
                    and not blocked
                ):
                    raise RuntimeError("Clarification exchange failed the Reviewer route gate")
                result_path = run_root / "results" / f"{run_id}.json"
                result_path.parent.mkdir(parents=True, exist_ok=True)
                result_path.write_text(
                    result.model_dump_json(indent=2) + "\n", encoding="utf-8"
                )
                output_hashes["result"] = hashlib.sha256(
                    result_path.read_bytes()
                ).hexdigest()
                if scorer is not None:
                    numeric_path = (
                        case.case_dir
                        / "groundtruth"
                        / "numeric_oracles"
                        / f"{episode.episode_id}.json"
                    )
                    numeric = (
                        json.loads(numeric_path.read_text(encoding="utf-8"))
                        if numeric_path.is_file()
                        else None
                    )
                    score = scorer.score(
                        episode=episode,
                        groundtruth=truths[episode.episode_id],
                        candidate=result,
                        numeric_oracle=numeric,
                    )
                    score_path = run_root / "scores" / f"{run_id}.json"
                    score_path.parent.mkdir(parents=True, exist_ok=True)
                    score_path.write_text(
                        score.model_dump_json(indent=2) + "\n", encoding="utf-8"
                    )
                    output_hashes["score"] = hashlib.sha256(
                        score_path.read_bytes()
                    ).hexdigest()
                    score_payload = score.model_dump(mode="json")
                request_hashes = result.llm_request_hashes
                elapsed_seconds = perf_counter() - started
                usage = llm_usage(
                    pipeline.translator.llm.record_root,
                    request_hashes,
                    result.llm_invocation_ids,
                )
                journal.emit("attempt.completed", result=result, score=score_payload, physics_attempts=physics_attempt_log)
                output_hashes["interaction_journal"] = hashlib.sha256(journal.path.read_bytes()).hexdigest()
                manifest = RunManifest(
                    run_id=run_id,
                    case_id=case.case_id,
                    episode_id=episode.episode_id,
                    method_id=(
                        f"{args.method}-end-to-end-{args.mode}"
                        if args.execute_physics
                        else f"{args.method}-contract-{args.mode}"
                    ),
                    git_commit=commit,
                    git_dirty=dirty,
                    llm_settings=pipeline.translator.llm.settings,
                    tool_versions=tool_versions,
                    llm_request_hashes=request_hashes,
                    llm_invocation_ids=result.llm_invocation_ids,
                    input_hashes=common_input_hashes,
                    output_hashes=output_hashes,
                    status="completed",
                )
                manifest.write(run_root / "manifests" / f"{run_id}.json")
                rows.append({
                    "run_id": run_id,
                    "interaction_journal": str(journal.path),
                    "physics_attempt_count": len(physics_attempt_log),
                    "status": "completed",
                    "workflow_status": result.workflow_status,
                    "method_id": args.method,
                    "episode_id": episode.episode_id,
                    "repeat": repeat + 1,
                    "task_spec_sha256": task_specification_sha256(
                        result.task_specification
                    ),
                    "plan_sha256": hashlib.sha256(
                        (
                            result.execution_plan.model_dump_json()
                            if result.execution_plan
                            else "null"
                        ).encode()
                    ).hexdigest(),
                    "review_sha256": hashlib.sha256(
                        result.reviewer_decision.model_dump_json().encode()
                    ).hexdigest(),
                    "verdict": result.reviewer_decision.verdict.value,
                    "decision": result.information_exchange.decision,
                    "completion_percentage": result.information_exchange.completion_percentage,
                    "physics_executed": result.physics_evidence is not None,
                    "physics_cache_status": physics_cache_status,
                    "physics_cache_key": physics_cache_key_value,
                    "elapsed_seconds": elapsed_seconds,
                    "retrieval_seconds": retrieval_seconds(result),
                    **usage,
                    "interaction_quality": (
                        score_payload["interaction_quality"] if score_payload else None
                    ),
                    "task_success": (
                        score_payload["task_success"] if score_payload else None
                    ),
                    "error": None,
                })
            except Exception as exc:
                elapsed_seconds = perf_counter() - started
                request_hashes = pipeline.translator.llm.request_hashes[request_offset:]
                invocation_ids = pipeline.translator.llm.invocation_ids[
                    invocation_offset:
                ]
                usage = llm_usage(
                    pipeline.translator.llm.record_root,
                    request_hashes,
                    invocation_ids,
                )
                error_path = run_root / "errors" / f"{run_id}.txt"
                error_path.parent.mkdir(parents=True, exist_ok=True)
                error_path.write_text(traceback.format_exc(), encoding="utf-8")
                output_hashes["error"] = hashlib.sha256(error_path.read_bytes()).hexdigest()
                if result is not None:
                    partial_path = run_root / "errors" / f"{run_id}.candidate.json"
                    partial_path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
                    output_hashes["unreleased_candidate"] = hashlib.sha256(partial_path.read_bytes()).hexdigest()
                if physics_attempt_log:
                    physics_cache_status = physics_attempt_log[-1]["cache_status"]
                    physics_cache_key_value = physics_attempt_log[-1]["cache_key"]
                journal.emit("attempt.failed", error_type=type(exc).__name__, error=str(exc),
                             traceback=traceback.format_exc(), candidate=result, physics_attempts=physics_attempt_log)
                output_hashes["interaction_journal"] = hashlib.sha256(journal.path.read_bytes()).hexdigest()
                manifest = RunManifest(
                    run_id=run_id,
                    case_id=case.case_id,
                    episode_id=episode.episode_id,
                    method_id=(
                        f"{args.method}-end-to-end-{args.mode}"
                        if args.execute_physics
                        else f"{args.method}-contract-{args.mode}"
                    ),
                    git_commit=commit,
                    git_dirty=dirty,
                    llm_settings=pipeline.translator.llm.settings,
                    tool_versions=tool_versions,
                    llm_request_hashes=request_hashes,
                    llm_invocation_ids=invocation_ids,
                    input_hashes=common_input_hashes,
                    output_hashes=output_hashes,
                    status="failed",
                    errors=[f"{type(exc).__name__}: {exc}"],
                )
                manifest.write(run_root / "manifests" / f"{run_id}.json")
                rows.append({
                    "run_id": run_id,
                    "interaction_journal": str(journal.path),
                    "physics_attempt_count": len(physics_attempt_log),
                    "status": "failed",
                    "method_id": args.method,
                    "episode_id": episode.episode_id,
                    "repeat": repeat + 1,
                    "task_spec_sha256": (
                        task_specification_sha256(result.task_specification)
                        if result is not None
                        else None
                    ),
                    "plan_sha256": None,
                    "review_sha256": None,
                    "verdict": None,
                    "decision": None,
                    "completion_percentage": None,
                    "physics_executed": False,
                    "physics_cache_status": physics_cache_status,
                    "physics_cache_key": physics_cache_key_value,
                    "elapsed_seconds": elapsed_seconds,
                    "retrieval_seconds": retrieval_seconds(result),
                    **usage,
                    "interaction_quality": None,
                    "task_success": None,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"failed_run={run_id} error={type(exc).__name__}: {exc}")
                if not args.continue_on_error or (args.stop_on_api_service_error and type(exc).__name__ in {
                    "APIConnectionError","APITimeoutError","RateLimitError","InternalServerError",
                    "AuthenticationError","PermissionDeniedError","APIStatusError","NotFoundError"}):
                    fatal_error = True
            finally:
                journal_context.__exit__(None, None, None)
            # Persist each completed/failed attempt before starting another API call.
            with (run_root / "index.jsonl").open("a", encoding="utf-8") as index_file:
                index_file.write(json.dumps(rows[-1], sort_keys=True) + "\n")
            print(f"progress={len(rows)}/{len(episodes) * args.repeats} run={run_id} status={rows[-1]['status']}", flush=True)
            if fatal_error:
                break
        if fatal_error:
            break
    (run_root / "index.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )
    completed_count = sum(row["status"] == "completed" for row in rows)
    failed_count = sum(row["status"] == "failed" for row in rows)
    blocked_count = sum(row.get("workflow_status") == "blocked" for row in rows)
    run_config.update(
        {
            "status": "failed" if failed_count else "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": (datetime.now(timezone.utc) - started_at).total_seconds(),
            "completed_runs": completed_count,
            "failed_runs": failed_count,
            "blocked_runs": blocked_count,
            "completed_scope": "Processing ended normally; includes controlled Reviewer blocks.",
        }
    )
    config_path.write_text(
        json.dumps(run_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"completed_runs={completed_count} blocked_runs={blocked_count} failed_runs={failed_count} output={run_root}"
    )
    return 1 if fatal_error or failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
