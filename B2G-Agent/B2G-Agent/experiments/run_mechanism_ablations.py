"""Mechanism ablations: ep-001..ep-200, three live repeats each.

Prepare by default. --execute is required to acquire the user-authorized batch.
Acquisition writes one raw suite directory per method. Final indicators are
calculated offline after acquisition and integrity review.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from b2g_agent.case_io import CaseBundle
from b2g_agent.evaluation.protocols import load_formal_protocol
from b2g_agent.evaluation.paper_metrics import (
    MECHANISM_ROLES, MECHANISM_VERSIONS, ACTIVE_MECHANISMS, AGENT_METRICS_PROTOCOL,
)
from b2g_agent.observability.journal import durable_write
from run_proposed_suite import freeze_sources, physics_cache_root, protocol_environment, write_json
from b2g_agent.artifact_paths import METHOD_DIRECTORIES


def selected_episodes(case):
    ids = sorted(ep.episode_id for ep in case.iter_episodes())
    if ids != [f"ep-{i:03d}" for i in range(1, 201)]:
        raise ValueError("Expected frozen 200 episode IDs")
    return ids


def worker_command(method, selected, root, index, protocol):
    command = [sys.executable, "-u", "experiments/run_contract_evaluation.py",
        "--mode", "live", "--method", method, "--split", "all", "--repeats", str(protocol["repeats"]),
        "--execute-physics", "--score", "--continue-on-error", "--output-root", str(root / f"w{index}"),
        "--physics-cache-root", str(physics_cache_root(root))]
    for episode in selected:
        command += ["--episode-id", episode]
    return command


def suite_directory(method, stamp):
    return REPO_ROOT / "outputs/benchmark" / METHOD_DIRECTORIES[method]


def check_record_paths(root, workers):
    """Exercise the actual Windows atomic-write depth before any API calls."""
    for index in range(workers):
        probe = root / "lr" / str(index) / ("0" * 64) / "i" / "000000-preflight" / "manifest.json"
        if os.name == "nt" and len(str(probe)) + 14 > 259:
            raise ValueError("LLM record paths exceed the Windows atomic-write limit")
        durable_write(probe, "{}\n")
        if probe.read_text(encoding="utf-8") != "{}\n":
            raise RuntimeError("LLM record storage preflight failed")
        # Delete only the exact disposable probe we just created, never a run tree.
        probe.unlink()
        for directory in (probe.parent, probe.parent.parent, probe.parent.parent.parent):
            directory.rmdir()


def run_method(method, campaign, case, workers):
    protocol = load_formal_protocol(case, REPO_ROOT, method)
    ids = selected_episodes(case)
    root = suite_directory(method, campaign.name.rsplit("-", 1)[-1])
    root.mkdir(parents=True, exist_ok=False)
    check_record_paths(root, workers)
    manifest = dict(status="running", started_at=datetime.now(timezone.utc).isoformat(),
        method=method, split="all", repeats=protocol["repeats"], protocol=protocol,
        episode_ids=ids, expected_attempts=600, workers=workers,
        agent_metrics_protocol_id=AGENT_METRICS_PROTOCOL,
        evaluation_scope=f"mechanism_ablation_v{MECHANISM_VERSIONS[method]}_full200",
        generalization_claim="Fixed 200-task corpus, not an untouched holdout.",
        reviewer_memory_policy="Seed access only when enabled; exclude current source; online writes frozen.",
        source_and_case_bundle_sha256=freeze_sources(root),
        physics_cache_root=str(physics_cache_root(root)),
        physics_cache_policy="Isolated per method; reuse exact deterministic inputs only within this suite.",
        failed_attempt_policy="Retain all method failures without selective retries.", workers_config=[])
    if method == "reviewer_no_feedback":
        manifest["reviewer_ablation_policy"] = {
            "version": 4, "corrective_feedback_enabled": False,
            "review_implementation": "unchanged_full_reviewer",
            "seed_memory_enabled": True, "schema_repairs_unchanged": True,
            "translator_actor_implementations_unchanged": True,
            "normal_initial_execution_and_sender_clarification_enabled": True,
            "diagnostic_routes_are_recommendations": True,
        }
    jobs = []
    for index in range(workers):
        command = worker_command(method, ids[index::workers], root, index, protocol)
        env = protocol_environment(protocol, root, index)
        if env.get("B2G_EMBEDDING_BACKEND") == "hash_test_only":
            raise ValueError("Test embeddings cannot enter acquisition")
        manifest["workers_config"].append(dict(worker=index, episode_ids=ids[index::workers],
            output_root=str(root / f"w{index}"), llm_record_root=str(root / "lr" / str(index)), command=command))
        jobs.append((index, command, env))
    write_json(root / "suite_manifest.json", manifest)
    status_path = campaign / "status.json"
    write_json(status_path, dict(status="running", active_method=method, active_suite=str(root), expected_attempts=600))
    print(f"suite_root={root}", flush=True)

    def worker(job):
        index, command, env = job
        with (root / f"worker-{index}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        return dict(worker=index, exit_code=result.returncode)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        outcomes = list(executor.map(worker, jobs))
    rows = [json.loads(line) for p in root.glob("w*/*/index.jsonl")
            for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    observed = Counter((r["episode_id"], r["repeat"]) for r in rows)
    expected = {(e, r) for e in ids for r in range(1, 4)}
    coverage = set(observed) == expected and all(n == 1 for n in observed.values())
    manifest.update(status="completed" if coverage else "incomplete", completed_at=datetime.now(timezone.utc).isoformat(),
        observed_attempts=len(rows), coverage_verified=coverage, worker_outcomes=outcomes,
        missing_attempts=sorted(expected-set(observed)), duplicate_attempts=[k for k,v in observed.items() if v > 1])
    write_json(root / "suite_manifest.json", manifest)
    if not coverage:
        raise RuntimeError(f"Incomplete acquisition; preserve records and inspect {root}")
    return root


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--method", choices=tuple(MECHANISM_ROLES), action="append")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use 1..4 workers")
    case = CaseBundle.default(REPO_ROOT)
    ids = selected_episodes(case)
    methods = args.method or list(ACTIVE_MECHANISMS)
    if len(methods) != len(set(methods)):
        raise ValueError("Duplicate methods are not allowed")
    protocols = {m:load_formal_protocol(case, REPO_ROOT, m) for m in methods}
    if not args.execute:
        print(json.dumps(dict(status="prepared_only", episode_ids=ids, repeats=3,
            attempts_per_method=600, total_attempts=600*len(methods), protocols=protocols,
            api_calls=0, physics_runs=0, metrics_per_method=12), indent=2))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    campaign = REPO_ROOT / "outputs/campaigns" / ("mechanisms-full200-" + stamp)
    campaign.mkdir(parents=True, exist_ok=False)
    manifest = dict(version="mechanism-ablations-versioned", method_versions={m:MECHANISM_VERSIONS[m] for m in methods},
        started_at=datetime.now(timezone.utc).isoformat(),
        episode_ids=ids, repeats=3, total_attempts=600*len(methods), methods=methods, suites={},
        proposed_reference="outputs/benchmark/proposed", comparison_selection="same 200 IDs x same 3 repeats")
    write_json(campaign / "campaign_manifest.json", manifest)
    print(f"campaign_root={campaign}", flush=True)
    try:
        for method in methods:
            manifest["suites"][method] = str(run_method(method, campaign, case, args.workers))
            write_json(campaign / "campaign_manifest.json", manifest)
        write_json(campaign / "status.json", dict(status="completed", suites=manifest["suites"], total_attempts=600*len(methods)))
    except Exception as exc:
        write_json(campaign / "status.json", dict(status="needs_attention", error=str(exc), suites=manifest["suites"]))
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
