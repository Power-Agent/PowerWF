"""Prepare all 200 episodes x 3 repeats; run only with explicit --execute.

Every repetition invokes the live LLM. Only identical deterministic physics
inputs/plans may reuse content-addressed simulator evidence. Failed attempts
are retained, never replaced by a more successful rerun.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.case_io import CaseBundle
from b2g_agent.evaluation import summarize_evaluation_runs, write_evaluation_summary
from b2g_agent.evaluation.protocols import CORE_METHODS as METHODS, load_formal_protocol
from b2g_agent.evaluation.paper_metrics import AGENT_METRICS_PROTOCOL, paper_metric_count
from b2g_agent.artifact_paths import METHOD_DIRECTORIES


def load_protocol(case, method="full_multi_agent"):
    return load_formal_protocol(case, REPO_ROOT, method)


def suite_directory(method, stamp):
    if method not in METHODS:
        raise ValueError(f"Unsupported formal method: {method}")
    return REPO_ROOT / "outputs/benchmark" / METHOD_DIRECTORIES[method]


def physics_cache_root(root: Path) -> Path:
    """Keep suite isolation without exceeding Windows simulator path limits."""
    owner = hashlib.sha256(str(root.resolve()).encode("utf-8")).hexdigest()[:16]
    return root.parents[1] / "physics" / "cache" / owner


def protocol_environment(protocol, root, worker):
    """Fresh formal outputs; no personal settings or caches can enter this run."""
    env = os.environ.copy()
    env.update(PYTHONDONTWRITEBYTECODE="1", B2G_LLM_MODE="live", B2G_LLM_ENABLED="true",
        B2G_MODEL=protocol["model"], B2G_REASONING_EFFORT=protocol["reasoning_effort"],
        B2G_MAX_OUTPUT_TOKENS=str(protocol["max_output_tokens_per_call"]), B2G_MAX_TOTAL_TOKENS_PER_TASK="",
        B2G_MAX_AGENT_REVISIONS=str(protocol["maximum_agent_revisions"]),
        B2G_MAX_SCHEMA_REPAIRS=str(protocol["maximum_schema_repairs"]), B2G_ONLINE_MEMORY_ENABLED="false",
        B2G_PROTOCOL_ID=protocol["protocol_id"], B2G_LLM_RECORD_ROOT=str(root / "lr" / str(worker)),
        B2G_JOURNAL_ROOT=str(root / f"w{worker}" / "interactions"),
        B2G_SIMULATION_OUTPUT_ROOT=str(physics_cache_root(root)))
    for key in ("B2G_PERSONAL_ROOT", "B2G_EVENT_JOURNAL", "B2G_TRACE_ID"):
        env.pop(key, None)
    return env


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def freeze_sources(root: Path) -> str:
    paths = [REPO_ROOT / name for name in ("pyproject.toml", "AGENTS.md")]
    for directory in ("src", "experiments", "tests", "cases", "knowledge"):
        paths.extend(path for path in (REPO_ROOT / directory).rglob("*")
                     if path.is_file() and path.suffix in {".py", ".json", ".jsonl", ".csv", ".md", ".yaml", ".yml"})
    hashes = {}
    with zipfile.ZipFile(root / "frozen_source_and_case.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            relative = path.relative_to(REPO_ROOT).as_posix()
            hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            archive.write(path, relative)
    write_json(root / "frozen_file_hashes.json", hashes)
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--method", choices=METHODS, default="full_multi_agent")
    parser.add_argument("--split", choices=["all", "test", "development", "memory_seed"], default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize this invocation to run the live suite.")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use between one and four workers")
    case = CaseBundle.default(REPO_ROOT)
    protocol = load_protocol(case, args.method)
    if args.split != protocol["split"] or args.repeats != protocol["repeats"]:
        raise ValueError("This formal protocol requires all 200 episodes x 3; use the private UI for other trials")
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    episode_ids = [ep.episode_id for ep in case.iter_episodes() if args.split == "all" or ep.split.value == args.split]
    expected_episodes = {"all": 200, "test": 150, "development": 30, "memory_seed": 20}[args.split]
    if len(episode_ids) != expected_episodes or len(set(episode_ids)) != expected_episodes:
        raise ValueError(f"Expected exactly {expected_episodes} distinct frozen episodes")
    if not args.execute:
        print(json.dumps({"status": "prepared_only", "api_calls": 0, "physics_runs": 0,
            "split": args.split, "episodes": len(episode_ids), "repeats": args.repeats,
            "expected_attempts": len(episode_ids) * args.repeats,
            "method": args.method, "paper_metric_count": paper_metric_count(args.method),
            "planned_suite_directory": str(suite_directory(args.method, "<UTC-timestamp>")),
            "protocol": protocol, "output_root": str(REPO_ROOT / "outputs/benchmark"),
            "scope": "fixed_corpus_descriptive_benchmark", "execution_requires": "--execute"}, indent=2))
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = suite_directory(args.method, stamp)
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "method": args.method, "split": args.split, "repeats": args.repeats,
        "protocol": protocol,
        "agent_metrics_protocol_id": AGENT_METRICS_PROTOCOL,
        "episode_ids": episode_ids, "expected_attempts": len(episode_ids) * args.repeats, "workers": args.workers,
        "evaluation_scope": "fixed_corpus_descriptive_benchmark",
        "generalization_claim": "None: includes development and memory-source tasks; not an untouched holdout.",
        "reviewer_memory_policy": "Exclude current source episode; freeze all online updates throughout the suite.",
        "source_and_case_bundle_sha256": freeze_sources(root),
        "physics_cache_policy": "Only exact task/plan/code/input hashes; all LLM repetitions are live.",
        "physics_cache_root": str(physics_cache_root(root)),
        "failed_attempt_policy": "Retain failures; no selective reruns; include every scheduled attempt in Y denominator.",
        "workers_config": [],
    }
    jobs = []
    for index in range(args.workers):
        selected = episode_ids[index::args.workers]
        output_root = root / f"w{index}"
        record_root = root / "lr" / str(index)
        command = [sys.executable, "-u", "experiments/run_contract_evaluation.py",
                   "--mode", "live", "--method", args.method, "--split", args.split,
                   "--repeats", str(args.repeats), "--execute-physics", "--score", "--continue-on-error",
                   "--output-root", str(output_root), "--physics-cache-root", str(physics_cache_root(root))]
        for episode_id in selected:
            command.extend(["--episode-id", episode_id])
        env = protocol_environment(protocol, root, index)
        # Do not inherit test-only embedding configuration into the live suite.
        if env.get("B2G_EMBEDDING_BACKEND") == "hash_test_only":
            raise ValueError("Test-only embeddings cannot be used for the live suite")
        jobs.append((index, command, env))
        manifest["workers_config"].append({"worker": index, "episode_ids": selected,
            "output_root": str(output_root), "llm_record_root": str(record_root), "command": command})
    write_json(root / "suite_manifest.json", manifest)
    print(f"suite_root={root}", flush=True)

    def run_worker(job):
        index, command, env = job
        with (root / f"worker-{index}.log").open("w", encoding="utf-8") as handle:
            process = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
        return {"worker": index, "exit_code": process.returncode}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        outcomes = list(executor.map(run_worker, jobs))
    run_roots = sorted(path.parent for path in root.glob("w*/*/index.jsonl"))
    rows = [json.loads(line) for path in run_roots for line in (path / "index.jsonl").read_text(encoding="utf-8").splitlines() if line]
    expected = {(episode_id, repeat) for episode_id in episode_ids for repeat in range(1, args.repeats + 1)}
    counts = Counter((row["episode_id"], row["repeat"]) for row in rows)
    coverage_ok = set(counts) == expected and all(count == 1 for count in counts.values())
    manifest.update(status=("completed" if coverage_ok else "incomplete"),
                    completed_at=datetime.now(timezone.utc).isoformat(), worker_outcomes=outcomes,
                    observed_attempts=len(rows), coverage_verified=coverage_ok,
                    missing_attempts=sorted(expected - set(counts)),
                    duplicate_attempts=[pair for pair, count in counts.items() if count > 1],
                    run_roots=[str(path) for path in run_roots])
    write_json(root / "suite_manifest.json", manifest)
    if run_roots:
        summary = summarize_evaluation_runs(run_roots)
        write_evaluation_summary(summary, root)
        print(json.dumps(summary["methods"][args.method], ensure_ascii=False), flush=True)
    if coverage_ok:
        # Offline postprocessing of this same suite; no extra diagnostic API calls.
        report = subprocess.run([sys.executable, "-B", "experiments/report_proposed_suite.py",
                                 "--suite", str(root)], cwd=REPO_ROOT, check=False)
        manifest["report_exit_code"] = report.returncode
        write_json(root / "suite_manifest.json", manifest)
        if report.returncode:
            return report.returncode
    return 0 if coverage_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
