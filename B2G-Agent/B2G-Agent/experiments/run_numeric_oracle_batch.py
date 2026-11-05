"""Run and optionally freeze deterministic physics oracles in a resumable batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.observability.reproducibility import (  # noqa: E402
    physics_implementation_bundle_sha256,
)


GROUNDTRUTH_DIR = REPO_ROOT / "cases" / "ieee33_10bem_15min" / "groundtruth"
CONTRACT_PATH = GROUNDTRUTH_DIR / "contract_oracles.jsonl"
JOINT_CONFIG_PATH = (
    REPO_ROOT / "cases" / "ieee33_10bem_15min" / "optimization" / "joint_dispatch.json"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contracts() -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in CONTRACT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def workflow_for(contract: dict[str, object]) -> str:
    families = contract["task_specification"]["task_families"]
    return (
        "bidirectional_flexibility_envelope"
        if "baseline_flexibility" in families
        else "closed_loop"
    )


def evidence_fingerprints(evidence_path: Path) -> set[tuple[str, str]]:
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence.get("workflow") == "bidirectional_flexibility_envelope":
        hashes: set[str] = set()
        for record in evidence["directions"].values():
            hashes.update(
                evidence_fingerprints(evidence_path.parent / record["evidence_package"])
            )
        return hashes
    final_iteration = evidence["iterations"][-1]
    validation_path = evidence_path.parent / final_iteration["coupled_validation"]["path"]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    hashes = validation["input_hashes"]
    return {
        (
            str(hashes["joint_dispatch_config"]),
            str(hashes.get("physics_implementation_bundle", "<missing>")),
        )
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / ".local-runs" / "numeric-oracle-batch",
    )
    parser.add_argument(
        "--episodes",
        nargs="*",
        help="Explicit episode IDs. By default, select every pending numeric oracle.",
    )
    parser.add_argument(
        "--all-physics",
        action="store_true",
        help="Select every physics-required episode, including already verified records.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--maximum-replans", type=int, default=1)
    parser.add_argument(
        "--freeze",
        action="store_true",
        help="Commit each validated evidence package into groundtruth/numeric_oracles.",
    )
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Refresh an existing verified numeric oracle from newly generated evidence.",
    )
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=7200,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command[1:3])}\n"
            f"{completed.stderr[-4000:]}"
        )
    return completed


def main() -> None:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env", override=False)
    contracts = load_contracts()
    by_id = {row["episode_id"]: row for row in contracts}
    if args.episodes and args.all_physics:
        raise ValueError("--episodes and --all-physics are mutually exclusive")
    if args.all_physics and args.freeze and not args.replace_existing:
        raise ValueError(
            "--all-physics --freeze requires --replace-existing so stale verified records are refreshed"
        )
    if args.episodes:
        unknown = sorted(set(args.episodes) - set(by_id))
        if unknown:
            raise ValueError(f"Unknown episodes: {unknown}")
        selected = [by_id[episode_id] for episode_id in args.episodes]
    elif args.all_physics:
        selected = [
            row
            for row in contracts
            if row["physics_truth_status"] != "not_required_for_current_turn"
        ]
    else:
        selected = [
            row for row in contracts if row["physics_truth_status"] == "pending"
        ]
    invalid = [
        row["episode_id"]
        for row in selected
        if row["physics_truth_status"] == "not_required_for_current_turn"
    ]
    if invalid:
        raise ValueError(f"Clarification-only episodes cannot run physics: {invalid}")
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        selected = selected[: args.limit]

    run_root = args.run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    batch_started = perf_counter()
    for position, contract in enumerate(selected, start=1):
        episode_started = perf_counter()
        episode_id = str(contract["episode_id"])
        workflow = workflow_for(contract)
        evidence_path = run_root / episode_id / "evidence_package.json"
        record: dict[str, object] = {
            "episode_id": episode_id,
            "workflow": workflow,
            "position": position,
            "total": len(selected),
        }
        print(
            json.dumps(
                {"event": "start", **record}, separators=(",", ":")
            ),
            flush=True,
        )
        try:
            if not evidence_path.is_file():
                script = (
                    "run_flexibility_envelope_episode.py"
                    if workflow == "bidirectional_flexibility_envelope"
                    else "run_closed_loop_episode.py"
                )
                run_checked(
                    [
                        sys.executable,
                        str(REPO_ROOT / "experiments" / script),
                        "--episode",
                        episode_id,
                        "--run-root",
                        str(run_root),
                        "--maximum-replans",
                        str(args.maximum_replans),
                    ]
                )
                record["reused"] = False
            else:
                recorded_fingerprints = evidence_fingerprints(evidence_path)
                current_fingerprint = (
                    sha256_file(JOINT_CONFIG_PATH),
                    physics_implementation_bundle_sha256(REPO_ROOT),
                )
                if recorded_fingerprints != {current_fingerprint}:
                    raise RuntimeError(
                        "Existing evidence uses stale configuration or implementation code; "
                        "choose a new --run-root instead of mixing versions. "
                        f"recorded={sorted(recorded_fingerprints)}, "
                        f"current={current_fingerprint}"
                    )
                record["reused"] = True
            evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            record.update(
                {
                    "status": evidence["status"],
                    "evidence_package": str(evidence_path),
                    "evidence_sha256": sha256_file(evidence_path),
                }
            )
            if args.freeze:
                numeric_path = GROUNDTRUTH_DIR / "numeric_oracles" / f"{episode_id}.json"
                freeze_command = [
                    sys.executable,
                    str(REPO_ROOT / "experiments" / "freeze_numeric_oracle.py"),
                    "--evidence-package",
                    str(evidence_path),
                ]
                if numeric_path.is_file() and args.replace_existing:
                    freeze_command.append("--replace")
                if not numeric_path.is_file() or args.replace_existing:
                    run_checked(freeze_command)
                record["numeric_oracle"] = str(numeric_path)
                record["numeric_oracle_sha256"] = sha256_file(numeric_path)
            record["result"] = "pass"
        except Exception as exc:
            record["elapsed_seconds"] = perf_counter() - episode_started
            record["result"] = "failed"
            record["error"] = str(exc)
            records.append(record)
            print(json.dumps({"event": "failed", **record}), flush=True)
            if not args.continue_on_error:
                break
            continue
        record["elapsed_seconds"] = perf_counter() - episode_started
        records.append(record)
        print(json.dumps({"event": "complete", **record}), flush=True)

    summary = {
        "schema_version": "1.0",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "contract_oracles_sha256_at_completion": sha256_file(CONTRACT_PATH),
        "selected_count": len(selected),
        "passed_count": sum(record["result"] == "pass" for record in records),
        "failed_count": sum(record["result"] == "failed" for record in records),
        "elapsed_seconds": perf_counter() - batch_started,
        "freeze": bool(args.freeze),
        "records": records,
    }
    summary_path = run_root / "batch_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "summary_path": str(summary_path)}, indent=2))
    if summary["failed_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
