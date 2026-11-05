"""Validate both sides of one baseline-flexibility envelope offer."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument(
        "--task-specification",
        type=Path,
        default=None,
        help="Optional translated TaskSpecification JSON for live or evaluation execution.",
    )
    parser.add_argument(
        "--execution-plan",
        type=Path,
        default=None,
        help="Reviewed ExecutionPlan JSON bound to both envelope directions.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / ".local-runs" / "flexibility-envelope",
    )
    parser.add_argument("--maximum-replans", type=int, default=1)
    return parser.parse_args()


def run_direction(
    episode: str,
    direction: str,
    output_root: Path,
    maximum_replans: int,
    task_specification: Path | None = None,
    execution_plan: Path | None = None,
) -> tuple[Path, dict[str, object], dict[str, object]]:
    command = [
        sys.executable,
        str(REPO_ROOT / "experiments" / "run_closed_loop_episode.py"),
        "--episode",
        episode,
        "--run-root",
        str(output_root),
        "--maximum-replans",
        str(maximum_replans),
        "--service-direction",
        direction,
    ]
    if task_specification is not None:
        command.extend(["--task-specification", str(task_specification)])
    if execution_plan is not None:
        command.extend(["--execution-plan", str(execution_plan)])
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
            f"{direction} envelope validation failed ({completed.returncode})\n"
            f"{completed.stderr[-3000:]}"
        )
    evidence_path = output_root / episode / "evidence_package.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    final_iteration = evidence["iterations"][-1]
    validation_path = evidence_path.parent / final_iteration["coupled_validation"]["path"]
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    return evidence_path, evidence, validation


def envelope_outcome(
    *,
    reduction_committed_kw: float,
    absorption_committed_kw: float,
    reduction_completion: float,
    absorption_completion: float,
    reduction_release_eligible: bool,
    absorption_release_eligible: bool,
    reduction_full_request_satisfied: bool,
    absorption_full_request_satisfied: bool,
) -> dict[str, object]:
    """Apply product-level hard gates to a bidirectional flexibility offer."""

    envelope_release_eligible = bool(
        reduction_release_eligible and absorption_release_eligible
    )
    if not envelope_release_eligible:
        return {
            "release_eligible": False,
            "receiver_decision": "rejected",
            "completion_percentage": 0.0,
            "validated_downward_flexibility_kw": 0.0,
            "validated_upward_flexibility_kw": 0.0,
            "symmetric_envelope_capacity_kw": 0.0,
            "full_request_satisfied": False,
        }

    downward_kw = max(0.0, float(reduction_committed_kw))
    upward_kw = max(0.0, float(absorption_committed_kw))
    symmetric_kw = min(downward_kw, upward_kw)
    both_full = bool(
        reduction_full_request_satisfied and absorption_full_request_satisfied
    )
    decision = "accepted" if both_full else "modified" if symmetric_kw > 0.0 else "rejected"
    completion = min(float(reduction_completion), float(absorption_completion))
    if decision == "accepted":
        completion = 100.0
    elif decision == "rejected":
        completion = 0.0
    return {
        "release_eligible": True,
        "receiver_decision": decision,
        "completion_percentage": completion,
        "validated_downward_flexibility_kw": downward_kw,
        "validated_upward_flexibility_kw": upward_kw,
        "symmetric_envelope_capacity_kw": symmetric_kw,
        "full_request_satisfied": both_full,
    }


def main() -> None:
    args = parse_args()
    episode_root = (args.run_root / args.episode).resolve()
    if episode_root.exists():
        raise FileExistsError(episode_root)
    episode_root.mkdir(parents=True)
    task_specification_path: Path | None = None
    execution_plan_path: Path | None = None
    execution_plan: dict[str, object] | None = None
    if args.task_specification is not None:
        specification = json.loads(args.task_specification.resolve().read_text(encoding="utf-8"))
        if specification.get("episode_id") != args.episode:
            raise ValueError("Task specification and --episode IDs differ")
        task_specification_path = episode_root / "input_task_specification.json"
        task_specification_path.write_text(
            json.dumps(specification, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.execution_plan is not None:
        execution_plan = json.loads(args.execution_plan.resolve().read_text(encoding="utf-8"))
        if execution_plan.get("episode_id") != args.episode:
            raise ValueError("Execution plan and --episode IDs differ")
        if not execution_plan.get("valid") or execution_plan.get("validation_errors"):
            raise ValueError("Execution plan is not release eligible")
        execution_plan_path = episode_root / "input_execution_plan.json"
        execution_plan_path.write_text(
            json.dumps(execution_plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    direction_records: dict[str, dict[str, object]] = {}
    validations: dict[str, dict[str, object]] = {}
    for direction in ("reduction", "absorption"):
        direction_root = episode_root / "direction_runs" / direction
        evidence_path, evidence, validation = run_direction(
            args.episode,
            direction,
            direction_root,
            args.maximum_replans,
            task_specification_path,
            execution_plan_path,
        )
        direction_records[direction] = {
            "status": evidence["status"],
            "receiver_decision": evidence["final"]["receiver_decision"],
            "candidate_committed_power_kw": evidence["final"].get(
                "candidate_committed_power_kw", evidence["final"]["committed_power_kw"]
            ),
            "committed_power_kw": evidence["final"]["committed_power_kw"],
            "completion_percentage": evidence["final"]["completion_percentage"],
            "release_eligible": bool(validation["release_eligible"]),
            "evidence_package": str(evidence_path.relative_to(episode_root)).replace(
                "\\", "/"
            ),
            "sha256": sha256_file(evidence_path),
        }
        validations[direction] = validation

    reduction = validations["reduction"]
    absorption = validations["absorption"]
    reduction_metrics = reduction["business_metrics"]
    absorption_metrics = absorption["business_metrics"]
    baseline_kw = float(reduction_metrics["counterfactual_baseline_kw"])
    candidate_downward_kw = float(
        reduction.get(
            "candidate_committed_power_kw", reduction["validated_committed_power_kw"]
        )
    )
    candidate_upward_kw = float(
        absorption.get(
            "candidate_committed_power_kw", absorption["validated_committed_power_kw"]
        )
    )
    outcome = envelope_outcome(
        reduction_committed_kw=candidate_downward_kw,
        absorption_committed_kw=candidate_upward_kw,
        reduction_completion=float(reduction["validated_completion_percentage"]),
        absorption_completion=float(absorption["validated_completion_percentage"]),
        reduction_release_eligible=bool(reduction["release_eligible"]),
        absorption_release_eligible=bool(absorption["release_eligible"]),
        reduction_full_request_satisfied=bool(reduction["full_request_satisfied"]),
        absorption_full_request_satisfied=bool(absorption["full_request_satisfied"]),
    )
    downward_kw = float(outcome["validated_downward_flexibility_kw"])
    upward_kw = float(outcome["validated_upward_flexibility_kw"])
    completion = float(outcome["completion_percentage"])
    decision = str(outcome["receiver_decision"])
    both_full = bool(outcome["full_request_satisfied"])
    requested_output_values = dict(reduction_metrics["requested_output_values"])
    requested_output_values.update(
        {
            "counterfactual_baseline_kw": baseline_kw,
            "flexibility_lower_kw": baseline_kw - downward_kw,
            "flexibility_upper_kw": baseline_kw + upward_kw,
        }
    )
    combined_metrics = {
        **reduction_metrics,
        "counterfactual_baseline_kw": baseline_kw,
        "flexibility_lower_kw": baseline_kw - downward_kw,
        "flexibility_upper_kw": baseline_kw + upward_kw,
        "validated_downward_flexibility_kw": downward_kw,
        "validated_upward_flexibility_kw": upward_kw,
        "candidate_downward_flexibility_kw": candidate_downward_kw,
        "candidate_upward_flexibility_kw": candidate_upward_kw,
        "symmetric_envelope_capacity_kw": float(
            outcome["symmetric_envelope_capacity_kw"]
        ),
        "directional_release_eligible": {
            "reduction": bool(reduction["release_eligible"]),
            "absorption": bool(absorption["release_eligible"]),
        },
        "envelope_release_eligible": bool(outcome["release_eligible"]),
        "reduction_rebound_kw": reduction_metrics["rebound_kw"],
        "absorption_rebound_kw": absorption_metrics["rebound_kw"],
        "requested_output_values": requested_output_values,
        "envelope_rule": (
            "Counterfactual target-BEM baseline plus separately validated downward "
            "and upward aggregate service capacities over the frozen task windows."
        ),
    }
    payload = {
        "schema_version": "1.0",
        "workflow": "bidirectional_flexibility_envelope",
        "episode_id": args.episode,
        "status": "validated_full" if both_full else "validated_partial",
        "receiver_decision": decision,
        "completion_percentage": completion,
        "business_metrics": combined_metrics,
        "directions": direction_records,
        "task_specification": (
            {
                "path": "input_task_specification.json",
                "sha256": sha256_file(task_specification_path),
            }
            if task_specification_path is not None
            else None
        ),
        "execution_plan": (
            {
                "plan_id": execution_plan.get("plan_id"),
                "path": "input_execution_plan.json",
                "sha256": sha256_file(execution_plan_path),
            }
            if execution_plan_path is not None and execution_plan is not None
            else None
        ),
    }
    output_path = episode_root / "evidence_package.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "evidence_package": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
