"""Run one numeric episode through optimize, EnergyPlus, pandapower, and one replan."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        help="Reviewed ExecutionPlan JSON bound to this physical run.",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / ".local-runs" / "closed-loop",
    )
    parser.add_argument("--maximum-replans", type=int, default=1)
    parser.add_argument(
        "--service-direction",
        choices=("native", "reduction", "absorption"),
        default="native",
    )
    return parser.parse_args()


def run_command(command: list[str], *, log: list[dict[str, object]]) -> None:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    log.append(
        {
            "command": [Path(command[0]).name, Path(command[1]).name, *command[2:]],
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {Path(command[1]).name}\n"
            f"{completed.stderr[-2000:]}"
        )


def maximum_building_adjustment(optimizer_path: Path) -> float:
    payload = json.loads(optimizer_path.read_text(encoding="utf-8"))
    return max(
        (
            abs(float(value))
            for schedule in payload["schedules"].values()
            for value in schedule["building_reduction_kw"]
        ),
        default=0.0,
    )


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
        source_specification_path = args.task_specification.resolve()
        specification = json.loads(
            source_specification_path.read_text(encoding="utf-8")
        )
        if specification.get("episode_id") != args.episode:
            raise ValueError("Task specification and --episode IDs differ")
        task_specification_path = episode_root / "input_task_specification.json"
        task_specification_path.write_text(
            json.dumps(specification, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    if args.execution_plan is not None:
        source_plan_path = args.execution_plan.resolve()
        execution_plan = json.loads(source_plan_path.read_text(encoding="utf-8"))
        if execution_plan.get("episode_id") != args.episode:
            raise ValueError("Execution plan and --episode IDs differ")
        if not execution_plan.get("valid") or execution_plan.get("validation_errors"):
            raise ValueError("Execution plan is not release eligible")
        execution_plan_path = episode_root / "input_execution_plan.json"
        execution_plan_path.write_text(
            json.dumps(execution_plan, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    command_log: list[dict[str, object]] = []
    iterations: list[dict[str, object]] = []
    multiplier = 1.0
    fixed_response_path: Path | None = None
    final_validation: dict[str, object] | None = None

    for iteration in range(args.maximum_replans + 1):
        iteration_root = episode_root / f"iteration_{iteration:02d}"
        optimizer_root = iteration_root / "optimizer"
        optimizer_command = [
            sys.executable,
            str(REPO_ROOT / "experiments" / "solve_joint_dispatch.py"),
            "--episode",
            args.episode,
            "--building-flexibility-multiplier",
            str(multiplier),
            "--output-root",
            str(optimizer_root),
            "--service-direction",
            args.service_direction,
        ]
        if task_specification_path is not None:
            optimizer_command.extend(
                ["--task-specification", str(task_specification_path)]
            )
        if fixed_response_path is not None:
            optimizer_command.extend(
                ["--fixed-building-response", str(fixed_response_path)]
            )
        run_command(optimizer_command, log=command_log)
        optimizer_path = optimizer_root / args.episode / "optimizer_result.json"
        event_validation_path = fixed_response_path
        reused_energyplus_response = fixed_response_path is not None
        if (
            maximum_building_adjustment(optimizer_path) > 1e-8
            and fixed_response_path is None
        ):
            event_idfs = iteration_root / "event-idfs"
            run_command(
                [
                    sys.executable,
                    str(REPO_ROOT / "experiments" / "prepare_event_idfs.py"),
                    "--optimizer-result",
                    str(optimizer_path),
                    "--output-root",
                    str(event_idfs),
                ],
                log=command_log,
            )
            energyplus_root = iteration_root / "energyplus"
            run_command(
                [
                    sys.executable,
                    str(REPO_ROOT / "experiments" / "run_energyplus_event_validation.py"),
                    "--optimizer-result",
                    str(optimizer_path),
                    "--event-idf-root",
                    str(event_idfs),
                    "--run-root",
                    str(energyplus_root),
                ],
                log=command_log,
            )
            event_validation_path = energyplus_root / "energyplus_event_validation.json"
        validation_path = iteration_root / "coupled_validation.json"
        validation_command = [
            sys.executable,
            str(REPO_ROOT / "experiments" / "validate_joint_dispatch.py"),
            "--optimizer-result",
            str(optimizer_path),
            "--output",
            str(validation_path),
        ]
        if event_validation_path:
            validation_command.extend(
                ["--energyplus-validation", str(event_validation_path)]
            )
        if task_specification_path is not None:
            validation_command.extend(
                ["--task-specification", str(task_specification_path)]
            )
        run_command(validation_command, log=command_log)
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        iteration_record = {
            "iteration": iteration,
            "building_flexibility_multiplier": multiplier,
            "fixed_energyplus_response_repair": reused_energyplus_response,
            "optimizer_result": {
                "path": str(optimizer_path.relative_to(episode_root)).replace("\\", "/"),
                "sha256": sha256_file(optimizer_path),
            },
            "energyplus_validation": (
                {
                    "path": str(event_validation_path.relative_to(episode_root)).replace("\\", "/"),
                    "sha256": sha256_file(event_validation_path),
                }
                if event_validation_path
                else None
            ),
            "coupled_validation": {
                "path": str(validation_path.relative_to(episode_root)).replace("\\", "/"),
                "sha256": sha256_file(validation_path),
            },
            "release_eligible": validation["release_eligible"],
            "receiver_decision": validation["receiver_decision"],
            "completion_percentage": validation["validated_completion_percentage"],
        }
        iterations.append(iteration_record)
        final_validation = validation
        if validation["release_eligible"] and validation["full_request_satisfied"]:
            break
        if iteration >= args.maximum_replans:
            break
        if event_validation_path is not None:
            event_payload = json.loads(event_validation_path.read_text(encoding="utf-8"))
            safe_bems = [
                record["bem_id"]
                for record in event_payload["records"]
                if int(record["comfort_violation_intervals"]) == 0
                and int(record["event_severe_error_count"]) == 0
            ]
            if safe_bems and not reused_energyplus_response:
                # Freeze every safe high-fidelity building response, set the
                # remaining building capacities to zero, and repair only the
                # storage/grid dispatch. One unsafe BEM does not erase the
                # validated flexibility of all other BEMs.
                fixed_response_path = event_validation_path
                multiplier = 0.0
                continue
        # If EnergyPlus comfort validation failed, remove the invalid building
        # capacity and perform the one conservative storage-only repair.
        multiplier = 0.0

    assert final_validation is not None
    log_path = episode_root / "command_log.json"
    log_path.write_text(json.dumps(command_log, indent=2) + "\n", encoding="utf-8")
    package = {
        "schema_version": "1.0",
        "episode_id": args.episode,
        "service_direction": args.service_direction,
        "status": (
            "validated_full"
            if final_validation["release_eligible"]
            and final_validation["full_request_satisfied"]
            else "validated_partial"
        ),
        "replan_count": len(iterations) - 1,
        "iterations": iterations,
        "final": {
            "receiver_decision": final_validation["receiver_decision"],
            "release_eligible": final_validation["release_eligible"],
            "candidate_committed_power_kw": final_validation[
                "candidate_committed_power_kw"
            ],
            "committed_power_kw": final_validation["validated_committed_power_kw"],
            "completion_percentage": final_validation["validated_completion_percentage"],
            "business_metrics": final_validation["business_metrics"],
            "minimum_voltage_pu": final_validation["nonlinear_metrics"]["minimum_voltage_pu"],
            "maximum_line_loading_pct": final_validation["nonlinear_metrics"][
                "maximum_line_loading_pct"
            ],
            "comfort_violation_intervals": final_validation[
                "comfort_violation_intervals"
            ],
        },
        "command_log": {
            "path": "command_log.json",
            "sha256": sha256_file(log_path),
        },
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
    package_path = episode_root / "evidence_package.json"
    package_path.write_text(json.dumps(package, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**package["final"], "status": package["status"], "replan_count": package["replan_count"], "evidence_package": str(package_path)}, indent=2))


if __name__ == "__main__":
    main()
