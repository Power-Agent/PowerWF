"""Solve one frozen episode with the deterministic linear RC--OPF candidate scheduler."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.contracts import EpisodeGroundTruth, TaskSpecification  # noqa: E402
from b2g_agent.simulation.joint_rc_opf import (  # noqa: E402
    FrozenJointRCOPF,
    result_payload,
    write_result,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True)
    parser.add_argument(
        "--task-specification",
        type=Path,
        default=None,
        help="Optional translated TaskSpecification JSON. When supplied, do not read benchmark ground truth.",
    )
    parser.add_argument(
        "--building-flexibility-multiplier",
        type=float,
        default=1.0,
        help="uniform 0..1 derating applied to target-BEM building flexibility during replanning",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / ".local-runs" / "joint-dispatch",
    )
    parser.add_argument(
        "--service-direction",
        choices=("native", "reduction", "absorption"),
        default="native",
        help="Override the signed service direction for bidirectional envelope validation.",
    )
    parser.add_argument(
        "--fixed-building-response",
        type=Path,
        default=None,
        help="EnergyPlus validation JSON whose actual BEM response is fixed during storage/grid repair.",
    )
    return parser.parse_args()


def load_oracle(episode_id: str) -> EpisodeGroundTruth:
    path = REPO_ROOT / "cases" / "ieee33_10bem_15min" / "groundtruth" / "contract_oracles.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload["episode_id"] == episode_id:
                return EpisodeGroundTruth.model_validate(payload)
    raise KeyError(f"Unknown episode: {episode_id}")


def load_specification(args: argparse.Namespace) -> TaskSpecification:
    if args.task_specification is None:
        oracle = load_oracle(args.episode)
        if oracle.physics_truth_status not in {"pending", "verified"}:
            raise ValueError(f"{args.episode} does not require a numeric physics oracle")
        return oracle.task_specification
    path = args.task_specification.resolve()
    specification = TaskSpecification.model_validate(
        json.loads(path.read_text(encoding="utf-8"))
    )
    if specification.episode_id != args.episode:
        raise ValueError("Task specification and --episode IDs differ")
    if specification.case_id != "ieee33_10bem_15min":
        raise ValueError(f"Unsupported case: {specification.case_id}")
    if specification.clarification_required:
        raise ValueError("A task requiring clarification cannot enter the physics executor")
    return specification


def main() -> None:
    args = parse_args()
    specification = load_specification(args)
    data_root = Path(os.getenv("B2G_DATA_ROOT", REPO_ROOT / ".local-data"))
    fixed_building_response = None
    if args.fixed_building_response is not None:
        response_path = args.fixed_building_response.resolve()
        response = json.loads(response_path.read_text(encoding="utf-8"))
        if response["episode_id"] != args.episode:
            raise ValueError("Fixed EnergyPlus response must match the episode")
        safe_bems = {
            record["bem_id"]
            for record in response["records"]
            if int(record["comfort_violation_intervals"]) == 0
            and int(record["event_severe_error_count"]) == 0
        }
        fixed_building_response = {
            bem_id: values
            for bem_id, values in response[
                "actual_building_reduction_kw_by_bem"
            ].items()
            if bem_id in safe_bems
        }
        if not fixed_building_response:
            raise ValueError("Fixed EnergyPlus response contains no safe BEM profiles")
    optimizer = FrozenJointRCOPF(REPO_ROOT, data_root=data_root)
    result = optimizer.solve(
        specification,
        building_flexibility_multiplier_by_bem={
            bem_id: args.building_flexibility_multiplier
            for bem_id in specification.target_bems
        },
        service_direction_override={
            "native": None,
            "reduction": 1,
            "absorption": -1,
        }[args.service_direction],
        fixed_building_response_by_bem=fixed_building_response,
    )
    output_path = args.output_root / args.episode / "optimizer_result.json"
    digest = write_result(result, output_path)
    summary = {
        "episode_id": result.episode_id,
        "status": result.status,
        "requested_power_kw": result.requested_power_kw,
        "committed_power_kw": result.committed_power_kw,
        "delivery_ratio": result.delivery_ratio,
        "predicted_minimum_voltage_pu": result.predicted_minimum_voltage_pu,
        "predicted_maximum_line_loading_pct": result.predicted_maximum_line_loading_pct,
        "output": str(output_path),
        "sha256": digest,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
