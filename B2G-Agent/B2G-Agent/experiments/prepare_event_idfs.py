"""Translate one optimizer result into auditable two-day EnergyPlus event IDFs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.simulation.energyplus_controls import build_event_idfs  # noqa: E402


CASE_ID = "ieee33_10bem_15min"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_baseline_day(path: Path, target_date: str) -> dict[str, list[float]]:
    values: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["timestamp"][:10] != target_date:
                continue
            values.setdefault(row["bem_id"], []).append(float(row["portfolio_kw"]))
    if len(values) != 10 or any(len(profile) != 96 for profile in values.values()):
        raise ValueError("Baseline day extraction did not produce ten 96-interval profiles")
    return values


def main() -> None:
    args = parse_args()
    optimizer_path = args.optimizer_result.resolve()
    optimizer = json.loads(optimizer_path.read_text(encoding="utf-8"))
    data_root = Path(os.getenv("B2G_DATA_ROOT", REPO_ROOT / ".local-data")).resolve()
    local_case = data_root / CASE_ID
    case_dir = REPO_ROOT / "cases" / CASE_ID
    target_days = json.loads((case_dir / "days" / "target_days.json").read_text(encoding="utf-8"))[
        "target_days"
    ]
    oracle_path = case_dir / "groundtruth" / "contract_oracles.jsonl"
    operating_day = None
    with oracle_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload["episode_id"] == optimizer["episode_id"]:
                operating_day = payload["task_specification"]["operating_day"]
                break
    if operating_day is None:
        raise KeyError(optimizer["episode_id"])
    target_date_value = target_days[operating_day]
    baseline_path = local_case / "profiles" / "bem_baseline_15min.csv"
    baseline = load_baseline_day(baseline_path, target_date_value)
    config = json.loads(
        (case_dir / "optimization" / "joint_dispatch.json").read_text(encoding="utf-8")
    )
    maximum_shift = float(
        config["energyplus_control_translation"]["maximum_setpoint_shift_c"]
    )
    control = config["energyplus_control_translation"]
    controllable_fraction = float(control["assumed_controllable_electric_fraction"])
    minimum_multiplier = float(
        control["minimum_electric_equipment_schedule_multiplier"]
    )
    maximum_multiplier = float(
        control["maximum_electric_equipment_schedule_multiplier"]
    )
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(output_root)
    records = []
    for bem_id, schedule in optimizer["schedules"].items():
        offsets = []
        load_multipliers = []
        for interval, adjustment in enumerate(schedule["building_reduction_kw"]):
            denominator = baseline[bem_id][interval]
            fraction = float(adjustment) / denominator if denominator > 1e-9 else 0.0
            # Symmetric heating/cooling offsets are safe only when they widen
            # the thermostat deadband.  Load-increase requests therefore use
            # ElectricEquipment and storage charging, while setpoint offsets
            # remain at zero instead of contracting or inverting the deadband.
            offsets.append(max(0.0, min(1.0, fraction)) * maximum_shift)
            if denominator > 1e-9:
                raw_multiplier = 1.0 - float(adjustment) / (
                    denominator * controllable_fraction
                )
            else:
                raw_multiplier = 1.0
            load_multipliers.append(
                max(minimum_multiplier, min(maximum_multiplier, raw_multiplier))
            )
        bem_dir = output_root / bem_id
        metadata = build_event_idfs(
            source_idf=local_case / "buildings" / bem_id / "energyplus_23_1.idf",
            baseline_idf=bem_dir / "baseline.idf",
            event_idf=bem_dir / "event.idf",
            target_date=date.fromisoformat(target_date_value),
            offsets_c=offsets,
            load_multipliers=load_multipliers,
        )
        records.append(
            {
                "bem_id": bem_id,
                **asdict(metadata),
                "offsets_c": offsets,
                "load_multipliers": load_multipliers,
            }
        )
    manifest = {
        "schema_version": "1.0",
        "episode_id": optimizer["episode_id"],
        "operating_day": operating_day,
        "target_date": target_date_value,
        "translation_rule": config["energyplus_control_translation"]["translation_rule"],
        "records": records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "bem_count": len(records)}, indent=2))


if __name__ == "__main__":
    main()
