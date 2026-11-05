"""Run paired two-day EnergyPlus baseline/event IDFs for one candidate dispatch."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.simulation.adapters import (  # noqa: E402
    FormalEnergyPlusAdapter,
    FormalEnergyPlusConfig,
)
from b2g_agent.simulation.joint_rc_opf import interval_index  # noqa: E402


CASE_ID = "ieee33_10bem_15min"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-result", type=Path, required=True)
    parser.add_argument("--event-idf-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    return parser.parse_args()


def aggregation_factors(path: Path) -> dict[str, float]:
    values: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values.setdefault(row["bem_id"], float(row["aggregation_factor"]))
    if len(values) != 10:
        raise ValueError("Ten BEM aggregation factors are required")
    return values


def main() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    args = parse_args()
    optimizer_path = args.optimizer_result.resolve()
    optimizer = json.loads(optimizer_path.read_text(encoding="utf-8"))
    event_root = args.event_idf_root.resolve()
    manifest = json.loads((event_root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["episode_id"] != optimizer["episode_id"]:
        raise ValueError("Optimizer and event-IDF episode IDs differ")
    run_root = args.run_root.resolve()
    if run_root.exists():
        raise FileExistsError(run_root)
    data_root = Path(os.getenv("B2G_DATA_ROOT", REPO_ROOT / ".local-data")).resolve()
    local_case = data_root / CASE_ID
    factors = aggregation_factors(local_case / "profiles" / "bem_baseline_15min.csv")
    config = json.loads(
        (
            REPO_ROOT
            / "cases"
            / CASE_ID
            / "optimization"
            / "joint_dispatch.json"
        ).read_text(encoding="utf-8")
    )
    energyplus = FormalEnergyPlusAdapter(
        FormalEnergyPlusConfig.from_case(data_root=data_root, case_id=CASE_ID)
    )
    month_day = date.fromisoformat(manifest["target_date"]).strftime("%m/%d")
    occupied_start = interval_index(config["comfort"]["occupied_start"])
    occupied_end = interval_index(config["comfort"]["occupied_end"])
    weekend = date.fromisoformat(manifest["target_date"]).weekday() >= 5
    records = []
    actual_by_bem: dict[str, list[float]] = {}
    total_comfort_violations = 0
    total_baseline_violations = 0
    for record in manifest["records"]:
        bem_id = record["bem_id"]
        baseline = energyplus.run(
            bem_id=bem_id,
            idf_path=event_root / bem_id / "baseline.idf",
            date_mm_dd=month_day,
            run_dir=run_root / bem_id / "baseline",
        )
        event = energyplus.run(
            bem_id=bem_id,
            idf_path=event_root / bem_id / "event.idf",
            date_mm_dd=month_day,
            run_dir=run_root / bem_id / "event",
        )
        factor = factors[bem_id]
        actual = [
            (baseline.electricity_kw[index] - event.electricity_kw[index]) * factor
            for index in range(96)
        ]
        actual_by_bem[bem_id] = actual
        violations = 0
        baseline_violations = 0
        if (
            baseline.minimum_zone_temperature_c is None
            or baseline.maximum_zone_temperature_c is None
            or event.minimum_zone_temperature_c is None
            or event.maximum_zone_temperature_c is None
        ):
            raise RuntimeError(f"Zone temperatures are unavailable for {bem_id}")
        tolerance = float(config["comfort"]["linear_screening_tolerance_c"])
        for interval in range(96):
            occupied = not weekend and occupied_start <= interval < occupied_end
            lower = float(
                config["comfort"]["occupied_minimum_c"]
                if occupied
                else config["comfort"]["unoccupied_minimum_c"]
            )
            upper = float(
                config["comfort"]["occupied_maximum_c"]
                if occupied
                else config["comfort"]["unoccupied_maximum_c"]
            )
            baseline_low = baseline.minimum_zone_temperature_c[interval]
            baseline_high = baseline.maximum_zone_temperature_c[interval]
            event_low = event.minimum_zone_temperature_c[interval]
            event_high = event.maximum_zone_temperature_c[interval]
            if baseline_low < lower or baseline_high > upper:
                baseline_violations += 1
            effective_lower = min(lower, baseline_low) - tolerance
            effective_upper = max(upper, baseline_high) + tolerance
            if event_low < effective_lower or event_high > effective_upper:
                violations += 1
        total_comfort_violations += violations
        total_baseline_violations += baseline_violations
        desired = optimizer["schedules"][bem_id]["building_reduction_kw"]
        records.append(
            {
                "bem_id": bem_id,
                "aggregation_factor": factor,
                "desired_building_reduction_kw": desired,
                "actual_building_reduction_kw": actual,
                "mean_absolute_tracking_error_kw": sum(
                    abs(float(desired[index]) - actual[index]) for index in range(96)
                )
                / 96.0,
                "maximum_absolute_tracking_error_kw": max(
                    abs(float(desired[index]) - actual[index]) for index in range(96)
                ),
                "comfort_violation_intervals": violations,
                "baseline_absolute_comfort_violation_intervals": baseline_violations,
                "baseline_warning_count": baseline.warning_count,
                "event_warning_count": event.warning_count,
                "baseline_severe_error_count": baseline.severe_error_count,
                "event_severe_error_count": event.severe_error_count,
                "baseline_input_hashes": baseline.input_hashes,
                "event_input_hashes": event.input_hashes,
            }
        )
        print(
            f"validated {bem_id}: comfort_violations={violations} "
            f"mae_kw={records[-1]['mean_absolute_tracking_error_kw']:.6f}"
        )
    payload = {
        "schema_version": "1.0",
        "episode_id": optimizer["episode_id"],
        "target_date": manifest["target_date"],
        "status": "pass" if total_comfort_violations == 0 else "failed",
        "response_comfort_violation_intervals": total_comfort_violations,
        "baseline_absolute_comfort_violation_intervals": total_baseline_violations,
        "comfort_validation_rule": config["comfort"]["validation_rule"],
        "actual_building_reduction_kw_by_bem": actual_by_bem,
        "records": records,
        "energyplus_version": energyplus.version(),
    }
    output_path = run_root / "energyplus_event_validation.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output_path)}, indent=2))


if __name__ == "__main__":
    main()
