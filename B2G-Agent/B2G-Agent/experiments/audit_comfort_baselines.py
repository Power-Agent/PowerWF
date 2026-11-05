"""Audit frozen target-day EnergyPlus baselines against configured comfort bounds."""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "cases" / "ieee33_10bem_15min"
PROFILE_PATH = (
    REPO_ROOT / ".local-data" / "ieee33_10bem_15min" / "profiles" / "bem_thermal_15min.csv"
)


def main() -> None:
    config = json.loads(
        (CASE_DIR / "optimization" / "joint_dispatch.json").read_text(encoding="utf-8")
    )
    comfort = config["comfort"]
    target_days = json.loads(
        (CASE_DIR / "days" / "target_days.json").read_text(encoding="utf-8")
    )["target_days"]
    by_bem: dict[str, list[dict[str, str]]] = {}
    with PROFILE_PATH.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            by_bem.setdefault(row["bem_id"], []).append(row)
    occupied_start = int(comfort["occupied_start"].split(":")[0]) * 4
    occupied_end = int(comfort["occupied_end"].split(":")[0]) * 4
    records: list[dict[str, object]] = []
    for day_name, day_value in target_days.items():
        day = date.fromisoformat(day_value)
        day_offset = (day - date(2025, 1, 1)).days * 96
        weekend = day.weekday() >= 5
        for bem_id, rows in by_bem.items():
            violations = 0
            minimum = float("inf")
            maximum = float("-inf")
            for interval, row in enumerate(rows[day_offset : day_offset + 96]):
                occupied = not weekend and occupied_start <= interval < occupied_end
                lower = float(
                    comfort["occupied_minimum_c"]
                    if occupied
                    else comfort["unoccupied_minimum_c"]
                )
                upper = float(
                    comfort["occupied_maximum_c"]
                    if occupied
                    else comfort["unoccupied_maximum_c"]
                )
                zone_minimum = float(row["minimum_zone_c"])
                zone_maximum = float(row["maximum_zone_c"])
                minimum = min(minimum, zone_minimum)
                maximum = max(maximum, zone_maximum)
                violations += zone_minimum < lower or zone_maximum > upper
            records.append(
                {
                    "operating_day": day_name,
                    "date": day_value,
                    "bem_id": bem_id,
                    "minimum_zone_c": minimum,
                    "maximum_zone_c": maximum,
                    "absolute_comfort_violation_intervals": violations,
                }
            )
    payload = {
        "schema_version": "1.0",
        "comfort_bounds": comfort,
        "record_count": len(records),
        "records_with_violations": sum(
            record["absolute_comfort_violation_intervals"] > 0 for record in records
        ),
        "total_violation_intervals": sum(
            int(record["absolute_comfort_violation_intervals"]) for record in records
        ),
        "global_minimum_zone_c": min(float(record["minimum_zone_c"]) for record in records),
        "global_maximum_zone_c": max(float(record["maximum_zone_c"]) for record in records),
        "records": records,
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
