"""Select and freeze ten physical task days without using episode outcomes."""

from __future__ import annotations

import csv
import json
import os
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "cases" / "ieee33_10bem_15min"
DATA_ROOT = Path(os.getenv("B2G_DATA_ROOT", REPO_ROOT / ".local-data")).resolve()
METRICS = DATA_ROOT / "ieee33_10bem_15min" / "profiles" / "daily_metrics.csv"


def main() -> None:
    split = json.loads((CASE_DIR / "days" / "calendar_split.json").read_text(encoding="utf-8"))
    candidates = set(split["test_candidate_dates"])
    references = set(split["reference_dates"])
    with METRICS.open("r", encoding="utf-8", newline="") as handle:
        rows = [{**row, "month": int(row["date"][5:7])} for row in csv.DictReader(handle)]
    by_date = {row["date"]: row for row in rows}
    if not candidates <= by_date.keys() or not references <= by_date.keys():
        raise ValueError("daily_metrics.csv does not cover all frozen candidate/reference dates")
    unused = set(candidates)
    selected: dict[str, str] = {}

    def take(label: str, eligible: set[str], metric: str, *, reverse: bool) -> None:
        ranked = sorted(
            eligible & unused,
            key=lambda day: ((-1 if reverse else 1) * float(by_date[day][metric]), day),
        )
        if not ranked:
            raise RuntimeError(f"No unused candidate for {label}")
        selected[label] = ranked[0]
        unused.remove(ranked[0])

    winter = {day for day in candidates if int(day[5:7]) in {12, 1, 2}}
    summer = {day for day in candidates if int(day[5:7]) in {6, 7, 8}}
    shoulder = candidates - winter - summer
    take("winter_peak_01", winter, "feeder_peak_kw", reverse=True)
    take("winter_peak_02", winter, "feeder_peak_kw", reverse=True)
    take("summer_peak_01", summer, "feeder_peak_kw", reverse=True)
    take("summer_peak_02", summer, "feeder_peak_kw", reverse=True)
    take("pv_surplus_01", candidates, "gross_pv_to_load_ratio", reverse=True)
    take("pv_surplus_02", candidates, "gross_pv_to_load_ratio", reverse=True)
    take("voltage_risk_01", candidates, "minimum_voltage_pu", reverse=False)
    take("congestion_risk_01", candidates, "maximum_line_loading_pct", reverse=True)
    shoulder_peaks = sorted(float(by_date[day]["feeder_peak_kw"]) for day in shoulder & unused)
    median = shoulder_peaks[len(shoulder_peaks) // 2]
    for label in ("shoulder_01", "shoulder_02"):
        day = min(
            shoulder & unused,
            key=lambda value: (abs(float(by_date[value]["feeder_peak_kw"]) - median), value),
        )
        selected[label] = day
        unused.remove(day)

    reference_days: dict[str, list[str]] = {}
    for label, target in selected.items():
        target_date = date.fromisoformat(target)
        target_temp = float(by_date[target]["mean_dry_bulb_c"])
        ranked = sorted(
            references,
            key=lambda day: (
                date.fromisoformat(day).weekday() != target_date.weekday(),
                abs(float(by_date[day]["mean_dry_bulb_c"]) - target_temp),
                day,
            ),
        )
        reference_days[label] = ranked[:5]
    payload = {
        "schema_version": "1.0",
        "target_days": selected,
        "reference_days": reference_days,
        "source_daily_metrics": str(METRICS.relative_to(DATA_ROOT)).replace("\\", "/"),
        "selection_rule": "cases/ieee33_10bem_15min/days/day_selection.json",
    }
    output = CASE_DIR / "days" / "target_days.json"
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
