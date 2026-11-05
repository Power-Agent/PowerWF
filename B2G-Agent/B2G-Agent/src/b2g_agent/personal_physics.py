"""Event inputs for private interactive tasks, without changing the formal kernel.

The formal prepare_event_idfs.py remains immutable. This adapter binds its same
control translation to the reviewed task's operating day instead of an oracle ID.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
from datetime import date
import hashlib
import json
import math
from pathlib import Path
import re
import sys

if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from b2g_agent.case_io import CaseBundle, find_repo_root
from b2g_agent.contracts import TaskSpecification
from b2g_agent.contracts.readiness import execution_readiness
from b2g_agent.simulation.energyplus_controls import build_event_idfs, idf_objects


def enable_native_csv(path: Path) -> None:
    """Select EnergyPlus's own CSV writer; leave every physical object intact."""
    text = path.read_text(encoding="utf-8")
    pattern = re.compile(r"^[ \t]*OutputControl:Files\s*(?:,[^;]*)?;", re.I | re.M)
    matches = list(pattern.finditer(text))
    if len(matches) > 1:
        raise ValueError("Multiple OutputControl:Files objects are not valid")
    if matches:
        match = matches[0]
        fields = idf_objects(match.group())[0][1]
        if len(fields) == 1:
            fields.append("Yes")
        else:
            fields[1] = "Yes"
        text = text[:match.start()] + ",".join(fields) + ";" + text[match.end():]
    else:
        text += "\nOutputControl:Files,Yes;\n"
    path.write_text(text, encoding="utf-8")


def prepare_event_inputs(*, task_specification: Path, optimizer_result: Path,
                         output_root: Path, repo_root: Path | None = None,
                         native_csv: bool = False) -> Path:
    root = repo_root or find_repo_root()
    case = CaseBundle.default(root)
    source = task_specification.read_bytes()
    specification = TaskSpecification.model_validate_json(source)
    optimizer = json.loads(optimizer_result.read_text(encoding="utf-8"))
    if specification.episode_id != optimizer["episode_id"]:
        raise ValueError("Interactive task and optimizer IDs differ")
    problems = execution_readiness(specification, case=case)
    if problems:
        raise ValueError("Interactive physics inputs are unresolved: " + "; ".join(p.message for p in problems))
    target_days = json.loads((case.case_dir / "days/target_days.json").read_text(encoding="utf-8"))["target_days"]
    target_date = target_days[specification.operating_day]
    local_case = case.local_data_root / case.case_id
    baseline: dict[str, list[float]] = {}
    with (local_case / "profiles/bem_baseline_15min.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["timestamp"][:10] == target_date:
                baseline.setdefault(row["bem_id"], []).append(float(row["portfolio_kw"]))
    if len(baseline) != 10 or any(len(values) != 96 for values in baseline.values()):
        raise ValueError("Baseline day extraction did not produce ten 96-interval profiles")
    config = json.loads((case.case_dir / "optimization/joint_dispatch.json").read_text(encoding="utf-8"))
    control = config["energyplus_control_translation"]
    maximum_shift = float(control["maximum_setpoint_shift_c"])
    controllable_fraction = float(control["assumed_controllable_electric_fraction"])
    minimum_multiplier = float(control["minimum_electric_equipment_schedule_multiplier"])
    maximum_multiplier = float(control["maximum_electric_equipment_schedule_multiplier"])
    schedules = optimizer["schedules"]
    if not schedules or set(schedules) - set(baseline):
        raise ValueError("Optimizer schedules contain missing or unknown BEMs")
    for schedule in schedules.values():
        values = schedule["building_reduction_kw"]
        if len(values) != 96 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
            raise ValueError("Optimizer schedules require 96 finite power values")
    if output_root.exists():
        raise FileExistsError(output_root)
    records = []
    for bem_id, schedule in schedules.items():
        offsets, multipliers = [], []
        for interval, adjustment in enumerate(schedule["building_reduction_kw"]):
            denominator = baseline[bem_id][interval]
            fraction = float(adjustment) / denominator if denominator > 1e-9 else 0.0
            offsets.append(max(0.0, min(1.0, fraction)) * maximum_shift)
            raw_multiplier = (1.0 - float(adjustment) / (denominator * controllable_fraction)
                              if denominator > 1e-9 else 1.0)
            multipliers.append(max(minimum_multiplier, min(maximum_multiplier, raw_multiplier)))
        bem_dir = output_root / bem_id
        metadata = build_event_idfs(
            source_idf=local_case / "buildings" / bem_id / "energyplus_23_1.idf",
            baseline_idf=bem_dir / "baseline.idf", event_idf=bem_dir / "event.idf",
            target_date=date.fromisoformat(target_date), offsets_c=offsets,
            load_multipliers=multipliers,
        )
        if native_csv:
            enable_native_csv(bem_dir / "baseline.idf")
            enable_native_csv(bem_dir / "event.idf")
            metadata = replace(metadata,
                baseline_sha256=hashlib.sha256((bem_dir / "baseline.idf").read_bytes()).hexdigest(),
                event_sha256=hashlib.sha256((bem_dir / "event.idf").read_bytes()).hexdigest())
        records.append({"bem_id": bem_id, **asdict(metadata), "offsets_c": offsets, "load_multipliers": multipliers})
    manifest = {
        "schema_version": "1.0", "episode_id": specification.episode_id,
        "operating_day": specification.operating_day, "target_date": target_date,
        "translation_rule": control["translation_rule"], "records": records,
        "interactive_task_input": {"sha256": hashlib.sha256(source).hexdigest()},
    }
    if native_csv:
        manifest["interactive_task_input"]["csv_export"] = "energyplus_native"
    path = output_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-specification", type=Path, required=True)
    parser.add_argument("--optimizer-result", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--native-csv", action="store_true")
    args = parser.parse_args()
    path = prepare_event_inputs(task_specification=args.task_specification.resolve(),
        optimizer_result=args.optimizer_result.resolve(), output_root=args.output_root.resolve(),
        native_csv=args.native_csv)
    print(json.dumps({"manifest": str(path)}))


if __name__ == "__main__":
    main()
