"""Run ten annual EnergyPlus baselines and freeze 15-minute BEM profiles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandapower as pp
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.case_io import CaseBundle  # noqa: E402
from b2g_agent.simulation.adapters.energyplus import (  # noqa: E402
    _interval_value_to_kw,
    _matching_column,
    _severe_count,
    _stable_version,
    _warning_count,
)
from b2g_agent.simulation.energyplus_controls import controlled_zone_names  # noqa: E402


EXPECTED_INTERVALS = 365 * 96


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def executable_path() -> Path:
    configured = os.getenv("B2G_ENERGYPLUS_EXE", "energyplus").strip()
    candidate = Path(configured).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    discovered = shutil.which(configured)
    if discovered:
        return Path(discovered).resolve()
    raise FileNotFoundError("EnergyPlus is unavailable; configure B2G_ENERGYPLUS_EXE in .env")


def read_annual_output(
    path: Path, *, zone_names: set[str]
) -> tuple[list[float], list[tuple[float, float, float]]]:
    electricity: list[float] = []
    temperatures: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"EnergyPlus CSV has no header: {path}")
        date_column = _matching_column(reader.fieldnames, "Date/Time")
        meter_column = _matching_column(reader.fieldnames, "Electricity:Facility")
        temperature_columns = [
            column
            for column in reader.fieldnames
            if "zone mean air temperature" in column.lower()
            and column.split(":", 1)[0].strip().upper() in zone_names
        ]
        if not temperature_columns:
            raise ValueError(f"No thermostat-controlled zone temperature outputs in {path}")
        for row in reader:
            if not re.match(r"\s*\d+\s*/\s*\d+", row.get(date_column, "")):
                continue
            electricity.append(
                _interval_value_to_kw(float(row[meter_column]), meter_column, 15)
            )
            values = [float(row[column]) for column in temperature_columns]
            temperatures.append((min(values), sum(values) / len(values), max(values)))
    if len(electricity) != EXPECTED_INTERVALS:
        raise ValueError(
            f"Expected {EXPECTED_INTERVALS} annual EnergyPlus rows in {path}, "
            f"found {len(electricity)}"
        )
    if len(temperatures) != EXPECTED_INTERVALS:
        raise ValueError(f"Temperature row count mismatch in {path}")
    return electricity, temperatures


def main() -> None:
    load_dotenv(REPO_ROOT / ".env", override=False)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / ".local-runs" / "energyplus-baseline",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args()

    case = CaseBundle.default(REPO_ROOT)
    local_case = case.local_data_root / case.case_id
    weather = (
        local_case
        / "weather"
        / "USA_NY_Buffalo.Niagara.Intl_.AP_.725280_TMY3.epw"
    )
    if not weather.is_file():
        raise FileNotFoundError(f"Weather file is missing: {weather}")
    executable = executable_path()
    version_run = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    version = _stable_version(
        (version_run.stdout or version_run.stderr).strip().splitlines()[0]
    )
    results_root = args.results_root.resolve()
    profiles: dict[str, list[float]] = {}
    thermal: dict[str, list[tuple[float, float, float]]] = {}
    records: list[dict[str, object]] = []
    record_by_bem: dict[str, dict[str, object]] = {}

    for index in range(1, 11):
        bem_id = f"bem_{index:02d}"
        metadata_path = case.case_dir / "buildings" / bem_id / "building.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        idf = local_case / "buildings" / bem_id / "energyplus_23_1.idf"
        run_dir = results_root / bem_id
        output_dir = run_dir / "output"
        if not idf.is_file():
            raise FileNotFoundError(f"Runtime IDF is missing: {idf}")
        replicate_from = metadata.get("replicate_runtime_from")
        if replicate_from:
            if replicate_from not in profiles:
                raise ValueError(
                    f"{bem_id} declares replicate_runtime_from={replicate_from}, "
                    "but the source profile has not been processed"
                )
            source_record = record_by_bem[str(replicate_from)]
            profiles[bem_id] = profiles[str(replicate_from)].copy()
            thermal[bem_id] = thermal[str(replicate_from)].copy()
            record = {
                "bem_id": bem_id,
                "execution_mode": "byte_identical_runtime_replica",
                "replicated_from": replicate_from,
                "idf_sha256": sha256_file(idf),
                "warning_count": source_record["warning_count"],
                "severe_error_count": 0,
                "error_report_sha256": source_record["error_report_sha256"],
                "csv_sha256": source_record["csv_sha256"],
                "source_result_directory": source_record["result_directory"],
            }
            records.append(record)
            record_by_bem[bem_id] = record
            print(f"replicated annual baseline {bem_id} from {replicate_from}")
            continue

        error_path = output_dir / "eplusout.err"
        csv_path = output_dir / "eplusout.csv"
        complete_existing = error_path.is_file() and csv_path.is_file()
        reused_existing = args.reuse_existing and complete_existing
        if not reused_existing:
            if run_dir.exists():
                raise FileExistsError(
                    f"Baseline run is incomplete or reuse was not requested: {run_dir}; "
                    "remove this specific run directory or choose a new --results-root"
                )
            output_dir.mkdir(parents=True)
            command = [
                str(executable),
                "--annual",
                "--weather",
                str(weather),
                "--output-directory",
                str(output_dir),
                "--readvars",
                str(idf),
            ]
            completed = subprocess.run(
                command,
                cwd=run_dir,
                capture_output=True,
                text=True,
                timeout=int(os.getenv("B2G_ENERGYPLUS_TIMEOUT_SECONDS", "1800")),
                check=False,
            )
            (run_dir / "command.json").write_text(
                json.dumps(
                    {
                        "command": [
                            executable.name,
                            "--annual",
                            "--weather",
                            weather.name,
                            "--output-directory",
                            "output",
                            "--readvars",
                            idf.name,
                        ],
                        "returncode": completed.returncode,
                        "stdout_tail": completed.stdout[-4000:],
                        "stderr_tail": completed.stderr[-4000:],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            (run_dir / "input_manifest.json").write_text(
                json.dumps(
                    {
                        "bem_id": bem_id,
                        "idf_sha256": sha256_file(idf),
                        "weather_sha256": sha256_file(weather),
                        "energyplus_version": version,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            if completed.returncode != 0:
                raise RuntimeError(f"EnergyPlus returned {completed.returncode} for {bem_id}")
        if not error_path.is_file() or not csv_path.is_file():
            raise FileNotFoundError(f"Incomplete EnergyPlus artifacts for {bem_id}")
        severe_count = _severe_count(error_path)
        if severe_count:
            raise RuntimeError(f"EnergyPlus produced {severe_count} Severe records for {bem_id}")
        zones = controlled_zone_names(idf.read_text(encoding="utf-8", errors="replace"))
        profile, temperature = read_annual_output(csv_path, zone_names=zones)
        profiles[bem_id] = profile
        thermal[bem_id] = temperature
        input_manifest_path = run_dir / "input_manifest.json"
        input_binding_verified = False
        if input_manifest_path.is_file():
            run_inputs = json.loads(input_manifest_path.read_text(encoding="utf-8"))
            input_binding_verified = (
                run_inputs.get("idf_sha256") == sha256_file(idf)
                and run_inputs.get("weather_sha256") == sha256_file(weather)
                and _stable_version(str(run_inputs.get("energyplus_version"))) == version
            )
            if not input_binding_verified:
                raise RuntimeError(f"Existing EnergyPlus inputs do not match for {bem_id}")
        record = {
            "bem_id": bem_id,
            "execution_mode": "reused_existing" if reused_existing else "executed",
            "input_binding_verified": input_binding_verified,
            "controlled_zone_count": len(zones),
            "idf_sha256": sha256_file(idf),
            "warning_count": _warning_count(error_path),
            "severe_error_count": severe_count,
            "error_report_sha256": sha256_file(error_path),
            "csv_sha256": sha256_file(csv_path),
            "result_directory": str(run_dir.relative_to(REPO_ROOT)).replace("\\", "/"),
        }
        records.append(record)
        record_by_bem[bem_id] = record
        print(f"validated annual baseline {bem_id}: warnings={record['warning_count']}")

    net = pp.from_json(str(case.case_dir / "network" / "source_case33bw.json"))
    base_p_kw = {int(row.bus) + 1: float(row.p_mw) * 1000.0 for row in net.load.itertuples()}
    mapping = case.bem_mapping()
    scaling = {
        bem_id: (0.40 * base_p_kw[paper_bus]) / max(profiles[bem_id])
        for bem_id, paper_bus in mapping.items()
    }
    profile_dir = local_case / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = profile_dir / "bem_baseline_15min.csv"
    thermal_path = profile_dir / "bem_thermal_15min.csv"
    start = datetime(2025, 1, 1)
    with baseline_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "bem_id",
                "paper_bus",
                "single_building_kw",
                "aggregation_factor",
                "portfolio_kw",
            ]
        )
        for bem_id in sorted(mapping):
            factor = scaling[bem_id]
            for interval, single_kw in enumerate(profiles[bem_id]):
                timestamp = start + timedelta(minutes=15 * interval)
                writer.writerow(
                    [
                        timestamp.isoformat(timespec="seconds"),
                        bem_id,
                        mapping[bem_id],
                        f"{single_kw:.9f}",
                        f"{factor:.12f}",
                        f"{single_kw * factor:.9f}",
                    ]
                )
    with thermal_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["timestamp", "bem_id", "minimum_zone_c", "mean_zone_c", "maximum_zone_c"]
        )
        for bem_id in sorted(mapping):
            for interval, (minimum, mean, maximum) in enumerate(thermal[bem_id]):
                timestamp = start + timedelta(minutes=15 * interval)
                writer.writerow(
                    [
                        timestamp.isoformat(timespec="seconds"),
                        bem_id,
                        f"{minimum:.6f}",
                        f"{mean:.6f}",
                        f"{maximum:.6f}",
                    ]
                )

    manifest = {
        "schema_version": "1.0",
        "case_id": case.case_id,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "energyplus_version": version,
        "weather_sha256": sha256_file(weather),
        "model_count_by_bem": {bem_id: 1 for bem_id in sorted(mapping)},
        "aggregation_factor_by_bem": scaling,
        "aggregation_rule": "one official prototype per BEM, scaled so its annual portfolio peak is 40% of the original case33bw node P load; temperatures are never scaled",
        "records": records,
        "outputs": [
            {
                "path": str(baseline_path.relative_to(case.local_data_root)).replace("\\", "/"),
                "row_count": EXPECTED_INTERVALS * 10,
                "sha256": sha256_file(baseline_path),
            },
            {
                "path": str(thermal_path.relative_to(case.local_data_root)).replace("\\", "/"),
                "row_count": EXPECTED_INTERVALS * 10,
                "sha256": sha256_file(thermal_path),
            },
        ],
    }
    manifest_path = profile_dir / "bem_baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote baseline_rows={EXPECTED_INTERVALS * 10} manifest={manifest_path}")


if __name__ == "__main__":
    main()
