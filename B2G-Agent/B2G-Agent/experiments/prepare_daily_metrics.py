"""Compute deterministic full-year baseline metrics used to select task days."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandapower as pp


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "ieee33_10bem_15min"
CASE_DIR = REPO_ROOT / "cases" / CASE_ID
DATA_ROOT = Path(os.getenv("B2G_DATA_ROOT", REPO_ROOT / ".local-data")).resolve()
LOCAL_CASE = DATA_ROOT / CASE_ID
PROFILE_DIR = LOCAL_CASE / "profiles"
EXPECTED_INTERVALS = 365 * 96


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_background(path: Path) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    p_by_bus = {bus: np.zeros(EXPECTED_INTERVALS) for bus in range(2, 34)}
    q_by_bus = {bus: np.zeros(EXPECTED_INTERVALS) for bus in range(2, 34)}
    counts = {bus: 0 for bus in range(2, 34)}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            bus = int(row["paper_bus"])
            index = counts[bus]
            if index >= EXPECTED_INTERVALS:
                raise ValueError(f"Too many background rows for bus {bus}")
            p_by_bus[bus][index] = float(row["p_kw"])
            q_by_bus[bus][index] = float(row["q_kvar"])
            counts[bus] += 1
    if any(count != EXPECTED_INTERVALS for count in counts.values()):
        raise ValueError(f"Background interval counts are incomplete: {counts}")
    return p_by_bus, q_by_bus


def load_bem_profile(
    path: Path, *, value_column: str
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    values: dict[str, np.ndarray] = {}
    buses: dict[str, int] = {}
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            bem_id = row["bem_id"]
            if bem_id not in values:
                values[bem_id] = np.zeros(EXPECTED_INTERVALS)
                buses[bem_id] = int(row["paper_bus"])
                counts[bem_id] = 0
            index = counts[bem_id]
            if index >= EXPECTED_INTERVALS:
                raise ValueError(f"Too many {path.name} rows for {bem_id}")
            values[bem_id][index] = float(row[value_column])
            counts[bem_id] += 1
    if len(values) != 10 or any(count != EXPECTED_INTERVALS for count in counts.values()):
        raise ValueError(f"{path.name} must contain 35040 rows for each of ten BEMs")
    return values, buses


def daily_dry_bulb() -> np.ndarray:
    try:
        import pvlib
    except ImportError as exc:
        raise RuntimeError("pvlib is required to read the frozen EPW") from exc
    weather_path = (
        LOCAL_CASE / "weather" / "USA_NY_Buffalo.Niagara.Intl_.AP_.725280_TMY3.epw"
    )
    weather, _ = pvlib.iotools.read_epw(weather_path)
    values = weather["temp_air"].to_numpy(dtype=float)
    if values.size != 365 * 24:
        raise ValueError("The frozen EPW must contain 8760 hourly dry-bulb values")
    return values.reshape(365, 24).mean(axis=1)


def main() -> None:
    input_paths = {
        "background": PROFILE_DIR / "background_load_15min.csv",
        "bem_baseline": PROFILE_DIR / "bem_baseline_15min.csv",
        "pv": PROFILE_DIR / "pv_15min.csv",
        "line_limits": CASE_DIR / "network" / "line_limits.csv",
        "transformer": CASE_DIR / "network" / "transformer.json",
    }
    missing = [str(path) for path in input_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Daily-metric inputs are missing: {missing}")

    background_p, background_q = load_background(input_paths["background"])
    bem, bem_buses = load_bem_profile(
        input_paths["bem_baseline"], value_column="portfolio_kw"
    )
    pv, pv_buses = load_bem_profile(input_paths["pv"], value_column="pv_kw")
    if bem_buses != pv_buses:
        raise ValueError("BEM baseline and PV bus mappings differ")

    net = pp.from_json(str(CASE_DIR / "network" / "source_case33bw.json"))
    load_index_by_bus = {int(row.bus) + 1: int(row.Index) for row in net.load.itertuples()}
    q_to_p_by_bus = {
        int(row.bus) + 1: float(row.q_mvar) / float(row.p_mw)
        for row in net.load.itertuples()
    }
    line_limits: dict[int, float] = {}
    with input_paths["line_limits"].open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            line_limits[int(row["line_index"])] = float(row["max_i_ka"])
    if set(line_limits) != set(int(index) for index in net.line[net.line.in_service].index):
        raise ValueError("Line-limit indices do not match the in-service feeder")

    bem_p_by_bus = {bus: np.zeros(EXPECTED_INTERVALS) for bus in range(2, 34)}
    pv_by_bus = {bus: np.zeros(EXPECTED_INTERVALS) for bus in range(2, 34)}
    for bem_id, values in bem.items():
        bem_p_by_bus[bem_buses[bem_id]] += values
        pv_by_bus[pv_buses[bem_id]] += pv[bem_id]

    daily_peak = np.full(365, -np.inf)
    daily_min_voltage = np.full(365, np.inf)
    daily_max_loading = np.full(365, -np.inf)
    gross_load_energy = np.zeros(365)
    gross_pv_energy = np.zeros(365)

    for interval in range(EXPECTED_INTERVALS):
        day_index = interval // 96
        for paper_bus in range(2, 34):
            p_kw = (
                background_p[paper_bus][interval]
                + bem_p_by_bus[paper_bus][interval]
                - pv_by_bus[paper_bus][interval]
            )
            q_kvar = (
                background_q[paper_bus][interval]
                + bem_p_by_bus[paper_bus][interval] * q_to_p_by_bus[paper_bus]
            )
            load_index = load_index_by_bus[paper_bus]
            net.load.at[load_index, "p_mw"] = p_kw / 1000.0
            net.load.at[load_index, "q_mvar"] = q_kvar / 1000.0
        options: dict[str, object] = {
            "algorithm": "nr",
            "calculate_voltage_angles": False,
            "init": "auto" if interval == 0 else "results",
            "tolerance_mva": 1e-8,
            "numba": False,
            "check_connectivity": interval == 0,
        }
        if interval:
            options["recycle"] = {"bus_pq": True, "trafo": False, "gen": False}
        pp.runpp(net, **options)
        if not net.converged:
            raise RuntimeError(f"Baseline power flow did not converge at interval {interval}")
        feeder_kw = float(net.res_ext_grid.p_mw.sum()) * 1000.0
        minimum_voltage = float(net.res_bus.vm_pu.min())
        maximum_loading = max(
            float(net.res_line.at[line_index, "i_ka"]) / limit * 100.0
            for line_index, limit in line_limits.items()
        )
        daily_peak[day_index] = max(daily_peak[day_index], feeder_kw)
        daily_min_voltage[day_index] = min(daily_min_voltage[day_index], minimum_voltage)
        daily_max_loading[day_index] = max(daily_max_loading[day_index], maximum_loading)
        gross_load_energy[day_index] += sum(
            background_p[bus][interval] + bem_p_by_bus[bus][interval]
            for bus in range(2, 34)
        ) * 0.25
        gross_pv_energy[day_index] += sum(
            pv_by_bus[bus][interval] for bus in range(2, 34)
        ) * 0.25
        if interval and interval % (30 * 96) == 0:
            print(f"processed_intervals={interval}")

    dry_bulb = daily_dry_bulb()
    output_path = PROFILE_DIR / "daily_metrics.csv"
    first_day = date(2025, 1, 1)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "date",
                "mean_dry_bulb_c",
                "feeder_peak_kw",
                "minimum_voltage_pu",
                "maximum_line_loading_pct",
                "gross_pv_to_load_ratio",
            ]
        )
        for day_index in range(365):
            writer.writerow(
                [
                    (first_day + timedelta(days=day_index)).isoformat(),
                    f"{dry_bulb[day_index]:.6f}",
                    f"{daily_peak[day_index]:.6f}",
                    f"{daily_min_voltage[day_index]:.9f}",
                    f"{daily_max_loading[day_index]:.6f}",
                    f"{gross_pv_energy[day_index] / gross_load_energy[day_index]:.9f}",
                ]
            )

    manifest = {
        "schema_version": "1.0",
        "case_id": CASE_ID,
        "derivation_script": "experiments/prepare_daily_metrics.py",
        "inputs": {
            name: {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/")
                if path.is_relative_to(REPO_ROOT)
                else str(path.relative_to(DATA_ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
            for name, path in input_paths.items()
        },
        "power_flow": {
            "pandapower_version": pp.__version__,
            "algorithm": "nr",
            "tolerance_mva": 1e-8,
            "bem_reactive_rule": "BEM portfolio P uses the original case33bw node Q/P ratio; PV is unity power factor",
        },
        "output": {
            "path": str(output_path.relative_to(DATA_ROOT)).replace("\\", "/"),
            "row_count": 365,
            "sha256": sha256_file(output_path),
        },
    }
    manifest_path = PROFILE_DIR / "daily_metrics_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path} rows=365 sha256={manifest['output']['sha256']}")


if __name__ == "__main__":
    main()
