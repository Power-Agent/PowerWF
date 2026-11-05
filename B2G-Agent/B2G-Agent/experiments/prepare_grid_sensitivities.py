"""Freeze target-day AC finite-difference sensitivities for the linear RC-OPF."""

from __future__ import annotations

import csv
import hashlib
import json
import os
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


def load_background(path: Path) -> tuple[np.ndarray, np.ndarray]:
    active = np.zeros((32, EXPECTED_INTERVALS))
    reactive = np.zeros((32, EXPECTED_INTERVALS))
    counts = np.zeros(32, dtype=int)
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            bus_index = int(row["paper_bus"]) - 2
            interval = counts[bus_index]
            if interval >= EXPECTED_INTERVALS:
                raise ValueError(f"Too many background rows for bus {bus_index + 2}")
            active[bus_index, interval] = float(row["p_kw"])
            reactive[bus_index, interval] = float(row["q_kvar"])
            counts[bus_index] += 1
    if np.any(counts != EXPECTED_INTERVALS):
        raise ValueError("Background profile counts are incomplete")
    return active, reactive


def load_bem(path: Path, column: str) -> tuple[dict[str, np.ndarray], dict[str, int]]:
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
            interval = counts[bem_id]
            values[bem_id][interval] = float(row[column])
            counts[bem_id] += 1
    if len(values) != 10 or any(count != EXPECTED_INTERVALS for count in counts.values()):
        raise ValueError(f"Incomplete BEM profile: {path}")
    return values, buses


def read_line_limits(path: Path) -> tuple[list[int], dict[int, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        values = {int(row["line_index"]): float(row["max_i_ka"]) for row in csv.DictReader(handle)}
    return sorted(values), values


def run_power_flow(
    net: object,
    load_index_by_bus: dict[int, int],
    active_kw: np.ndarray,
    reactive_kvar: np.ndarray,
    line_indices: list[int],
    line_limits: dict[int, float],
    transformer_kva: float,
    *,
    init: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    for paper_bus in range(2, 34):
        load_index = load_index_by_bus[paper_bus]
        net.load.at[load_index, "p_mw"] = active_kw[paper_bus - 2] / 1000.0
        net.load.at[load_index, "q_mvar"] = reactive_kvar[paper_bus - 2] / 1000.0
    pp.runpp(
        net,
        algorithm="nr",
        calculate_voltage_angles=False,
        init=init,
        tolerance_mva=1e-8,
        numba=False,
        check_connectivity=init == "auto",
        recycle=None if init == "auto" else {"bus_pq": True, "trafo": False, "gen": False},
    )
    if not net.converged:
        raise RuntimeError("AC power flow did not converge during sensitivity preparation")
    voltage = net.res_bus.vm_pu.to_numpy(dtype=float)
    loading = np.array(
        [float(net.res_line.at[index, "i_ka"]) / line_limits[index] * 100.0 for index in line_indices]
    )
    p_kw = float(net.res_ext_grid.p_mw.sum()) * 1000.0
    q_kvar = float(net.res_ext_grid.q_mvar.sum()) * 1000.0
    transformer = np.hypot(p_kw, q_kvar) / transformer_kva * 100.0
    return voltage, loading, float(transformer)


def main() -> None:
    config_path = CASE_DIR / "optimization" / "joint_dispatch.json"
    target_path = CASE_DIR / "days" / "target_days.json"
    background_path = PROFILE_DIR / "background_load_15min.csv"
    baseline_path = PROFILE_DIR / "bem_baseline_15min.csv"
    pv_path = PROFILE_DIR / "pv_15min.csv"
    network_path = CASE_DIR / "network" / "source_case33bw.json"
    limits_path = CASE_DIR / "network" / "line_limits.csv"
    transformer_path = CASE_DIR / "network" / "transformer.json"
    inputs = {
        "config": config_path,
        "target_days": target_path,
        "background": background_path,
        "bem_baseline": baseline_path,
        "pv": pv_path,
        "network": network_path,
        "line_limits": limits_path,
        "transformer": transformer_path,
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Sensitivity inputs are missing: {missing}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    target_days = json.loads(target_path.read_text(encoding="utf-8"))["target_days"]
    day_names = list(target_days)
    dates = [target_days[name] for name in day_names]
    perturbation_kw = float(config["finite_difference_reduction_kw"])
    background_p, background_q = load_background(background_path)
    bem, bem_buses = load_bem(baseline_path, "portfolio_kw")
    pv, pv_buses = load_bem(pv_path, "pv_kw")
    if bem_buses != pv_buses:
        raise ValueError("BEM and PV mappings differ")
    bem_ids = sorted(bem)
    net = pp.from_json(str(network_path))
    load_index_by_bus = {int(row.bus) + 1: int(row.Index) for row in net.load.itertuples()}
    q_to_p = {
        int(row.bus) + 1: float(row.q_mvar) / float(row.p_mw)
        for row in net.load.itertuples()
    }
    line_indices, line_limits = read_line_limits(limits_path)
    transformer_kva = float(
        json.loads(transformer_path.read_text(encoding="utf-8"))["rating_kva"]
    )

    shape = (len(day_names), 96)
    baseline_voltage = np.zeros(shape + (33,))
    baseline_line = np.zeros(shape + (len(line_indices),))
    baseline_transformer = np.zeros(shape)
    voltage_sensitivity = np.zeros(shape + (33, len(bem_ids)))
    line_sensitivity = np.zeros(shape + (len(line_indices), len(bem_ids)))
    transformer_sensitivity = np.zeros(shape + (len(bem_ids),))

    for day_index, day_value in enumerate(dates):
        start = (np.datetime64(day_value) - np.datetime64("2025-01-01")).astype(int) * 96
        for interval in range(96):
            source_interval = int(start + interval)
            active = background_p[:, source_interval].copy()
            reactive = background_q[:, source_interval].copy()
            for bem_id in bem_ids:
                bus = bem_buses[bem_id]
                active[bus - 2] += bem[bem_id][source_interval] - pv[bem_id][source_interval]
                reactive[bus - 2] += bem[bem_id][source_interval] * q_to_p[bus]
            voltage, loading, transformer = run_power_flow(
                net,
                load_index_by_bus,
                active,
                reactive,
                line_indices,
                line_limits,
                transformer_kva,
                init="auto" if day_index == 0 and interval == 0 else "results",
            )
            baseline_voltage[day_index, interval] = voltage
            baseline_line[day_index, interval] = loading
            baseline_transformer[day_index, interval] = transformer
            for bem_index, bem_id in enumerate(bem_ids):
                bus = bem_buses[bem_id]
                changed_p = active.copy()
                changed_q = reactive.copy()
                changed_p[bus - 2] -= perturbation_kw
                changed_q[bus - 2] -= perturbation_kw * q_to_p[bus]
                perturbed_voltage, perturbed_line, perturbed_transformer = run_power_flow(
                    net,
                    load_index_by_bus,
                    changed_p,
                    changed_q,
                    line_indices,
                    line_limits,
                    transformer_kva,
                    init="results",
                )
                voltage_sensitivity[day_index, interval, :, bem_index] = (
                    perturbed_voltage - voltage
                ) / perturbation_kw
                line_sensitivity[day_index, interval, :, bem_index] = (
                    perturbed_line - loading
                ) / perturbation_kw
                transformer_sensitivity[day_index, interval, bem_index] = (
                    perturbed_transformer - transformer
                ) / perturbation_kw
        print(f"prepared sensitivities {day_names[day_index]}={day_value}")

    output_dir = LOCAL_CASE / "models" / "grid_sensitivities"
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {
        "baseline_bus_voltage_pu": baseline_voltage,
        "baseline_line_loading_pct": baseline_line,
        "baseline_transformer_loading_pct": baseline_transformer,
        "bus_voltage_pu_per_kw_reduction": voltage_sensitivity,
        "line_loading_pct_per_kw_reduction": line_sensitivity,
        "transformer_loading_pct_per_kw_reduction": transformer_sensitivity,
    }
    array_records: dict[str, dict[str, object]] = {}
    for name, values in arrays.items():
        path = output_dir / f"{name}.npy"
        np.save(path, values, allow_pickle=False)
        array_records[name] = {
            "path": str(path.relative_to(DATA_ROOT)).replace("\\", "/"),
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "sha256": sha256_file(path),
        }
    manifest = {
        "schema_version": "1.0",
        "case_id": CASE_ID,
        "pandapower_version": pp.__version__,
        "day_names": day_names,
        "dates": dates,
        "bem_ids": bem_ids,
        "bem_buses": [bem_buses[bem_id] for bem_id in bem_ids],
        "paper_bus_order": list(range(1, 34)),
        "line_index_order": line_indices,
        "finite_difference_reduction_kw": perturbation_kw,
        "arrays": array_records,
        "inputs": {
            name: {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/")
                if path.is_relative_to(REPO_ROOT)
                else str(path.relative_to(DATA_ROOT)).replace("\\", "/"),
                "sha256": sha256_file(path),
            }
            for name, path in inputs.items()
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path} arrays={len(arrays)}")


if __name__ == "__main__":
    main()
