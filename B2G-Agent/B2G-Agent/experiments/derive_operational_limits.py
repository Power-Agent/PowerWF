"""Derive experiment-specific network limits from development-only profiles."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandapower as pp


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "cases" / "ieee33_10bem_15min"
DATA_ROOT = Path(os.getenv("B2G_DATA_ROOT", REPO_ROOT / ".local-data")).resolve()
PROFILE = DATA_ROOT / "ieee33_10bem_15min" / "profiles" / "background_load_15min.csv"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_development_profiles() -> dict[str, dict[int, tuple[float, float]]]:
    split = json.loads((CASE_DIR / "days" / "calendar_split.json").read_text(encoding="utf-8"))
    development_dates = set(split["development_dates"])
    profiles: dict[str, dict[int, tuple[float, float]]] = defaultdict(dict)
    with PROFILE.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            timestamp = row["timestamp"]
            day = timestamp[:10]
            if day not in development_dates:
                continue
            profiles[timestamp][int(row["paper_bus"])] = (float(row["p_kw"]), float(row["q_kvar"]))
    expected = len(development_dates) * 96
    if len(profiles) != expected:
        raise ValueError(f"Expected {expected} development timestamps, found {len(profiles)}")
    return profiles


def main() -> None:
    if not PROFILE.is_file():
        raise FileNotFoundError(f"Background profile is missing: {PROFILE}")
    net = pp.from_json(str(CASE_DIR / "network" / "source_case33bw.json"))
    load_by_bus = {int(row.bus) + 1: int(index) for index, row in net.load.iterrows()}
    profiles = load_development_profiles()
    line_currents: dict[int, list[float]] = {int(index): [] for index in net.line[net.line.in_service].index}
    slack_apparent_kva: list[float] = []
    for interval_index, timestamp in enumerate(sorted(profiles)):
        values = profiles[timestamp]
        for paper_bus, (p_kw, q_kvar) in values.items():
            if paper_bus in load_by_bus:
                load_index = load_by_bus[paper_bus]
                net.load.at[load_index, "p_mw"] = p_kw / 1000.0
                net.load.at[load_index, "q_mvar"] = q_kvar / 1000.0
        run_options = {
            "algorithm": "nr",
            "calculate_voltage_angles": False,
            "init": "auto" if interval_index == 0 else "results",
            "tolerance_mva": 1e-8,
            "numba": False,
            "check_connectivity": interval_index == 0,
        }
        if interval_index:
            run_options["recycle"] = {"bus_pq": True, "trafo": False, "gen": False}
        pp.runpp(net, **run_options)
        if not net.converged:
            raise RuntimeError(f"Development power flow did not converge: {timestamp}")
        for line_index in line_currents:
            line_currents[line_index].append(float(net.res_line.at[line_index, "i_ka"]))
        p_kw = float(net.res_ext_grid.p_mw.sum()) * 1000.0
        q_kvar = float(net.res_ext_grid.q_mvar.sum()) * 1000.0
        slack_apparent_kva.append(math.sqrt(p_kw * p_kw + q_kvar * q_kvar))

    limits_path = CASE_DIR / "network" / "line_limits.csv"
    with limits_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["line_index", "max_i_ka", "development_p99_i_ka", "multiplier", "minimum_floor_ka"])
        for line_index, values in sorted(line_currents.items()):
            p99 = float(np.quantile(values, 0.99))
            limit = max(0.05, 1.20 * p99)
            writer.writerow([line_index, f"{limit:.9f}", f"{p99:.9f}", "1.20", "0.05"])
    p99_kva = float(np.quantile(slack_apparent_kva, 0.99))
    transformer = {
        "label": "experiment-specific upstream transformer",
        "development_p99_kva": p99_kva,
        "rating_kva": 1.20 * p99_kva,
        "multiplier": 1.20,
        "development_interval_count": len(slack_apparent_kva),
        "source_profile": str(PROFILE.relative_to(DATA_ROOT)).replace("\\", "/"),
        "source_profile_sha256": sha256_file(PROFILE),
        "derivation_script": "experiments/derive_operational_limits.py",
    }
    (CASE_DIR / "network" / "transformer.json").write_text(
        json.dumps(transformer, indent=2) + "\n", encoding="utf-8"
    )
    print(f"derived_lines={len(line_currents)} development_intervals={len(slack_apparent_kva)}")


if __name__ == "__main__":
    main()
