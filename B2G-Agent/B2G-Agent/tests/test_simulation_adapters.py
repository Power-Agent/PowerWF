from __future__ import annotations

import csv
from pathlib import Path

import pandapower as pp

from b2g_agent.simulation.adapters.energyplus import (
    _stable_version,
    _warning_count,
    read_15_minute_day,
)
from b2g_agent.simulation.adapters.pandapower import FormalPandapowerAdapter, FormalPandapowerConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "cases" / "ieee33_10bem_15min"


def test_energyplus_timestep_joules_are_converted_to_kw(tmp_path: Path) -> None:
    path = tmp_path / "eplusout.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Date/Time", "Electricity:Facility [J](TimeStep)", "ZONE:Zone Mean Air Temperature [C](TimeStep)"])
        for index in range(96):
            hour = index // 4 + 1
            minute = (index % 4 + 1) * 15
            writer.writerow([f" 07/15  {hour:02d}:{minute:02d}:00", 900_000, 24.0])
    power, temperature = read_15_minute_day(path, date_mm_dd="07/15")
    assert power == [1.0] * 96
    assert temperature == [24.0] * 96


def test_energyplus_final_warning_count_includes_suppressed_recurring_warnings(
    tmp_path: Path,
) -> None:
    error_file = tmp_path / "eplusout.err"
    error_file.write_text(
        "   ** Warning ** one displayed warning\n"
        "EnergyPlus Completed Successfully-- 492542 Warning; 0 Severe Errors;\n",
        encoding="utf-8",
    )
    assert _warning_count(error_file) == 492542


def test_energyplus_version_drops_dynamic_ymd_suffix() -> None:
    assert (
        _stable_version("EnergyPlus, Version 23.1.0-87ed9199d4, YMD=2026.10.24 21:46")
        == "EnergyPlus, Version 23.1.0-87ed9199d4"
    )


def test_pandapower_adapter_runs_96_multi_node_intervals(tmp_path: Path) -> None:
    network = CASE_DIR / "network" / "source_case33bw.json"
    net = pp.from_json(str(network))
    limits = tmp_path / "line_limits.csv"
    with limits.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["line_index", "max_i_ka"])
        for line_index in net.line[net.line.in_service].index:
            writer.writerow([line_index, 1.0])
    adapter = FormalPandapowerAdapter(FormalPandapowerConfig(
        network_path=network, line_limits_path=limits, upstream_transformer_kva=10_000.0
    ))
    result = adapter.run({3: [100.0] * 96, 5: [150.0] * 96})
    assert len(result.feeder_kw) == 96
    assert all(result.converged)
    assert len(result.bus_voltage_pu[33]) == 96
    assert max(result.maximum_line_loading_pct) < 100.0
