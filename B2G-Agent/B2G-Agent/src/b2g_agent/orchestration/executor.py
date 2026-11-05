from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from b2g_agent.simulation.adapters import FormalEnergyPlusAdapter, FormalPandapowerAdapter


@dataclass(frozen=True)
class CoupledRunInputs:
    episode_id: str
    date_mm_dd: str
    bem_to_bus: dict[str, int]
    baseline_idfs: dict[str, Path]
    event_idfs: dict[str, Path]
    aggregation_factors: dict[str, float]
    background_node_kw: dict[int, list[float]]
    pv_kw_by_bem: dict[str, list[float]]
    storage_dispatch_kw_by_bem: dict[str, list[float]]
    maximum_zone_temperature_c: float


class PhysicalExecutor:
    """Executes the deterministic EnergyPlus → mapping → pandapower boundary.

    Event IDFs are explicit inputs. Generating device schedules belongs to the
    optimizer/action layer and is never improvised inside this executor.
    """

    def __init__(self, energyplus: FormalEnergyPlusAdapter, pandapower: FormalPandapowerAdapter) -> None:
        self.energyplus = energyplus
        self.pandapower = pandapower

    def run(self, inputs: CoupledRunInputs, *, run_root: Path) -> dict[str, Any]:
        _validate_inputs(inputs)
        if run_root.exists():
            raise FileExistsError(f"Run root already exists: {run_root}")
        run_root.mkdir(parents=True)
        baseline_node = {bus: list(values) for bus, values in inputs.background_node_kw.items()}
        event_node = {bus: list(values) for bus, values in inputs.background_node_kw.items()}
        comfort_violations: dict[str, int] = {}
        building_hashes: dict[str, dict[str, str]] = {}
        building_diagnostics: dict[str, dict[str, int]] = {}

        for bem_id, paper_bus in inputs.bem_to_bus.items():
            baseline = self.energyplus.run(
                bem_id=bem_id,
                idf_path=inputs.baseline_idfs[bem_id],
                date_mm_dd=inputs.date_mm_dd,
                run_dir=run_root / "energyplus" / bem_id / "baseline",
            )
            event = self.energyplus.run(
                bem_id=bem_id,
                idf_path=inputs.event_idfs[bem_id],
                date_mm_dd=inputs.date_mm_dd,
                run_dir=run_root / "energyplus" / bem_id / "event",
            )
            factor = inputs.aggregation_factors[bem_id]
            baseline_node.setdefault(paper_bus, [0.0] * 96)
            event_node.setdefault(paper_bus, [0.0] * 96)
            for interval in range(96):
                pv = inputs.pv_kw_by_bem[bem_id][interval]
                storage = inputs.storage_dispatch_kw_by_bem[bem_id][interval]
                baseline_node[paper_bus][interval] += baseline.electricity_kw[interval] * factor - pv
                event_node[paper_bus][interval] += event.electricity_kw[interval] * factor - pv - storage
            temperatures = event.maximum_zone_temperature_c or ()
            comfort_violations[bem_id] = sum(
                temperature > inputs.maximum_zone_temperature_c for temperature in temperatures
            )
            building_hashes[bem_id] = {
                **{f"baseline_{key}": value for key, value in baseline.input_hashes.items()},
                **{f"event_{key}": value for key, value in event.input_hashes.items()},
            }
            building_diagnostics[bem_id] = {
                "baseline_warning_count": baseline.warning_count,
                "baseline_severe_error_count": baseline.severe_error_count,
                "event_warning_count": event.warning_count,
                "event_severe_error_count": event.severe_error_count,
            }

        baseline_grid = self.pandapower.run(
            baseline_node,
            output_path=run_root / "pandapower" / "baseline.json",
        )
        event_grid = self.pandapower.run(
            event_node,
            output_path=run_root / "pandapower" / "event.json",
        )
        result = {
            "schema_version": "1.0",
            "episode_id": inputs.episode_id,
            "date_mm_dd": inputs.date_mm_dd,
            "intervals": 96,
            "delivered_reduction_kw": [
                round(baseline_grid.feeder_kw[index] - event_grid.feeder_kw[index], 6)
                for index in range(96)
            ],
            "minimum_voltage_pu": min(event_grid.minimum_voltage_pu),
            "maximum_voltage_pu": max(event_grid.maximum_voltage_pu),
            "maximum_line_loading_pct": max(event_grid.maximum_line_loading_pct),
            "maximum_transformer_loading_pct": max(event_grid.transformer_loading_pct),
            "comfort_violation_intervals_by_bem": comfort_violations,
            "energyplus_diagnostics_by_bem": building_diagnostics,
            "converged": all(event_grid.converged),
            "tool_versions": {
                "energyplus": self.energyplus.version(),
                "pandapower": event_grid.version,
            },
            "input_hashes": {
                "buildings": building_hashes,
                "network": event_grid.input_hashes,
            },
        }
        result_path = run_root / "coupled_result.json"
        result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        result["output_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
        return result


def _validate_inputs(inputs: CoupledRunInputs) -> None:
    bem_ids = set(inputs.bem_to_bus)
    for name, mapping in {
        "baseline_idfs": inputs.baseline_idfs,
        "event_idfs": inputs.event_idfs,
        "aggregation_factors": inputs.aggregation_factors,
        "pv_kw_by_bem": inputs.pv_kw_by_bem,
        "storage_dispatch_kw_by_bem": inputs.storage_dispatch_kw_by_bem,
    }.items():
        if set(mapping) != bem_ids:
            raise ValueError(f"{name} must contain exactly the BEM mapping keys")
    for bem_id, factor in inputs.aggregation_factors.items():
        if factor <= 0:
            raise ValueError(f"Aggregation factor must be positive for {bem_id}")
    for name, profiles in {
        "background_node_kw": inputs.background_node_kw,
        "pv_kw_by_bem": inputs.pv_kw_by_bem,
        "storage_dispatch_kw_by_bem": inputs.storage_dispatch_kw_by_bem,
    }.items():
        for key, values in profiles.items():
            if len(values) != 96:
                raise ValueError(f"{name}[{key}] has {len(values)} intervals; expected 96")
