"""Replay one RC--OPF candidate through nonlinear pandapower and gate release."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.contracts import (  # noqa: E402
    EpisodeGroundTruth,
    TaskFamily,
    TaskSpecification,
)
from b2g_agent.observability.reproducibility import (  # noqa: E402
    physics_implementation_bundle_sha256,
    physics_implementation_hashes,
)
from b2g_agent.simulation.adapters import (  # noqa: E402
    FormalPandapowerAdapter,
    FormalPandapowerConfig,
)
from b2g_agent.simulation.joint_rc_opf import interval_index, window_indices  # noqa: E402


CASE_ID = "ieee33_10bem_15min"
CASE_DIR = REPO_ROOT / "cases" / CASE_ID
EXPECTED_INTERVALS = 365 * 96


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_oracle(episode_id: str) -> EpisodeGroundTruth:
    path = CASE_DIR / "groundtruth" / "contract_oracles.jsonl"
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            if payload["episode_id"] == episode_id:
                return EpisodeGroundTruth.model_validate(payload)
    raise KeyError(episode_id)


def load_background_day(path: Path, day_offset: int) -> tuple[dict[int, list[float]], dict[int, list[float]]]:
    active = {bus: [0.0] * 96 for bus in range(2, 34)}
    reactive = {bus: [0.0] * 96 for bus in range(2, 34)}
    counts = {bus: 0 for bus in range(2, 34)}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            bus = int(row["paper_bus"])
            index = counts[bus]
            if day_offset <= index < day_offset + 96:
                local = index - day_offset
                active[bus][local] = float(row["p_kw"])
                reactive[bus][local] = float(row["q_kvar"])
            counts[bus] += 1
    if any(value != EXPECTED_INTERVALS for value in counts.values()):
        raise ValueError("Background profile is incomplete")
    return active, reactive


def load_bem_day(
    path: Path,
    day_offset: int,
    value_columns: tuple[str, ...],
    *,
    known_buses: dict[str, int] | None = None,
) -> tuple[dict[str, dict[str, list[float]]], dict[str, int]]:
    values: dict[str, dict[str, list[float]]] = {}
    buses: dict[str, int] = {}
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            bem_id = row["bem_id"]
            if bem_id not in values:
                values[bem_id] = {column: [0.0] * 96 for column in value_columns}
                buses[bem_id] = (
                    known_buses[bem_id] if known_buses is not None else int(row["paper_bus"])
                )
                counts[bem_id] = 0
            index = counts[bem_id]
            if day_offset <= index < day_offset + 96:
                local = index - day_offset
                for column in value_columns:
                    values[bem_id][column][local] = float(row[column])
            counts[bem_id] += 1
    if len(values) != 10 or any(value != EXPECTED_INTERVALS for value in counts.values()):
        raise ValueError(f"BEM profile is incomplete: {path}")
    return values, buses


def load_scalar_day(path: Path, day_offset: int, column: str) -> list[float]:
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            values.append(float(row[column]))
    if len(values) != EXPECTED_INTERVALS:
        raise ValueError(f"Profile must contain {EXPECTED_INTERVALS} intervals: {path}")
    return values[day_offset : day_offset + 96]


def family_interval_indices(specification: object) -> dict[str, list[int]]:
    """Reproduce the optimizer's frozen positional family/window binding."""
    families = list(specification.task_families)
    windows = list(specification.windows)
    if len(families) == len(windows) and len(families) > 1:
        bindings = list(zip(families, windows, strict=True))
    else:
        bindings = [(family, window) for family in families for window in windows]
    result: dict[str, list[int]] = {}
    for family, window in bindings:
        key = family.value
        result.setdefault(key, [])
        result[key].extend(window_indices(window.start, window.end))
    return {key: sorted(set(values)) for key, values in result.items()}


def violation_area(
    series_by_asset: dict[int, tuple[float, ...]],
    indices: list[int],
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> float:
    area = 0.0
    for values in series_by_asset.values():
        for index in indices:
            value = float(values[index])
            if lower is not None:
                area += max(0.0, lower - value) * 0.25
            if upper is not None:
                area += max(0.0, value - upper) * 0.25
    return area


def build_business_metrics(
    *,
    specification: object,
    requested_signed: list[float],
    actual_service: list[float],
    remaining_gap: list[float],
    baseline_grid: object,
    response_grid: object,
    baseline: dict[str, dict[str, list[float]]],
    prices: list[float],
    config: dict[str, object],
    validated_committed_power: float,
    validated_delivery_ratio: float,
) -> dict[str, object]:
    """Calculate frozen, task-facing metrics from baseline and response replays."""
    family_indices = family_interval_indices(specification)
    active = [index for index, value in enumerate(requested_signed) if abs(value) > 1e-9]
    peak_indices = family_indices.get(TaskFamily.PEAK_SHAVING.value, active)
    congestion_indices = family_indices.get(
        TaskFamily.CONGESTION_MANAGEMENT.value, active
    )
    voltage_indices = family_indices.get(TaskFamily.VOLTAGE_MANAGEMENT.value, active)
    positive = [index for index in active if requested_signed[index] > 0]
    negative = [index for index in active if requested_signed[index] < 0]
    directed = {
        index: (1.0 if requested_signed[index] > 0 else -1.0) * actual_service[index]
        for index in active
    }
    event_average = (
        sum(directed.values()) / len(directed) if directed else 0.0
    )
    delivered_energy = sum(max(0.0, value) for value in directed.values()) * 0.25
    last_active = max(active) if active else 95
    rebound_indices = list(range(last_active + 1, 96))
    rebound = (
        max(
            0.0,
            max(
                float(response_grid.feeder_kw[index])
                - float(baseline_grid.feeder_kw[index])
                for index in rebound_indices
            ),
        )
        if rebound_indices
        else None
    )
    limits = config["linear_grid_constraints"]
    event_line_overloads = sum(
        float(response_grid.maximum_line_loading_pct[index])
        > float(limits["maximum_line_loading_pct"])
        for index in congestion_indices
    )
    event_voltage_violations = sum(
        float(response_grid.minimum_voltage_pu[index])
        < float(limits["minimum_voltage_pu"])
        or float(response_grid.maximum_voltage_pu[index])
        > float(limits["maximum_voltage_pu"])
        for index in voltage_indices
    )
    target_baseline = [
        sum(float(baseline[bem_id]["portfolio_kw"][index]) for bem_id in specification.target_bems)
        for index in range(96)
    ]
    counterfactual_baseline = (
        sum(target_baseline[index] for index in active) / len(active) if active else 0.0
    )
    positive_service = [max(0.0, actual_service[index]) for index in positive]
    negative_service = [max(0.0, -actual_service[index]) for index in negative]
    absorption_delivered = sum(
        min(abs(requested_signed[index]), max(0.0, -actual_service[index]))
        for index in negative
    ) * 0.25
    absorption_requested = sum(abs(requested_signed[index]) for index in negative) * 0.25
    absorption_gap = sum(
        max(0.0, abs(requested_signed[index]) - max(0.0, -actual_service[index]))
        for index in negative
    ) * 0.25
    if absorption_gap < 1e-9:
        absorption_gap = 0.0
    offer_field = specification.fields.get("offer_price")
    offer_price = (
        float(offer_field.value)
        if offer_field is not None and offer_field.value is not None
        else None
    )
    market_indices = family_indices.get(TaskFamily.MARKET_TRADING.value, [])
    market_delivered_energy = sum(
        min(
            abs(requested_signed[index]),
            max(
                0.0,
                (1.0 if requested_signed[index] > 0 else -1.0)
                * actual_service[index],
            ),
        )
        for index in market_indices
    ) * 0.25
    settlement = (
        market_delivered_energy * offer_price / 1000.0
        if offer_price is not None
        else None
    )
    reserve_indices = family_indices.get(TaskFamily.RESERVE_SERVICE.value, [])
    baseline_flexibility_indices = family_indices.get(
        TaskFamily.BASELINE_FLEXIBILITY.value, []
    )
    reserve_full_intervals = sum(
        max(
            0.0,
            abs(requested_signed[index])
            - (1.0 if requested_signed[index] > 0 else -1.0)
            * actual_service[index],
        )
        <= float(config["validation_tolerances"]["service_kw"])
        for index in reserve_indices
    )
    reserve_tracking_mae = (
        sum(
            abs(
                abs(requested_signed[index])
                - max(
                    0.0,
                    (1.0 if requested_signed[index] > 0 else -1.0)
                    * actual_service[index],
                )
            )
            for index in reserve_indices
        )
        / len(reserve_indices)
        if reserve_indices
        else None
    )
    baseline_peak = (
        max(float(baseline_grid.feeder_kw[index]) for index in peak_indices)
        if peak_indices
        else 0.0
    )
    response_peak = (
        max(float(response_grid.feeder_kw[index]) for index in peak_indices)
        if peak_indices
        else 0.0
    )
    return {
        "interval_minutes": 15,
        "active_interval_indices": active,
        "family_interval_indices": family_indices,
        "delivered_reduction_kw": validated_committed_power,
        "event_average_directed_delivery_kw": event_average,
        "delivered_energy_kwh": delivered_energy,
        "baseline_feeder_peak_kw": baseline_peak,
        "feeder_peak_kw": response_peak,
        "feeder_peak_reduction_kw": baseline_peak - response_peak,
        "rebound_kw": rebound,
        "baseline_minimum_voltage_pu": min(
            float(baseline_grid.minimum_voltage_pu[index]) for index in voltage_indices
        ) if voltage_indices else None,
        "baseline_maximum_voltage_pu": max(
            float(baseline_grid.maximum_voltage_pu[index]) for index in voltage_indices
        ) if voltage_indices else None,
        "minimum_voltage_pu": min(
            float(response_grid.minimum_voltage_pu[index]) for index in voltage_indices
        ) if voltage_indices else None,
        "maximum_voltage_pu": max(
            float(response_grid.maximum_voltage_pu[index]) for index in voltage_indices
        ) if voltage_indices else None,
        "voltage_violation_intervals": event_voltage_violations,
        "baseline_voltage_violation_area_pu_h": violation_area(
            baseline_grid.bus_voltage_pu,
            voltage_indices,
            lower=float(limits["minimum_voltage_pu"]),
            upper=float(limits["maximum_voltage_pu"]),
        ),
        "voltage_violation_area_pu_h": violation_area(
            response_grid.bus_voltage_pu,
            voltage_indices,
            lower=float(limits["minimum_voltage_pu"]),
            upper=float(limits["maximum_voltage_pu"]),
        ),
        "baseline_maximum_line_loading_pct": max(
            float(baseline_grid.maximum_line_loading_pct[index])
            for index in congestion_indices
        ) if congestion_indices else None,
        "maximum_line_loading_pct": max(
            float(response_grid.maximum_line_loading_pct[index])
            for index in congestion_indices
        ) if congestion_indices else None,
        "overload_intervals": event_line_overloads,
        "baseline_congestion_violation_area_pct_h": violation_area(
            baseline_grid.line_loading_pct,
            congestion_indices,
            upper=float(limits["maximum_line_loading_pct"]),
        ),
        "congestion_violation_area_pct_h": violation_area(
            response_grid.line_loading_pct,
            congestion_indices,
            upper=float(limits["maximum_line_loading_pct"]),
        ),
        "reserved_capacity_kw": (
            validated_committed_power if reserve_indices else None
        ),
        "reserve_requested_duration_h": (
            len(reserve_indices) * 0.25 if reserve_indices else None
        ),
        "reserve_delivered_duration_h": (
            reserve_full_intervals * 0.25 if reserve_indices else None
        ),
        "reserve_tracking_mae_kw": reserve_tracking_mae,
        "delivery_ratio": validated_delivery_ratio,
        "absorbed_energy_kwh": absorption_delivered if negative else None,
        "counterfactual_pv_curtailment_kwh": (
            absorption_requested if negative else None
        ),
        "pv_curtailment_kwh": absorption_gap if negative else None,
        "renewable_metric_interpretation": config["business_metrics"][
            "renewable_absorption_rule"
        ],
        "counterfactual_baseline_kw": (
            counterfactual_baseline if baseline_flexibility_indices else None
        ),
        "flexibility_lower_kw": (
            counterfactual_baseline - min(positive_service)
            if baseline_flexibility_indices and positive_service
            else None
        ),
        "flexibility_upper_kw": (
            counterfactual_baseline + min(negative_service)
            if baseline_flexibility_indices and negative_service
            else None
        ),
        "flexibility_envelope_status": (
            "one_sided_response_validation_only"
            if baseline_flexibility_indices
            else "not_applicable"
        ),
        "offer_price_usd_per_mwh": offer_price,
        "mean_rt_lmp_usd_per_mwh": (
            sum(prices[index] for index in market_indices) / len(market_indices)
            if market_indices
            else None
        ),
        "accepted_capacity_kw": (
            validated_committed_power if market_indices else None
        ),
        "market_delivered_energy_kwh": (
            market_delivered_energy if market_indices else None
        ),
        "settlement_usd": settlement,
        "market_settlement_rule": config["business_metrics"]["market_settlement_rule"],
        "maximum_remaining_service_gap_kw": (
            0.0
            if max(remaining_gap, default=0.0) < 1e-9
            else max(remaining_gap, default=0.0)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimizer-result", type=Path, required=True)
    parser.add_argument("--energyplus-validation", type=Path, default=None)
    parser.add_argument(
        "--task-specification",
        type=Path,
        default=None,
        help="Optional translated TaskSpecification JSON. When supplied, do not read benchmark ground truth.",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def linearization_error(actual: float, predicted: object) -> float | None:
    """Compare nonlinear and linear results when a prediction exists.

    Solver-failed or deliberately infeasible candidates have no linear
    prediction, but still require nonlinear baseline/zero-response evidence so
    they can become traceable rejections instead of crashing validation.
    """

    if predicted is None:
        return None
    return float(actual) - float(predicted)


def receiver_outcome(
    *,
    release_eligible: bool,
    service_request_satisfied: bool,
    total_delivered: float,
    service_tolerance: float,
    delivery_ratio: float,
) -> tuple[str, bool, float]:
    """Apply physical hard gates before assigning the receiver's outcome."""
    if not release_eligible:
        return "rejected", False, 0.0
    if service_request_satisfied:
        return "accepted", True, 1.0
    if total_delivered > service_tolerance:
        return "modified", False, delivery_ratio
    return "rejected", False, 0.0


def main() -> None:
    args = parse_args()
    optimizer_path = args.optimizer_result.resolve()
    optimizer = json.loads(optimizer_path.read_text(encoding="utf-8"))
    event_validation_path = (
        args.energyplus_validation.resolve() if args.energyplus_validation else None
    )
    event_validation = (
        json.loads(event_validation_path.read_text(encoding="utf-8"))
        if event_validation_path
        else None
    )
    if event_validation and event_validation["episode_id"] != optimizer["episode_id"]:
        raise ValueError("Optimizer and EnergyPlus validation episode IDs differ")
    task_specification_path = (
        args.task_specification.resolve() if args.task_specification else None
    )
    if task_specification_path is None:
        specification = load_oracle(optimizer["episode_id"]).task_specification
    else:
        specification = TaskSpecification.model_validate(
            json.loads(task_specification_path.read_text(encoding="utf-8"))
        )
        if specification.episode_id != optimizer["episode_id"]:
            raise ValueError("Optimizer and task specification episode IDs differ")
        if specification.clarification_required:
            raise ValueError("A task requiring clarification cannot enter validation")
    data_root = Path(os.getenv("B2G_DATA_ROOT", REPO_ROOT / ".local-data")).resolve()
    local_case = data_root / CASE_ID
    profiles = local_case / "profiles"
    target_days_path = CASE_DIR / "days" / "target_days.json"
    target_days = json.loads(target_days_path.read_text(encoding="utf-8"))["target_days"]
    day_value = target_days[specification.operating_day]
    day_offset = (date.fromisoformat(day_value) - date(2025, 1, 1)).days * 96
    background_path = profiles / "background_load_15min.csv"
    baseline_path = profiles / "bem_baseline_15min.csv"
    thermal_path = profiles / "bem_thermal_15min.csv"
    pv_path = profiles / "pv_15min.csv"
    prices_path = profiles / "prices_15min.csv"
    config_path = CASE_DIR / "optimization" / "joint_dispatch.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    background_p, background_q = load_background_day(background_path, day_offset)
    baseline, bem_buses = load_bem_day(baseline_path, day_offset, ("portfolio_kw",))
    thermal, thermal_buses = load_bem_day(
        thermal_path,
        day_offset,
        ("minimum_zone_c", "mean_zone_c", "maximum_zone_c"),
        known_buses=bem_buses,
    )
    pv, pv_buses = load_bem_day(pv_path, day_offset, ("pv_kw",))
    prices = load_scalar_day(prices_path, day_offset, "rt_lmp_usd_per_mwh")
    if bem_buses != pv_buses or bem_buses != thermal_buses:
        raise ValueError("BEM profile mappings differ")

    import pandapower as pp

    network_path = CASE_DIR / "network" / "source_case33bw.json"
    source_net = pp.from_json(str(network_path))
    q_to_p = {
        int(row.bus) + 1: float(row.q_mvar) / float(row.p_mw)
        for row in source_net.load.itertuples()
    }
    baseline_node_p = {bus: list(values) for bus, values in background_p.items()}
    baseline_node_q = {bus: list(values) for bus, values in background_q.items()}
    response_node_p = {bus: list(values) for bus, values in background_p.items()}
    response_node_q = {bus: list(values) for bus, values in background_q.items()}
    schedules = optimizer["schedules"]
    fixed_building_response_bems = set(
        optimizer.get("fixed_building_response_bems", [])
    )
    maximum_building_adjustment = 0.0
    actual_aggregate_service = [0.0] * 96
    for bem_id, bus in bem_buses.items():
        schedule = schedules.get(bem_id)
        use_energyplus_actual = bool(
            event_validation
            and bem_id in event_validation["actual_building_reduction_kw_by_bem"]
            and (
                not optimizer.get("fixed_building_response_bems")
                or bem_id in fixed_building_response_bems
            )
        )
        if use_energyplus_actual:
            building_reduction = event_validation["actual_building_reduction_kw_by_bem"][bem_id]
        else:
            building_reduction = schedule["building_reduction_kw"] if schedule else [0.0] * 96
        discharge = schedule["storage_discharge_kw"] if schedule else [0.0] * 96
        charge = schedule["storage_charge_kw"] if schedule else [0.0] * 96
        maximum_building_adjustment = max(
            maximum_building_adjustment,
            max(abs(float(value)) for value in building_reduction),
        )
        for interval in range(96):
            baseline_building = baseline[bem_id]["portfolio_kw"][interval]
            baseline_node_p[bus][interval] += (
                baseline_building - pv[bem_id]["pv_kw"][interval]
            )
            baseline_node_q[bus][interval] += baseline_building * q_to_p[bus]
            actual_aggregate_service[interval] += (
                float(building_reduction[interval])
                + float(discharge[interval])
                - float(charge[interval])
            )
            adjusted_building = baseline[bem_id]["portfolio_kw"][interval] - float(
                building_reduction[interval]
            )
            response_node_p[bus][interval] += (
                adjusted_building
                - pv[bem_id]["pv_kw"][interval]
                - float(discharge[interval])
                + float(charge[interval])
            )
            response_node_q[bus][interval] += adjusted_building * q_to_p[bus]

    transformer = json.loads(
        (CASE_DIR / "network" / "transformer.json").read_text(encoding="utf-8")
    )
    adapter = FormalPandapowerAdapter(
        FormalPandapowerConfig(
            network_path=network_path,
            line_limits_path=CASE_DIR / "network" / "line_limits.csv",
            upstream_transformer_kva=float(transformer["rating_kva"]),
        )
    )
    output_path = args.output or optimizer_path.with_name("nonlinear_validation.json")
    if output_path.exists():
        raise FileExistsError(output_path)
    powerflow_path = output_path.with_name("nonlinear_pandapower.json")
    baseline_powerflow_path = output_path.with_name("baseline_pandapower.json")
    baseline_grid = adapter.run(
        baseline_node_p,
        node_reactive_kvar=baseline_node_q,
        output_path=baseline_powerflow_path,
    )
    grid = adapter.run(
        response_node_p,
        node_reactive_kvar=response_node_q,
        output_path=powerflow_path,
    )
    grid_limits = config["linear_grid_constraints"]
    grid_violations = {
        "nonconverged_intervals": sum(not value for value in grid.converged),
        "low_voltage_intervals": sum(
            value < float(grid_limits["minimum_voltage_pu"])
            for value in grid.minimum_voltage_pu
        ),
        "high_voltage_intervals": sum(
            value > float(grid_limits["maximum_voltage_pu"])
            for value in grid.maximum_voltage_pu
        ),
        "line_overload_intervals": sum(
            value > float(grid_limits["maximum_line_loading_pct"])
            for value in grid.maximum_line_loading_pct
        ),
        "transformer_overload_intervals": sum(
            value > float(grid_limits["maximum_transformer_loading_pct"])
            for value in grid.transformer_loading_pct
        ),
    }
    grid_pass = not any(grid_violations.values())

    comfort = config["comfort"]
    occupied_start = interval_index(comfort["occupied_start"])
    occupied_end = interval_index(comfort["occupied_end"])
    weekend = date.fromisoformat(day_value).weekday() >= 5
    comfort_violations = 0
    baseline_absolute_comfort_violations = 0
    if event_validation:
        if fixed_building_response_bems:
            relevant_records = [
                record
                for record in event_validation["records"]
                if record["bem_id"] in fixed_building_response_bems
            ]
            comfort_violations = sum(
                int(record["comfort_violation_intervals"])
                for record in relevant_records
            )
            energyplus_severe_errors = sum(
                int(record["event_severe_error_count"])
                for record in relevant_records
            )
        else:
            comfort_violations = int(
                event_validation["response_comfort_violation_intervals"]
            )
            energyplus_severe_errors = sum(
                int(record["event_severe_error_count"])
                for record in event_validation["records"]
            )
        energyplus_status = (
            "event_validation_pass"
            if comfort_violations == 0 and energyplus_severe_errors == 0
            else "event_validation_failed"
        )
    elif maximum_building_adjustment <= 1e-8:
        for bem_id in specification.target_bems:
            for interval in range(96):
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
                if (
                    thermal[bem_id]["minimum_zone_c"][interval] < lower
                    or thermal[bem_id]["maximum_zone_c"][interval] > upper
                ):
                    baseline_absolute_comfort_violations += 1
        energyplus_status = "baseline_replay_pass"
    else:
        energyplus_status = "pending_event_schedule"

    energyplus_pass = energyplus_status in {"baseline_replay_pass", "event_validation_pass"}
    requested_signed = [float(value) for value in optimizer["requested_service_kw"]]
    active_intervals = [index for index, value in enumerate(requested_signed) if abs(value) > 1e-9]
    requested_magnitude = [abs(requested_signed[index]) for index in active_intervals]
    directed_delivery = [
        (1.0 if requested_signed[index] > 0 else -1.0) * actual_aggregate_service[index]
        for index in active_intervals
    ]
    remaining_gap = [
        max(0.0, requested - max(0.0, delivered))
        for requested, delivered in zip(requested_magnitude, directed_delivery, strict=True)
    ]
    total_requested = sum(requested_magnitude)
    total_delivered = sum(
        min(requested, max(0.0, delivered))
        for requested, delivered in zip(requested_magnitude, directed_delivery, strict=True)
    )
    validated_delivery_ratio = total_delivered / total_requested if total_requested else 0.0
    candidate_committed_power = (
        max(0.0, min(directed_delivery)) if directed_delivery else 0.0
    )
    service_tolerance = float(config["validation_tolerances"]["service_kw"])
    service_request_satisfied = bool(
        active_intervals and max(remaining_gap, default=0.0) <= service_tolerance
    )
    release_eligible = bool(grid_pass and energyplus_pass)
    receiver_decision, full_request_satisfied, validated_delivery_ratio = (
        receiver_outcome(
            release_eligible=release_eligible,
            service_request_satisfied=service_request_satisfied,
            total_delivered=total_delivered,
            service_tolerance=service_tolerance,
            delivery_ratio=validated_delivery_ratio,
        )
    )
    validated_committed_power = candidate_committed_power if release_eligible else 0.0
    maximum_remaining_service_gap = max(remaining_gap, default=0.0)
    if maximum_remaining_service_gap < 1e-9:
        maximum_remaining_service_gap = 0.0
    business_metrics = build_business_metrics(
        specification=specification,
        requested_signed=requested_signed,
        actual_service=actual_aggregate_service,
        remaining_gap=remaining_gap,
        baseline_grid=baseline_grid,
        response_grid=grid,
        baseline=baseline,
        prices=prices,
        config=config,
        validated_committed_power=validated_committed_power,
        validated_delivery_ratio=validated_delivery_ratio,
    )
    business_metrics["comfort_violation_minutes"] = (
        comfort_violations * 15 if energyplus_pass else None
    )
    requested_output_values: dict[str, object] = {}
    for field_name in specification.requested_outputs:
        metric_name = (
            "voltage_violation_intervals"
            if field_name == "violation_intervals"
            else field_name
        )
        if metric_name not in business_metrics:
            raise KeyError(f"No business metric implementation for {field_name}")
        requested_output_values[field_name] = business_metrics[metric_name]
    business_metrics["requested_output_values"] = requested_output_values
    implementation_hashes = physics_implementation_hashes(REPO_ROOT)
    payload = {
        "schema_version": "1.0",
        "episode_id": specification.episode_id,
        "operating_day": specification.operating_day,
        "date": day_value,
        "optimizer_status": optimizer["status"],
        "nonlinear_grid_status": "pass" if grid_pass else "failed",
        "energyplus_status": energyplus_status,
        "release_eligible": release_eligible,
        "receiver_decision": receiver_decision,
        "full_request_satisfied": full_request_satisfied,
        "candidate_committed_power_kw": candidate_committed_power,
        "validated_committed_power_kw": validated_committed_power,
        "validated_delivery_ratio": validated_delivery_ratio,
        "validated_completion_percentage": validated_delivery_ratio * 100.0,
        "maximum_remaining_service_gap_kw": maximum_remaining_service_gap,
        "actual_aggregate_service_kw": actual_aggregate_service,
        "maximum_building_adjustment_kw": maximum_building_adjustment,
        "comfort_violation_intervals": comfort_violations if energyplus_pass or event_validation else None,
        "baseline_absolute_comfort_violation_intervals": (
            int(event_validation["baseline_absolute_comfort_violation_intervals"])
            if event_validation
            else baseline_absolute_comfort_violations
        ),
        "grid_violations": grid_violations,
        "business_metrics": business_metrics,
        "baseline_nonlinear_metrics": {
            "minimum_voltage_pu": min(baseline_grid.minimum_voltage_pu),
            "maximum_voltage_pu": max(baseline_grid.maximum_voltage_pu),
            "maximum_line_loading_pct": max(baseline_grid.maximum_line_loading_pct),
            "maximum_transformer_loading_pct": max(
                baseline_grid.transformer_loading_pct
            ),
        },
        "nonlinear_metrics": {
            "minimum_voltage_pu": min(grid.minimum_voltage_pu),
            "maximum_voltage_pu": max(grid.maximum_voltage_pu),
            "maximum_line_loading_pct": max(grid.maximum_line_loading_pct),
            "maximum_transformer_loading_pct": max(grid.transformer_loading_pct),
        },
        "linearization_error": {
            "minimum_voltage_pu": linearization_error(
                min(grid.minimum_voltage_pu),
                optimizer["predicted_minimum_voltage_pu"],
            ),
            "maximum_line_loading_pct": linearization_error(
                max(grid.maximum_line_loading_pct),
                optimizer["predicted_maximum_line_loading_pct"],
            ),
            "maximum_transformer_loading_pct": linearization_error(
                max(grid.transformer_loading_pct),
                optimizer["predicted_maximum_transformer_loading_pct"],
            ),
        },
        "input_hashes": {
            "optimizer_result": sha256_file(optimizer_path),
            "background": sha256_file(background_path),
            "bem_baseline": sha256_file(baseline_path),
            "bem_thermal": sha256_file(thermal_path),
            "pv": sha256_file(pv_path),
            "prices": sha256_file(prices_path),
            "target_days": sha256_file(target_days_path),
            "joint_dispatch_config": sha256_file(config_path),
            "physics_implementation_bundle": physics_implementation_bundle_sha256(
                REPO_ROOT
            ),
            **(
                {"task_specification": sha256_file(task_specification_path)}
                if task_specification_path
                else {}
            ),
            **(
                {"energyplus_event_validation": sha256_file(event_validation_path)}
                if event_validation_path
                else {}
            ),
        },
        "implementation_hashes": implementation_hashes,
        "artifacts": {
            "pandapower": str(powerflow_path),
            "baseline_pandapower": str(baseline_powerflow_path),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
