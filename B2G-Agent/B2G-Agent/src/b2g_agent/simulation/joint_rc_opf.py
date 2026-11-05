from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
from scipy import __version__ as scipy_version
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from b2g_agent.contracts import TaskFamily, TaskSpecification


INTERVALS = 96
DT_HOURS = 0.25


@dataclass(frozen=True)
class JointDispatchResult:
    episode_id: str
    status: str
    solver_status: int
    solver_message: str
    objective_value: float | None
    requested_power_kw: float
    committed_power_kw: float
    delivery_ratio: float
    aggregate_service_kw: tuple[float, ...]
    requested_service_kw: tuple[float, ...]
    unserved_request_kw: tuple[float, ...]
    schedules: dict[str, dict[str, tuple[float, ...]]]
    predicted_minimum_voltage_pu: float | None
    predicted_maximum_voltage_pu: float | None
    predicted_maximum_line_loading_pct: float | None
    predicted_maximum_transformer_loading_pct: float | None
    predicted_temperature_range_c_by_bem: dict[str, tuple[float, float]]
    fixed_building_response_bems: tuple[str, ...]
    assumptions: tuple[str, ...]
    versions: dict[str, str]
    input_hashes: dict[str, str]


def interval_index(value: str) -> int:
    hour, minute = (int(part) for part in value.split(":"))
    if minute % 15:
        raise ValueError(f"Time is not on the 15-minute grid: {value}")
    return hour * 4 + minute // 15


def window_indices(start: str, end: str) -> list[int]:
    first = interval_index(start)
    last = interval_index(end)
    if start == "00:00" and end == "23:45":
        return list(range(INTERVALS))
    if last <= first:
        raise ValueError(f"Overnight or empty windows are not supported: {start}-{end}")
    return list(range(first, last))


def family_direction(family: TaskFamily) -> int:
    return -1 if family == TaskFamily.RENEWABLE_ABSORPTION else 1


def request_binding(specification: TaskSpecification) -> tuple[np.ndarray, np.ndarray]:
    """Return requested magnitudes and signs using the frozen priority rule.

    A single window shared by a composite request follows the first task
    family, which is the explicitly declared highest-priority objective in the
    benchmark corpus.  Separate windows bind positionally to task families.
    """
    field = specification.fields.get("target_power")
    if field is None or field.value is None:
        raise ValueError("target_power must be resolved before joint optimization")
    target = float(field.value)
    if target < 0:
        raise ValueError("target_power must be nonnegative; direction comes from task family")
    requested = np.zeros(INTERVALS)
    direction = np.zeros(INTERVALS)
    families = specification.task_families
    windows = specification.windows
    if len(families) == len(windows) and len(families) > 1:
        bindings = zip(families, windows, strict=True)
    elif len(windows) == 1 or len(families) == 1:
        bindings = ((families[0], window) for window in windows)
    else:
        raise ValueError(
            "Composite tasks must provide either one shared window or one window per family"
        )
    for family, window in bindings:
        sign = family_direction(family)
        for index in window_indices(window.start, window.end):
            if direction[index] not in {0, sign}:
                raise ValueError("Conflicting service directions overlap in one interval")
            requested[index] = target
            direction[index] = sign
    return requested, direction


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FrozenJointRCOPF:
    """Deterministic linear RC--OPF candidate scheduler for the frozen case."""

    def __init__(self, repo_root: Path, *, data_root: Path | None = None) -> None:
        self.repo_root = repo_root.resolve()
        self.case_dir = self.repo_root / "cases" / "ieee33_10bem_15min"
        self.data_root = (
            data_root
            or Path(os.getenv("B2G_DATA_ROOT", self.repo_root / ".local-data"))
        ).resolve()
        self.local_case = self.data_root / "ieee33_10bem_15min"
        self.profile_dir = self.local_case / "profiles"
        self.config_path = self.case_dir / "optimization" / "joint_dispatch.json"
        self.storage_path = self.case_dir / "der" / "storage.json"
        self.rc_path = self.local_case / "models" / "rc_parameters.json"
        self.sensitivity_dir = self.local_case / "models" / "grid_sensitivities"
        self.sensitivity_manifest_path = self.sensitivity_dir / "manifest.json"
        self.target_days_path = self.case_dir / "days" / "target_days.json"
        self.daily_metrics_path = self.profile_dir / "daily_metrics.csv"
        self.baseline_path = self.profile_dir / "bem_baseline_15min.csv"
        self.thermal_path = self.profile_dir / "bem_thermal_15min.csv"
        self.prices_path = self.profile_dir / "prices_15min.csv"
        required = [
            self.config_path,
            self.storage_path,
            self.rc_path,
            self.sensitivity_manifest_path,
            self.target_days_path,
            self.daily_metrics_path,
            self.baseline_path,
            self.thermal_path,
            self.prices_path,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Joint RC-OPF inputs are missing: {missing}")
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.storage = json.loads(self.storage_path.read_text(encoding="utf-8"))
        self.rc = json.loads(self.rc_path.read_text(encoding="utf-8"))
        self.sensitivity_manifest = json.loads(
            self.sensitivity_manifest_path.read_text(encoding="utf-8")
        )
        self.target_days = json.loads(self.target_days_path.read_text(encoding="utf-8"))[
            "target_days"
        ]
        self._verify_sensitivity_arrays()
        self.arrays = {
            name: np.load(self.data_root / record["path"], allow_pickle=False)
            for name, record in self.sensitivity_manifest["arrays"].items()
        }
        self.bem_ids = list(self.sensitivity_manifest["bem_ids"])
        self.day_names = list(self.sensitivity_manifest["day_names"])
        self.baseline, self.aggregation = self._load_bem_profile(
            self.baseline_path, "portfolio_kw", aggregation=True
        )
        self.temperature, _ = self._load_bem_profile(
            self.thermal_path, "mean_zone_c", aggregation=False
        )
        self.mean_outdoor = self._load_daily_outdoor()
        self.prices = self._load_prices()

    def _verify_sensitivity_arrays(self) -> None:
        for record in self.sensitivity_manifest["arrays"].values():
            path = self.data_root / record["path"]
            if not path.is_file() or sha256_file(path) != record["sha256"]:
                raise RuntimeError(f"Grid sensitivity hash mismatch: {path}")

    def _load_bem_profile(
        self, path: Path, column: str, *, aggregation: bool
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        values: dict[str, np.ndarray] = {}
        factors: dict[str, float] = {}
        counts: dict[str, int] = {}
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                bem_id = row["bem_id"]
                if bem_id not in values:
                    values[bem_id] = np.zeros(365 * INTERVALS)
                    counts[bem_id] = 0
                    if aggregation:
                        factors[bem_id] = float(row["aggregation_factor"])
                index = counts[bem_id]
                values[bem_id][index] = float(row[column])
                counts[bem_id] += 1
        if set(values) != set(self.bem_ids) or any(value != 365 * INTERVALS for value in counts.values()):
            raise ValueError(f"Invalid BEM profile {path}")
        return values, factors

    def _load_daily_outdoor(self) -> dict[str, float]:
        with self.daily_metrics_path.open("r", encoding="utf-8", newline="") as handle:
            return {row["date"]: float(row["mean_dry_bulb_c"]) for row in csv.DictReader(handle)}

    def _load_prices(self) -> np.ndarray:
        values: list[float] = []
        with self.prices_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                values.append(float(row["rt_lmp_usd_per_mwh"]))
        if len(values) != 365 * INTERVALS:
            raise ValueError("Price profile must contain 35,040 intervals")
        return np.array(values)

    def solve(
        self,
        specification: TaskSpecification,
        *,
        building_flexibility_multiplier_by_bem: dict[str, float] | None = None,
        service_direction_override: int | None = None,
        fixed_building_response_by_bem: dict[str, list[float]] | None = None,
    ) -> JointDispatchResult:
        if specification.case_id != "ieee33_10bem_15min":
            raise ValueError(f"Unsupported case: {specification.case_id}")
        if specification.clarification_required:
            raise ValueError("A task requiring clarification cannot enter RC-OPF")
        if not specification.operating_day or specification.operating_day not in self.target_days:
            raise ValueError(f"Unknown operating day: {specification.operating_day}")
        if not specification.target_bems:
            raise ValueError("At least one target BEM is required")
        unknown = set(specification.target_bems) - set(self.bem_ids)
        if unknown:
            raise ValueError(f"Unknown target BEMs: {sorted(unknown)}")

        requested, direction = request_binding(specification)
        if service_direction_override is not None:
            if service_direction_override not in {-1, 1}:
                raise ValueError("service_direction_override must be -1 or 1")
            direction[requested > 0] = service_direction_override
        day_name = specification.operating_day
        day_value = self.target_days[day_name]
        day_offset = (date.fromisoformat(day_value) - date(2025, 1, 1)).days * INTERVALS
        day_slice = slice(day_offset, day_offset + INTERVALS)
        target_bems = list(specification.target_bems)
        fixed_building_response_by_bem = fixed_building_response_by_bem or {}
        unknown_fixed = set(fixed_building_response_by_bem) - set(target_bems)
        if unknown_fixed:
            raise ValueError(f"Fixed building responses contain non-target BEMs: {unknown_fixed}")
        for bem_id, values in fixed_building_response_by_bem.items():
            if len(values) != INTERVALS or any(not math.isfinite(float(value)) for value in values):
                raise ValueError(f"Fixed building response must contain 96 finite values: {bem_id}")
        fixed_response_digest = (
            hashlib.sha256(
                json.dumps(
                    fixed_building_response_by_bem,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if fixed_building_response_by_bem
            else None
        )
        flexibility_multipliers = {
            bem_id: float((building_flexibility_multiplier_by_bem or {}).get(bem_id, 1.0))
            for bem_id in target_bems
        }
        if any(value < 0.0 or value > 1.0 for value in flexibility_multipliers.values()):
            raise ValueError("Building flexibility multipliers must lie in [0, 1]")
        n_bem = len(target_bems)
        n_bt = n_bem * INTERVALS
        red_offset = 0
        inc_offset = n_bt
        dis_offset = 2 * n_bt
        ch_offset = 3 * n_bt
        soc_offset = 4 * n_bt
        short_offset = soc_offset + n_bem * (INTERVALS + 1)
        n_variables = short_offset + INTERVALS

        def bt(offset: int, bem: int, interval: int) -> int:
            return offset + bem * INTERVALS + interval

        def soc(bem: int, interval: int) -> int:
            return soc_offset + bem * (INTERVALS + 1) + interval

        bounds: list[tuple[float, float]] = [(0.0, 0.0)] * n_variables
        reduction_fraction = float(
            self.config["building_flexibility"]["maximum_reduction_fraction_of_baseline"]
        )
        increase_fraction = float(
            self.config["building_flexibility"]["maximum_increase_fraction_of_baseline"]
        )
        maximum_charge = float(self.storage["maximum_charge_kw_per_bem"])
        maximum_discharge = float(self.storage["maximum_discharge_kw_per_bem"])
        capacity = float(self.storage["capacity_kwh_per_bem"])
        minimum_soc = capacity * float(self.storage["minimum_soc_fraction"])
        maximum_soc = capacity * float(self.storage["maximum_soc_fraction"])
        initial_soc = capacity * float(self.storage["initial_soc_fraction"])
        terminal_tolerance = capacity * float(self.storage["terminal_soc_tolerance_fraction"])
        for bem_index, bem_id in enumerate(target_bems):
            baseline = self.baseline[bem_id][day_slice]
            flexibility_multiplier = flexibility_multipliers[bem_id]
            for interval in range(INTERVALS):
                if bem_id in fixed_building_response_by_bem:
                    fixed = float(fixed_building_response_by_bem[bem_id][interval])
                    bounds[bt(red_offset, bem_index, interval)] = (
                        max(0.0, fixed),
                        max(0.0, fixed),
                    )
                    bounds[bt(inc_offset, bem_index, interval)] = (
                        max(0.0, -fixed),
                        max(0.0, -fixed),
                    )
                else:
                    bounds[bt(red_offset, bem_index, interval)] = (
                        0.0,
                        max(
                            0.0,
                            baseline[interval]
                            * reduction_fraction
                            * flexibility_multiplier,
                        ),
                    )
                    bounds[bt(inc_offset, bem_index, interval)] = (
                        0.0,
                        max(
                            0.0,
                            baseline[interval]
                            * increase_fraction
                            * flexibility_multiplier,
                        ),
                    )
                bounds[bt(dis_offset, bem_index, interval)] = (0.0, maximum_discharge)
                bounds[bt(ch_offset, bem_index, interval)] = (0.0, maximum_charge)
            for interval in range(INTERVALS + 1):
                bounds[soc(bem_index, interval)] = (minimum_soc, maximum_soc)
            bounds[soc(bem_index, 0)] = (initial_soc, initial_soc)
            bounds[soc(bem_index, INTERVALS)] = (
                initial_soc - terminal_tolerance,
                initial_soc + terminal_tolerance,
            )
        for interval in range(INTERVALS):
            bounds[short_offset + interval] = (
                (0.0, float(requested[interval]))
                if requested[interval] > 0
                else (0.0, 0.0)
            )

        objective = np.zeros(n_variables)
        weights = self.config["objective_weights"]
        for bem_index in range(n_bem):
            for interval in range(INTERVALS):
                objective[bt(red_offset, bem_index, interval)] = float(
                    weights["building_adjustment_kw"]
                )
                objective[bt(inc_offset, bem_index, interval)] = float(
                    weights["building_adjustment_kw"]
                )
                objective[bt(dis_offset, bem_index, interval)] = float(
                    weights["storage_throughput_kw"]
                )
                objective[bt(ch_offset, bem_index, interval)] = float(
                    weights["storage_throughput_kw"]
                )
        for interval in range(INTERVALS):
            objective[short_offset + interval] = float(weights["unserved_request_kw"])

        ub_rows: list[int] = []
        ub_cols: list[int] = []
        ub_data: list[float] = []
        ub_rhs: list[float] = []
        eq_rows: list[int] = []
        eq_cols: list[int] = []
        eq_data: list[float] = []
        eq_rhs: list[float] = []

        def add_ub(coefficients: dict[int, float], right_hand_side: float) -> None:
            row = len(ub_rhs)
            for column, value in coefficients.items():
                if abs(value) > 1e-15:
                    ub_rows.append(row)
                    ub_cols.append(column)
                    ub_data.append(float(value))
            ub_rhs.append(float(right_hand_side))

        def add_eq(coefficients: dict[int, float], right_hand_side: float) -> None:
            row = len(eq_rhs)
            for column, value in coefficients.items():
                if abs(value) > 1e-15:
                    eq_rows.append(row)
                    eq_cols.append(column)
                    eq_data.append(float(value))
            eq_rhs.append(float(right_hand_side))

        charge_efficiency = math.sqrt(float(self.storage["round_trip_efficiency"]))
        discharge_efficiency = charge_efficiency
        for bem_index in range(n_bem):
            for interval in range(INTERVALS):
                add_eq(
                    {
                        soc(bem_index, interval + 1): 1.0,
                        soc(bem_index, interval): -1.0,
                        bt(ch_offset, bem_index, interval): -charge_efficiency * DT_HOURS,
                        bt(dis_offset, bem_index, interval): DT_HOURS / discharge_efficiency,
                    },
                    0.0,
                )

        for interval in range(INTERVALS):
            if requested[interval] <= 0:
                continue
            service: dict[int, float] = {}
            for bem_index in range(n_bem):
                service[bt(red_offset, bem_index, interval)] = 1.0
                service[bt(inc_offset, bem_index, interval)] = -1.0
                service[bt(dis_offset, bem_index, interval)] = 1.0
                service[bt(ch_offset, bem_index, interval)] = -1.0
            sign = float(direction[interval])
            add_ub(
                {
                    **{column: -sign * value for column, value in service.items()},
                    short_offset + interval: -1.0,
                },
                -requested[interval],
            )
            add_ub({column: -sign * value for column, value in service.items()}, 0.0)

        regime = self._regime(day_value)
        comfort = self.config["comfort"]
        occupied_start = interval_index(comfort["occupied_start"])
        occupied_end = interval_index(comfort["occupied_end"])
        is_weekend = date.fromisoformat(day_value).weekday() >= 5
        tolerance = float(comfort["linear_screening_tolerance_c"])
        predicted_temperature_coefficients: dict[str, list[dict[int, float]]] = {}
        for bem_index, bem_id in enumerate(target_bems):
            if bem_id in fixed_building_response_by_bem:
                predicted_temperature_coefficients[bem_id] = [
                    {} for _ in range(INTERVALS)
                ]
                continue
            model = self.rc["models"][bem_id][regime]
            coefficients = model["state_equation"]["coefficients"]
            state = float(coefficients["mean_zone_temperature_c"])
            control = float(coefficients["single_building_electricity_kw"])
            factor = self.aggregation[bem_id]
            rows: list[dict[int, float]] = []
            for interval in range(INTERVALS):
                row: dict[int, float] = {}
                for prior in range(interval):
                    response = (state ** (interval - 1 - prior)) * control / factor
                    row[bt(red_offset, bem_index, prior)] = -response
                    row[bt(inc_offset, bem_index, prior)] = response
                rows.append(row)
                occupied = not is_weekend and occupied_start <= interval < occupied_end
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
                baseline_temperature = float(self.temperature[bem_id][day_offset + interval])
                effective_lower = min(lower, baseline_temperature) - tolerance
                effective_upper = max(upper, baseline_temperature) + tolerance
                add_ub(row, effective_upper - baseline_temperature)
                add_ub(
                    {column: -value for column, value in row.items()},
                    baseline_temperature - effective_lower,
                )
            predicted_temperature_coefficients[bem_id] = rows

        day_index = self.day_names.index(day_name)
        bem_positions = [self.bem_ids.index(bem_id) for bem_id in target_bems]
        baseline_voltage = self.arrays["baseline_bus_voltage_pu"][day_index]
        baseline_line = self.arrays["baseline_line_loading_pct"][day_index]
        baseline_transformer = self.arrays["baseline_transformer_loading_pct"][day_index]
        voltage_sensitivity = self.arrays["bus_voltage_pu_per_kw_reduction"][day_index]
        line_sensitivity = self.arrays["line_loading_pct_per_kw_reduction"][day_index]
        transformer_sensitivity = self.arrays[
            "transformer_loading_pct_per_kw_reduction"
        ][day_index]
        grid = self.config["linear_grid_constraints"]
        for interval in range(INTERVALS):
            service_columns: list[tuple[int, int, int, int]] = [
                (
                    bt(red_offset, bem_index, interval),
                    bt(inc_offset, bem_index, interval),
                    bt(dis_offset, bem_index, interval),
                    bt(ch_offset, bem_index, interval),
                )
                for bem_index in range(n_bem)
            ]

            def sensitivity_row(values: np.ndarray) -> dict[int, float]:
                row: dict[int, float] = {}
                for local_bem, global_bem in enumerate(bem_positions):
                    sensitivity = float(values[global_bem])
                    red, inc, dis, ch = service_columns[local_bem]
                    row[red] = sensitivity
                    row[inc] = -sensitivity
                    row[dis] = sensitivity
                    row[ch] = -sensitivity
                return row

            for bus_index in range(33):
                row = sensitivity_row(voltage_sensitivity[interval, bus_index])
                add_ub(
                    row,
                    float(grid["maximum_voltage_pu"]) - baseline_voltage[interval, bus_index],
                )
                add_ub(
                    {column: -value for column, value in row.items()},
                    baseline_voltage[interval, bus_index] - float(grid["minimum_voltage_pu"]),
                )
            for line_index in range(baseline_line.shape[1]):
                row = sensitivity_row(line_sensitivity[interval, line_index])
                add_ub(
                    row,
                    float(grid["maximum_line_loading_pct"])
                    - baseline_line[interval, line_index],
                )
            row = sensitivity_row(transformer_sensitivity[interval])
            add_ub(
                row,
                float(grid["maximum_transformer_loading_pct"])
                - baseline_transformer[interval],
            )

        a_ub = coo_matrix(
            (ub_data, (ub_rows, ub_cols)), shape=(len(ub_rhs), n_variables)
        ).tocsr()
        a_eq = coo_matrix(
            (eq_data, (eq_rows, eq_cols)), shape=(len(eq_rhs), n_variables)
        ).tocsr()
        solved = linprog(
            objective,
            A_ub=a_ub,
            b_ub=np.array(ub_rhs),
            A_eq=a_eq,
            b_eq=np.array(eq_rhs),
            bounds=bounds,
            method="highs",
            options={"presolve": True},
        )
        if not solved.success or solved.x is None:
            return JointDispatchResult(
                episode_id=specification.episode_id,
                status="solver_failed",
                solver_status=int(solved.status),
                solver_message=str(solved.message),
                objective_value=None,
                requested_power_kw=float(specification.fields["target_power"].value),
                committed_power_kw=0.0,
                delivery_ratio=0.0,
                aggregate_service_kw=tuple(0.0 for _ in range(INTERVALS)),
                requested_service_kw=tuple(float(value) for value in requested * direction),
                unserved_request_kw=tuple(float(value) for value in requested),
                schedules={},
                predicted_minimum_voltage_pu=None,
                predicted_maximum_voltage_pu=None,
                predicted_maximum_line_loading_pct=None,
                predicted_maximum_transformer_loading_pct=None,
                predicted_temperature_range_c_by_bem={},
                fixed_building_response_bems=tuple(
                    sorted(fixed_building_response_by_bem)
                ),
                assumptions=(
                    *self._assumptions(),
                    f"building_flexibility_multipliers={json.dumps(flexibility_multipliers, sort_keys=True)}",
                    f"fixed_building_response_sha256={fixed_response_digest}",
                ),
                versions={"scipy": scipy_version, "solver": "HiGHS via scipy.optimize.linprog"},
                input_hashes=self._input_hashes(),
            )

        solution = solved.x
        aggregate_service = np.zeros(INTERVALS)
        schedules: dict[str, dict[str, tuple[float, ...]]] = {}
        temperature_ranges: dict[str, tuple[float, float]] = {}
        for bem_index, bem_id in enumerate(target_bems):
            red = np.array(
                [solution[bt(red_offset, bem_index, interval)] for interval in range(INTERVALS)]
            )
            inc = np.array(
                [solution[bt(inc_offset, bem_index, interval)] for interval in range(INTERVALS)]
            )
            dis = np.array(
                [solution[bt(dis_offset, bem_index, interval)] for interval in range(INTERVALS)]
            )
            ch = np.array(
                [solution[bt(ch_offset, bem_index, interval)] for interval in range(INTERVALS)]
            )
            service = red - inc + dis - ch
            aggregate_service += service
            temperature = self.temperature[bem_id][day_slice].copy()
            for interval, row in enumerate(predicted_temperature_coefficients[bem_id]):
                temperature[interval] += sum(value * solution[column] for column, value in row.items())
            soc_values = np.array(
                [solution[soc(bem_index, interval)] for interval in range(INTERVALS + 1)]
            )
            schedules[bem_id] = {
                "building_reduction_kw": tuple(float(value) for value in red - inc),
                "storage_discharge_kw": tuple(float(value) for value in dis),
                "storage_charge_kw": tuple(float(value) for value in ch),
                "net_service_kw": tuple(float(value) for value in service),
                "state_of_charge_kwh": tuple(float(value) for value in soc_values),
                "predicted_mean_zone_temperature_c": tuple(float(value) for value in temperature),
            }
            temperature_ranges[bem_id] = (float(temperature.min()), float(temperature.max()))

        shortfall = np.array(
            [solution[short_offset + interval] for interval in range(INTERVALS)]
        )
        requested_total = float(requested.sum())
        delivered_total = float(np.sum(np.maximum(0.0, requested - shortfall)))
        ratio = min(1.0, delivered_total / requested_total) if requested_total > 0 else 0.0
        active = requested > 0
        directed_service = direction[active] * aggregate_service[active]
        committed = max(0.0, float(directed_service.min())) if np.any(active) else 0.0
        predicted_voltage = baseline_voltage.copy()
        predicted_line = baseline_line.copy()
        predicted_transformer = baseline_transformer.copy()
        for local_bem, global_bem in enumerate(bem_positions):
            predicted_voltage += (
                voltage_sensitivity[:, :, global_bem] * aggregate_service[:, None]
                if n_bem == 1
                else voltage_sensitivity[:, :, global_bem]
                * np.array(schedules[target_bems[local_bem]]["net_service_kw"])[:, None]
            )
            predicted_line += line_sensitivity[:, :, global_bem] * np.array(
                schedules[target_bems[local_bem]]["net_service_kw"]
            )[:, None]
            predicted_transformer += transformer_sensitivity[:, global_bem] * np.array(
                schedules[target_bems[local_bem]]["net_service_kw"]
            )
        status = "optimal_full" if float(shortfall.sum()) <= 1e-5 else "optimal_partial"
        return JointDispatchResult(
            episode_id=specification.episode_id,
            status=status,
            solver_status=int(solved.status),
            solver_message=str(solved.message),
            objective_value=float(solved.fun),
            requested_power_kw=float(specification.fields["target_power"].value),
            committed_power_kw=committed,
            delivery_ratio=ratio,
            aggregate_service_kw=tuple(float(value) for value in aggregate_service),
            requested_service_kw=tuple(float(value) for value in requested * direction),
            unserved_request_kw=tuple(float(value) for value in shortfall),
            schedules=schedules,
            predicted_minimum_voltage_pu=float(predicted_voltage.min()),
            predicted_maximum_voltage_pu=float(predicted_voltage.max()),
            predicted_maximum_line_loading_pct=float(predicted_line.max()),
            predicted_maximum_transformer_loading_pct=float(predicted_transformer.max()),
            predicted_temperature_range_c_by_bem=temperature_ranges,
            fixed_building_response_bems=tuple(
                sorted(fixed_building_response_by_bem)
            ),
            assumptions=(
                *self._assumptions(),
                f"building_flexibility_multipliers={json.dumps(flexibility_multipliers, sort_keys=True)}",
                f"service_direction_override={service_direction_override}",
                f"fixed_building_response_sha256={fixed_response_digest}",
            ),
            versions={"scipy": scipy_version, "solver": "HiGHS via scipy.optimize.linprog"},
            input_hashes=self._input_hashes(),
        )

    def _regime(self, day_value: str) -> str:
        outdoor = self.mean_outdoor[day_value]
        if outdoor <= 15.0:
            return "heating"
        if outdoor >= 18.0:
            return "cooling"
        return "shoulder"

    def _assumptions(self) -> tuple[str, ...]:
        return (
            self.config["service_sign_convention"],
            self.config["reactive_power_approximation"],
            self.config["validation_policy"],
            self.rc["limitations"][0],
        )

    def _input_hashes(self) -> dict[str, str]:
        return {
            "joint_dispatch_config": sha256_file(self.config_path),
            "storage": sha256_file(self.storage_path),
            "rc_parameters": sha256_file(self.rc_path),
            "grid_sensitivity_manifest": sha256_file(self.sensitivity_manifest_path),
            "target_days": sha256_file(self.target_days_path),
            "bem_baseline": sha256_file(self.baseline_path),
            "bem_thermal": sha256_file(self.thermal_path),
            "prices": sha256_file(self.prices_path),
        }


def result_payload(result: JointDispatchResult) -> dict[str, Any]:
    return asdict(result)


def write_result(result: JointDispatchResult, output_path: Path) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=False)
    payload = result_payload(result)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return sha256_file(output_path)
