from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class PandapowerUnavailable(RuntimeError):
    pass


class PandapowerExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class FormalPandapowerConfig:
    network_path: Path
    line_limits_path: Path
    upstream_transformer_kva: float
    power_factor: float = 0.95
    intervals_per_day: int = 96


@dataclass(frozen=True)
class Pandapower15MinuteResult:
    converged: tuple[bool, ...]
    feeder_kw: tuple[float, ...]
    feeder_kvar: tuple[float, ...]
    minimum_voltage_pu: tuple[float, ...]
    maximum_voltage_pu: tuple[float, ...]
    maximum_line_loading_pct: tuple[float, ...]
    transformer_loading_pct: tuple[float, ...]
    bus_voltage_pu: dict[int, tuple[float, ...]]
    line_loading_pct: dict[int, tuple[float, ...]]
    version: str
    input_hashes: dict[str, str]


class FormalPandapowerAdapter:
    """Runs deterministic 15-minute AC power flow for total node profiles."""

    def __init__(self, config: FormalPandapowerConfig, *, pp_module: Any | None = None) -> None:
        self.config = config
        self._pp_module = pp_module

    def run(
        self,
        node_active_kw: dict[int, list[float]],
        *,
        node_reactive_kvar: dict[int, list[float]] | None = None,
        output_path: Path | None = None,
    ) -> Pandapower15MinuteResult:
        pp = self._module()
        if not self.config.network_path.is_file():
            raise PandapowerUnavailable(f"Frozen network is missing: {self.config.network_path}")
        limits = _read_line_limits(self.config.line_limits_path)
        net = pp.from_json(str(self.config.network_path))
        _validate_profiles(node_active_kw, self.config.intervals_per_day)
        if node_reactive_kvar is not None:
            _validate_profiles(node_reactive_kvar, self.config.intervals_per_day)

        load_rows: dict[int, list[int]] = {}
        for load_index, load in net.load.iterrows():
            load_rows.setdefault(int(load.bus), []).append(int(load_index))
        original_p = {int(index): float(row.p_mw) for index, row in net.load.iterrows()}
        original_q = {int(index): float(row.q_mvar) for index, row in net.load.iterrows()}

        converged: list[bool] = []
        feeder_kw: list[float] = []
        feeder_kvar: list[float] = []
        minimum_voltage: list[float] = []
        maximum_voltage: list[float] = []
        maximum_line: list[float] = []
        transformer_loading: list[float] = []
        bus_voltage: dict[int, list[float]] = {paper_bus: [] for paper_bus in range(1, 34)}
        line_loading: dict[int, list[float]] = {index: [] for index in limits}

        for interval in range(self.config.intervals_per_day):
            for load_index in net.load.index:
                net.load.at[load_index, "p_mw"] = original_p[int(load_index)]
                net.load.at[load_index, "q_mvar"] = original_q[int(load_index)]
            for paper_bus, values in node_active_kw.items():
                internal_bus = paper_bus - 1
                p_mw = values[interval] / 1000.0
                q_kvar = (
                    node_reactive_kvar[paper_bus][interval]
                    if node_reactive_kvar and paper_bus in node_reactive_kvar
                    else values[interval] * math.tan(math.acos(self.config.power_factor))
                )
                q_mvar = q_kvar / 1000.0
                rows = load_rows.get(internal_bus)
                if not rows:
                    rows = [int(pp.create_load(net, internal_bus, p_mw=0.0, q_mvar=0.0, name=f"b2g_bus_{paper_bus}"))]
                    load_rows[internal_bus] = rows
                weights = [max(original_p.get(row, 0.0), 0.0) for row in rows]
                denominator = sum(weights)
                if denominator <= 0:
                    weights = [1.0] * len(rows)
                    denominator = float(len(rows))
                for row, weight in zip(rows, weights, strict=True):
                    net.load.at[row, "p_mw"] = p_mw * weight / denominator
                    net.load.at[row, "q_mvar"] = q_mvar * weight / denominator
            try:
                options: dict[str, Any] = {
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
            except Exception as exc:
                raise PandapowerExecutionError(f"AC power flow failed at interval {interval}") from exc
            interval_converged = bool(net.converged)
            if not interval_converged:
                raise PandapowerExecutionError(f"AC power flow did not converge at interval {interval}")
            converged.append(interval_converged)
            p_kw = float(net.res_ext_grid.p_mw.sum()) * 1000.0
            q_kvar = float(net.res_ext_grid.q_mvar.sum()) * 1000.0
            feeder_kw.append(p_kw)
            feeder_kvar.append(q_kvar)
            voltages = [float(value) for value in net.res_bus.vm_pu]
            minimum_voltage.append(min(voltages))
            maximum_voltage.append(max(voltages))
            for internal_bus, value in enumerate(voltages):
                bus_voltage[internal_bus + 1].append(value)
            interval_line: list[float] = []
            for line_index, maximum_ka in limits.items():
                current = float(net.res_line.at[line_index, "i_ka"])
                loading = current / maximum_ka * 100.0
                line_loading[line_index].append(loading)
                interval_line.append(loading)
            maximum_line.append(max(interval_line))
            apparent_kva = math.sqrt(p_kw * p_kw + q_kvar * q_kvar)
            transformer_loading.append(apparent_kva / self.config.upstream_transformer_kva * 100.0)

        result = Pandapower15MinuteResult(
            converged=tuple(converged),
            feeder_kw=tuple(feeder_kw),
            feeder_kvar=tuple(feeder_kvar),
            minimum_voltage_pu=tuple(minimum_voltage),
            maximum_voltage_pu=tuple(maximum_voltage),
            maximum_line_loading_pct=tuple(maximum_line),
            transformer_loading_pct=tuple(transformer_loading),
            bus_voltage_pu={key: tuple(value) for key, value in bus_voltage.items()},
            line_loading_pct={key: tuple(value) for key, value in line_loading.items()},
            version=str(pp.__version__),
            input_hashes={
                "network": _sha256(self.config.network_path),
                "line_limits": _sha256(self.config.line_limits_path),
            },
        )
        if output_path:
            _write_result(result, output_path)
        return result

    def _module(self) -> Any:
        if self._pp_module is not None:
            return self._pp_module
        try:
            import pandapower as pp
        except ImportError as exc:
            raise PandapowerUnavailable("Install the simulation extra to use pandapower") from exc
        return pp


def _validate_profiles(profiles: dict[int, list[float]], expected: int) -> None:
    for paper_bus, values in profiles.items():
        if paper_bus < 1 or paper_bus > 33:
            raise ValueError(f"Paper bus must be in 1..33: {paper_bus}")
        if len(values) != expected:
            raise ValueError(f"Bus {paper_bus} has {len(values)} values; expected {expected}")


def _read_line_limits(path: Path) -> dict[int, float]:
    if not path.is_file():
        raise PandapowerUnavailable(
            "Experiment-specific line_limits.csv is missing; placeholder case33bw max_i_ka values are not used"
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = csv.DictReader(handle)
        limits = {int(row["line_index"]): float(row["max_i_ka"]) for row in rows}
    if not limits or any(value <= 0 for value in limits.values()):
        raise ValueError("line_limits.csv must contain positive max_i_ka values")
    return limits


def _write_result(result: Pandapower15MinuteResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "converged": result.converged,
        "feeder_kw": result.feeder_kw,
        "feeder_kvar": result.feeder_kvar,
        "minimum_voltage_pu": result.minimum_voltage_pu,
        "maximum_voltage_pu": result.maximum_voltage_pu,
        "maximum_line_loading_pct": result.maximum_line_loading_pct,
        "transformer_loading_pct": result.transformer_loading_pct,
        "bus_voltage_pu": result.bus_voltage_pu,
        "line_loading_pct": result.line_loading_pct,
        "version": result.version,
        "input_hashes": result.input_hashes,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
