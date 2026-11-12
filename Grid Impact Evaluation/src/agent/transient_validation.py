"""Deterministic comparison of transient trajectories from independent simulators.

The module is deliberately simulator- and MCP-agnostic. Adapters are responsible
for extracting channel samples and declaring their units; this module handles
validation, common-window alignment, interpolation, metrics and tolerances.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


class TrajectoryValidationError(ValueError):
    """Raised when trajectory inputs cannot be compared safely."""


@dataclass(frozen=True)
class ChannelSeries:
    """Samples for one physical channel."""

    time_s: Sequence[float]
    values: Sequence[float]
    unit: str
    base_id: str | None = None


@dataclass(frozen=True)
class ChannelMetrics:
    """Numerical comparison metrics for one aligned channel."""

    reference_channel: str
    candidate_channel: str
    unit: str
    sample_count: int
    mae: float
    rmse: float
    bias: float
    max_absolute_error: float
    reference_peak_magnitude: float
    candidate_peak_magnitude: float
    peak_magnitude_error: float
    max_absolute_error_time_s: float
    tolerance: float | None
    passed: bool | None


@dataclass(frozen=True)
class TrajectoryComparison:
    """Complete comparison report, including its exact alignment window."""

    status: str
    reference_engine: str
    candidate_engine: str
    start_time_s: float
    end_time_s: float
    sample_count: int
    alignment: str
    channels: tuple[ChannelMetrics, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable report."""
        return asdict(self)


# Convert only unambiguous units within the same physical dimension.
_UNIT_TABLE: dict[str, tuple[str, float, str]] = {
    "pu": ("per_unit", 1.0, "pu"),
    "mw": ("power", 1.0, "MW"),
    "kw": ("power", 1e-3, "MW"),
    "w": ("power", 1e-6, "MW"),
    "mva": ("apparent_power", 1.0, "MVA"),
    "kva": ("apparent_power", 1e-3, "MVA"),
    "va": ("apparent_power", 1e-6, "MVA"),
    "kv": ("voltage", 1.0, "kV"),
    "v": ("voltage", 1e-3, "kV"),
    "ka": ("current", 1.0, "kA"),
    "a": ("current", 1e-3, "kA"),
    "mvar": ("reactive_power", 1.0, "Mvar"),
    "kvar": ("reactive_power", 1e-3, "Mvar"),
    "var": ("reactive_power", 1e-6, "Mvar"),
    "hz": ("frequency", 1.0, "Hz"),
    "rad": ("angle", 180.0 / math.pi, "deg"),
    "deg": ("angle", 1.0, "deg"),
    "s": ("time", 1.0, "s"),
    "ms": ("time", 1e-3, "s"),
}


def _unit_info(unit: str) -> tuple[str, float, str]:
    if not isinstance(unit, str) or not unit.strip():
        raise TrajectoryValidationError("Every channel must declare a non-empty unit.")
    key = unit.strip().lower()
    try:
        return _UNIT_TABLE[key]
    except KeyError as exc:
        raise TrajectoryValidationError(
            f"Unsupported or ambiguous unit {unit!r}; provide a supported unit "
            "or normalize it in the simulator adapter."
        ) from exc


def _validate_series(name: str, series: ChannelSeries) -> tuple[list[float], list[float]]:
    if not isinstance(series, ChannelSeries):
        raise TrajectoryValidationError(
            f"Channel {name!r} must be provided as a ChannelSeries."
        )
    if not isinstance(series.unit, str) or not series.unit.strip():
        raise TrajectoryValidationError(f"Channel {name!r} has no unit.")
    if isinstance(series.time_s, (str, bytes)) or isinstance(series.values, (str, bytes)):
        raise TrajectoryValidationError(
            f"Channel {name!r} time and value samples must be numeric sequences."
        )
    try:
        raw_times = list(series.time_s)
        raw_values = list(series.values)
        if any(isinstance(value, bool) for value in raw_times + raw_values):
            raise ValueError("boolean samples are not valid engineering measurements")
        times = [float(value) for value in raw_times]
        values = [float(value) for value in raw_values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise TrajectoryValidationError(
            f"Channel {name!r} contains a non-numeric or invalid sample."
        ) from exc
    if len(times) != len(values) or len(times) < 2:
        raise TrajectoryValidationError(
            f"Channel {name!r} needs equal time/value arrays with at least two samples."
        )
    if any(not math.isfinite(value) for value in times + values):
        raise TrajectoryValidationError(f"Channel {name!r} contains NaN or infinity.")
    if any(right <= left for left, right in zip(times, times[1:])):
        raise TrajectoryValidationError(
            f"Channel {name!r} timestamps must be strictly increasing; "
            "sort or resolve duplicate timestamps in the adapter."
        )
    return times, values


def _interpolate(times: Sequence[float], values: Sequence[float], point: float) -> float:
    """Linearly interpolate inside the sampled interval; never extrapolate."""
    index = bisect_left(times, point)
    if index < len(times) and times[index] == point:
        return values[index]
    if index == 0 or index == len(times):
        raise TrajectoryValidationError(
            f"Requested alignment time {point} s is outside a trajectory's sample range."
        )
    left_t, right_t = times[index - 1], times[index]
    fraction = (point - left_t) / (right_t - left_t)
    return values[index - 1] + fraction * (values[index] - values[index - 1])


def _time_weighted_metrics(
    errors: Sequence[float], times: Sequence[float]
) -> tuple[float, float, float]:
    """Integrate piecewise-linear errors exactly for MAE, RMSE and bias."""
    duration = times[-1] - times[0]
    if not math.isfinite(duration) or duration <= 0:
        raise TrajectoryValidationError("Time-weighted metrics require finite positive duration.")

    absolute_error_area = 0.0
    squared_error_area = 0.0
    bias_area = 0.0
    for left, right, left_time, right_time in zip(
        errors, errors[1:], times, times[1:]
    ):
        interval = right_time - left_time
        left_abs, right_abs = abs(left), abs(right)
        if left == 0 or right == 0 or (left > 0) == (right > 0):
            absolute_area = 0.5 * (left_abs + right_abs) * interval
        else:
            # The linear error crosses zero inside this interval; integrate
            # the two triangular areas instead of overestimating with a trapezoid.
            absolute_area = (
                0.5
                * interval
                * (left_abs * left_abs + right_abs * right_abs)
                / (left_abs + right_abs)
            )
        absolute_error_area += absolute_area
        squared_error_area += (
            interval * (left * left + left * right + right * right) / 3.0
        )
        bias_area += 0.5 * (left + right) * interval

    metrics = (
        absolute_error_area / duration,
        math.sqrt(squared_error_area / duration),
        bias_area / duration,
    )
    if any(not math.isfinite(value) for value in metrics):
        raise TrajectoryValidationError(
            "Metric calculation overflowed; rescale the input quantities before comparison."
        )
    return metrics


def compare_transient_trajectories(
    reference: Mapping[str, ChannelSeries],
    candidate: Mapping[str, ChannelSeries],
    *,
    reference_engine: str = "reference",
    candidate_engine: str = "candidate",
    channel_map: Mapping[str, str] | None = None,
    tolerances: Mapping[str, float] | None = None,
) -> TrajectoryComparison:
    """Compare two simulator outputs on a common, non-extrapolated time grid.

    Args:
        reference: Reference channels keyed by stable engineering channel IDs.
        candidate: Candidate channels keyed by stable engineering channel IDs.
        reference_engine: Human-readable reference engine name.
        candidate_engine: Human-readable candidate engine name.
        channel_map: Optional mapping from reference channel IDs to candidate IDs.
            Without a map, only exact channel-name intersections are compared.
        tolerances: Optional maximum-absolute-error limits keyed by reference
            channel ID. Channels without a limit have an unevaluated pass state.

    Returns:
        A deterministic report with MAE, RMSE, bias, maximum absolute error,
        and peak-magnitude error for each compared channel.

    Raises:
        TrajectoryValidationError: If inputs, channel mappings, units, or time
            overlap are invalid. No implicit extrapolation or unit guessing occurs.
    """
    if not isinstance(reference, Mapping) or not isinstance(candidate, Mapping):
        raise TrajectoryValidationError("Both engine outputs must be channel mappings.")
    if not reference or not candidate:
        raise TrajectoryValidationError("Both engines must provide at least one channel.")
    if any(
        not isinstance(channel_id, str) or not channel_id.strip()
        for channel_id in (*reference.keys(), *candidate.keys())
    ):
        raise TrajectoryValidationError("Channel identifiers must be non-empty strings.")
    if (
        not isinstance(reference_engine, str)
        or not reference_engine.strip()
        or not isinstance(candidate_engine, str)
        or not candidate_engine.strip()
    ):
        raise TrajectoryValidationError("Engine names must be non-empty strings.")
    if reference_engine.strip().casefold() == candidate_engine.strip().casefold():
        raise TrajectoryValidationError("Reference and candidate engine names must differ.")

    if channel_map is not None and not isinstance(channel_map, Mapping):
        raise TrajectoryValidationError("channel_map must be a mapping of reference to candidate IDs.")
    mapping = dict(channel_map) if channel_map is not None else {
        name: name for name in sorted(set(reference) & set(candidate))
    }
    if not mapping:
        raise TrajectoryValidationError("No matching channels were selected.")
    if any(
        not isinstance(ref_name, str) or not ref_name.strip()
        or not isinstance(cand_name, str) or not cand_name.strip()
        for ref_name, cand_name in mapping.items()
    ):
        raise TrajectoryValidationError("Channel mapping keys and values must be non-empty strings.")
    if len(set(mapping.values())) != len(mapping):
        raise TrajectoryValidationError("Channel mapping must be one-to-one.")

    normalized_tolerances: dict[str, float] = {}
    if tolerances is not None:
        if not isinstance(tolerances, Mapping):
            raise TrajectoryValidationError("Tolerances must be a mapping keyed by reference channel ID.")
        unknown_tolerances = set(tolerances) - set(mapping)
        if unknown_tolerances:
            raise TrajectoryValidationError(
                "Tolerances reference channels that are not being compared: "
                + ", ".join(sorted(map(str, unknown_tolerances)))
            )
        if any(isinstance(value, bool) for value in tolerances.values()):
            raise TrajectoryValidationError("Tolerances must be numeric values, not booleans.")
        try:
            normalized_tolerances = {
                name: float(value) for name, value in tolerances.items()
            }
        except (TypeError, ValueError, OverflowError) as exc:
            raise TrajectoryValidationError(
                "Tolerances must be finite, non-negative values."
            ) from exc
        if any(
            not math.isfinite(value) or value < 0
            for value in normalized_tolerances.values()
        ):
            raise TrajectoryValidationError("Tolerances must be finite, non-negative values.")

    validated: dict[tuple[str, str], tuple[list[float], list[float], list[float], list[float], str]] = {}
    all_time_ranges: list[tuple[float, float]] = []
    for ref_name, cand_name in mapping.items():
        if ref_name not in reference:
            raise TrajectoryValidationError(f"Reference channel {ref_name!r} is missing.")
        if cand_name not in candidate:
            raise TrajectoryValidationError(f"Candidate channel {cand_name!r} is missing.")
        ref_series, cand_series = reference[ref_name], candidate[cand_name]
        ref_t, ref_v = _validate_series(ref_name, ref_series)
        cand_t, cand_v = _validate_series(cand_name, cand_series)
        ref_dimension, ref_scale, canonical_unit = _unit_info(ref_series.unit)
        cand_dimension, cand_scale, _ = _unit_info(cand_series.unit)
        if ref_dimension != cand_dimension:
            raise TrajectoryValidationError(
                f"Incompatible units for {ref_name!r}/{cand_name!r}: "
                f"{ref_series.unit!r} versus {cand_series.unit!r}."
            )
        if ref_dimension == "per_unit" and (
            not isinstance(ref_series.base_id, str)
            or not ref_series.base_id.strip()
            or not isinstance(cand_series.base_id, str)
            or not cand_series.base_id.strip()
            or ref_series.base_id.strip() != cand_series.base_id.strip()
        ):
            raise TrajectoryValidationError(
                f"Per-unit channel pair {ref_name!r}/{cand_name!r} requires "
                "the same explicit, verified base_id on both engines."
            )
        channel_start = max(ref_t[0], cand_t[0])
        channel_end = min(ref_t[-1], cand_t[-1])
        if channel_end <= channel_start:
            raise TrajectoryValidationError(
                f"Channel pair {ref_name!r}/{cand_name!r} has no common time interval."
            )
        all_time_ranges.append((channel_start, channel_end))
        normalized_ref = [value * ref_scale for value in ref_v]
        normalized_cand = [value * cand_scale for value in cand_v]
        if any(not math.isfinite(value) for value in normalized_ref + normalized_cand):
            raise TrajectoryValidationError(
                f"Unit normalization overflowed for channel {ref_name!r}/{cand_name!r}."
            )
        validated[(ref_name, cand_name)] = (
            ref_t, normalized_ref, cand_t, normalized_cand, canonical_unit,
        )

    # Use one grid valid for every selected channel pair. Its points are the
    # union of observed samples, clipped to the intersection of all valid ranges.
    start_time = max(start for start, _ in all_time_ranges)
    end_time = min(end for _, end in all_time_ranges)
    if end_time <= start_time:
        raise TrajectoryValidationError(
            "Selected channels do not share a common comparison interval."
        )
    grid = sorted(
        {
            time
            for values in validated.values()
            for times in (values[0], values[2])
            for time in times
            if start_time <= time <= end_time
        }
        | {start_time, end_time}
    )
    if len(grid) < 2:
        raise TrajectoryValidationError("The common time interval has fewer than two points.")

    metrics: list[ChannelMetrics] = []
    for (ref_name, cand_name), (ref_t, ref_v, cand_t, cand_v, unit) in validated.items():
        errors: list[float] = []
        reference_values: list[float] = []
        candidate_values: list[float] = []
        for point in grid:
            left = _interpolate(ref_t, ref_v, point)
            right = _interpolate(cand_t, cand_v, point)
            reference_values.append(left)
            candidate_values.append(right)
            errors.append(right - left)

        absolute_errors = [abs(value) for value in errors]
        # Time-weighted metrics avoid biasing the score toward whichever engine
        # happened to export a denser or irregularly sampled trajectory.
        mae, rmse, bias = _time_weighted_metrics(errors, grid)
        max_error_index = max(range(len(absolute_errors)), key=absolute_errors.__getitem__)
        max_error = absolute_errors[max_error_index]
        ref_peak = max(abs(value) for value in reference_values)
        cand_peak = max(abs(value) for value in candidate_values)
        tolerance = normalized_tolerances.get(ref_name)
        passed = None if tolerance is None else max_error <= tolerance
        if not math.isfinite(max_error) or not math.isfinite(ref_peak) or not math.isfinite(cand_peak):
            raise TrajectoryValidationError(
                f"Metric calculation overflowed for channel {ref_name!r}/{cand_name!r}."
            )
        metrics.append(
            ChannelMetrics(
                reference_channel=ref_name,
                candidate_channel=cand_name,
                unit=unit,
                sample_count=len(grid),
                mae=mae,
                rmse=rmse,
                bias=bias,
                max_absolute_error=max_error,
                reference_peak_magnitude=ref_peak,
                candidate_peak_magnitude=cand_peak,
                peak_magnitude_error=abs(cand_peak - ref_peak),
                max_absolute_error_time_s=grid[max_error_index],
                tolerance=None if tolerance is None else float(tolerance),
                passed=passed,
            )
        )

    evaluated = [item.passed for item in metrics if item.passed is not None]
    if not evaluated:
        status = "not_evaluated"
    elif not all(evaluated):
        status = "failed"
    elif len(evaluated) == len(metrics):
        status = "passed"
    else:
        status = "partially_evaluated"
    return TrajectoryComparison(
        status=status,
        reference_engine=reference_engine,
        candidate_engine=candidate_engine,
        start_time_s=start_time,
        end_time_s=end_time,
        sample_count=len(grid),
        alignment="linear_interpolation_on_union_of_observed_times_within_common_window",
        channels=tuple(metrics),
    )
