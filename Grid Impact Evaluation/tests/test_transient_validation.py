"""Tests for deterministic cross-engine transient trajectory comparison."""

import math

import pytest

from agent.transient_validation import (
    ChannelSeries,
    TrajectoryValidationError,
    compare_transient_trajectories,
)


def series(times, values, unit="pu", base_id="system_base"):
    return ChannelSeries(time_s=times, values=values, unit=unit, base_id=base_id)


def test_identical_trajectories_pass_zero_error():
    report = compare_transient_trajectories(
        {"bus:1:voltage": series([0, 1, 2], [1.0, 0.98, 1.0])},
        {"bus:1:voltage": series([0, 0.5, 1, 2], [1.0, 0.99, 0.98, 1.0])},
        reference_engine="PowerWorld",
        candidate_engine="PSS/E",
        tolerances={"bus:1:voltage": 1e-12},
    )
    metric = report.channels[0]
    assert report.status == "passed"
    assert metric.max_absolute_error == pytest.approx(0.0)
    assert metric.rmse == pytest.approx(0.0)
    assert report.start_time_s == 0.0
    assert report.end_time_s == 2.0


def test_aligns_different_time_steps_without_extrapolation():
    report = compare_transient_trajectories(
        {"gen:1:angle": series([0, 1, 2, 3], [0, 1, 2, 3], "deg")},
        {"gen:1:angle": series([0.5, 1.5, 2.5], [0.5, 1.5, 2.5], "deg")},
        tolerances={"gen:1:angle": 1e-12},
    )
    assert report.start_time_s == 0.5
    assert report.end_time_s == 2.5
    assert report.channels[0].mae == pytest.approx(0.0)


def test_converts_supported_units_before_comparison():
    report = compare_transient_trajectories(
        {"gen:1:p": series([0, 1], [1.0, 2.0], "MW")},
        {"gen:1:p": series([0, 1], [1000.0, 2000.0], "kW")},
        tolerances={"gen:1:p": 1e-12},
    )
    assert report.status == "passed"
    assert report.channels[0].unit == "MW"


def test_reports_quantitative_error_and_tolerance_failure():
    report = compare_transient_trajectories(
        {"bus:1:voltage": series([0, 1], [1.0, 0.9])},
        {"bus:1:voltage": series([0, 1], [1.0, 0.8])},
        tolerances={"bus:1:voltage": 0.05},
    )
    metric = report.channels[0]
    assert report.status == "failed"
    assert metric.mae == pytest.approx(0.05)
    assert metric.rmse == pytest.approx(math.sqrt(0.01 / 3.0))
    assert metric.max_absolute_error == pytest.approx(0.1)
    assert metric.passed is False


def test_channel_mapping_is_explicit_and_reported():
    report = compare_transient_trajectories(
        {"BUS 1 V pu": series([0, 1], [1, 1])},
        {"Bus1:Voltage": series([0, 1], [1, 1])},
        channel_map={"BUS 1 V pu": "Bus1:Voltage"},
    )
    assert report.channels[0].candidate_channel == "Bus1:Voltage"


@pytest.mark.parametrize(
    "times,values",
    [
        ([0, 1, 1], [1, 1, 1]),
        ([1, 0], [1, 1]),
        ([0, float("nan")], [1, 1]),
        ([0, 1], [1]),
    ],
)
def test_rejects_invalid_time_series(times, values):
    with pytest.raises(TrajectoryValidationError):
        compare_transient_trajectories(
            {"x": series(times, values)}, {"x": series([0, 1], [1, 1])}
        )


def test_rejects_no_overlap_and_never_extrapolates():
    with pytest.raises(TrajectoryValidationError, match="no common time"):
        compare_transient_trajectories(
            {"x": series([0, 1], [1, 1])},
            {"x": series([2, 3], [1, 1])},
        )


def test_rejects_incompatible_units_and_missing_channels():
    with pytest.raises(TrajectoryValidationError, match="Incompatible units"):
        compare_transient_trajectories(
            {"x": series([0, 1], [1, 1], "MW")},
            {"x": series([0, 1], [1, 1], "Hz")},
        )
    with pytest.raises(TrajectoryValidationError, match="missing"):
        compare_transient_trajectories(
            {"x": series([0, 1], [1, 1])},
            {"y": series([0, 1], [1, 1])},
            channel_map={"x": "x"},
        )


def test_rejects_duplicate_engine_names_and_non_one_to_one_mapping():
    with pytest.raises(TrajectoryValidationError, match="must differ"):
        compare_transient_trajectories(
            {"x": series([0, 1], [1, 1])},
            {"x": series([0, 1], [1, 1])},
            reference_engine="same",
            candidate_engine="same",
        )
    with pytest.raises(TrajectoryValidationError, match="one-to-one"):
        compare_transient_trajectories(
            {"x": series([0, 1], [1, 1]), "y": series([0, 1], [1, 1])},
            {"z": series([0, 1], [1, 1])},
            channel_map={"x": "z", "y": "z"},
        )


def test_report_is_json_serializable():
    import json

    report = compare_transient_trajectories(
        {"x": series([0, 1], [1, 1])},
        {"x": series([0, 1], [1, 1])},
    )
    encoded = json.dumps(report.to_dict())
    assert '"status": "not_evaluated"' in encoded



def test_time_weighted_metrics_do_not_depend_on_export_sample_density():
    coarse = compare_transient_trajectories(
        {"x": series([0.0, 1.0], [0.0, 0.0])},
        {"x": series([0.0, 1.0], [0.0, 1.0])},
    )
    dense = compare_transient_trajectories(
        {"x": series([0.0, 0.25, 0.5, 0.75, 1.0], [0.0] * 5)},
        {"x": series([0.0, 0.5, 1.0], [0.0, 0.5, 1.0])},
    )
    assert dense.channels[0].mae == pytest.approx(coarse.channels[0].mae)
    assert dense.channels[0].rmse == pytest.approx(coarse.channels[0].rmse)


def test_partial_tolerance_coverage_is_not_reported_as_passed():
    report = compare_transient_trajectories(
        {
            "x": series([0.0, 1.0], [1.0, 1.0]),
            "y": series([0.0, 1.0], [2.0, 2.0]),
        },
        {
            "x": series([0.0, 1.0], [1.0, 1.0]),
            "y": series([0.0, 1.0], [2.0, 2.0]),
        },
        tolerances={"x": 0.01},
    )
    assert report.status == "partially_evaluated"



def test_rejects_per_unit_comparison_without_matching_verified_bases():
    with pytest.raises(TrajectoryValidationError, match="verified base_id"):
        compare_transient_trajectories(
            {"x": series([0, 1], [1, 1], base_id=None)},
            {"x": series([0, 1], [1, 1], base_id=None)},
        )
    with pytest.raises(TrajectoryValidationError, match="verified base_id"):
        compare_transient_trajectories(
            {"x": series([0, 1], [1, 1], base_id="100-MVA-230-kV")},
            {"x": series([0, 1], [1, 1], base_id="100-MVA-115-kV")},
        )


def test_rejects_tolerance_for_unselected_channel():
    with pytest.raises(TrajectoryValidationError, match="not being compared"):
        compare_transient_trajectories(
            {"x": series([0, 1], [1, 1])},
            {"x": series([0, 1], [1, 1])},
            tolerances={"unmapped-channel": 0.01},
        )


def test_rejects_unit_normalization_overflow():
    with pytest.raises(TrajectoryValidationError, match="normalization overflowed"):
        compare_transient_trajectories(
            {"x": series([0, 1], [1e308, 1e308], unit="rad")},
            {"x": series([0, 1], [1e308, 1e308], unit="deg")},
        )


def test_rejects_metric_overflow():
    with pytest.raises(TrajectoryValidationError, match="Metric calculation overflowed"):
        compare_transient_trajectories(
            {"x": series([0, 1], [1e308, 1e308], unit="MW")},
            {"x": series([0, 1], [-1e308, -1e308], unit="MW")},
        )



def test_rejects_missing_unit_and_invalid_channel_identifiers():
    with pytest.raises(TrajectoryValidationError, match="no unit"):
        compare_transient_trajectories(
            {"x": ChannelSeries([0, 1], [1, 1], None)},
            {"x": series([0, 1], [1, 1])},
        )
    with pytest.raises(TrajectoryValidationError, match="identifiers"):
        compare_transient_trajectories(
            {"": series([0, 1], [1, 1])},
            {"": series([0, 1], [1, 1])},
        )



@pytest.mark.parametrize(
    "invalid_series",
    [
        ChannelSeries([0, True], [1, 1], "MW"),
        ChannelSeries([0, 1], [1, False], "MW"),
        ChannelSeries("01", [1, 1], "MW"),
        ChannelSeries([0, 1], "11", "MW"),
        {"time_s": [0, 1], "values": [1, 1], "unit": "MW"},
    ],
)
def test_rejects_boolean_and_malformed_channel_samples(invalid_series):
    with pytest.raises(TrajectoryValidationError):
        compare_transient_trajectories(
            {"x": invalid_series},
            {"x": series([0, 1], [1, 1], "MW")},
        )


def test_rejects_non_mapping_engine_outputs_and_channel_map():
    with pytest.raises(TrajectoryValidationError, match="channel mappings"):
        compare_transient_trajectories(
            [series([0, 1], [1, 1], "MW")],
            {"x": series([0, 1], [1, 1], "MW")},
        )
    with pytest.raises(TrajectoryValidationError, match="channel_map"):
        compare_transient_trajectories(
            {"x": series([0, 1], [1, 1], "MW")},
            {"x": series([0, 1], [1, 1], "MW")},
            channel_map=[("x", "x")],
        )


def test_rejects_boolean_tolerances():
    with pytest.raises(TrajectoryValidationError, match="not booleans"):
        compare_transient_trajectories(
            {"x": series([0, 1], [1, 1], "MW")},
            {"x": series([0, 1], [1, 1], "MW")},
            tolerances={"x": True},
        )



def test_time_weighted_metrics_integrate_zero_crossings_exactly():
    report = compare_transient_trajectories(
        {"x": series([0.0, 2.0], [0.0, 0.0], "MW")},
        {"x": series([0.0, 2.0], [1.0, -1.0], "MW")},
    )
    metric = report.channels[0]
    assert metric.mae == pytest.approx(0.5)
    assert metric.rmse == pytest.approx((1.0 / 3.0) ** 0.5)
    assert metric.bias == pytest.approx(0.0)


@pytest.mark.parametrize("tolerance", [-0.1, float("nan"), float("inf")])
def test_rejects_negative_or_non_finite_tolerances(tolerance):
    with pytest.raises(TrajectoryValidationError, match="finite, non-negative"):
        compare_transient_trajectories(
            {"x": series([0.0, 1.0], [1.0, 1.0], "MW")},
            {"x": series([0.0, 1.0], [1.0, 1.0], "MW")},
            tolerances={"x": tolerance},
        )



def test_converts_common_voltage_current_and_apparent_power_units():
    reference = {
        "bus:1:voltage": series([0.0, 1.0], [230.0, 231.0], "kV"),
        "line:1:current": series([0.0, 1.0], [1.0, 1.2], "kA"),
        "transformer:1:power": series([0.0, 1.0], [10.0, 12.0], "MVA"),
    }
    candidate = {
        "bus:1:voltage": series([0.0, 1.0], [230000.0, 231000.0], "V"),
        "line:1:current": series([0.0, 1.0], [1000.0, 1200.0], "A"),
        "transformer:1:power": series([0.0, 1.0], [10000.0, 12000.0], "kVA"),
    }
    report = compare_transient_trajectories(
        reference,
        candidate,
        tolerances={
            "bus:1:voltage": 1e-12,
            "line:1:current": 1e-12,
            "transformer:1:power": 1e-12,
        },
    )
    assert report.status == "passed"
    assert [item.unit for item in report.channels] == ["kV", "kA", "MVA"]
    assert all(item.max_absolute_error == pytest.approx(0.0) for item in report.channels)
