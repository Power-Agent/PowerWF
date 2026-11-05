from types import SimpleNamespace

import pytest

from b2g_agent.contracts import TaskFamily, TaskField, FieldStatus, TimeWindow
from experiments.validate_joint_dispatch import (
    build_business_metrics,
    linearization_error,
    receiver_outcome,
)


def test_linearization_error_is_absent_when_solver_has_no_prediction() -> None:
    assert linearization_error(0.96, None) is None
    assert linearization_error(0.96, 0.95) == pytest.approx(0.01)


def test_physical_hard_gate_precedes_service_delivery() -> None:
    assert receiver_outcome(
        release_eligible=False,
        service_request_satisfied=True,
        total_delivered=800.0,
        service_tolerance=0.5,
        delivery_ratio=1.0,
    ) == ("rejected", False, 0.0)
    assert receiver_outcome(
        release_eligible=True,
        service_request_satisfied=True,
        total_delivered=800.0,
        service_tolerance=0.5,
        delivery_ratio=0.999999999999,
    ) == ("accepted", True, 1.0)


def _grid(*, feeder: float, voltage: float, line: float) -> SimpleNamespace:
    return SimpleNamespace(
        feeder_kw=tuple(feeder for _ in range(96)),
        minimum_voltage_pu=tuple(voltage for _ in range(96)),
        maximum_voltage_pu=tuple(1.0 for _ in range(96)),
        maximum_line_loading_pct=tuple(line for _ in range(96)),
        bus_voltage_pu={1: tuple(voltage for _ in range(96))},
        line_loading_pct={0: tuple(line for _ in range(96))},
    )


def _config() -> dict[str, object]:
    return {
        "linear_grid_constraints": {
            "minimum_voltage_pu": 0.95,
            "maximum_voltage_pu": 1.05,
            "maximum_line_loading_pct": 100.0,
        },
        "validation_tolerances": {"service_kw": 0.5},
        "business_metrics": {
            "renewable_absorption_rule": "request-relative",
            "market_settlement_rule": "pay-as-bid",
        },
    }


def test_baseline_and_response_metrics_use_event_window_and_rebound() -> None:
    specification = SimpleNamespace(
        task_families=[TaskFamily.PEAK_SHAVING],
        windows=[TimeWindow(start="16:00", end="18:00")],
        target_bems=["bem_01"],
        fields={},
    )
    requested = [0.0] * 96
    actual = [0.0] * 96
    for index in range(64, 72):
        requested[index] = 10.0
        actual[index] = 10.0
    baseline_grid = _grid(feeder=100.0, voltage=0.94, line=110.0)
    response_grid = _grid(feeder=90.0, voltage=0.96, line=90.0)
    response_feeder = list(response_grid.feeder_kw)
    response_feeder[72] = 105.0
    response_grid.feeder_kw = tuple(response_feeder)
    metrics = build_business_metrics(
        specification=specification,
        requested_signed=requested,
        actual_service=actual,
        remaining_gap=[0.0] * 8,
        baseline_grid=baseline_grid,
        response_grid=response_grid,
        baseline={"bem_01": {"portfolio_kw": [30.0] * 96}},
        prices=[50.0] * 96,
        config=_config(),
        validated_committed_power=10.0,
        validated_delivery_ratio=1.0,
    )

    assert metrics["active_interval_indices"] == list(range(64, 72))
    assert metrics["feeder_peak_reduction_kw"] == pytest.approx(10.0)
    assert metrics["rebound_kw"] == pytest.approx(5.0)
    assert metrics["baseline_voltage_violation_area_pu_h"] == pytest.approx(0.02)
    assert metrics["voltage_violation_area_pu_h"] == pytest.approx(0.0)
    assert metrics["baseline_maximum_voltage_pu"] == pytest.approx(1.0)
    assert metrics["maximum_voltage_pu"] == pytest.approx(1.0)
    assert metrics["baseline_congestion_violation_area_pct_h"] == pytest.approx(20.0)
    assert metrics["congestion_violation_area_pct_h"] == pytest.approx(0.0)
    assert metrics["counterfactual_baseline_kw"] is None


def test_market_metrics_use_frozen_pay_as_bid_rule() -> None:
    specification = SimpleNamespace(
        task_families=[TaskFamily.MARKET_TRADING],
        windows=[TimeWindow(start="16:00", end="18:00")],
        target_bems=["bem_01"],
        fields={
            "offer_price": TaskField(
                value=50.0,
                unit="USD/MWh",
                status=FieldStatus.DECLARED,
                confidence=1.0,
            )
        },
    )
    requested = [0.0] * 96
    actual = [0.0] * 96
    for index in range(64, 72):
        requested[index] = 10.0
        actual[index] = 10.0
    grid = _grid(feeder=100.0, voltage=1.0, line=50.0)
    metrics = build_business_metrics(
        specification=specification,
        requested_signed=requested,
        actual_service=actual,
        remaining_gap=[0.0] * 8,
        baseline_grid=grid,
        response_grid=grid,
        baseline={"bem_01": {"portfolio_kw": [30.0] * 96}},
        prices=[60.0] * 96,
        config=_config(),
        validated_committed_power=10.0,
        validated_delivery_ratio=1.0,
    )

    assert metrics["accepted_capacity_kw"] == pytest.approx(10.0)
    assert metrics["market_delivered_energy_kwh"] == pytest.approx(20.0)
    assert metrics["settlement_usd"] == pytest.approx(1.0)
    assert metrics["mean_rt_lmp_usd_per_mwh"] == pytest.approx(60.0)


def test_composite_metrics_use_each_family_bound_window() -> None:
    specification = SimpleNamespace(
        task_families=[TaskFamily.RENEWABLE_ABSORPTION, TaskFamily.PEAK_SHAVING],
        windows=[
            TimeWindow(start="11:00", end="13:00"),
            TimeWindow(start="17:00", end="19:00"),
        ],
        target_bems=["bem_01"],
        fields={},
    )
    requested = [0.0] * 96
    actual = [0.0] * 96
    for index in range(44, 52):
        requested[index] = -10.0
        actual[index] = -10.0
    for index in range(68, 76):
        requested[index] = 10.0
        actual[index] = 10.0
    baseline_grid = _grid(feeder=100.0, voltage=1.0, line=50.0)
    response_grid = _grid(feeder=100.0, voltage=1.0, line=50.0)
    feeder = list(response_grid.feeder_kw)
    for index in range(44, 52):
        feeder[index] = 120.0
    for index in range(68, 76):
        feeder[index] = 90.0
    response_grid.feeder_kw = tuple(feeder)
    metrics = build_business_metrics(
        specification=specification,
        requested_signed=requested,
        actual_service=actual,
        remaining_gap=[0.0] * 16,
        baseline_grid=baseline_grid,
        response_grid=response_grid,
        baseline={"bem_01": {"portfolio_kw": [30.0] * 96}},
        prices=[50.0] * 96,
        config=_config(),
        validated_committed_power=10.0,
        validated_delivery_ratio=1.0,
    )

    assert metrics["absorbed_energy_kwh"] == pytest.approx(20.0)
    assert metrics["feeder_peak_kw"] == pytest.approx(90.0)
    assert metrics["feeder_peak_reduction_kw"] == pytest.approx(10.0)
