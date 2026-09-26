from __future__ import annotations

from b2g_agent.collaboration.models import ResearchScenario, ScenarioParameters
from b2g_agent.collaboration.simulator import ResidentialCommunitySimulator


def test_vertical_scenario_detects_baseline_constraints() -> None:
    run = ResidentialCommunitySimulator().run(ScenarioParameters(), trigger="test")

    assert run.metrics.feasible is False
    assert run.metrics.maximum_line_loading_pct > 100.0
    assert run.metrics.maximum_transformer_loading_pct > 100.0
    assert len(run.timeseries) == 24


def test_joint_building_grid_plan_can_become_feasible() -> None:
    simulator = ResidentialCommunitySimulator()
    baseline = simulator.run(ScenarioParameters(), trigger="baseline")
    joint = simulator.run(
        ScenarioParameters(
            retrofit_level="standard",
            demand_response_pct=12,
            transformer_capacity_kva=500,
            line_capacity_kw=500,
        ),
        trigger="joint plan",
    )

    assert joint.metrics.feasible is True
    assert joint.metrics.feeder_peak_kw < baseline.metrics.feeder_peak_kw
    assert joint.metrics.minimum_voltage_pu > baseline.metrics.minimum_voltage_pu


def test_demand_response_scenario_establishes_baseline_and_delivers_target() -> None:
    simulator = ResidentialCommunitySimulator()
    run = simulator.run(
        ScenarioParameters(
            building_count=1,
            target_bus="bus_6",
            transformer_capacity_kva=500,
            line_capacity_kw=500,
            baseline_method="weather_adjusted",
            dr_target_kw_per_building=0.8,
            max_rebound_pct=20,
        ),
        trigger="test demand response enrollment",
        scenario_id=ResearchScenario.DEMAND_RESPONSE,
    )

    assert run.scenario_id == ResearchScenario.DEMAND_RESPONSE
    assert run.metrics.baseline_peak_kw_per_building > 0
    assert run.metrics.dr_delivery_pct >= 90
    assert run.metrics.rebound_pct <= 20
    assert run.metrics.feasible is True
    assert sum(point.event_active for point in run.timeseries) == 3


def test_excessive_demand_response_commitment_is_rejected() -> None:
    run = ResidentialCommunitySimulator().run(
        ScenarioParameters(
            building_count=1,
            target_bus="bus_6",
            transformer_capacity_kva=500,
            line_capacity_kw=500,
            dr_target_kw_per_building=4.5,
        ),
        trigger="overcommitment",
        scenario_id=ResearchScenario.DEMAND_RESPONSE,
    )

    assert run.metrics.dr_delivery_pct < 90
    assert run.metrics.feasible is False
