from __future__ import annotations

import math
from uuid import uuid4

from b2g_agent.collaboration.models import (
    BaselineMethod,
    MetricSummary,
    ResearchScenario,
    RetrofitLevel,
    ScenarioParameters,
    SimulationRun,
    SimulationScope,
    TimeseriesPoint,
)


class ResidentialCommunitySimulator:
    """Deterministic vertical-scenario backend for auditable local demonstrations.

    The equations are deliberately transparent. They are not presented as a
    calibrated replacement for EnergyPlus or OpenDSS; they provide a stable
    end-to-end backend while MCP adapters are connected and validated.
    """

    backend_name = "deterministic-vertical-scenario"

    def run(
        self,
        parameters: ScenarioParameters,
        *,
        trigger: str,
        scope: SimulationScope = SimulationScope.COUPLED,
        label: str | None = None,
        scenario_id: ResearchScenario | str = ResearchScenario.GRID_UPGRADE,
    ) -> SimulationRun:
        normalized = ResearchScenario(scenario_id)
        if normalized == ResearchScenario.DEMAND_RESPONSE:
            timeseries = self._demand_response_timeseries(parameters)
            metrics = self._demand_response_metrics(parameters, timeseries)
            backend = "deterministic-demand-response-baseline"
            findings = self._demand_response_findings(parameters, metrics)
        else:
            timeseries = self._timeseries(parameters)
            metrics = self._metrics(parameters, timeseries)
            backend = self.backend_name
            findings = self._findings(metrics)
        run_id = uuid4().hex[:10]
        return SimulationRun(
            run_id=run_id,
            label=label or f"Scenario {run_id[:5]}",
            trigger=trigger,
            scope=scope,
            scenario_id=normalized,
            backend=backend,
            parameters=parameters.model_copy(deep=True),
            metrics=metrics,
            timeseries=timeseries,
            findings=findings,
        )

    def _timeseries(self, parameters: ScenarioParameters) -> list[TimeseriesPoint]:
        retrofit_multiplier = {
            RetrofitLevel.NONE: 1.0,
            RetrofitLevel.STANDARD: 0.88,
            RetrofitLevel.DEEP: 0.72,
        }[parameters.retrofit_level]
        setpoint_multiplier = max(0.55, 1.0 + (24.0 - parameters.cooling_setpoint_c) * 0.18)
        bus_distance = {
            "bus_3": 0.75,
            "bus_4": 0.85,
            "bus_5": 0.95,
            "bus_6": 1.05,
            "bus_7": 1.15,
            "bus_8": 1.25,
        }[parameters.target_bus]

        points: list[TimeseriesPoint] = []
        for hour in range(24):
            morning = _gaussian(hour, 7.0, 2.2)
            evening = _gaussian(hour, 20.0, 2.8)
            afternoon = _gaussian(hour, 16.0, 4.0)

            appliance_per_building = 0.42 + 0.18 * morning + 0.40 * evening
            cooling_per_building = 1.65 * afternoon * setpoint_multiplier * retrofit_multiplier
            envelope_base = 0.18 * retrofit_multiplier
            gross_per_building = appliance_per_building + cooling_per_building + envelope_base

            if 15 <= hour <= 20:
                gross_per_building *= 1.0 - parameters.demand_response_pct / 100.0
            elif hour == 21:
                gross_per_building *= 1.0 + parameters.demand_response_pct / 300.0

            pv_shape = _gaussian(hour, 13.0, 3.0) if 6 <= hour <= 19 else 0.0
            pv_kw = parameters.building_count * parameters.pv_kw_per_building * pv_shape
            building_load_kw = parameters.building_count * gross_per_building
            community_net_kw = max(0.0, building_load_kw - pv_kw)

            existing_feeder_kw = 142.0 + 18.0 * morning + 27.0 * evening + 12.0 * afternoon
            feeder_kw = existing_feeder_kw + community_net_kw
            line_loading = feeder_kw / parameters.line_capacity_kw * 100.0
            transformer_loading = feeder_kw / 0.95 / parameters.transformer_capacity_kva * 100.0
            min_voltage = 1.0 - (feeder_kw / 100.0) * 0.0115 * bus_distance

            points.append(
                TimeseriesPoint(
                    hour=hour,
                    building_load_kw=round(building_load_kw, 3),
                    pv_generation_kw=round(pv_kw, 3),
                    community_net_load_kw=round(community_net_kw, 3),
                    feeder_kw=round(feeder_kw, 3),
                    min_voltage_pu=round(min_voltage, 5),
                    line_loading_pct=round(line_loading, 3),
                    transformer_loading_pct=round(transformer_loading, 3),
                )
            )
        return points

    def _metrics(
        self,
        parameters: ScenarioParameters,
        points: list[TimeseriesPoint],
    ) -> MetricSummary:
        minimum_voltage = min(point.min_voltage_pu for point in points)
        maximum_line_loading = max(point.line_loading_pct for point in points)
        maximum_transformer_loading = max(point.transformer_loading_pct for point in points)
        voltage_violations = sum(point.min_voltage_pu < 0.95 for point in points)
        line_overloads = sum(point.line_loading_pct > 100.0 for point in points)
        transformer_overloads = sum(point.transformer_loading_pct > 100.0 for point in points)

        retrofit_cost_per_building = {
            RetrofitLevel.NONE: 0.0,
            RetrofitLevel.STANDARD: 18.0,
            RetrofitLevel.DEEP: 42.0,
        }[parameters.retrofit_level]
        retrofit_cost = retrofit_cost_per_building * parameters.building_count
        pv_cost = parameters.pv_kw_per_building * parameters.building_count * 2.1
        demand_response_cost = parameters.demand_response_pct * parameters.building_count * 0.08
        transformer_cost = max(0.0, parameters.transformer_capacity_kva - 350.0) * 0.35
        line_cost = max(0.0, parameters.line_capacity_kw - 330.0) * 0.22
        estimated_cost = retrofit_cost + pv_cost + demand_response_cost + transformer_cost + line_cost

        retrofit_bonus = {
            RetrofitLevel.NONE: 0.0,
            RetrofitLevel.STANDARD: 3.0,
            RetrofitLevel.DEEP: 6.0,
        }[parameters.retrofit_level]
        building_score = 100.0 - abs(parameters.cooling_setpoint_c - 23.5) * 10.0
        building_score += retrofit_bonus - parameters.demand_response_pct * 0.35
        building_score = max(0.0, min(100.0, building_score))

        grid_score = 100.0
        grid_score -= voltage_violations * 7.0
        grid_score -= line_overloads * 8.0
        grid_score -= transformer_overloads * 8.0
        grid_score -= max(0.0, maximum_line_loading - 85.0) * 0.3
        grid_score -= max(0.0, maximum_transformer_loading - 85.0) * 0.3
        grid_score = max(0.0, min(100.0, grid_score))

        feasible = voltage_violations == 0 and line_overloads == 0 and transformer_overloads == 0
        return MetricSummary(
            peak_building_load_kw=round(max(point.building_load_kw for point in points), 3),
            peak_community_net_load_kw=round(max(point.community_net_load_kw for point in points), 3),
            feeder_peak_kw=round(max(point.feeder_kw for point in points), 3),
            minimum_voltage_pu=round(minimum_voltage, 5),
            maximum_line_loading_pct=round(maximum_line_loading, 3),
            maximum_transformer_loading_pct=round(maximum_transformer_loading, 3),
            voltage_violation_hours=voltage_violations,
            line_overload_hours=line_overloads,
            transformer_overload_hours=transformer_overloads,
            daily_building_energy_kwh=round(sum(point.building_load_kw for point in points), 3),
            daily_grid_energy_kwh=round(sum(point.feeder_kw for point in points), 3),
            estimated_capital_cost_kusd=round(estimated_cost, 1),
            building_satisfaction_score=round(building_score, 1),
            grid_reliability_score=round(grid_score, 1),
            feasible=feasible,
        )

    def _demand_response_timeseries(self, parameters: ScenarioParameters) -> list[TimeseriesPoint]:
        method_multiplier = {
            BaselineMethod.RECENT_TEN_DAY: 1.0,
            BaselineMethod.WEATHER_ADJUSTED: 1.03,
            BaselineMethod.MATCHED_DAY: 0.98,
        }[parameters.baseline_method]
        adjustment = 1.0 + parameters.baseline_adjustment_pct / 100.0
        setpoint_multiplier = max(0.65, 1.0 + (24.0 - parameters.cooling_setpoint_c) * 0.15)
        bus_distance = {
            "bus_3": 0.75,
            "bus_4": 0.85,
            "bus_5": 0.95,
            "bus_6": 1.05,
            "bus_7": 1.15,
            "bus_8": 1.25,
        }[parameters.target_bus]
        event_hours = {
            (parameters.dr_event_start_hour + offset) % 24
            for offset in range(parameters.dr_event_duration_hours)
        }
        rebound_hour = (
            parameters.dr_event_start_hour + parameters.dr_event_duration_hours
        ) % 24
        expected_delivery = 0.0
        points: list[TimeseriesPoint] = []

        for hour in range(24):
            morning = _gaussian(hour, 8.0, 2.1)
            afternoon = _gaussian(hour, 16.5, 3.2)
            evening = _gaussian(hour, 20.0, 2.4)
            cooling = 1.55 * afternoon * setpoint_multiplier
            native_per_building = 0.62 + 0.24 * morning + cooling + 0.32 * evening
            baseline_per_building = native_per_building * method_multiplier * adjustment
            delivered_per_building = 0.0

            if hour in event_hours:
                curtailable_per_building = 0.22 + cooling * 0.68
                delivered_per_building = min(
                    parameters.dr_target_kw_per_building,
                    curtailable_per_building,
                    baseline_per_building * 0.48,
                )
                expected_delivery += delivered_per_building

            actual_per_building = baseline_per_building - delivered_per_building
            if hour == rebound_hour and parameters.dr_event_duration_hours:
                average_delivery = expected_delivery / parameters.dr_event_duration_hours
                actual_per_building += average_delivery * 0.15

            baseline_load_kw = parameters.building_count * baseline_per_building
            actual_load_kw = parameters.building_count * actual_per_building
            reduction_kw = parameters.building_count * delivered_per_building
            existing_feeder_kw = 142.0 + 18.0 * morning + 27.0 * evening + 12.0 * afternoon
            feeder_kw = existing_feeder_kw + actual_load_kw
            line_loading = feeder_kw / parameters.line_capacity_kw * 100.0
            transformer_loading = feeder_kw / 0.95 / parameters.transformer_capacity_kva * 100.0
            min_voltage = 1.0 - (feeder_kw / 100.0) * 0.0115 * bus_distance

            points.append(
                TimeseriesPoint(
                    hour=hour,
                    building_load_kw=round(actual_load_kw, 3),
                    baseline_building_load_kw=round(baseline_load_kw, 3),
                    dr_reduction_kw=round(reduction_kw, 3),
                    event_active=hour in event_hours,
                    pv_generation_kw=0.0,
                    community_net_load_kw=round(actual_load_kw, 3),
                    feeder_kw=round(feeder_kw, 3),
                    min_voltage_pu=round(min_voltage, 5),
                    line_loading_pct=round(line_loading, 3),
                    transformer_loading_pct=round(transformer_loading, 3),
                )
            )
        return points

    def _demand_response_metrics(
        self,
        parameters: ScenarioParameters,
        points: list[TimeseriesPoint],
    ) -> MetricSummary:
        event_points = [point for point in points if point.event_active]
        rebound_hour = (
            parameters.dr_event_start_hour + parameters.dr_event_duration_hours
        ) % 24
        rebound_point = next(point for point in points if point.hour == rebound_hour)
        building_count = parameters.building_count
        target_energy = (
            parameters.dr_target_kw_per_building
            * building_count
            * parameters.dr_event_duration_hours
        )
        delivered_energy = sum(point.dr_reduction_kw for point in event_points)
        delivery_pct = 100.0 if target_energy == 0 else delivered_energy / target_energy * 100.0
        delivered_per_building = (
            delivered_energy / building_count / parameters.dr_event_duration_hours
        )
        rebound_per_building = max(
            0.0,
            (rebound_point.building_load_kw - rebound_point.baseline_building_load_kw)
            / building_count,
        )
        rebound_pct = (
            0.0
            if parameters.dr_target_kw_per_building == 0
            else rebound_per_building / parameters.dr_target_kw_per_building * 100.0
        )
        confidence = {
            BaselineMethod.RECENT_TEN_DAY: 86.0,
            BaselineMethod.WEATHER_ADJUSTED: 94.0,
            BaselineMethod.MATCHED_DAY: 90.0,
        }[parameters.baseline_method]
        confidence = max(0.0, confidence - abs(parameters.baseline_adjustment_pct) * 0.8)
        minimum_voltage = min(point.min_voltage_pu for point in points)
        maximum_line_loading = max(point.line_loading_pct for point in points)
        maximum_transformer_loading = max(point.transformer_loading_pct for point in points)
        voltage_violations = sum(point.min_voltage_pu < 0.95 for point in points)
        line_overloads = sum(point.line_loading_pct > 100.0 for point in points)
        transformer_overloads = sum(point.transformer_loading_pct > 100.0 for point in points)
        feasible = (
            confidence >= 80.0
            and delivery_pct >= 90.0
            and rebound_pct <= parameters.max_rebound_pct
            and voltage_violations == 0
            and line_overloads == 0
            and transformer_overloads == 0
        )
        building_score = max(
            0.0,
            min(
                100.0,
                100.0
                - parameters.dr_target_kw_per_building * 7.0
                - rebound_pct * 0.25
                - abs(parameters.cooling_setpoint_c - 23.5) * 4.0,
            ),
        )
        grid_score = max(
            0.0,
            min(
                100.0,
                delivery_pct
                - max(0.0, maximum_line_loading - 90.0)
                - max(0.0, maximum_transformer_loading - 90.0),
            ),
        )
        estimated_cost = 12.0 + building_count * parameters.dr_target_kw_per_building * 8.0

        return MetricSummary(
            peak_building_load_kw=round(max(point.building_load_kw for point in points), 3),
            peak_community_net_load_kw=round(max(point.community_net_load_kw for point in points), 3),
            feeder_peak_kw=round(max(point.feeder_kw for point in points), 3),
            minimum_voltage_pu=round(minimum_voltage, 5),
            maximum_line_loading_pct=round(maximum_line_loading, 3),
            maximum_transformer_loading_pct=round(maximum_transformer_loading, 3),
            voltage_violation_hours=voltage_violations,
            line_overload_hours=line_overloads,
            transformer_overload_hours=transformer_overloads,
            daily_building_energy_kwh=round(sum(point.building_load_kw for point in points), 3),
            daily_grid_energy_kwh=round(sum(point.feeder_kw for point in points), 3),
            estimated_capital_cost_kusd=round(estimated_cost, 1),
            building_satisfaction_score=round(building_score, 1),
            grid_reliability_score=round(grid_score, 1),
            feasible=feasible,
            baseline_peak_kw_per_building=round(
                max(point.baseline_building_load_kw for point in points) / building_count,
                3,
            ),
            event_baseline_energy_kwh=round(
                sum(point.baseline_building_load_kw for point in event_points),
                3,
            ),
            delivered_reduction_kw_per_building=round(delivered_per_building, 3),
            dr_delivery_pct=round(delivery_pct, 1),
            rebound_peak_kw_per_building=round(rebound_per_building, 3),
            rebound_pct=round(rebound_pct, 1),
            baseline_confidence_score=round(confidence, 1),
        )

    @staticmethod
    def _findings(metrics: MetricSummary) -> list[str]:
        findings = [
            f"Minimum voltage is {metrics.minimum_voltage_pu:.3f} p.u.",
            f"Maximum line loading is {metrics.maximum_line_loading_pct:.1f}%.",
            f"Maximum transformer loading is {metrics.maximum_transformer_loading_pct:.1f}%.",
        ]
        if metrics.feasible:
            findings.append("All three hard grid constraints are satisfied in this scenario.")
        else:
            if metrics.voltage_violation_hours:
                findings.append(f"Voltage is below 0.95 p.u. for {metrics.voltage_violation_hours} hour(s).")
            if metrics.line_overload_hours:
                findings.append(f"The feeder line is overloaded for {metrics.line_overload_hours} hour(s).")
            if metrics.transformer_overload_hours:
                findings.append(
                    f"The transformer is overloaded for {metrics.transformer_overload_hours} hour(s)."
                )
        return findings

    @staticmethod
    def _demand_response_findings(
        parameters: ScenarioParameters,
        metrics: MetricSummary,
    ) -> list[str]:
        findings = [
            (
                f"The {parameters.baseline_method.value} baseline peaks at "
                f"{metrics.baseline_peak_kw_per_building:.2f} kW per building."
            ),
            (
                f"Average delivered reduction is {metrics.delivered_reduction_kw_per_building:.2f} "
                f"kW per building ({metrics.dr_delivery_pct:.1f}% of the commitment)."
            ),
            (
                f"Post-event rebound is {metrics.rebound_peak_kw_per_building:.2f} kW per building "
                f"({metrics.rebound_pct:.1f}% of the target)."
            ),
            f"Baseline confidence score is {metrics.baseline_confidence_score:.1f}/100.",
        ]
        if metrics.feasible:
            findings.append("The baseline, delivery, rebound, and grid acceptance checks are satisfied.")
        else:
            if metrics.dr_delivery_pct < 90.0:
                findings.append("The proposed commitment is not fully deliverable during the event window.")
            if metrics.rebound_pct > parameters.max_rebound_pct:
                findings.append("The modeled rebound exceeds the agreed post-event limit.")
            if metrics.baseline_confidence_score < 80.0:
                findings.append("The selected baseline adjustment is not sufficiently credible for enrollment.")
            if metrics.voltage_violation_hours or metrics.line_overload_hours or metrics.transformer_overload_hours:
                findings.append("At least one grid operating constraint is violated.")
        return findings


def _gaussian(hour: float, center: float, width: float) -> float:
    distance = min(abs(hour - center), 24.0 - abs(hour - center))
    return math.exp(-0.5 * (distance / width) ** 2)
