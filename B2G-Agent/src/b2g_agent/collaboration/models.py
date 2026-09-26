from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class EngineerRole(str, Enum):
    BUILDING = "building_engineer"
    POWER = "distribution_power_engineer"


class ResearchScenario(str, Enum):
    GRID_UPGRADE = "distribution_grid_upgrade"
    DEMAND_RESPONSE = "demand_response_service"


class RetrofitLevel(str, Enum):
    NONE = "none"
    STANDARD = "standard"
    DEEP = "deep"


class BaselineMethod(str, Enum):
    RECENT_TEN_DAY = "recent_10_day_average"
    WEATHER_ADJUSTED = "weather_adjusted"
    MATCHED_DAY = "matched_day"


class SimulationScope(str, Enum):
    NONE = "none"
    BUILDING = "building"
    GRID = "grid"
    COUPLED = "coupled"


class ScenarioParameters(BaseModel):
    """Validated shared state shared by both research scenarios."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    cooling_setpoint_c: float = Field(default=24.0, ge=20.0, le=28.0)
    building_count: int = Field(default=80, ge=1, le=300)
    retrofit_level: RetrofitLevel = RetrofitLevel.NONE
    pv_kw_per_building: float = Field(default=1.5, ge=0.0, le=15.0)
    demand_response_pct: float = Field(default=0.0, ge=0.0, le=35.0)
    target_bus: Literal["bus_3", "bus_4", "bus_5", "bus_6", "bus_7", "bus_8"] = "bus_8"
    transformer_capacity_kva: float = Field(default=350.0, ge=250.0, le=1500.0)
    line_capacity_kw: float = Field(default=330.0, ge=200.0, le=1500.0)
    baseline_method: BaselineMethod = BaselineMethod.WEATHER_ADJUSTED
    baseline_adjustment_pct: float = Field(default=0.0, ge=-15.0, le=15.0)
    dr_event_start_hour: int = Field(default=16, ge=0, le=23)
    dr_event_duration_hours: int = Field(default=3, ge=1, le=6)
    dr_target_kw_per_building: float = Field(default=0.8, ge=0.0, le=5.0)
    max_rebound_pct: float = Field(default=20.0, ge=0.0, le=50.0)


class TimeseriesPoint(BaseModel):
    hour: int
    building_load_kw: float
    pv_generation_kw: float
    community_net_load_kw: float
    feeder_kw: float
    min_voltage_pu: float
    line_loading_pct: float
    transformer_loading_pct: float
    baseline_building_load_kw: float = 0.0
    dr_reduction_kw: float = 0.0
    event_active: bool = False


class MetricSummary(BaseModel):
    peak_building_load_kw: float
    peak_community_net_load_kw: float
    feeder_peak_kw: float
    minimum_voltage_pu: float
    maximum_line_loading_pct: float
    maximum_transformer_loading_pct: float
    voltage_violation_hours: int
    line_overload_hours: int
    transformer_overload_hours: int
    daily_building_energy_kwh: float
    daily_grid_energy_kwh: float
    estimated_capital_cost_kusd: float
    building_satisfaction_score: float
    grid_reliability_score: float
    feasible: bool
    baseline_peak_kw_per_building: float = 0.0
    event_baseline_energy_kwh: float = 0.0
    delivered_reduction_kw_per_building: float = 0.0
    dr_delivery_pct: float = 0.0
    rebound_peak_kw_per_building: float = 0.0
    rebound_pct: float = 0.0
    baseline_confidence_score: float = 0.0


class SimulationRun(BaseModel):
    run_id: str
    label: str
    trigger: str
    scope: SimulationScope
    scenario_id: ResearchScenario = ResearchScenario.GRID_UPGRADE
    backend: str = "deterministic-vertical-scenario"
    created_at: datetime = Field(default_factory=utc_now)
    parameters: ScenarioParameters
    metrics: MetricSummary
    timeseries: list[TimeseriesPoint]
    findings: list[str] = Field(default_factory=list)


class ChatMessage(BaseModel):
    message_id: str
    speaker: Literal["user", "mediator", "npc", "system"]
    role: EngineerRole | None = None
    text: str
    created_at: datetime = Field(default_factory=utc_now)


class MediatorPlan(BaseModel):
    """Structured, validated result from the LLM planning pass."""

    model_config = ConfigDict(extra="forbid")

    intent: Literal["discuss", "clarify", "propose", "simulate", "finalize"]
    interpreted_intent: str
    translation_for_counterpart: str
    requested_changes: dict[str, Any] = Field(default_factory=dict)
    simulation_scope: SimulationScope = SimulationScope.NONE
    needs_confirmation: bool = False
    clarification_question: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class MediatedReply(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mediator_message: str
    counterpart_message: str
    next_question: str
    decision_note: str | None = None


class TurnResult(BaseModel):
    plan: MediatorPlan
    messages: list[ChatMessage]
    parameters: ScenarioParameters
    simulation_run: SimulationRun | None = None
    decision_ledger: list[str] = Field(default_factory=list)
    llm_backend: str


class FinalPlan(BaseModel):
    selected_run: SimulationRun
    status: Literal["ready", "needs_revision"]
    summary: str
    accepted_decisions: list[str]
    unresolved_items: list[str]
    report_markdown: str


class SessionSnapshot(BaseModel):
    session_id: str
    user_role: EngineerRole
    counterpart_role: EngineerRole
    scenario_id: ResearchScenario
    parameters: ScenarioParameters
    messages: list[ChatMessage]
    runs: list[SimulationRun]
    decision_ledger: list[str]
    created_at: datetime
    updated_at: datetime
    llm_backend: str
