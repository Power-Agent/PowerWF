from __future__ import annotations

from typing import Any

from b2g_agent.collaboration.models import EngineerRole, ResearchScenario, ScenarioParameters


ROLE_LABELS = {
    EngineerRole.BUILDING: "Building Engineer",
    EngineerRole.POWER: "Distribution Power Engineer",
}


ALLOWED_PARAMETER_DESCRIPTIONS = {
    "cooling_setpoint_c": "Cooling setpoint in degrees Celsius (20-28).",
    "building_count": "Number of participating buildings (1-300).",
    "retrofit_level": "Envelope/HVAC retrofit: none, standard, or deep.",
    "pv_kw_per_building": "Average rooftop PV capacity per building in kW (0-15).",
    "demand_response_pct": "Peak-period demand reduction percentage (0-35).",
    "target_bus": "Connection bus: bus_3 through bus_8.",
    "transformer_capacity_kva": "Substation transformer capacity in kVA (250-1500).",
    "line_capacity_kw": "Feeder thermal capacity proxy in kW (200-1500).",
    "baseline_method": (
        "Demand-response baseline method: recent_10_day_average, weather_adjusted, or matched_day."
    ),
    "baseline_adjustment_pct": "Adjustment applied to the estimated baseline (-15 to 15 percent).",
    "dr_event_start_hour": "Demand-response event start hour in local time (0-23).",
    "dr_event_duration_hours": "Demand-response event duration (1-6 hours).",
    "dr_target_kw_per_building": "Committed load reduction per building in kW (0-5).",
    "max_rebound_pct": "Maximum acceptable post-event rebound as a percentage of the target (0-50).",
}


SCENARIO_PARAMETER_KEYS = {
    ResearchScenario.GRID_UPGRADE: [
        "cooling_setpoint_c",
        "building_count",
        "retrofit_level",
        "pv_kw_per_building",
        "demand_response_pct",
        "target_bus",
        "transformer_capacity_kva",
        "line_capacity_kw",
    ],
    ResearchScenario.DEMAND_RESPONSE: [
        "baseline_method",
        "baseline_adjustment_pct",
        "building_count",
        "cooling_setpoint_c",
        "dr_event_start_hour",
        "dr_event_duration_hours",
        "dr_target_kw_per_building",
        "max_rebound_pct",
        "target_bus",
        "transformer_capacity_kva",
        "line_capacity_kw",
    ],
}


_SCENARIOS: dict[ResearchScenario, dict[str, Any]] = {
    ResearchScenario.GRID_UPGRADE: {
        "title": "Distribution Grid Upgrade",
        "short_title": "Grid upgrade",
        "kicker": "Scenario 01 · Load growth and upgrade",
        "selection_summary": (
            "Coordinate an 80-home expansion, building retrofit choices, and targeted feeder upgrades."
        ),
        "location": "A representative U.S. coastal community",
        "summary": (
            "The study district plans a phased residential expansion and retrofit program. "
            "The building team must preserve occupant comfort and control retrofit cost, "
            "while the distribution team must keep voltage and equipment loading within limits."
        ),
        "room_title": "Distribution Grid Upgrade Room",
        "highlights": [
            {"label": "Homes", "value": "80"},
            {"label": "Connection", "value": "Bus 8"},
            {"label": "Study", "value": "24 hours"},
            {"label": "Voltage floor", "value": "0.95 p.u."},
        ],
        "hard_constraints": [
            "Minimum service voltage must remain at or above 0.95 p.u.",
            "Feeder line loading must not exceed 100%.",
            "Transformer loading must not exceed 100%.",
        ],
        "building_mission": (
            "Define comfort, retrofit, electrification, PV, and demand-flexibility choices."
        ),
        "power_mission": "Define connection, feeder capacity, voltage, and upgrade choices.",
        "building_objectives": [
            "Maintain a defensible cooling setpoint and occupant comfort.",
            "Limit retrofit cost and avoid unnecessary equipment changes.",
            "Use PV and demand flexibility where they improve the joint plan.",
        ],
        "grid_objectives": [
            "Serve the planned buildings without voltage violations.",
            "Avoid thermal overloads and preserve operating margin.",
            "Use targeted upgrades rather than unconstrained expansion.",
        ],
        "simulation_note": (
            "This research prototype uses a deterministic residential-load and radial-feeder model. "
            "EnergyPlus-MCP and PowerMCP are planned adapters and are not called by this release."
        ),
    },
    ResearchScenario.DEMAND_RESPONSE: {
        "title": "Demand Response Service",
        "short_title": "Demand response service",
        "kicker": "Scenario 02 · Baseline and flexibility",
        "selection_summary": (
            "Agree on a defensible building baseline and a reliable load-reduction commitment for a grid event."
        ),
        "location": "A civic building and its distribution feeder",
        "summary": (
            "A civic building is preparing to enroll in a utility demand-response service. "
            "The two engineers must agree on the counterfactual baseline load, event window, achievable "
            "reduction, comfort implications, post-event rebound, and the value delivered to the feeder."
        ),
        "room_title": "Demand Response Enrollment Room",
        "highlights": [
            {"label": "Asset", "value": "1 civic building"},
            {"label": "Baseline", "value": "Weather-adjusted"},
            {"label": "Event", "value": "16:00-19:00"},
            {"label": "Connection", "value": "Bus 6"},
        ],
        "hard_constraints": [
            "Both engineers must approve the baseline method before enrollment.",
            "Delivered event reduction must reach at least 90% of the committed target.",
            "Post-event rebound must remain within the agreed limit.",
            "Voltage and equipment loading must remain within grid limits.",
        ],
        "building_mission": (
            "Defend the building baseline, comfort limits, controllable end uses, event duration, and rebound risk."
        ),
        "power_mission": (
            "Validate baseline credibility, requested reduction, feeder benefit, delivery performance, and reliability."
        ),
        "building_objectives": [
            "Use a baseline that reflects weather and normal operation.",
            "Commit only load that can be reduced without unacceptable comfort impact.",
            "Limit rebound after the event and document operational assumptions.",
        ],
        "grid_objectives": [
            "Obtain measurable reduction during the event window.",
            "Reject inflated baselines and unreliable commitments.",
            "Confirm that the service improves feeder loading without creating a rebound problem.",
        ],
        "simulation_note": (
            "This research prototype compares an explicit counterfactual baseline with a deterministic event-day "
            "load response. It is not a settlement-grade baseline or a calibrated controls model."
        ),
    },
}


def counterpart_role(role: EngineerRole) -> EngineerRole:
    if role == EngineerRole.BUILDING:
        return EngineerRole.POWER
    return EngineerRole.BUILDING


def role_label(role: EngineerRole) -> str:
    return ROLE_LABELS[role]


def normalize_scenario_id(scenario_id: ResearchScenario | str) -> ResearchScenario:
    try:
        return ResearchScenario(scenario_id)
    except ValueError as exc:
        raise ValueError(f"Unknown research scenario: {scenario_id}") from exc


def scenario_catalog() -> dict[str, Any]:
    scenarios = []
    for scenario_id, definition in _SCENARIOS.items():
        scenarios.append(
            {
                "scenario_id": scenario_id.value,
                "title": definition["title"],
                "short_title": definition["short_title"],
                "kicker": definition["kicker"],
                "selection_summary": definition["selection_summary"],
            }
        )
    return {
        "user_roles": _role_payload(ResearchScenario.GRID_UPGRADE),
        "scenarios": scenarios,
    }


def scenario_brief(scenario_id: ResearchScenario | str = ResearchScenario.GRID_UPGRADE) -> dict[str, Any]:
    normalized = normalize_scenario_id(scenario_id)
    definition = _SCENARIOS[normalized]
    keys = SCENARIO_PARAMETER_KEYS[normalized]
    return {
        "scenario_id": normalized.value,
        **definition,
        "user_roles": _role_payload(normalized),
        "allowed_parameters": {key: ALLOWED_PARAMETER_DESCRIPTIONS[key] for key in keys},
        "parameter_keys": keys,
        "defaults": default_parameters(normalized).model_dump(mode="json"),
    }


def default_parameters(scenario_id: ResearchScenario | str) -> ScenarioParameters:
    normalized = normalize_scenario_id(scenario_id)
    if normalized == ResearchScenario.DEMAND_RESPONSE:
        return ScenarioParameters(
            building_count=1,
            cooling_setpoint_c=23.5,
            pv_kw_per_building=0.0,
            target_bus="bus_6",
            transformer_capacity_kva=500.0,
            line_capacity_kw=500.0,
            baseline_method="weather_adjusted",
            baseline_adjustment_pct=0.0,
            dr_event_start_hour=16,
            dr_event_duration_hours=3,
            dr_target_kw_per_building=0.8,
            max_rebound_pct=20.0,
        )
    return ScenarioParameters()


def allowed_parameter_descriptions(scenario_id: ResearchScenario | str) -> dict[str, str]:
    normalized = normalize_scenario_id(scenario_id)
    return {
        key: ALLOWED_PARAMETER_DESCRIPTIONS[key]
        for key in SCENARIO_PARAMETER_KEYS[normalized]
    }


def validate_parameter_changes(
    current: ScenarioParameters,
    changes: dict[str, Any],
    scenario_id: ResearchScenario | str = ResearchScenario.GRID_UPGRADE,
) -> ScenarioParameters:
    allowed = set(SCENARIO_PARAMETER_KEYS[normalize_scenario_id(scenario_id)])
    unknown = sorted(set(changes) - allowed)
    if unknown:
        raise ValueError(f"Unsupported parameter(s) for this scenario: {', '.join(unknown)}")
    payload = current.model_dump(mode="json")
    payload.update(changes)
    return ScenarioParameters.model_validate(payload)


def compact_case_context(
    parameters: ScenarioParameters,
    scenario_id: ResearchScenario | str = ResearchScenario.GRID_UPGRADE,
) -> str:
    values = parameters.model_dump(mode="json")
    keys = SCENARIO_PARAMETER_KEYS[normalize_scenario_id(scenario_id)]
    return "; ".join(f"{name}={values[name]}" for name in keys)


def scenario_title(scenario_id: ResearchScenario | str) -> str:
    return str(_SCENARIOS[normalize_scenario_id(scenario_id)]["title"])


def scenario_room_title(scenario_id: ResearchScenario | str) -> str:
    return str(_SCENARIOS[normalize_scenario_id(scenario_id)]["room_title"])


def _role_payload(scenario_id: ResearchScenario) -> list[dict[str, str]]:
    definition = _SCENARIOS[scenario_id]
    return [
        {
            "id": EngineerRole.BUILDING.value,
            "label": role_label(EngineerRole.BUILDING),
            "mission": definition["building_mission"],
        },
        {
            "id": EngineerRole.POWER.value,
            "label": role_label(EngineerRole.POWER),
            "mission": definition["power_mission"],
        },
    ]
