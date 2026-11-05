"""Generate the frozen 200-episode B2G interaction corpus deterministically."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.contracts import (  # noqa: E402
    ActionType,
    AgentRole,
    Constraint,
    Episode,
    EpisodeDifficulty,
    EpisodeGroundTruth,
    EpisodeSplit,
    FieldStatus,
    Objective,
    PerfectExecutionOracle,
    ReviewVerdict,
    TaskFamily,
    TaskField,
    TaskSpecification,
    TimeWindow,
)


SEED = 2026
GENERATOR_VERSION = "episode-generator-v7"
CASE_ID = "ieee33_10bem_15min"
CASE_DIR = REPO_ROOT / "cases" / CASE_ID
FAMILY_COUNTS = {
    TaskFamily.PEAK_SHAVING: 24,
    TaskFamily.CONGESTION_MANAGEMENT: 24,
    TaskFamily.VOLTAGE_MANAGEMENT: 22,
    TaskFamily.RESERVE_SERVICE: 20,
    TaskFamily.RENEWABLE_ABSORPTION: 20,
    TaskFamily.BASELINE_FLEXIBILITY: 20,
    TaskFamily.MARKET_TRADING: 20,
}
BEMS = [f"bem_{index:02d}" for index in range(1, 11)]
BUSES = [3, 5, 8, 11, 15, 18, 21, 27, 29, 33]
DAYS = [
    "winter_peak_01",
    "winter_peak_02",
    "shoulder_01",
    "shoulder_02",
    "summer_peak_01",
    "summer_peak_02",
    "pv_surplus_01",
    "pv_surplus_02",
    "voltage_risk_01",
    "congestion_risk_01",
]
COMMUNICATION_ASSIGNMENT = (
    ["direct"] * 100
    + ["implicit_unit"] * 34
    + ["missing_field"] * 33
    + ["conflicting_constraints"] * 33
)
random.Random(SEED + 1).shuffle(COMMUNICATION_ASSIGNMENT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset-verified-physics",
        action="store_true",
        help=(
            "Explicitly replace contract-oracle physics statuses with the freshly "
            "generated pending state. Existing numeric files are not deleted."
        ),
    )
    return parser.parse_args()


def require_safe_regeneration(*, reset_verified_physics: bool) -> None:
    contract_path = CASE_DIR / "groundtruth" / "contract_oracles.jsonl"
    if not contract_path.is_file():
        return
    verified = [
        row["episode_id"]
        for row in (
            json.loads(line)
            for line in contract_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
        if row.get("physics_truth_status") == "verified"
    ]
    if verified and not reset_verified_physics:
        raise RuntimeError(
            "Refusing to reset "
            f"{len(verified)} verified numeric-oracle statuses. Re-run with "
            "--reset-verified-physics only when you intend to regenerate and "
            "revalidate every affected numeric oracle."
        )


def targets(index: int) -> tuple[list[str], list[int], str]:
    mode = index % 5
    if mode in {0, 1}:
        selected = [index % 10]
        scope = "single_bem"
    elif mode in {2, 3}:
        selected = sorted({index % 10, (index + 3) % 10, (index + 6) % 10})
        scope = "multi_bem"
    else:
        selected = list(range(10))
        scope = "feeder_wide"
    return [BEMS[item] for item in selected], [BUSES[item] for item in selected], scope


def windows(
    index: int, families: list[TaskFamily]
) -> tuple[list[TimeWindow], str]:
    mode = index % 6
    if mode in {0, 1, 2}:
        selected = [TimeWindow(start="16:00", end="18:00")]
        scope = "single_window"
    elif mode in {3, 4}:
        selected = [
            TimeWindow(start="11:00", end="13:00"),
            TimeWindow(start="17:00", end="19:00"),
        ]
        scope = "multi_window"
    else:
        selected = [TimeWindow(start="00:00", end="23:45")]
        scope = "full_day"

    # Renewable absorption increases net load; the other benchmark families use
    # a reduction request. A shared window cannot carry both directions without
    # making one objective physically undefined, so bind the two families to
    # separate, non-overlapping windows.
    has_absorption = TaskFamily.RENEWABLE_ABSORPTION in families
    has_reduction = any(
        family != TaskFamily.RENEWABLE_ABSORPTION for family in families
    )
    if len(families) > 1 and has_absorption and has_reduction and len(selected) == 1:
        return [
            TimeWindow(start="11:00", end="13:00"),
            TimeWindow(start="17:00", end="19:00"),
        ], "multi_window"

    # Rebound is defined after the final active interval on the same operating
    # day. Reserve-service tasks request rebound explicitly, so a full-day event
    # would make its own ground-truth field undefined.
    if TaskFamily.RESERVE_SERVICE in families and scope == "full_day":
        return [TimeWindow(start="16:00", end="18:00")], "single_window"
    return selected, scope


def family_outputs(families: list[TaskFamily]) -> list[str]:
    mapping = {
        TaskFamily.PEAK_SHAVING: ["feeder_peak_kw", "delivered_reduction_kw", "comfort_violation_minutes"],
        TaskFamily.CONGESTION_MANAGEMENT: ["maximum_line_loading_pct", "overload_intervals"],
        TaskFamily.VOLTAGE_MANAGEMENT: ["minimum_voltage_pu", "maximum_voltage_pu", "violation_intervals"],
        TaskFamily.RESERVE_SERVICE: ["reserved_capacity_kw", "delivery_ratio", "rebound_kw"],
        TaskFamily.RENEWABLE_ABSORPTION: ["pv_curtailment_kwh", "absorbed_energy_kwh"],
        TaskFamily.BASELINE_FLEXIBILITY: ["counterfactual_baseline_kw", "flexibility_lower_kw", "flexibility_upper_kw"],
        TaskFamily.MARKET_TRADING: ["accepted_capacity_kw", "settlement_usd", "delivery_ratio"],
    }
    return sorted({output for family in families for output in mapping[family]})


def action_types(families: list[TaskFamily]) -> list[ActionType]:
    actions = [
        ActionType.LOAD_CASE,
        ActionType.SELECT_BUILDING_MODELS,
        ActionType.LOAD_REFERENCE_PROFILES,
        ActionType.RUN_ENERGYPLUS_BASELINE,
        ActionType.CALIBRATE_RC_MODELS,
        ActionType.SOLVE_JOINT_RC_OPF,
        ActionType.RUN_ENERGYPLUS_EVENT,
        ActionType.ALIGN_15_MIN_TIMESERIES,
        ActionType.SCALE_BEM_PORTFOLIOS,
    ]
    if any(family in {TaskFamily.RENEWABLE_ABSORPTION, TaskFamily.RESERVE_SERVICE, TaskFamily.MARKET_TRADING} for family in families):
        actions.append(ActionType.APPLY_DER_DISPATCH)
    actions.extend([ActionType.MAP_LOADS_TO_BUSES, ActionType.RUN_PANDAPOWER])
    if TaskFamily.MARKET_TRADING in families:
        actions.append(ActionType.CALCULATE_MARKET_SETTLEMENT)
    actions.append(ActionType.EXTRACT_METRICS)
    return actions


def objectives(families: list[TaskFamily], target_kw: float | None) -> list[Objective]:
    mapping = {
        TaskFamily.PEAK_SHAVING: ("feeder_peak_kw", "minimize", None, "kW"),
        TaskFamily.CONGESTION_MANAGEMENT: ("maximum_line_loading_pct", "at_most", 100.0, "%"),
        TaskFamily.VOLTAGE_MANAGEMENT: ("minimum_voltage_pu", "at_least", 0.95, "p.u."),
        TaskFamily.RESERVE_SERVICE: ("reserved_capacity_kw", "at_least", target_kw, "kW"),
        TaskFamily.RENEWABLE_ABSORPTION: ("pv_curtailment_kwh", "minimize", None, "kWh"),
        TaskFamily.BASELINE_FLEXIBILITY: ("baseline_nrmse", "minimize", None, "%"),
        TaskFamily.MARKET_TRADING: ("settlement_usd", "maximize", None, "USD"),
    }
    return [
        Objective(
            name=family.value,
            metric=mapping[family][0],
            operator=mapping[family][1],
            value=mapping[family][2],
            unit=mapping[family][3],
            priority=index + 1,
        )
        for index, family in enumerate(families)
    ]


def instruction(
    families: list[TaskFamily],
    operating_day: str,
    target_bems: list[str],
    buses: list[int],
    event_windows: list[TimeWindow],
    target_kw: float,
    communication: str,
    start_role: AgentRole,
    offer_price: float | None,
    style_index: int,
) -> str:
    family_text = " and ".join(family.value.replace("_", " ") for family in families)
    asset_text = ", ".join(target_bems)
    bus_text = ", ".join(str(bus) for bus in buses)
    window_text = "; ".join(f"{item.start}-{item.end}" for item in event_windows)
    amount = f"{target_kw:.0f} kW"
    day_text = operating_day.replace("_", " ")
    if communication == "implicit_unit":
        amount = f"{target_kw:.0f}"
    elif communication == "missing_field":
        amount = "the required amount"
    if start_role == AgentRole.BUILDING_ENERGY_MANAGER:
        commitment_evidence = (
            "comfort and rebound evidence"
            if TaskFamily.RESERVE_SERVICE in families
            else "comfort evidence"
        )
        offer_openings = [
            (
                f"On the {day_text} operating day, for {family_text}, the portfolio formed by "
                f"{asset_text} at buses {bus_text} is ready to offer an aggregate flexibility "
                f"envelope of {amount} during {window_text}. We need you to review the offer for "
                "network feasibility."
            ),
            (
                f"For {family_text}, please assess our offer from {asset_text} at buses {bus_text} "
                f"on the {day_text} operating day. We can provide an aggregate flexibility envelope "
                f"of {amount} during {window_text}."
            ),
            (
                f"For {family_text}, the portfolio formed by {asset_text} at buses {bus_text} would "
                f"like to submit an aggregate flexibility envelope of {amount} during {window_text} "
                f"on the {day_text} operating day. We need a network decision on this offer."
            ),
            (
                f"On the {day_text} operating day, for {family_text}, please review an offer from "
                f"{asset_text} at buses {bus_text}. Our offer is an aggregate flexibility envelope "
                f"of {amount} during {window_text}."
            ),
            (
                f"For {family_text}, please check this proposal on the {day_text} operating day: "
                f"the portfolio formed by {asset_text} at buses {bus_text} offers an aggregate "
                f"flexibility envelope of {amount} during {window_text}. We need confirmation that "
                "the network can support it."
            ),
            (
                f"For {family_text}, the portfolio formed by {asset_text} at buses {bus_text} can "
                f"make an aggregate flexibility envelope of {amount} available during {window_text} "
                f"on the {day_text} operating day. We'd like a network-feasibility review."
            ),
            (
                f"On the {day_text} operating day, for {family_text}, please evaluate our portfolio "
                f"of {asset_text} at buses {bus_text}. We are offering an aggregate flexibility "
                f"envelope of {amount} during {window_text}."
            ),
            (
                f"For {family_text}, the portfolio formed by {asset_text} at buses {bus_text} needs "
                f"a network decision on an aggregate flexibility envelope of {amount} during "
                f"{window_text} on the {day_text} operating day. We need a clear answer before "
                "finalizing the offer."
            ),
            (
                f"For {family_text}, please look at our proposal on the {day_text} operating day "
                f"from {asset_text} at buses {bus_text}. We'd like to offer an aggregate "
                f"flexibility envelope of {amount} during {window_text}."
            ),
            (
                f"On the {day_text} operating day, for {family_text}, the portfolio formed by "
                f"{asset_text} at buses {bus_text} is asking for clearance on an aggregate "
                f"flexibility envelope of {amount} during {window_text}."
            ),
        ]
        decision_requests = [
            f"Please tell us whether it is accepted or rejected, or return a quantified counteroffer; we'll attach {commitment_evidence} to any commitment.",
            f"We need an accept/reject decision or a quantified counteroffer after the network check. Any commitment will include {commitment_evidence}.",
            f"Could you screen it for network feasibility and send back an acceptance, a rejection, or a quantified counteroffer? {commitment_evidence.capitalize()} will accompany any commitment.",
            f"Please return either an acceptance, a rejection, or a quantified counteroffer after checking the network. We'll provide {commitment_evidence} with any commitment.",
            f"We need you to check the network and respond with an acceptance, rejection, or quantified counteroffer; any commitment will be backed by {commitment_evidence}.",
        ]
        sentence = (
            offer_openings[style_index % len(offer_openings)]
            + " "
            + decision_requests[(style_index // 2) % len(decision_requests)]
        )
        if offer_price is not None:
            price_sentences = [
                f"Our offer price is {offer_price:.0f} USD/MWh.",
                f"We're quoting {offer_price:.0f} USD/MWh for this offer.",
                f"Please use {offer_price:.0f} USD/MWh as the offer price.",
                f"The bid price is {offer_price:.0f} USD/MWh.",
            ]
            sentence += " " + price_sentences[style_index % len(price_sentences)]
    else:
        if TaskFamily.RESERVE_SERVICE in families:
            evidence_request = "report comfort and rebound evidence"
        elif TaskFamily.PEAK_SHAVING in families:
            evidence_request = "report comfort evidence"
        else:
            evidence_request = "confirm that the comfort constraint remains satisfied"
        request_openings = [
            (
                f"On the {day_text} operating day, for {family_text}, please have {asset_text} at "
                f"buses {bus_text} provide {amount} during {window_text}. We need that response for "
                "the scheduled grid operation."
            ),
            (
                f"For {family_text}, please schedule {asset_text} at buses {bus_text} to provide "
                f"{amount} during {window_text} on the {day_text} operating day. We need a clear "
                "commitment for those intervals."
            ),
            (
                f"For {family_text}, please check whether {asset_text} at buses {bus_text} can "
                f"provide {amount} during {window_text} on the {day_text} operating day. I need a "
                "firm answer for the dispatch plan."
            ),
            (
                f"On the {day_text} operating day, for {family_text}, please arrange for "
                f"{asset_text} at buses {bus_text} to provide {amount} during {window_text}. We need "
                "to know what can actually be committed."
            ),
            (
                f"For {family_text}, please use {asset_text} at buses {bus_text} to provide "
                f"{amount} during {window_text} on the {day_text} operating day. I need the response "
                "confirmed before the operating window."
            ),
            (
                f"On the {day_text} operating day, for {family_text}, please make sure {asset_text} "
                f"at buses {bus_text} provide {amount} during {window_text}. We need the feasible "
                "commitment, not just a nominal target."
            ),
            (
                f"For {family_text}, please plan on {asset_text} at buses {bus_text} to provide "
                f"{amount} during {window_text} on the {day_text} operating day. We'd like the "
                "commitment checked before release."
            ),
            (
                f"For {family_text}, please coordinate {asset_text} at buses {bus_text} so they "
                f"provide {amount} during {window_text} on the {day_text} operating day. We need a "
                "validated response for the final plan."
            ),
            (
                f"On the {day_text} operating day, for {family_text}, please ask {asset_text} at "
                f"buses {bus_text} to provide {amount} during {window_text}. I need to know whether "
                "the full request is workable."
            ),
            (
                f"For {family_text}, please confirm a plan in which {asset_text} at buses {bus_text} "
                f"provide {amount} during {window_text} on the {day_text} operating day. We need a "
                "network-safe commitment."
            ),
        ]
        operating_limits = [
            f"Please keep voltage within 0.95-1.05 p.u. and line loading at or below 100%, and {evidence_request}.",
            f"We still need voltage held within 0.95-1.05 p.u. and line loading no higher than 100%; please {evidence_request}.",
            f"Make sure voltage stays inside 0.95-1.05 p.u. and line loading does not exceed 100%; please {evidence_request}.",
            f"The response must respect the 0.95-1.05 p.u. voltage band and the 100% line-loading ceiling, and it must {evidence_request}.",
            f"Use 0.95-1.05 p.u. as the voltage band and 100% as the line-loading limit, and {evidence_request}.",
            f"Please verify that voltage remains between 0.95 and 1.05 p.u. and line loading remains at or below 100%, then {evidence_request}.",
        ]
        sentence = (
            request_openings[style_index % len(request_openings)]
            + " "
            + operating_limits[(style_index // 2) % len(operating_limits)]
        )
    has_absorption = TaskFamily.RENEWABLE_ABSORPTION in families
    has_reduction = any(
        family != TaskFamily.RENEWABLE_ABSORPTION for family in families
    )
    if has_absorption and has_reduction and len(event_windows) == len(families):
        first_family = families[0].value.replace("_", " ")
        second_family = families[1].value.replace("_", " ")
        sentence += (
            f" Specifically, the first listed window is for {first_family}, and the second "
            f"is for {second_family}; the same requested amount applies to each window. "
            "Renewable absorption increases net demand, while the other service reduces "
            "net demand."
        )
    elif has_absorption:
        sentence += (
            " For renewable absorption, increase net demand by the requested amount "
            "during the stated window."
        )
    if len(families) > 1:
        first = families[0].value.replace("_", " ")
        second = families[1].value.replace("_", " ")
        priority_sentences = [
            f"If a trade-off is necessary, treat {first} as the first priority and {second} as the second priority.",
            f"Our priority order is {first} first and {second} second.",
            f"We need {first} handled as the first priority, with {second} second.",
            f"Please rank {first} first and {second} second when making the decision.",
        ]
        sentence += " " + priority_sentences[style_index % len(priority_sentences)]
    if communication == "conflicting_constraints":
        conflict_limit = max(10.0, target_kw * 0.55)
        conflict_sentences = [
            f"At the same time, do not change aggregate load by more than {conflict_limit:.0f} kW; both clauses are stated as highest priority.",
            f"I also need aggregate load change capped at {conflict_limit:.0f} kW, and both this cap and the requested response are highest-priority requirements.",
            f"At the same time, aggregate load must not change by more than {conflict_limit:.0f} kW; treat both requirements as highest priority.",
            f"However, do not change aggregate load by more than {conflict_limit:.0f} kW. This limit and the requested response are both highest priority.",
        ]
        sentence += " " + conflict_sentences[style_index % len(conflict_sentences)]
    return sentence


def make_episode(
    number: int,
    families: list[TaskFamily],
    primary: TaskFamily,
) -> tuple[Episode, EpisodeGroundTruth]:
    index = number - 1
    target_bems, target_buses, spatial_scope = targets(index)
    event_windows, temporal_scope = windows(index, families)
    communication = COMMUNICATION_ASSIGNMENT[index]
    feasibility = "infeasible" if index % 4 == 3 else "feasible"
    target_kw = float(60 + 15 * (index % 13)) * len(target_bems) ** 0.5
    if feasibility == "infeasible":
        target_kw *= 2.4
    target_kw = round(target_kw, 1)
    # The benchmark begins with m^(1), so its reference contract must use the
    # quantity actually visible in m^(1), not the generator's hidden decimal.
    communicated_target_kw = float(f"{target_kw:.0f}")
    start_role = (
        AgentRole.BUILDING_ENERGY_MANAGER
        if families[0] in {TaskFamily.BASELINE_FLEXIBILITY, TaskFamily.MARKET_TRADING}
        else AgentRole.DISTRIBUTION_GRID_OPERATOR
    )
    receiver = (
        AgentRole.BUILDING_ENERGY_MANAGER
        if start_role == AgentRole.DISTRIBUTION_GRID_OPERATOR
        else AgentRole.DISTRIBUTION_GRID_OPERATOR
    )
    offer_price = (
        float(35 + 5 * (index % 9))
        if TaskFamily.MARKET_TRADING in families
        else None
    )
    text = instruction(
        families,
        DAYS[index % len(DAYS)],
        target_bems,
        target_buses,
        event_windows,
        target_kw,
        communication,
        start_role,
        offer_price,
        number,
    )
    clarification = communication in {"implicit_unit", "missing_field", "conflicting_constraints"}
    target_status = FieldStatus.DECLARED if communication in {"direct", "conflicting_constraints"} else FieldStatus.MISSING
    target_unit = "kW" if communication in {"direct", "conflicting_constraints"} else None
    requested = family_outputs(families)
    fields = {
        "target_power": TaskField(
        value=communicated_target_kw if communication != "missing_field" else None,
            unit=target_unit,
            status=target_status,
            confidence=1.0 if target_status == FieldStatus.DECLARED else 0.0,
            note="Ground-truth interpretation of the source instruction.",
        ),
        "time_resolution": TaskField(value=15, unit="min", status=FieldStatus.INFERRED, confidence=1.0),
        "target_buses": TaskField(value=target_buses, status=FieldStatus.DECLARED, confidence=1.0),
    }
    if communication == "conflicting_constraints":
        fields["priority_conflict"] = TaskField(
            value=True,
            status=FieldStatus.MISSING,
            confidence=1.0,
            note="Two incompatible highest-priority power requirements require clarification.",
        )
    if offer_price is not None:
        fields["offer_price"] = TaskField(
            value=offer_price,
            unit="USD/MWh",
            status=FieldStatus.DECLARED,
            confidence=1.0,
            note="Deterministic simplified market offer used by the case contract.",
        )
    for output in requested:
        fields[output] = TaskField(status=FieldStatus.TO_BE_SIMULATED, confidence=1.0)
    specification = TaskSpecification(
        episode_id=f"ep-{number:03d}",
        case_id=CASE_ID,
        sender_role=start_role,
        receiver_role=receiver,
        task_families=families,
        operating_day=DAYS[index % len(DAYS)],
        windows=event_windows,
        target_bems=target_bems,
        target_buses=target_buses,
        objectives=objectives(
            families,
            communicated_target_kw if communication != "missing_field" else None,
        ),
        constraints=[
            Constraint(name="voltage_floor", metric="minimum_voltage_pu", operator="at_least", value=0.95, unit="p.u."),
            Constraint(name="voltage_ceiling", metric="maximum_voltage_pu", operator="at_most", value=1.05, unit="p.u."),
            Constraint(name="line_limit", metric="maximum_line_loading_pct", operator="at_most", value=100.0, unit="%"),
            Constraint(name="comfort", metric="comfort_violation_minutes", operator="at_most", value=0.0, unit="min"),
        ],
        fields=fields,
        requested_outputs=requested,
        clarification_required=clarification,
        clarification_question=(
            "Confirm the target power with units and resolve the priority conflict before execution."
            if clarification
            else None
        ),
        source_text=text,
    )
    expected_verdict = ReviewVerdict.REVISE if clarification else ReviewVerdict.PASS
    issue_codes = ["T-MISSING-OR-CONFLICTING-FIELD"] if clarification else []
    episode_id = f"ep-{number:03d}"
    episode = Episode(
        episode_id=episode_id,
        case_id=CASE_ID,
        split=EpisodeSplit.TEST,
        group_id=f"physical-{number:03d}",
        primary_family=primary,
        task_families=families,
        start_role=start_role,
        operating_day=DAYS[index % len(DAYS)],
        target_bems=target_bems,
        target_buses=target_buses,
        windows=event_windows,
        difficulty=EpisodeDifficulty(
            spatial_scope=spatial_scope,
            temporal_scope=temporal_scope,
            feasibility=feasibility,
            communication=communication,
        ),
        sender_message=text,
        generator_version=GENERATOR_VERSION,
        seed=SEED,
    )
    canonical_receiver = canonical_receiver_instruction(specification)
    if clarification:
        receiver_feedback = (
            "I cannot make a commitment because the requested power, its unit, or the priority "
            "rules are not uniquely executable. I am returning this task for clarification."
        )
        sender_final = (
            "The receiver made no commitment because the initial message was not uniquely "
            "executable. This task is 0% complete."
        )
        route = "clarify_sender"
        decision_rule = "Request one focused clarification and do not run physical tools."
        completion_rule = "An unexecutable one-round benchmark task has 0% completion."
        expected_actions: list[ActionType] = []
        numeric_outputs: list[str] = []
        numeric_path = None
    else:
        receiver_feedback = None
        sender_final = None
        route = "execute_physics_then_release"
        decision_rule = (
            "Use the frozen joint RC/dispatch optimization as the candidate decision, then accept, "
            "modify, or reject it only after EnergyPlus and pandapower validation of comfort, "
            "voltage, thermal, convergence, and business constraints."
        )
        completion_rule = (
            "Report 100% only when every requested term is accepted; otherwise report the frozen "
            "business-specific delivered-versus-requested ratio and all residual gaps."
        )
        expected_actions = action_types(families)
        numeric_outputs = requested
        numeric_path = f"numeric_oracles/{episode_id}.json"
    truth = EpisodeGroundTruth(
        episode_id=episode_id,
        case_id=CASE_ID,
        source_message_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
        task_specification=specification,
        perfect_execution=PerfectExecutionOracle(
            route=route,
            canonical_receiver_instruction=canonical_receiver,
            required_semantic_fields=[
                "sender_role",
                "receiver_role",
                "task_families",
                "operating_day",
                "windows",
                "target_bems",
                "target_buses",
                "target_power",
                "objectives",
                "constraints",
                "requested_outputs",
                *(
                    ["offer_price"]
                    if TaskFamily.MARKET_TRADING in families
                    else []
                ),
            ],
            expected_action_types=expected_actions,
            required_numeric_outputs=numeric_outputs,
            receiver_decision_rule=decision_rule,
            sender_completion_rule=completion_rule,
            expected_receiver_feedback=receiver_feedback,
            expected_sender_final=sender_final,
            expected_completion_percentage=0.0 if clarification else None,
        ),
        minimum_action_types=expected_actions,
        expected_reviewer_verdict=expected_verdict,
        expected_issue_codes=issue_codes,
        oracle_kind="deterministic_physics" if not clarification else "deterministic_contract",
        physics_truth_status=(
            "pending" if not clarification else "not_required_for_current_turn"
        ),
        numeric_oracle_path=numeric_path,
    )
    return episode, truth


def canonical_receiver_instruction(specification: TaskSpecification) -> str:
    families = " and ".join(
        family.value.replace("_", " ") for family in specification.task_families
    )
    windows_text = "; ".join(
        f"{window.start}-{window.end}" for window in specification.windows
    )
    bems = ", ".join(specification.target_bems)
    buses = ", ".join(str(bus) for bus in specification.target_buses)
    target = specification.fields["target_power"]
    amount = (
        f"{float(target.value):g} {target.unit}"
        if isinstance(target.value, (int, float)) and target.unit
        else "a power amount that requires clarification"
    )
    if specification.receiver_role == AgentRole.BUILDING_ENERGY_MANAGER:
        return (
            f"For the {specification.operating_day} operating day, assess whether {bems} can "
            f"provide {amount} for {families} during {windows_text}. Preserve comfort and rebound "
            f"limits; the response will be validated at buses {buses} against the 0.95-1.05 p.u. "
            "voltage band and 100% line-loading limit."
        )
    return (
        f"For the {specification.operating_day} operating day, assess the {amount} flexibility "
        f"offer from {bems} at buses {buses} for {families} during {windows_text}. Return the "
        "network-feasible accepted quantity, residual need, risks, and settlement consequences."
    )


def build() -> tuple[list[Episode], list[EpisodeGroundTruth]]:
    episodes: list[Episode] = []
    truths: list[EpisodeGroundTruth] = []
    number = 1
    for family, count in FAMILY_COUNTS.items():
        for _ in range(count):
            episode, truth = make_episode(number, [family], family)
            episodes.append(episode)
            truths.append(truth)
            number += 1
    families = list(FAMILY_COUNTS)
    for composite_index in range(50):
        left = families[composite_index % len(families)]
        right = families[(composite_index * 3 + 2) % len(families)]
        if right == left:
            right = families[(families.index(left) + 1) % len(families)]
        episode, truth = make_episode(number, [left, right], TaskFamily.COMPOSITE)
        episodes.append(episode)
        truths.append(truth)
        number += 1
    if len(episodes) != 200:
        raise RuntimeError(f"Expected 200 episodes, created {len(episodes)}")

    clarification_indices = [
        index for index, episode in enumerate(episodes)
        if truths[index].expected_issue_codes
    ]
    random.Random(SEED).shuffle(clarification_indices)
    memory_seed = set(clarification_indices[:20])
    remaining = [index for index in range(200) if index not in memory_seed]
    random.Random(SEED + 2).shuffle(remaining)
    development = set(remaining[:30])
    for index, episode in enumerate(episodes):
        if index in development:
            episode.split = EpisodeSplit.DEVELOPMENT
        elif index in memory_seed:
            episode.split = EpisodeSplit.MEMORY_SEED
        else:
            episode.split = EpisodeSplit.TEST
    return episodes, truths


def contract_semantic_sha256(truth: EpisodeGroundTruth) -> str:
    """Hash the contract semantics independently of mutable verification state."""

    payload = truth.model_dump(mode="json")
    payload.pop("contract_truth_status", None)
    payload.pop("physics_truth_status", None)
    payload.pop("numeric_oracle_path", None)
    # Runtime retrieval diagnostics are not part of the frozen task semantics.
    payload.get("task_specification", {}).pop("retrieval_trace", None)
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write(episodes: list[Episode], truths: list[EpisodeGroundTruth]) -> None:
    if [episode.episode_id for episode in episodes] != [truth.episode_id for truth in truths]:
        raise ValueError("episode and ground-truth identifiers are not one-to-one and ordered")
    groundtruth_dir = CASE_DIR / "groundtruth"
    previous_contract_path = groundtruth_dir / "contract_oracles.jsonl"
    previous_truths = (
        {
            row["episode_id"]: row
            for row in (
                json.loads(line)
                for line in previous_contract_path.read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            )
        }
        if previous_contract_path.is_file()
        else {}
    )
    for truth in truths:
        previous = previous_truths.get(truth.episode_id)
        if not previous or previous.get("physics_truth_status") != "verified":
            continue
        previous_specification = dict(previous["task_specification"])
        current_specification = truth.task_specification.model_dump(mode="json")
        # Natural-language-only edits do not invalidate a deterministic physics
        # result. Every physical field, window, objective, and constraint must
        # remain exactly equal for the verified status to be retained.
        previous_specification.pop("source_text", None)
        current_specification.pop("source_text", None)
        if previous_specification == current_specification:
            truth.physics_truth_status = "verified"

    tasks_dir = CASE_DIR / "tasks"
    splits_dir = tasks_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    corpus = tasks_dir / "episodes.jsonl"
    corpus.write_text(
        "".join(episode.model_dump_json() + "\n" for episode in episodes), encoding="utf-8"
    )
    sender_corpus = tasks_dir / "sender_messages.jsonl"
    sender_corpus.write_text(
        "".join(
            json.dumps(
                {
                    "episode_id": episode.episode_id,
                    "sender_role": episode.start_role.value,
                    "sender_message": episode.sender_message,
                },
                ensure_ascii=False,
            )
            + "\n"
            for episode in episodes
        ),
        encoding="utf-8",
    )
    groundtruth_dir.mkdir(parents=True, exist_ok=True)
    contract_oracles = groundtruth_dir / "contract_oracles.jsonl"
    contract_oracles.write_text(
        "".join(truth.model_dump_json() + "\n" for truth in truths),
        encoding="utf-8",
    )
    verification_path = groundtruth_dir / "contract_verification.jsonl"
    verification_records = [
        {
            "schema_version": "1.0",
            "episode_id": truth.episode_id,
            "status": "programmatically_verified",
            "contract_semantic_sha256": contract_semantic_sha256(truth),
            "verification_protocol": "contract-oracle-validator-v1",
            "checks": [
                "source_message_hash",
                "episode_contract_alignment",
                "family_output_mapping",
                "execution_route",
                "service_direction_window_binding",
                "required_output_definedness",
            ],
        }
        for truth in truths
    ]
    verification_path.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
            for record in verification_records
        ),
        encoding="utf-8",
    )
    for split in EpisodeSplit:
        identifiers = [episode.episode_id for episode in episodes if episode.split == split]
        (splits_dir / f"{split.value}.txt").write_text("\n".join(identifiers) + "\n", encoding="utf-8")

    family_counts = Counter(episode.primary_family.value for episode in episodes)
    split_counts = Counter(episode.split.value for episode in episodes)
    communication_counts = Counter(episode.difficulty.communication for episode in episodes)
    spatial_counts = Counter(episode.difficulty.spatial_scope for episode in episodes)
    temporal_counts = Counter(episode.difficulty.temporal_scope for episode in episodes)
    feasibility_counts = Counter(episode.difficulty.feasibility for episode in episodes)
    physics_truth_counts = Counter(
        truth.physics_truth_status for truth in truths
    )
    start_role_counts = Counter(episode.start_role.value for episode in episodes)
    summary = {
        "schema_version": "3.0",
        "case_id": CASE_ID,
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "episode_count": len(episodes),
        "frozen_sender_messages_per_episode": 1,
        "runtime_communication_acts": 2,
        "runtime_information_states": 4,
        "single_family_count": sum(len(episode.task_families) == 1 for episode in episodes),
        "composite_count": sum(len(episode.task_families) == 2 for episode in episodes),
        "primary_family_counts": dict(sorted(family_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "communication_counts": dict(sorted(communication_counts.items())),
        "spatial_scope_counts": dict(sorted(spatial_counts.items())),
        "temporal_scope_counts": dict(sorted(temporal_counts.items())),
        "feasibility_counts": dict(sorted(feasibility_counts.items())),
        "physics_truth_status_counts": {
            status: physics_truth_counts.get(status, 0)
            for status in ("not_required_for_current_turn", "pending", "verified")
        },
        "start_role_counts": dict(sorted(start_role_counts.items())),
        "unique_episode_ids": len({episode.episode_id for episode in episodes}),
        "unique_group_ids": len({episode.group_id for episode in episodes}),
        "unique_sender_messages": len({episode.sender_message for episode in episodes}),
        "sender_messages_sha256": hashlib.sha256(sender_corpus.read_bytes()).hexdigest(),
        "groundtruth_contract_sha256": hashlib.sha256(contract_oracles.read_bytes()).hexdigest(),
        "split_leakage": False,
        "split_leakage_definition": "No exact physical/group identity appears in more than one split.",
        "structural_template_overlap": True,
        "note": (
            "Each episode freezes only the sender's initial natural-language message; receiver-side "
            "interpretation, receiver feedback, and sender-side final interpretation are runtime records. "
            "Each exact physical group has one canonical wording in v7, so no paraphrase of the same "
            "scenario can cross splits. Structural task-family templates do span splits and must be "
            "reported as such. "
            "Feasibility is a preregistered design label until physics_truth_status becomes verified."
        )
    }
    (tasks_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    groundtruth_manifest = {
        "schema_version": "1.0",
        "case_id": CASE_ID,
        "generator_version": GENERATOR_VERSION,
        "seed": SEED,
        "oracle_count": len(truths),
        "contract_oracle_count": len(truths),
        "programmatically_verified_contract_count": len(verification_records),
        "expert_verified_contract_count": sum(
            truth.contract_truth_status == "expert_verified" for truth in truths
        ),
        "physics_verified_count": sum(
            truth.physics_truth_status == "verified" for truth in truths
        ),
        "physics_pending_count": sum(
            truth.physics_truth_status == "pending" for truth in truths
        ),
        "physics_not_required_count": sum(
            truth.physics_truth_status == "not_required_for_current_turn" for truth in truths
        ),
        "programmatically_verified_numeric_count": sum(
            truth.physics_truth_status == "verified" for truth in truths
        ),
        "episodes_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "contract_oracles_sha256": hashlib.sha256(contract_oracles.read_bytes()).hexdigest(),
        "contract_verification_path": "contract_verification.jsonl",
        "contract_verification_sha256": hashlib.sha256(
            verification_path.read_bytes()
        ).hexdigest(),
        "numeric_oracle_directory": "numeric_oracles",
        "publication_gate": (
            "Contract oracles must pass deterministic schema, direction, window, and source-hash "
            "checks; direct-task numeric oracles require the frozen RC/dispatch optimization "
            "followed by EnergyPlus and pandapower validation."
        ),
        "numeric_oracles": [
            {
                "episode_id": path.stem,
                "path": f"numeric_oracles/{path.name}",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in sorted((groundtruth_dir / "numeric_oracles").glob("ep-*.json"))
            if next(
                truth.physics_truth_status
                for truth in truths
                if truth.episode_id == path.stem
            )
            == "verified"
        ],
    }
    (groundtruth_dir / "manifest.json").write_text(
        json.dumps(groundtruth_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    # Reviewer memory is built separately by build_reviewer_memory.py.
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    arguments = parse_args()
    require_safe_regeneration(
        reset_verified_physics=arguments.reset_verified_physics
    )
    write(*build())
