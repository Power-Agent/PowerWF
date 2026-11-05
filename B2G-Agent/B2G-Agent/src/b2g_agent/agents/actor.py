from __future__ import annotations

import hashlib
import json
import re
from b2g_agent.observability.journal import emit, traced_stage

from b2g_agent.agents.llm import StructuredLLMClient
from b2g_agent.contracts import (
    ActionType,
    ExecutionPlan,
    PlanAction,
    PlanProposal,
    TaskFamily,
    TaskSpecification,
)


SUPPORTED_REQUESTED_OUTPUTS = {
    "absorbed_energy_kwh",
    "accepted_capacity_kw",
    "comfort_violation_minutes",
    "counterfactual_baseline_kw",
    "delivered_reduction_kw",
    "delivery_ratio",
    "feeder_peak_kw",
    "flexibility_lower_kw",
    "flexibility_upper_kw",
    "maximum_line_loading_pct",
    "maximum_voltage_pu",
    "minimum_voltage_pu",
    "overload_intervals",
    "pv_curtailment_kwh",
    "rebound_kw",
    "reserved_capacity_kw",
    "settlement_usd",
    "violation_intervals",
}

ACTION_TO_TOOL = {
    ActionType.LOAD_CASE: "case_io",
    ActionType.SELECT_BUILDING_MODELS: "case_io",
    ActionType.LOAD_REFERENCE_PROFILES: "case_io",
    ActionType.RUN_ENERGYPLUS_BASELINE: "energyplus",
    ActionType.CALIBRATE_RC_MODELS: "rc_model",
    ActionType.SOLVE_JOINT_RC_OPF: "opf",
    ActionType.RUN_ENERGYPLUS_EVENT: "energyplus",
    ActionType.ALIGN_15_MIN_TIMESERIES: "metrics",
    ActionType.SCALE_BEM_PORTFOLIOS: "metrics",
    ActionType.APPLY_DER_DISPATCH: "optimizer",
    ActionType.MAP_LOADS_TO_BUSES: "metrics",
    ActionType.RUN_PANDAPOWER: "pandapower",
    ActionType.CALCULATE_MARKET_SETTLEMENT: "metrics",
    ActionType.EXTRACT_METRICS: "metrics",
}

ACTION_COST = {
    ActionType.LOAD_CASE: 1,
    ActionType.SELECT_BUILDING_MODELS: 1,
    ActionType.LOAD_REFERENCE_PROFILES: 1,
    ActionType.RUN_ENERGYPLUS_BASELINE: 20,
    ActionType.CALIBRATE_RC_MODELS: 5,
    ActionType.SOLVE_JOINT_RC_OPF: 8,
    ActionType.RUN_ENERGYPLUS_EVENT: 20,
    ActionType.ALIGN_15_MIN_TIMESERIES: 2,
    ActionType.SCALE_BEM_PORTFOLIOS: 1,
    ActionType.APPLY_DER_DISPATCH: 3,
    ActionType.MAP_LOADS_TO_BUSES: 2,
    ActionType.RUN_PANDAPOWER: 8,
    ActionType.CALCULATE_MARKET_SETTLEMENT: 1,
    ActionType.EXTRACT_METRICS: 1,
}

ACTION_OUTPUTS = {
    ActionType.LOAD_CASE: ["case_manifest", "bem_bus_mapping"],
    ActionType.SELECT_BUILDING_MODELS: ["building_models"],
    ActionType.LOAD_REFERENCE_PROFILES: [
        "background_load", "pv_forecast", "price_profile"
    ],
    ActionType.RUN_ENERGYPLUS_BASELINE: [
        "bem_baseline_kw", "baseline_zone_temperature_c"
    ],
    ActionType.CALIBRATE_RC_MODELS: [
        "calibrated_rc_models", "rc_validation_metrics"
    ],
    ActionType.SOLVE_JOINT_RC_OPF: [
        "candidate_building_schedules",
        "candidate_der_dispatch",
        "joint_optimization_status",
    ],
    ActionType.RUN_ENERGYPLUS_EVENT: [
        "bem_event_kw", "event_zone_temperature_c"
    ],
    ActionType.ALIGN_15_MIN_TIMESERIES: ["aligned_profiles"],
    ActionType.SCALE_BEM_PORTFOLIOS: ["portfolio_profiles"],
    ActionType.APPLY_DER_DISPATCH: ["der_dispatch", "net_portfolio_profiles"],
    ActionType.MAP_LOADS_TO_BUSES: ["node_p_kw", "node_q_kvar"],
    ActionType.RUN_PANDAPOWER: [
        "bus_voltage_pu", "line_loading_pct", "substation_power_kw"
    ],
    ActionType.CALCULATE_MARKET_SETTLEMENT: ["settlement_usd"],
}

ACTION_INPUT_SCHEMA = {
    ActionType.LOAD_CASE: ["case_id", "selection_policy", "estimated_cost_units"],
    ActionType.SELECT_BUILDING_MODELS: ["target_bems"],
    ActionType.LOAD_REFERENCE_PROFILES: ["operating_day"],
    ActionType.RUN_ENERGYPLUS_BASELINE: [],
    ActionType.CALIBRATE_RC_MODELS: [],
    ActionType.SOLVE_JOINT_RC_OPF: ["objectives", "constraints", "maximum_iterations"],
    ActionType.RUN_ENERGYPLUS_EVENT: ["windows"],
    ActionType.ALIGN_15_MIN_TIMESERIES: ["resolution_minutes"],
    ActionType.SCALE_BEM_PORTFOLIOS: [],
    ActionType.APPLY_DER_DISPATCH: [],
    ActionType.MAP_LOADS_TO_BUSES: ["target_buses"],
    ActionType.RUN_PANDAPOWER: [],
    ActionType.CALCULATE_MARKET_SETTLEMENT: [],
    ActionType.EXTRACT_METRICS: ["requested_outputs"],
}


class PlanCompiler:
    """Compiles a finite, schema-checked minimal DAG for physical evidence."""

    version = "backward-capability-plan-compiler-v3"

    def compile(self, specification: TaskSpecification) -> ExecutionPlan:
        actions: list[PlanAction] = []
        selected_types = _required_action_types(specification)
        selected_type_set = set(selected_types)

        def add(
            action_type: ActionType,
            tool: str,
            *,
            depends_on: list[str] | None = None,
            inputs: dict[str, object] | None = None,
            outputs: list[str] | None = None,
        ) -> str:
            action_id = f"a{len(actions) + 1:02d}_{action_type.value}"
            actions.append(
                PlanAction(
                    action_id=action_id,
                    action_type=action_type,
                    tool=tool,  # type: ignore[arg-type]
                    depends_on=depends_on or [],
                    inputs=inputs or {},
                    outputs=outputs or [],
                )
            )
            return action_id

        load_case = add(
            ActionType.LOAD_CASE,
            "case_io",
            inputs={
                "case_id": specification.case_id,
                "selection_policy": "backward_minimum_cost",
                "estimated_cost_units": sum(ACTION_COST[item] for item in selected_types),
            },
            outputs=["case_manifest", "bem_bus_mapping"],
        )
        select = add(
            ActionType.SELECT_BUILDING_MODELS,
            "case_io",
            depends_on=[load_case],
            inputs={"target_bems": specification.target_bems},
            outputs=["building_models"],
        )
        profiles = add(
            ActionType.LOAD_REFERENCE_PROFILES,
            "case_io",
            depends_on=[load_case],
            inputs={"operating_day": specification.operating_day},
            outputs=["background_load", "pv_forecast", "price_profile"],
        )
        baseline = add(
            ActionType.RUN_ENERGYPLUS_BASELINE,
            "energyplus",
            depends_on=[select, profiles],
            outputs=["bem_baseline_kw", "baseline_zone_temperature_c"],
        )
        rc_models = add(
            ActionType.CALIBRATE_RC_MODELS,
            "rc_model",
            depends_on=[select, baseline, profiles],
            outputs=["calibrated_rc_models", "rc_validation_metrics"],
        )
        joint_dispatch = add(
            ActionType.SOLVE_JOINT_RC_OPF,
            "opf",
            depends_on=[rc_models, profiles],
            inputs={
                "objectives": [item.model_dump() for item in specification.objectives],
                "constraints": [item.model_dump(exclude={"origin", "evidence"}) for item in specification.constraints],
                "maximum_iterations": 10,
            },
            outputs=[
                "candidate_building_schedules",
                "candidate_der_dispatch",
                "joint_optimization_status",
            ],
        )
        event = add(
            ActionType.RUN_ENERGYPLUS_EVENT,
            "energyplus",
            depends_on=[select, baseline, joint_dispatch],
            inputs={"windows": [window.model_dump() for window in specification.windows]},
            outputs=["bem_event_kw", "event_zone_temperature_c"],
        )
        align = add(
            ActionType.ALIGN_15_MIN_TIMESERIES,
            "metrics",
            depends_on=[baseline, event, profiles],
            inputs={"resolution_minutes": 15},
            outputs=["aligned_profiles"],
        )
        scale = add(
            ActionType.SCALE_BEM_PORTFOLIOS,
            "metrics",
            depends_on=[align],
            outputs=["portfolio_profiles"],
        )
        previous = scale
        if ActionType.APPLY_DER_DISPATCH in selected_type_set:
            previous = add(
                ActionType.APPLY_DER_DISPATCH,
                "optimizer",
                depends_on=[scale, profiles, joint_dispatch],
                outputs=["der_dispatch", "net_portfolio_profiles"],
            )
        mapped = add(
            ActionType.MAP_LOADS_TO_BUSES,
            "metrics",
            depends_on=[previous, profiles],
            inputs={"target_buses": specification.target_buses},
            outputs=["node_p_kw", "node_q_kvar"],
        )
        power_flow = add(
            ActionType.RUN_PANDAPOWER,
            "pandapower",
            depends_on=[mapped],
            outputs=["bus_voltage_pu", "line_loading_pct", "substation_power_kw"],
        )
        final_dependencies = [power_flow, event]
        if ActionType.CALCULATE_MARKET_SETTLEMENT in selected_type_set:
            settlement = add(
                ActionType.CALCULATE_MARKET_SETTLEMENT,
                "metrics",
                depends_on=[power_flow, profiles],
                outputs=["settlement_usd"],
            )
            final_dependencies.append(settlement)
        add(
            ActionType.EXTRACT_METRICS,
            "metrics",
            depends_on=final_dependencies,
            inputs={"requested_outputs": specification.requested_outputs},
            outputs=specification.requested_outputs,
        )

        plan_payload = [action.model_dump(mode="json") for action in actions]
        plan_id = hashlib.sha256(
            json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        plan = ExecutionPlan(
            episode_id=specification.episode_id,
            plan_id=plan_id,
            actions=actions,
            requested_outputs=specification.requested_outputs,
            compiler_version=self.version,
        )
        errors = self.validate(plan, specification)
        plan.validation_errors = errors
        plan.valid = not errors
        return plan

    def compile_proposal(
        self,
        specification: TaskSpecification,
        proposal: PlanProposal,
    ) -> ExecutionPlan:
        # Cost is compiler bookkeeping, not a physical input or an LLM decision.
        # Compute cost from selected actions. Only redundant transitive edges
        # may be removed; never add missing edges or alter physical bindings.
        actions = [action.model_copy(deep=True) for action in proposal.actions]
        normalizations = self._normalize_transitive_edges(actions)
        for action in actions:
            if action.action_type == ActionType.LOAD_CASE:
                action.inputs["estimated_cost_units"] = sum(
                    ACTION_COST[item.action_type] for item in actions
                )
        plan_payload = [action.model_dump(mode="json") for action in actions]
        plan_id = hashlib.sha256(
            json.dumps(plan_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:16]
        plan = ExecutionPlan(
            episode_id=specification.episode_id,
            plan_id=plan_id,
            actions=actions,
            requested_outputs=specification.requested_outputs,
            compiler_version=self.version,
            normalization_log=normalizations,
        )
        errors = self.validate(plan, specification)
        plan.validation_errors = errors
        plan.valid = not errors
        return plan

    @staticmethod
    def _normalize_transitive_edges(actions: list[PlanAction]) -> list[dict[str, object]]:
        positions = {a.action_id: i for i, a in enumerate(actions)}
        if len(positions) != len(actions) or any(
            d not in positions or positions[d] >= positions[a.action_id]
            for a in actions for d in a.depends_on
        ):
            return []  # Malformed/cyclic graphs must be rejected, never repaired here.
        by_id = {a.action_id: a for a in actions}
        required = _required_dependencies({a.action_type for a in actions})
        changes = []

        def ancestors(node):
            found, stack = set(), list(by_id[node].depends_on)
            while stack:
                dep = stack.pop()
                if dep not in found:
                    found.add(dep)
                    stack.extend(by_id[dep].depends_on)
            return found

        for action in actions:
            required_types = required.get(action.action_type, set())
            kept = [d for d in action.depends_on if by_id[d].action_type in required_types]
            reachable = set().union(*(ancestors(d) for d in kept)) if kept else set()
            removable = [d for d in action.depends_on if d not in kept and d in reachable]
            if removable:
                before = list(action.depends_on)
                action.depends_on = [d for d in before if d not in removable]
                changes.append({"action_id": action.action_id, "before": before,
                                "after": list(action.depends_on), "removed": removable,
                                "reason": "Redundant transitive edge; required direct bindings and graph reachability preserved."})
        if changes:
            emit("plan.normalized", changes=changes)
        return changes

    def diagnostics(self, plan: ExecutionPlan, specification: TaskSpecification) -> list[dict[str, object]]:
        if plan.compiler_version == "unstructured-tool-plan-v2":
            return [{"validation_error": error} for error in self.validate(plan, specification)]
        by_id = {a.action_id: a.action_type for a in plan.actions}
        required = _required_dependencies(set(by_id.values()))
        bindings = _required_input_bindings(specification, _required_action_types(specification))
        details = []
        for action in plan.actions:
            observed = {by_id[d] for d in action.depends_on if d in by_id}
            expected = required.get(action.action_type, set())
            if observed != expected:
                details.append({"action": action.action_type.value,
                                "missing_dependencies": sorted(x.value for x in expected - observed),
                                "extra_dependencies": sorted(x.value for x in observed - expected)})
            if action.inputs != bindings.get(action.action_type, {}):
                details.append({"action": action.action_type.value, "expected_inputs": bindings.get(action.action_type, {}),
                                "observed_inputs": action.inputs})
        return details

    def validate(self, plan: ExecutionPlan, specification: TaskSpecification) -> list[str]:
        if plan.compiler_version == "unstructured-tool-plan-v2":
            from b2g_agent.agents.mechanism_ablations import validate_unstructured_plan
            return validate_unstructured_plan(plan, specification)
        errors: list[str] = []
        if plan.episode_id != specification.episode_id:
            errors.append("episode_binding_mismatch")
        if plan.requested_outputs != specification.requested_outputs:
            errors.append("requested_output_binding_mismatch")
        ids = [action.action_id for action in plan.actions]
        if len(ids) != len(set(ids)):
            errors.append("duplicate_action_id")
        positions = {action_id: index for index, action_id in enumerate(ids)}
        for action in plan.actions:
            if len(action.depends_on) != len(set(action.depends_on)):
                errors.append(f"duplicate_dependency:{action.action_id}")
            expected_tool = ACTION_TO_TOOL.get(action.action_type)
            if expected_tool is not None and action.tool != expected_tool:
                errors.append(
                    f"wrong_tool:{action.action_type.value}:{action.tool}:{expected_tool}"
                )
            for dependency in action.depends_on:
                if dependency not in positions:
                    errors.append(f"unknown_dependency:{action.action_id}:{dependency}")
                elif positions[dependency] >= positions[action.action_id]:
                    errors.append(f"non_dag_dependency:{action.action_id}:{dependency}")
        action_types = {action.action_type for action in plan.actions}
        action_type_by_id = {
            action.action_id: action.action_type for action in plan.actions
        }
        mandatory = {
            ActionType.LOAD_CASE,
            ActionType.ALIGN_15_MIN_TIMESERIES,
            ActionType.MAP_LOADS_TO_BUSES,
            ActionType.RUN_PANDAPOWER,
            ActionType.EXTRACT_METRICS,
        }
        for missing in sorted(mandatory - action_types, key=lambda item: item.value):
            errors.append(f"missing_action:{missing.value}")
        expected_types = _required_action_types(specification)
        observed_types = [action.action_type for action in plan.actions]
        for missing in sorted(set(expected_types) - set(observed_types), key=lambda item: item.value):
            errors.append(f"missing_required_action:{missing.value}")
        for redundant in sorted(set(observed_types) - set(expected_types), key=lambda item: item.value):
            errors.append(f"redundant_action:{redundant.value}")
        if len(observed_types) != len(set(observed_types)):
            errors.append("duplicate_action_type")
        expected_dependencies = _required_dependencies(action_types)
        expected_inputs = _required_input_bindings(specification, expected_types)
        for action in plan.actions:
            observed_dependency_types = {
                action_type_by_id[dependency]
                for dependency in action.depends_on
                if dependency in action_type_by_id
            }
            required_dependency_types = expected_dependencies.get(
                action.action_type, set()
            )
            if observed_dependency_types != required_dependency_types:
                errors.append(
                    "dependency_contract_mismatch:"
                    f"{action.action_type.value}"
                )
            if action.inputs != expected_inputs.get(action.action_type, {}):
                errors.append(f"input_binding_mismatch:{action.action_type.value}")
            required_outputs = (
                specification.requested_outputs
                if action.action_type == ActionType.EXTRACT_METRICS
                else ACTION_OUTPUTS.get(action.action_type, [])
            )
            if action.outputs != required_outputs:
                errors.append(f"output_contract_mismatch:{action.action_type.value}")
        if plan.time_resolution_minutes != 15:
            errors.append("invalid_time_resolution")
        if not specification.target_buses or any(bus < 1 or bus > 33 for bus in specification.target_buses):
            errors.append("invalid_target_bus")
        if specification.clarification_required:
            errors.append("clarification_required")
        for unsupported in sorted(
            set(specification.requested_outputs) - SUPPORTED_REQUESTED_OUTPUTS
        ):
            errors.append(f"unsupported_requested_output:{unsupported}")
        return errors


class ActorAgent:
    def __init__(
        self,
        compiler: PlanCompiler | None = None,
        *,
        llm: StructuredLLMClient | None = None,
    ) -> None:
        self.compiler = compiler or PlanCompiler()
        self.llm = llm

    @traced_stage("actor.plan")
    def plan(
        self,
        specification: TaskSpecification,
        *,
        reviewer_feedback: list[dict[str, object]] | None = None,
        attempt: int = 0,
    ) -> ExecutionPlan | None:
        if specification.clarification_required:
            return None
        if self.llm is None or not self.llm.available:
            return self.compiler.compile(specification)
        proposal = self.llm.call(
            PlanProposal,
            agent_name="actor",
            system=(
                "You are the B2G-Agent Actor planner. Select only from the supplied finite action "
                "types and tools. Return the smallest dependency-valid action DAG that resolves every "
                "requested output. Preserve 15-minute alignment and explicit EnergyPlus-to-pandapower "
                "mapping. List actions in topological order, so each dependency appears earlier. "
                "Use the catalog's exact input binding rules and conditional prerequisites; inputs "
                "contain concrete values, not variable names or references. Do not invent simulation "
                "results. Copy the supplied per-action input bindings exactly; do not add a second "
                "dispatch action when the joint optimizer already supplies that dispatch. Return JSON only."
            ),
            user_payload={
                "task_specification": specification.model_dump(mode="json"),
                "tool_capability_graph": _capability_catalog(),
                "exact_action_input_bindings": {
                    action.value: values for action, values in
                    _required_input_bindings(specification, _required_action_types(specification)).items()
                },
                "requested_output_contract": specification.requested_outputs,
                "reviewer_feedback": reviewer_feedback or [],
                "planning_attempt": attempt,
            },
        )
        return self.compiler.compile_proposal(specification, proposal)


def _capability_catalog() -> list[dict[str, object]]:
    """Expose the same typed capability budget to every planning method."""

    binding_rules = {
        ActionType.LOAD_CASE: "case_id = task_specification.case_id; selection_policy = 'backward_minimum_cost'; estimated_cost_units = sum(cost_units for every selected action).",
        ActionType.SELECT_BUILDING_MODELS: "target_bems = task_specification.target_bems.",
        ActionType.LOAD_REFERENCE_PROFILES: "operating_day = task_specification.operating_day.",
        ActionType.SOLVE_JOINT_RC_OPF: "objectives = exact complete list from task_specification; constraints = exact complete list of name, metric, operator, value, unit, hard (origin/evidence are contract provenance, not tool inputs); maximum_iterations = 10.",
        ActionType.RUN_ENERGYPLUS_EVENT: "windows = task_specification.windows, including timezone.",
        ActionType.ALIGN_15_MIN_TIMESERIES: "resolution_minutes = 15.",
        ActionType.MAP_LOADS_TO_BUSES: "target_buses = task_specification.target_buses.",
        ActionType.EXTRACT_METRICS: "requested_outputs = task_specification.requested_outputs; outputs must be the same list in the same order.",
    }
    conditions = {
        ActionType.APPLY_DER_DISPATCH: "Include exactly when task_families contains renewable_absorption, reserve_service, or market_trading; omit otherwise.",
        ActionType.CALCULATE_MARKET_SETTLEMENT: "Include exactly when requested_outputs contains settlement_usd; omit otherwise.",
        ActionType.MAP_LOADS_TO_BUSES: "Depend on load_reference_profiles and exactly one of apply_der_dispatch (when included) or scale_bem_portfolios (otherwise).",
        ActionType.EXTRACT_METRICS: "Depend on run_pandapower and run_energyplus_event; also depend on calculate_market_settlement when included.",
    }
    reference_dependencies: dict[ActionType, tuple[ActionType, ...]] = {
        ActionType.SELECT_BUILDING_MODELS: (ActionType.LOAD_CASE,),
        ActionType.LOAD_REFERENCE_PROFILES: (ActionType.LOAD_CASE,),
        ActionType.RUN_ENERGYPLUS_BASELINE: (
            ActionType.SELECT_BUILDING_MODELS,
            ActionType.LOAD_REFERENCE_PROFILES,
        ),
        ActionType.CALIBRATE_RC_MODELS: (
            ActionType.SELECT_BUILDING_MODELS,
            ActionType.RUN_ENERGYPLUS_BASELINE,
            ActionType.LOAD_REFERENCE_PROFILES,
        ),
        ActionType.SOLVE_JOINT_RC_OPF: (
            ActionType.CALIBRATE_RC_MODELS,
            ActionType.LOAD_REFERENCE_PROFILES,
        ),
        ActionType.RUN_ENERGYPLUS_EVENT: (
            ActionType.SELECT_BUILDING_MODELS,
            ActionType.RUN_ENERGYPLUS_BASELINE,
            ActionType.SOLVE_JOINT_RC_OPF,
        ),
        ActionType.ALIGN_15_MIN_TIMESERIES: (
            ActionType.RUN_ENERGYPLUS_BASELINE,
            ActionType.RUN_ENERGYPLUS_EVENT,
            ActionType.LOAD_REFERENCE_PROFILES,
        ),
        ActionType.SCALE_BEM_PORTFOLIOS: (ActionType.ALIGN_15_MIN_TIMESERIES,),
        ActionType.APPLY_DER_DISPATCH: (
            ActionType.SCALE_BEM_PORTFOLIOS,
            ActionType.LOAD_REFERENCE_PROFILES,
            ActionType.SOLVE_JOINT_RC_OPF,
        ),
        ActionType.MAP_LOADS_TO_BUSES: (
            ActionType.SCALE_BEM_PORTFOLIOS,
            ActionType.LOAD_REFERENCE_PROFILES,
            ActionType.APPLY_DER_DISPATCH,
        ),
        ActionType.RUN_PANDAPOWER: (ActionType.MAP_LOADS_TO_BUSES,),
        ActionType.CALCULATE_MARKET_SETTLEMENT: (
            ActionType.RUN_PANDAPOWER,
            ActionType.LOAD_REFERENCE_PROFILES,
        ),
        ActionType.EXTRACT_METRICS: (
            ActionType.RUN_PANDAPOWER,
            ActionType.RUN_ENERGYPLUS_EVENT,
        ),
    }
    return [
        {
            "action_type": action.value,
            "tool": ACTION_TO_TOOL[action],
            "cost_units": ACTION_COST[action],
            "possible_prerequisites": [
                dependency.value
                for dependency in reference_dependencies.get(action, ())
            ],
            "required_input_fields": ACTION_INPUT_SCHEMA[action],
            "input_binding_rule": binding_rules.get(action, "No inputs: use an empty mapping."),
            "conditional_rule": conditions.get(action, "All listed prerequisites are required when this action is selected."),
            "output_contract": (
                ["<requested_output_fields>"]
                if action == ActionType.EXTRACT_METRICS
                else ACTION_OUTPUTS.get(action, [])
            ),
        }
        for action in ActionType
    ]


def _required_dependencies(
    selected_types: set[ActionType],
) -> dict[ActionType, set[ActionType]]:
    mapping: dict[ActionType, set[ActionType]] = {
        ActionType.LOAD_CASE: set(),
        ActionType.SELECT_BUILDING_MODELS: {ActionType.LOAD_CASE},
        ActionType.LOAD_REFERENCE_PROFILES: {ActionType.LOAD_CASE},
        ActionType.RUN_ENERGYPLUS_BASELINE: {
            ActionType.SELECT_BUILDING_MODELS,
            ActionType.LOAD_REFERENCE_PROFILES,
        },
        ActionType.CALIBRATE_RC_MODELS: {
            ActionType.SELECT_BUILDING_MODELS,
            ActionType.RUN_ENERGYPLUS_BASELINE,
            ActionType.LOAD_REFERENCE_PROFILES,
        },
        ActionType.SOLVE_JOINT_RC_OPF: {
            ActionType.CALIBRATE_RC_MODELS,
            ActionType.LOAD_REFERENCE_PROFILES,
        },
        ActionType.RUN_ENERGYPLUS_EVENT: {
            ActionType.SELECT_BUILDING_MODELS,
            ActionType.RUN_ENERGYPLUS_BASELINE,
            ActionType.SOLVE_JOINT_RC_OPF,
        },
        ActionType.ALIGN_15_MIN_TIMESERIES: {
            ActionType.RUN_ENERGYPLUS_BASELINE,
            ActionType.RUN_ENERGYPLUS_EVENT,
            ActionType.LOAD_REFERENCE_PROFILES,
        },
        ActionType.SCALE_BEM_PORTFOLIOS: {
            ActionType.ALIGN_15_MIN_TIMESERIES
        },
        ActionType.APPLY_DER_DISPATCH: {
            ActionType.SCALE_BEM_PORTFOLIOS,
            ActionType.LOAD_REFERENCE_PROFILES,
            ActionType.SOLVE_JOINT_RC_OPF,
        },
        ActionType.MAP_LOADS_TO_BUSES: {
            (
                ActionType.APPLY_DER_DISPATCH
                if ActionType.APPLY_DER_DISPATCH in selected_types
                else ActionType.SCALE_BEM_PORTFOLIOS
            ),
            ActionType.LOAD_REFERENCE_PROFILES,
        },
        ActionType.RUN_PANDAPOWER: {ActionType.MAP_LOADS_TO_BUSES},
        ActionType.CALCULATE_MARKET_SETTLEMENT: {
            ActionType.RUN_PANDAPOWER,
            ActionType.LOAD_REFERENCE_PROFILES,
        },
        ActionType.EXTRACT_METRICS: {
            ActionType.RUN_PANDAPOWER,
            ActionType.RUN_ENERGYPLUS_EVENT,
        },
    }
    if ActionType.CALCULATE_MARKET_SETTLEMENT in selected_types:
        mapping[ActionType.EXTRACT_METRICS].add(
            ActionType.CALCULATE_MARKET_SETTLEMENT
        )
    return mapping


def _required_input_bindings(
    specification: TaskSpecification,
    selected_types: list[ActionType],
) -> dict[ActionType, dict[str, object]]:
    return {
        ActionType.LOAD_CASE: {
            "case_id": specification.case_id,
            "selection_policy": "backward_minimum_cost",
            "estimated_cost_units": sum(ACTION_COST[item] for item in selected_types),
        },
        ActionType.SELECT_BUILDING_MODELS: {
            "target_bems": specification.target_bems
        },
        ActionType.LOAD_REFERENCE_PROFILES: {
            "operating_day": specification.operating_day
        },
        ActionType.RUN_ENERGYPLUS_BASELINE: {},
        ActionType.CALIBRATE_RC_MODELS: {},
        ActionType.SOLVE_JOINT_RC_OPF: {
            "objectives": [item.model_dump() for item in specification.objectives],
            "constraints": [item.model_dump(exclude={"origin", "evidence"}) for item in specification.constraints],
            "maximum_iterations": 10,
        },
        ActionType.RUN_ENERGYPLUS_EVENT: {
            "windows": [window.model_dump() for window in specification.windows]
        },
        ActionType.ALIGN_15_MIN_TIMESERIES: {"resolution_minutes": 15},
        ActionType.SCALE_BEM_PORTFOLIOS: {},
        ActionType.APPLY_DER_DISPATCH: {},
        ActionType.MAP_LOADS_TO_BUSES: {
            "target_buses": specification.target_buses
        },
        ActionType.RUN_PANDAPOWER: {},
        ActionType.CALCULATE_MARKET_SETTLEMENT: {},
        ActionType.EXTRACT_METRICS: {
            "requested_outputs": specification.requested_outputs
        },
    }


def _required_action_types(specification: TaskSpecification) -> list[ActionType]:
    """Back-chain requested evidence through the typed capability graph."""

    dependencies: dict[ActionType, set[ActionType]] = {
        ActionType.SELECT_BUILDING_MODELS: {ActionType.LOAD_CASE},
        ActionType.LOAD_REFERENCE_PROFILES: {ActionType.LOAD_CASE},
        ActionType.RUN_ENERGYPLUS_BASELINE: {
            ActionType.SELECT_BUILDING_MODELS, ActionType.LOAD_REFERENCE_PROFILES,
        },
        ActionType.CALIBRATE_RC_MODELS: {
            ActionType.SELECT_BUILDING_MODELS, ActionType.RUN_ENERGYPLUS_BASELINE,
            ActionType.LOAD_REFERENCE_PROFILES,
        },
        ActionType.SOLVE_JOINT_RC_OPF: {
            ActionType.CALIBRATE_RC_MODELS, ActionType.LOAD_REFERENCE_PROFILES,
        },
        ActionType.RUN_ENERGYPLUS_EVENT: {
            ActionType.SELECT_BUILDING_MODELS, ActionType.RUN_ENERGYPLUS_BASELINE,
            ActionType.SOLVE_JOINT_RC_OPF,
        },
        ActionType.ALIGN_15_MIN_TIMESERIES: {
            ActionType.RUN_ENERGYPLUS_BASELINE, ActionType.RUN_ENERGYPLUS_EVENT,
            ActionType.LOAD_REFERENCE_PROFILES,
        },
        ActionType.SCALE_BEM_PORTFOLIOS: {ActionType.ALIGN_15_MIN_TIMESERIES},
        ActionType.APPLY_DER_DISPATCH: {
            ActionType.SCALE_BEM_PORTFOLIOS, ActionType.LOAD_REFERENCE_PROFILES,
            ActionType.SOLVE_JOINT_RC_OPF,
        },
        ActionType.MAP_LOADS_TO_BUSES: {
            ActionType.SCALE_BEM_PORTFOLIOS, ActionType.LOAD_REFERENCE_PROFILES,
        },
        ActionType.RUN_PANDAPOWER: {ActionType.MAP_LOADS_TO_BUSES},
        ActionType.CALCULATE_MARKET_SETTLEMENT: {
            ActionType.RUN_PANDAPOWER, ActionType.LOAD_REFERENCE_PROFILES,
        },
        ActionType.EXTRACT_METRICS: {
            ActionType.RUN_PANDAPOWER, ActionType.RUN_ENERGYPLUS_EVENT,
        },
    }
    roots = {ActionType.EXTRACT_METRICS}
    if any(
        family
        in {
            TaskFamily.RENEWABLE_ABSORPTION,
            TaskFamily.RESERVE_SERVICE,
            TaskFamily.MARKET_TRADING,
        }
        for family in specification.task_families
    ):
        roots.add(ActionType.APPLY_DER_DISPATCH)
        dependencies[ActionType.MAP_LOADS_TO_BUSES].add(ActionType.APPLY_DER_DISPATCH)
    if "settlement_usd" in specification.requested_outputs:
        roots.add(ActionType.CALCULATE_MARKET_SETTLEMENT)

    selected: set[ActionType] = set()

    def visit(action: ActionType) -> None:
        if action in selected:
            return
        selected.add(action)
        for dependency in dependencies.get(action, set()):
            visit(dependency)

    for root in roots:
        visit(root)
    return [action for action in ActionType if action in selected]
