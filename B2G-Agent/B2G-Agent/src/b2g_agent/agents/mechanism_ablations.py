"""Selected Translator v2 and Actor v2 mechanisms.

These replacements remove specialist assistance, not task information or physics.
No runtime path reads task-generation metadata or evaluation ground truth.
"""
from __future__ import annotations

import hashlib
import json

from b2g_agent.agents.actor import (
    ACTION_COST, ACTION_INPUT_SCHEMA, ACTION_OUTPUTS, ACTION_TO_TOOL,
    SUPPORTED_REQUESTED_OUTPUTS, PlanCompiler, _required_dependencies,
)
from b2g_agent.agents.free_plan_text import parse_free_plan
from b2g_agent.agents.translator import _conversation_payload
from b2g_agent.case_io.operating_policy import operating_constraints, public_review_context
from b2g_agent.contracts import (
    ActionType, AgentRole, ExecutionPlan, InteractionContext,
    ReverseTransferState, TaskSpecification,
)
from b2g_agent.observability.journal import emit, traced_stage

UNSTRUCTURED_VERSION = "unstructured-tool-plan-v2"


class PlainTranslator:
    """Direct same-LLM translation without retrieval or semantic prefilling."""

    def __init__(self, *, llm, case):
        self.llm, self.case = llm, case
        self.retrieval_enabled = False

    @traced_stage("translator.translate")
    def translate(self, text, *, sender, episode=None, context=None,
                  prior_specification=None, conversation_history=None, turn_index=1):
        if episode is not None and context is not None:
            raise ValueError("Provide episode or context, not both")
        if episode is not None:
            context = InteractionContext(interaction_id=episode.episode_id,
                case_id=episode.case_id, origin="benchmark_episode")
        if context is None:
            raise ValueError("Translation requires public interaction identity")
        if self.llm is None or not self.llm.available:
            raise RuntimeError("Plain-Translator requires live or locked replay")
        receiver = (AgentRole.BUILDING_ENERGY_MANAGER
                    if sender == AgentRole.DISTRIBUTION_GRID_OPERATOR
                    else AgentRole.DISTRIBUTION_GRID_OPERATOR)
        emit("retrieval.completed", agent="translator", direction="forward",
             elapsed_seconds=0.0, evidence_count=0, disabled=True)
        spec = self.llm.call(TaskSpecification, agent_name="translator",
            system=("Translate the sender request into the supplied task contract and a complete "
                "receiver_message. Interpret the original message yourself using the public case "
                "configuration. Preserve quantities, directions, units, dates, objects and priorities. "
                "Public operating bounds are case policy, not sender preferences. Ask a focused "
                "clarification for genuinely missing or conflicting critical information. Fields "
                "requiring calculation remain to_be_simulated. Do not invent measurements. "
                "Do not invent knowledge citations. Return JSON only."),
            user_payload={"source_text": text, "sender_role": sender.value,
                "receiver_role": receiver.value,
                "interaction_context": {"interaction_id": context.interaction_id,
                    "case_id": context.case_id, "origin": context.origin},
                "public_case_context": public_review_context(self.case),
                "public_operating_constraints": [c.model_dump(mode="json") for c in operating_constraints(self.case)],
                "supported_requested_outputs": sorted(SUPPORTED_REQUESTED_OUTPUTS),
                "available_operating_days": list(json.loads((self.case.case_dir /
                    "days/target_days.json").read_text(encoding="utf-8"))["target_days"]),
                # Field names/types are the executor's public API, not answers.
                # Omit bound values, request-specific mappings and family/output rules.
                "execution_field_contract": {
                    "target_power": "Exact fields key for a requested power target when applicable. Numeric value in kW; do not guess an absent or ambiguous unit.",
                    "time_resolution": "Exact fields key for the public case time resolution, expressed in min.",
                    "operating_day": "Exact fields key for the operating-day identifier; keep consistent with the top-level value. Select the exact ID from available_operating_days matching the sender's day words; do not invent an ID.",
                    "windows": "Exact fields key for the event window list; keep consistent with the top-level windows.",
                    "target_bems": "Exact fields key for the building-identifier list; keep consistent with the top-level list.",
                    "target_buses": "Exact fields key for the bus-identifier list; keep consistent with the top-level list.",
                    "objective_names": "Use the corresponding task-family identifier as objectives[].name.",
                    "resolved_statuses": ["declared", "inferred", "resolved"],
                },
                "prior_task_specification": prior_specification.model_dump(mode="json") if prior_specification else None,
                **_conversation_payload(conversation_history)})
        # Only public identity and actual provenance are adapter-owned. Never
        # fill semantic fields, select outputs, convert units, or decide ambiguity.
        spec.episode_id, spec.case_id = context.interaction_id, context.case_id
        spec.sender_role, spec.receiver_role = sender, receiver
        spec.source_text, spec.turn_index, spec.context_origin = text, turn_index, context.origin
        spec.retrieval_evidence = []
        spec.retrieval_trace = {"method_id": "plain_translator_v2", "elapsed_seconds": 0.0, "evidence_count": 0}
        for field in spec.fields.values():
            field.evidence = []
        return spec

    @traced_stage("translator.translate_feedback")
    def translate_feedback(self, feedback, *, specification, physics_evidence=None,
                           conversation_history=None):
        if self.llm is None or not self.llm.available:
            raise RuntimeError("Plain-Translator requires live or locked replay")
        emit("retrieval.completed", agent="translator", direction="reverse",
             elapsed_seconds=0.0, evidence_count=0, disabled=True)
        state = self.llm.call(ReverseTransferState, agent_name="translator_feedback",
            system=("Translate the receiver feedback to the original sender using the supplied "
                "reverse-state interface. Preserve the decision, committed_power, "
                "completion_percentage, completion_basis and requested numerical fields. Use "
                "actual physics evidence where available; never generate physical measurements. "
                "Write translated_message in the sender's terminology. Return JSON only."),
            user_payload={"receiver_feedback": feedback.model_dump(mode="json"),
                "task_specification": specification.model_dump(mode="json"),
                "physics_evidence": physics_evidence,
                **_conversation_payload(conversation_history)})
        state.episode_id = specification.episode_id
        state.source_role, state.target_role = specification.receiver_role, specification.sender_role
        state.source_text = feedback.text
        state.retrieval_evidence = []
        state.retrieval_trace = {"method_id": "plain_translator_v2", "elapsed_seconds": 0.0, "evidence_count": 0}
        return state


def ordinary_tool_catalog():
    """Individual tool interfaces only: no DAG, selection rules or bound values."""
    notes = {
        "load_case": "Load the public case. selection_policy is the executor label 'backward_minimum_cost'; estimated_cost_units is declared bookkeeping only.",
        "select_building_models": "Select building models using a list of BEM identifiers.",
        "load_reference_profiles": "Load background, PV and price profiles for an operating-day identifier.",
        "run_energyplus_baseline": "Calculate baseline building load and zone temperature from models and reference profiles.",
        "calibrate_rc_models": "Identify RC models from building models, baseline simulation and reference profiles.",
        "solve_joint_rc_opf": "Optimize schedules using calibrated RC models and reference profiles. objectives and constraints are lists; constraint entries have name, metric, operator, value, unit, hard. maximum_iterations is 10.",
        "run_energyplus_event": "Validate candidate schedules with building models and baseline simulation for the declared windows.",
        "align_15_min_timeseries": "Align baseline, response and reference profiles to the public time resolution.",
        "scale_bem_portfolios": "Aggregate aligned building profiles into BEM portfolios.",
        "apply_der_dispatch": "Apply candidate DER dispatch to portfolio and reference profiles for services using DER.",
        "map_loads_to_buses": "Map net portfolio and reference profiles to the requested bus list.",
        "run_pandapower": "Run nonlinear AC power flow on mapped nodal loads.",
        "calculate_market_settlement": "Calculate market settlement from grid validation and price profiles.",
        "extract_metrics": "Collect requested outputs from grid and event simulation and, when requested, market settlement.",
    }
    return [{"action_type": a.value, "tool": ACTION_TO_TOOL[a],
        "description": notes[a.value], "input_fields": ACTION_INPUT_SCHEMA[a],
        "output_fields": ACTION_OUTPUTS.get(a, ["<requested_outputs>"]),
        "cost_units": ACTION_COST[a]} for a in ActionType]


class UnstructuredActor:
    """Unassisted text tool planner with the unchanged physical capability set."""

    def __init__(self, *, llm):
        self.llm = llm

    @traced_stage("actor.free_plan")
    def plan(self, specification, *, reviewer_feedback=None, attempt=0):
        if specification.clarification_required:
            return None
        if self.llm is None or not self.llm.available:
            raise RuntimeError("Unstructured-Actor requires live or locked replay")
        text = self.llm.call_text(agent_name="free_plan_actor",
            system=("Plan the tool calls needed to fulfil the task using the ordinary tool "
                "interfaces. Choose tools, execution order, input values and data dependencies "
                "yourself. Write an explicit text call sequence. For each call use a paragraph "
                "starting 'action_name using tool_name', followed by Depends on:, Inputs:, and "
                "Outputs:. Dependencies reference earlier action names or none. Inputs are named "
                "concrete JSON values; use {} for no inputs. Outputs are explicit names. These "
                "clauses are only a readable execution interface, not a reference plan. Do not "
                "invent results. No omitted action, value, output or dependency will be supplied "
                "by the adapter. Respond with the plan only."),
            user_payload={"task_specification": specification.model_dump(mode="json"),
                "tools": ordinary_tool_catalog(), "reviewer_feedback": reviewer_feedback or [],
                "planning_attempt": attempt})
        actions, errors = parse_free_plan(text)
        if not actions:
            emit("free_plan.parse_failed", errors=errors)
            return None
        digest = hashlib.sha256(json.dumps([a.model_dump(mode="json") for a in actions],
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
        plan = ExecutionPlan(episode_id=specification.episode_id, plan_id=digest,
            actions=actions, requested_outputs=list(specification.requested_outputs),
            compiler_version=UNSTRUCTURED_VERSION)
        plan.validation_errors = [*validate_unstructured_plan(plan, specification), *errors]
        plan.valid = not plan.validation_errors
        return plan


def validate_unstructured_plan(plan, specification):
    """Same finite tools/physical input checks; dependency reachability is legal.

    Do not demand one exact edge list, normalize a model's parameters, add
    actions, or fill its output contracts. Cost bookkeeping has no physics role.
    """
    legacy = plan.model_copy(update={"compiler_version": PlanCompiler.version})
    errors = PlanCompiler().validate(legacy, specification)
    errors = [e for e in errors if not e.startswith("dependency_contract_mismatch:")]
    for action in plan.actions:
        if action.action_type == ActionType.LOAD_CASE:
            # Only this nonphysical annotation is ignored, not case/policy binding.
            from b2g_agent.agents.actor import _required_input_bindings, _required_action_types
            expected = _required_input_bindings(specification, _required_action_types(specification))[ActionType.LOAD_CASE]
            if {k:v for k,v in action.inputs.items() if k != "estimated_cost_units"} == {k:v for k,v in expected.items() if k != "estimated_cost_units"}:
                errors = [e for e in errors if e != "input_binding_mismatch:load_case"]
    by_id = {a.action_id:a for a in plan.actions}
    required = _required_dependencies({a.action_type for a in plan.actions})
    for action in plan.actions:
        seen, pending = set(), list(action.depends_on)
        while pending:
            node = pending.pop()
            if node in seen or node not in by_id:
                continue
            seen.add(node)
            pending.extend(by_id[node].depends_on)
        found = {by_id[n].action_type for n in seen}
        if not required.get(action.action_type, set()).issubset(found):
            errors.append(f"missing_data_dependency:{action.action_type.value}")
    return errors


