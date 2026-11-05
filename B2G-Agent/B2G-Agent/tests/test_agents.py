from __future__ import annotations

import json
from pathlib import Path

import pytest

from b2g_agent.agents import ActorAgent, ReviewerAgent, TranslatorAgent
from b2g_agent.agents.actor import PlanCompiler
from b2g_agent.case_io import CaseBundle
from b2g_agent.contracts import (
    ActionType,
    AgentRole,
    InteractionContext,
    PlanProposal,
    ReceiverFeedback,
    ReviewVerdict,
    TaskFamily,
    TaskSpecification,
)
from b2g_agent.orchestration import B2GPipeline, apply_receiver_feedback


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_direct_episode_compiles_a_valid_dag_and_waits_for_evidence() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = next(item for item in case.iter_episodes() if item.difficulty.communication == "direct")
    specification = TranslatorAgent(REPO_ROOT / "knowledge").translate(
        episode.sender_message, episode=episode, sender=episode.start_role
    )
    plan = ActorAgent().plan(specification)
    decision = ReviewerAgent().review(specification, plan, simulation_status="not_run")
    assert specification.clarification_required is False
    assert plan is not None and plan.valid is True
    assert {action.action_type for action in plan.actions} >= {
        ActionType.CALIBRATE_RC_MODELS,
        ActionType.SOLVE_JOINT_RC_OPF,
        ActionType.RUN_ENERGYPLUS_EVENT,
        ActionType.RUN_PANDAPOWER,
    }
    assert len({action.action_id for action in plan.actions}) == len(plan.actions)
    assert decision.verdict == ReviewVerdict.ABSTAIN
    assert decision.semantic_risk_score == 0.0
    assert decision.evidence_risk_score > 0.0
    assert decision.routing_decision == "actor"


def test_voltage_range_is_parsed_as_a_positive_band() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = case.get_episode("ep-001")
    specification = TranslatorAgent(REPO_ROOT / "knowledge").translate(
        episode.sender_message,
        episode=episode,
        sender=episode.start_role,
    )

    assert specification.fields["voltage_band"].value == [0.95, 1.05]
    assert "voltage_threshold" not in specification.fields
    assert specification.task_families == [TaskFamily.PEAK_SHAVING]
    assert [objective.name for objective in specification.objectives] == ["peak_shaving"]


def test_actor_rejects_an_output_the_physics_layer_cannot_produce() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = case.get_episode("ep-001")
    specification = TranslatorAgent(REPO_ROOT / "knowledge").translate(
        episode.sender_message,
        episode=episode,
        sender=episode.start_role,
    )
    specification.requested_outputs = [
        *specification.requested_outputs,
        "unsupported_metric",
    ]

    plan = ActorAgent().plan(specification)

    assert plan is not None and plan.valid is False
    assert "unsupported_requested_output:unsupported_metric" in plan.validation_errors


def test_implicit_unit_is_routed_back_to_translator() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = next(item for item in case.iter_episodes() if item.difficulty.communication == "implicit_unit")
    specification = TranslatorAgent(REPO_ROOT / "knowledge").translate(
        episode.sender_message, episode=episode, sender=episode.start_role
    )
    plan = ActorAgent().plan(specification)
    decision = ReviewerAgent().review(specification, plan)
    assert specification.clarification_required is True
    assert isinstance(specification.fields["target_power"].value, float)
    assert specification.fields["target_power"].unit is None
    assert plan is None
    assert decision.verdict == ReviewVerdict.REVISE
    assert decision.issues[0].responsible_agent == "translator"
    assert decision.semantic_risk_score == 1.0
    assert decision.routing_decision == "clarify_sender"


def test_plan_hash_is_stable() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = next(item for item in case.iter_episodes() if item.difficulty.communication == "direct")
    specification = TranslatorAgent(REPO_ROOT / "knowledge").translate(
        episode.sender_message, episode=episode, sender=episode.start_role
    )
    actor = ActorAgent()
    first = actor.plan(specification)
    second = actor.plan(specification)
    assert first is not None and second is not None
    assert first.plan_id == second.plan_id


def test_actor_rejects_incomplete_dependencies_and_unbound_inputs() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = case.get_episode("ep-001")
    specification = TranslatorAgent(REPO_ROOT / "knowledge").translate(
        episode.sender_message, episode=episode, sender=episode.start_role
    )
    compiler = PlanCompiler()
    reference = compiler.compile(specification)
    actions = [action.model_copy(deep=True) for action in reference.actions]
    event = next(
        action
        for action in actions
        if action.action_type == ActionType.RUN_ENERGYPLUS_EVENT
    )
    event.depends_on = []
    mapping = next(
        action for action in actions if action.action_type == ActionType.MAP_LOADS_TO_BUSES
    )
    mapping.inputs = {"target_buses": [33]}

    invalid = compiler.compile_proposal(
        specification,
        PlanProposal(actions=actions),
    )

    assert invalid.valid is False
    assert "dependency_contract_mismatch:run_energyplus_event" in invalid.validation_errors
    assert "input_binding_mismatch:map_loads_to_buses" in invalid.validation_errors


def test_compiler_derives_cost_without_repairing_physical_inputs() -> None:
    episode = CaseBundle.default(REPO_ROOT).get_episode("ep-021")
    specification = TranslatorAgent(REPO_ROOT / "knowledge").translate(
        episode.sender_message, episode=episode, sender=episode.start_role
    )
    compiler = PlanCompiler()
    reference = compiler.compile(specification)
    proposal = PlanProposal(actions=[action.model_copy(deep=True) for action in reference.actions])
    proposal.actions[0].inputs["estimated_cost_units"] = -100
    plan = compiler.compile_proposal(specification, proposal)
    assert plan.valid
    assert plan.actions[0].inputs == reference.actions[0].inputs
    assert proposal.actions[0].inputs["estimated_cost_units"] == -100
    mapping = next(action for action in proposal.actions if action.action_type == ActionType.MAP_LOADS_TO_BUSES)
    mapping.inputs["target_buses"] = [33]
    assert not compiler.compile_proposal(specification, proposal).valid


class _StructuredStub:
    available = True

    def __init__(self) -> None:
        self.request_hashes: list[str] = []
        self.agents: list[str] = []

    def call(self, response_model, *, system, user_payload, agent_name):  # type: ignore[no-untyped-def]
        del system
        self.agents.append(agent_name)
        self.request_hashes.append(f"stub-{agent_name}")
        if agent_name == "translator":
            return response_model.model_validate(user_payload["deterministic_draft"])
        if agent_name == "actor":
            specification = TaskSpecification.model_validate(
                user_payload["task_specification"]
            )
            return response_model.model_validate(
                {
                    "actions": [
                        action.model_dump(mode="json")
                        for action in PlanCompiler().compile(specification).actions
                    ]
                }
            )
        from reviewer_fixtures import categorical_fixture
        return response_model.model_validate(categorical_fixture(
            {
                "status": "passed",
                "issues": [],
                "checks": [
                    {
                        "check_id": check_id,
                        "channel": "semantic",
                        "score": 0.0,
                        "message": "The reviewed semantic field is consistent.",
                    }
                    for check_id in (
                        "semantic_objective",
                        "semantic_action_direction",
                        "semantic_objects",
                        "semantic_quantities_units",
                        "semantic_time_windows",
                        "semantic_constraints",
                        "semantic_requested_outputs",
                        "semantic_decision_completion",
                    )
                ],
            }, user_payload["review_phase"]
        ))


def test_live_contract_path_invokes_all_three_schema_constrained_agents() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = next(item for item in case.iter_episodes() if item.difficulty.communication == "direct")
    llm = _StructuredStub()
    pipeline = B2GPipeline.default(repo_root=REPO_ROOT, llm_client=llm)  # type: ignore[arg-type]
    result = pipeline.run_contract_stage(
        episode.sender_message,
        episode=episode,
        sender=episode.start_role,
        simulation_status="not_run",
    )
    assert llm.agents == ["translator", "actor", "reviewer"]
    assert result.llm_request_hashes == ["stub-translator", "stub-actor", "stub-reviewer"]
    assert result.execution_plan is not None and result.execution_plan.valid
    assert result.reviewer_decision.semantic_review_status == "passed"


def test_direct_task_stops_before_inventing_receiver_feedback() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = next(item for item in case.iter_episodes() if item.difficulty.communication == "direct")
    result = B2GPipeline.default(repo_root=REPO_ROOT).run_contract_stage(
        episode.sender_message,
        episode=episode,
        sender=episode.start_role,
    )
    exchange = result.information_exchange
    assert exchange.sender_sent == episode.sender_message
    assert exchange.receiver_received
    assert exchange.decision == "pending_physics"
    assert exchange.receiver_sent is None
    assert exchange.sender_received is None
    assert exchange.completion_percentage is None
    pending_decision = result.reviewer_decision
    assert pending_decision.evidence_risk_score == pytest.approx(2.5 / 9.0)
    assert pending_decision.routing_decision == "actor"


def test_receiver_feedback_closes_exchange_with_quantified_completion() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = next(item for item in case.iter_episodes() if item.difficulty.communication == "direct")
    pending = B2GPipeline.default(repo_root=REPO_ROOT).run_contract_stage(
        episode.sender_message,
        episode=episode,
        sender=episode.start_role,
    ).information_exchange
    accepted = apply_receiver_feedback(
        pending,
        ReceiverFeedback(
            decision="accepted",
            text=pending.receiver_received,
            completion_percentage=100.0,
            completion_basis="The receiver accepted every requested term.",
        ),
    )
    assert accepted.receiver_sent == accepted.receiver_received
    assert accepted.completion_percentage == 100.0
    assert "100% complete" in accepted.sender_received

    modified = apply_receiver_feedback(
        pending,
        ReceiverFeedback(
            decision="modified",
            text="I can deliver 75% of the requested service while preserving comfort.",
            completion_percentage=75.0,
            completion_basis="Deliverable service divided by requested service.",
        ),
    )
    assert modified.completion_percentage == 75.0
    assert "75%" in modified.sender_received


def test_single_clarification_feedback_closes_unexecutable_task_at_zero_percent() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = next(
        item for item in case.iter_episodes()
        if item.difficulty.communication == "conflicting_constraints"
    )
    exchange = B2GPipeline.default(repo_root=REPO_ROOT).run_contract_stage(
        episode.sender_message,
        episode=episode,
        sender=episode.start_role,
    ).information_exchange
    assert exchange.decision == "needs_clarification"
    assert exchange.receiver_sent
    assert exchange.sender_received
    assert exchange.completion_percentage == 0.0
    assert "0% complete" in exchange.sender_received


def test_live_translator_accepts_unfrozen_wording_and_carries_context_forward() -> None:
    translator = TranslatorAgent(REPO_ROOT / "knowledge")
    first = translator.translate(
        (
            "On 2026-07-15, ask BEM 01 at bus 3 to provide 80 kW of peak "
            "shaving during 16:00-18:00."
        ),
        sender=AgentRole.DISTRIBUTION_GRID_OPERATOR,
        context=InteractionContext(
            interaction_id="live-turn-001",
            case_id="ieee33_10bem_15min",
            origin="interactive_inferred",
        ),
    )
    assert first.clarification_required is False
    assert first.task_families == [TaskFamily.PEAK_SHAVING]
    assert first.operating_day == "2026-07-15"
    assert first.target_bems == ["bem_01"]
    assert first.target_buses == [3]

    follow_up = translator.translate(
        "For the next turn, reduce the request to 60 kW during 17:00-18:00.",
        sender=AgentRole.BUILDING_ENERGY_MANAGER,
        context=InteractionContext(
            interaction_id="live-turn-002",
            case_id="ieee33_10bem_15min",
            origin="interactive_inferred",
        ),
        prior_specification=first,
        conversation_history=[{"speaker": "sender", "text": first.source_text}],
        turn_index=2,
    )
    assert follow_up.turn_index == 2
    assert follow_up.sender_role == AgentRole.BUILDING_ENERGY_MANAGER
    assert follow_up.task_families == [TaskFamily.PEAK_SHAVING]
    assert follow_up.operating_day == "2026-07-15"
    assert follow_up.target_bems == ["bem_01"]
    assert follow_up.target_buses == [3]
    assert follow_up.fields["target_power"].value == 60.0
    assert follow_up.clarification_required is False


def test_live_follow_up_can_resolve_an_implicit_power_unit() -> None:
    translator = TranslatorAgent(REPO_ROOT / "knowledge")
    context = InteractionContext(
        interaction_id="live-unit-resolution",
        case_id="ieee33_10bem_15min",
        origin="interactive_inferred",
    )
    first = translator.translate(
        (
            "On 2026-07-15, for peak shaving, request BEM 01 at bus 3 "
            "to provide 80 during 16:00-18:00."
        ),
        sender=AgentRole.DISTRIBUTION_GRID_OPERATOR,
        context=context,
    )
    assert first.fields["target_power"].value == 80.0
    assert first.fields["target_power"].unit is None
    assert first.clarification_required is True

    resolved = translator.translate(
        "Use kW as the unit.",
        sender=AgentRole.DISTRIBUTION_GRID_OPERATOR,
        context=context,
        prior_specification=first,
        turn_index=2,
    )
    assert resolved.fields["target_power"].value == 80.0
    assert resolved.fields["target_power"].unit == "kW"
    assert resolved.fields["target_power"].status.value == "resolved"
    assert resolved.clarification_required is False


def test_all_frozen_messages_match_contract_oracles_without_hidden_metadata() -> None:
    case = CaseBundle.default(REPO_ROOT)
    oracle_path = case.case_dir / "groundtruth" / "contract_oracles.jsonl"
    oracles = {
        payload["episode_id"]: payload["task_specification"]
        for payload in (
            json.loads(line)
            for line in oracle_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    translator = TranslatorAgent(REPO_ROOT / "knowledge")
    for episode in case.iter_episodes():
        observed = translator.translate(
            episode.sender_message,
            episode=episode,
            sender=episode.start_role,
        )
        expected = oracles[episode.episode_id]
        expected_power = expected["fields"]["target_power"]
        assert [item.value for item in observed.task_families] == expected["task_families"]
        assert observed.operating_day == expected["operating_day"]
        assert observed.target_bems == expected["target_bems"]
        assert observed.target_buses == expected["target_buses"]
        assert observed.model_dump(mode="json")["windows"] == expected["windows"]
        assert observed.fields["target_power"].value == expected_power["value"]
        assert observed.fields["target_power"].unit == expected_power["unit"]
        assert observed.fields["target_power"].status.value == expected_power["status"]
        assert observed.model_dump(mode="json")["objectives"] == expected["objectives"]
        assert observed.requested_outputs == expected["requested_outputs"]
        assert observed.clarification_required == expected["clarification_required"]
