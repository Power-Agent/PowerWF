from pathlib import Path

from b2g_agent.agents import ActorAgent, TranslatorAgent
from b2g_agent.case_io import CaseBundle
from b2g_agent.contracts import ReceiverFeedback
from b2g_agent.evaluation import EvaluationMethodRegistry
from b2g_agent.orchestration import B2GPipeline

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_comparison_registry_declares_selected_methods() -> None:
    registry = EvaluationMethodRegistry.load(
        REPO_ROOT / "cases/ieee33_10bem_15min/evaluation/methods.json"
    )
    assert {method.method_id for method in registry.methods} == {
        "full_multi_agent", "same_llm_single_agent", "plain_translator",
        "unstructured_actor", "reviewer_no_feedback",
    }
    assert all(method.ready for method in registry.methods)


class _SingleAgentStub:
    available = True

    def __init__(self, specification, plan) -> None:  # type: ignore[no-untyped-def]
        self.specification = specification
        self.plan = plan
        self.request_hashes: list[str] = []

    def call(self, response_model, *, system, user_payload, agent_name):  # type: ignore[no-untyped-def]
        del system
        self.request_hashes.append(f"stub-{agent_name}")
        if agent_name == "same_llm_single_agent_contract":
            return response_model.model_validate(
                {
                    "task_specification": self.specification.model_dump(mode="json"),
                    "plan": {
                        "actions": [
                            action.model_dump(mode="json")
                            for action in self.plan.actions
                        ]
                    },
                    "receiver_received": "Assess the translated request with the listed tools.",
                }
            )
        if agent_name == "same_llm_single_agent_result":
            return response_model.model_validate({**user_payload["tool_result"],
                "sender_message": "The requested service was accepted with validated completion."})
        raise AssertionError(agent_name)


class _HallucinatingResultStub(_SingleAgentStub):
    def call(self, response_model, *, system, user_payload, agent_name):  # type: ignore[no-untyped-def]
        if agent_name != "same_llm_single_agent_result":
            return super().call(
                response_model,
                system=system,
                user_payload=user_payload,
                agent_name=agent_name,
            )
        self.request_hashes.append(f"stub-{agent_name}")
        return response_model(
            decision="modified",
            text="I can only provide a generated partial commitment.",
            committed_power_kw=1.0,
            completion_percentage=10.0,
            completion_basis="Generated rather than tool validated.",
            sender_message="A generated partial commitment was offered.",
        )


def test_same_llm_baseline_uses_one_contract_call_without_role_split() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = case.get_episode("ep-001")
    specification = TranslatorAgent(REPO_ROOT / "knowledge").translate(
        episode.sender_message,
        episode=episode,
        sender=episode.start_role,
    )
    plan = ActorAgent().plan(specification)
    assert plan is not None
    llm = _SingleAgentStub(specification, plan)
    pipeline = B2GPipeline.default(
        repo_root=REPO_ROOT,
        method_id="same_llm_single_agent",
        llm_client=llm,  # type: ignore[arg-type]
    )
    result = pipeline.run_contract_stage(
        episode.sender_message,
        episode=episode,
        sender=episode.start_role,
    )
    assert result.execution_plan is not None and result.execution_plan.valid
    assert result.llm_request_hashes == ["stub-same_llm_single_agent_contract"]
    assert result.reviewer_decision.checked_memory_ids == []

    feedback = pipeline.single_agent.propose_feedback(  # type: ignore[union-attr]
        specification=result.task_specification,
        physics_evidence={"status": "validated_full"},
        tool_feedback=ReceiverFeedback(
            decision="accepted",
            text="I accept the physically validated request.",
            committed_power_kw=60.0,
            completion_percentage=100.0,
            completion_basis="Validated tool result.",
        ),
    )
    assert feedback.decision == "accepted"
    assert llm.request_hashes[-1] == "stub-same_llm_single_agent_result"


def test_same_llm_result_cannot_override_physics_fields() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = case.get_episode("ep-001")
    specification = TranslatorAgent(REPO_ROOT / "knowledge").translate(
        episode.sender_message,
        episode=episode,
        sender=episode.start_role,
    )
    plan = ActorAgent().plan(specification)
    assert plan is not None
    llm = _HallucinatingResultStub(specification, plan)
    pipeline = B2GPipeline.default(
        repo_root=REPO_ROOT,
        method_id="same_llm_single_agent",
        llm_client=llm,  # type: ignore[arg-type]
    )
    tool_feedback = ReceiverFeedback(
        decision="accepted",
        text="The simulator validates the full request.",
        committed_power_kw=60.0,
        completion_percentage=100.0,
        completion_basis="Validated tool result.",
    )
    feedback = pipeline.single_agent.propose_feedback(  # type: ignore[union-attr]
        specification=specification,
        physics_evidence={"status": "validated_full"},
        tool_feedback=tool_feedback,
    )
    assert feedback.text.startswith("I can only provide")
    assert feedback.decision == "accepted"
    assert feedback.committed_power_kw == 60.0
    assert feedback.completion_percentage == 100.0
    assert feedback.completion_basis == "Validated tool result."
