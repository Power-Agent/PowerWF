import pytest

from b2g_agent.collaboration.mediator import B2GMediator
from b2g_agent.collaboration.models import EngineerRole, ResearchScenario
from b2g_agent.collaboration.scenario import default_parameters


def test_placeholder_api_key_is_not_treated_as_configured(monkeypatch) -> None:
    monkeypatch.setenv("B2G_LLM_ENABLED", "true")
    monkeypatch.setenv("OPENAI_API_KEY", "your_openai_api_key_here")

    mediator = B2GMediator()

    assert mediator.configured is False
    assert mediator.last_backend == "not-configured"
    with pytest.raises(RuntimeError, match="requires your LLM API"):
        mediator.plan_turn(
            text="Explain the baseline.",
            scenario_id=ResearchScenario.DEMAND_RESPONSE,
            user_role=EngineerRole.POWER,
            counterpart_role=EngineerRole.BUILDING,
            parameters=default_parameters(ResearchScenario.DEMAND_RESPONSE),
            recent_messages=[],
        )
