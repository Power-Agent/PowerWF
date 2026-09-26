from __future__ import annotations

from pathlib import Path

from b2g_agent.collaboration.models import MediatedReply, MediatorPlan, SimulationScope
from b2g_agent.collaboration.session import CollaborationSession


class FakeMediator:
    configured = True
    last_backend = "fake-llm"

    def plan_turn(self, **_kwargs):  # type: ignore[no-untyped-def]
        return MediatorPlan(
            intent="simulate",
            interpreted_intent="Test a joint retrofit and capacity proposal.",
            translation_for_counterpart="The proposal changes both load and network headroom.",
            requested_changes={
                "retrofit_level": "standard",
                "demand_response_pct": 12,
                "transformer_capacity_kva": 500,
                "line_capacity_kw": 500,
            },
            simulation_scope=SimulationScope.COUPLED,
            confidence=0.98,
        )

    def compose_reply(self, **kwargs):  # type: ignore[no-untyped-def]
        run = kwargs["run"]
        return MediatedReply(
            mediator_message=f"Run {run.run_id} is grounded in the vertical backend.",
            counterpart_message="I can accept this feasible candidate for review.",
            next_question="Should we finalize it?",
            decision_note=f"Accepted candidate {run.run_id}.",
        )


def test_session_records_mediated_evidence_and_final_plan(tmp_path: Path) -> None:
    session = CollaborationSession(
        user_role="building_engineer",
        mediator=FakeMediator(),  # type: ignore[arg-type]
        output_root=tmp_path,
    )

    result = session.handle_message("Use a standard retrofit and targeted grid upgrade.")

    assert result.simulation_run is not None
    assert result.simulation_run.metrics.feasible is True
    assert len(result.messages) == 2
    assert result.llm_backend == "fake-llm"
    assert result.decision_ledger

    final = session.finalize()
    assert final.status == "ready"
    assert "OPENAI_API_KEY" not in final.report_markdown
    assert (tmp_path / session.session_id / "session.json").exists()
    assert (tmp_path / session.session_id / "final_plan.md").exists()


class DemandResponseMediator(FakeMediator):
    def plan_turn(self, **_kwargs):  # type: ignore[no-untyped-def]
        return MediatorPlan(
            intent="simulate",
            interpreted_intent="Confirm a weather-adjusted baseline and 0.8 kW commitment.",
            translation_for_counterpart="Evaluate baseline credibility, delivery, and rebound.",
            requested_changes={
                "baseline_method": "weather_adjusted",
                "dr_target_kw_per_building": 0.8,
                "max_rebound_pct": 20,
            },
            simulation_scope=SimulationScope.COUPLED,
            confidence=0.99,
        )


def test_demand_response_session_produces_enrollment_evidence(tmp_path: Path) -> None:
    session = CollaborationSession(
        user_role="distribution_power_engineer",
        scenario_id="demand_response_service",
        mediator=DemandResponseMediator(),  # type: ignore[arg-type]
        output_root=tmp_path,
    )

    result = session.handle_message("Use the weather-adjusted baseline and test 0.8 kW.")

    assert result.simulation_run is not None
    assert result.simulation_run.metrics.dr_delivery_pct >= 90
    assert session.snapshot().scenario_id.value == "demand_response_service"
    final = session.finalize()
    assert final.status == "ready"
    assert "Baseline method" in final.report_markdown
