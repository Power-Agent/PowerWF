from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from b2g_agent.collaboration.models import MediatedReply, MediatorPlan, SimulationScope
from b2g_agent.collaboration.session import SessionStore
from b2g_agent.web.app import create_app


class FakeMediator:
    configured = True
    last_backend = "fake-llm"

    def plan_turn(self, **_kwargs):  # type: ignore[no-untyped-def]
        return MediatorPlan(
            intent="simulate",
            interpreted_intent="Run a validated joint scenario.",
            translation_for_counterpart="Translate the building change into grid evidence.",
            requested_changes={"transformer_capacity_kva": 500, "line_capacity_kw": 500},
            simulation_scope=SimulationScope.COUPLED,
            confidence=1.0,
        )

    def compose_reply(self, **kwargs):  # type: ignore[no-untyped-def]
        run = kwargs["run"]
        return MediatedReply(
            mediator_message="The coupled run completed.",
            counterpart_message="I reviewed the supplied evidence.",
            next_question="Compare or finalize?",
            decision_note=f"Reviewed {run.run_id}.",
        )


def test_web_vertical_slice_api(tmp_path: Path) -> None:
    store = SessionStore(mediator_factory=FakeMediator, output_root=tmp_path)  # type: ignore[arg-type]
    client = TestClient(create_app(store))

    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["service"] == "B2G-Agent"

    scenarios = client.get("/api/scenarios")
    assert scenarios.status_code == 200
    catalog = scenarios.json()["scenarios"]
    assert [item["scenario_id"] for item in catalog] == [
        "distribution_grid_upgrade",
        "demand_response_service",
    ]
    assert [item["title"] for item in catalog] == [
        "Distribution Grid Upgrade",
        "Demand Response Service",
    ]

    created = client.post(
        "/api/sessions",
        json={
            "role": "building_engineer",
            "scenario_id": "distribution_grid_upgrade",
        },
    )
    assert created.status_code == 200
    session = created.json()
    assert session["counterpart_role"] == "distribution_power_engineer"
    assert "OPENAI_API_KEY" not in created.text

    turn = client.post(
        f"/api/sessions/{session['session_id']}/messages",
        json={"text": "Test the joint capacity plan."},
    )
    assert turn.status_code == 200
    assert turn.json()["simulation_run"]["scope"] == "coupled"

    final = client.post(f"/api/sessions/{session['session_id']}/finalize")
    assert final.status_code == 200
    assert "report_markdown" in final.json()


def test_web_api_creates_demand_response_session(tmp_path: Path) -> None:
    store = SessionStore(mediator_factory=FakeMediator, output_root=tmp_path)  # type: ignore[arg-type]
    client = TestClient(create_app(store))

    brief = client.get("/api/scenarios/demand_response_service")
    assert brief.status_code == 200
    assert "baseline_method" in brief.json()["allowed_parameters"]

    created = client.post(
        "/api/sessions",
        json={
            "role": "distribution_power_engineer",
            "scenario_id": "demand_response_service",
        },
    )
    assert created.status_code == 200
    payload = created.json()
    assert payload["scenario_id"] == "demand_response_service"
    assert payload["runs"][0]["metrics"]["baseline_peak_kw_per_building"] > 0
