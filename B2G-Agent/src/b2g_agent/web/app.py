from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from b2g_agent import __version__
from b2g_agent.collaboration.models import EngineerRole, ResearchScenario
from b2g_agent.collaboration.scenario import scenario_brief, scenario_catalog
from b2g_agent.collaboration.session import SessionStore


STATIC_DIR = Path(__file__).resolve().parent / "static"


class CreateSessionRequest(BaseModel):
    role: EngineerRole
    scenario_id: ResearchScenario


class MessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)


def create_app(store: SessionStore | None = None) -> FastAPI:
    app = FastAPI(
        title="B2G-Agent",
        version=__version__,
        description="Simulation-grounded mediation for building-grid co-design.",
    )
    app.state.sessions = store or SessionStore()
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "service": "B2G-Agent", "version": __version__}

    @app.get("/api/scenarios")
    def get_scenarios() -> dict[str, object]:
        return scenario_catalog()

    @app.get("/api/scenarios/{scenario_id}")
    def get_scenario(scenario_id: ResearchScenario) -> dict[str, object]:
        return scenario_brief(scenario_id)

    @app.post("/api/sessions")
    def create_session(request: CreateSessionRequest) -> dict[str, object]:
        try:
            session = app.state.sessions.create(request.role, request.scenario_id)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return session.snapshot().model_dump(mode="json")

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, object]:
        session = _get_session(app.state.sessions, session_id)
        return session.snapshot().model_dump(mode="json")

    @app.post("/api/sessions/{session_id}/messages")
    def post_message(session_id: str, request: MessageRequest) -> dict[str, object]:
        session = _get_session(app.state.sessions, session_id)
        try:
            result = session.handle_message(request.text)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return result.model_dump(mode="json")

    @app.post("/api/sessions/{session_id}/simulate")
    def simulate(session_id: str) -> dict[str, object]:
        session = _get_session(app.state.sessions, session_id)
        run = session.simulate_current()
        return run.model_dump(mode="json")

    @app.post("/api/sessions/{session_id}/finalize")
    def finalize(session_id: str) -> dict[str, object]:
        session = _get_session(app.state.sessions, session_id)
        plan = session.finalize()
        return plan.model_dump(mode="json")

    return app


def _get_session(store: SessionStore, session_id: str):  # type: ignore[no-untyped-def]
    try:
        return store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


app = create_app()
