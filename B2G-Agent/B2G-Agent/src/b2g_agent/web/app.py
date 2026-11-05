from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated, Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from b2g_agent import __version__
from b2g_agent.case_io import find_repo_root
from b2g_agent.collaboration import SessionStore
from b2g_agent.contracts import AgentRole, ReceiverFeedback, TaskFamily
from b2g_agent.simulation.status import simulator_status
from b2g_agent.web.interface import InterfaceRuntime, interface_snapshot


STATIC_DIR = Path(__file__).resolve().parent / "static"
UiOperationId = Annotated[str | None, Header(alias="X-UI-Operation-ID", pattern=r"^[A-Za-z0-9_-]{1,64}$")]


class InterfaceStaticFiles(StaticFiles):
    """Revalidate local UI assets instead of retaining an older visual build."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


class InterfaceSettingsRequest(BaseModel):
    llm_mode: Literal["offline", "live", "replay"]


class CreateSessionRequest(BaseModel):
    role: AgentRole
    mode: Literal["interactive", "benchmark"] = "interactive"
    episode_id: str | None = None


class MessageRequest(BaseModel):
    text: str = Field(min_length=1, max_length=8000)
    sender_role: AgentRole | None = None


class FeedbackRequest(BaseModel):
    turn_index: int | None = Field(default=None, ge=1)
    decision: Literal["accepted", "modified", "rejected"]
    text: str = Field(min_length=1, max_length=8000)
    committed_power_kw: float | None = Field(default=None, ge=0.0)
    completion_percentage: float = Field(ge=0.0, le=100.0)
    completion_basis: str = Field(min_length=1, max_length=2000)


class ExecuteRequest(BaseModel):
    turn_index: int | None = Field(default=None, ge=1)


def create_app(store: SessionStore | None = None) -> FastAPI:
    root = find_repo_root()
    load_dotenv(root / ".env", override=False)
    app = FastAPI(
        title="B2G-Agent",
        version=__version__,
        description="Private Translator-Actor-Reviewer interface; personal sessions are isolated from formal evaluations.",
    )
    app.state.sessions = store or SessionStore(repo_root=root)
    interface = InterfaceRuntime(app.state.sessions, root)
    app.state.interface = interface
    app.mount("/static", InterfaceStaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> HTMLResponse:
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        for filename, marker in (
            ("styles.css", "__B2G_STYLE_VERSION__"),
            ("app.js", "__B2G_SCRIPT_VERSION__"),
        ):
            version = hashlib.sha256((STATIC_DIR / filename).read_bytes()).hexdigest()[:16]
            html = html.replace(marker, version)
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @app.get("/api/health")
    def health() -> dict[str, object]:
        pipeline = app.state.sessions.pipeline
        return {
            "status": "ok",
            "service": "B2G-Agent",
            "version": __version__,
            "case_id": pipeline.case.case_id,
            "case_readiness": pipeline.case.readiness().as_dict(),
            "simulators": simulator_status(pipeline.case),
            "llm_mode": pipeline.translator.llm.mode if pipeline.translator.llm else "offline",
            "storage_scope": "personal" if app.state.sessions.workspace else "injected_test_or_custom_store",
            "llm_settings": (
                pipeline.translator.llm.settings.model_dump(mode="json")
                if pipeline.translator.llm else None
            ),
        }

    @app.get("/api/case")
    def get_case() -> dict[str, object]:
        return app.state.sessions.case.public_summary()

    @app.get("/api/ui/settings")
    def get_interface_settings() -> dict:
        with interface.settings_lock:
            return interface.settings()

    @app.put("/api/ui/settings")
    def update_interface_settings(request: InterfaceSettingsRequest) -> dict:
        return interface.update_settings(request.llm_mode)

    @app.get("/api/episodes")
    def get_episodes(
        family: TaskFamily | None = None,
        split: str | None = None,
        role: AgentRole | None = None,
        limit: int = Query(default=200, ge=1, le=200),
    ) -> dict[str, object]:
        episodes = app.state.sessions.case.list_episodes(family=family, split=split)
        if role:
            episodes = [episode for episode in episodes if episode.start_role == role]
        episodes = episodes[:limit]
        return {
            "count": len(episodes),
            "episodes": [_public_episode(episode) for episode in episodes],
        }

    @app.get("/api/episodes/{episode_id}")
    def get_episode(episode_id: str) -> dict[str, object]:
        try:
            episode = app.state.sessions.case.get_episode(episode_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return _public_episode(episode)

    @app.post("/api/sessions")
    def create_session(request: CreateSessionRequest) -> dict[str, object]:
        try:
            with interface.settings_lock:
                session = app.state.sessions.create(
                    request.role,
                    request.episode_id,
                    mode=request.mode,
                )
        except (KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return interface_snapshot(session)

    @app.get("/api/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, object]:
        return interface_snapshot(_session(app.state.sessions, session_id))

    @app.get("/api/sessions/{session_id}/progress")
    def get_progress(
        session_id: str,
        operation_id: str | None = Query(default=None, pattern=r"^[A-Za-z0-9_-]{1,64}$"),
    ) -> dict:
        _session(app.state.sessions, session_id)
        return interface.progress(session_id, operation_id)

    @app.post("/api/sessions/{session_id}/messages")
    def post_message(session_id: str, request: MessageRequest, operation_id: UiOperationId = None) -> dict[str, object]:
        session = _session(app.state.sessions, session_id)
        try:
            with interface.operation(session, "message", operation_id):
                result = session.handle_message(
                    request.text,
                    sender_role=request.sender_role,
                )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "result": result.model_dump(mode="json"),
            "snapshot": interface_snapshot(session),
        }

    @app.post("/api/sessions/{session_id}/feedback")
    def post_feedback(
        session_id: str,
        request: FeedbackRequest,
        operation_id: UiOperationId = None,
    ) -> dict[str, object]:
        session = _session(app.state.sessions, session_id)
        try:
            feedback = ReceiverFeedback.model_validate(
                request.model_dump(exclude={"turn_index"})
            )
            with interface.operation(session, "feedback", operation_id):
                result = session.handle_feedback(
                    feedback,
                    turn_index=request.turn_index,
                )
        except (ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "result": result.model_dump(mode="json"),
            "snapshot": interface_snapshot(session),
        }

    @app.post("/api/sessions/{session_id}/execute")
    def execute_physics(
        session_id: str,
        request: ExecuteRequest,
        operation_id: UiOperationId = None,
    ) -> dict[str, object]:
        session = _session(app.state.sessions, session_id)
        try:
            with interface.operation(session, "physics", operation_id):
                result = session.execute_physics(turn_index=request.turn_index)
        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "result": result.model_dump(mode="json"),
            "snapshot": interface_snapshot(session),
        }

    @app.post("/api/sessions/{session_id}/finalize")
    def finalize(session_id: str, operation_id: UiOperationId = None) -> dict[str, object]:
        try:
            session = _session(app.state.sessions, session_id)
            with interface.operation(session, "report", operation_id):
                return session.finalize().model_dump(mode="json")
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app


def _public_episode(episode: object) -> dict[str, object]:
    return {
        "episode_id": episode.episode_id,
        "split": episode.split.value,
        "primary_family": episode.primary_family.value,
        "task_families": [family.value for family in episode.task_families],
        "start_role": episode.start_role.value,
        "operating_day": episode.operating_day,
        "target_bems": episode.target_bems,
        "target_buses": episode.target_buses,
        "windows": [window.model_dump(mode="json") for window in episode.windows],
        "difficulty": episode.difficulty.model_dump(mode="json"),
        "sender_message": episode.sender_message,
    }


def _session(store: SessionStore, session_id: str):  # type: ignore[no-untyped-def]
    try:
        return store.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


app = create_app()
