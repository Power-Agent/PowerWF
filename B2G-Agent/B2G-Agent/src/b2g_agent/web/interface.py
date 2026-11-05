"""Presentation-only settings and progress adapters for the personal web UI.

Agent methods, their arguments, and their return values are left unchanged.
Progress is a small, allowlisted view of the existing per-turn event journal.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from threading import RLock
import time
from uuid import uuid4

from fastapi import HTTPException

from b2g_agent.agents.llm import _usable_key
from b2g_agent.observability.journal import EventJournal, durable_write
from b2g_agent.personal_workspace import PersonalWorkspace


@dataclass
class _Activity:
    operation_id: str
    action: str
    journal_path: Path
    offset: int
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    status: str = "running"
    workflow_status: str | None = None
    events: list[dict] = field(default_factory=list)


class InterfaceRuntime:
    def __init__(self, store, repo_root: Path):
        self.store = store
        self.repo_root = repo_root
        self.settings_lock = RLock()
        self._lock = RLock()
        self._activities: dict[str, _Activity] = {}

    def settings(self) -> dict:
        llm = self.store.pipeline.translator.llm
        key = os.getenv("OPENAI_API_KEY", "").strip()
        return {
            "llm_mode": llm.mode if llm else "offline",
            "model": llm.settings.model if llm else None,
            "api_key_configured": _usable_key(key),
            "editable": self.store.workspace is not None,
            "applies_to": "new_sessions",
        }

    def update_settings(self, mode: str) -> dict:
        with self.settings_lock:
            workspace = self.store.workspace
            if workspace is None:
                raise HTTPException(409, "Settings are managed by the injected session store.")
            if mode == "live" and not self.settings()["api_key_configured"]:
                raise HTTPException(422, "No API credential is configured. The current mode was kept.")
            # Re-read the personal file to preserve settings edited outside the UI.
            settings_path = workspace.root / "settings.json"
            try:
                settings = type(workspace.settings).model_validate_json(settings_path.read_text(encoding="utf-8"))
                settings = settings.model_copy(update={"llm_mode": mode})
                candidate = PersonalWorkspace(workspace.root, settings)
                pipeline = candidate.pipeline(self.repo_root)
                durable_write(settings_path, settings.model_dump_json(indent=2) + "\n")
            except (OSError, ValueError, KeyError) as exc:
                raise HTTPException(422, "Could not apply the selected mode. The current mode was kept.") from exc
            # Existing CollaborationSession objects retain their original pipeline.
            self.store.pipeline = pipeline
            self.store.workspace = candidate
            return self.settings()

    @contextmanager
    def operation(self, session, action: str, operation_id: str | None):
        with self._lock:
            previous = self._activities.get(session.session_id)
            if previous and previous.status == "running":
                raise HTTPException(409, "This session is already processing a request.")
            prior_result = session.latest_result if action != "message" else None
            trace_id = prior_result.trace_id if prior_result else uuid4().hex
            # Match the core journal's safe identifier contract without accepting paths.
            if not trace_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in trace_id):
                raise HTTPException(422, "Invalid interaction trace identifier.")
            journal = EventJournal(session.pipeline.journal_root / f"{trace_id}.jsonl", trace_id)
            activity = _Activity(
                operation_id or uuid4().hex, action, journal.path,
                journal.path.stat().st_size if journal.path.is_file() else 0,
            )
            self._activities[session.session_id] = activity
        try:
            # This is the same journal context used by the existing pipeline decorators.
            with journal.activate():
                yield
        except BaseException:
            with self._lock:
                activity.status = "failed"
            raise
        else:
            with self._lock:
                activity.status = "completed"
                activity.workflow_status = session.latest_result.workflow_status if session.latest_result else None
        finally:
            with self._lock:
                activity.finished = time.monotonic()

    def progress(self, session_id: str, operation_id: str | None = None) -> dict:
        with self._lock:
            activity = self._activities.get(session_id)
            if activity is None or (operation_id and activity.operation_id != operation_id):
                return {"operation_id": operation_id, "status": "waiting" if operation_id else "idle", "events": []}
            self._read_events(activity)
            return {
                "operation_id": activity.operation_id,
                "action": activity.action,
                "status": activity.status,
                "elapsed_seconds": round((activity.finished or time.monotonic()) - activity.started, 1),
                "workflow_status": activity.workflow_status,
                "events": list(activity.events),
            }

    @staticmethod
    def _read_events(activity: _Activity) -> None:
        if not activity.journal_path.is_file():
            return
        with activity.journal_path.open("rb") as stream:
            stream.seek(activity.offset)
            for _ in range(512):
                position = stream.tell()
                line = stream.readline()
                if not line or not line.endswith(b"\n"):
                    activity.offset = position
                    break
                activity.offset = stream.tell()
                try:
                    record = json.loads(line)
                except (ValueError, UnicodeDecodeError):
                    continue
                event = record.get("event")
                if event not in {
                    "stage.started", "stage.completed", "stage.failed",
                    "llm.request", "llm.response", "llm.format_repair", "reviewer.return",
                    "physics.started", "physics.completed", "physics.failed",
                    "workflow.blocked", "exchange.clarification", "exchange.released",
                }:
                    continue
                payload = record.get("payload", {})
                # Never forward prompts, source text, results, exception traces, or paths.
                activity.events.append({
                    "id": record.get("event_id"),
                    "event": event,
                    "timestamp": record.get("timestamp"),
                    "span_id": record.get("span_id"),
                    "stage": payload.get("stage"),
                    "agent": payload.get("agent"),
                    "elapsed_seconds": payload.get("elapsed_seconds"),
                    "revision": payload.get("revision"),
                    "route": payload.get("route"),
                })
            activity.events[:] = activity.events[-120:]


def interface_snapshot(session) -> dict:
    snapshot = session.snapshot().model_dump(mode="json")
    llm = session.pipeline.translator.llm
    snapshot["ui"] = {
        "llm_mode": llm.mode if llm else "offline",
        "model": llm.settings.model if llm else None,
    }
    return snapshot
