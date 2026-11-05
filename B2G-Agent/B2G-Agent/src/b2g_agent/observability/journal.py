"""Durable, append-only interaction events shared by agents and subprocesses."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
import inspect
import dataclasses
import json
import os
from pathlib import Path
import threading
import time
import traceback
from uuid import uuid4

from pydantic import BaseModel


_active: ContextVar[EventJournal | None] = ContextVar("b2g_journal", default=None)
_span: ContextVar[str | None] = ContextVar("b2g_span", default=None)
_lock = threading.RLock()


def json_value(value):
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: json_value(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, datetime):
        return value.isoformat()
    if type(value).__module__.startswith("numpy") and hasattr(value, "tolist"):
        return json_value(value.tolist())
    if type(value).__name__ == "DataFrame" and hasattr(value, "to_dict"):
        return json_value(value.to_dict(orient="split"))
    if isinstance(value, (Path, Enum)):
        return str(value) if isinstance(value, Path) else value.value
    if isinstance(value, dict):
        return {str(k): (
            "[redacted]" if str(k).lower() in {"api_key", "openai_api_key", "authorization"}
            else {"variable_names": sorted(v)} if str(k).lower() in {"env", "environment"} and isinstance(v, dict)
            else json_value(v)
        ) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, type):
        return value.__name__
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return type(value).__name__


def durable_write(path: Path, text: str) -> None:
    """Atomically replace a checkpoint after flushing its bytes to local storage."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex[:8] + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    # Windows scanners/sync clients can briefly hold the destination. Retry
    # only the local atomic rename, never the LLM or physical computation.
    for attempt in range(5):
        try:
            os.replace(temporary, path)
            break
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05 * (2 ** attempt))


class EventJournal:
    def __init__(self, path: Path, trace_id: str):
        self.path = path.resolve()
        self.trace_id = trace_id

    def emit(self, event: str, **payload) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # The OS lock also serializes child simulators using this same journal.
        with _lock, self.path.with_suffix(".lock").open("a+b") as lockfile:
            lockfile.seek(0, os.SEEK_END)
            if lockfile.tell() == 0:
                lockfile.write(b"0")
                lockfile.flush()
            lockfile.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lockfile.fileno(), msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(lockfile.fileno(), fcntl.LOCK_EX)
            try:
                record = {"schema_version": "1.0", "event_id": uuid4().hex,
                          "trace_id": self.trace_id, "event": event,
                          "timestamp": datetime.now(timezone.utc).isoformat(),
                          "process_id": os.getpid(), "span_id": _span.get(),
                          "payload": json_value(payload)}
                with self.path.open("a", encoding="utf-8", newline="\n") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
            finally:
                lockfile.seek(0)
                if os.name == "nt":
                    msvcrt.locking(lockfile.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    fcntl.flock(lockfile.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def activate(self):
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)


def current_journal() -> EventJournal | None:
    journal = _active.get()
    if journal is None and os.getenv("B2G_EVENT_JOURNAL"):
        path = Path(os.environ["B2G_EVENT_JOURNAL"])
        journal = EventJournal(path, os.getenv("B2G_TRACE_ID", path.stem))
    return journal


def emit(event: str, **payload) -> None:
    journal = current_journal()
    if journal is not None:
        journal.emit(event, **payload)


def subprocess_environment() -> dict[str, str]:
    environment = os.environ.copy()
    journal = current_journal()
    if journal is not None:
        environment.update(B2G_EVENT_JOURNAL=str(journal.path), B2G_TRACE_ID=journal.trace_id)
    return environment


def traced_stage(name: str):
    """Persist inputs before work, then outputs or the full exception immediately."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            parent = _span.get()
            token = _span.set(uuid4().hex)
            try:
                bound = inspect.signature(function).bind(*args, **kwargs)
                inputs = {k: v for k, v in bound.arguments.items() if k not in {"self", "cls"}}
                emit("stage.started", stage=name, parent_span_id=parent, inputs=inputs)
                stage_started = time.perf_counter()
                try:
                    result = function(*args, **kwargs)
                except BaseException as exc:
                    emit("stage.failed", stage=name, error_type=type(exc).__name__,
                         elapsed_seconds=time.perf_counter() - stage_started,
                         error=str(exc), traceback=traceback.format_exc(),
                         stdout=getattr(exc, "stdout", None), stderr=getattr(exc, "stderr", None))
                    raise
                emit("stage.completed", stage=name, output=result,
                     elapsed_seconds=time.perf_counter() - stage_started)
                return result
            finally:
                _span.reset(token)
        return wrapped
    return decorate


def journaled_turn(function):
    """Open/continue a per-turn journal for every public pipeline entry point."""
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        previous = args[0] if args and hasattr(args[0], "trace_id") else kwargs.get("result")
        active = current_journal()
        trace_id = previous.trace_id if previous is not None else uuid4().hex
        if not trace_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in trace_id):
            raise ValueError("Unsafe interaction trace ID")
        journal = active or EventJournal(self.journal_root / f"{trace_id}.jsonl", trace_id)
        with journal.activate():
            @wraps(function)
            def checkpoint(*inner_args, **inner_kwargs):
                result = function(*inner_args, **inner_kwargs)
                result.trace_id = journal.trace_id
                result.interaction_journal = str(journal.path)
                route, exchange = result.reviewer_decision.routing_decision, result.information_exchange
                if route == "clarify_sender" and exchange.decision == "needs_clarification":
                    result.workflow_status = "needs_clarification"
                elif route == "release" and exchange.decision in {"accepted", "modified", "rejected"}:
                    result.workflow_status = "released"
                elif (route == "actor" and exchange.decision == "pending_physics"
                      and all(i.code == "E-MISSING-EVIDENCE" for i in result.reviewer_decision.issues)):
                    result.workflow_status = "awaiting_physics"
                else:
                    result.workflow_status = "blocked"
                emit("pipeline.checkpoint", entrypoint=function.__name__, result=result)
                return result
            return traced_stage("pipeline." + function.__name__)(checkpoint)(self, *args, **kwargs)
    return wrapped
