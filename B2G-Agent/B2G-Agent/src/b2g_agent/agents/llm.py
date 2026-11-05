from __future__ import annotations

import json
import os
import hashlib
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypeVar
from uuid import uuid4

from pydantic import BaseModel

from b2g_agent.observability import LLMSettings, request_hash
from b2g_agent.agents.wire_schema import decode_wire_value, strict_wire_schema
from b2g_agent.agents.response_validation import validate_response
from b2g_agent.agents.token_budget import TaskTokenBudget, conservative_prompt_tokens
from b2g_agent.observability.journal import current_journal, durable_write, emit, traced_stage


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class ResponseValidationError(ValueError):
    """A received reply failed local parsing, distinct from transport errors."""


class StructuredLLMClient:
    """Strict JSON LLM calls with request hashing and locked replay support."""

    def __init__(
        self,
        *,
        mode: str,
        settings: LLMSettings,
        record_root: Path,
        client: Any | None = None,
    ) -> None:
        if mode not in {"offline", "live", "replay"}:
            raise ValueError("B2G_LLM_MODE must be offline, live, or replay")
        if mode == "live" and settings.maximum_retries:
            raise ValueError("Complete invocation recording requires maximum_retries=0")
        self.mode = mode
        self.settings = settings
        self.record_root = record_root
        self._client = client
        self.last_request_hash: str | None = None
        self.request_hashes: list[str] = []
        self.invocation_ids: list[str] = []
        self._replay_cursors: dict[str, int] = {}
        self._task_budgets: dict[str, TaskTokenBudget] = {}
        self._standalone_budget = (
            TaskTokenBudget(settings.max_total_tokens_per_task)
            if settings.max_total_tokens_per_task is not None else None
        )

    @classmethod
    def from_env(cls, *, repo_root: Path, client: Any | None = None,
                 mode: str | None = None, settings: LLMSettings | None = None,
                 record_root: Path | None = None) -> StructuredLLMClient:
        enabled = os.getenv("B2G_LLM_ENABLED", "true").strip().lower() not in {"0", "false", "no"}
        configured_mode = os.getenv("B2G_LLM_MODE", "").strip().lower()
        if mode is not None:
            pass
        elif configured_mode:
            mode = configured_mode
        else:
            mode = "live" if enabled and _usable_key(os.getenv("OPENAI_API_KEY", "")) else "offline"
        model = os.getenv("B2G_MODEL", "gpt-6-luna").strip()
        configured_effort = os.getenv("B2G_REASONING_EFFORT", "").strip().lower()
        reasoning_effort = configured_effort or (
            "none" if model == "gpt-6-luna" else None
        )
        configured_total = os.getenv("B2G_MAX_TOTAL_TOKENS_PER_TASK", "").strip()
        settings = settings or LLMSettings(
            model=model,
            reasoning_effort=reasoning_effort,
            max_total_tokens_per_task=int(configured_total) if configured_total else None,
            max_output_tokens=int(os.getenv("B2G_MAX_OUTPUT_TOKENS", "9000")),
            maximum_schema_repairs=int(os.getenv("B2G_MAX_SCHEMA_REPAIRS", "2")),
        )
        if mode == "live" and client is None:
            from openai import OpenAI

            client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                timeout=settings.request_timeout_seconds,
                max_retries=settings.maximum_retries,
            )
        record_root = record_root or Path(
            os.getenv(
                "B2G_LLM_RECORD_ROOT",
                str(repo_root / ".local-runs" / "llm_records"),
            )
        ).resolve()
        return cls(mode=mode, settings=settings, record_root=record_root, client=client)

    @property
    def available(self) -> bool:
        return self.mode in {"live", "replay"}

    def begin_task(self) -> None:
        """Start a fresh standalone task (pipeline turns use their journal path)."""
        self._standalone_budget = (
            TaskTokenBudget(self.settings.max_total_tokens_per_task)
            if self.settings.max_total_tokens_per_task is not None else None
        )

    def _task_budget(self) -> TaskTokenBudget | None:
        if self.settings.max_total_tokens_per_task is None:
            return None
        journal = current_journal()
        if journal is None:
            return self._standalone_budget
        key = str(journal.path)
        if key not in self._task_budgets:
            self._task_budgets[key] = TaskTokenBudget(
                self.settings.max_total_tokens_per_task, journal.path.with_suffix(".tokens.json")
            )
        return self._task_budgets[key]

    @traced_stage("llm.structured")
    def call(
        self,
        response_model: type[ResponseModel],
        *,
        system: str,
        user_payload: dict[str, Any],
        agent_name: str,
    ) -> ResponseModel:
        # Reviewer already owns a bounded shape-repair loop. Other agents need
        # the same protection against truncated JSON and paired-list failures.
        attempts = 0 if agent_name == "reviewer" else self.settings.maximum_schema_repairs
        payload = user_payload
        for repair in range(attempts + 1):
            try:
                return self._invoke(
                    system=system, user_payload=payload, agent_name=agent_name,
                    response_model=response_model,
                )
            except ResponseValidationError as exc:
                if repair >= attempts:
                    raise
                emit("llm.format_repair", agent=agent_name, repair=repair + 1,
                     error=str(exc), previous_request_sha256=self.last_request_hash)
                payload = {**user_payload, "format_repair": {
                    "attempt": repair + 1, "validation_error": str(exc)[:2500],
                    "instruction": "Return one concise schema-valid JSON object. Preserve source facts, "
                    "all required targets and outputs. Keep paired building/bus lists equal in length "
                    "and order. Do not duplicate evidence or invent missing values; clarify when needed.",
                }}

    @traced_stage("llm.free_text")
    def call_text(
        self,
        *,
        system: str,
        user_payload: dict[str, Any],
        agent_name: str,
    ) -> str:
        return self._invoke(
            system=system, user_payload=user_payload, agent_name=agent_name,
            response_model=None,
        )

    def _invoke(self, *, system, user_payload, agent_name, response_model):
        user_payload = _stable_prompt_payload(user_payload)
        strict_schema = _strict_json_schema(response_model) if response_model is not None else None
        response_format = (
            {"type": "json_schema", "json_schema": {
                "name": f"b2g_{agent_name}_response", "strict": True, "schema": strict_schema,
            }} if strict_schema is not None else None
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]
        payload = {
            "agent": agent_name, "model": self.settings.model,
            "settings": self.settings.model_dump(mode="json", exclude_none=True),
            "system": system, "user": user_payload,
            **({"response_schema": strict_schema} if strict_schema is not None
               else {"response_format": "unconstrained_text"}),
        }
        parsed_name = "parsed.json" if response_model is not None else "parsed.txt"
        budget = self._task_budget()
        # The optional task budget is inactive under the per-call output protocol.
        with budget.lock if budget is not None else nullcontext():
            estimated_input = conservative_prompt_tokens(messages, response_format) if budget is not None else 0
            output_limit = (
                budget.output_allowance(estimated_input, self.settings.max_output_tokens)
                if budget is not None else self.settings.max_output_tokens
            )
            # The actual request limit is part of the replay identity.
            payload["max_completion_tokens"] = output_limit
            digest = request_hash(payload)
            self.last_request_hash = digest
            self.request_hashes.append(digest)
            emit("llm.request", request_sha256=digest, request=payload, mode=self.mode)
            if self.mode == "replay":
                parsed_path, invocation_id = self._replay_artifact(digest, parsed_name)
                if budget is not None:
                    reserved = budget.reserve(estimated_input, output_limit)
                    response_json = json.loads((parsed_path.parent / "response.json").read_text(encoding="utf-8"))
                    budget.settle(reserved, response_json)
                self.invocation_ids.append(invocation_id)
                content = parsed_path.read_text(encoding="utf-8")
                return response_model.model_validate_json(content) if response_model is not None else content
            if self.mode != "live" or self._client is None:
                raise RuntimeError("An LLM call was requested while B2G_LLM_MODE=offline")

            invocation_id, invocation_dir = self._start_invocation(digest, payload)
            response_json = None
            request_latency = 0.0
            try:
                reserved = budget.reserve(estimated_input, output_limit) if budget is not None else 0
                request = {
                    "model": self.settings.model,
                    "temperature": self.settings.temperature,
                    "top_p": self.settings.top_p,
                    "seed": self.settings.seed,
                    "max_completion_tokens": output_limit,
                    "messages": messages,
                }
                if self.settings.reasoning_effort is not None:
                    request["reasoning_effort"] = self.settings.reasoning_effort
                if response_format is not None:
                    request["response_format"] = response_format
                manifest_path = invocation_dir / "manifest.json"
                running = json.loads(manifest_path.read_text(encoding="utf-8"))
                running["api_request_attempted"] = True
                durable_write(manifest_path, json.dumps(running, indent=2) + "\n")
                request_started = time.perf_counter()
                try:
                    response = self._client.chat.completions.create(**request)
                finally:
                    request_latency = time.perf_counter() - request_started
                response_json = response.model_dump(mode="json") if hasattr(response, "model_dump") else response
                # Account before validation, including truncated/malformed replies.
                emit("llm.response", invocation_id=invocation_id, request_sha256=digest,
                     response=response_json, latency_seconds=request_latency)
                durable_write(invocation_dir / "response.json", json.dumps(response_json, ensure_ascii=False, indent=2, default=str) + "\n")
                if budget is not None:
                    budget.settle(reserved, response_json)
            except Exception as exc:
                # With an optional task budget, transport errors retain their
                # reservation. SDK retries are disabled for complete recording.
                self._finish_failed_invocation(invocation_dir, invocation_id, digest, exc, response_json,
                                               latency_seconds=request_latency)
                raise
            try:
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("The LLM returned an empty response")
                normalizations = []
                if response_model is not None:
                    parsed, normalizations = validate_response(
                        response_model, decode_wire_value(json.loads(content), response_model.model_json_schema())
                    )
                    parsed_text = parsed.model_dump_json(indent=2) + "\n"
                else:
                    parsed = parsed_text = content
            except Exception as exc:
                self._finish_failed_invocation(invocation_dir, invocation_id, digest, exc, response_json,
                                               latency_seconds=request_latency)
                raise ResponseValidationError(str(exc)) from exc
            self._finish_invocation(
                invocation_dir, invocation_id=invocation_id, request_digest=digest,
                response_json=response_json, parsed_text=parsed_text,
                parsed_name=parsed_name, normalizations=normalizations,
                latency_seconds=request_latency,
            )
            if normalizations:
                emit("llm.normalized", invocation_id=invocation_id, changes=normalizations)
            return parsed

    def _start_invocation(
        self, digest: str, payload: dict[str, Any]
    ) -> tuple[str, Path]:
        request_dir = self.record_root / digest
        request_dir.mkdir(parents=True, exist_ok=True)
        request_path = request_dir / "request.json"
        serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if request_path.is_file() and request_hash(
            json.loads(request_path.read_text(encoding="utf-8"))
        ) != digest:
            raise ValueError(f"Stored LLM request does not match its content hash: {request_path}")
        if not request_path.is_file():
            durable_write(request_path, serialized)
        # Keep the immutable layout short enough for default Windows path limits.
        invocations_root = request_dir / "i"
        invocations_root.mkdir(parents=True, exist_ok=True)
        ordinal = len([path for path in invocations_root.iterdir() if path.is_dir()]) + 1
        invocation_id = f"{ordinal:06d}-{uuid4().hex[:8]}"
        invocation_dir = invocations_root / invocation_id
        invocation_dir.mkdir(parents=False, exist_ok=False)
        self.invocation_ids.append(invocation_id)
        durable_write(invocation_dir / "manifest.json",
            json.dumps(
                {
                    "schema_version": "1.0",
                    "invocation_id": invocation_id,
                    "request_sha256": digest,
                    "status": "running",
                    "api_request_attempted": False,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "started_epoch_seconds": time.time(),
                    "agent": payload.get("agent"),
                    "model": payload.get("model"),
                },
                indent=2,
            )
            + "\n",
        )
        return invocation_id, invocation_dir

    @staticmethod
    def _finish_invocation(
        invocation_dir: Path,
        *,
        invocation_id: str,
        request_digest: str,
        response_json: Any,
        parsed_text: str,
        parsed_name: str,
        normalizations: list[dict[str, Any]] | None = None,
        latency_seconds: float | None = None,
    ) -> None:
        response_text = (
            json.dumps(response_json, ensure_ascii=False, indent=2, default=str) + "\n"
        )
        response_path = invocation_dir / "response.json"
        parsed_path = invocation_dir / parsed_name
        _write_response_once(response_path, response_text)
        durable_write(parsed_path, parsed_text)
        response_sha = hashlib.sha256(response_path.read_bytes()).hexdigest()
        parsed_sha = hashlib.sha256(parsed_path.read_bytes()).hexdigest()
        usage = response_json.get("usage", {}) if isinstance(response_json, dict) else {}
        running = json.loads((invocation_dir / "manifest.json").read_text(encoding="utf-8"))
        latency = latency_seconds if latency_seconds is not None else max(0.0, time.time() - float(running["started_epoch_seconds"]))
        durable_write(invocation_dir / "manifest.json",
            json.dumps(
                {
                    "schema_version": "1.0",
                    "invocation_id": invocation_id,
                    "request_sha256": request_digest,
                    "response_sha256": response_sha,
                    "parsed_sha256": parsed_sha,
                    "parsed_artifact": parsed_name,
                    "status": "completed",
                    "api_request_attempted": running.get("api_request_attempted", True),
                    "agent": running.get("agent"),
                    "model": running.get("model"),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "latency_seconds": latency,
                    "latency_scope": "api_request_wall" if latency_seconds is not None else "legacy_invocation_wall",
                    "usage": usage,
                    "normalizations": normalizations or [],
                },
                indent=2,
                default=str,
            )
            + "\n",
        )

    @staticmethod
    def _finish_failed_invocation(
        invocation_dir: Path,
        invocation_id: str,
        request_digest: str,
        exc: Exception,
        response_json: Any = None,
        *, latency_seconds: float | None = None,
    ) -> None:
        running = json.loads((invocation_dir / "manifest.json").read_text(encoding="utf-8"))
        latency = latency_seconds if latency_seconds is not None else max(0.0, time.time() - float(running["started_epoch_seconds"]))
        response_metadata = {}
        if response_json is not None:
            response_path = invocation_dir / "response.json"
            _write_response_once(response_path, json.dumps(response_json, ensure_ascii=False, indent=2, default=str) + "\n")
            response_metadata = {
                "response_sha256": hashlib.sha256(response_path.read_bytes()).hexdigest(),
                "usage": response_json.get("usage", {}),
            }
        durable_write(invocation_dir / "manifest.json",
            json.dumps(
                {
                    "schema_version": "1.0",
                    "invocation_id": invocation_id,
                    "request_sha256": request_digest,
                    "status": "failed",
                    "api_request_attempted": running.get("api_request_attempted", True),
                    "agent": running.get("agent"),
                    "model": running.get("model"),
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "latency_seconds": latency,
                    "latency_scope": "api_request_wall" if latency_seconds is not None else "legacy_invocation_wall",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    **response_metadata,
                },
                indent=2,
            )
            + "\n",
        )

    def _replay_artifact(self, digest: str, filename: str) -> tuple[Path, str]:
        request_dir = self.record_root / digest
        legacy = request_dir / filename
        if legacy.is_file():
            return legacy, f"legacy-{digest[:16]}"
        invocations_root = request_dir / "i"
        if not invocations_root.is_dir():
            invocations_root = request_dir / "invocations"
        candidates: list[tuple[str, Path]] = []
        if invocations_root.is_dir():
            for invocation_dir in sorted(path for path in invocations_root.iterdir() if path.is_dir()):
                manifest_path = invocation_dir / "manifest.json"
                artifact_path = invocation_dir / filename
                if not manifest_path.is_file() or not artifact_path.is_file():
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "completed":
                    continue
                expected = manifest.get("parsed_sha256")
                observed = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
                if expected != observed:
                    raise ValueError(f"Locked replay artifact hash mismatch: {artifact_path}")
                candidates.append((str(manifest["invocation_id"]), artifact_path))
        cursor = self._replay_cursors.get(digest, 0)
        if cursor >= len(candidates):
            raise RuntimeError(
                f"Locked replay is missing completed invocation {cursor + 1} for request {digest}"
            )
        self._replay_cursors[digest] = cursor + 1
        return candidates[cursor][1], candidates[cursor][0]


def _write_response_once(path: Path, text: str) -> None:
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError("Refusing to overwrite a different immutable provider response")
        return
    durable_write(path, text)


def _usable_key(value: str) -> bool:
    stripped = value.strip().lower()
    return bool(stripped) and not stripped.startswith(("replace_", "your_", "example_", "<"))


def _stable_prompt_payload(value: Any) -> Any:
    """Keep measured retrieval latency local; it is not semantic model input."""
    if isinstance(value, list):
        return [_stable_prompt_payload(item) for item in value]
    if isinstance(value, dict):
        return {
            key: _stable_prompt_payload(
                {name: val for name, val in item.items() if name != "elapsed_seconds"}
                if key == "retrieval_trace" and isinstance(item, dict)
                else item
            )
            for key, item in value.items()
        }
    return value


def _strict_json_schema(response_model: type[BaseModel]) -> dict[str, Any]:
    """Use the installed OpenAI SDK's strict-schema normalizer.

    Raw Pydantic schemas leave defaulted properties optional, which violates
    the provider's strict Structured Outputs contract.
    """

    try:
        from openai.lib._pydantic import to_strict_json_schema
    except ImportError as exc:
        raise RuntimeError(
            "The installed OpenAI SDK does not expose strict Pydantic schema support"
        ) from exc
    return strict_wire_schema(to_strict_json_schema(response_model))
