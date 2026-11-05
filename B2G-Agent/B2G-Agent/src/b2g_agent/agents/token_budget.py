"""Local, cumulative input/output accounting for one task execution.

Chat Completions does not expose an exact offline prompt counter. Reserve one
token per UTF-8 byte of the messages and schema, plus framing headroom. This is
deliberately conservative, not a provider-exact tokenizer. Never divide bytes
by four: that can underestimate JSON, identifiers and multilingual requests.
"""
from __future__ import annotations

import json
from pathlib import Path
from threading import RLock
from typing import Any

from b2g_agent.observability.journal import durable_write, emit


class TaskTokenBudgetExceeded(RuntimeError):
    """No further request fits within this task's input + output budget."""


class TokenUsageUnavailable(RuntimeError):
    """Keep the reservation when provider usage is unknown; do not retry."""


def conservative_prompt_tokens(messages: list[dict], response_format: dict | None) -> int:
    payload: dict[str, Any] = {"messages": messages}
    if response_format is not None:
        payload["response_format"] = response_format
    return len(json.dumps(payload, ensure_ascii=False).encode("utf-8")) + 1024


class TaskTokenBudget:
    def __init__(self, limit: int, state_path: Path | None = None) -> None:
        self.limit = limit
        self.state_path = state_path
        self.lock = RLock()
        self.state = {
            "limit": limit,
            "scope": "one_task_execution_all_agents_and_revisions",
            "counting": "input_plus_output_including_cached_input",
            "preflight_estimator": "utf8_bytes_plus_1024_framing_headroom",
            "charged_tokens": 0,
            "observed_total_tokens": 0,
            "reserved_tokens": 0,
            "completed_calls": 0,
        }
        if state_path is not None and state_path.is_file():
            self.state = json.loads(state_path.read_text(encoding="utf-8"))
            if self.state["limit"] != limit:
                raise ValueError("Cannot change the token limit during an existing task")

    def snapshot(self) -> dict[str, Any]:
        return {**self.state, "remaining_tokens": max(0, self.limit - self.state["charged_tokens"])}

    def output_allowance(self, prompt_tokens: int, per_call_limit: int) -> int:
        if self.state["reserved_tokens"]:
            # An earlier interrupted/failed request has unknown billed usage.
            return 0
        return max(0, min(per_call_limit, self.limit - self.state["charged_tokens"] - prompt_tokens))

    def reserve(self, prompt_tokens: int, output_tokens: int) -> int:
        amount = prompt_tokens + output_tokens
        if output_tokens < 1 or amount > self.limit - self.state["charged_tokens"]:
            emit("llm.budget_blocked", budget=self.snapshot(), estimated_input_tokens=prompt_tokens)
            raise TaskTokenBudgetExceeded(
                f"Task token budget {self.limit} (input + output): "
                f"{self.state['charged_tokens']} already charged; next input conservatively "
                f"requires {prompt_tokens} tokens. No API request was sent."
            )
        self.state["charged_tokens"] += amount
        self.state["reserved_tokens"] += amount
        self._persist("llm.budget_reserved")
        return amount

    def settle(self, reserved: int, response: dict[str, Any]) -> None:
        usage = response.get("usage") or {}
        prompt, completion, total = (usage.get(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens"))
        if not all(type(n) is int and n >= 0 for n in (prompt, completion)):
            self._persist("llm.budget_usage_unknown")
            raise TokenUsageUnavailable("Provider input/output usage is missing; reservation retained, no automatic retry")
        if total is None:
            total = prompt + completion
        if type(total) is not int or total != prompt + completion:
            self._persist("llm.budget_usage_unknown")
            raise TokenUsageUnavailable("Provider token usage is inconsistent; reservation retained")
        self.state["charged_tokens"] += total - reserved
        self.state["reserved_tokens"] -= reserved
        self.state["observed_total_tokens"] += total
        self.state["completed_calls"] += 1
        self._persist("llm.budget_settled")
        if self.state["charged_tokens"] > self.limit:
            raise TaskTokenBudgetExceeded("Provider usage exceeded the local reservation; task stopped")

    def _persist(self, event: str) -> None:
        if self.state_path is not None:
            durable_write(self.state_path, json.dumps(self.state, indent=2) + "\n")
        emit(event, budget=self.snapshot())
