from __future__ import annotations

import json
import pytest
from pathlib import Path
from types import SimpleNamespace

from b2g_agent.agents.llm import StructuredLLMClient, _strict_json_schema, _stable_prompt_payload
from b2g_agent.contracts import PlanProposal, ReceiverFeedback, SemanticReviewResult, TaskSpecification
from b2g_agent.agents.wire_schema import decode_wire_value
from b2g_agent.contracts.semantic_audit import SemanticAuditResult
from b2g_agent.observability import LLMSettings


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]
        self._content = content

    def model_dump(self, *, mode: str) -> dict[str, object]:
        assert mode == "json"
        return {
            "choices": [{"message": {"content": self._content}}],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(self.content)


def _fake_client(content: str) -> tuple[object, _FakeCompletions]:
    completions = _FakeCompletions(content)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions)), completions


def test_strict_schema_requires_every_top_level_property() -> None:
    schema = _strict_json_schema(SemanticReviewResult)
    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False


def test_retrieval_latency_cannot_change_prompt_or_mutate_local_telemetry() -> None:
    payload = {"spec": {"retrieval_trace": {"elapsed_seconds": 0.12, "evidence_count": 6}}, "value": 165}
    frozen = _stable_prompt_payload(payload)
    assert payload["spec"]["retrieval_trace"]["elapsed_seconds"] == 0.12
    payload["spec"]["retrieval_trace"]["elapsed_seconds"] = 1.0
    assert _stable_prompt_payload(payload) == frozen
    payload["value"] = 170
    assert _stable_prompt_payload(payload) != frozen


@pytest.mark.parametrize("model", [TaskSpecification, PlanProposal, SemanticReviewResult, SemanticAuditResult, ReceiverFeedback])
def test_live_schema_contains_no_open_objects_or_untyped_values(model) -> None:
    def check(node):
        assert any(key in node for key in ("type", "$ref", "anyOf", "allOf"))
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])
            for value in node["properties"].values():
                check(value)
        for value in node.get("$defs", {}).values():
            check(value)
        if "items" in node:
            check(node["items"])
        for name in ("anyOf", "allOf"):
            for value in node.get(name, []):
                check(value)
    check(_strict_json_schema(model))


def test_dynamic_plan_input_transport_preserves_types_and_rejects_duplicate_keys() -> None:
    action = {
        "action_id": "load", "action_type": "load_case", "tool": "case_io",
        "depends_on": [], "outputs": [],
        "inputs": [
            {"key": "case_id", "value": "research-case"},
            {"key": "target_buses", "value": [3,18]},
            {"key": "settings", "value": {"entries": [
                {"key": "limit", "value": 165}, {"key": "enabled", "value": True},
                {"key": "missing", "value": None},
            ]}},
        ],
    }
    decoded = decode_wire_value({"actions": [action]}, PlanProposal.model_json_schema())
    plan = PlanProposal.model_validate(decoded)
    assert plan.actions[0].inputs == {
        "case_id": "research-case", "target_buses": [3, 18],
        "settings": {"limit": 165, "enabled": True, "missing": None},
    }
    action["inputs"].append({"key": "case_id", "value": "duplicate"})
    with pytest.raises(ValueError, match="Duplicate mapping key"):
        decode_wire_value({"actions": [action]}, PlanProposal.model_json_schema())


def test_failed_validation_retains_raw_response_and_usage(tmp_path: Path) -> None:
    client_object, _ = _fake_client('{"decision":"invalid"}')
    live = StructuredLLMClient(mode="live", settings=LLMSettings(), record_root=tmp_path, client=client_object)
    with pytest.raises(ValueError):
        live.call(ReceiverFeedback, system="Test", user_payload={}, agent_name="test")
    root = tmp_path / live.request_hashes[0] / "i" / live.invocation_ids[0]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["usage"]["total_tokens"] == 15
    assert (root / "response.json").is_file()


def test_gpt6_luna_uses_reproducible_chat_profile(
    tmp_path: Path,
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("B2G_LLM_MODE", "offline")
    monkeypatch.setenv("B2G_MODEL", "gpt-6-luna")
    monkeypatch.delenv("B2G_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("B2G_MAX_TOTAL_TOKENS_PER_TASK", raising=False)
    configured = StructuredLLMClient.from_env(repo_root=tmp_path)
    assert configured.settings.reasoning_effort == "none"

    result = ReceiverFeedback(
        decision="accepted",
        text="I accept the validated request.",
        committed_power_kw=10.0,
        completion_percentage=100.0,
        completion_basis="Validated tool result.",
    )
    client_object, completions = _fake_client(result.model_dump_json())
    live = StructuredLLMClient(
        mode="live",
        settings=LLMSettings(model="gpt-6-luna", reasoning_effort="none"),
        record_root=tmp_path / "records",
        client=client_object,
    )
    live.call(
        ReceiverFeedback,
        system="Return the fixed response.",
        user_payload={"request": "fixed"},
        agent_name="gpt6_profile_test",
    )
    request = completions.calls[0]
    assert request["reasoning_effort"] == "none"
    assert request["model"] == "gpt-6-luna"
    assert request["temperature"] == 0
    assert request["top_p"] == 1
    assert request["max_completion_tokens"] == 9000
    assert configured.settings.max_total_tokens_per_task is None
    assert configured.settings.maximum_retries == 0
    assert "max_tokens" not in request


def test_repeated_live_calls_are_immutable_and_replay_in_order(
    tmp_path: Path,
) -> None:
    first = ReceiverFeedback(
        decision="accepted",
        text="I accept the complete validated request.",
        committed_power_kw=10.0,
        completion_percentage=100.0,
        completion_basis="All requested service is validated.",
    )
    client_object, completions = _fake_client(first.model_dump_json())
    live = StructuredLLMClient(
        mode="live",
        settings=LLMSettings(max_output_tokens=256),
        record_root=tmp_path,
        client=client_object,
    )
    for _ in range(2):
        observed = live.call(
            ReceiverFeedback,
            system="Return the fixed response.",
            user_payload={"request": "fixed"},
            agent_name="record_test",
        )
        assert observed == first
    assert len(completions.calls) == 2
    assert len(set(live.request_hashes)) == 1
    assert len(set(live.invocation_ids)) == 2
    request_root = tmp_path / live.request_hashes[0]
    invocation_dirs = sorted((request_root / "i").iterdir())
    assert len(invocation_dirs) == 2
    assert all((path / "manifest.json").is_file() for path in invocation_dirs)
    assert all(
        json.loads((path / "manifest.json").read_text(encoding="utf-8"))["status"]
        == "completed"
        for path in invocation_dirs
    )

    replay = StructuredLLMClient(
        mode="replay",
        settings=LLMSettings(max_output_tokens=256),
        record_root=tmp_path,
    )
    replayed = [
        replay.call(
            ReceiverFeedback,
            system="Return the fixed response.",
            user_payload={"request": "fixed"},
            agent_name="record_test",
        )
        for _ in range(2)
    ]
    assert replayed == [first, first]
    assert replay.invocation_ids == live.invocation_ids
