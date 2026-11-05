"""Budget fault injection uses local doubles only; no API or physics calls."""
import json
from types import SimpleNamespace

import pytest

from b2g_agent.agents.llm import StructuredLLMClient, ResponseValidationError
from b2g_agent.agents.token_budget import (
    TaskTokenBudgetExceeded, TokenUsageUnavailable, conservative_prompt_tokens,
)
from b2g_agent.contracts import ReceiverFeedback
from b2g_agent.observability import LLMSettings
from b2g_agent.observability.journal import EventJournal


class CompletionsDouble:
    def __init__(self):
        self.calls = []
        self.usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        self.content = "I accept."
        self.error = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        data = {"choices": [{"message": {"content": self.content}}], "usage": self.usage}
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))],
            model_dump=lambda **_: data,
        )


def make_client(tmp_path, **settings):
    # Cumulative budgeting is opt-in; these tests explicitly enable it.
    settings.setdefault("max_total_tokens_per_task", 9000)
    double = CompletionsDouble()
    client = StructuredLLMClient(
        mode="live", settings=LLMSettings(**settings), record_root=tmp_path,
        client=SimpleNamespace(chat=SimpleNamespace(completions=double)),
    )
    return client, double


def text_call(client, agent="translator"):
    return client.call_text(system="Translate.", user_payload={}, agent_name=agent)


def test_total_budget_is_shared_between_agents_and_includes_input(tmp_path):
    client, double = make_client(tmp_path)
    text_call(client, "translator")
    text_call(client, "actor")
    text_call(client, "reviewer")
    limits = [r["max_completion_tokens"] for r in double.calls]
    assert limits[0] - limits[1] == 15  # input 10 + output 5, not just output
    assert limits[1] - limits[2] == 15
    assert client._task_budget().snapshot()["observed_total_tokens"] == 45


def test_prompt_and_schema_are_budgeted_and_oversize_never_sent(tmp_path):
    client, double = make_client(tmp_path, max_total_tokens_per_task=1200)
    text_call(client)
    with pytest.raises(TaskTokenBudgetExceeded):
        client.call(ReceiverFeedback, system="Translate.", user_payload={}, agent_name="reviewer")
    assert len(double.calls) == 1
    last = tmp_path / client.request_hashes[-1] / "i" / client.invocation_ids[-1]
    manifest = json.loads((last / "manifest.json").read_text())
    assert manifest["error_type"] == "TaskTokenBudgetExceeded"
    assert manifest["api_request_attempted"] is False
    assert not (last / "response.json").exists()


def test_exhaustion_and_new_repetition_gets_fresh_budget(tmp_path):
    messages = [{"role": "system", "content": "Translate."}, {"role": "user", "content": "{}"}]
    estimate = conservative_prompt_tokens(messages, None)
    client, double = make_client(tmp_path, max_total_tokens_per_task=estimate + 21)
    text_call(client)
    text_call(client)
    with pytest.raises(TaskTokenBudgetExceeded):
        text_call(client)
    assert [r["max_completion_tokens"] for r in double.calls] == [21, 6]
    client.begin_task()
    text_call(client)
    assert double.calls[-1]["max_completion_tokens"] == 21


def test_journal_keeps_feedback_budget_and_isolates_new_turns(tmp_path):
    client, double = make_client(tmp_path / "records")
    first = EventJournal(tmp_path / "turn1.jsonl", "turn1")
    second = EventJournal(tmp_path / "turn2.jsonl", "turn2")
    with first.activate():
        text_call(client, "translator")
    with second.activate():
        text_call(client, "translator")
    with first.activate():
        text_call(client, "translator_feedback")
    assert double.calls[0]["max_completion_tokens"] == double.calls[1]["max_completion_tokens"]
    assert double.calls[2]["max_completion_tokens"] == double.calls[0]["max_completion_tokens"] - 15
    ledger = json.loads((tmp_path / "turn1.tokens.json").read_text())
    assert ledger["charged_tokens"] == 30
    restored, restored_double = make_client(tmp_path / "records")
    with first.activate():
        text_call(restored, "reviewer")
    assert restored_double.calls[0]["max_completion_tokens"] == double.calls[0]["max_completion_tokens"] - 30


def test_invalid_structured_reply_still_consumes_tokens(tmp_path):
    client, double = make_client(tmp_path)
    double.content = '{"decision":"invalid"}'
    with pytest.raises(ResponseValidationError):
        client.call(ReceiverFeedback, system="Translate.", user_payload={}, agent_name="reviewer")
    assert client._task_budget().snapshot()["observed_total_tokens"] == 15
    text_call(client)
    assert client._task_budget().snapshot()["observed_total_tokens"] == 30


@pytest.mark.parametrize("failure", ["transport", "missing_usage", "inconsistent_usage"])
def test_unknown_usage_blocks_followup_and_retains_response(tmp_path, failure):
    client, double = make_client(tmp_path, max_output_tokens=256)
    if failure == "transport":
        double.error = TimeoutError("Offline injected timeout")
    elif failure == "missing_usage":
        double.usage = None
    else:
        double.usage["total_tokens"] = 999
    with pytest.raises((TimeoutError, TokenUsageUnavailable)):
        text_call(client)
    with pytest.raises(TaskTokenBudgetExceeded):
        text_call(client)
    assert len(double.calls) == 1
    assert client._task_budget().snapshot()["reserved_tokens"] > 0
    if failure != "transport":
        assert list(tmp_path.glob("*/i/*/response.json"))


def test_provider_overrun_is_recorded_and_stops_task(tmp_path):
    client, double = make_client(tmp_path)
    double.usage = {"prompt_tokens": 9000, "completion_tokens": 1, "total_tokens": 9001}
    with pytest.raises(TaskTokenBudgetExceeded):
        text_call(client)
    assert client._task_budget().snapshot()["observed_total_tokens"] == 9001
    with pytest.raises(TaskTokenBudgetExceeded):
        text_call(client)
    assert len(double.calls) == 1


def test_hidden_sdk_retries_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="maximum_retries=0"):
        make_client(tmp_path, maximum_retries=2)


def test_usage_report_does_not_count_a_local_block_as_an_api_call(tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "experiments"))
    from run_contract_evaluation import llm_usage
    client, double = make_client(tmp_path, max_total_tokens_per_task=1200)
    text_call(client)
    with pytest.raises(TaskTokenBudgetExceeded):
        client.call(ReceiverFeedback, system="Translate.", user_payload={}, agent_name="reviewer")
    usage = llm_usage(tmp_path, client.request_hashes, client.invocation_ids)
    assert usage["llm_call_count"] == len(double.calls) == 1
    assert usage["local_request_blocks"] == 1
    assert usage["total_tokens"] == 15


def test_per_call_protocol_preserves_9000_after_large_inputs_and_prior_calls(tmp_path):
    client, double = make_client(tmp_path / "records", max_total_tokens_per_task=None)
    double.usage = {"prompt_tokens": 12000, "completion_tokens": 8000, "total_tokens": 20000}
    with EventJournal(tmp_path / "turn.jsonl", "turn").activate():
        for agent in ("translator", "actor", "reviewer", "translator_feedback"):
            client.call_text(system="Translate.", user_payload={"context": "x" * 12000}, agent_name=agent)
    assert len(double.calls) == 4
    assert all(request["max_completion_tokens"] == 9000 for request in double.calls)
    assert client._task_budget() is None
    assert not (tmp_path / "turn.tokens.json").exists()
    manifests = [json.loads(p.read_text()) for p in (tmp_path / "records").glob("*/i/*/manifest.json")]
    assert len(manifests) == 4
    assert sum(m["usage"]["total_tokens"] for m in manifests) == 80000


def test_empty_task_budget_environment_disables_cumulative_limit(tmp_path, monkeypatch):
    monkeypatch.setenv("B2G_LLM_MODE", "offline")
    monkeypatch.setenv("B2G_MAX_TOTAL_TOKENS_PER_TASK", "")
    monkeypatch.setenv("B2G_MAX_OUTPUT_TOKENS", "9000")
    client = StructuredLLMClient.from_env(repo_root=tmp_path)
    assert client.settings.max_total_tokens_per_task is None
    assert client.settings.max_output_tokens == 9000
