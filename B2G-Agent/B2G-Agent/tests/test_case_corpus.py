from __future__ import annotations

import json
import hashlib
from collections import Counter
from pathlib import Path

from b2g_agent.case_io import CaseBundle


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_formal_case_has_ten_bems_and_200_unique_episodes() -> None:
    case = CaseBundle.default(REPO_ROOT)
    case_definition = case.manifest
    episodes = list(case.iter_episodes())
    assert len(case.bem_mapping()) == 10
    assert case_definition["building_model_policy"] == {
        "representative_idfs_per_bem": 1,
        "power_portfolio_scaling_enabled": True,
        "temperature_scaling_enabled": False,
    }
    assert len(episodes) == 200
    assert len({episode.episode_id for episode in episodes}) == 200
    assert len({episode.group_id for episode in episodes}) == 200
    assert len({episode.sender_message for episode in episodes}) == 200
    assert sum(len(episode.task_families) == 1 for episode in episodes) == 150
    assert sum(len(episode.task_families) == 2 for episode in episodes) == 50
    assert all(episode.sender_message.strip() for episode in episodes)


def test_frozen_corpus_contains_only_the_initial_sender_message() -> None:
    corpus = REPO_ROOT / "cases" / "ieee33_10bem_15min" / "tasks" / "episodes.jsonl"
    records = [json.loads(line) for line in corpus.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 200
    assert all(record["schema_version"] == "3.0" for record in records)
    assert all(isinstance(record["sender_message"], str) for record in records)
    assert all("conversation" not in record for record in records)
    assert all("ground_truth" not in record for record in records)

    sender_corpus = corpus.with_name("sender_messages.jsonl")
    sender_records = [
        json.loads(line)
        for line in sender_corpus.read_text(encoding="utf-8").splitlines()
    ]
    assert len(sender_records) == 200
    assert all(
        set(record) == {"episode_id", "sender_role", "sender_message"}
        for record in sender_records
    )
    assert [record["sender_message"] for record in sender_records] == [
        record["sender_message"] for record in records
    ]


def test_sender_messages_use_varied_human_request_language() -> None:
    corpus = REPO_ROOT / "cases" / "ieee33_10bem_15min" / "tasks" / "episodes.jsonl"
    messages = [
        json.loads(line)["sender_message"]
        for line in corpus.read_text(encoding="utf-8").splitlines()
    ]
    human_markers = (
        "we need",
        "i need",
        "we'd like",
        "we're quoting",
        "could you",
        "our offer",
        "our portfolio",
    )
    assert all(any(marker in message.lower() for marker in human_markers) for message in messages)
    assert len({message.split(".", 1)[0] for message in messages}) >= 100
    assert not any(
        message.lower().startswith("on the ") and ", request " in message.lower()
        for message in messages
    )


def test_groundtruth_is_separate_complete_and_honestly_statused() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episodes = list(case.iter_episodes())
    truths = list(case.iter_groundtruth())
    assert len(truths) == 200
    assert case.validate_groundtruth_alignment() == ()
    assert {item.episode_id for item in truths} == {
        item.episode_id for item in episodes
    }
    assert all(
        item.contract_truth_status
        == "programmatically_generated_pending_expert_review"
        for item in truths
    )
    verification_path = (
        REPO_ROOT
        / "cases"
        / "ieee33_10bem_15min"
        / "groundtruth"
        / "contract_verification.jsonl"
    )
    verification = [
        json.loads(line)
        for line in verification_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(verification) == 200
    assert {record["episode_id"] for record in verification} == {
        item.episode_id for item in truths
    }
    assert all(
        record["status"] == "programmatically_verified"
        for record in verification
    )
    pending = {item.episode_id for item in truths if item.physics_truth_status == "pending"}
    verified = {
        item.episode_id for item in truths if item.physics_truth_status == "verified"
    }
    assert len(pending) + len(verified) == 100
    assert sum(
        item.physics_truth_status == "not_required_for_current_turn"
        for item in truths
    ) == 100
    assert sum(
        item.perfect_execution.route == "execute_physics_then_release"
        for item in truths
    ) == 100
    assert sum(
        item.perfect_execution.route == "clarify_sender" for item in truths
    ) == 100
    assert not any(
        "rebound_kw" in item.perfect_execution.required_numeric_outputs
        and any(
            window.start == "00:00" and window.end == "23:45"
            for window in item.task_specification.windows
        )
        for item in truths
    )
    assert not any(
        "renewable_absorption" in {
            family.value for family in item.task_specification.task_families
        }
        and len(item.task_specification.task_families) > 1
        and len(item.task_specification.windows) == 1
        for item in truths
    )
    numeric_ids = {
        path.stem
        for path in (
            REPO_ROOT
            / "cases"
            / "ieee33_10bem_15min"
            / "groundtruth"
            / "numeric_oracles"
        ).glob("ep-*.json")
    }
    assert numeric_ids == verified


def test_episode_allocations_and_splits_are_frozen() -> None:
    episodes = list(CaseBundle.default(REPO_ROOT).iter_episodes())
    family_counts = Counter(episode.primary_family.value for episode in episodes)
    split_counts = Counter(episode.split.value for episode in episodes)
    assert family_counts == {
        "peak_shaving": 24, "congestion_management": 24, "voltage_management": 22,
        "reserve_service": 20, "renewable_absorption": 20, "baseline_flexibility": 20,
        "market_trading": 20, "composite": 50,
    }
    assert split_counts == {"development": 30, "memory_seed": 20, "test": 150}
    groups_by_split: dict[str, set[str]] = {}
    for episode in episodes:
        groups_by_split.setdefault(episode.split.value, set()).add(episode.group_id)
    assert groups_by_split["development"].isdisjoint(groups_by_split["test"])
    assert groups_by_split["memory_seed"].isdisjoint(groups_by_split["test"])


def test_case_summary_is_contract_ready_without_claiming_physics_ready() -> None:
    summary = CaseBundle.default(REPO_ROOT).public_summary()
    assert summary["readiness"]["ready_for_contract_evaluation"] is True
    assert isinstance(summary["readiness"]["ready_for_physics"], bool)
    assert len(summary["episode_sha256"]) == 64
    assert summary["groundtruth_count"] == 200
    assert len(summary["groundtruth_sha256"]) == 64
    assert summary["groundtruth_alignment_errors"] == []
    task_summary = json.loads(
        (
            REPO_ROOT
            / "cases"
            / "ieee33_10bem_15min"
            / "tasks"
            / "summary.json"
        ).read_text(encoding="utf-8")
    )
    assert task_summary["frozen_sender_messages_per_episode"] == 1
    assert task_summary["runtime_communication_acts"] == 2
    assert task_summary["runtime_information_states"] == 4
    assert task_summary["unique_sender_messages"] == 200
    groundtruth_path = (
        REPO_ROOT
        / "cases"
        / "ieee33_10bem_15min"
        / "groundtruth"
        / "contract_oracles.jsonl"
    )
    assert task_summary["groundtruth_contract_sha256"] == hashlib.sha256(
        groundtruth_path.read_bytes()
    ).hexdigest()
    physics_counts = task_summary["physics_truth_status_counts"]
    assert physics_counts["not_required_for_current_turn"] == 100
    assert physics_counts["pending"] + physics_counts["verified"] == 100
