"""Refresh sender wording while preserving task semantics and frozen physics."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.generate_episodes import GENERATOR_VERSION, build  # noqa: E402


CASE_DIR = REPO_ROOT / "cases" / "ieee33_10bem_15min"
TASKS_DIR = CASE_DIR / "tasks"
GROUNDTRUTH_DIR = CASE_DIR / "groundtruth"
EPISODES_PATH = TASKS_DIR / "episodes.jsonl"
SENDER_MESSAGES_PATH = TASKS_DIR / "sender_messages.jsonl"
CONTRACTS_PATH = GROUNDTRUTH_DIR / "contract_oracles.jsonl"
TASK_SUMMARY_PATH = TASKS_DIR / "summary.json"
MANIFEST_PATH = GROUNDTRUTH_DIR / "manifest.json"
NUMERIC_ORACLE_DIR = GROUNDTRUTH_DIR / "numeric_oracles"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]], *, sort_keys: bool = False) -> None:
    path.write_text(
        "".join(
            json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=sort_keys,
            )
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def numeric_hashes() -> dict[str, str]:
    return {
        path.name: sha256_file(path)
        for path in sorted(NUMERIC_ORACLE_DIR.glob("ep-*.json"))
    }


def without_wording_episode(row: dict[str, Any]) -> dict[str, Any]:
    comparable = json.loads(json.dumps(row))
    comparable.pop("sender_message")
    comparable.pop("generator_version")
    return comparable


def without_wording_contract(row: dict[str, Any]) -> dict[str, Any]:
    comparable = json.loads(json.dumps(row))
    comparable.pop("source_message_sha256")
    comparable["task_specification"].pop("source_text")
    comparable.pop("physics_truth_status")
    return comparable


def main() -> None:
    generated_episodes, generated_truths = build()
    generated_episode_rows = {
        row["episode_id"]: row
        for row in (episode.model_dump(mode="json") for episode in generated_episodes)
    }
    generated_truth_rows = {
        row["episode_id"]: row
        for row in (truth.model_dump(mode="json") for truth in generated_truths)
    }
    episode_rows = read_jsonl(EPISODES_PATH)
    contract_rows = read_jsonl(CONTRACTS_PATH)
    episode_by_id = {row["episode_id"]: row for row in episode_rows}
    contract_by_id = {row["episode_id"]: row for row in contract_rows}
    expected_ids = set(generated_episode_rows)
    if set(episode_by_id) != expected_ids or set(contract_by_id) != expected_ids:
        raise RuntimeError("Existing episodes and contracts do not match the deterministic 200-task build")

    for episode_id in sorted(expected_ids):
        existing_episode = episode_by_id[episode_id]
        generated_episode = generated_episode_rows[episode_id]
        if without_wording_episode(existing_episode) != without_wording_episode(generated_episode):
            raise RuntimeError(f"Non-wording episode drift detected for {episode_id}")
        existing_contract = contract_by_id[episode_id]
        generated_contract = generated_truth_rows[episode_id]
        if without_wording_contract(existing_contract) != without_wording_contract(generated_contract):
            raise RuntimeError(f"Non-wording contract drift detected for {episode_id}")

    old_numeric_hashes = numeric_hashes()
    old_episode_hash = sha256_file(EPISODES_PATH)
    old_contract_hash = sha256_file(CONTRACTS_PATH)
    old_messages = {
        row["episode_id"]: row["sender_message"] for row in episode_rows
    }

    for row in episode_rows:
        generated = generated_episode_rows[row["episode_id"]]
        row["sender_message"] = generated["sender_message"]
        row["generator_version"] = GENERATOR_VERSION
    write_jsonl(EPISODES_PATH, episode_rows)

    sender_rows = [
        {
            "episode_id": row["episode_id"],
            "sender_role": row["start_role"],
            "sender_message": row["sender_message"],
        }
        for row in episode_rows
    ]
    write_jsonl(SENDER_MESSAGES_PATH, sender_rows)

    message_by_id = {row["episode_id"]: row["sender_message"] for row in episode_rows}
    for row in contract_rows:
        message = message_by_id[row["episode_id"]]
        row["source_message_sha256"] = sha256_bytes(message.encode("utf-8"))
        row["task_specification"]["source_text"] = message
    write_jsonl(CONTRACTS_PATH, contract_rows)

    summary = json.loads(TASK_SUMMARY_PATH.read_text(encoding="utf-8"))
    summary["generator_version"] = GENERATOR_VERSION
    summary["unique_sender_messages"] = len(set(message_by_id.values()))
    summary["sender_messages_sha256"] = sha256_file(SENDER_MESSAGES_PATH)
    summary["groundtruth_contract_sha256"] = sha256_file(CONTRACTS_PATH)
    TASK_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["generator_version"] = GENERATOR_VERSION
    manifest["episodes_sha256"] = sha256_file(EPISODES_PATH)
    manifest["contract_oracles_sha256"] = sha256_file(CONTRACTS_PATH)
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    new_numeric_hashes = numeric_hashes()
    if old_numeric_hashes != new_numeric_hashes:
        raise RuntimeError("A frozen numeric oracle changed during the wording-only migration")

    print(
        json.dumps(
            {
                "generator_version": GENERATOR_VERSION,
                "episode_count": len(episode_rows),
                "changed_message_count": sum(
                    old_messages[row["episode_id"]] != row["sender_message"]
                    for row in episode_rows
                ),
                "old_episodes_sha256": old_episode_hash,
                "new_episodes_sha256": sha256_file(EPISODES_PATH),
                "old_contract_oracles_sha256": old_contract_hash,
                "new_contract_oracles_sha256": sha256_file(CONTRACTS_PATH),
                "numeric_oracle_count": len(new_numeric_hashes),
                "numeric_oracles_unchanged": True,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
