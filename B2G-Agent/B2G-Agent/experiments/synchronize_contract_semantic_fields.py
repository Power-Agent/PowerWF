"""Synchronize paper-defined semantic evaluation fields without touching physics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "cases" / "ieee33_10bem_15min"
CONTRACT_PATH = CASE_DIR / "groundtruth" / "contract_oracles.jsonl"
MANIFEST_PATH = CASE_DIR / "groundtruth" / "manifest.json"
TASK_SUMMARY_PATH = CASE_DIR / "tasks" / "summary.json"


BASE_FIELDS = [
    "sender_role",
    "receiver_role",
    "task_families",
    "operating_day",
    "windows",
    "target_bems",
    "target_buses",
    "target_power",
    "objectives",
    "constraints",
    "requested_outputs",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def required_fields(families: list[str]) -> list[str]:
    return [
        *BASE_FIELDS,
        *(["offer_price"] if "market_trading" in families else []),
    ]


def main() -> None:
    rows = [
        json.loads(line)
        for line in CONTRACT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    changed = 0
    for row in rows:
        expected = required_fields(row["task_specification"]["task_families"])
        oracle = row["perfect_execution"]
        if oracle["required_semantic_fields"] != expected:
            oracle["required_semantic_fields"] = expected
            changed += 1
    CONTRACT_PATH.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    contract_hash = sha256(CONTRACT_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["contract_oracles_sha256"] = contract_hash
    MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = json.loads(TASK_SUMMARY_PATH.read_text(encoding="utf-8"))
    summary["groundtruth_contract_sha256"] = contract_hash
    TASK_SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "episode_count": len(rows),
                "changed_count": changed,
                "contract_oracles_sha256": contract_hash,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
