"""Finalize numeric-oracle review metadata after deterministic validation."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.observability.reproducibility import (  # noqa: E402
    physics_implementation_bundle_sha256,
)


CASE_DIR = REPO_ROOT / "cases" / "ieee33_10bem_15min"
GROUNDTRUTH_DIR = CASE_DIR / "groundtruth"
CONTRACT_PATH = GROUNDTRUTH_DIR / "contract_oracles.jsonl"
MANIFEST_PATH = GROUNDTRUTH_DIR / "manifest.json"
REVIEW_CHECKS = [
    "physics_implementation_bundle",
    "stable_input_hashes",
    "required_output_definedness",
    "natural_language_completion",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    contracts = [
        json.loads(line)
        for line in CONTRACT_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    current_bundle = physics_implementation_bundle_sha256(REPO_ROOT)
    verified_count = 0
    for contract in contracts:
        if contract["physics_truth_status"] == "not_required_for_current_turn":
            continue
        episode_id = contract["episode_id"]
        if contract["physics_truth_status"] != "verified":
            raise ValueError(f"Numeric oracle remains pending: {episode_id}")
        path = GROUNDTRUTH_DIR / "numeric_oracles" / f"{episode_id}.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        oracle = json.loads(path.read_text(encoding="utf-8"))
        hash_groups = (
            list(oracle.get("input_hashes_by_direction", {}).values())
            if "input_hashes_by_direction" in oracle
            else [oracle.get("input_hashes", {})]
        )
        if not hash_groups or any(
            group.get("physics_implementation_bundle") != current_bundle
            for group in hash_groups
        ):
            raise ValueError(f"Stale physics implementation: {episode_id}")
        values = oracle.get("business_metrics", {}).get(
            "requested_output_values", {}
        )
        for field in contract["perfect_execution"]["required_numeric_outputs"]:
            if field not in values or values[field] is None:
                raise ValueError(f"Undefined required output: {episode_id}:{field}")
        final = oracle.get("final_decision", {})
        if not final.get("expected_receiver_feedback"):
            raise ValueError(f"Missing receiver feedback: {episode_id}")
        if not final.get("expected_sender_final"):
            raise ValueError(f"Missing sender final: {episode_id}")
        oracle["review_status"] = "programmatically_verified"
        oracle["review_checks"] = REVIEW_CHECKS
        path.write_text(json.dumps(oracle, indent=2) + "\n", encoding="utf-8")
        verified_count += 1

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["numeric_oracles"] = [
        {
            "episode_id": path.stem,
            "path": f"numeric_oracles/{path.name}",
            "sha256": sha256(path),
        }
        for path in sorted((GROUNDTRUTH_DIR / "numeric_oracles").glob("ep-*.json"))
    ]
    manifest["programmatically_verified_numeric_count"] = verified_count
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "programmatically_verified_numeric_count": verified_count,
                "physics_implementation_bundle": current_bundle,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
