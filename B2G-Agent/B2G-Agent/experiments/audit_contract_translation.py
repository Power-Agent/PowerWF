"""Audit deterministic m1-to-TaskSpecification translation without hidden episode metadata."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.agents import TranslatorAgent  # noqa: E402
from b2g_agent.case_io import CaseBundle  # noqa: E402
from b2g_agent.contracts import EpisodeGroundTruth  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit deterministic sender-message translation for all episodes."
    )
    parser.add_argument(
        "--mismatch-limit",
        type=int,
        default=20,
        help="Maximum number of mismatch records to print (counts always cover all episodes).",
    )
    parser.add_argument(
        "--only-failed-check",
        choices=(
            "task_families",
            "operating_day",
            "windows",
            "target_bems",
            "target_buses",
            "target_power_value",
            "target_power_unit",
            "target_power_status",
            "objectives",
            "constraints",
            "clarification_required",
            "requested_outputs",
        ),
        help="Restrict displayed mismatch examples to records failing this check.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case = CaseBundle.default(REPO_ROOT)
    translator = TranslatorAgent(REPO_ROOT / "knowledge")
    truth_path = case.case_dir / "groundtruth" / "contract_oracles.jsonl"
    truths = {
        row.episode_id: row
        for row in (
            EpisodeGroundTruth.model_validate(json.loads(line))
            for line in truth_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }
    field_counts: Counter[str] = Counter()
    mismatches: list[dict[str, object]] = []
    for episode in case.iter_episodes():
        expected = truths[episode.episode_id].task_specification
        observed = translator.translate(
            episode.sender_message,
            episode=episode,
            sender=episode.start_role,
        )
        checks = {
            "task_families": observed.task_families == expected.task_families,
            "operating_day": observed.operating_day == expected.operating_day,
            "windows": observed.windows == expected.windows,
            "target_bems": observed.target_bems == expected.target_bems,
            "target_buses": observed.target_buses == expected.target_buses,
            "target_power_value": (
                observed.fields["target_power"].value
                == expected.fields["target_power"].value
            ),
            "target_power_unit": (
                observed.fields["target_power"].unit
                == expected.fields["target_power"].unit
            ),
            "target_power_status": (
                observed.fields["target_power"].status
                == expected.fields["target_power"].status
            ),
            "objectives": observed.objectives == expected.objectives,
            "constraints": (
                sorted(
                    (item.metric, item.operator, item.value, item.unit, item.hard)
                    for item in observed.constraints
                )
                == sorted(
                    (item.metric, item.operator, item.value, item.unit, item.hard)
                    for item in expected.constraints
                )
            ),
            "clarification_required": (
                observed.clarification_required == expected.clarification_required
            ),
            "requested_outputs": (
                observed.requested_outputs == expected.requested_outputs
            ),
        }
        for name, passed in checks.items():
            field_counts[f"{name}_pass" if passed else f"{name}_fail"] += 1
        if not all(checks.values()):
            mismatches.append(
                {
                    "episode_id": episode.episode_id,
                    "communication": episode.difficulty.communication,
                    "failed_checks": [name for name, passed in checks.items() if not passed],
                    "sender_message": episode.sender_message,
                    "expected": {
                        "task_families": [item.value for item in expected.task_families],
                        "target_power": expected.fields["target_power"].model_dump(mode="json"),
                        "requested_outputs": expected.requested_outputs,
                    },
                    "observed": {
                        "task_families": [item.value for item in observed.task_families],
                        "target_power": observed.fields["target_power"].model_dump(mode="json"),
                        "requested_outputs": observed.requested_outputs,
                    },
                }
            )
    display_mismatches = mismatches
    if args.only_failed_check:
        display_mismatches = [
            item
            for item in mismatches
            if args.only_failed_check in item["failed_checks"]
        ]
    payload = {
        "schema_version": "1.0",
        "episode_count": len(truths),
        "fully_matching_count": len(truths) - len(mismatches),
        "mismatch_count": len(mismatches),
        "field_counts": dict(sorted(field_counts.items())),
        "display_filter": args.only_failed_check,
        "display_match_count": len(display_mismatches),
        "mismatches": display_mismatches[: max(args.mismatch_limit, 0)],
        "mismatches_truncated": len(display_mismatches) > max(args.mismatch_limit, 0),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
