"""Score one saved PipelineResult using the paper-aligned F/A/V/G/Q protocol."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.artifact_paths import resolve_recorded_path
from b2g_agent.case_io import CaseBundle  # noqa: E402
from b2g_agent.contracts import EpisodeGroundTruth, PipelineResult  # noqa: E402
from b2g_agent.evaluation import InteractionScorer  # noqa: E402


CASE_DIR = REPO_ROOT / "cases" / "ieee33_10bem_15min"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_groundtruth(episode_id: str) -> EpisodeGroundTruth:
    path = CASE_DIR / "groundtruth" / "contract_oracles.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload["episode_id"] == episode_id:
            return EpisodeGroundTruth.model_validate(payload)
    raise KeyError(episode_id)


def main() -> None:
    args = parse_args()
    result_path = resolve_recorded_path(args.result).resolve()
    candidate = PipelineResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    groundtruth = load_groundtruth(candidate.task_specification.episode_id)
    case = CaseBundle.default(REPO_ROOT)
    episode = case.get_episode(groundtruth.episode_id)
    numeric_path = CASE_DIR / "groundtruth" / "numeric_oracles" / f"{episode.episode_id}.json"
    numeric = (
        json.loads(numeric_path.read_text(encoding="utf-8"))
        if numeric_path.is_file()
        else None
    )
    scorer = InteractionScorer(CASE_DIR / "evaluation" / "scoring.json")
    score = scorer.score(
        episode=episode,
        groundtruth=groundtruth,
        candidate=candidate,
        numeric_oracle=numeric,
    )
    output = score.model_dump_json(indent=2) + "\n"
    if args.output is not None:
        output_path = args.output.resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output, encoding="utf-8")
    print(output, end="")


if __name__ == "__main__":
    main()
