"""Aggregate one or more completed evaluation directories by method."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.evaluation import (  # noqa: E402
    summarize_evaluation_runs,
    write_evaluation_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_roots", type=Path, nargs="+")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / ".local-runs" / "evaluation_summaries",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = summarize_evaluation_runs(args.run_roots)
    json_path, csv_path = write_evaluation_summary(summary, args.output_dir)
    print(
        f"methods={summary['method_count']} json={json_path} csv={csv_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
