"""Read-only Reviewer reference extraction used by the final feedback audit."""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from b2g_agent.case_io import CaseBundle
from b2g_agent.evaluation import InteractionScorer
from b2g_agent.evaluation.reviewer_process_metrics import (
    audit_opportunity, extract_dispatches, extract_reviews, independent_reference,
)
from retained_paths import ABLATION_SUITES

SOURCES = {
    "proposed": ROOT / "outputs/benchmark/proposed",
    "reviewer_v4": ABLATION_SUITES["reviewer_no_feedback"],
}


def forbid_network(*_args, **_kwargs):
    raise RuntimeError("Network/API access is forbidden during this offline audit")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


from b2g_agent.evaluation.reviewer_feedback_metrics import first_reviews
