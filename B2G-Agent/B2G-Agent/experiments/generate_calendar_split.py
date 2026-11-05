"""Freeze leakage-safe calendar pools for full-year profile derivation."""

from __future__ import annotations

import json
import random
from calendar import monthrange
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = REPO_ROOT / "cases" / "ieee33_10bem_15min" / "days" / "calendar_split.json"
YEAR = 2025
SEED = 2026


def main() -> None:
    development: list[str] = []
    reference: list[str] = []
    test_candidate: list[str] = []
    for month in range(1, 13):
        days = [date(YEAR, month, day).isoformat() for day in range(1, monthrange(YEAR, month)[1] + 1)]
        random.Random(SEED + month).shuffle(days)
        development_count = round(len(days) * 0.70)
        reference_count = round(len(days) * 0.10)
        development.extend(days[:development_count])
        reference.extend(days[development_count : development_count + reference_count])
        test_candidate.extend(days[development_count + reference_count :])
    payload = {
        "schema_version": "1.0",
        "nominal_year": YEAR,
        "seed": SEED,
        "stratification": "independent shuffle within each month; 70% development, 10% reference, 20% test-candidate by nearest rounded counts",
        "development_dates": sorted(development),
        "reference_dates": sorted(reference),
        "test_candidate_dates": sorted(test_candidate),
    }
    if sum(len(payload[key]) for key in ("development_dates", "reference_dates", "test_candidate_dates")) != 365:
        raise RuntimeError("Calendar split must contain 365 dates")
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print({key: len(payload[key]) for key in ("development_dates", "reference_dates", "test_candidate_dates")})


if __name__ == "__main__":
    main()
