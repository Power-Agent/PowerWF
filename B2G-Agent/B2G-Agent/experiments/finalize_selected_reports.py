"""Publish the selected five-method, 18-indicator view without any API calls.

The acquired run logs and frozen source snapshots are immutable. This script
uses the already checked final comparison and Reviewer impact audit to replace
obsolete derived indicator tables in the visible method directories.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

from retained_paths import ABLATION_SUITES, BENCHMARK, ROOT
from b2g_agent.evaluation.reviewer_feedback_metrics import REVIEWER_METRICS, reviewer_feedback_metrics

COMPARISON = ROOT / "outputs/comparison/final"
IMPACTS = ROOT / "outputs/analysis/reviewer-feedback/comparison.json"
SOURCES = {
    "proposed": BENCHMARK / "proposed",
    "baseline": BENCHMARK / "baseline",
    "translator_v2": ABLATION_SUITES["plain_translator"],
    "actor_v2": ABLATION_SUITES["unstructured_actor"],
    "reviewer_v4": ABLATION_SUITES["reviewer_no_feedback"],
}
REVIEWER = {
    "F_rew_1": ("first_review_appropriate_handling", "all scheduled attempts", 600),
    "F_rew_2": ("executable_reverse_review_reach", "source-executable attempts", 300),
    "F_rew_3": ("executable_task_success", "source-executable attempts", 300),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    with (COMPARISON / "metrics.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({row["metric"] for row in rows}) == 18
    outcomes = [json.loads(line) for line in (IMPACTS.parent / "run_outcomes.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    for method in ("proposed", "reviewer_v4"):
        cohort = [row for row in outcomes if row["method"] == method]
        assert len(cohort) == 600 and sum(row["source_executable"] for row in cohort) == 300
        computed = reviewer_feedback_metrics(cohort)
        for index, name in enumerate(REVIEWER_METRICS, 1):
            row = next(row for row in rows if row["metric"] == f"F_rew_{index}")
            assert abs(float(row[method]) - computed[name]["value"]) < 1e-12, (method, name)
    report_paths = {}
    for method, suite in SOURCES.items():
        selected = []
        for row in rows:
            if not row[method]:
                continue
            name = row["metric"]
            datum = {"metric": name, "value": float(row[method]), "unit": "points / 100" if name == "Q_qua" else
                     "tokens" if name == "total_token_consumption" else "seconds / attempt" if name in {"llm_inference_latency", "retrieval_time"} else "ratio"}
            if name in REVIEWER:
                definition, cohort, denominator = REVIEWER[name]
                datum.update(definition=definition, cohort=cohort, denominator=denominator)
            else:
                datum["definition"] = "See cases/ieee33_10bem_15min/evaluation/scoring.json and final comparison."
            selected.append(datum)
        assert len(selected) == (9 if method == "baseline" else 18 if method == "proposed" else 12)
        payload = {"method": method, "tasks": 200, "repeats": 3, "attempts": 600,
                   "metric_protocol": "selected-18-feedback-impact-20261103", "metrics": selected,
                   "comparison": str(COMPARISON / "metrics.csv"), "api_calls": 0, "physics_runs": 0}
        out = suite / "FINAL_METRICS.json"
        out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        with (suite / "FINAL_METRICS.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("metric", "value", "unit", "definition", "cohort", "denominator"))
            writer.writeheader()
            writer.writerows(selected)
        report_paths[str(out.relative_to(ROOT))] = sha256(out)
    provenance = json.loads((COMPARISON / "provenance.json").read_text(encoding="utf-8"))
    provenance["source_sha256"] = report_paths | {str(IMPACTS.relative_to(ROOT)): sha256(IMPACTS)}
    provenance["relocated_method_suites"] = {name: str(path.relative_to(ROOT)) for name, path in SOURCES.items()}
    provenance["original_acquisition_paths_recorded_in_immutable_manifests"] = True
    (COMPARISON / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"methods": list(SOURCES), "indicator_rows": len(rows), "per_method_reports": report_paths,
                      "api_calls": 0, "physics_runs": 0}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
