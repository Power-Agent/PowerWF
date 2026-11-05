"""Offline audit of consequences after a correctable first forward review error.

The protocol uses recorded drafts, independent structured references and the
unchanged task scores. It does not alter any acquisition or formal paper metric.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import socket

from reviewer_feedback_reference import (
    ROOT, SOURCES, CaseBundle, InteractionScorer, audit_opportunity, digest,
    extract_dispatches, first_reviews, forbid_network,
)
from b2g_agent.publication import recorded_hash_matches
from retained_paths import resolve_recorded_path
from b2g_agent.evaluation.reviewer_feedback_metrics import feedback_outcome


def summarize(rows: list[dict]) -> dict:
    denominator = len(rows)
    if not denominator:
        return {"denominator": 0}
    counts = {
        "appropriate_handling": sum(row["appropriate_handling"] for row in rows),
        "error_eliminated": sum(row["error_eliminated"] for row in rows),
        "reached_reverse_review": sum(row["reached_reverse_review"] for row in rows),
        "task_success": sum(row["task_success"] for row in rows),
        "feedback_delivered_diagnostic": sum(row["correct_feedback_delivered"] for row in rows),
        "feedback_based_closure_diagnostic": sum(row["repaired"] for row in rows),
    }
    return {
        "denominator": denominator,
        "counts": counts,
        "rates": {key: value / denominator for key, value in counts.items()},
        "mean_final_quality": sum(row["final_quality"] for row in rows) / denominator,
        "workflow_status": dict(Counter(row["workflow_status"] for row in rows)),
    }


def main() -> None:
    socket.socket.connect = forbid_network
    socket.socket.connect_ex = forbid_network
    socket.create_connection = forbid_network
    output = ROOT / "outputs/analysis/reviewer-feedback"
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "status": "exploratory_post_acquisition",
        "source": "existing 200 tasks x 3 repeats for Proposed and Reviewer v4",
        "eligibility": "first forward Reviewer input independently has a correctable internal semantic or evidence error; source-required clarification and ordinary pending physics excluded",
        "comparison_cohort": "same 200 tasks x 3 repeats; first indicator includes all 600, second and third include the same 300 source-executable runs in both methods; paired-error intersection is diagnostic only",
        "indicators": {
            "appropriate_handling": "all 600 runs: source clarification requires first Reviewer clarify_sender route and original task_success; clean first input requires actor route and subsequent reverse review; internal error requires final later forward draft independently clean; missing review gets no credit",
            "executable_reverse_review_reach": "among 300 source-executable runs, a reverse Reviewer stage appears; no corrective-return event is required by the formula",
            "executable_task_success": "among the same 300 source-executable runs, unchanged original task_success is true",
        },
        "diagnostics_only": ["correct_feedback_delivered", "feedback_based_closure"],
        "reference_policy": "reviewer-process-offline-v1.1 required-supported",
        "score_policy": "original interaction-quality-f085-v2; no rescoring",
        "analysis_script_sha256": digest(Path(__file__)),
        "api_calls": 0,
        "physics_runs": 0,
    }
    (output / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    case = CaseBundle.default(ROOT)
    truths = {truth.episode_id: truth for truth in case.iter_groundtruth()}
    numeric = {}
    for task in range(1, 201):
        episode = f"ep-{task:03d}"
        path = case.case_dir / "groundtruth/numeric_oracles" / f"{episode}.json"
        numeric[episode] = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    scorer = InteractionScorer(case.case_dir / "evaluation/scoring.json")
    all_rows, eligible = {}, {}
    for method, suite in SOURCES.items():
        method_rows = []
        for task in range(1, 201):
            episode = f"ep-{task:03d}"
            truth = truths[episode]
            for repeat in range(1, 4):
                run_id = f"{episode}-r{repeat:02d}"
                record = json.loads((suite / "interaction_records" / run_id / "record.json").read_text(encoding="utf-8"))
                score = record.get("score") or {}
                journal = resolve_recorded_path(record["source_journal"])
                if not recorded_hash_matches(journal, record["source_journal_sha256"]):
                    raise AssertionError(f"Source journal hash mismatch: {method} {run_id}")
                events = [json.loads(line) for line in journal.read_text(encoding="utf-8").splitlines() if line.strip()]
                method_rows.append(feedback_outcome(
                    events, truth, numeric[episode], scorer, score, run_id=run_id,
                    workflow_status=record["index_row"].get("workflow_status")))

        assert len(method_rows) == 600
        all_rows[method] = method_rows
        eligible[method] = {row["run_id"] for row in method_rows if row["affected"]}
        print(json.dumps({"method": method, "eligible": len(eligible[method]),
                          "all_successes": sum(row["task_success"] for row in method_rows),
                          "all_quality": sum(row["final_quality"] for row in method_rows) / 600},
                         ensure_ascii=False), flush=True)
    common = eligible["proposed"] & eligible["reviewer_v4"]
    result = {"protocol": protocol, "generated_at": datetime.now(timezone.utc).isoformat(),
              "common_affected_run_count": len(common),
              "eligibility_overlap": {
                  "proposed_only": len(eligible["proposed"] - common),
                  "reviewer_v4_only": len(eligible["reviewer_v4"] - common),
              },
              "methods": {}}
    for method, rows in all_rows.items():
        affected = [row for row in rows if row["affected"]]
        matched = [row for row in rows if row["run_id"] in common]
        result["methods"][method] = {
            "all_runs": summarize(rows),
            "executable_runs": summarize([row for row in rows if row["source_executable"]]),
            "natural_affected_runs": summarize(affected),
            "matched_affected_runs": summarize(matched),
        }
        print(json.dumps({"method": method,
                          "natural_affected": result["methods"][method]["natural_affected_runs"],
                          "matched_affected": result["methods"][method]["matched_affected_runs"]},
                         ensure_ascii=False), flush=True)
    (output / "comparison.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "run_outcomes.jsonl").open("w", encoding="utf-8") as stream:
        for method, rows in all_rows.items():
            for row in rows:
                stream.write(json.dumps({"method": method, "matched_affected": row["run_id"] in common, **row}, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output / "comparison.json"), "matched_affected": len(common)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
