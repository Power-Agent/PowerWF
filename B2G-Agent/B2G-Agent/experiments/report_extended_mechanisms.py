"""Audit disjoint acquisition segments and aggregate the original 200 x 3 metrics.

This creates derived reports only. Original journals, manifests and scores are
read-only, and a failed method attempt remains in the denominator.
"""
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from b2g_agent.case_io import CaseBundle
from b2g_agent.contracts import PipelineResult
from b2g_agent.evaluation import InteractionScorer
from b2g_agent.evaluation.agent_metrics import evaluate_agents
from b2g_agent.evaluation.paper_metrics import AGENTS, agent_metric_keys, write_paper_metrics
from report_proposed_suite import (
    aggregate, artifact_references, digest, inspect_journal, load,
    relative, verify_invocations, write_json, write_report,
)
from retained_paths import resolve_recorded_path


def check_coverage(rows, episode_ids):
    counts = Counter((row["episode_id"], row["repeat"]) for row in rows)
    expected = {(episode, repeat) for episode in episode_ids for repeat in (1, 2, 3)}
    if set(counts) != expected or any(n != 1 for n in counts.values()):
        raise ValueError("Missing, duplicate or unexpected task/repeat in merged cohort")
    return len(expected)


def audit_segments(method, segments, output, protocol):
    output = resolve_recorded_path(output)
    if (output / "FINAL_METRICS.json").is_file():
        from finalize_selected_reports import main as finalize
        finalize()
        return load(output / "FINAL_METRICS.json")
    if output.exists():
        cached = output / "paper_metrics.json"
        if cached.is_file() and load(cached).get("status") == "completed":
            raise ValueError("A completed derived report must never be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    case = CaseBundle.default(ROOT)
    scorer = InteractionScorer(case.case_dir / "evaluation/scoring.json")
    records, errors, warnings, provenance = [], [], [], []
    physical, physical_files = {}, {}
    calls_verified = journals_verified = four_messages = 0
    source_checks=[]
    started, ended = [], []
    for source in map(resolve_recorded_path, segments):
        suite = load(source / "suite_manifest.json")
        if suite["method"] != method or not suite.get("coverage_verified"):
            raise ValueError("Source segment is incomplete or has a different method")
        frozen=load(source/"frozen_file_hashes.json")
        if hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()!=suite["source_and_case_bundle_sha256"]:
            raise ValueError("Source bundle manifest hash mismatch")
        with zipfile.ZipFile(source/"frozen_source_and_case.zip") as archive:
            if any(hashlib.sha256(archive.read(name)).hexdigest()!=sha for name,sha in frozen.items()):
                raise ValueError("Archived source file hash mismatch")
        # The frozen archive is the source of truth after post-acquisition cleanup.
        changed=[]
        source_checks.append({"suite":str(source),"archived_source_verified":True,
            "current_runtime_compared":bool(suite.get("continuation_of")),"changed_files":changed})
        started.append(suite["started_at"])
        ended.append(suite["completed_at"])
        provenance.append({"suite": str(source), "manifest_sha256": digest(source / "suite_manifest.json"),
            "episodes": suite["episode_ids"], "protocol": suite["protocol"]})
        workers = {str(item["worker"]): resolve_recorded_path(item["llm_record_root"]) for item in suite["workers_config"]}
        for index in sorted(source.glob("w*/*/index.jsonl")):
            run_root = index.parent
            config = load(run_root / "run_config.json")
            if config["method_id"] != method or config["llm_mode"] != "live":
                raise ValueError("Source worker method/mode mismatch")
            if config.get("protocol_id") != suite["protocol"]["protocol_id"]:
                raise ValueError("Source worker protocol identity mismatch")
            for field, expected in (("model", protocol["model"]),
                ("max_output_tokens", protocol["max_output_tokens_per_call"]),
                ("temperature", protocol["temperature"]), ("top_p", protocol["top_p"]),
                ("seed", protocol["seed"]), ("reasoning_effort", protocol["reasoning_effort"]),
                ("maximum_schema_repairs", protocol["maximum_schema_repairs"]),
                ("max_total_tokens_per_task", None)):
                if config["llm_settings"].get(field) != expected:
                    raise ValueError(f"Source worker setting mismatch: {field}")
            if config["maximum_agent_revisions"] != protocol["maximum_agent_revisions"]:
                raise ValueError("Source revision budget mismatch")
            record_root = workers[run_root.parent.name[1:]]
            for line in index.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                rid = row["run_id"]
                manifest_path = run_root / "manifests" / f"{rid}.json"
                manifest = load(manifest_path)
                journal_path = resolve_recorded_path(row["interaction_journal"])
                if digest(journal_path) != manifest["output_hashes"]["interaction_journal"]:
                    raise ValueError(f"Journal hash mismatch: {rid}")
                journals_verified += 1
                directory = output / "interaction_records" / rid
                journal = inspect_journal(journal_path, directory / "transcript.md", rid)
                expected_terminal = "attempt.completed" if row["status"] == "completed" else "attempt.failed"
                if journal["terminal"] != expected_terminal:
                    raise ValueError(f"Nonterminal source attempt: {rid}")
                four_messages += journal["four_messages_present"]
                for path, sha in journal["physics_evidence_files"].items():
                    if path not in physical:
                        physical[path] = digest(Path(path))
                        for name, reference_sha in artifact_references(load(path)):
                            reference = (resolve_recorded_path(path).parent / name).resolve()
                            physical_files.setdefault(str(reference), digest(reference))
                            if physical_files[str(reference)] != reference_sha:
                                raise ValueError(f"Physical artifact hash mismatch: {reference}")
                    if physical[path] != sha:
                        raise ValueError(f"Physical evidence hash mismatch: {path}")
                calls = verify_invocations(manifest, record_root, errors, warnings, journal["llm_responses"])
                calls_verified += len(calls)
                links=[f"# {rid} — original LLM records", "", "| Call | Agent | Status | Request | Raw response | Parsed output | Manifest |",
                    "| ---: | --- | --- | --- | --- | --- | --- |"]
                for number,call in enumerate(calls,1):
                    stored=resolve_recorded_path(call["directory"])
                    paths=[resolve_recorded_path(call["request_path"]),stored/"response.json",stored/call["parsed_artifact"],stored/"manifest.json"]
                    cells=[f"[{label}](<{relative(path,directory)}>)" if path.is_file() else "Not produced"
                        for label,path in zip(("request","response","parsed","manifest"),paths)]
                    links.append(f"| {number} | {call['agent']} | {call['status']} | "+" | ".join(cells)+" |")
                (directory/"calls.md").write_text("\n".join(links)+"\n",encoding="utf-8")
                result_path = run_root / "results" / f"{rid}.json"
                candidate = run_root / "errors" / f"{rid}.candidate.json"
                result = load(result_path) if result_path.exists() else load(candidate) if candidate.exists() else None
                for key, path in [("result", result_path), ("score", run_root / "scores" / f"{rid}.json")]:
                    if key in manifest["output_hashes"] and digest(path) != manifest["output_hashes"][key]:
                        raise ValueError(f"Acquired {key} hash mismatch: {rid}")
                numeric = case.case_dir / "groundtruth/numeric_oracles" / f"{row['episode_id']}.json"
                score = scorer.score(episode=case.get_episode(row["episode_id"]),
                    groundtruth=case.get_groundtruth(row["episode_id"]),
                    candidate=PipelineResult.model_validate(result),
                    numeric_oracle=load(numeric) if numeric.exists() else None).model_dump(mode="json") if row["status"] == "completed" and result else None
                observed_tokens = sum(int(call["usage"].get("total_tokens") or 0) for call in calls)
                audited = row
                if observed_tokens != int(row.get("total_tokens") or 0):
                    per_agent = {}
                    for call in calls:
                        usage = call["usage"]
                        per_agent.setdefault(call["agent"], Counter()).update(
                            call_count=int(call["api_request_attempted"]),
                            local_request_blocks=int(not call["api_request_attempted"]),
                            prompt_tokens=int(usage.get("prompt_tokens") or 0),
                            completion_tokens=int(usage.get("completion_tokens") or 0),
                            total_tokens=int(usage.get("total_tokens") or 0),
                            latency_seconds=call["latency_seconds"])
                    audited = dict(row, total_tokens=observed_tokens,
                        prompt_tokens=sum(v["prompt_tokens"] for v in per_agent.values()),
                        completion_tokens=sum(v["completion_tokens"] for v in per_agent.values()),
                        llm_latency_seconds=sum(v["latency_seconds"] for v in per_agent.values()),
                        per_agent_llm_usage={k:dict(v) for k,v in per_agent.items()})
                records.append({"row":audited,"score":score,"result":result,"journal":journal})
                write_json(directory / "record.json", {"run_id":rid,"index_row":row,"score":score,
                    "audited_usage_row":audited if audited is not row else None,
                    "source_journal":str(journal_path),"source_journal_sha256":digest(journal_path),
                    "manifest":str(manifest_path),"llm_invocations":calls,
                    "events":{k:v for k,v in journal.items() if not k.startswith("first_")},
                    "information_exchange":result.get("information_exchange") if result else None})
    ids = [f"ep-{n:03d}" for n in range(1,201)]
    expected = check_coverage([r["row"] for r in records], ids)
    agents = evaluate_agents(records, case, scorer, method_id=method)
    for filename, key in [("reviewer_diagnostic_labels.jsonl","reviewer_samples"),
        ("mapping_field_audit.jsonl","mapping_audit"),("computation_cost_records.jsonl","computation_cost_records")]:
        (output / filename).write_text("".join(json.dumps(item,ensure_ascii=False)+"\n" for item in agents[key]),encoding="utf-8")
    with (output / "agent_metrics.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["metric","value","denominator"])
        writer.writeheader()
        for name in (AGENTS[key][1] for key in agent_metric_keys(method)):
            writer.writerow({"metric":name,**{k:agents["metrics"][name][k] for k in ("value","denominator")}})
    combined_suite = {"episode_ids":ids,"repeats":3,"started_at":min(started),"completed_at":max(ended)}
    summary = {"schema_version":"3.0","method":method,"status":"invalid" if errors else "completed",
        "generated_at":datetime.now(timezone.utc).isoformat(),"source_suite":str(output),
        "postprocessor_sha256":digest(Path(__file__)),"api_calls_by_postprocessor":0,
        "physics_runs_by_postprocessor":0,"expected_attempts":expected,"protocol":protocol,
        "results":aggregate(records,case,combined_suite),
        "agent_level":{k:v for k,v in agents.items() if k not in {"reviewer_samples","mapping_audit","computation_cost_records"}},
        "scoring_configuration":scorer.config,"acquisition_segments":provenance,
        "evaluation_scope":"fixed_corpus_descriptive_benchmark_extended_from_first50",
        "integrity":{"journals_verified":journals_verified,"llm_invocations_verified":calls_verified,
            "unique_physics_packages_verified":len(physical),"physics_referenced_artifacts_verified":len(physical_files),
            "four_message_records":four_messages,"frozen_files_unchanged":all(not c["changed_files"] for c in source_checks),
            "source_checks":source_checks,"source_verification_scope":"Historical archives verified against their frozen manifests; continuation runtime also compared to current files.","errors":errors,
            "recording_warnings":warnings,"usage_adjustments":[]}}
    rows=write_paper_metrics(output,summary)
    summary[f"metrics_{len(rows)}"]=rows
    summary["interaction_quality"]={row["metric"]:row for row in rows if row["layer"]=="interaction_quality"}
    summary["computation_cost"]=summary["results"]["efficiency"]["framework"]
    write_json(output / "paper_metrics.json",summary)
    write_json(output / "computation_cost.json",summary["computation_cost"])
    write_report(output / "RESULTS.md",summary)
    (output/"interaction_records/INDEX.md").write_text("# Acquired records\n\n"+"\n".join(
        f"- [{rid}]({rid}/transcript.md)" for rid in sorted(r["row"]["run_id"] for r in records))+"\n",encoding="utf-8")
    if errors:
        raise RuntimeError("Derived report failed invocation integrity checks")
    return summary


def compare_reports(reports, output):
    tables={name:{row["metric"]:row for row in report.get("metrics", report.get("metrics_9", report.get("metrics_12",report.get("metrics_18",[]))))}
        for name,report in reports.items()}
    keys=list(tables["proposed"])
    rows=[{"metric":key,**{name:table.get(key,{}).get("value") for name,table in tables.items()}} for key in keys]
    with (output/"comparison_metrics.csv").open("w",encoding="utf-8",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["metric",*tables]);writer.writeheader();writer.writerows(rows)
    lines=["# Proposed and retained ablations on all 200 tasks", "",
        "Each method contains the same ep-001..ep-200 x repetitions 1-3 (600 attempts). Ablations retain the old first-50 acquisition and add only ep-051..200. All model failures remain in the denominator. Metrics are reaggregated from individual records, not averaged across segment means.", "",
        "| Metric | "+" | ".join(tables)+" |","| --- | "+" | ".join("---:" for _ in tables)+" |"]
    for row in rows:
        values=[]
        for name in tables:
            value=row[name]
            values.append("—" if row["metric"] not in tables[name] else "N/A" if value is None else
                f"{value:.2f}" if row["metric"]=="Q_qua" else f"{value:,.0f}" if row["metric"]=="total_token_consumption" else
                f"{value:.3f} s" if row["metric"] in {"llm_inference_latency","retrieval_time"} else f"{100*value:.2f}%")
        lines.append("| "+row["metric"]+" | "+" | ".join(values)+" |")
    lines += ["", "Each ablation reports its target module's three metrics. Reviewer metrics use final feedback-impact definitions and source-executable denominators.",
        "The original quality thresholds, terminal Actor sampling and first-draft Translator sampling and final Reviewer feedback-impact sampling are unchanged. This is a fixed-corpus descriptive comparison; repeated calls are not independent tasks and this corpus is not an untouched holdout."]
    (output/"COMPARISON.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    write_json(output/"comparison.json",{"api_calls":0,"physics_runs":0,"attempts_per_method":600,
        "quality_and_agent_metric_definitions_changed":False,"reports":{name:report.get("metrics", report.get("metrics_9", report.get("metrics_12",report.get("metrics_18")))) for name,report in reports.items()}})


if __name__ == "__main__":
    from finalize_selected_reports import main
    raise SystemExit(main())
