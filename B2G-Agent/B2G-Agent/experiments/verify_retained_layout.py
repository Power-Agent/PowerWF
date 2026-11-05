"""Read-only integrity check for the five retained 200-task cohorts."""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from b2g_agent.publication import recorded_hash_matches
from b2g_agent.evaluation.methods import EvaluationMethodRegistry
from retained_paths import ABLATION_SUITES, BENCHMARK, resolve_recorded_path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify():
    expected = {"proposed", "baseline", "ablation-translator", "ablation-actor", "ablation-reviewer"}
    actual = {p.name for p in BENCHMARK.iterdir() if p.is_dir()}
    assert actual == expected, {"unexpected": sorted(actual - expected), "missing": sorted(expected - actual)}
    registry = EvaluationMethodRegistry.load(ROOT / "cases/ieee33_10bem_15min/evaluation/methods.json")
    assert {m.method_id for m in registry.methods} == {"full_multi_agent", "same_llm_single_agent", *ABLATION_SUITES}
    report = {}
    physical = {}

    def verify_evidence(value):
        if isinstance(value, dict):
            recorded, claimed = value.get("evidence_path"), value.get("evidence_sha256")
            if recorded and claimed and len(str(claimed)) == 64:
                path = resolve_recorded_path(recorded)
                if path not in physical:
                    assert path.is_file(), f"Missing physics evidence: {path}"
                    physical[path] = sha256(path)
                assert recorded_hash_matches(path, claimed, actual_sha256=physical[path]), f"Physics evidence hash mismatch: {path}"
            for child in value.values():
                verify_evidence(child)
        elif isinstance(value, list):
            for child in value:
                verify_evidence(child)
    for name, directory in (("proposed", BENCHMARK / "proposed"),
                            ("baseline", BENCHMARK / "baseline")):
        manifest = json.loads((directory / "suite_manifest.json").read_text(encoding="utf-8"))
        assert manifest["coverage_verified"] and len(manifest["episode_ids"]) == 200
        frozen = json.loads((directory / "frozen_file_hashes.json").read_text(encoding="utf-8"))
        with zipfile.ZipFile(directory / "frozen_source_and_case.zip") as archive:
            assert set(archive.namelist()) == set(frozen)
            assert all(recorded_hash_matches(directory / "frozen_source_and_case.zip", frozen[filename], data=archive.read(filename), member=filename) for filename in frozen)
        record_root = directory / "interaction_records" if name == "proposed" else directory / "acquisition_history/interactions"
        checked = 0
        for record_path in record_root.glob("*/record.json"):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            verify_evidence(record)
            journal = resolve_recorded_path(record["source_journal"])
            recorded_sha = record.get("source_journal_sha256", record.get("journal_sha256"))
            assert journal.is_file() and recorded_hash_matches(journal, recorded_sha)
            for line in journal.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    verify_evidence(json.loads(line))
            checked += 1
        assert checked == 600, (name, checked)
        report[name] = {"slots": 600, "journals_verified": checked, "source_archives_verified": 1}
    for method, suite in ABLATION_SUITES.items():
        assert suite.is_dir()
        observed = Counter()
        archives = 0
        for phase, low, high in (("first50", 1, 50), ("remaining150", 51, 200)):
            source = suite / phase
            manifest = json.loads((source / "suite_manifest.json").read_text(encoding="utf-8"))
            assert manifest["method"] == method and manifest["coverage_verified"]
            assert manifest["episode_ids"] == [f"ep-{n:03d}" for n in range(low, high + 1)]
            frozen = json.loads((source / "frozen_file_hashes.json").read_text(encoding="utf-8"))
            with zipfile.ZipFile(source / "frozen_source_and_case.zip") as archive:
                assert set(archive.namelist()) == set(frozen)
                assert all(recorded_hash_matches(source / "frozen_source_and_case.zip", frozen[name], data=archive.read(name), member=name) for name in frozen)
            archives += 1
            for index in source.glob("w*/*/index.jsonl"):
                for line in index.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    assert row["method_id"] == method
                    observed[(row["episode_id"], row["repeat"])] += 1
        expected_slots = {(f"ep-{n:03d}", repeat) for n in range(1, 201) for repeat in range(1, 4)}
        assert set(observed) == expected_slots and all(count == 1 for count in observed.values())
        checked = 0
        for record_path in (suite / "interaction_records").glob("*/record.json"):
            record = json.loads(record_path.read_text(encoding="utf-8"))
            verify_evidence(record)
            journal = resolve_recorded_path(record["source_journal"])
            assert journal.is_file() and recorded_hash_matches(journal, record["source_journal_sha256"]), record["run_id"]
            for line in journal.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    verify_evidence(json.loads(line))
            checked += 1
        assert checked == 600, (method, checked)
        if method == "plain_translator":
            excluded = json.loads((suite / "remaining150/excluded-transport-or-host-rows.json").read_text(encoding="utf-8"))
            assert len(excluded) == 7 and (suite / "attempt_history/suite_manifest.json").is_file()
        report[method] = {"slots": len(observed), "journals_verified": checked, "source_archives_verified": archives}
    final = ROOT / "outputs/comparison/final/metrics.csv"
    with final.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({r["metric"] for r in rows}) == 18
    assert set(rows[0]) == {"metric", "proposed", "baseline", "translator_v2", "actor_v2", "reviewer_v4"}
    suite_for = {"proposed": BENCHMARK / "proposed",
                 "baseline": BENCHMARK / "baseline",
                 "translator_v2": ABLATION_SUITES["plain_translator"],
                 "actor_v2": ABLATION_SUITES["unstructured_actor"],
                 "reviewer_v4": ABLATION_SUITES["reviewer_no_feedback"]}
    for method, suite in suite_for.items():
        final_report = json.loads((suite / "FINAL_METRICS.json").read_text(encoding="utf-8"))
        values = {entry["metric"]: entry["value"] for entry in final_report["metrics"]}
        expected_values = {row["metric"]: float(row[method]) for row in rows if row[method]}
        assert values == expected_values, method
    return {"status": "verified", "benchmark_directories": sorted(actual), "ablations": report,
            "final_metrics": len(rows), "physics_evidence_packages_verified": len(physical),
            "api_calls": 0, "physics_runs": 0}


if __name__ == "__main__":
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
