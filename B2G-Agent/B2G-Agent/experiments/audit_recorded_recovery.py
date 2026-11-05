"""Check current parsers/compilers against existing failures without API or physics calls."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from b2g_agent.agents.actor import PlanCompiler
from b2g_agent.agents.response_validation import validate_response
from b2g_agent.agents.wire_schema import decode_wire_value
from b2g_agent.contracts import PlanProposal, SemanticReviewResult, TaskSpecification
from b2g_agent.contracts.models import ReviewCheck, ReviewIssue
from b2g_agent.contracts.semantic_audit import SemanticAuditResult, SemanticFinding
from b2g_agent.contracts.readiness import execution_readiness
from b2g_agent.case_io import CaseBundle
from b2g_agent.observability import physics_implementation_bundle_sha256


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=REPO / "outputs/verification/recorded-recovery")
    args = parser.parse_args()
    suite = load(args.suite / "suite_manifest.json")
    roots = {str(w["worker"]): Path(w["llm_record_root"]) for w in suite["workers_config"]}
    counts, errors, details = Counter(), [], []
    compiler, case = PlanCompiler(), CaseBundle.default(REPO)
    for path in sorted(args.suite.glob("w*/*/manifests/*.json")):
        manifest = load(path)
        root = roots[path.parent.parent.parent.name[1:]]
        for digest, invocation in zip(manifest["llm_request_hashes"], manifest["llm_invocation_ids"], strict=True):
            request = load(root / digest / "request.json")
            if request["agent"] != "reviewer":
                continue
            directory = root / digest / "i" / invocation
            metadata = load(directory / "manifest.json")
            response_path = directory / "response.json"
            if hashlib.sha256(response_path.read_bytes()).hexdigest() != metadata["response_sha256"]:
                raise ValueError(f"Original response hash mismatch: {response_path}")
            content = load(response_path)["choices"][0]["message"]["content"]
            model = (SemanticAuditResult if request.get("response_schema", {}).get("title") == "SemanticAuditResult"
                     else SemanticReviewResult)
            payload = decode_wire_value(json.loads(content), model.model_json_schema())
            counts["reviewer_raw_responses"] += 1
            try:
                parsed, changes = validate_response(model, payload)
                check_model = SemanticFinding if model is SemanticAuditResult else ReviewCheck
                assert parsed.checks == [check_model.model_validate(c) for c in payload["checks"]]
                if model is SemanticAuditResult:
                    parsed.to_review_result()  # Verify local score derivation, retaining categorical findings.
                else:
                    assert parsed.issues == [ReviewIssue.model_validate(i) for i in payload["issues"]]
                counts["parsed_with_all_findings_preserved"] += 1
                counts["status_normalizations"] += bool(changes)
                counts["previously_failed_now_parseable"] += metadata["status"] == "failed"
            except Exception as exc:
                errors.append({"run_id": manifest["run_id"], "invocation_id": invocation, "error": str(exc)})
        error = "\n".join(manifest.get("errors", []))
        candidate_path = path.parent.parent / "errors" / f"{manifest['run_id']}.candidate.json"
        if "did not approve a valid execution plan" in error and candidate_path.exists():
            candidate = load(candidate_path)
            specification = TaskSpecification.model_validate(candidate["task_specification"])
            proposal = PlanProposal.model_validate({"actions": candidate["execution_plan"]["actions"]})
            revised = compiler.compile_proposal(specification, proposal)
            counts["formerly_blocked_plans"] += 1
            counts["plans_with_audited_transitive_edge_removal"] += bool(revised.normalization_log)
            counts["plans_now_contract_valid"] += revised.valid
            counts["plans_still_requiring_agent_revision"] += not revised.valid
            details.append({"run_id": manifest["run_id"], "plan_valid": revised.valid,
                            "normalization_log": revised.normalization_log,
                            "remaining_validation_errors": revised.validation_errors})
        if "target_power must be resolved" in error and candidate_path.exists():
            candidate = load(candidate_path)
            issues = execution_readiness(TaskSpecification.model_validate(candidate["task_specification"]), case=case)
            assert any(i.structured_field == "fields.target_power" for i in issues)
            counts["historical_missing_target_blocked_before_execution"] += 1
    first = load(next(args.suite.glob("w*/*/manifests/*.json")))
    frozen = load(args.suite / "frozen_file_hashes.json")
    unchanged = all(hashlib.sha256((REPO / name).read_bytes()).hexdigest() == digest
                    for name, digest in frozen.items() if name.startswith(("cases/", "knowledge/")))
    report = {
        "audit_kind": "offline_stage_regression_only", "api_calls": 0, "physical_simulations": 0,
        "counts": dict(counts), "errors": errors,
        "case_and_knowledge_files_unchanged": unchanged,
        "numeric_kernel_fingerprint_unchanged": physics_implementation_bundle_sha256(REPO) == first["input_hashes"]["physics_implementation_bundle"],
        "plan_details": details,
        "interpretation": "Parseability and plan legality are not end-to-end task success. New Translator/Actor replies and real recovery outcomes were not generated; original results remain unchanged.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "plan_details"}, ensure_ascii=False, indent=2))
    return int(bool(errors) or not unchanged or not report["numeric_kernel_fingerprint_unchanged"])


if __name__ == "__main__":
    raise SystemExit(main())
