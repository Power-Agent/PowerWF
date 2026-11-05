from __future__ import annotations

"""Fail-closed readiness gate for formal B2G-Agent evaluations."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.agents import TranslatorAgent  # noqa: E402
from b2g_agent.agents.llm import StructuredLLMClient  # noqa: E402
from b2g_agent.case_io import CaseBundle  # noqa: E402
from b2g_agent.evaluation import EvaluationMethodRegistry  # noqa: E402
from b2g_agent.simulation.status import simulator_status  # noqa: E402


EXPECTED_METHODS = {
    "full_multi_agent",
    "same_llm_single_agent",
    "plain_translator",
    "unstructured_actor",
    "reviewer_no_feedback",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("live", "replay"),
        default="live",
        help="LLM mode intended for the next formal run; no API call is made.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(REPO_ROOT / ".env", override=False)
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, object] = {}

    if os.getenv("B2G_EMBEDDING_BACKEND", "").strip() == "hash_test_only":
        errors.append("test-only hash embeddings are forbidden in formal evaluation")
    os.environ["B2G_EMBEDDING_BACKEND"] = "sentence_transformers_local"
    os.environ["B2G_LLM_MODE"] = args.mode

    case = CaseBundle.default(REPO_ROOT)
    readiness = case.readiness()
    checks["case_readiness"] = readiness.as_dict()
    if not readiness.ready_for_contract_evaluation:
        errors.append("episode/ground-truth contract alignment is not ready")
    if not readiness.ready_for_physics:
        errors.extend(
            f"missing physics asset: {item}" for item in readiness.missing_local_assets
        )

    status = simulator_status(case)
    checks["simulators"] = status
    if not status["energyplus"]["ready"]:
        errors.append("EnergyPlus executable is not available")
    if not status["pandapower"]["ready"]:
        errors.append("pandapower is not available")

    validate = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "validate_case.py"),
            "--require-physics",
            "--require-publishable",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    checks["groundtruth_gate"] = {
        "returncode": validate.returncode,
        "summary": _json_or_tail(validate.stdout, validate.stderr),
    }
    if validate.returncode != 0:
        errors.append("publishable ground-truth validation failed")

    embedding = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "experiments" / "prepare_embedding_model.py"),
            "--verify-only",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    checks["embedding_model"] = {
        "returncode": embedding.returncode,
        "summary": _json_or_tail(embedding.stdout, embedding.stderr),
    }
    if embedding.returncode != 0:
        errors.append("frozen all-MiniLM-L6-v2 model verification failed")

    knowledge_manifest_path = REPO_ROOT / "knowledge" / "manifest.json"
    knowledge_manifest = json.loads(
        knowledge_manifest_path.read_text(encoding="utf-8")
    )
    domain_counts = {
        domain: int(record["record_count"])
        for domain, record in knowledge_manifest.get("outputs", {}).items()
    }
    checks["knowledge_bases"] = {
        "record_counts": domain_counts,
        "knowledge_release": knowledge_manifest.get("knowledge_release"),
        "asset_lock_sha256": knowledge_manifest.get("asset_lock_sha256"),
    }
    if set(domain_counts.values()) != {180} or set(domain_counts) != {
        "building",
        "grid",
        "cross_domain",
    }:
        errors.append("knowledge partitions must each contain 180 frozen records")

    memory_path = case.case_dir / "reviewer_memory" / "verified_seed.jsonl"
    memory_manifest = json.loads(
        (case.case_dir / "reviewer_memory" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    memory_records = [
        json.loads(line)
        for line in memory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    checks["reviewer_memory"] = {
        "record_count": len(memory_records),
        "sha256": hashlib.sha256(memory_path.read_bytes()).hexdigest(),
        "source_splits": sorted({row["source_split"] for row in memory_records}),
    }
    if len(memory_records) != 150:
        errors.append("Reviewer memory must contain 150 verified seeds")
    if any(row.get("source_split") == "test" for row in memory_records):
        errors.append("Reviewer memory contains test-derived leakage")
    if memory_manifest.get("sha256") != hashlib.sha256(memory_path.read_bytes()).hexdigest():
        errors.append("Reviewer memory manifest hash mismatch")


    methods = EvaluationMethodRegistry.load(
        case.case_dir / "evaluation" / "methods.json"
    )
    method_ids = {method.method_id for method in methods.methods}
    checks["methods"] = {
        "method_ids": sorted(method_ids),
        "ready": all(method.ready for method in methods.methods),
    }
    checks["evaluation_protocol"] = {
        "episodes": 200, "repeats": 3, "expected_attempts": 600,
        "scope": "fixed_corpus_descriptive_benchmark_not_independent_holdout",
        "quality_metrics": ["Q_qua", "Q_tsr", "Q_F", "Q_A", "Q_V", "Q_G"],
        "agent_metrics": [f"F_{role}_{number}" for role in ("tra", "act", "rew") for number in (1, 2, 3)],
        "computation_cost_metrics": ["total_token_consumption", "llm_inference_latency", "retrieval_time"],
        "cost_scope": "whole_framework",
        "metric_count": 18,
        "applicable_metric_counts": {"full_multi_agent": 18, "same_llm_single_agent": 9,
                                     "plain_translator": 12, "unstructured_actor": 12, "reviewer_no_feedback": 12},
        "reviewer_metrics": "appropriate first handling / executable reverse-review reach / executable task success",
        "reviewer_memory_policy": "exclude current source episode; freeze online updates",
        "maximum_agent_revisions": int(os.getenv("B2G_MAX_AGENT_REVISIONS", "4")),
    }
    if not 0 <= checks["evaluation_protocol"]["maximum_agent_revisions"] <= 8:
        errors.append("B2G_MAX_AGENT_REVISIONS must be between 0 and 8")
    if method_ids != EXPECTED_METHODS or not all(
        method.ready for method in methods.methods
    ):
        errors.append("five-method registry is incomplete or not ready")

    llm = StructuredLLMClient.from_env(repo_root=REPO_ROOT)
    protocol = json.loads((case.case_dir / "evaluation/proposed_protocol.json").read_text(encoding="utf-8"))
    checks["evaluation_protocol"]["protocol_id"] = protocol["protocol_id"]
    for field, expected in (("model", protocol["model"]), ("temperature", protocol["temperature"]),
            ("top_p", protocol["top_p"]), ("seed", protocol["seed"]),
            ("max_output_tokens", protocol["max_output_tokens_per_call"]),
            ("max_total_tokens_per_task", protocol["max_total_tokens_per_task"]),
            ("reasoning_effort", protocol["reasoning_effort"]),
            ("maximum_schema_repairs", protocol["maximum_schema_repairs"])):
        if getattr(llm.settings, field) != expected:
            errors.append(f"LLM setting differs from the proposed protocol: {field}")
    if checks["evaluation_protocol"]["maximum_agent_revisions"] != protocol["maximum_agent_revisions"]:
        errors.append("Agent revision limit differs from the proposed protocol")
    checks["llm"] = {
        "mode": llm.mode,
        "model": llm.settings.model,
        "temperature": llm.settings.temperature,
        "top_p": llm.settings.top_p,
        "seed": llm.settings.seed,
        "parallel_tool_calls": llm.settings.parallel_tool_calls,
        "reasoning_effort": llm.settings.reasoning_effort,
        "max_total_tokens_per_task": llm.settings.max_total_tokens_per_task,
        "token_budget_scope": (
            "per_llm_call_output_only" if llm.settings.max_total_tokens_per_task is None
            else "per_call_output_and_cumulative_task_input_plus_output"
        ),
        "max_output_tokens_per_call_ceiling": llm.settings.max_output_tokens,
        "sdk_automatic_retries": llm.settings.maximum_retries,
        "maximum_schema_repairs": llm.settings.maximum_schema_repairs,
        "credential_configured": bool(
            os.getenv("OPENAI_API_KEY", "").strip()
            and not os.getenv("OPENAI_API_KEY", "").strip().lower().startswith(
                ("replace_", "your_", "example_", "<")
            )
        ),
    }
    if llm.mode != args.mode:
        errors.append("LLM mode does not match preflight mode")
    if args.mode == "live" and not checks["llm"]["credential_configured"]:
        errors.append("OPENAI_API_KEY is not configured for a live run")
    if (
        llm.settings.temperature != 0.0
        or llm.settings.top_p != 1.0
        or llm.settings.seed != 2026
        or llm.settings.parallel_tool_calls is not False
        or llm.settings.maximum_retries != 0
    ):
        errors.append("LLM reproducibility settings are not frozen")
    if llm.settings.model == "gpt-6-luna" and llm.settings.reasoning_effort != "none":
        errors.append("gpt-6-luna requires B2G_REASONING_EFFORT=none for this protocol")
    if llm.settings.max_total_tokens_per_task is not None:
        warnings.append(
            "An optional cumulative input/output task budget is enabled. "
            "The current paper configuration uses only a per-call output limit; "
            "leave B2G_MAX_TOTAL_TOKENS_PER_TASK empty to match it."
        )
    if llm.settings.model == "gpt-4.1-mini-2025-04-14":
        warnings.append(
            "The frozen model remains available but is deprecated and scheduled for "
            "shutdown on 2027-01-14; finish live acquisition before that date."
        )

    if embedding.returncode == 0:
        translator = TranslatorAgent(REPO_ROOT / "knowledge", llm=None)
        evidence_counts: dict[int, int] = {}
        partition_counts = {"building": 0, "grid": 0, "cross_domain": 0}
        for episode in case.iter_episodes():
            specification = translator.translate(
                episode.sender_message,
                episode=episode,
                sender=episode.start_role,
            )
            count = len(specification.retrieval_evidence)
            evidence_counts[count] = evidence_counts.get(count, 0) + 1
            for evidence_ref in specification.retrieval_evidence:
                domain = evidence_ref.source.split(":", 2)[1]
                partition_counts[domain] += 1
        checks["retrieval_budget"] = {
            "evidence_count_distribution": evidence_counts,
            "partition_totals": partition_counts,
        }
        if evidence_counts != {6: 200} or set(partition_counts.values()) != {400}:
            errors.append("Translator retrieval budget is not 2+2+2 for all 200 tasks")

    incomplete_records = _incomplete_llm_records(llm.record_root)
    checks["llm_record_store"] = {
        "root": str(llm.record_root),
        "incomplete_invocation_count": incomplete_records,
    }
    if incomplete_records:
        warnings.append(
            f"{incomplete_records} incomplete historical LLM invocation records are ignored "
            "by locked replay."
        )

    payload = {
        "schema_version": "1.0",
        "status": "READY" if not errors else "NOT_READY",
        "checks": checks,
        "warnings": warnings,
        "errors": errors,
        "llm_api_called": False,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    return 0 if not errors else 1


def _json_or_tail(stdout: str, stderr: str) -> object:
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return (stderr or stdout)[-2000:]


def _incomplete_llm_records(record_root: Path) -> int:
    if not record_root.is_dir():
        return 0
    incomplete = 0
    for request_dir in record_root.iterdir():
        if not request_dir.is_dir():
            continue
        invocation_root = request_dir / "i"
        legacy_root = request_dir / "invocations"
        roots = [root for root in (invocation_root, legacy_root) if root.is_dir()]
        if not roots and (request_dir / "request.json").is_file():
            if not (request_dir / "parsed.json").is_file() and not (
                request_dir / "parsed.txt"
            ).is_file():
                incomplete += 1
        for root in roots:
            for invocation in root.iterdir():
                manifest_path = invocation / "manifest.json"
                if not manifest_path.is_file():
                    incomplete += 1
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "completed":
                    incomplete += 1
    return incomplete


if __name__ == "__main__":
    raise SystemExit(main())
