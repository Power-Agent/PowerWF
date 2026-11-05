"""Evidence-backed non-execution reports; never authorize an unsafe action."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from b2g_agent.publication import recorded_hash_matches
from b2g_agent.artifact_paths import resolve_recorded_path
from typing import Any


def validated_non_execution_report(evidence: Any) -> bool:
    """Verify a zero-service rejection against its persisted physical package.

    This establishes what the simulated workflow concluded, not a mathematical
    proof of global infeasibility. No benchmark answer is read by this gate.
    """
    if evidence is None:
        return False
    data = evidence.model_dump(mode="json") if hasattr(evidence, "model_dump") else evidence
    if not isinstance(data, dict) or data.get("receiver_decision") != "rejected":
        return False
    if data.get("committed_power_kw") != 0 or data.get("completion_percentage") != 0:
        return False
    if data.get("converged") is not True:
        return False
    try:
        path = resolve_recorded_path(data["evidence_path"])
        raw = path.read_bytes()
        if not recorded_hash_matches(path, data.get("evidence_sha256"), data=raw):
            return False
        package = json.loads(raw)
        if not isinstance(package, dict) or package.get("status") not in {"validated_full", "validated_partial"}:
            return False
        if data.get("workflow") == "bidirectional_flexibility_envelope":
            if (package.get("receiver_decision") != "rejected"
                    or package.get("completion_percentage") != 0
                    or package.get("business_metrics") != data.get("business_metrics")):
                return False
            directions = package.get("directions", {})
            if not isinstance(directions, dict) or set(directions) != {"reduction", "absorption"}:
                return False
            rejected_branches = 0
            for branch in directions.values():
                if not isinstance(branch, dict):
                    return False
                child = resolve_recorded_path(path.parent / branch["evidence_package"]).resolve()
                child.relative_to(path.parent.resolve())
                child_bytes = child.read_bytes()
                if not recorded_hash_matches(child, branch.get("sha256"), data=child_bytes):
                    return False
                child_package = json.loads(child_bytes)
                if not _validated_closed_loop(child_package):
                    return False
                child_final = child_package["final"]
                if any(child_final.get(key) != branch.get(key) for key in (
                        "receiver_decision", "committed_power_kw", "completion_percentage", "release_eligible")):
                    return False
                if _zero_service_rejection(child_final):
                    rejected_branches += 1
                elif not (child_final.get("release_eligible") is True
                          and child_final.get("receiver_decision") in {"accepted", "modified"}
                          and float(child_final.get("committed_power_kw") or 0) > 0):
                    return False
            # A symmetric envelope is zero if either direction cannot be served.
            # The aggregate report releases neither candidate branch as an action.
            return rejected_branches > 0
        final = package.get("final", {})
        return bool(
            _closed_loop_rejection(package)
            and final.get("business_metrics") == data.get("business_metrics")
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


def _zero_service_rejection(final):
    return (isinstance(final, dict) and final.get("receiver_decision") == "rejected"
            and final.get("committed_power_kw") == 0 and final.get("completion_percentage") == 0
            and final.get("release_eligible") is False)


def _closed_loop_rejection(package):
    return _validated_closed_loop(package) and _zero_service_rejection(package.get("final"))


def _validated_closed_loop(package):
    return (isinstance(package, dict)
            and package.get("status") in {"validated_full", "validated_partial"}
            and isinstance(package.get("final"), dict)
            and isinstance(package.get("iterations"), list) and bool(package["iterations"]))


CONSTRAINT_NAMES = {
    "low_voltage_intervals": "the lower voltage bound",
    "high_voltage_intervals": "the upper voltage bound",
    "line_overload_intervals": "the line thermal limit",
    "transformer_overload_intervals": "the transformer capacity limit",
    "comfort_violation_intervals": "the building comfort bounds",
}


def constraint_failure_codes(validation):
    counts = {**validation.get("grid_violations", {}),
              "comfort_violation_intervals": validation.get("comfort_violation_intervals", 0)}
    return sorted(name for name in CONSTRAINT_NAMES if float(counts.get(name) or 0) > 0)


def rejection_causes(evidence):
    """Read failed constraints from hash-verified local validation outputs."""
    if not validated_non_execution_report(evidence):
        return {}
    data = evidence.model_dump(mode="json") if hasattr(evidence, "model_dump") else evidence
    try:
        path = resolve_recorded_path(data["evidence_path"])
        package = json.loads(path.read_text(encoding="utf-8"))
        branches = {"event": path} if data["workflow"] == "closed_loop" else {
            name: resolve_recorded_path(path.parent / branch["evidence_package"]) for name, branch in package["directions"].items()}
        causes = {}
        for direction, child in branches.items():
            child_package = json.loads(child.read_text(encoding="utf-8"))
            reference = child_package["iterations"][-1]["coupled_validation"]
            validation_path = resolve_recorded_path(child.parent / reference["path"]).resolve()
            validation_path.relative_to(child.parent.resolve())
            raw = validation_path.read_bytes()
            if not recorded_hash_matches(validation_path, reference["sha256"], data=raw):
                return {}
            codes = constraint_failure_codes(json.loads(raw))
            if codes:
                causes[direction] = codes
        return causes
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return {}


def annotate_non_execution_feedback(feedback, evidence):
    """Explain the simulated rejection, also when physical evidence was cached."""
    causes = rejection_causes(evidence) if feedback.decision == "rejected" else {}
    if not causes:
        return feedback, evidence
    reason = "; ".join(f"{direction}: " + ", ".join(CONSTRAINT_NAMES[code] for code in codes)
                       for direction, codes in sorted(causes.items()))
    feedback = feedback.model_copy(update={
        "text": "I cannot commit service because the simulated candidate violates " + reason
                + ". No schedule is released, and completion is 0%.",
        "completion_basis": "Failed hard constraints under the simulated conditions: " + reason + ".",
    })
    evidence = evidence.model_copy(update={"execution_log": {
        **evidence.execution_log, "rejection_causes": causes}})
    return feedback, evidence
