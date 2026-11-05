from __future__ import annotations

import hashlib
import json
from typing import Any


VALIDATOR_VERSION = "reviewer-failure-validator-v1"
SEMANTIC_SIGNATURES = (
    "S-OBJECT-OMISSION",
    "S-ACTION-DIRECTION-REVERSAL",
    "S-BEM-BUS-MISMATCH",
    "S-UNIT-MISMATCH",
    "S-TIME-WINDOW-DRIFT",
    "S-CONSTRAINT-OMISSION",
    "S-REQUESTED-OUTPUT-OMISSION",
)
EVIDENCE_SIGNATURES = (
    "E-INVALID-DAG",
    "E-WRONG-TOOL",
    "E-NONCONVERGENCE",
    "E-MISSING-OUTPUT",
    "E-STALE-PROVENANCE",
    "E-HARD-CONSTRAINT-VIOLATION",
    "E-TIME-ALIGNMENT",
    "E-BEM-BUS-MAPPING",
)
ALL_SIGNATURES = SEMANTIC_SIGNATURES + EVIDENCE_SIGNATURES


def validate_controlled_failure(signature: str, artifact: dict[str, Any]) -> bool:
    """Independently detect one controlled semantic or evidence perturbation."""

    checks = {
        "S-OBJECT-OMISSION": lambda a: bool(a["original_objects"]) and not a["mutated_objects"],
        "S-ACTION-DIRECTION-REVERSAL": lambda a: a["original_direction"] != a["mutated_direction"],
        "S-BEM-BUS-MISMATCH": lambda a: a["expected_mapping"] != a["mutated_mapping"],
        "S-UNIT-MISMATCH": lambda a: a["source_unit"] != a["translated_unit"],
        "S-TIME-WINDOW-DRIFT": lambda a: a["source_windows"] != a["translated_windows"],
        "S-CONSTRAINT-OMISSION": lambda a: bool(set(a["source_constraints"]) - set(a["translated_constraints"])),
        "S-REQUESTED-OUTPUT-OMISSION": lambda a: bool(set(a["requested_outputs"]) - set(a["translated_outputs"])),
        "E-INVALID-DAG": lambda a: bool(set(a["action_ids"]) & set(a["dependencies"])),
        "E-WRONG-TOOL": lambda a: a["expected_tool"] != a["observed_tool"],
        "E-NONCONVERGENCE": lambda a: a["converged"] is False,
        "E-MISSING-OUTPUT": lambda a: any(value is None for value in a["requested_output_values"].values()),
        "E-STALE-PROVENANCE": lambda a: a["claimed_sha256"] != _canonical_sha256(a["evidence_payload"]),
        "E-HARD-CONSTRAINT-VIOLATION": lambda a: a["hard_constraints_passed"] is False,
        "E-TIME-ALIGNMENT": lambda a: a["building_intervals"] != a["grid_intervals"],
        "E-BEM-BUS-MAPPING": lambda a: a["expected_mapping"] != a["observed_mapping"],
    }
    if signature not in checks:
        raise ValueError(f"Unknown failure signature: {signature}")
    try:
        return bool(checks[signature](artifact))
    except (KeyError, TypeError, ValueError):
        return False


def validator_sha256() -> str:
    return hashlib.sha256((VALIDATOR_VERSION + "|" + "|".join(ALL_SIGNATURES)).encode("utf-8")).hexdigest()


def _canonical_sha256(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
