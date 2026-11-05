"""Public constraint provenance for agents, outside the numerical kernel."""
from __future__ import annotations

import hashlib
import json

from b2g_agent.case_io.loader import CaseBundle
from b2g_agent.contracts import Constraint, EvidenceRef


def public_review_context(case: CaseBundle) -> dict:
    """Established operational context, not per-task actions or oracle values."""
    path = case.case_dir / "optimization" / "joint_dispatch.json"
    raw = path.read_bytes()
    policy = json.loads(raw)
    files = [case.case_file, path, case.case_dir / "network/bem_to_bus.csv"]
    return {
        "case_id": case.case_id, "timezone": case.manifest["timezone"],
        "time_resolution_minutes": case.manifest["time_resolution_minutes"],
        "bem_to_bus": case.bem_mapping(),
        "service_sign_convention": policy["service_sign_convention"],
        "multiobjective_window_binding": policy["multiobjective_window_binding"],
        "sources": [{"locator": f"cases/{case.case_id}/{p.relative_to(case.case_dir).as_posix()}",
                     "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files],
    }


def operating_constraints(case: CaseBundle) -> list[Constraint]:
    """Read engineering policy without reading tasks or evaluation oracles."""
    path = case.case_dir / "optimization" / "joint_dispatch.json"
    raw = path.read_bytes()
    policy = json.loads(raw)
    grid = policy["linear_grid_constraints"]
    definitions = [
        ("service_voltage_floor", "minimum_voltage_pu", "at_least", grid["minimum_voltage_pu"], "p.u.", "linear_grid_constraints.minimum_voltage_pu"),
        ("service_voltage_ceiling", "maximum_voltage_pu", "at_most", grid["maximum_voltage_pu"], "p.u.", "linear_grid_constraints.maximum_voltage_pu"),
        ("line_thermal_limit", "maximum_line_loading_pct", "at_most", grid["maximum_line_loading_pct"], "%", "linear_grid_constraints.maximum_line_loading_pct"),
        # Response-induced violations of effective bounds, not absolute
        # nominal-band violations in the baseline.
        ("comfort_limit", "comfort_violation_minutes", "at_most", 0.0, "min", "comfort.validation_rule"),
    ]
    source = f"cases/{case.case_id}/optimization/joint_dispatch.json"
    return [Constraint(name=name, metric=metric, operator=operator, value=value, unit=unit,
        origin="case_policy", evidence=[EvidenceRef(evidence_id=f"case-policy:{metric}",
            source=source, locator=f"{source}#{key}", sha256=hashlib.sha256(raw).hexdigest(),
            excerpt=(policy["comfort"]["validation_rule"] + " " + policy["replanning_policy"]
                     if metric == "comfort_violation_minutes" else f"{metric} {operator} {value} {unit}"))])
        for name, metric, operator, value, unit, key in definitions]
