"""Synthetic control-flow fixtures. This adapter is never used by runtime agents."""
from b2g_agent.contracts import SemanticReviewResult


def categorical_fixture(payload, phase="sender_instruction"):
    """Keep existing numeric fault-injection tests, with explicit synthetic meanings."""
    if "review_phase" in payload:
        return payload
    previous = SemanticReviewResult.model_validate(payload)
    checks = []
    for check in previous.checks:
        issue = next((i for i in previous.issues if i.code == check.check_id), None)
        passed = check.score == 0
        checks.append({
            "check_id": check.check_id,
            "finding": "consistent" if passed else "material_ambiguity" if check.score == 0.5 else "contradiction",
            "expected_meaning": "Controlled expected meaning",
            "observed_meaning": "Controlled expected meaning" if passed else "Controlled different meaning",
            "message": check.message,
            "source_excerpt": check.source_excerpt or "Controlled source",
            "structured_field": check.structured_field or "fixture.field",
            "basis_locator": check.basis_locator or "test:controlled_fixture",
            "responsible_agent": "none" if passed else check.responsible_agent if check.responsible_agent != "none" else "translator",
            "correction_required": None if passed else issue.correction_required if issue else "Resolve the controlled difference.",
            "ambiguity": {"alternatives": ["Controlled interpretation A", "Controlled interpretation B"],
                "execution_difference": check.uncertainty_reason or "Controlled alternative actions differ.",
                "unresolved_after_context": True} if check.score == 0.5 else None,
        })
    return {"review_phase": phase, "checks": checks,
            "clarification_verdict": previous.clarification_verdict,
            "clarification_basis": previous.clarification_basis}
