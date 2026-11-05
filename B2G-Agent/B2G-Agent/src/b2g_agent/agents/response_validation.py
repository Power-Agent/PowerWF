"""Conservative response normalization; original model replies remain immutable."""
from copy import deepcopy

from b2g_agent.contracts import SemanticReviewResult


def validate_response(response_model, payload):
    normalizations = []
    if response_model is SemanticReviewResult and isinstance(payload, dict):
        payload = deepcopy(payload)
        checks = payload.get("checks", [])
        findings = bool(payload.get("issues")) or (isinstance(checks, list) and any(
            isinstance(c, dict) and c.get("score") in (0.5, 1.0) for c in checks))
        if payload.get("status") == "passed" and findings:
            items = [*(checks if isinstance(checks, list) else []), *(payload.get("issues") or [])]
            status = "failed" if any(isinstance(c, dict) and c.get("score", 1) == 1 for c in items) else "uncertain"
            normalizations.append({"field": "status", "original": "passed", "normalized": status,
                                   "reason": "Nonzero checks/issues take precedence; no finding was removed."})
            payload["status"] = status
    # Missing/invalid checks, invalid scores and an unsupported bare failure are
    # still rejected. A malformed response can never be converted into a pass.
    return response_model.model_validate(payload), normalizations
