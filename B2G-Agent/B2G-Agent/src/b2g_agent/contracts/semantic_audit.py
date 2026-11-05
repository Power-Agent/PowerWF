"""Categorical semantic findings; numeric risk is derived locally, never voted twice."""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from b2g_agent.contracts.models import (
    SEMANTIC_REVIEW_CHECK_IDS, ReviewCheck, ReviewIssue, SemanticReviewResult, StrictModel,
)


class MaterialAmbiguity(StrictModel):
    alternatives: list[str] = Field(min_length=2, max_length=3)
    execution_difference: str = Field(min_length=1)
    unresolved_after_context: bool

    @model_validator(mode="after")
    def distinct_interpretations(self):
        alternatives = [value.strip().casefold() for value in self.alternatives]
        if any(not value for value in alternatives) or len(set(alternatives)) != len(alternatives):
            raise ValueError("Material ambiguity needs distinct, nonempty executable interpretations")
        if not self.unresolved_after_context:
            raise ValueError("An ambiguity resolved by available context is not unresolved")
        if not self.execution_difference.strip():
            raise ValueError("Name the material execution or decision difference")
        return self


class SemanticFinding(StrictModel):
    check_id: str
    finding: Literal["consistent", "context_resolved", "material_ambiguity", "contradiction", "omission"]
    expected_meaning: str = Field(min_length=1)
    observed_meaning: str = Field(min_length=1)
    message: str = Field(min_length=1)
    source_excerpt: str = Field(min_length=1)
    structured_field: str = Field(min_length=1)
    basis_locator: str = Field(min_length=1)
    responsible_agent: Literal["translator", "actor", "human", "none"]
    correction_required: str | None = None
    ambiguity: MaterialAmbiguity | None = None

    @model_validator(mode="after")
    def coherent_finding(self):
        passed = self.finding in {"consistent", "context_resolved"}
        if passed and (self.responsible_agent != "none" or self.correction_required is not None or self.ambiguity is not None):
            raise ValueError("Consistent/context-resolved findings have no correction owner, correction or ambiguity")
        if not passed and (self.responsible_agent == "none" or not (self.correction_required or "").strip()):
            raise ValueError("A material finding needs a correction owner and actionable correction")
        if (self.finding == "material_ambiguity") != (self.ambiguity is not None):
            raise ValueError("Only material_ambiguity requires two interpretations and their execution difference")
        if not passed and self.expected_meaning.strip().casefold() == self.observed_meaning.strip().casefold():
            raise ValueError("Identical expected and observed meanings do not establish a material semantic difference")
        return self


class SemanticAuditResult(StrictModel):
    """Live response schema. Legacy numeric records remain readable in their own schema."""

    review_phase: Literal["sender_instruction", "receiver_feedback"]
    checks: list[SemanticFinding] = Field(min_length=8, max_length=8)
    clarification_verdict: Literal["not_applicable", "required", "unsupported"]
    clarification_basis: str | None = None

    @model_validator(mode="after")
    def complete_audit(self):
        ids = [check.check_id for check in self.checks]
        if len(set(ids)) != 8 or set(ids) != SEMANTIC_REVIEW_CHECK_IDS:
            raise ValueError("Semantic audit must report the complete fixed eight-check checklist")
        if self.clarification_verdict != "not_applicable" and not (self.clarification_basis or "").strip():
            raise ValueError("Clarification adjudication requires a source-grounded basis")
        if self.clarification_verdict == "unsupported" and not any(
            check.check_id == "semantic_decision_completion"
            and check.finding in {"contradiction", "omission"}
            and check.responsible_agent == "translator" for check in self.checks
        ):
            raise ValueError("Unsupported clarification must identify a Translator decision/completion failure")
        return self

    def to_review_result(self) -> SemanticReviewResult:
        scores = {"consistent": 0.0, "context_resolved": 0.0, "material_ambiguity": 0.5,
                  "contradiction": 1.0, "omission": 1.0}
        checks, issues = [], []
        for finding in self.checks:
            score = scores[finding.finding]
            uncertainty = (" / ".join(finding.ambiguity.alternatives) + "; " + finding.ambiguity.execution_difference
                           if finding.ambiguity else None)
            check = ReviewCheck(check_id=finding.check_id, channel="semantic", score=score,
                message=finding.message, responsible_agent=finding.responsible_agent,
                source_excerpt=finding.source_excerpt, structured_field=finding.structured_field,
                basis_locator=finding.basis_locator, uncertainty_reason=uncertainty)
            checks.append(check)
            if score:
                issues.append(ReviewIssue(code=check.check_id, score=score, message=check.message,
                    responsible_agent=check.responsible_agent, correction_required=finding.correction_required,
                    evidence_locator=check.structured_field, source_excerpt=check.source_excerpt,
                    structured_field=check.structured_field, basis_locator=check.basis_locator))
        status = "failed" if any(c.score == 1 for c in checks) else "uncertain" if any(c.score == 0.5 for c in checks) else "passed"
        return SemanticReviewResult(status=status, checks=checks, issues=issues,
            clarification_verdict=self.clarification_verdict, clarification_basis=self.clarification_basis)
