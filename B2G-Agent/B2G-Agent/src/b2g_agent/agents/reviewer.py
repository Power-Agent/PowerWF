from __future__ import annotations

import hashlib
import json
from pathlib import Path
from b2g_agent.artifact_paths import resolve_recorded_path
from b2g_agent.publication import recorded_hash_matches
from typing import Any, Literal
from pydantic import ValidationError
from b2g_agent.agents.actor import PlanCompiler, SUPPORTED_REQUESTED_OUTPUTS
from b2g_agent.contracts.readiness import execution_readiness
from b2g_agent.case_io import CaseBundle
from b2g_agent.case_io.operating_policy import operating_constraints, public_review_context
from b2g_agent.agents.reviewer_prompt import SEMANTIC_REVIEW_PROMPT
from b2g_agent.contracts.semantic_audit import SemanticAuditResult
from b2g_agent.contracts.outcomes import validated_non_execution_report, rejection_causes
from b2g_agent.observability.journal import emit, traced_stage

from b2g_agent.agents.llm import StructuredLLMClient, ResponseValidationError
from b2g_agent.contracts import (
    ExecutionPlan,
    FieldStatus,
    PhysicsEvidence,
    ReviewCheck,
    ReviewIssue,
    ReviewerDecision,
    ReviewVerdict,
    TaskSpecification,
)


SEMANTIC_RISK_THRESHOLD = 0.25
EVIDENCE_RISK_THRESHOLD = 0.0

EVIDENCE_CHECKS: tuple[tuple[str, str], ...] = (
    ("contract_schema", "Task and field schemas are complete and internally consistent."),
    ("plan_validity", "The tool plan is acyclic, typed, and dependency-complete."),
    ("solver_convergence", "All required optimization and simulation tools converged."),
    ("hard_constraints", "All hard comfort, device, voltage, and loading constraints pass."),
    ("requested_outputs", "Every requested output is bound to deterministic evidence."),
    ("provenance", "Persisted evidence exists and matches its claimed content hash."),
    ("time_alignment", "Building and grid evidence share the frozen 15-minute index."),
    ("bem_bus_mapping", "Every BEM injection uses the frozen BEM-to-bus mapping."),
    ("execution_plan_binding", "The executed physics workflow is bound to the reviewed plan."),
)


class ReviewerAgent:
    """Hard checks first; memory-guided semantic review remains explicit."""

    def __init__(
        self,
        memory_path: Path | None = None,
        *,
        llm: StructuredLLMClient | None = None,
        online_memory_path: Path | None = None,
        allow_memory_writes: bool = True,
        case: CaseBundle | None = None,
        semantic_enabled: bool = True,
    ) -> None:
        self.memory_path = memory_path
        self.llm = llm
        self.online_memory_path = online_memory_path
        self.allow_memory_writes = allow_memory_writes
        self.case = case
        self.semantic_enabled = semantic_enabled

    @traced_stage("reviewer.review")
    def review(
        self,
        specification: TaskSpecification,
        plan: ExecutionPlan | None,
        *,
        simulation_status: str = "not_run",
        draft: str | None = None,
        physics_evidence: PhysicsEvidence | dict[str, object] | None = None,
        execution_log: dict[str, object] | None = None,
        source_text: str | None = None,
        require_physics_evidence: bool = False,
        review_direction: str | None = None,
        semantic_state: dict[str, object] | None = None,
        review_history: list[dict[str, Any]] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> ReviewerDecision:
        issues: list[ReviewIssue] = []
        if not specification.clarification_required:
            issues.extend(execution_readiness(specification, case=self.case))
            unsupported = sorted(set(specification.requested_outputs) - SUPPORTED_REQUESTED_OUTPUTS)
            if unsupported:
                issues.append(ReviewIssue(code="T-UNSUPPORTED-OUTPUT", responsible_agent="translator",
                    message=f"Requested output names are not bound to available evidence: {unsupported}.",
                    evidence_locator="task_specification.requested_outputs",
                    correction_required="Map the sender's meaning to supported evidence fields without dropping requirements; otherwise request clarification. Supported fields: " + ", ".join(sorted(SUPPORTED_REQUESTED_OUTPUTS))))
        missing = sorted(
            name for name, field in specification.fields.items() if field.status == FieldStatus.MISSING
        )
        if missing and not specification.clarification_required:
            issues.append(
                ReviewIssue(
                    code="T-UNDECLARED-MISSING-FIELD",
                    message=f"Missing fields were not routed for clarification: {', '.join(missing)}.",
                    responsible_agent="translator",
                    evidence_locator="task_specification.fields",
                    correction_required="Mark execution-critical missing fields for clarification.",
                )
            )
        if plan is None and not specification.clarification_required:
            issues.append(
                ReviewIssue(
                    code="A-MISSING-PLAN",
                    message="A simulation-grounded task has no execution plan.",
                    responsible_agent="actor",
                    evidence_locator="execution_plan",
                    correction_required="Compile the finite action DAG.",
                )
            )
        # Never trust a stale 'valid' flag after a contract or plan was edited.
        plan_errors = PlanCompiler().validate(plan, specification) if plan is not None else []
        if plan is not None and (not plan.valid or plan_errors):
            non_clarification = [
                error for error in sorted(set(plan.validation_errors + plan_errors)) if error != "clarification_required"
            ]
            if non_clarification:
                issues.append(
                    ReviewIssue(
                        code="A-INVALID-PLAN",
                        message=f"Plan compiler errors: {', '.join(non_clarification)}.",
                        responsible_agent="actor",
                        evidence_locator="execution_plan.validation_errors",
                        correction_required=("Recompile with these exact differences: " + json.dumps(
                            PlanCompiler().diagnostics(plan, specification), ensure_ascii=False
                        )),
                    )
                )
        if simulation_status in {"failed", "non_converged", "severe_errors"}:
            issues.append(
                ReviewIssue(
                    code="A-INVALID-SIMULATION-EVIDENCE",
                    message=f"Physical evidence status is {simulation_status}.",
                    responsible_agent="actor",
                    evidence_locator="simulation_manifest.status",
                    correction_required="Resolve the simulator failure; never fill unresolved values with LLM output.",
                )
            )
        issues.extend(
            self._validate_evidence(
                specification,
                plan,
                physics_evidence,
                execution_log=execution_log,
                required=(
                    require_physics_evidence
                    or (
                        simulation_status == "completed"
                        and bool(specification.requested_outputs)
                    )
                ),
            )
        )

        deterministic_passed = not issues
        memory_records = self._retrieve_memories(
            specification,
            {issue.code for issue in issues},
            direction_override=review_direction,
        )
        semantic_status = "not_run"
        clarification_verdict = "not_reviewed"
        clarification_required = specification.clarification_required
        semantic_checks: list[ReviewCheck] = []
        if self.semantic_enabled and self.llm is not None and self.llm.available:
            assessment = self._semantic_assessment(
                SemanticAuditResult,
                agent_name="reviewer",
                system=SEMANTIC_REVIEW_PROMPT,
                user_payload={
                    "review_phase": "receiver_feedback" if semantic_state is not None else "sender_instruction",
                    "semantic_channel_scope": "Meaning preservation, not physical feasibility. Pending physics alone is not a semantic ambiguity.",
                    "source_text": specification.source_text,
                    "reviewed_source_text": source_text or specification.source_text,
                    "natural_language_draft": draft,
                    "structured_transfer_state": (
                        semantic_state
                        if semantic_state is not None
                        else specification.model_dump(mode="json")
                    ),
                    "execution_plan": plan.model_dump(mode="json") if plan else None,
                    "simulation_status": simulation_status,
                    "physics_evidence": (
                        physics_evidence.model_dump(mode="json")
                        if isinstance(physics_evidence, PhysicsEvidence)
                        else physics_evidence
                    ),
                    "execution_log": execution_log,
                    "verified_failure_memories": memory_records,
                    "clarification_requested": specification.clarification_required,
                    "clarification_question": specification.clarification_question,
                    "public_operating_constraints": ([c.model_dump(mode="json") for c in operating_constraints(self.case)]
                        if self.case is not None and self.case.case_id == specification.case_id else []),
                    "public_case_context": (public_review_context(self.case)
                        if self.case is not None and self.case.case_id == specification.case_id else None),
                    "review_history": list(review_history or [])[-4:],
                    "conversation_history": [m for m in (conversation_history or []) if m.get("speaker") != "reviewer"][-8:],
                    "revision_changes": _revision_changes(review_history, semantic_state or specification.model_dump(mode="json"), draft),
                },
            )
            semantic_status = assessment.status
            clarification_verdict = assessment.clarification_verdict
            if specification.clarification_required:
                clarification_required = clarification_verdict != "unsupported"
            semantic_checks = assessment.checks
            above_threshold = _weighted_risk(semantic_checks) > SEMANTIC_RISK_THRESHOLD
            # Nonblocking uncertainty remains in checks and the raw response.
            # Only actionable findings enter the correction loop.
            issues.extend(i for i in assessment.issues if i.score == 1 or i.responsible_agent == "human" or above_threshold)
            for check in semantic_checks:
                if (check.score == 1 or (check.score == 0.5 and (
                    above_threshold or check.responsible_agent == "human"))) and not any(
                    issue.code == check.check_id for issue in assessment.issues
                ):
                    issues.append(
                        ReviewIssue(
                            code=check.check_id,
                            message=check.message,
                            responsible_agent=(
                                check.responsible_agent
                                if check.responsible_agent != "none"
                                else "translator"
                            ),
                            evidence_locator=check.structured_field,
                            correction_required=(
                                "Correct the cited semantic field and rerun downstream planning."
                            ),
                            score=0.5 if check.score == 0.5 else 1.0,
                            source_excerpt=check.source_excerpt,
                            structured_field=check.structured_field,
                            basis_locator=check.basis_locator,
                        )
                    )
        elif self.llm is not None and not self.llm.available:
            semantic_status = "unavailable"

        if clarification_required:
            deterministic_passed = False
            issues.append(ReviewIssue(code="T-MISSING-OR-CONFLICTING-FIELD",
                message=specification.clarification_question or "Material input is unresolved.",
                responsible_agent="translator", evidence_locator="task_specification.fields",
                correction_required="Return to the sender for one focused clarification."))

        memory_ids = [str(record["memory_id"]) for record in memory_records]
        semantic_issue_scores = [
            issue.score for issue in issues if issue.responsible_agent == "translator"
        ]
        evidence_issue_scores = [
            issue.score for issue in issues if issue.responsible_agent == "actor"
        ]
        if not semantic_checks:
            semantic_checks = _semantic_checks_from_issues(issues, source_text or specification.source_text)
        evidence_checks = _evidence_checks(
            issues,
            specification=specification,
            simulation_status=simulation_status,
            physics_evidence=physics_evidence,
        )
        semantic_risk = _weighted_risk(semantic_checks)
        evidence_risk = _weighted_risk(evidence_checks)
        has_semantic_failure = bool(semantic_issue_scores) or semantic_risk > SEMANTIC_RISK_THRESHOLD
        has_evidence_failure = bool(evidence_issue_scores) or evidence_risk > EVIDENCE_RISK_THRESHOLD
        needs_human = any(issue.responsible_agent == "human" for issue in issues)
        if clarification_required or (needs_human and clarification_verdict != "unsupported"):
            routing = "clarify_sender"
        elif has_semantic_failure and has_evidence_failure:
            routing = "translator_then_actor"
        elif has_semantic_failure:
            routing = "translator"
        elif has_evidence_failure or evidence_risk > 0.0:
            routing = "actor"
        else:
            routing = "release"
        if issues:
            if all(issue.responsible_agent == "human" for issue in issues):
                verdict = ReviewVerdict.ABSTAIN
                final_message = "The evidence is insufficient for an autonomous decision; human confirmation is required."
            else:
                verdict = ReviewVerdict.REVISE
                final_message = "The contract must be revised by the responsible stage before release."
        elif evidence_risk > EVIDENCE_RISK_THRESHOLD:
            verdict = ReviewVerdict.ABSTAIN
            final_message = (
                "The contract is structurally valid, but required physical evidence has not "
                "passed the release gate."
            )
        else:
            verdict = ReviewVerdict.PASS
            final_message = "The contract passed deterministic checks and has traceable physical evidence."
        return ReviewerDecision(
            episode_id=specification.episode_id,
            verdict=verdict,
            issues=issues,
            checked_memory_ids=memory_ids,
            deterministic_checks_passed=deterministic_passed,
            semantic_review_status=semantic_status,
            clarification_verdict=clarification_verdict,
            semantic_risk_score=semantic_risk,
            evidence_risk_score=evidence_risk,
            semantic_checks=semantic_checks,
            evidence_checks=evidence_checks,
            semantic_risk_threshold=SEMANTIC_RISK_THRESHOLD,
            evidence_risk_threshold=EVIDENCE_RISK_THRESHOLD,
            routing_decision=routing,
            final_message=final_message,
        )

    def _semantic_assessment(self, response_model, **kwargs):
        """Retry malformed Reviewer output itself; never invent a domain finding."""
        assert self.llm is not None
        payload = dict(kwargs.pop("user_payload"))
        repair_limit = getattr(getattr(self.llm, "settings", None), "maximum_schema_repairs", 2)
        for attempt in range(repair_limit + 1):
            try:
                audit = self.llm.call(response_model, user_payload=payload, **kwargs)
                if audit.review_phase != payload["review_phase"]:
                    raise ResponseValidationError("Review phase must match the supplied source direction; requests and executed outcomes have different evidence requirements")
                assessment = audit.to_review_result()
                requested = payload.get("clarification_requested", False)
                if requested and assessment.clarification_verdict == "not_applicable":
                    raise ResponseValidationError("Adjudicate the requested clarification as required or unsupported with a source-grounded basis")
                if not requested and assessment.clarification_verdict != "not_applicable":
                    raise ResponseValidationError("No Translator clarification was requested; use not_applicable and report any newly discovered ambiguity as a human issue")
                emit("reviewer.semantic_scoring", audit=audit, assessment=assessment,
                     rule="consistent/context_resolved=0; material_ambiguity=0.5; contradiction/omission=1")
                return assessment
            except (ValidationError, json.JSONDecodeError, ResponseValidationError) as exc:
                emit("reviewer.response_invalid", attempt=attempt, error=str(exc))
                if attempt == repair_limit:
                    raise RuntimeError(f"Reviewer response remained invalid after {repair_limit} format-repair attempts; release withheld") from exc
                payload = {**payload, "response_validation_error": str(exc),
                           "response_repair_instruction": "Return all eight categorical findings for the supplied review_phase. Identify any remaining material difference after using public context. Correct contradictory findings, retaining real failures; do not force either a pass or a failure. Do not output numeric scores or an overall status.",
                           "response_repair_attempt": attempt + 1}

    def record_verified_failure(
        self,
        record: dict[str, Any],
        *,
        confirmation_source: str,
        source_split: str,
        evaluation_closed: bool = False,
    ) -> Path:
        """Append Reflexion memory only after program/expert confirmation.

        Test-derived failures remain write-locked until evaluation is closed.
        """

        if not self.allow_memory_writes:
            raise ValueError("Memory writes are disabled for this runtime")
        if confirmation_source not in {"program", "expert"}:
            raise ValueError("confirmation_source must be program or expert")
        if source_split == "test" and not evaluation_closed:
            raise ValueError("test failures cannot enter memory before evaluation closes")
        required = {
            "failure_signature", "failure_channel", "responsible_agent", "direction",
            "task_families", "affected_fields", "trigger", "root_cause", "wrong_output",
            "correction", "generalized_rule", "applicability_boundary",
        }
        missing = sorted(required - set(record))
        if missing:
            raise ValueError(f"verified failure record is missing: {', '.join(missing)}")
        if self.online_memory_path is not None:
            output_path = self.online_memory_path
        else:
            output_dir = (
                self.memory_path if self.memory_path and self.memory_path.is_dir()
                else self.memory_path.parent if self.memory_path else Path.cwd()
            )
            output_path = output_dir / "verified_online.jsonl"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            **record,
            "schema_version": "2.0",
            "memory_id": str(record.get("memory_id") or f"online-{hashlib.sha256(json.dumps(record, sort_keys=True, default=str).encode()).hexdigest()[:16]}"),
            "source_split": source_split,
            "generation_method": "online_reflexion",
            "verification_evidence": {"confirmation_source": confirmation_source},
            "status": "verified_online",
            "created_before_test": source_split != "test",
        }
        if output_path.is_file():
            existing_ids = {
                str(json.loads(line).get("memory_id"))
                for line in output_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            if payload["memory_id"] in existing_ids:
                return output_path
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        return output_path

    def record_program_confirmed_failures(
        self,
        specification: TaskSpecification,
        decision: ReviewerDecision,
        *,
        source_split: str,
        evaluation_closed: bool = False,
    ) -> list[Path]:
        """Persist only failures independently established by deterministic checks."""

        if self.online_memory_path is None:
            return []
        confirmed_codes = {
            "A-MISSING-PLAN",
            "A-INVALID-PLAN",
            "A-INVALID-SIMULATION-EVIDENCE",
            "E-MISSING-EVIDENCE",
            "E-NONCONVERGENCE",
            "E-HARD-CONSTRAINT-VIOLATION",
            "E-MISSING-OUTPUT",
            "E-STALE-PROVENANCE",
            "E-MISSING-PROVENANCE",
            "E-TIME-ALIGNMENT",
            "E-BEM-BUS-MAPPING",
            "E-PLAN-NOT-BOUND",
        }
        paths: list[Path] = []
        direction = f"{specification.sender_role.value}->{specification.receiver_role.value}"
        for issue in decision.issues:
            if issue.code not in confirmed_codes or issue.responsible_agent != "actor":
                continue
            affected = [issue.structured_field or issue.evidence_locator or "physics_evidence"]
            paths.append(
                self.record_verified_failure(
                    {
                        "failure_signature": issue.code,
                        "failure_channel": "evidence",
                        "responsible_agent": "actor",
                        "direction": direction,
                        "task_families": [item.value for item in specification.task_families],
                        "affected_fields": affected,
                        "trigger": issue.message,
                        "root_cause": "deterministic Reviewer validation failure",
                        "wrong_output": issue.message,
                        "correction": issue.correction_required,
                        "generalized_rule": issue.correction_required,
                        "applicability_boundary": "Applies when the same deterministic check fails.",
                    },
                    confirmation_source="program",
                    source_split=source_split,
                    evaluation_closed=evaluation_closed,
                )
            )
        return paths

    @staticmethod
    def _validate_evidence(
        specification: TaskSpecification,
        plan: ExecutionPlan | None,
        physics_evidence: PhysicsEvidence | dict[str, object] | None,
        *,
        execution_log: dict[str, object] | None,
        required: bool,
    ) -> list[ReviewIssue]:
        issues: list[ReviewIssue] = []
        if physics_evidence is None:
            if required:
                issues.append(_actor_issue(
                    "E-MISSING-EVIDENCE",
                    "Physical completion was requested without a PhysicsEvidence package.",
                    "physics_evidence",
                    "Execute the typed plan and attach traceable evidence before release.",
                ))
            return issues
        payload = (
            physics_evidence.model_dump(mode="json")
            if isinstance(physics_evidence, PhysicsEvidence)
            else physics_evidence
        )
        if not bool(payload.get("converged")):
            issues.append(_actor_issue(
                "E-NONCONVERGENCE", "The physical solver did not converge.",
                "physics_evidence.converged", "Replan or abstain; do not release the result.",
            ))
        if not bool(payload.get("hard_constraints_passed")) and not validated_non_execution_report(payload):
            issues.append(_actor_issue(
                "E-HARD-CONSTRAINT-VIOLATION", "At least one hard physical constraint failed.",
                "physics_evidence.hard_constraints_passed", "Modify the commitment and rerun validation.",
            ))
        if payload.get("receiver_decision") == "rejected" and validated_non_execution_report(payload):
            causes = rejection_causes(payload)
            if not causes or payload.get("execution_log", {}).get("rejection_causes") != causes:
                issues.append(_actor_issue(
                    "E-REJECTION-CAUSE-MISSING", "The rejection lacks verified failed-constraint causes.",
                    "physics_evidence.execution_log.rejection_causes",
                    "Bind the failed constraints from the hashed validation output and explain them in the reply.",
                ))
        metrics = payload.get("business_metrics")
        if not isinstance(metrics, dict):
            metrics = {}
        requested_values = metrics.get("requested_output_values")
        if not isinstance(requested_values, dict):
            requested_values = metrics
        missing_outputs = [
            name for name in specification.requested_outputs
            if name not in requested_values or requested_values[name] is None
        ]
        if missing_outputs:
            issues.append(_actor_issue(
                "E-MISSING-OUTPUT",
                f"Requested evidence outputs are unresolved: {', '.join(missing_outputs)}.",
                "physics_evidence.business_metrics",
                "Produce each requested metric or mark the task incomplete and replan.",
            ))
        evidence_path = resolve_recorded_path(str(payload.get("evidence_path", "")))
        claimed_hash = str(payload.get("evidence_sha256", ""))
        if evidence_path.is_file():
            observed_hash = hashlib.sha256(evidence_path.read_bytes()).hexdigest()
            if not recorded_hash_matches(evidence_path, claimed_hash, actual_sha256=observed_hash):
                issues.append(_actor_issue(
                    "E-STALE-PROVENANCE", "The evidence file hash does not match the claimed hash.",
                    "physics_evidence.evidence_sha256", "Reload the exact evidence artifact and recompute its hash.",
                ))
        elif required:
            issues.append(_actor_issue(
                "E-MISSING-PROVENANCE",
                "The claimed physical evidence artifact is not available at its recorded path.",
                "physics_evidence.evidence_path",
                "Persist the evidence package before review and bind its verified content hash.",
            ))
        if plan is not None:
            executed_plan_id = payload.get("execution_plan_id")
            expected_plan_hash = hashlib.sha256(
                json.dumps(
                    plan.model_dump(mode="json"),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            if (
                executed_plan_id != plan.plan_id
                or payload.get("execution_plan_sha256") != expected_plan_hash
            ):
                issues.append(_actor_issue(
                    "E-PLAN-NOT-BOUND",
                    "The physical evidence is not bound to the reviewed execution plan.",
                    "physics_evidence.execution_plan_id",
                    "Execute the reviewed plan and bind its content hash to the evidence package.",
                ))
        if execution_log is not None and execution_log.get("time_alignment_passed") is False:
            issues.append(_actor_issue(
                "E-TIME-ALIGNMENT", "Building and grid execution intervals are not aligned.",
                "execution_log.time_alignment_passed", "Align both domains to the same 15-minute index.",
            ))
        if execution_log is not None and execution_log.get("bem_bus_mapping_passed") is False:
            issues.append(_actor_issue(
                "E-BEM-BUS-MAPPING", "Execution used an invalid BEM-to-bus mapping.",
                "execution_log.bem_bus_mapping_passed", "Rebind injections through the frozen case mapping.",
            ))
        return issues

    @traced_stage("retrieval.reviewer_memory")
    def _retrieve_memories(
        self,
        specification: TaskSpecification,
        issue_codes: set[str],
        *,
        limit: int = 5,
        direction_override: str | None = None,
    ) -> list[dict[str, object]]:
        if not self.memory_path or not self.memory_path.exists():
            return []
        scored: list[tuple[int, str, dict[str, object]]] = []
        direction = direction_override or f"{specification.sender_role.value}->{specification.receiver_role.value}"
        families = {family.value for family in specification.task_families}
        material_fields = set(specification.fields) | {"execution_plan", "physics_evidence"}
        for memory_file in self._memory_files():
          with memory_file.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                if not raw_line.strip():
                    continue
                record = json.loads(raw_line)
                if record.get("episode_id") == specification.episode_id or record.get("source_episode_id") == specification.episode_id:
                    # A task cannot retrieve a failure example derived from itself.
                    continue
                if record.get("status") not in {
                    "verified_seed", "programmatically_verified_seed", "verified_online"
                }:
                    continue
                score = 0
                if record.get("failure_signature") in issue_codes:
                    score += 8
                if record.get("direction") == direction:
                    score += 3
                score += len(families & set(record.get("task_families", []))) * 2
                score += len(material_fields & set(record.get("affected_fields", [])))
                if score:
                    scored.append((score, str(record["memory_id"]), record))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [record for _, _, record in scored[:limit]]

    def _memory_files(self) -> list[Path]:
        assert self.memory_path is not None
        files = (
            [self.memory_path]
            if self.memory_path.is_file()
            else sorted(self.memory_path.glob("*.jsonl"))
        )
        if self.online_memory_path is not None and self.online_memory_path.is_file():
            files.append(self.online_memory_path)
        return sorted(set(files))


def review_state(state: dict[str, Any]) -> dict[str, Any]:
    """Compact semantic snapshot; full requests still live in the durable journal."""
    snapshot = {k: v for k, v in state.items() if k not in {"retrieval_trace", "retrieval_evidence", "source_text"}}
    if isinstance(snapshot.get("fields"), dict):
        snapshot["fields"] = {name: {k: v for k, v in field.items() if k != "evidence"}
                              for name, field in snapshot["fields"].items()}
    return snapshot


def _revision_changes(history, state, draft):
    if not history:
        return None
    previous = history[-1]
    before = previous.get("structured_transfer_state", {})
    after = review_state(state)
    return {"changed_fields": sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k)),
            "draft_changed": previous.get("natural_language_draft") != draft,
            "previous_direction": previous.get("direction"),
            "previous_simulation_status": previous.get("simulation_status")}


def _weighted_risk(checks: list[ReviewCheck]) -> float:
    if not checks:
        return 0.0
    denominator = sum(check.weight for check in checks)
    return sum(check.score * check.weight for check in checks) / denominator


def _semantic_checks_from_issues(
    issues: list[ReviewIssue], source_text: str
) -> list[ReviewCheck]:
    semantic = [issue for issue in issues if issue.responsible_agent == "translator"]
    if not semantic:
        return [
            ReviewCheck(
                check_id="semantic_contract_consistency",
                channel="semantic",
                score=0.0,
                message="No deterministic semantic inconsistency was found.",
                source_excerpt=source_text[:280],
                structured_field="task_specification",
                basis_locator="deterministic_semantic_gate",
            )
        ]
    return [
        ReviewCheck(
            check_id=issue.code,
            channel="semantic",
            score=issue.score,
            message=issue.message,
            responsible_agent="translator",
            source_excerpt=issue.source_excerpt or source_text[:280],
            structured_field=issue.structured_field or issue.evidence_locator,
            basis_locator=issue.basis_locator or issue.evidence_locator,
        )
        for issue in semantic
    ]


def _evidence_checks(
    issues: list[ReviewIssue],
    *,
    specification: TaskSpecification,
    simulation_status: str,
    physics_evidence: PhysicsEvidence | dict[str, object] | None,
) -> list[ReviewCheck]:
    code_to_check = {
        "T-MISSING-OR-CONFLICTING-FIELD": "contract_schema",
        "T-UNDECLARED-MISSING-FIELD": "contract_schema",
        "A-MISSING-PLAN": "plan_validity",
        "A-INVALID-PLAN": "plan_validity",
        "A-INVALID-SIMULATION-EVIDENCE": "solver_convergence",
        "E-NONCONVERGENCE": "solver_convergence",
        "E-HARD-CONSTRAINT-VIOLATION": "hard_constraints",
        "E-MISSING-EVIDENCE": "requested_outputs",
        "E-MISSING-OUTPUT": "requested_outputs",
        "E-STALE-PROVENANCE": "provenance",
        "E-MISSING-PROVENANCE": "provenance",
        "E-TIME-ALIGNMENT": "time_alignment",
        "E-BEM-BUS-MAPPING": "bem_bus_mapping",
        "E-PLAN-NOT-BOUND": "execution_plan_binding",
    }
    failures: dict[str, ReviewIssue] = {
        code_to_check[issue.code]: issue
        for issue in issues
        if issue.responsible_agent == "actor" and issue.code in code_to_check
    }
    pending = (
        physics_evidence is None
        and bool(specification.requested_outputs)
        and simulation_status in {"not_run", "declared_by_receiver"}
    )
    checks: list[ReviewCheck] = []
    for check_id, message in EVIDENCE_CHECKS:
        issue = failures.get(check_id)
        score: Literal[0.0, 0.5, 1.0]
        if issue is not None:
            score = 0.5 if issue.score == 0.5 else 1.0
        elif pending and check_id in {
            "solver_convergence",
            "hard_constraints",
            "requested_outputs",
            "provenance",
            "execution_plan_binding",
        }:
            score = 0.5
        else:
            score = 0.0
        checks.append(
            ReviewCheck(
                check_id=check_id,
                channel="evidence",
                score=score,
                message=issue.message if issue is not None else message,
                responsible_agent="actor" if score > 0.0 else "none",
                structured_field=(
                    issue.structured_field or issue.evidence_locator
                    if issue is not None
                    else "physics_evidence"
                ),
                basis_locator=(
                    issue.basis_locator or issue.evidence_locator
                    if issue is not None
                    else "deterministic_evidence_validator"
                ),
            )
        )
    return checks


def _actor_issue(code: str, message: str, locator: str, correction: str) -> ReviewIssue:
    return ReviewIssue(
        code=code,
        message=message,
        responsible_agent="actor",
        evidence_locator=locator,
        correction_required=correction,
        structured_field=locator,
        basis_locator="deterministic_evidence_validator",
    )
