from __future__ import annotations

import os
from b2g_agent.contracts.outcomes import annotate_non_execution_feedback
from pathlib import Path
from uuid import uuid4

from b2g_agent.agents import (
    ActorAgent,
    ReviewerAgent,
    SameLLMSingleAgent,
    TranslatorAgent,
)
from b2g_agent.agents.llm import StructuredLLMClient
from b2g_agent.case_io import CaseBundle, find_repo_root
from b2g_agent.contracts import (
    AgentRole,
    Episode,
    EvidenceRef,
    FieldStatus,
    InformationExchange,
    InteractionContext,
    PhysicsEvidence,
    PipelineResult,
    ReceiverFeedback,
    ReviewVerdict,
    TaskField,
    TaskSpecification,
)
from b2g_agent.evaluation import EvaluationMethodRegistry
from b2g_agent.orchestration.exchange import start_exchange
from b2g_agent.orchestration.exchange import apply_receiver_feedback
from b2g_agent.orchestration.task_physics import TaskPhysicsExecutor
from b2g_agent.agents.actor import PlanCompiler
from b2g_agent.agents.reviewer import review_state
from b2g_agent.contracts.readiness import execution_readiness
from b2g_agent.observability.journal import emit, journaled_turn, current_journal


class B2GPipeline:
    def __init__(
        self,
        case: CaseBundle,
        translator: TranslatorAgent,
        actor: ActorAgent,
        reviewer: ReviewerAgent,
        physics_executor: TaskPhysicsExecutor,
        *,
        method_id: str = "full_multi_agent",
        single_agent: SameLLMSingleAgent | None = None,
        maximum_agent_revisions: int = 4,
        journal_root: Path | None = None,
    ) -> None:
        self.case = case
        self.translator = translator
        self.actor = actor
        self.reviewer = reviewer
        self.physics_executor = physics_executor
        self.method_id = method_id
        self.single_agent = single_agent
        if not 0 <= maximum_agent_revisions <= 8:
            raise ValueError("maximum_agent_revisions must be between 0 and 8")
        self.maximum_agent_revisions = maximum_agent_revisions
        self.journal_root = journal_root or Path(os.getenv("B2G_JOURNAL_ROOT",
            str(find_repo_root(case.case_dir) / "outputs" / "interactions")))
        if not self.journal_root.is_absolute():
            self.journal_root = find_repo_root(case.case_dir) / self.journal_root

    @classmethod
    def default(
        cls,
        *,
        repo_root: Path | None = None,
        llm_client: StructuredLLMClient | None = None,
        method_id: str = "full_multi_agent",
        formal_evaluation: bool = False,
        runtime_root: Path | None = None,
    ) -> B2GPipeline:
        root = repo_root or find_repo_root()
        if formal_evaluation and runtime_root is not None:
            raise ValueError("A formal evaluation cannot use a personal runtime root")
        case = CaseBundle.default(root)
        mechanism_v2 = method_id in {"plain_translator", "unstructured_actor"}
        registry = EvaluationMethodRegistry.load(case.case_dir / "evaluation" / "methods.json")
        method = registry.get(method_id)
        llm = llm_client or StructuredLLMClient.from_env(repo_root=root,
            record_root=runtime_root / "lr" if runtime_root else None)
        single_agent = (
            SameLLMSingleAgent(root / "knowledge", llm=llm, case=case)
            if method.pipeline_kind == "single_agent"
            else None
        )
        reviewer_type = ReviewerAgent
        if method_id == "reviewer_no_feedback":
            from b2g_agent.agents.no_feedback_reviewer import NoCorrectiveFeedbackReviewer
            reviewer_type = NoCorrectiveFeedbackReviewer
        pipeline = cls(
            case,
            TranslatorAgent(
                root / "knowledge",
                llm=llm,
                retrieval_enabled=method.retrieval_enabled,
                case=case,
            ),
            ActorAgent(llm=llm),
            reviewer_type(
                (
                    case.case_dir / "reviewer_memory" / "verified_seed.jsonl"
                    if method.reviewer_memory_enabled
                    else None
                ),
                llm=llm if method.pipeline_kind == "multi_agent" else None,
                allow_memory_writes=not formal_evaluation,
                case=case,
                online_memory_path=(
                    (runtime_root / "memory" / "verified_online.jsonl" if runtime_root else root
                    / ".local-runs"
                    / "reviewer_memory"
                    / case.case_id
                    / "verified_online.jsonl")
                    if (
                        method.reviewer_memory_enabled
                        and not formal_evaluation
                        and os.getenv("B2G_ONLINE_MEMORY_ENABLED", "true").strip().lower()
                        not in {"0", "false", "no"}
                    )
                    else None
                ),
            ),
            TaskPhysicsExecutor(root),
            method_id=method.method_id,
            single_agent=single_agent,
            maximum_agent_revisions=int(os.getenv("B2G_MAX_AGENT_REVISIONS", "4")),
            journal_root=runtime_root / "interactions" if runtime_root else None,
        )
        if mechanism_v2:
            from b2g_agent.agents.mechanism_ablations import PlainTranslator, UnstructuredActor
            if method_id == "plain_translator":
                pipeline.translator = PlainTranslator(llm=llm, case=case)
            elif method_id == "unstructured_actor":
                pipeline.actor = UnstructuredActor(llm=llm)
        return pipeline

    def run_contract_stage(
        self,
        text: str,
        *,
        episode: Episode,
        sender: AgentRole,
        simulation_status: str = "not_run",
    ) -> PipelineResult:
        """Run one frozen benchmark turn using its preregistered case context."""
        context = InteractionContext(
            interaction_id=episode.episode_id,
            case_id=episode.case_id,
            origin="benchmark_episode",
            task_families=episode.task_families,
            operating_day=episode.operating_day,
            windows=episode.windows,
            target_bems=episode.target_bems,
            target_buses=episode.target_buses,
            communication_hint=episode.difficulty.communication,
        )
        return self.run_message(
            text,
            sender=sender,
            context=context,
            episode=episode,
            simulation_status=simulation_status,
        )

    @journaled_turn
    def run_message(
        self,
        text: str,
        *,
        sender: AgentRole,
        context: InteractionContext,
        episode: Episode | None = None,
        prior_specification: TaskSpecification | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        turn_index: int = 1,
        simulation_status: str = "not_run",
    ) -> PipelineResult:
        """Run any live turn; benchmark restrictions are enforced by the session layer."""
        if self.single_agent is not None:
            return self._run_single_agent_message(
                text,
                sender=sender,
                context=context,
                prior_specification=prior_specification,
                conversation_history=conversation_history,
                turn_index=turn_index,
                simulation_status=simulation_status,
            )
        llm = self.translator.llm
        request_offset = len(llm.request_hashes) if llm else 0
        invocation_offset = len(getattr(llm, "invocation_ids", [])) if llm else 0
        history = list(conversation_history or [])
        specification = self.translator.translate(
            text,
            sender=sender,
            episode=episode,
            context=None if episode is not None else context,
            prior_specification=prior_specification,
            conversation_history=history,
            turn_index=turn_index,
        )
        exchange = start_exchange(specification)
        plan = self.actor.plan(specification)
        review_history = []
        decision = self._review(
            specification,
            plan,
            review_history=review_history,
            simulation_status=simulation_status,
            draft=exchange.receiver_received,
            source_text=text,
            conversation_history=history,
        )
        specification, exchange, plan, decision = self._contract_corrections(
            specification, exchange, plan, decision, text=text, sender=sender,
            context=context, episode=episode, history=history,
            turn_index=turn_index, simulation_status=simulation_status,
            review_history=review_history,
        )
        hashes = llm.request_hashes[request_offset:] if llm else []
        return PipelineResult(
            episode_id=context.interaction_id, turn_index=turn_index,
            information_exchange=exchange, task_specification=specification,
            execution_plan=plan, reviewer_decision=decision, trace_id=uuid4().hex,
            llm_request_hashes=hashes, llm_invocation_ids=_new_invocation_ids(llm, invocation_offset),
            interaction_context=context, conversation_history=history,
            review_history=review_history,
        )

    def _review(self, specification, plan, *, review_history, **kwargs):
        decision = self.reviewer.review(specification, plan,
            review_history=review_history[-4:], **kwargs)
        review_history.append({
            "direction": kwargs.get("review_direction") or f"{specification.sender_role.value}->{specification.receiver_role.value}",
            "simulation_status": kwargs.get("simulation_status", "not_run"),
            "reviewed_source_text": kwargs.get("source_text") or specification.source_text,
            "structured_transfer_state": review_state(kwargs.get("semantic_state") or specification.model_dump(mode="json")),
            "natural_language_draft": kwargs.get("draft"),
            "execution_plan": plan.model_dump(mode="json") if plan else None,
            "decision": decision.model_dump(mode="json"),
        })
        del review_history[:-4]
        return decision

    def _withhold_corrective_feedback(self, decision, *, direction):
        if getattr(self.reviewer, "corrective_feedback_enabled", True):
            return False
        emit("reviewer.correction_withheld", direction=direction,
             route=decision.routing_decision, decision=decision,
             reason="reviewer_corrective_feedback_disabled")
        return True

    def _contract_corrections(self, specification, exchange, plan, decision, *,
                              text, sender, context, episode=None, history=None,
                              turn_index=1, simulation_status="not_run", review_history=None):
        history = list(history or [])
        review_history = review_history if review_history is not None else []
        revision_count = 0
        while revision_count < self.maximum_agent_revisions:
            semantic_revision = decision.routing_decision in {
                "translator",
                "translator_then_actor",
            }
            actor_revision = (
                decision.routing_decision in {"actor", "translator_then_actor"}
                and bool(decision.issues)
            )
            if not semantic_revision and not actor_revision:
                break
            if self._withhold_corrective_feedback(decision, direction="forward"):
                break
            revision_count += 1
            review_payload = [issue.model_dump(mode="json") for issue in decision.issues]
            emit("reviewer.return", direction="forward", route=decision.routing_decision,
                 revision=revision_count, decision=decision)
            if semantic_revision:
                history.append(
                    {
                        "speaker": "reviewer",
                        "text": (
                            "Revise the prior translation using these structured findings: "
                            + str(review_payload)
                        ),
                    }
                )
                specification = self.translator.translate(
                    text,
                    sender=sender,
                    episode=episode,
                    context=None if episode is not None else context,
                    prior_specification=specification,
                    conversation_history=history,
                    turn_index=turn_index,
                )
                exchange = start_exchange(specification)
            plan = self.actor.plan(
                specification,
                reviewer_feedback=review_payload,
                attempt=revision_count,
            )
            decision = self._review(
                specification,
                plan,
                review_history=review_history,
                simulation_status=simulation_status,
                draft=exchange.receiver_received,
                source_text=text,
                conversation_history=history,
            )
        decision.revision_count = revision_count
        if decision.routing_decision == "clarify_sender":
            specification, exchange = self._clarification_exchange(specification, decision)
        elif decision.issues:
            emit("workflow.blocked", reason=("contract_revision_budget_exhausted"
                if getattr(self.reviewer, "corrective_feedback_enabled", True)
                else "reviewer_corrective_feedback_disabled"), decision=decision)
        return specification, exchange, plan, decision

    @staticmethod
    def _clarification_exchange(specification, decision):
        question = specification.clarification_question or (
            "Please clarify or confirm these unresolved terms: "
            + "; ".join(dict.fromkeys(issue.message for issue in decision.issues))
        )
        specification = TaskSpecification.model_validate({
            **specification.model_dump(mode="json"), "clarification_required": True,
            "clarification_question": question,
        })
        exchange = start_exchange(specification)
        # Include the actual question, not only the generic unresolved-field label.
        exchange.receiver_sent = f"I cannot make a commitment yet. {question}"
        exchange.sender_received = f"The receiver requires clarification before committing: {question} The request is currently 0% complete."
        emit("exchange.clarification", exchange=exchange, decision=decision)
        return specification, exchange

    def _run_single_agent_message(
        self,
        text: str,
        *,
        sender: AgentRole,
        context: InteractionContext,
        prior_specification: TaskSpecification | None,
        conversation_history: list[dict[str, str]] | None,
        turn_index: int,
        simulation_status: str,
        validation_feedback: list[dict] | None = None,
    ) -> PipelineResult:
        if self.single_agent is None:
            raise RuntimeError("single-agent runtime is not configured")
        llm = self.single_agent.llm
        request_offset = len(llm.request_hashes)
        invocation_offset = len(getattr(llm, "invocation_ids", []))
        # This branch never invokes role-specific LLMs. The shared Reviewer
        # object has llm=None and supplies only deterministic tool guards.
        if self.reviewer.llm is not None:
            raise ValueError("Single-agent baseline cannot use an LLM Reviewer")
        feedback = validation_feedback or []
        for revision in range(self.maximum_agent_revisions + 1):
            proposal, plan = self.single_agent.propose_contract(
                text, sender=sender, context=context,
                prior_specification=prior_specification,
                conversation_history=conversation_history, turn_index=turn_index,
                validation_feedback=feedback,
            )
            specification = proposal.task_specification
            emit("single_agent.candidate", specification=specification, plan=plan,
                 revision=revision)
            exchange_payload = start_exchange(specification).model_dump(mode="json")
            exchange_payload["receiver_received"] = proposal.receiver_received
            if specification.clarification_required:
                exchange_payload["receiver_sent"] = proposal.clarification_receiver_feedback
                exchange_payload["sender_received"] = proposal.clarification_sender_result
            exchange = InformationExchange.model_validate(exchange_payload)
            decision = self.reviewer.review(specification, plan, simulation_status=simulation_status)
            ready = (decision.routing_decision == "actor" and plan is not None and plan.valid
                     and all(i.code == "E-MISSING-EVIDENCE" for i in decision.issues))
            if ready or (decision.routing_decision == "clarify_sender" and specification.clarification_required):
                break
            if revision == self.maximum_agent_revisions:
                emit("workflow.blocked", reason="single_agent_contract_revision_budget_exhausted",
                     decision=decision)
                break
            feedback = [i.model_dump(mode="json") for i in decision.issues]
            prior_specification = specification
            emit("single_agent.revision", direction="forward", revision=revision + 1,
                 decision=decision)
        return PipelineResult(
            episode_id=context.interaction_id,
            turn_index=turn_index,
            information_exchange=exchange,
            task_specification=specification,
            execution_plan=plan,
            reviewer_decision=decision,
            trace_id=uuid4().hex,
            llm_request_hashes=llm.request_hashes[request_offset:],
            llm_invocation_ids=_new_invocation_ids(llm, invocation_offset),
            interaction_context=context, conversation_history=conversation_history or [],
        )

    @journaled_turn
    def execute_physics(
        self,
        result: PipelineResult,
        *,
        run_root: Path,
        cache=None,
        attempt_log: list[dict] | None = None,
    ) -> PipelineResult:
        if result.information_exchange.decision != "pending_physics":
            raise ValueError("Only a pending physics turn can be executed")
        if result.workflow_status == "blocked":
            emit("workflow.blocked", reason="reviewer_withheld_physics_authorization",
                 decision=result.reviewer_decision)
            return result.model_copy(deep=True)
        pending = result.model_copy(deep=True)
        review_history = list(pending.review_history)
        llm = self.translator.llm
        request_offset = len(llm.request_hashes) if llm else 0
        invocation_offset = len(getattr(llm, "invocation_ids", [])) if llm else 0
        attempts = attempt_log if attempt_log is not None else []

        def finish(completed):
            # Include *all* forward, reverse and Actor revisions, including ones
            # preceding the final candidate. Never drop intermediate invocations.
            completed.llm_request_hashes = [*result.llm_request_hashes, *(llm.request_hashes[request_offset:] if llm else [])]
            completed.llm_invocation_ids = [*result.llm_invocation_ids, *_new_invocation_ids(llm, invocation_offset)]
            completed.physics_attempts = list(attempts)
            completed.review_history = list(review_history)
            return completed

        for attempt in range(self.maximum_agent_revisions + 1):
            specification, plan = pending.task_specification, pending.execution_plan
            readiness = execution_readiness(specification, case=self.case)
            emit("execution.preflight", issues=readiness, specification=specification, plan=plan)
            if readiness:
                decision = pending.reviewer_decision.model_copy(update={
                    "routing_decision": "clarify_sender", "verdict": ReviewVerdict.REVISE,
                    "issues": readiness, "deterministic_checks_passed": False,
                    "semantic_risk_score": 1.0,
                    "final_message": "Execution-critical input or case data requires clarification.",
                })
                spec, exchange = self._clarification_exchange(specification, decision)
                pending.task_specification, pending.information_exchange = spec, exchange
                pending.reviewer_decision = decision
                return finish(pending)
            errors = PlanCompiler().validate(plan, specification) if plan is not None else ["missing_plan"]
            if plan is None or not plan.valid or errors:
                raise ValueError(f"A freshly validated Actor plan is required before physics execution: {errors}")
            blocking_issues = [i for i in pending.reviewer_decision.issues if i.code != "E-MISSING-EVIDENCE"]
            if pending.reviewer_decision.routing_decision != "actor" or blocking_issues:
                raise ValueError("Physics dispatch requires a clean Reviewer actor route")
            attempt_root = run_root if attempt == 0 else run_root / f"reviewer-actor-revision-{attempt:02d}"
            cached = cache.load(specification, plan) if cache is not None else None
            cache_key = None
            status = "disabled" if cache is None else "hit" if cached else "miss"
            if cache is not None:
                if cached:
                    cache_key = cached[2]
                else:
                    attempt_root, cache_key = cache.allocate_run_root(specification, plan)
            entry = {"attempt": attempt, "cache_status": status, "cache_key": cache_key,
                     "run_root": str(attempt_root), "status": "running", "plan_id": plan.plan_id}
            attempts.append(entry)
            emit("physics.started", attempt=entry, specification=specification, plan=plan)
            try:
                if cached:
                    feedback, evidence, _ = cached
                else:
                    feedback, evidence = self.physics_executor.run(specification, plan, run_root=attempt_root)
                    if cache is not None:
                        cache.store(specification, execution_plan=plan, feedback=feedback, evidence=evidence)
            except Exception as exc:
                entry.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                emit("physics.failed", attempt=entry)
                # The exception becomes evidence-channel feedback, never an LLM
                # replacement for missing physical results.
                decision = self._review(specification, plan, simulation_status="failed", review_history=review_history,
                    conversation_history=pending.conversation_history,
                    draft=pending.information_exchange.receiver_received, source_text=specification.source_text)
                completed = pending.model_copy(deep=True)
                completed.reviewer_decision = decision
            else:
                feedback, evidence = annotate_non_execution_feedback(feedback, evidence)
                entry.update(status="completed", evidence_path=evidence.evidence_path,
                             evidence_sha256=evidence.evidence_sha256)
                emit("physics.completed", attempt=entry, feedback=feedback, evidence=evidence)
                pending.review_history = list(review_history)
                completed = self.complete_with_physics(pending, feedback=feedback, evidence=evidence)
                review_history[:] = completed.review_history
            if completed.reviewer_decision.routing_decision == "release":
                emit("exchange.released", exchange=completed.information_exchange)
                return finish(completed)
            if completed.reviewer_decision.routing_decision == "clarify_sender":
                spec, exchange = self._clarification_exchange(specification, completed.reviewer_decision)
                completed.task_specification, completed.information_exchange = spec, exchange
                return finish(completed)
            if self._withhold_corrective_feedback(completed.reviewer_decision, direction="physics"):
                emit("workflow.blocked", reason="reviewer_corrective_feedback_disabled", result=completed)
                return finish(completed)
            if (
                completed.reviewer_decision.routing_decision
                not in {"actor", "translator_then_actor"}
                or attempt >= self.maximum_agent_revisions
            ):
                emit("workflow.blocked", reason="physics_or_reverse_revision_budget_exhausted", result=completed)
                self.reviewer.record_program_confirmed_failures(
                    completed.task_specification,
                    completed.reviewer_decision,
                    source_split="interactive",
                )
                return finish(completed)
            # Reverse semantic edits are handled before Actor in complete_with_physics.
            # If that budget was exhausted, do not skip the unresolved Translator.
            if completed.physics_evidence is not None and completed.reviewer_decision.routing_decision == "translator_then_actor":
                emit("workflow.blocked", reason="reverse_semantic_revision_budget_exhausted", result=completed)
                return finish(completed)
            emit("reviewer.return", direction="physics", route=completed.reviewer_decision.routing_decision,
                 revision=attempt + 1, decision=completed.reviewer_decision)
            review_payload = [
                issue.model_dump(mode="json")
                for issue in completed.reviewer_decision.issues
            ]
            review_payload.append({"code": "PHYSICS-ATTEMPT-CONTEXT", "attempt": entry,
                                   "evidence": completed.physics_evidence.model_dump(mode="json") if completed.physics_evidence else None})
            context = pending.interaction_context or InteractionContext(
                interaction_id=specification.episode_id, case_id=specification.case_id,
                origin=specification.context_origin, task_families=specification.task_families,
                operating_day=specification.operating_day, windows=specification.windows,
                target_bems=specification.target_bems, target_buses=specification.target_buses,
            )
            # An execution exception can expose a forward contract issue too.
            if self.single_agent is not None:
                emit("single_agent.revision", direction="physics", revision=attempt + 1,
                     decision=completed.reviewer_decision)
                revised = self._run_single_agent_message(
                    specification.source_text, sender=specification.sender_role, context=context,
                    prior_specification=specification, conversation_history=pending.conversation_history,
                    turn_index=pending.turn_index, simulation_status="not_run",
                    validation_feedback=review_payload,
                )
                # Keep the complete invocation stream through finish(), including
                # failed physical candidates; never borrow Translator/Actor calls.
                pending = revised
                if (pending.information_exchange.decision == "needs_clarification"
                        or pending.execution_plan is None or not pending.execution_plan.valid
                        or any(i.code != "E-MISSING-EVIDENCE" for i in pending.reviewer_decision.issues)):
                    return finish(pending)
                continue
            if completed.reviewer_decision.routing_decision == "translator_then_actor":
                specification = self.translator.translate(specification.source_text,
                    sender=specification.sender_role, context=context, prior_specification=specification,
                    conversation_history=[*pending.conversation_history,
                        {"speaker": "reviewer", "text": str(review_payload)}], turn_index=pending.turn_index)
            revised_plan = self.actor.plan(
                specification,
                reviewer_feedback=review_payload,
                attempt=attempt + 1,
            )
            plan_review = self._review(
                specification,
                revised_plan,
                review_history=review_history,
                simulation_status="not_run",
                conversation_history=pending.conversation_history,
                draft=start_exchange(specification).receiver_received,
                source_text=specification.source_text,
            )
            spec, exchange, revised_plan, plan_review = self._contract_corrections(
                specification, start_exchange(specification), revised_plan, plan_review,
                text=specification.source_text, sender=specification.sender_role, context=context,
                history=pending.conversation_history, turn_index=pending.turn_index,
                review_history=review_history,
            )
            pending = PipelineResult.model_validate(
                {
                    **pending.model_dump(mode="json"),
                    "task_specification": spec.model_dump(mode="json"),
                    "information_exchange": exchange.model_dump(mode="json"),
                    "execution_plan": (
                        revised_plan.model_dump(mode="json") if revised_plan else None
                    ),
                    "reviewer_decision": plan_review.model_dump(mode="json"),
                    "physics_evidence": None,
                    "reverse_transfer": None,
                    "proposed_receiver_feedback": None,
                    "review_history": list(review_history),
                }
            )
            if exchange.decision == "needs_clarification" or revised_plan is None or not revised_plan.valid or plan_review.issues:
                emit("workflow.blocked", reason="revised_contract_not_executable", result=pending)
                return finish(pending)
        raise AssertionError("Bounded physics loop must return a terminal candidate")

    @journaled_turn
    def complete_with_physics(
        self,
        result: PipelineResult,
        *,
        feedback: ReceiverFeedback,
        evidence: PhysicsEvidence,
    ) -> PipelineResult:
        """Finish a turn from validated physics, including cached evidence.

        The deterministic simulator output may be shared across repeated methods
        when the task contract, physics inputs, and implementation fingerprint
        are identical. Agent-specific feedback/review calls still run here for
        every method and repetition.
        """
        if result.information_exchange.decision != "pending_physics":
            raise ValueError("Only a pending physics turn can be completed")
        if result.execution_plan is None or not result.execution_plan.valid:
            raise ValueError("A valid Actor plan is required before physics completion")
        if evidence.receiver_decision != feedback.decision:
            raise ValueError("Cached physics evidence and receiver feedback disagree")
        if evidence.completion_percentage != feedback.completion_percentage:
            raise ValueError("Cached physics completion and receiver feedback disagree")
        review_history = list(result.review_history)
        if self.single_agent is not None:
            llm = self.single_agent.llm
            request_offset = len(llm.request_hashes)
            invocation_offset = len(getattr(llm, "invocation_ids", []))
            feedback = self.single_agent.propose_feedback(
                specification=result.task_specification,
                physics_evidence=evidence.model_dump(mode="json"),
                tool_feedback=feedback,
            )
            exchange = apply_receiver_feedback(result.information_exchange, feedback,
                translated_sender_message=feedback.sender_message)
            reviewer = self._review(
                result.task_specification,
                result.execution_plan,
                review_history=review_history,
                simulation_status="completed",
                conversation_history=result.conversation_history,
                draft=exchange.sender_received,
                physics_evidence=evidence,
                execution_log=evidence.execution_log,
                source_text=feedback.text,
                require_physics_evidence=True,
            )
            return PipelineResult.model_validate(
                {
                    **result.model_dump(mode="json"),
                    "information_exchange": exchange.model_dump(mode="json"),
                    "reviewer_decision": reviewer.model_dump(mode="json"),
                    "physics_evidence": evidence.model_dump(mode="json"),
                    "review_history": review_history,
                    "llm_request_hashes": [
                        *result.llm_request_hashes,
                        *llm.request_hashes[request_offset:],
                    ],
                    "llm_invocation_ids": [
                        *result.llm_invocation_ids,
                        *_new_invocation_ids(llm, invocation_offset),
                    ],
                }
            )
        llm = self.reviewer.llm
        request_offset = len(llm.request_hashes) if llm else 0
        invocation_offset = len(getattr(llm, "invocation_ids", [])) if llm else 0
        translated_feedback = self.translator.translate_feedback(
            feedback,
            specification=result.task_specification,
            physics_evidence=evidence.model_dump(mode="json"),
        )
        exchange = apply_receiver_feedback(
            result.information_exchange,
            feedback,
            translated_sender_message=translated_feedback.translated_message,
        )
        completed_specification = _backfill_physics_fields(
            result.task_specification,
            evidence,
        )
        reviewer = self._review(
            completed_specification,
            result.execution_plan,
            review_history=review_history,
            simulation_status="completed",
            conversation_history=result.conversation_history,
            draft=translated_feedback.translated_message,
            physics_evidence=evidence,
            execution_log=evidence.execution_log,
            source_text=feedback.text,
            require_physics_evidence=True,
            review_direction=(
                f"{result.task_specification.receiver_role.value}->"
                f"{result.task_specification.sender_role.value}"
            ),
            semantic_state=translated_feedback.model_dump(mode="json"),
        )
        reverse_revision_count = 0
        reverse_history: list[dict[str, str]] = []
        while (
            reviewer.routing_decision in {"translator", "translator_then_actor"}
            and reverse_revision_count < self.maximum_agent_revisions
        ):
            if self._withhold_corrective_feedback(reviewer, direction="reverse"):
                break
            reverse_revision_count += 1
            emit("reviewer.return", direction="reverse", route=reviewer.routing_decision,
                 revision=reverse_revision_count, decision=reviewer)
            reverse_history.append(
                {
                    "speaker": "reviewer",
                    "text": str(
                        [
                            issue.model_dump(mode="json")
                            for issue in reviewer.issues
                            if issue.responsible_agent == "translator"
                        ]
                    ),
                }
            )
            translated_feedback = self.translator.translate_feedback(
                feedback,
                specification=completed_specification,
                physics_evidence=evidence.model_dump(mode="json"),
                conversation_history=reverse_history,
            )
            exchange = apply_receiver_feedback(
                result.information_exchange,
                feedback,
                translated_sender_message=translated_feedback.translated_message,
            )
            reviewer = self._review(
                completed_specification,
                result.execution_plan,
                review_history=review_history,
                simulation_status="completed",
                conversation_history=result.conversation_history,
                draft=translated_feedback.translated_message,
                physics_evidence=evidence,
                execution_log=evidence.execution_log,
                source_text=feedback.text,
                require_physics_evidence=True,
                review_direction=(
                    f"{result.task_specification.receiver_role.value}->"
                    f"{result.task_specification.sender_role.value}"
                ),
                semantic_state=translated_feedback.model_dump(mode="json"),
            )
        reviewer.revision_count = reverse_revision_count
        new_hashes = llm.request_hashes[request_offset:] if llm else []
        return PipelineResult.model_validate(
            {
                **result.model_dump(mode="json"),
                "information_exchange": exchange.model_dump(mode="json"),
                "task_specification": completed_specification.model_dump(mode="json"),
                "reviewer_decision": reviewer.model_dump(mode="json"),
                "physics_evidence": evidence.model_dump(mode="json"),
                "reverse_transfer": translated_feedback.model_dump(mode="json"),
                "review_history": review_history,
                "proposed_receiver_feedback": None,
                "llm_request_hashes": [*result.llm_request_hashes, *new_hashes],
                "llm_invocation_ids": [
                    *result.llm_invocation_ids,
                    *_new_invocation_ids(llm, invocation_offset),
                ],
            }
        )

    @journaled_turn
    def complete_with_receiver_feedback(
        self,
        result: PipelineResult,
        *,
        feedback: ReceiverFeedback,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> PipelineResult:
        """Close an interactive turn from an actual receiver response."""

        if result.information_exchange.decision != "pending_physics":
            raise ValueError("Only a pending turn can receive feedback")
        review_history = list(result.review_history)
        llm = self.translator.llm
        request_offset = len(llm.request_hashes) if llm else 0
        invocation_offset = len(getattr(llm, "invocation_ids", [])) if llm else 0
        translated = self.translator.translate_feedback(
            feedback,
            specification=result.task_specification,
            conversation_history=conversation_history,
        )
        requires_validation = translated.requires_simulation_validation
        exchange = (
            result.information_exchange
            if requires_validation
            else apply_receiver_feedback(
                result.information_exchange,
                feedback,
                translated_sender_message=translated.translated_message,
            )
        )
        reviewer = self._review(
            result.task_specification,
            result.execution_plan,
            review_history=review_history,
            simulation_status="declared_by_receiver",
            conversation_history=conversation_history or result.conversation_history,
            draft=translated.translated_message,
            source_text=feedback.text,
            require_physics_evidence=requires_validation,
            review_direction=(
                f"{result.task_specification.receiver_role.value}->"
                f"{result.task_specification.sender_role.value}"
            ),
            semantic_state=translated.model_dump(mode="json"),
        )
        reverse_history = list(conversation_history or [])
        revision = 0
        while reviewer.routing_decision in {"translator", "translator_then_actor"} and revision < self.maximum_agent_revisions:
            if self._withhold_corrective_feedback(reviewer, direction="receiver_feedback"):
                break
            revision += 1
            emit("reviewer.return", direction="receiver_feedback", route=reviewer.routing_decision,
                 revision=revision, decision=reviewer)
            reverse_history.append({"speaker": "reviewer", "text": str([
                issue.model_dump(mode="json") for issue in reviewer.issues if issue.responsible_agent == "translator"
            ])})
            translated = self.translator.translate_feedback(feedback,
                specification=result.task_specification, conversation_history=reverse_history)
            requires_validation = translated.requires_simulation_validation
            exchange = result.information_exchange if requires_validation else apply_receiver_feedback(
                result.information_exchange, feedback, translated_sender_message=translated.translated_message)
            reviewer = self._review(result.task_specification, result.execution_plan,
                review_history=review_history,
                conversation_history=conversation_history or result.conversation_history,
                simulation_status="declared_by_receiver", draft=translated.translated_message,
                source_text=feedback.text, require_physics_evidence=requires_validation,
                review_direction=f"{result.task_specification.receiver_role.value}->{result.task_specification.sender_role.value}",
                semantic_state=translated.model_dump(mode="json"))
        reviewer.revision_count = revision
        final_specification = result.task_specification
        if reviewer.routing_decision == "clarify_sender":
            final_specification, exchange = self._clarification_exchange(final_specification, reviewer)
        new_hashes = llm.request_hashes[request_offset:] if llm else []
        return PipelineResult.model_validate(
            {
                **result.model_dump(mode="json"),
                "information_exchange": exchange.model_dump(mode="json"),
                "task_specification": final_specification.model_dump(mode="json"),
                "reviewer_decision": reviewer.model_dump(mode="json"),
                "reverse_transfer": translated.model_dump(mode="json"),
                "review_history": review_history,
                "proposed_receiver_feedback": (
                    feedback.model_dump(mode="json") if requires_validation else None
                ),
                "llm_request_hashes": [*result.llm_request_hashes, *new_hashes],
                "llm_invocation_ids": [
                    *result.llm_invocation_ids,
                    *_new_invocation_ids(llm, invocation_offset),
                ],
            }
        )


def _backfill_physics_fields(
    specification: TaskSpecification,
    evidence: PhysicsEvidence,
) -> TaskSpecification:
    fields = dict(specification.fields)
    evidence_ref = EvidenceRef(
        evidence_id=f"physics-{specification.episode_id}",
        source="physics_evidence",
        locator=evidence.evidence_path,
        sha256=evidence.evidence_sha256,
        excerpt=f"{evidence.status}; converged={evidence.converged}",
    )
    requested_values = evidence.business_metrics.get("requested_output_values")
    if not isinstance(requested_values, dict):
        requested_values = evidence.business_metrics
    for output in specification.requested_outputs:
        prior = fields.get(output)
        value = requested_values.get(output)
        fields[output] = TaskField(
            value=value,
            unit=prior.unit if prior else None,
            status=FieldStatus.RESOLVED if value is not None else FieldStatus.UNRESOLVED,
            confidence=1.0,
            evidence=[evidence_ref],
            note=(
                None if value is not None
                else "The completed physics evidence did not contain this requested output."
            ),
        )
    return TaskSpecification.model_validate(
        {**specification.model_dump(mode="json"), "fields": fields}
    )


def _new_invocation_ids(llm: object | None, offset: int) -> list[str]:
    if llm is None:
        return []
    return list(getattr(llm, "invocation_ids", [])[offset:])
