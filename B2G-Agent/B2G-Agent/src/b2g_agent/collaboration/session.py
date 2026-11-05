from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from b2g_agent.case_io import CaseBundle, find_repo_root
from b2g_agent.collaboration.models import (
    ChatMessage,
    FinalReview,
    SessionMode,
    SessionSnapshot,
)
from b2g_agent.contracts import (
    AgentRole,
    InteractionContext,
    PipelineResult,
    ReceiverFeedback,
)
from b2g_agent.orchestration import B2GPipeline
from b2g_agent.personal_workspace import PersonalWorkspace
from b2g_agent.observability.journal import durable_write


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CollaborationSession:
    """One persisted conversation containing one or many four-state exchanges."""

    def __init__(
        self,
        *,
        session_id: str,
        user_role: AgentRole,
        pipeline: B2GPipeline,
        output_root: Path,
        mode: SessionMode = "interactive",
        episode_id: str | None = None,
    ) -> None:
        self.session_id = session_id
        self.user_role = user_role
        self.pipeline = pipeline
        self.mode = mode
        self.episode = (
            pipeline.case.get_episode(episode_id) if episode_id is not None else None
        )
        if mode == "benchmark" and self.episode is None:
            raise ValueError("Benchmark mode requires a frozen episode")
        if (
            mode == "benchmark"
            and self.episode is not None
            and self.episode.start_role != user_role
        ):
            raise ValueError(
                f"Episode {episode_id} starts with {self.episode.start_role.value}; "
                "select that role or another episode"
            )
        self.output_root = output_root
        self.created_at = _now()
        self.updated_at = self.created_at
        if self.mode == "benchmark":
            assert self.episode is not None
            system_text = (
                f"Loaded frozen task {self.episode.episode_id}: "
                f"{self.episode.primary_family.value}. This private rehearsal permits "
                "one sender message and one receiver feedback; it is excluded from formal results."
            )
        elif self.episode is not None:
            system_text = (
                f"Started a multi-turn interactive session seeded by {self.episode.episode_id}. "
                "The frozen case metadata is context, not a restriction on later wording or turns."
            )
        else:
            system_text = (
                "Started a multi-turn interactive session. Send natural-language requests from "
                "either role; unresolved execution-critical fields will be surfaced explicitly."
            )
        self.messages = [
            ChatMessage(
                message_id=uuid4().hex,
                speaker="system",
                text=system_text,
            )
        ]
        self.results: list[PipelineResult] = []
        self._persist()

    @property
    def latest_result(self) -> PipelineResult | None:
        return self.results[-1] if self.results else None

    def handle_message(
        self,
        text: str,
        *,
        sender_role: AgentRole | None = None,
    ) -> PipelineResult:
        if self.mode == "benchmark" and self.results:
            raise ValueError(
                "Benchmark mode permits exactly one sender message; use interactive mode for multiple turns"
            )
        if (
            self.latest_result is not None
            and self.latest_result.information_exchange.decision == "pending_physics"
            and self.latest_result.workflow_status != "blocked"
        ):
            raise ValueError(
                "The previous turn is awaiting receiver feedback; complete it before starting another turn"
            )
        stripped = text.strip()
        if not stripped:
            raise ValueError("Message cannot be empty")
        actual_sender = sender_role or self.user_role
        if (
            self.mode == "benchmark"
            and self.episode is not None
            and actual_sender != self.episode.start_role
        ):
            raise ValueError("The benchmark sender role is fixed by the selected episode")

        turn_index = len(self.results) + 1
        self.messages.append(
            ChatMessage(
                message_id=uuid4().hex,
                speaker="sender",
                role=actual_sender,
                text=stripped,
            )
        )
        self._persist()  # Retain the sender input even if a later API call fails.
        if self.mode == "benchmark":
            assert self.episode is not None
            result = self.pipeline.run_contract_stage(
                stripped,
                episode=self.episode,
                sender=actual_sender,
            )
        else:
            context = self._interactive_context(turn_index)
            result = self.pipeline.run_message(
                stripped,
                sender=actual_sender,
                context=context,
                prior_specification=(
                    self.latest_result.task_specification
                    if self.latest_result is not None
                    else None
                ),
                conversation_history=[
                    {"speaker": message.speaker, "text": message.text}
                    for message in self.messages[:-1]
                ],
                turn_index=turn_index,
            )
        self.results.append(result)
        self._append_exchange_messages(result)
        self.updated_at = _now()
        self._persist()
        return result

    def handle_feedback(
        self,
        feedback: ReceiverFeedback,
        *,
        turn_index: int | None = None,
    ) -> PipelineResult:
        if not self.results:
            raise ValueError("No sender turn is available for receiver feedback")
        index = len(self.results) - 1 if turn_index is None else turn_index - 1
        if index < 0 or index >= len(self.results):
            raise ValueError(f"Unknown turn index: {turn_index}")
        result = self.results[index]
        if result.information_exchange.decision != "pending_physics":
            raise ValueError("That turn is already completed or requires sender clarification")
        updated_result = self.pipeline.complete_with_receiver_feedback(
            result,
            feedback=feedback,
            conversation_history=[
                {"speaker": message.speaker, "text": message.text}
                for message in self.messages
            ],
        )
        updated_exchange = updated_result.information_exchange
        self.results[index] = updated_result
        if updated_result.workflow_status in {"released", "needs_clarification"}:
            self._append_released_completion(updated_result)
        else:
            self.messages.append(
                ChatMessage(
                    message_id=uuid4().hex,
                    speaker="system",
                    text=(
                        "Reviewer stored the receiver declaration as a proposal but withheld it "
                        "because physical or semantic validation remains unresolved."
                    ),
                )
            )
        self.updated_at = _now()
        self._persist()
        return updated_result

    def execute_physics(self, *, turn_index: int | None = None) -> PipelineResult:
        """Execute the Actor plan and close one pending exchange with traced evidence."""

        if not self.results:
            raise ValueError("No sender turn is available for physics execution")
        index = len(self.results) - 1 if turn_index is None else turn_index - 1
        if index < 0 or index >= len(self.results):
            raise ValueError(f"Unknown turn index: {turn_index}")
        result = self.results[index]
        if result.information_exchange.decision != "pending_physics":
            raise ValueError("That turn is already completed or requires sender clarification")
        updated_result = self.pipeline.execute_physics(
            result,
            run_root=self.output_root / self.session_id / "physics" / f"t{index + 1:03d}",
        )
        self.results[index] = updated_result
        if updated_result.workflow_status in {"released", "needs_clarification"}:
            self._append_released_completion(updated_result)
        else:
            self.messages.append(
                ChatMessage(
                    message_id=uuid4().hex,
                    speaker="system",
                    text="Reviewer blocked the simulated result; no external completion message was released.",
                )
            )
        self.updated_at = _now()
        self._persist()
        return updated_result

    def snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=self.session_id,
            mode=self.mode,
            user_role=self.user_role,
            episode=self.episode,
            messages=self.messages,
            results=self.results,
            latest_result=self.latest_result,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def finalize(self) -> FinalReview:
        if self.latest_result is None:
            raise ValueError("Run at least one interaction before final review")
        report = _report(self.snapshot(), self.pipeline.case.case_id)
        report_path = self.output_root / self.session_id / "final-review.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
        return FinalReview(
            session_id=self.session_id,
            case_id=self.pipeline.case.case_id,
            episode_id=self.episode.episode_id if self.episode else None,
            turn_count=len(self.results),
            verdict=self.latest_result.reviewer_decision.verdict,
            report_markdown=report,
            trace_id=self.latest_result.trace_id,
        )

    def _interactive_context(self, turn_index: int) -> InteractionContext:
        episode = self.episode
        return InteractionContext(
            interaction_id=f"{self.session_id}-turn-{turn_index:03d}",
            case_id=self.pipeline.case.case_id,
            origin="interactive_seeded" if episode else "interactive_inferred",
            task_families=episode.task_families if episode else [],
            operating_day=episode.operating_day if episode else None,
            windows=episode.windows if episode else [],
            target_bems=episode.target_bems if episode else [],
            target_buses=episode.target_buses if episode else [],
            communication_hint=None,
        )

    def _append_exchange_messages(self, result: PipelineResult) -> None:
        exchange = result.information_exchange
        if exchange.decision == "pending_physics":
            contract_released = (
                result.reviewer_decision.routing_decision == "actor"
                and result.execution_plan is not None
                and result.execution_plan.valid
                and not any(
                    issue.responsible_agent == "translator"
                    for issue in result.reviewer_decision.issues
                )
            )
            if contract_released:
                self.messages.append(
                    ChatMessage(
                        message_id=uuid4().hex,
                        speaker="agent_to_receiver",
                        role=exchange.receiver_role,
                        text=exchange.receiver_received,
                    )
                )
            self.messages.append(
                ChatMessage(
                    message_id=uuid4().hex,
                    speaker="system",
                    text=(
                        (
                            "The translated request and Actor plan passed the contract gate. The "
                            "completion response remains withheld until physical evidence passes "
                            "the Reviewer's release gate."
                        )
                        if contract_released
                        else (
                            "The receiver-facing draft is withheld because its semantic or plan "
                            "review has not passed."
                        )
                    ),
                )
            )
            return
        self.messages.append(
            ChatMessage(
                message_id=uuid4().hex,
                speaker="agent_to_receiver",
                role=exchange.receiver_role,
                text=exchange.receiver_received,
            )
        )
        if exchange.receiver_sent is not None:
            self.messages.append(
                ChatMessage(
                    message_id=uuid4().hex,
                    speaker="receiver",
                    role=exchange.receiver_role,
                    text=exchange.receiver_sent,
                )
            )
        if exchange.sender_received is not None:
            self.messages.append(
                ChatMessage(
                    message_id=uuid4().hex,
                    speaker="agent_to_sender",
                    role=exchange.sender_role,
                    text=exchange.sender_received,
                )
            )

    def _append_released_completion(self, result: PipelineResult) -> None:
        exchange = result.information_exchange
        self.messages.extend(
            [
                ChatMessage(
                    message_id=uuid4().hex,
                    speaker="receiver",
                    role=exchange.receiver_role,
                    text=exchange.receiver_sent or "",
                ),
                ChatMessage(
                    message_id=uuid4().hex,
                    speaker="agent_to_sender",
                    role=exchange.sender_role,
                    text=exchange.sender_received or "",
                ),
            ]
        )

    def _persist(self) -> None:
        path = self.output_root / self.session_id / "session.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        durable_write(path,
            json.dumps(
                self.snapshot().model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
        )


class SessionStore:
    def __init__(
        self,
        *,
        pipeline: B2GPipeline | None = None,
        output_root: Path | None = None,
        repo_root: Path | None = None,
    ) -> None:
        root = repo_root or find_repo_root()
        self.workspace = PersonalWorkspace.open(root) if pipeline is None else None
        self.pipeline = pipeline if pipeline is not None else self.workspace.pipeline(root)
        self.output_root = (output_root or (self.workspace.root / "sessions" if self.workspace else root / ".personal" / "sessions")).resolve()
        if self.workspace and not self.output_root.is_relative_to(self.workspace.root):
            raise ValueError("Personal session output must remain inside its private workspace")
        self._sessions: dict[str, CollaborationSession] = {}
        self._lock = RLock()

    @property
    def case(self) -> CaseBundle:
        return self.pipeline.case

    def create(
        self,
        role: AgentRole,
        episode_id: str | None = None,
        *,
        mode: SessionMode = "interactive",
    ) -> CollaborationSession:
        with self._lock:
            session_id = uuid4().hex[:12]
            session = CollaborationSession(
                session_id=session_id,
                user_role=role,
                episode_id=episode_id,
                mode=mode,
                pipeline=self.pipeline,
                output_root=self.output_root,
            )
            self._sessions[session_id] = session
            return session

    def get(self, session_id: str) -> CollaborationSession:
        try:
            return self._sessions[session_id]
        except KeyError as exc:
            raise KeyError(f"Unknown session: {session_id}") from exc


def _report(snapshot: SessionSnapshot, case_id: str) -> str:
    result = snapshot.latest_result
    assert result is not None
    episode_id = snapshot.episode.episode_id if snapshot.episode else "none (live input)"
    sections = "\n\n".join(_turn_report(turn) for turn in snapshot.results)
    return f"""# B2G-Agent final review

- Session: `{snapshot.session_id}`
- Mode: `{snapshot.mode}`
- Episode seed: `{episode_id}`
- Case: `{case_id}`
- Initial role: `{snapshot.user_role.value}`
- Turn count: **{len(snapshot.results)}**
- Latest reviewer verdict: **{result.reviewer_decision.verdict.value}**
- Latest trace: `{result.trace_id}`

{sections}

## Reproducibility note

Turns with a `physics_evidence` record were executed through the joint RC/OPF,
EnergyPlus, and nonlinear pandapower loop; the evidence package and all frozen
input hashes are retained with the session. Other turns make no physical-success
claim. LLM live requests, when enabled, are stored by content hash. Private
benchmark ground truth is not included in session artifacts or API responses.
"""


def _turn_report(result: PipelineResult) -> str:
    issues = [
        f"{issue.code}: {issue.message}"
        for issue in result.reviewer_decision.issues
    ]
    issue_lines = "\n".join(f"- {issue}" for issue in issues) or "- None"
    plan_summary = (
        f"{len(result.execution_plan.actions)} actions; valid={result.execution_plan.valid}"
        if result.execution_plan
        else "Blocked by clarification gate"
    )
    exchange = result.information_exchange
    receiver_sent = (
        exchange.receiver_sent
        or "Pending physical evaluation; no feedback was invented."
    )
    sender_received = exchange.sender_received or "Pending receiver feedback."
    completion = (
        f"{exchange.completion_percentage:g}%"
        if exchange.completion_percentage is not None
        else "pending"
    )
    physics = (
        f"{result.physics_evidence.status}; `{result.physics_evidence.evidence_sha256}`"
        if result.physics_evidence is not None
        else "not executed"
    )
    return f"""## Turn {result.turn_index}

- Interaction: `{result.episode_id}`
- Reviewer verdict: **{result.reviewer_decision.verdict.value}**
- Actor plan: {plan_summary}
- Exchange decision: **{exchange.decision}**
- Completion: **{completion}**
- Physics evidence: {physics}

### Four information states

1. Sender sent: {exchange.sender_sent}
2. Receiver received (agent-translated): {exchange.receiver_received}
3. Receiver sent: {receiver_sent}
4. Sender received (agent-translated): {sender_received}

### Reviewer issues

{issue_lines}"""
