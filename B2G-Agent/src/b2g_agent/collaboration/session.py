from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from b2g_agent.collaboration.mediator import B2GMediator
from b2g_agent.collaboration.models import (
    ChatMessage,
    EngineerRole,
    FinalPlan,
    MediatedReply,
    MediatorPlan,
    ResearchScenario,
    ScenarioParameters,
    SessionSnapshot,
    SimulationRun,
    SimulationScope,
    TurnResult,
)
from b2g_agent.collaboration.scenario import (
    counterpart_role,
    default_parameters,
    normalize_scenario_id,
    role_label,
    scenario_title,
    validate_parameter_changes,
)
from b2g_agent.collaboration.simulator import ResidentialCommunitySimulator


def _now() -> datetime:
    return datetime.now(timezone.utc)


class CollaborationSession:
    def __init__(
        self,
        *,
        user_role: EngineerRole,
        scenario_id: ResearchScenario | str = ResearchScenario.GRID_UPGRADE,
        mediator: B2GMediator | None = None,
        simulator: ResidentialCommunitySimulator | None = None,
        session_id: str | None = None,
        output_root: Path | None = None,
    ) -> None:
        self.session_id = session_id or uuid4().hex[:12]
        self.scenario_id = normalize_scenario_id(scenario_id)
        self.user_role = user_role
        self.counterpart_role = counterpart_role(user_role)
        self.parameters = default_parameters(self.scenario_id)
        self.mediator = mediator or B2GMediator()
        self.simulator = simulator or ResidentialCommunitySimulator()
        self.created_at = _now()
        self.updated_at = self.created_at
        self.messages: list[ChatMessage] = []
        self.runs: list[SimulationRun] = []
        self.decision_ledger: list[str] = []
        self.pending_changes: dict[str, object] | None = None
        repo_root = Path(__file__).resolve().parents[3]
        self.output_dir = (output_root or repo_root / "outputs" / "web_sessions") / self.session_id

        baseline = self.simulator.run(
            self.parameters,
            trigger="Scenario baseline",
            label="Baseline",
            scenario_id=self.scenario_id,
        )
        self.runs.append(baseline)
        self._append_message(
            "mediator",
            (
                f"Welcome to the {scenario_title(self.scenario_id)} session. "
                f"You are the {role_label(self.user_role)}. "
                f"I will mediate with the AI {role_label(self.counterpart_role)}, translate both domains, "
                "and ground proposed decisions in explicit simulation evidence."
            ),
        )
        self._append_message(
            "npc",
            self._counterpart_intro(),
            role=self.counterpart_role,
        )
        self._persist()

    def handle_message(self, text: str) -> TurnResult:
        text = " ".join(text.strip().split())
        if not text:
            raise ValueError("Message cannot be empty.")
        if len(text) > 4000:
            raise ValueError("Message cannot exceed 4000 characters.")

        self._append_message("user", text, role=self.user_role)
        confirmed_plan = self._confirmed_pending_plan(text)
        if confirmed_plan is not None:
            plan = confirmed_plan
        else:
            plan = self.mediator.plan_turn(
                text=text,
                scenario_id=self.scenario_id,
                user_role=self.user_role,
                counterpart_role=self.counterpart_role,
                parameters=self.parameters,
                recent_messages=[
                    {"speaker": item.speaker, "role": item.role, "text": item.text}
                    for item in self.messages[-8:]
                ],
            )

        if plan.needs_confirmation and plan.requested_changes and confirmed_plan is None:
            self.pending_changes = dict(plan.requested_changes)
            mediator_text = (
                f"I interpreted your proposal as: {plan.interpreted_intent} "
                f"The material changes are {json.dumps(plan.requested_changes, ensure_ascii=False)}. "
                "Please reply 'confirm' to apply them and run the requested simulation, or restate the proposal."
            )
            messages = [
                self._append_message("mediator", mediator_text),
                self._append_message(
                    "npc",
                    "I will wait for confirmation before treating those model changes as an engineering proposal.",
                    role=self.counterpart_role,
                ),
            ]
            self.updated_at = _now()
            self._persist()
            return TurnResult(
                plan=plan,
                messages=messages,
                parameters=self.parameters,
                decision_ledger=self.decision_ledger,
                llm_backend=self.mediator.last_backend,
            )

        run: SimulationRun | None = None
        if plan.requested_changes:
            try:
                self.parameters = validate_parameter_changes(
                    self.parameters,
                    plan.requested_changes,
                    self.scenario_id,
                )
            except (TypeError, ValueError) as exc:
                error_message = self._append_message(
                    "mediator",
                    f"I could not apply that interpretation safely: {exc}. Please revise the parameter request.",
                )
                self.updated_at = _now()
                self._persist()
                return TurnResult(
                    plan=plan,
                    messages=[error_message],
                    parameters=self.parameters,
                    decision_ledger=self.decision_ledger,
                    llm_backend=self.mediator.last_backend,
                )

        if plan.simulation_scope != SimulationScope.NONE:
            run = self.simulator.run(
                self.parameters,
                trigger=plan.interpreted_intent,
                scope=plan.simulation_scope,
                scenario_id=self.scenario_id,
            )
            self.runs.append(run)

        reply = self.mediator.compose_reply(
            original_text=text,
            scenario_id=self.scenario_id,
            user_role=self.user_role,
            counterpart_role=self.counterpart_role,
            parameters=self.parameters,
            plan=plan,
            run=run,
        )
        messages = self._append_reply(reply)
        if reply.decision_note:
            self.decision_ledger.append(reply.decision_note)
        self.updated_at = _now()
        self._persist()
        return TurnResult(
            plan=plan,
            messages=messages,
            parameters=self.parameters,
            simulation_run=run,
            decision_ledger=self.decision_ledger,
            llm_backend=self.mediator.last_backend,
        )

    def simulate_current(self, trigger: str = "User requested a coupled rerun") -> SimulationRun:
        run = self.simulator.run(
            self.parameters,
            trigger=trigger,
            scope=SimulationScope.COUPLED,
            scenario_id=self.scenario_id,
        )
        self.runs.append(run)
        self.decision_ledger.append(
            f"Run {run.run_id}: {'feasible' if run.metrics.feasible else 'needs revision'} manual coupled rerun."
        )
        self.updated_at = _now()
        self._persist()
        return run

    def finalize(self) -> FinalPlan:
        feasible_runs = [run for run in self.runs if run.metrics.feasible]
        candidates = feasible_runs or self.runs
        selected = max(candidates, key=_run_value)
        status = "ready" if selected.metrics.feasible else "needs_revision"
        accepted_decisions = _accepted_decisions(self.scenario_id, selected)
        unresolved = _unresolved_items(self.scenario_id, selected)
        if selected.metrics.estimated_capital_cost_kusd > 2500:
            unresolved.append("Capital cost exceeds the scenario's indicative review threshold of $2.5M.")
        summary = (
            "A jointly feasible candidate is ready for professional review."
            if status == "ready"
            else "The best available candidate still requires engineering revision before approval."
        )
        report = _build_report(self, selected, status, accepted_decisions, unresolved)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "final_plan.md").write_text(report, encoding="utf-8")
        self.decision_ledger.append(f"Final review selected run {selected.run_id} with status={status}.")
        self.updated_at = _now()
        self._persist()
        return FinalPlan(
            selected_run=selected,
            status=status,
            summary=summary,
            accepted_decisions=accepted_decisions,
            unresolved_items=unresolved,
            report_markdown=report,
        )

    def snapshot(self) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=self.session_id,
            user_role=self.user_role,
            counterpart_role=self.counterpart_role,
            scenario_id=self.scenario_id,
            parameters=self.parameters,
            messages=self.messages,
            runs=self.runs,
            decision_ledger=self.decision_ledger,
            created_at=self.created_at,
            updated_at=self.updated_at,
            llm_backend=self.mediator.last_backend,
        )

    def _confirmed_pending_plan(self, text: str) -> MediatorPlan | None:
        if not self.pending_changes:
            return None
        if not re.fullmatch(r"(?i)(confirm|confirmed|yes|proceed|确认|同意|执行)[.!。 ]*", text):
            self.pending_changes = None
            return None
        changes = self.pending_changes
        self.pending_changes = None
        return MediatorPlan(
            intent="simulate",
            interpreted_intent="The user confirmed the pending material model changes.",
            translation_for_counterpart="The previously interpreted change is confirmed and ready for evidence generation.",
            requested_changes=changes,
            simulation_scope=SimulationScope.COUPLED,
            needs_confirmation=False,
            confidence=1.0,
        )

    def _append_reply(self, reply: MediatedReply) -> list[ChatMessage]:
        mediator_text = reply.mediator_message
        if reply.next_question and reply.next_question not in mediator_text:
            mediator_text = f"{mediator_text}\n\nNext question: {reply.next_question}"
        return [
            self._append_message("mediator", mediator_text),
            self._append_message("npc", reply.counterpart_message, role=self.counterpart_role),
        ]

    def _append_message(
        self,
        speaker: str,
        text: str,
        *,
        role: EngineerRole | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            message_id=uuid4().hex[:10],
            speaker=speaker,
            role=role,
            text=text,
        )
        self.messages.append(message)
        return message

    def _counterpart_intro(self) -> str:
        if self.scenario_id == ResearchScenario.DEMAND_RESPONSE:
            if self.counterpart_role == EngineerRole.POWER:
                return (
                    "I am the AI Distribution Power Engineer. I will challenge the baseline, verify delivered "
                    "event reduction, and protect the feeder from unreliable commitments or rebound."
                )
            return (
                "I am the AI Building Engineer. I will explain normal building operation, defend a credible "
                "baseline, and protect comfort while identifying controllable load."
            )
        if self.counterpart_role == EngineerRole.POWER:
            return (
                "I am the AI Distribution Power Engineer. I will protect voltage and equipment margins, "
                "but I am open to targeted upgrades when building-side flexibility is insufficient."
            )
        return (
            "I am the AI Building Engineer. I will protect occupant comfort and practical retrofit scope, "
            "while considering flexibility that avoids unnecessary grid expansion."
        )

    def _persist(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        payload = self.snapshot().model_dump(mode="json")
        (self.output_dir / "session.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


class SessionStore:
    def __init__(
        self,
        *,
        mediator_factory: type[B2GMediator] = B2GMediator,
        output_root: Path | None = None,
    ) -> None:
        self._sessions: dict[str, CollaborationSession] = {}
        self._lock = RLock()
        self._mediator_factory = mediator_factory
        self._output_root = output_root

    def create(
        self,
        role: EngineerRole,
        scenario_id: ResearchScenario | str = ResearchScenario.GRID_UPGRADE,
    ) -> CollaborationSession:
        mediator = self._mediator_factory()
        require_llm = os.getenv("B2G_REQUIRE_LLM", "true").strip().lower() not in {
            "0",
            "false",
            "no",
        }
        if require_llm and not mediator.configured:
            raise RuntimeError(
                "A personal OpenAI API key is required. Set OPENAI_API_KEY and "
                "B2G_LLM_ENABLED=true in .env, then restart b2g-web."
            )
        session = CollaborationSession(
            user_role=role,
            scenario_id=scenario_id,
            mediator=mediator,
            output_root=self._output_root,
        )
        with self._lock:
            self._sessions[session.session_id] = session
        return session

    def get(self, session_id: str) -> CollaborationSession:
        with self._lock:
            try:
                return self._sessions[session_id]
            except KeyError as exc:
                raise KeyError(f"Unknown session: {session_id}") from exc


def _run_value(run: SimulationRun) -> float:
    metrics = run.metrics
    feasibility_bonus = 100.0 if metrics.feasible else 0.0
    return (
        feasibility_bonus
        + metrics.building_satisfaction_score
        + metrics.grid_reliability_score
        - metrics.estimated_capital_cost_kusd / 100.0
    )


def _accepted_decisions(
    scenario_id: ResearchScenario,
    selected: SimulationRun,
) -> list[str]:
    parameters = selected.parameters
    if scenario_id == ResearchScenario.DEMAND_RESPONSE:
        return [
            f"Baseline method: {parameters.baseline_method.value}",
            f"Baseline adjustment: {parameters.baseline_adjustment_pct:+.1f}%",
            f"Participating assets: {parameters.building_count}",
            (
                f"Event window: {parameters.dr_event_start_hour:02d}:00 for "
                f"{parameters.dr_event_duration_hours} hour(s)"
            ),
            f"Committed reduction: {parameters.dr_target_kw_per_building:.2f} kW per building",
            f"Maximum rebound: {parameters.max_rebound_pct:.1f}% of target",
            f"Grid connection: {parameters.target_bus}",
        ]
    return [
        f"Cooling setpoint: {parameters.cooling_setpoint_c:.1f} °C",
        f"Building program: {parameters.building_count} homes, {parameters.retrofit_level.value} retrofit",
        f"PV: {parameters.pv_kw_per_building:.1f} kW per building",
        f"Demand response: {parameters.demand_response_pct:.1f}%",
        f"Grid connection: {parameters.target_bus}",
        (
            f"Transformer / line capacity: {parameters.transformer_capacity_kva:.0f} kVA / "
            f"{parameters.line_capacity_kw:.0f} kW"
        ),
    ]


def _unresolved_items(
    scenario_id: ResearchScenario,
    selected: SimulationRun,
) -> list[str]:
    metrics = selected.metrics
    unresolved: list[str] = []
    if scenario_id == ResearchScenario.DEMAND_RESPONSE:
        if metrics.baseline_confidence_score < 80.0:
            unresolved.append("Baseline confidence is below the 80/100 enrollment threshold.")
        if metrics.dr_delivery_pct < 90.0:
            unresolved.append("Delivered reduction is below 90% of the committed target.")
        if metrics.rebound_pct > selected.parameters.max_rebound_pct:
            unresolved.append("Post-event rebound exceeds the agreed limit.")
        if metrics.voltage_violation_hours or metrics.line_overload_hours or metrics.transformer_overload_hours:
            unresolved.append("At least one grid operating constraint remains violated.")
    elif not metrics.feasible:
        unresolved.append("At least one voltage or thermal hard constraint remains violated.")
    return unresolved


def _build_report(
    session: CollaborationSession,
    selected: SimulationRun,
    status: str,
    accepted_decisions: list[str],
    unresolved: list[str],
) -> str:
    metrics = selected.metrics
    lines = [
        f"# B2G-Agent Final Plan: {scenario_title(session.scenario_id)}",
        "",
        f"- Session: `{session.session_id}`",
        f"- Selected run: `{selected.run_id}`",
        f"- Review status: **{status}**",
        f"- Human role: {role_label(session.user_role)}",
        f"- AI counterpart: {role_label(session.counterpart_role)}",
        f"- Research scenario: {session.scenario_id.value}",
        f"- Simulation backend: `{selected.backend}`",
        "",
        "## Agreed Scenario",
        "",
        *[f"- {item}" for item in accepted_decisions],
        "",
        "## Joint Evidence",
        "",
        *_report_evidence(session.scenario_id, selected),
        "",
        "## Unresolved Items",
        "",
        *([f"- {item}" for item in unresolved] or ["- None recorded in this vertical scenario."]),
        "",
        "## Provenance And Limitation",
        "",
        (
            "This result was produced by a deterministic B2G-Agent research backend. It demonstrates the "
            "mediation and evidence workflow, but it is not a calibrated EnergyPlus/OpenDSS study or a "
            "settlement-grade demand-response baseline. EnergyPlus-MCP and PowerMCP are cited integration "
            "targets and are not called by this release."
        ),
        "",
    ]
    return "\n".join(lines)


def _report_evidence(
    scenario_id: ResearchScenario,
    selected: SimulationRun,
) -> list[str]:
    metrics = selected.metrics
    common = [
        f"- Feeder peak: {metrics.feeder_peak_kw:.1f} kW",
        f"- Minimum voltage: {metrics.minimum_voltage_pu:.3f} p.u.",
        f"- Maximum line loading: {metrics.maximum_line_loading_pct:.1f}%",
        f"- Maximum transformer loading: {metrics.maximum_transformer_loading_pct:.1f}%",
        f"- Building satisfaction score: {metrics.building_satisfaction_score:.1f}/100",
        f"- Grid reliability score: {metrics.grid_reliability_score:.1f}/100",
    ]
    if scenario_id == ResearchScenario.DEMAND_RESPONSE:
        return [
            f"- Baseline peak: {metrics.baseline_peak_kw_per_building:.2f} kW per building",
            f"- Baseline confidence: {metrics.baseline_confidence_score:.1f}/100",
            f"- Delivered reduction: {metrics.delivered_reduction_kw_per_building:.2f} kW per building",
            f"- Delivery performance: {metrics.dr_delivery_pct:.1f}% of commitment",
            f"- Post-event rebound: {metrics.rebound_pct:.1f}% of target",
            *common,
        ]
    return [
        *common,
        f"- Estimated capital cost: ${metrics.estimated_capital_cost_kusd:.1f}k",
    ]
