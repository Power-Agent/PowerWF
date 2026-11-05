from __future__ import annotations
from b2g_agent.observability.journal import emit, traced_stage

import re
from pathlib import Path
from time import perf_counter

from pydantic import BaseModel, ConfigDict, Field

from b2g_agent.agents.llm import StructuredLLMClient
from b2g_agent.agents.retrieval import DeterministicFusionRetriever
from b2g_agent.case_io import CaseBundle
from b2g_agent.case_io.operating_policy import operating_constraints
from b2g_agent.contracts import (
    AgentRole,
    Episode,
    EvidenceRef,
    FieldStatus,
    InteractionContext,
    Objective,
    ReceiverFeedback,
    ReverseTransferState,
    TaskFamily,
    TaskField,
    TaskSpecification,
    TimeWindow,
)


class TranslatedFeedback(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sender_message: str = Field(min_length=1)


FAMILY_OUTPUTS: dict[TaskFamily, list[str]] = {
    TaskFamily.PEAK_SHAVING: ["feeder_peak_kw", "delivered_reduction_kw", "comfort_violation_minutes"],
    TaskFamily.CONGESTION_MANAGEMENT: ["maximum_line_loading_pct", "overload_intervals"],
    TaskFamily.VOLTAGE_MANAGEMENT: ["minimum_voltage_pu", "maximum_voltage_pu", "violation_intervals"],
    TaskFamily.RESERVE_SERVICE: ["reserved_capacity_kw", "delivery_ratio", "rebound_kw"],
    TaskFamily.RENEWABLE_ABSORPTION: ["pv_curtailment_kwh", "absorbed_energy_kwh"],
    TaskFamily.BASELINE_FLEXIBILITY: ["counterfactual_baseline_kw", "flexibility_lower_kw", "flexibility_upper_kw"],
    TaskFamily.MARKET_TRADING: ["accepted_capacity_kw", "settlement_usd", "delivery_ratio"],
    TaskFamily.GENERAL_COORDINATION: [],
}

FAMILY_ALIASES: dict[TaskFamily, tuple[str, ...]] = {
    TaskFamily.PEAK_SHAVING: ("peak shaving", "peak reduction", "\u524a\u5cf0", "\u5cf0\u503c\u964d\u4f4e"),
    TaskFamily.CONGESTION_MANAGEMENT: ("congestion", "line overload", "\u963b\u585e", "\u7ebf\u8def\u8fc7\u8f7d"),
    TaskFamily.VOLTAGE_MANAGEMENT: ("voltage", "\u7535\u538b"),
    TaskFamily.RESERVE_SERVICE: ("reserve", "\u5907\u7528"),
    TaskFamily.RENEWABLE_ABSORPTION: ("renewable absorption", "pv absorption", "curtailment", "\u5149\u4f0f\u6d88\u7eb3", "\u5f03\u5149"),
    TaskFamily.BASELINE_FLEXIBILITY: ("baseline flexibility", "baseline", "flexibility envelope", "\u57fa\u7ebf", "\u7075\u6d3b\u6027"),
    TaskFamily.MARKET_TRADING: ("market trading", "offer price", "settlement", "\u5e02\u573a\u4ea4\u6613", "\u62a5\u4ef7", "\u7ed3\u7b97"),
}


class TranslatorAgent:
    """Role-conditioned translation for both benchmark and live messages."""

    def __init__(
        self,
        knowledge_root: Path,
        *,
        llm: StructuredLLMClient | None = None,
        retrieval_enabled: bool = True,
        case: CaseBundle | None = None,
    ) -> None:
        self.retriever = DeterministicFusionRetriever(knowledge_root)
        self.llm = llm
        self.retrieval_enabled = retrieval_enabled
        self.case = case or (CaseBundle.default(knowledge_root.parent)
            if (knowledge_root.parent / "cases/ieee33_10bem_15min/case.json").is_file() else None)

    @traced_stage("translator.translate")
    def translate(
        self,
        text: str,
        *,
        sender: AgentRole,
        episode: Episode | None = None,
        context: InteractionContext | None = None,
        prior_specification: TaskSpecification | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        turn_index: int = 1,
    ) -> TaskSpecification:
        if episode is not None and context is not None:
            raise ValueError("Provide episode or context, not both")
        if episode is not None:
            # Benchmark evaluation must begin from m^(1), not from hidden
            # generator metadata.  Preserve only public identity fields; all
            # semantic content must be recovered from the sender's message.
            context = InteractionContext(
                interaction_id=episode.episode_id,
                case_id=episode.case_id,
                origin="benchmark_episode",
            )
        if context is None:
            raise ValueError("Live translation requires an InteractionContext")

        receiver = _counterpart(sender)
        inferred_families = _infer_families(text)
        families = (
            inferred_families
            or context.task_families
            or (prior_specification.task_families if prior_specification else [])
            or [TaskFamily.GENERAL_COORDINATION]
        )
        prior_bems = prior_specification.target_bems if prior_specification else []
        prior_buses = prior_specification.target_buses if prior_specification else []
        prior_windows = prior_specification.windows if prior_specification else []
        target_bems = _parse_bems(text) or context.target_bems or prior_bems
        target_buses = _parse_buses(text) or context.target_buses or prior_buses
        event_windows = _parse_windows(text) or context.windows or prior_windows
        operating_day = (
            _parse_operating_day(text)
            or context.operating_day
            or (prior_specification.operating_day if prior_specification else None)
        )

        retrieval_started = perf_counter()
        evidence = (
            self.retriever.retrieve_for_transfer(
                f"{sender.value} {receiver.value} {' '.join(f.value for f in families)} {text}",
                direction=f"{sender.value}->{receiver.value}",
                business_types=(family.value for family in families),
                time_scales=("interval",),
                spatial_levels=(
                    "portfolio" if len(target_bems) > 1 else "building",
                    "bus",
                ),
                roles=(sender.value, receiver.value),
                per_partition=2,
            )
            if self.retrieval_enabled
            else []
        )
        retrieval_elapsed = perf_counter() - retrieval_started
        emit("retrieval.completed", agent="translator", direction="forward",
             elapsed_seconds=retrieval_elapsed, evidence_count=len(evidence))
        target_kw, target_unit, target_status = _parse_power_target(text)
        if target_kw is None and prior_specification is not None:
            prior_target = prior_specification.fields.get("target_power")
            if prior_target and isinstance(prior_target.value, (int, float)):
                if _declares_kw_unit(text):
                    target_kw = float(prior_target.value)
                    target_unit = "kW"
                    target_status = FieldStatus.RESOLVED
                elif prior_target.status in {FieldStatus.DECLARED, FieldStatus.RESOLVED}:
                    target_kw = float(prior_target.value)
                    target_unit = prior_target.unit
                    target_status = prior_target.status
        price = _first_number_before_unit(text, "usd")
        voltage_band = _parse_voltage_band(text)
        voltage = None if voltage_band is not None else _first_number_before_unit(text, "pu")
        conflict = (
            context.communication_hint == "conflicting_constraints"
            or _contains_priority_conflict(text)
        )

        missing: list[str] = []
        if families == [TaskFamily.GENERAL_COORDINATION]:
            missing.append("task_family")
        if not operating_day:
            missing.append("operating_day")
        if not event_windows:
            missing.append("windows")
        if not target_bems:
            missing.append("target_bems")
        if not target_buses:
            missing.append("target_buses")
        if (target_kw is None or target_unit != "kW") and _requires_power_target(families):
            missing.append("target_power")
        if conflict:
            missing.append("objective_priority")

        fields: dict[str, TaskField] = {
            "target_power": TaskField(
                value=target_kw,
                unit=target_unit,
                status=target_status,
                confidence=(
                    1.0
                    if target_status in {FieldStatus.DECLARED, FieldStatus.RESOLVED}
                    else 0.0
                ),
                evidence=evidence[:1] if target_kw is not None else [],
                note=(
                    "A numeric quantity was stated without a power unit."
                    if target_kw is not None and target_unit is None
                    else None
                ),
            ),
            "time_resolution": TaskField(
                value=15,
                unit="min",
                status=FieldStatus.INFERRED,
                confidence=1.0,
                evidence=evidence[:1],
                note="Fixed by the selected formal case; a different case may override it.",
            ),
            "operating_day": _context_field(operating_day, evidence),
            "windows": _context_field(
                [window.model_dump(mode="json") for window in event_windows] or None,
                evidence,
            ),
            "target_bems": _context_field(target_bems or None, evidence),
            "target_buses": _context_field(target_buses or None, evidence),
        }
        if conflict:
            fields["priority_conflict"] = TaskField(
                value=True,
                status=FieldStatus.MISSING,
                confidence=1.0,
                evidence=evidence[:1],
                note="Incompatible highest-priority requirements require human clarification.",
            )
        if price is not None:
            fields["offer_price"] = TaskField(
                value=price,
                unit="USD/MWh",
                status=FieldStatus.DECLARED,
                confidence=1.0,
                evidence=evidence[:1],
            )
        if voltage_band is not None:
            fields["voltage_band"] = TaskField(
                value=list(voltage_band),
                unit="p.u.",
                status=FieldStatus.DECLARED,
                confidence=1.0,
                evidence=evidence[:1],
            )
        elif voltage is not None:
            fields["voltage_threshold"] = TaskField(
                value=voltage,
                unit="p.u.",
                status=FieldStatus.DECLARED,
                confidence=1.0,
                evidence=evidence[:1],
            )

        requested_outputs = sorted(
            {
                output
                for family in families
                if family not in {TaskFamily.COMPOSITE, TaskFamily.GENERAL_COORDINATION}
                for output in FAMILY_OUTPUTS[family]
            }
        )
        for output in requested_outputs:
            fields.setdefault(
                output,
                TaskField(
                    status=FieldStatus.TO_BE_SIMULATED,
                    confidence=1.0,
                    evidence=evidence[:1],
                ),
            )

        question = None
        if missing:
            question = (
                "Please provide or resolve the following execution-critical information: "
                + ", ".join(sorted(set(missing)))
                + "."
            )
        draft = TaskSpecification(
            episode_id=context.interaction_id,
            case_id=context.case_id,
            sender_role=sender,
            receiver_role=receiver,
            task_families=families,
            operating_day=operating_day,
            windows=event_windows,
            target_bems=target_bems,
            target_buses=target_buses,
            objectives=_objectives(families, target_kw),
            constraints=(operating_constraints(self.case)
                         if self.case is not None and self.case.case_id == context.case_id else []),
            fields=fields,
            requested_outputs=requested_outputs,
            clarification_required=bool(missing),
            clarification_question=question,
            source_text=text,
            retrieval_evidence=evidence,
            retrieval_trace={
                "method_id": (
                    self.retriever.method_id
                    if self.retrieval_enabled
                    else "disabled_retrieval"
                ),
                "elapsed_seconds": retrieval_elapsed,
                "evidence_count": len(evidence),
            },
            turn_index=turn_index,
            context_origin=context.origin,
        )
        if self.llm is None or not self.llm.available:
            return draft
        translated = self.llm.call(
            TaskSpecification,
            agent_name="translator",
            system=(
                "You are the B2G-Agent Translator. Preserve sender intent and populate only the supplied "
                "TaskSpecification schema. Use the deterministic draft, conversation history, and retrieved "
                "evidence; never invent physical results. A live conversation may differ completely from the "
                "frozen benchmark messages. Carry forward only terms established in prior turns, keep "
                "unresolved fields missing or to_be_simulated, and ask for clarification when an "
                "execution-critical value is ambiguous. Populate receiver_message with a complete "
                "natural-language request appropriate to the receiver, consistent with the structured "
                "contract. When Reviewer feedback concerns wording, repair receiver_message as well "
                "as any affected fields; preserve the sender's requested decisions and constraints. "
                "Only active_reviewer_feedback contains current correction requests; earlier internal "
                "review opinions are not sender requirements. Preserve already-correct terms while "
                "repairing current findings, and never reintroduce a withdrawn clarification. "
                "Keep building and bus lists paired in the same order. Preserve source_text and "
                "retrieval identifiers exactly; do not rewrite or duplicate provenance. Explicit "
                "case defaults may remain inferred but must not become invented sender demands. "
                "Constraints with origin=case_policy are mandatory public operating bounds, not "
                "claims about the sender's wording. Preserve their provenance and distinguish them "
                "from sender constraints. Comfort means zero response-induced violations of the "
                "effective bounds, not zero baseline nominal-band violations. Do not request "
                "clarification merely because a public operating bound was not repeated by the sender. "
                "Be concise while retaining every object, quantity, unit and time window. Return JSON only."
            ),
            user_payload={
                "source_text": text,
                "source_target_pairs": list(zip(target_bems, target_buses)) if len(target_bems) == len(target_buses) else [],
                "sender_role": sender.value,
                "interaction_context": context.model_dump(mode="json"),
                "prior_task_specification": (
                    prior_specification.model_dump(mode="json")
                    if prior_specification else None
                ),
                **_conversation_payload(conversation_history),
                "deterministic_draft": draft.model_dump(mode="json"),
                "retrieval_method": (
                    self.retriever.method_id
                    if self.retrieval_enabled
                    else "disabled_retrieval"
                ),
            },
        )
        # Runtime measurements and source provenance are not generated by the LLM.
        translated.retrieval_trace = draft.retrieval_trace
        translated.retrieval_evidence = draft.retrieval_evidence
        translated.source_text = text
        # Attach trusted provenance only to exactly matching public bounds.
        # Changed/omitted bounds remain visible for independent review.
        for constraint in translated.constraints:
            match = next((c for c in draft.constraints if (
                c.metric, c.operator, c.value, c.unit, c.hard) == (
                constraint.metric, constraint.operator, constraint.value, constraint.unit, constraint.hard)), None)
            if match is not None and constraint.origin != "sender":
                constraint.origin = "case_policy"
                constraint.evidence = match.evidence
        return translated

    @traced_stage("translator.translate_feedback")
    def translate_feedback(
        self,
        feedback: ReceiverFeedback,
        *,
        specification: TaskSpecification,
        physics_evidence: dict[str, object] | None = None,
        conversation_history: list[dict[str, str]] | None = None,
    ) -> ReverseTransferState:
        """Translate feedback into a reverse-direction state and sender message."""

        retrieval_started = perf_counter()
        evidence = (
            self.retriever.retrieve_for_transfer(
                feedback.text,
                direction=f"{specification.receiver_role.value}->{specification.sender_role.value}",
                business_types=(family.value for family in specification.task_families),
                time_scales=("interval",),
                spatial_levels=(
                    "portfolio" if len(specification.target_bems) > 1 else "building",
                    "bus",
                ),
                roles=(specification.sender_role.value, specification.receiver_role.value),
                per_partition=1,
            )
            if self.retrieval_enabled
            else []
        )
        retrieval_elapsed = perf_counter() - retrieval_started
        emit("retrieval.completed", agent="translator", direction="reverse",
             elapsed_seconds=retrieval_elapsed, evidence_count=len(evidence))
        commitment = (
            f" The committed quantity is {feedback.committed_power_kw:g} kW."
            if feedback.committed_power_kw is not None
            else ""
        )
        if feedback.decision == "accepted":
            outcome = "All original requirements were accepted, so the initiated task is 100% complete."
        elif feedback.decision == "modified":
            outcome = (
                f"The receiver modified the request; quantified completion is "
                f"{feedback.completion_percentage:g}%."
            )
        else:
            outcome = "The receiver rejected the request, so no service was committed and completion is 0%."
        deterministic = TranslatedFeedback(
            sender_message=f"{feedback.text.strip()}{commitment} {outcome} Basis: {feedback.completion_basis}"
        )
        translated = deterministic
        if self.llm is not None and self.llm.available:
            translated = self.llm.call(
                TranslatedFeedback,
                agent_name="translator_feedback",
                system=(
                    "You are the B2G-Agent Translator handling the reverse domain transfer. Preserve the "
                    "receiver's decision, quantities, modifications, constraints, and completion percentage. "
                    "Use natural language appropriate for the original sender. Never alter physics evidence "
                    "or turn partial completion into full acceptance. Repair active_reviewer_feedback "
                    "while retaining already-correct meaning; internal review opinions are not receiver decisions. Return JSON only."
                ),
                user_payload={
                    "receiver_feedback": feedback.model_dump(mode="json"),
                    "task_specification": specification.model_dump(mode="json"),
                    "physics_evidence": physics_evidence,
                    **_conversation_payload(conversation_history),
                    "retrieved_evidence": [item.model_dump(mode="json") for item in evidence],
                    "deterministic_draft": deterministic.model_dump(mode="json"),
                },
            )

        physics_ref: EvidenceRef | None = None
        metrics: dict[str, object] = {}
        if physics_evidence is not None:
            path = str(physics_evidence.get("evidence_path") or "")
            digest = physics_evidence.get("evidence_sha256")
            physics_ref = EvidenceRef(
                evidence_id=f"physics-{specification.episode_id}",
                source="physics_evidence",
                locator=path or "in-memory-physics-evidence",
                sha256=str(digest) if digest else None,
                excerpt=str(physics_evidence.get("status") or "validated physics evidence"),
            )
            raw_metrics = physics_evidence.get("business_metrics")
            if isinstance(raw_metrics, dict):
                raw_values = raw_metrics.get("requested_output_values")
                metrics = raw_values if isinstance(raw_values, dict) else raw_metrics

        fields: dict[str, TaskField] = {
            "decision": TaskField(
                value=feedback.decision,
                status=FieldStatus.DECLARED,
                confidence=1.0,
            ),
            "committed_power": TaskField(
                value=feedback.committed_power_kw,
                unit="kW" if feedback.committed_power_kw is not None else None,
                status=FieldStatus.DECLARED,
                confidence=1.0,
            ),
            "completion_percentage": TaskField(
                value=feedback.completion_percentage,
                unit="%",
                status=FieldStatus.DECLARED,
                confidence=1.0,
            ),
            "completion_basis": TaskField(
                value=feedback.completion_basis,
                status=FieldStatus.DECLARED,
                confidence=1.0,
            ),
        }
        for output in specification.requested_outputs:
            value = metrics.get(output)
            fields[output] = TaskField(
                value=value,
                status=(
                    FieldStatus.RESOLVED
                    if value is not None
                    else FieldStatus.TO_BE_SIMULATED
                ),
                confidence=1.0 if value is not None else 0.0,
                evidence=[physics_ref] if value is not None and physics_ref else [],
                note=(
                    None
                    if value is not None
                    else "The receiver declaration requires physical validation before release."
                ),
            )
        return ReverseTransferState(
            episode_id=specification.episode_id,
            source_role=specification.receiver_role,
            target_role=specification.sender_role,
            decision=feedback.decision,
            fields=fields,
            source_text=feedback.text,
            translated_message=translated.sender_message,
            retrieval_evidence=evidence,
            retrieval_trace={
                "method_id": (
                    self.retriever.method_id
                    if self.retrieval_enabled
                    else "disabled_retrieval"
                ),
                "elapsed_seconds": retrieval_elapsed,
                "evidence_count": len(evidence),
            },
            requires_simulation_validation=(
                physics_evidence is None and bool(specification.requested_outputs)
            ),
        )


def _conversation_payload(history):
    history = list(history or [])
    return {"conversation_history": [item for item in history if item.get("speaker") != "reviewer"],
            "active_reviewer_feedback": next((item.get("text") for item in reversed(history)
                                               if item.get("speaker") == "reviewer"), None)}


def _context_field(value: object | None, evidence: list[object]) -> TaskField:
    return TaskField(
        value=value,
        status=FieldStatus.INFERRED if value is not None else FieldStatus.MISSING,
        confidence=1.0 if value is not None else 0.0,
        evidence=evidence[:1] if value is not None else [],  # type: ignore[arg-type]
    )


def _counterpart(sender: AgentRole) -> AgentRole:
    return (
        AgentRole.BUILDING_ENERGY_MANAGER
        if sender == AgentRole.DISTRIBUTION_GRID_OPERATOR
        else AgentRole.DISTRIBUTION_GRID_OPERATOR
    )


def _objectives(families: list[TaskFamily], target_kw: float | None) -> list[Objective]:
    mapping = {
        TaskFamily.PEAK_SHAVING: ("feeder_peak_kw", "minimize", None, "kW"),
        TaskFamily.CONGESTION_MANAGEMENT: ("maximum_line_loading_pct", "at_most", 100.0, "%"),
        TaskFamily.VOLTAGE_MANAGEMENT: ("minimum_voltage_pu", "at_least", 0.95, "p.u."),
        TaskFamily.RESERVE_SERVICE: ("reserved_capacity_kw", "at_least", target_kw, "kW"),
        TaskFamily.RENEWABLE_ABSORPTION: ("pv_curtailment_kwh", "minimize", None, "kWh"),
        TaskFamily.BASELINE_FLEXIBILITY: ("baseline_nrmse", "minimize", None, "%"),
        TaskFamily.MARKET_TRADING: ("settlement_usd", "maximize", None, "USD"),
    }
    objectives: list[Objective] = []
    for priority, family in enumerate(
        (
            item for item in families
            if item not in {TaskFamily.COMPOSITE, TaskFamily.GENERAL_COORDINATION}
        ),
        1,
    ):
        metric, operator, value, unit = mapping[family]
        objectives.append(
            Objective(
                name=family.value,
                metric=metric,
                operator=operator,  # type: ignore[arg-type]
                value=value,
                unit=unit,
                priority=priority,
            )
        )
    return objectives


def _infer_families(text: str) -> list[TaskFamily]:
    lowered = text.lower()
    # Frozen and live requests commonly state business objectives in a leading
    # "for ..., request/offer ..." clause and mention voltage/thermal terms
    # later only as hard constraints.  Prefer that explicit objective clause so
    # a voltage limit is not promoted into a voltage-management objective.
    objective_clause = re.search(
        r"\bfor\s+(.+?)(?=,\s*(?:request|the portfolio|ask|assess|please))",
        lowered,
        flags=re.IGNORECASE,
    )
    scope = objective_clause.group(1) if objective_clause else lowered
    located: list[tuple[int, TaskFamily]] = []
    for family, aliases in FAMILY_ALIASES.items():
        positions = [scope.find(alias) for alias in aliases if alias in scope]
        if positions:
            located.append((min(positions), family))
    return [family for _, family in sorted(located, key=lambda item: item[0])]


def _parse_bems(text: str) -> list[str]:
    values = {int(match) for match in re.findall(r"\bbem[_\s-]?(\d{1,2})\b", text, re.IGNORECASE)}
    return [f"bem_{value:02d}" for value in sorted(values) if 1 <= value <= 99]


def _parse_buses(text: str) -> list[int]:
    match = re.search(
        r"(?:bus(?:es)?|nodes?|\u8282\u70b9)\s*[:：]?\s*([0-9,，、\s]+)",
        text,
        re.IGNORECASE,
    )
    if not match:
        return []
    return sorted(
        {
            int(value)
            for value in re.findall(r"\d+", match.group(1))
            if 1 <= int(value) <= 33
        }
    )


def _parse_windows(text: str) -> list[TimeWindow]:
    matches = re.findall(
        r"\b([01]\d|2[0-3]):([0-5]\d)\s*[-–—\u81f3\u5230]\s*([01]\d|2[0-3]):([0-5]\d)\b",
        text,
    )
    return [
        TimeWindow(start=f"{start_h}:{start_m}", end=f"{end_h}:{end_m}")
        for start_h, start_m, end_h, end_m in matches
    ]


def _parse_operating_day(text: str) -> str | None:
    labelled = re.search(r"on the ([a-z0-9 _-]+?) operating day", text, re.IGNORECASE)
    if labelled:
        return re.sub(r"[\s-]+", "_", labelled.group(1).strip().lower())
    dated = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", text)
    if dated:
        return dated.group(1)
    chinese = re.search(r"([\w\u4e00-\u9fff-]+)\u8fd0\u884c\u65e5", text)
    return chinese.group(1) if chinese else None


def _requires_power_target(families: list[TaskFamily]) -> bool:
    return any(family != TaskFamily.GENERAL_COORDINATION for family in families)


def _contains_priority_conflict(text: str) -> bool:
    lowered = text.lower()
    return (
        "both clauses are stated as highest priority" in lowered
        or ("highest priority" in lowered and "at the same time" in lowered)
        or (
            re.search(r"highest[- ]priorit", lowered) is not None
            and "aggregate load" in lowered
        )
        or ("\u6700\u9ad8\u4f18\u5148\u7ea7" in text and "\u540c\u65f6" in text)
    )


def _first_number_before_unit(text: str, unit: str) -> float | None:
    aliases = {
        "kw": r"k\s*w|\u5343\u74e6",
        "usd": r"(?:usd|\$)\s*/?\s*mwh|\u7f8e\u5143\s*/?\s*\u5146\u74e6\u65f6",
        "pu": r"p\.?\s*u\.?|\u6807\u5e7a",
    }[unit]
    match = re.search(rf"(-?\d+(?:\.\d+)?)\s*(?:{aliases})", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _parse_power_target(text: str) -> tuple[float | None, str | None, FieldStatus]:
    explicit = _first_number_before_unit(text, "kw")
    if explicit is not None:
        return explicit, "kW", FieldStatus.DECLARED
    implicit = re.search(
        r"(?:\bprovide|\benvelope\s+of|\btarget(?:\s+is)?|\u63d0\u4f9b|\u8c03\u8282\u91cf(?:\u4e3a|\u662f)?)\s*"
        r"(-?\d+(?:\.\d+)?)\s*(?=(?:during|for\s+the\s+period|\u5728|\u4e8e)\b)",
        text,
        flags=re.IGNORECASE,
    )
    if implicit:
        return float(implicit.group(1)), None, FieldStatus.MISSING
    return None, None, FieldStatus.MISSING


def _declares_kw_unit(text: str) -> bool:
    return bool(
        re.search(
            r"(?:\bunit\s+(?:is|should\s+be)|\buse|\bin)\s*(?:k\s*w|kilowatts?|\u5343\u74e6)",
            text,
            flags=re.IGNORECASE,
        )
    )


def _parse_voltage_band(text: str) -> tuple[float, float] | None:
    match = re.search(
        r"(\d+(?:\.\d+)?)\s*[-–—\u81f3\u5230]\s*(\d+(?:\.\d+)?)\s*"
        r"(?:p\.?\s*u\.?|\u6807\u5e7a)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    lower, upper = float(match.group(1)), float(match.group(2))
    return (lower, upper) if lower <= upper else (upper, lower)
