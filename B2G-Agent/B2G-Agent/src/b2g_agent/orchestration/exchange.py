from __future__ import annotations
from b2g_agent.observability.journal import traced_stage

from b2g_agent.contracts import (
    AgentRole,
    FieldStatus,
    InformationExchange,
    ReceiverFeedback,
    TaskSpecification,
)


@traced_stage("exchange.start")
def start_exchange(specification: TaskSpecification) -> InformationExchange:
    """Create the four information states without inventing a physical response."""
    requested_power = _requested_power_kw(specification)
    receiver_received = specification.receiver_message or _receiver_domain_message(specification, requested_power)
    if not specification.clarification_required:
        return InformationExchange(
            episode_id=specification.episode_id,
            sender_role=specification.sender_role,
            receiver_role=specification.receiver_role,
            sender_sent=specification.source_text,
            receiver_received=receiver_received,
            decision="pending_physics",
            requested_power_kw=requested_power,
        )

    receiver_sent = _clarification_feedback(specification)
    sender_received = (
        f"{_role_name(specification.receiver_role)} made no commitment because the original "
        "message was not uniquely executable. The current request is therefore 0% complete; "
        "a later turn may provide the missing information."
    )
    return InformationExchange(
        episode_id=specification.episode_id,
        sender_role=specification.sender_role,
        receiver_role=specification.receiver_role,
        sender_sent=specification.source_text,
        receiver_received=receiver_received,
        receiver_sent=receiver_sent,
        sender_received=sender_received,
        decision="needs_clarification",
        requested_power_kw=requested_power,
        completion_percentage=0.0,
        completion_basis="No commitment was possible from the ambiguous or conflicting message.",
    )


@traced_stage("exchange.feedback")
def apply_receiver_feedback(
    exchange: InformationExchange,
    feedback: ReceiverFeedback,
    *,
    translated_sender_message: str | None = None,
) -> InformationExchange:
    """Translate the receiver's one feedback message back to the original sender."""
    requested = exchange.requested_power_kw
    if feedback.decision == "accepted":
        translated = (
            f"{_role_name(exchange.receiver_role)} accepted the original task without modification. "
            "The initiated task is 100% complete."
        )
    elif feedback.decision == "modified":
        quantity = (
            f" and committed {feedback.committed_power_kw:g} kW"
            if feedback.committed_power_kw is not None
            else ""
        )
        requested_text = f" against the requested {requested:g} kW" if requested is not None else ""
        translated = (
            f"{_role_name(exchange.receiver_role)} modified the original task{quantity}"
            f"{requested_text}. The quantified completion of the initiated task is "
            f"{feedback.completion_percentage:g}%."
        )
    else:
        translated = (
            f"{_role_name(exchange.receiver_role)} rejected the original task. "
            "No requested service was committed, so completion is 0%."
        )
    if translated_sender_message is not None:
        translated = translated_sender_message
    return InformationExchange.model_validate(
        {
            **exchange.model_dump(),
            "receiver_sent": feedback.text,
            "sender_received": translated,
            "decision": feedback.decision,
            "committed_power_kw": feedback.committed_power_kw,
            "completion_percentage": feedback.completion_percentage,
            "completion_basis": feedback.completion_basis,
        }
    )


def _receiver_domain_message(
    specification: TaskSpecification,
    requested_power: float | None,
) -> str:
    families = " and ".join(
        family.value.replace("_", " ") for family in specification.task_families
    )
    windows = "; ".join(f"{window.start}-{window.end}" for window in specification.windows) or "an unresolved time window"
    bems = ", ".join(specification.target_bems) or "an unresolved building portfolio"
    buses = ", ".join(str(bus) for bus in specification.target_buses) or "unresolved distribution buses"
    quantity = (
        f"{requested_power:g} kW"
        if requested_power is not None
        else "a power quantity that is not yet uniquely specified"
    )
    conflict = specification.fields.get("priority_conflict")
    conflict_text = (
        " The source contains incompatible highest-priority power clauses, so their priority "
        "must be clarified before a commitment can be made."
        if conflict and conflict.status == FieldStatus.MISSING
        else ""
    )
    offer = specification.fields.get("offer_price") or specification.fields.get("price")
    offer_text = (
        f" The offered price is {float(offer.value):g} {offer.unit}."
        if offer and isinstance(offer.value, (int, float)) and offer.unit
        else ""
    )
    if specification.receiver_role == AgentRole.BUILDING_ENERGY_MANAGER:
        return (
            f"Assess whether the portfolio {bems} can deliver {quantity} for {families} during "
            f"{windows}. Any commitment must respect building comfort and rebound limits, and it "
            f"will be evaluated at distribution buses {buses} under the 0.95-1.05 p.u. voltage "
            f"band and 100% line-loading limit.{offer_text}{conflict_text}"
        )
    return (
        f"Assess the {quantity} flexibility offer from {bems} at buses {buses} for {families} "
        f"during {windows}. Determine network feasibility, the accepted quantity, residual need, "
        f"and any applicable settlement consequence; the commitment must retain comfort and "
        f"rebound evidence.{offer_text}{conflict_text}"
    )


def _clarification_feedback(specification: TaskSpecification) -> str:
    missing = [
        name
        for name, field in specification.fields.items()
        if field.status == FieldStatus.MISSING
    ]
    detail = ", ".join(name.replace("_", " ") for name in missing) or "material terms"
    return (
        f"I cannot accept the task as written because {detail} is missing, ambiguous, or "
        "conflicting. I am making no commitment in this turn; please clarify these terms."
    )


def _requested_power_kw(specification: TaskSpecification) -> float | None:
    field = specification.fields.get("target_power")
    if not field or field.unit != "kW" or not isinstance(field.value, (int, float)):
        return None
    return float(field.value)


def _role_name(role: AgentRole) -> str:
    return (
        "The building energy manager"
        if role == AgentRole.BUILDING_ENERGY_MANAGER
        else "The distribution grid operator"
    )
