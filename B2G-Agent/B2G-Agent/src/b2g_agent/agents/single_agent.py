from __future__ import annotations

from pathlib import Path
from time import perf_counter
import json

from pydantic import BaseModel, ConfigDict, Field, model_validator

from b2g_agent.agents.actor import PlanCompiler, _capability_catalog, SUPPORTED_REQUESTED_OUTPUTS
from b2g_agent.agents.llm import StructuredLLMClient
from b2g_agent.agents.retrieval import DeterministicFusionRetriever
from b2g_agent.agents.translator import _infer_families, _parse_bems
from b2g_agent.case_io import CaseBundle
from b2g_agent.case_io.operating_policy import operating_constraints, public_review_context
from b2g_agent.observability.journal import emit, traced_stage
from b2g_agent.contracts import (
    AgentRole,
    ExecutionPlan,
    InteractionContext,
    PlanProposal,
    ReceiverFeedback,
    TaskSpecification,
)


class SingleAgentContractProposal(BaseModel):
    """One same-LLM baseline response before deterministic execution checks."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    task_specification: TaskSpecification
    plan: PlanProposal | None = None
    receiver_received: str = Field(min_length=1)
    clarification_receiver_feedback: str | None = Field(default=None,
        description="Required nonempty receiver reply when clarification_required is true; otherwise null.")
    clarification_sender_result: str | None = Field(default=None,
        description="Required nonempty sender-facing explanation when clarification_required is true; otherwise null.")

    @model_validator(mode="after")
    def validate_route(self) -> SingleAgentContractProposal:
        if self.task_specification.clarification_required:
            if self.plan is not None:
                raise ValueError("a clarification proposal cannot include an execution plan")
            if not self.clarification_receiver_feedback or not self.clarification_sender_result:
                raise ValueError("a clarification proposal requires both natural-language messages")
        elif self.plan is None:
            raise ValueError("an executable single-agent proposal requires a tool plan")
        return self


class SingleAgentFeedback(ReceiverFeedback):
    """Both return messages are written by the same baseline agent."""

    sender_message: str = Field(min_length=1)


class SameLLMSingleAgent:
    """Matched tool-enabled baseline without role-split module calls or memory."""

    def __init__(
        self,
        knowledge_root: Path,
        *,
        llm: StructuredLLMClient,
        compiler: PlanCompiler | None = None,
        case: CaseBundle | None = None,
    ) -> None:
        self.retriever = DeterministicFusionRetriever(knowledge_root)
        self.llm = llm
        self.compiler = compiler or PlanCompiler()
        self.case = case or CaseBundle.default(knowledge_root.parent)

    @traced_stage("single_agent.contract")
    def propose_contract(
        self,
        text: str,
        *,
        sender: AgentRole,
        context: InteractionContext,
        prior_specification: TaskSpecification | None = None,
        conversation_history: list[dict[str, str]] | None = None,
        turn_index: int = 1,
        validation_feedback: list[dict] | None = None,
    ) -> tuple[SingleAgentContractProposal, ExecutionPlan | None]:
        if not self.llm.available:
            raise RuntimeError(
                "same_llm_single_agent requires a live or locked-replay LLM"
            )
        receiver = (
            AgentRole.BUILDING_ENERGY_MANAGER
            if sender == AgentRole.DISTRIBUTION_GRID_OPERATOR
            else AgentRole.DISTRIBUTION_GRID_OPERATOR
        )
        families = _infer_families(text) or context.task_families
        target_bems = _parse_bems(text) or context.target_bems
        retrieval_started = perf_counter()
        evidence = self.retriever.retrieve_for_transfer(
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
        retrieval_elapsed = perf_counter() - retrieval_started
        emit("retrieval.completed", agent="single_agent", direction="forward",
             elapsed_seconds=retrieval_elapsed, evidence_count=len(evidence))
        proposal = self.llm.call(
            SingleAgentContractProposal,
            agent_name="same_llm_single_agent_contract",
            system=(
                "You are the same-LLM tool-enabled single-agent comparison. In one response, "
                "interpret the sender's message, write the receiver-facing natural-language "
                "instruction, and either request clarification or choose a tool plan. You have "
                "the same knowledge evidence and tool catalog as the proposed method, but no "
                "Translator/Actor/Reviewer role split, no inter-module routing, and no failure "
                "memory. Deterministic tool/validation feedback, when present, is returned to you alone; "
                "revise your own interpretation and plan without inventing sender information. "
                "Use public case policy and mappings as authoritative engineering context. Preserve "
                "explicit quantities, units, windows and objectives; clarify genuinely missing or "
                "conflicting execution inputs. Never invent physical results. Keep the response compact; "
                "Use execution_field_contract's exact field keys and available_operating_days IDs. "
                "When clarification_required=true, set plan=null and supply BOTH nonempty "
                "clarification_receiver_feedback and clarification_sender_result; an empty/null reply "
                "is invalid. Otherwise provide the tool plan and leave clarification replies null. "
                "return the evaluation adapter schema only."
            ),
            user_payload={
                "source_text": text,
                "sender_role": sender.value,
                "receiver_role": receiver.value,
                "public_context": {
                    "interaction_id": context.interaction_id,
                    "case_id": context.case_id,
                    "origin": context.origin,
                },
                "prior_task_specification": (
                    prior_specification.model_dump(mode="json")
                    if prior_specification
                    else None
                ),
                "conversation_history": conversation_history or [],
                "retrieved_knowledge": [item.model_dump(mode="json") for item in evidence],
                "tool_capability_graph": _capability_catalog(),
                "fixed_case_rules": {
                    "time_resolution_minutes": 15,
                    "voltage_band_pu": [0.95, 1.05],
                    "maximum_line_loading_pct": 100.0,
                },
                "turn_index": turn_index,
                "public_case_context": public_review_context(self.case),
                "operating_constraints": [c.model_dump(mode="json") for c in operating_constraints(self.case)],
                "validation_feedback": validation_feedback or [],
                "execution_field_contract": {
                    "target_power": "Exact required key in task_specification.fields. Numeric power in kW from the sender; preserve missing/ambiguous units for clarification, never guess.",
                    "time_resolution": "Exact required key in fields. Public case default value=15, unit=min, status=inferred.",
                    "offer_price": "Exact key in fields for market trading. Numeric price in USD/MWh from the sender; clarify if required but missing.",
                    "resolved_statuses": ["declared", "inferred", "resolved"],
                },
                "available_operating_days": list(json.loads((self.case.case_dir/
                    "days/target_days.json").read_text(encoding="utf-8"))["target_days"]),
                "supported_requested_outputs": sorted(SUPPORTED_REQUESTED_OUTPUTS),
            },
        )
        specification = proposal.task_specification
        # Identity and provenance are adapter-owned public values, not semantic
        # answers. Overriding them prevents an LLM typo from breaking run joins.
        specification.episode_id = context.interaction_id
        specification.case_id = context.case_id
        specification.sender_role = sender
        specification.receiver_role = receiver
        specification.source_text = text
        specification.turn_index = turn_index
        specification.context_origin = context.origin
        specification.retrieval_evidence = evidence
        specification.retrieval_trace = {
            "method_id": self.retriever.method_id,
            "elapsed_seconds": retrieval_elapsed,
            "evidence_count": len(evidence),
        }
        plan = (
            self.compiler.compile_proposal(specification, proposal.plan)
            if proposal.plan is not None
            else None
        )
        return proposal, plan

    @traced_stage("single_agent.feedback")
    def propose_feedback(
        self,
        *,
        specification: TaskSpecification,
        physics_evidence: dict[str, object],
        tool_feedback: ReceiverFeedback,
    ) -> SingleAgentFeedback:
        if not self.llm.available:
            raise RuntimeError(
                "same_llm_single_agent requires a live or locked-replay LLM"
            )
        proposed = self.llm.call(
            SingleAgentFeedback,
            agent_name="same_llm_single_agent_result",
            system=(
                "You are the same tool-enabled single agent after physical execution. Produce the "
                "receiver's one natural-language feedback message using only the supplied validated "
                "EnergyPlus/pandapower evidence. State accept, modify, or reject and quantify "
                "completion. Also write sender_message: convey the same conclusion in the original "
                "sender's domain terminology. These are the one receiver reply and its one sender-facing "
                "rendering, not a new conversational turn. Do not alter or invent physical values. Return JSON only."
            ),
            user_payload={
                "task_specification": specification.model_dump(mode="json"),
                "physics_evidence": physics_evidence,
                "tool_result": tool_feedback.model_dump(mode="json"),
            },
        )
        # The model may phrase the natural-language response, but the simulator
        # remains the sole authority for the decision and completion fields.
        # This makes the baseline fail closed under an instruction-following
        # error instead of allowing generated numbers to bypass tool evidence.
        return proposed.model_copy(
            update={
                "decision": tool_feedback.decision,
                "committed_power_kw": tool_feedback.committed_power_kw,
                "completion_percentage": tool_feedback.completion_percentage,
                "completion_basis": tool_feedback.completion_basis,
            }
        )
