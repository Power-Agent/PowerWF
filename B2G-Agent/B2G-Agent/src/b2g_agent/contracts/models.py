from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator, model_serializer

class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class AgentRole(str, Enum):
    BUILDING_ENERGY_MANAGER = "building_energy_manager"
    DISTRIBUTION_GRID_OPERATOR = "distribution_grid_operator"


class TaskFamily(str, Enum):
    PEAK_SHAVING = "peak_shaving"
    CONGESTION_MANAGEMENT = "congestion_management"
    VOLTAGE_MANAGEMENT = "voltage_management"
    RESERVE_SERVICE = "reserve_service"
    RENEWABLE_ABSORPTION = "renewable_absorption"
    BASELINE_FLEXIBILITY = "baseline_flexibility"
    MARKET_TRADING = "market_trading"
    COMPOSITE = "composite"
    GENERAL_COORDINATION = "general_coordination"


class EpisodeSplit(str, Enum):
    DEVELOPMENT = "development"
    MEMORY_SEED = "memory_seed"
    TEST = "test"


class FieldStatus(str, Enum):
    DECLARED = "declared"
    INFERRED = "inferred"
    MISSING = "missing"
    TO_BE_SIMULATED = "to_be_simulated"
    RESOLVED = "resolved"
    UNRESOLVED = "unresolved"


class EvidenceRef(StrictModel):
    evidence_id: str
    source: str
    locator: str
    sha256: str | None = None
    excerpt: str | None = None


class TaskField(StrictModel):
    value: Any | None = None
    unit: str | None = None
    status: FieldStatus
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[EvidenceRef] = Field(default_factory=list)
    note: str | None = None


class TimeWindow(StrictModel):
    start: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    end: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
    timezone: str = "America/New_York"


class InteractionContext(StrictModel):
    """Optional context for either a frozen benchmark task or a live message."""

    interaction_id: str
    case_id: str
    origin: Literal["benchmark_episode", "interactive_seeded", "interactive_inferred"]
    task_families: list[TaskFamily] = Field(default_factory=list)
    operating_day: str | None = None
    windows: list[TimeWindow] = Field(default_factory=list)
    target_bems: list[str] = Field(default_factory=list)
    target_buses: list[int] = Field(default_factory=list)
    communication_hint: str | None = None

    @model_validator(mode="after")
    def validate_optional_targets(self) -> InteractionContext:
        if self.target_bems and self.target_buses and len(self.target_bems) != len(self.target_buses):
            raise ValueError("context target_bems and target_buses must be one-to-one")
        return self


class Objective(StrictModel):
    name: str
    metric: str
    operator: Literal["minimize", "maximize", "at_least", "at_most", "equal", "reduce_by"]
    value: float | None = None
    unit: str | None = None
    priority: int = Field(default=1, ge=1)


class Constraint(StrictModel):
    name: str
    metric: str
    operator: Literal["at_least", "at_most", "equal", "between"]
    value: float | list[float]
    unit: str
    hard: bool = True
    origin: Literal["sender", "case_policy", "inferred"] | None = None
    evidence: list[EvidenceRef] = Field(default_factory=list)

    @model_serializer(mode="wrap")
    def serialize_provenance(self, handler):
        payload = handler(self)
        # Preserve legacy contract hashes when no provenance was supplied.
        if self.origin is None:
            payload.pop("origin", None)
        if not self.evidence:
            payload.pop("evidence", None)
        return payload


class TaskSpecification(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    episode_id: str
    case_id: str
    sender_role: AgentRole
    receiver_role: AgentRole
    task_families: list[TaskFamily] = Field(min_length=1)
    operating_day: str | None = None
    windows: list[TimeWindow] = Field(default_factory=list)
    target_bems: list[str] = Field(default_factory=list)
    target_buses: list[int] = Field(default_factory=list)
    objectives: list[Objective] = Field(default_factory=list)
    constraints: list[Constraint] = Field(default_factory=list)
    fields: dict[str, TaskField]
    requested_outputs: list[str] = Field(default_factory=list)
    clarification_required: bool = False
    clarification_question: str | None = None
    source_text: str
    receiver_message: str | None = Field(default=None, min_length=1)
    retrieval_evidence: list[EvidenceRef] = Field(default_factory=list)
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)
    turn_index: int = Field(default=1, ge=1)
    context_origin: Literal[
        "benchmark_episode",
        "interactive_seeded",
        "interactive_inferred",
    ] = "benchmark_episode"

    @model_validator(mode="after")
    def validate_targets_and_clarification(self) -> TaskSpecification:
        if self.target_bems and self.target_buses and len(self.target_bems) != len(self.target_buses):
            raise ValueError("target_bems and target_buses must be one-to-one")
        if self.clarification_required and not self.clarification_question:
            raise ValueError("clarification_question is required when clarification_required=true")
        return self

    @model_serializer(mode="wrap")
    def serialize_optional_draft(self, handler):
        payload = handler(self)
        # An absent communication draft adds no semantic content to legacy
        # contracts. A supplied draft is persisted for exact locked replay.
        if self.receiver_message is None:
            payload.pop("receiver_message", None)
        return payload


class ActionType(str, Enum):
    LOAD_CASE = "load_case"
    SELECT_BUILDING_MODELS = "select_building_models"
    LOAD_REFERENCE_PROFILES = "load_reference_profiles"
    RUN_ENERGYPLUS_BASELINE = "run_energyplus_baseline"
    CALIBRATE_RC_MODELS = "calibrate_rc_models"
    SOLVE_JOINT_RC_OPF = "solve_joint_rc_opf"
    RUN_ENERGYPLUS_EVENT = "run_energyplus_event"
    ALIGN_15_MIN_TIMESERIES = "align_15_min_timeseries"
    SCALE_BEM_PORTFOLIOS = "scale_bem_portfolios"
    APPLY_DER_DISPATCH = "apply_der_dispatch"
    MAP_LOADS_TO_BUSES = "map_loads_to_buses"
    RUN_PANDAPOWER = "run_pandapower"
    CALCULATE_MARKET_SETTLEMENT = "calculate_market_settlement"
    EXTRACT_METRICS = "extract_metrics"


class PlanAction(StrictModel):
    action_id: str
    action_type: ActionType
    tool: Literal[
        "case_io",
        "energyplus",
        "rc_model",
        "opf",
        "pandapower",
        "optimizer",
        "metrics",
    ]
    depends_on: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    outputs: list[str] = Field(default_factory=list)


class ExecutionPlan(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    episode_id: str
    plan_id: str
    time_resolution_minutes: Literal[15] = 15
    actions: list[PlanAction] = Field(min_length=1)
    requested_outputs: list[str]
    compiler_version: str
    valid: bool = False
    validation_errors: list[str] = Field(default_factory=list)
    normalization_log: list[dict[str, Any]] = Field(default_factory=list)


class PlanProposal(StrictModel):
    """LLM-authored finite-action proposal before deterministic compilation."""

    actions: list[PlanAction] = Field(min_length=1)


class ReviewVerdict(str, Enum):
    PASS = "pass"
    REVISE = "revise"
    ABSTAIN = "abstain"


class ReviewIssue(StrictModel):
    code: str
    message: str
    responsible_agent: Literal["translator", "actor", "human"]
    evidence_locator: str | None = None
    correction_required: str
    score: Literal[0.5, 1.0] = 1.0
    source_excerpt: str | None = None
    structured_field: str | None = None
    basis_locator: str | None = None


class ReviewCheck(StrictModel):
    """One auditable semantic or evidence-channel Reviewer judgement."""

    check_id: str
    channel: Literal["semantic", "evidence"]
    score: Literal[0.0, 0.5, 1.0]
    weight: float = Field(default=1.0, gt=0.0)
    message: str
    responsible_agent: Literal["translator", "actor", "human", "none"] = "none"
    source_excerpt: str | None = None
    structured_field: str | None = None
    basis_locator: str | None = None
    uncertainty_reason: str | None = None


class ReviewerDecision(StrictModel):
    episode_id: str
    verdict: ReviewVerdict
    issues: list[ReviewIssue] = Field(default_factory=list)
    checked_memory_ids: list[str] = Field(default_factory=list)
    deterministic_checks_passed: bool
    semantic_review_status: Literal["not_run", "passed", "uncertain", "failed", "unavailable"]
    clarification_verdict: Literal["not_reviewed", "not_applicable", "required", "unsupported"] = "not_reviewed"
    semantic_risk_score: float = Field(ge=0.0, le=1.0)
    evidence_risk_score: float = Field(ge=0.0, le=1.0)
    semantic_checks: list[ReviewCheck] = Field(default_factory=list)
    evidence_checks: list[ReviewCheck] = Field(default_factory=list)
    semantic_risk_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    evidence_risk_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    revision_count: int = Field(default=0, ge=0)
    routing_decision: Literal[
        "clarify_sender",
        "translator",
        "actor",
        "translator_then_actor",
        "release",
    ]
    final_message: str


SEMANTIC_REVIEW_CHECK_IDS = frozenset(
    {
        "semantic_objective",
        "semantic_action_direction",
        "semantic_objects",
        "semantic_quantities_units",
        "semantic_time_windows",
        "semantic_constraints",
        "semantic_requested_outputs",
        "semantic_decision_completion",
    }
)


class SemanticReviewResult(StrictModel):
    status: Literal["passed", "uncertain", "failed"]
    issues: list[ReviewIssue] = Field(default_factory=list)
    checks: list[ReviewCheck] = Field(min_length=8, max_length=8)
    clarification_verdict: Literal["not_applicable", "required", "unsupported"] = "not_applicable"
    clarification_basis: str | None = None

    @model_validator(mode="after")
    def validate_status(self) -> SemanticReviewResult:
        check_ids = [check.check_id for check in self.checks]
        if set(check_ids) != SEMANTIC_REVIEW_CHECK_IDS or len(set(check_ids)) != 8:
            raise ValueError("semantic review must report the complete fixed checklist")
        if any(check.channel != "semantic" for check in self.checks):
            raise ValueError("semantic review checks must use the semantic channel")
        scores = [check.score for check in self.checks] + [issue.score for issue in self.issues]
        expected = "failed" if 1.0 in scores else "uncertain" if 0.5 in scores else "passed"
        if self.status != expected:
            raise ValueError(f"Semantic status must be {expected} for the supplied scores; retain every finding")
        for check in self.checks:
            if check.score == 0 and check.responsible_agent != "none":
                raise ValueError("A passing semantic check cannot assign a correction owner")
            if check.score == 1 and check.responsible_agent == "none":
                raise ValueError("A failed semantic check must identify a correction owner")
            if check.score > 0 and not all((check.source_excerpt, check.structured_field, check.basis_locator)):
                raise ValueError("A nonzero semantic check requires a source excerpt, field and basis locator")
            if check.score == 0.5 and not (check.uncertainty_reason or "").strip():
                raise ValueError("An uncertain semantic check must identify the unresolved ambiguity in uncertainty_reason")
        for issue in self.issues:
            matching = next((c for c in self.checks if c.check_id == issue.code), None)
            if matching and (matching.score != issue.score or matching.responsible_agent != issue.responsible_agent):
                raise ValueError("An issue and its matching check must agree on score and correction owner")
        if self.clarification_verdict != "not_applicable" and not self.clarification_basis:
            raise ValueError("Clarification adjudication requires a source-grounded basis")
        if self.clarification_verdict == "unsupported" and not any(
            c.check_id == "semantic_decision_completion" and c.score == 1
            and c.responsible_agent == "translator" for c in self.checks
        ):
            raise ValueError("Unsupported clarification must identify a Translator decision/completion failure")
        return self


class EpisodeDifficulty(StrictModel):
    spatial_scope: Literal["single_bem", "multi_bem", "feeder_wide"]
    temporal_scope: Literal["single_window", "multi_window", "full_day"]
    feasibility: Literal["feasible", "infeasible"]
    communication: Literal["direct", "implicit_unit", "missing_field", "conflicting_constraints"]


class PerfectExecutionOracle(StrictModel):
    """Canonical semantic/plan outcome independent of a model's wording."""

    route: Literal["clarify_sender", "execute_physics_then_release"]
    canonical_receiver_instruction: str = Field(min_length=1)
    required_semantic_fields: list[str] = Field(min_length=1)
    expected_action_types: list[ActionType] = Field(default_factory=list)
    required_numeric_outputs: list[str] = Field(default_factory=list)
    receiver_decision_rule: str = Field(min_length=1)
    sender_completion_rule: str = Field(min_length=1)
    expected_receiver_feedback: str | None = None
    expected_sender_final: str | None = None
    expected_completion_percentage: float | None = Field(default=None, ge=0.0, le=100.0)

    @model_validator(mode="after")
    def validate_route(self) -> PerfectExecutionOracle:
        if self.route == "clarify_sender":
            if not self.expected_receiver_feedback or not self.expected_sender_final:
                raise ValueError("clarification oracle requires both natural-language outputs")
            if self.expected_completion_percentage != 0.0:
                raise ValueError("clarification oracle completion must be 0%")
        if self.route == "execute_physics_then_release" and not self.expected_action_types:
            raise ValueError("physics oracle requires an executable action contract")
        return self


class EpisodeGroundTruth(StrictModel):
    schema_version: Literal["2.0"] = "2.0"
    episode_id: str
    case_id: str
    source_message_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    task_specification: TaskSpecification
    perfect_execution: PerfectExecutionOracle
    minimum_action_types: list[ActionType]
    expected_reviewer_verdict: ReviewVerdict
    expected_issue_codes: list[str] = Field(default_factory=list)
    oracle_kind: Literal["deterministic_physics", "deterministic_contract", "expert_label"]
    contract_truth_status: Literal[
        "programmatically_generated_pending_expert_review",
        "expert_verified",
    ] = "programmatically_generated_pending_expert_review"
    physics_truth_status: Literal["pending", "verified", "not_required_for_current_turn"]
    numeric_oracle_path: str | None = None


class Episode(StrictModel):
    schema_version: Literal["3.0"] = "3.0"
    episode_id: str
    case_id: str
    split: EpisodeSplit
    group_id: str
    primary_family: TaskFamily
    task_families: list[TaskFamily] = Field(min_length=1)
    start_role: AgentRole
    operating_day: str
    target_bems: list[str] = Field(min_length=1)
    target_buses: list[int] = Field(min_length=1)
    windows: list[TimeWindow] = Field(min_length=1)
    difficulty: EpisodeDifficulty
    sender_message: str = Field(min_length=1)
    generator_version: str
    seed: int

    @model_validator(mode="after")
    def validate_episode_consistency(self) -> Episode:
        if len(self.target_bems) != len(self.target_buses):
            raise ValueError("episode target_bems and target_buses must be one-to-one")
        return self


class ReceiverFeedback(StrictModel):
    decision: Literal["accepted", "modified", "rejected"]
    text: str = Field(min_length=1)
    committed_power_kw: float | None = Field(default=None, ge=0.0)
    completion_percentage: float = Field(ge=0.0, le=100.0)
    completion_basis: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_completion(self) -> ReceiverFeedback:
        if self.decision == "accepted" and self.completion_percentage != 100.0:
            raise ValueError("accepted feedback must report 100% completion")
        if self.decision == "modified" and not 0.0 < self.completion_percentage < 100.0:
            raise ValueError("modified feedback must report completion strictly between 0 and 100")
        if self.decision == "rejected" and self.completion_percentage != 0.0:
            raise ValueError("rejected feedback must report 0% completion")
        if self.decision == "rejected" and self.committed_power_kw not in {None, 0.0}:
            raise ValueError("rejected feedback cannot contain a positive commitment")
        return self


class ReverseTransferState(StrictModel):
    """Structured state for the receiver-to-original-sender transfer."""

    episode_id: str
    source_role: AgentRole
    target_role: AgentRole
    decision: Literal["accepted", "modified", "rejected"]
    fields: dict[str, TaskField]
    source_text: str
    translated_message: str = Field(min_length=1)
    retrieval_evidence: list[EvidenceRef] = Field(default_factory=list)
    retrieval_trace: dict[str, Any] = Field(default_factory=dict)
    requires_simulation_validation: bool = False


class InformationExchange(StrictModel):
    """Two communications represented as four domain-specific information states."""

    schema_version: Literal["1.0"] = "1.0"
    episode_id: str
    sender_role: AgentRole
    receiver_role: AgentRole
    sender_sent: str = Field(min_length=1)
    receiver_received: str = Field(min_length=1)
    receiver_sent: str | None = None
    sender_received: str | None = None
    decision: Literal[
        "pending_physics",
        "needs_clarification",
        "accepted",
        "modified",
        "rejected",
    ]
    requested_power_kw: float | None = None
    committed_power_kw: float | None = Field(default=None, ge=0.0)
    completion_percentage: float | None = Field(default=None, ge=0.0, le=100.0)
    completion_basis: str | None = None

    @model_validator(mode="after")
    def validate_information_states(self) -> InformationExchange:
        completed = self.decision not in {"pending_physics"}
        if not completed and any(
            value is not None
            for value in (
                self.receiver_sent,
                self.sender_received,
                self.committed_power_kw,
                self.completion_percentage,
                self.completion_basis,
            )
        ):
            raise ValueError("a pending exchange cannot contain invented feedback or completion")
        if completed and (not self.receiver_sent or not self.sender_received):
            raise ValueError("a completed exchange requires receiver_sent and sender_received")
        if completed and self.completion_percentage is None:
            raise ValueError("a completed exchange requires quantified completion")
        if completed and not self.completion_basis:
            raise ValueError("a completed exchange requires a completion basis")
        if self.decision in {"needs_clarification", "rejected"} and self.completion_percentage != 0.0:
            raise ValueError("clarification/rejection must report 0% completion")
        if self.decision == "accepted" and self.completion_percentage != 100.0:
            raise ValueError("accepted exchange must report 100% completion")
        if (
            self.decision == "accepted"
            and self.requested_power_kw is not None
            and self.committed_power_kw is not None
            and self.requested_power_kw != self.committed_power_kw
        ):
            raise ValueError("accepted exchange cannot alter the requested power")
        if self.decision == "rejected" and self.committed_power_kw not in {None, 0.0}:
            raise ValueError("rejected exchange cannot contain a positive commitment")
        if self.decision == "modified" and not (
            self.completion_percentage is not None
            and 0.0 < self.completion_percentage < 100.0
        ):
            raise ValueError("modified exchange must quantify partial completion")
        return self


class PhysicsEvidence(StrictModel):
    """Traceable result of the deterministic closed-loop physics executor."""

    schema_version: Literal["1.0"] = "1.0"
    workflow: Literal["closed_loop", "bidirectional_flexibility_envelope"]
    status: Literal["validated_full", "validated_partial"]
    evidence_path: str = Field(min_length=1)
    evidence_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    receiver_decision: Literal["accepted", "modified", "rejected"]
    committed_power_kw: float = Field(ge=0.0)
    completion_percentage: float = Field(ge=0.0, le=100.0)
    replan_count: int | dict[str, int]
    converged: bool
    hard_constraints_passed: bool
    business_metrics: dict[str, Any]
    input_hashes: dict[str, Any]
    execution_plan_id: str | None = None
    execution_plan_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    execution_log: dict[str, Any] = Field(default_factory=dict)


class PipelineResult(StrictModel):
    episode_id: str
    turn_index: int = Field(default=1, ge=1)
    information_exchange: InformationExchange
    task_specification: TaskSpecification
    execution_plan: ExecutionPlan | None
    reviewer_decision: ReviewerDecision
    physics_evidence: PhysicsEvidence | None = None
    reverse_transfer: ReverseTransferState | None = None
    proposed_receiver_feedback: ReceiverFeedback | None = None
    trace_id: str
    interaction_journal: str | None = None
    workflow_status: Literal["awaiting_physics", "needs_clarification", "released", "blocked"] = "awaiting_physics"
    physics_attempts: list[dict[str, Any]] = Field(default_factory=list)
    interaction_context: InteractionContext | None = None
    conversation_history: list[dict[str, str]] = Field(default_factory=list)
    review_history: list[dict[str, Any]] = Field(default_factory=list)
    llm_request_hashes: list[str] = Field(default_factory=list)
    llm_invocation_ids: list[str] = Field(default_factory=list)
