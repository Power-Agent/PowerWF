import json
from pathlib import Path

from b2g_agent.agents import ActorAgent, ReviewerAgent
from b2g_agent.case_io import CaseBundle
from b2g_agent.contracts import (
    EpisodeGroundTruth,
    PhysicsEvidence,
    PipelineResult,
    ReceiverFeedback,
)
from b2g_agent.evaluation import InteractionScorer
from b2g_agent.orchestration.exchange import apply_receiver_feedback, start_exchange


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "cases" / "ieee33_10bem_15min"


def _truth(episode_id: str) -> EpisodeGroundTruth:
    for line in (CASE_DIR / "groundtruth" / "contract_oracles.jsonl").read_text(
        encoding="utf-8"
    ).splitlines():
        payload = json.loads(line)
        if payload["episode_id"] == episode_id:
            return EpisodeGroundTruth.model_validate(payload)
    raise KeyError(episode_id)


def _scorer() -> InteractionScorer:
    return InteractionScorer(CASE_DIR / "evaluation" / "scoring.json")


def test_perfect_physics_result_scores_one_hundred() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = case.get_episode("ep-001")
    truth = _truth(episode.episode_id)
    numeric = json.loads(
        (CASE_DIR / "groundtruth" / "numeric_oracles" / "ep-001.json").read_text(
            encoding="utf-8"
        )
    )
    specification = truth.task_specification
    plan = ActorAgent().plan(specification)
    assert plan is not None and plan.valid
    final = numeric["final_decision"]
    exchange = apply_receiver_feedback(
        start_exchange(specification),
        ReceiverFeedback(
            decision=final["receiver_decision"],
            text=final["expected_receiver_feedback"],
            committed_power_kw=final["requested_power_kw"],
            completion_percentage=final["completion_percentage"],
            completion_basis="Frozen physically validated delivered-versus-requested ratio.",
        ),
    )
    reviewer = ReviewerAgent().review(specification, plan, simulation_status="completed")
    evidence = PhysicsEvidence(
        workflow="closed_loop",
        status="validated_full",
        evidence_path="test/evidence_package.json",
        evidence_sha256="a" * 64,
        receiver_decision=final["receiver_decision"],
        committed_power_kw=final["requested_power_kw"],
        completion_percentage=final["completion_percentage"],
        replan_count=numeric["replan_count"],
        converged=True,
        hard_constraints_passed=True,
        business_metrics=numeric["business_metrics"],
        input_hashes=numeric["input_hashes"],
    )
    candidate = PipelineResult(
        episode_id=episode.episode_id,
        information_exchange=exchange,
        task_specification=specification,
        execution_plan=plan,
        reviewer_decision=reviewer,
        physics_evidence=evidence,
        trace_id="perfect-physics",
    )

    score = _scorer().score(
        episode=episode,
        groundtruth=truth,
        candidate=candidate,
        numeric_oracle=numeric,
    )

    assert score.semantic_fidelity == 1.0
    assert score.actionability == 1.0
    assert score.physical_feasibility == 1.0
    assert score.goal_achievement == 1.0
    assert score.interaction_quality == 100.0
    assert score.task_success is True

    blocked = _scorer().score(episode=episode, groundtruth=truth,
        candidate=candidate.model_copy(update={"workflow_status": "blocked"}), numeric_oracle=numeric)
    assert not blocked.actionability_checks["final_delivery"]
    assert blocked.actionability < 1 and not blocked.task_success
    assert blocked.physical_feasibility == score.physical_feasibility


def test_correct_clarification_scores_one_hundred_without_physics() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = next(
        item
        for item in case.iter_episodes()
        if item.difficulty.communication == "missing_field"
    )
    truth = _truth(episode.episode_id)
    specification = truth.task_specification
    exchange = start_exchange(specification)
    reviewer = ReviewerAgent().review(specification, None, simulation_status="not_run")
    candidate = PipelineResult(
        episode_id=episode.episode_id,
        information_exchange=exchange,
        task_specification=specification,
        execution_plan=None,
        reviewer_decision=reviewer,
        trace_id="perfect-clarification",
    )

    score = _scorer().score(
        episode=episode,
        groundtruth=truth,
        candidate=candidate,
        numeric_oracle=None,
    )

    assert score.interaction_quality == 100.0
    assert score.task_success is True
    assert score.agent_metrics["actor_tool_chain_success"] is True

    blocked = _scorer().score(episode=episode, groundtruth=truth,
        candidate=candidate.model_copy(update={"workflow_status": "blocked"}), numeric_oracle=None)
    assert blocked.actionability == 0 and not blocked.task_success


def test_first_transfer_semantics_penalize_an_incorrect_visible_quantity() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = case.get_episode("ep-003")
    truth = _truth(episode.episode_id)
    specification = truth.task_specification.model_copy(deep=True)
    specification.fields["target_power"].value = 999.0
    exchange = start_exchange(specification)
    reviewer = ReviewerAgent().review(specification, None, simulation_status="not_run")
    candidate = PipelineResult(
        episode_id=episode.episode_id,
        information_exchange=exchange,
        task_specification=specification,
        execution_plan=None,
        reviewer_decision=reviewer,
        trace_id="wrong-visible-quantity",
    )

    score = _scorer().score(
        episode=episode,
        groundtruth=truth,
        candidate=candidate,
        numeric_oracle=None,
    )

    assert score.first_transfer["field_checks"]["target_power"] is False
    assert score.semantic_fidelity < 1.0


def test_adopted_fidelity_threshold_retains_full_delivery_requirement() -> None:
    case = CaseBundle.default(REPO_ROOT)
    episode = case.get_episode("ep-002")
    truth = _truth(episode.episode_id)
    specification = truth.task_specification.model_copy(deep=True)
    assert specification.clarification_required
    # Correctly preserve the missing quantity and ask for clarification, while
    # two representation fields differ from the reference contract.
    specification.fields["target_power"].unit = "kW"
    specification.objectives[0].name = "assess_peak_shaving"
    candidate = PipelineResult(
        episode_id=episode.episode_id,
        information_exchange=start_exchange(specification),
        task_specification=specification, execution_plan=None,
        reviewer_decision=ReviewerAgent().review(specification, None, simulation_status="not_run"),
        trace_id="adopted-threshold-regression",
    )
    scorer = _scorer()
    assert scorer.config["semantic_success_threshold"] == 0.85
    current = scorer.score(episode=episode, groundtruth=truth, candidate=candidate, numeric_oracle=None)
    assert 0.85 <= current.semantic_fidelity < 0.9
    assert current.actionability == current.physical_feasibility == current.goal_achievement == 1
    assert current.task_success
    blocked = scorer.score(episode=episode, groundtruth=truth,
        candidate=candidate.model_copy(update={"workflow_status": "blocked"}), numeric_oracle=None)
    assert not blocked.task_success
    scorer.config["semantic_success_threshold"] = 0.9
    historical = scorer.score(episode=episode, groundtruth=truth, candidate=candidate, numeric_oracle=None)
    assert not historical.task_success
    assert historical.interaction_quality == current.interaction_quality
