"""Paper-aligned interaction and agent evaluation."""

from b2g_agent.evaluation.methods import EvaluationMethod, EvaluationMethodRegistry
from b2g_agent.evaluation.physics_cache import (
    PhysicsCacheRecord,
    PhysicsResultCache,
    physics_cache_key,
    physics_input_bundle_sha256,
    physics_input_hashes,
    task_specification_sha256,
    task_specification_physics_payload,
)
from b2g_agent.evaluation.scoring import InteractionScore, InteractionScorer
from b2g_agent.evaluation.summary import (
    summarize_evaluation_runs,
    write_evaluation_summary,
)

__all__ = [
    "EvaluationMethod",
    "EvaluationMethodRegistry",
    "InteractionScore",
    "InteractionScorer",
    "PhysicsCacheRecord",
    "PhysicsResultCache",
    "physics_cache_key",
    "physics_input_bundle_sha256",
    "physics_input_hashes",
    "task_specification_sha256",
    "task_specification_physics_payload",
    "summarize_evaluation_runs",
    "write_evaluation_summary",
]
