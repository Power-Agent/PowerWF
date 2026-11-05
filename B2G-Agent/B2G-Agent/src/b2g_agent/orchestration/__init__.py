"""End-to-end B2G-Agent orchestration."""

from b2g_agent.orchestration.pipeline import B2GPipeline
from b2g_agent.orchestration.executor import CoupledRunInputs, PhysicalExecutor
from b2g_agent.orchestration.exchange import apply_receiver_feedback, start_exchange
from b2g_agent.orchestration.task_physics import TaskPhysicsExecutor

__all__ = [
    "B2GPipeline",
    "CoupledRunInputs",
    "PhysicalExecutor",
    "TaskPhysicsExecutor",
    "apply_receiver_feedback",
    "start_exchange",
]
