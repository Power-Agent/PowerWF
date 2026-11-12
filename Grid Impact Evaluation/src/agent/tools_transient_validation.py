"""LangChain tool exposing deterministic transient comparison to PowerWF agents."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, StrictFloat

from agent.transient_validation import (
    ChannelSeries,
    TrajectoryValidationError,
    compare_transient_trajectories,
)


class ChannelPayload(BaseModel):
    """JSON payload for a simulator channel."""

    time_s: list[StrictFloat] = Field(min_length=2, description="Strictly increasing timestamps in seconds")
    values: list[StrictFloat] = Field(min_length=2, description="Samples corresponding to time_s")
    unit: str = Field(description="Explicit engineering unit for all samples")
    base_id: str | None = Field(
        default=None,
        description="Required verified base identifier when unit is per-unit (pu); must match across engines",
    )


class CompareTransientTrajectoriesInput(BaseModel):
    """Input schema for comparing outputs already collected from two MCPs."""

    reference_engine: str = Field(description="Name of the reference simulator/MCP")
    candidate_engine: str = Field(description="Name of the candidate simulator/MCP")
    reference_channels: dict[str, ChannelPayload] = Field(
        description="Reference channel ID to time_s/values/unit payload"
    )
    candidate_channels: dict[str, ChannelPayload] = Field(
        description="Candidate channel ID to time_s/values/unit payload"
    )
    channel_map: dict[str, str] | None = Field(
        default=None,
        description="Optional explicit mapping from reference channel IDs to candidate channel IDs",
    )
    tolerances: dict[str, StrictFloat] | None = Field(
        default=None,
        description="Optional maximum absolute error by reference channel ID, in the normalized unit",
    )


@tool(args_schema=CompareTransientTrajectoriesInput)
def compare_transient_trajectories_tool(
    reference_engine: str,
    candidate_engine: str,
    reference_channels: dict[str, ChannelPayload],
    candidate_channels: dict[str, ChannelPayload],
    channel_map: dict[str, str] | None = None,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compare two MCP-provided transient trajectories using deterministic metrics.

    Call this only after both MCPs have returned numerical time-series data.
    It does not run either simulator and does not infer channel equivalence.
    """
    try:
        reference = {
            name: ChannelSeries(**payload.model_dump())
            for name, payload in reference_channels.items()
        }
        candidate = {
            name: ChannelSeries(**payload.model_dump())
            for name, payload in candidate_channels.items()
        }
        report = compare_transient_trajectories(
            reference,
            candidate,
            reference_engine=reference_engine,
            candidate_engine=candidate_engine,
            channel_map=channel_map,
            tolerances=tolerances,
        )
        return {"status": "success", "report": report.to_dict()}
    except (TrajectoryValidationError, ValueError, TypeError) as exc:
        return {"status": "error", "message": str(exc)}


TRANSIENT_VALIDATION_TOOLS = [compare_transient_trajectories_tool]
