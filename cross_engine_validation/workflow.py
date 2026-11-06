"""Orchestrate equivalent transient studies across independent simulation MCPs.

The module deliberately does not implement simulator-specific MCP transports or
numerical trajectory metrics. PowerWF supplies callable MCP adapters and a
comparator (for example, the deterministic validator exposed by PowerMCP).
"""

from __future__ import annotations

import asyncio
import copy
import inspect
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, Awaitable, Callable, Mapping, Protocol


JsonObject = Mapping[str, Any]


class AsyncStudyTool(Protocol):
    """Callable adapter for one simulation MCP."""

    def __call__(self, study: JsonObject) -> Awaitable[Mapping[str, Any]]: ...


class Comparator(Protocol):
    """Callable adapter for a deterministic trajectory comparison service."""

    def __call__(
        self, first: Mapping[str, Any], second: Mapping[str, Any]
    ) -> Mapping[str, Any] | Awaitable[Mapping[str, Any]]: ...


class ComparisonStatus(StrEnum):
    """Terminal state of a cross-engine workflow."""

    COMPARED = "compared"
    INCOMPATIBLE_STUDY = "incompatible_study"
    ENGINE_FAILURE = "engine_failure"
    COMPARISON_FAILURE = "comparison_failure"


@dataclass(frozen=True)
class StudySpec:
    """Canonical, engine-neutral transient study definition."""

    study_id: str
    case_id: str
    parameters: Mapping[str, Any]
    disturbance: Mapping[str, Any]
    start_time_s: float
    end_time_s: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_payload(self) -> dict[str, Any]:
        """Return a detached payload shared by both engine adapters."""
        if not self.study_id.strip() or not self.case_id.strip():
            raise ValueError("study_id and case_id must be non-empty")
        if self.start_time_s < 0 or self.end_time_s <= self.start_time_s:
            raise ValueError("study time window must satisfy 0 <= start < end")
        if not self.disturbance:
            raise ValueError("disturbance must define the transient event")
        return copy.deepcopy(asdict(self))


@dataclass(frozen=True)
class EngineEndpoint:
    """Named simulation MCP endpoint and its callable adapter."""

    name: str
    run: AsyncStudyTool


@dataclass
class EngineRun:
    """Outcome from one engine, including raw payload for auditability."""

    engine: str
    status: str
    result: Mapping[str, Any] | None = None
    error: str | None = None


@dataclass
class TransientComparisonReport:
    """Auditable result of a two-engine transient comparison."""

    study: Mapping[str, Any]
    status: ComparisonStatus
    runs: list[EngineRun]
    comparison: Mapping[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report without discarding raw engine outputs."""
        return {
            "study": copy.deepcopy(dict(self.study)),
            "status": self.status.value,
            "runs": [asdict(run) for run in self.runs],
            "comparison": (
                copy.deepcopy(dict(self.comparison))
                if self.comparison is not None
                else None
            ),
            "error": self.error,
        }


async def _invoke(endpoint: EngineEndpoint, payload: Mapping[str, Any]) -> EngineRun:
    """Run one endpoint and convert exceptions into explicit outcomes."""
    try:
        result = endpoint.run(copy.deepcopy(dict(payload)))
        if not inspect.isawaitable(result):
            raise TypeError("MCP adapter must return an awaitable mapping")
        value = await result
        if not isinstance(value, Mapping):
            raise TypeError("MCP adapter result must be a mapping")
        return EngineRun(engine=endpoint.name, status="success", result=dict(value))
    except Exception as exc:  # keep the other engine's result for diagnosis
        return EngineRun(
            engine=endpoint.name,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


async def run_cross_engine_transient_study(
    study: StudySpec,
    first: EngineEndpoint,
    second: EngineEndpoint,
    comparator: Comparator,
) -> TransientComparisonReport:
    """Execute two equivalent MCP studies and compare successful trajectories.

    Both engines receive the same canonical study definition, detached to
    prevent one adapter mutating the payload observed by the other. Numerical
    comparison is delegated to the injected PowerMCP validation adapter.
    """
    payload = study.to_payload()
    if first.name.strip().casefold() == second.name.strip().casefold():
        return TransientComparisonReport(
            study=payload,
            status=ComparisonStatus.INCOMPATIBLE_STUDY,
            runs=[],
            error="Two distinct engine endpoints are required",
        )
    if not first.name.strip() or not second.name.strip():
        raise ValueError("engine endpoint names must be non-empty")

    runs = list(
        await asyncio.gather(
            _invoke(first, payload),
            _invoke(second, payload),
        )
    )
    if any(run.status != "success" for run in runs):
        return TransientComparisonReport(
            study=payload,
            status=ComparisonStatus.ENGINE_FAILURE,
            runs=runs,
            error="At least one engine failed; comparison was not attempted",
        )

    try:
        comparison = comparator(runs[0].result or {}, runs[1].result or {})
        if inspect.isawaitable(comparison):
            comparison = await comparison
        if not isinstance(comparison, Mapping):
            raise TypeError("comparator result must be a mapping")
    except Exception as exc:
        return TransientComparisonReport(
            study=payload,
            status=ComparisonStatus.COMPARISON_FAILURE,
            runs=runs,
            error=f"{type(exc).__name__}: {exc}",
        )

    return TransientComparisonReport(
        study=payload,
        status=ComparisonStatus.COMPARED,
        runs=runs,
        comparison=dict(comparison),
    )
