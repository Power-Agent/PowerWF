from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from b2g_agent.contracts import AgentRole, Episode, PipelineResult, ReviewVerdict


SessionMode = Literal["interactive", "benchmark"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message_id: str
    speaker: Literal[
        "sender",
        "agent_to_receiver",
        "receiver",
        "agent_to_sender",
        "system",
    ]
    role: AgentRole | None = None
    text: str
    created_at: datetime = Field(default_factory=utc_now)


class SessionSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    mode: SessionMode
    user_role: AgentRole
    episode: Episode | None = None
    messages: list[ChatMessage]
    results: list[PipelineResult] = Field(default_factory=list)
    latest_result: PipelineResult | None = None
    created_at: datetime
    updated_at: datetime


class FinalReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session_id: str
    case_id: str
    episode_id: str | None = None
    turn_count: int = Field(ge=1)
    verdict: ReviewVerdict
    report_markdown: str
    trace_id: str | None = None
