"""Dedicated LangGraph workflow for cross-engine transient validation.

This graph is intentionally separate from the PowerWorld-only grid-impact
workflow: it compares trajectories already returned by two simulator MCPs and
does not invoke either simulator.
"""

from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

from agent.tools_transient_validation import TRANSIENT_VALIDATION_TOOLS


class TransientValidationState(TypedDict):
    """Conversation state for a single transient-comparison request."""

    messages: Annotated[list[Any], add_messages]


_SYSTEM_PROMPT = """You are PowerWF's transient-trajectory validation assistant.

The user supplies numerical time-series results already collected from two
independent simulator MCPs. Your task is to call
compare_transient_trajectories_tool with those supplied values, then explain
the returned metrics and status. Do not invoke simulators or invent samples,
channel mappings, units, tolerances, or results.

Use explicit channel identifiers and an explicit channel_map when identifiers
differ. Supply a tolerance only when the user or study specification provides
one. A passed result means only that all compared channels met the supplied
maximum-absolute-error tolerances; it is not a general model-validation or
operational-security certification.

For per-unit (pu) channels, call the tool only if the input explicitly confirms
that both engines use the same engineering base. Set the same verified base_id
on both channel payloads. If bases are missing or differ, ask for the base
definitions instead of comparing pu values.

If the supplied data is incomplete or ambiguous, ask a focused clarification
rather than guessing. After the tool returns, report the comparison window,
alignment method, units, channel metrics, tolerance coverage, and limitations.
"""


def build_transient_validation_graph(model: Any | None = None):
    """Build the workflow; model is injectable for deterministic tests."""

    def compare_node(state: TransientValidationState, config: RunnableConfig):
        llm = model
        if llm is None:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY is required to run transient validation.")
            llm = ChatOpenAI(
                model="gpt-4o",
                temperature=0,
                max_retries=2,
                api_key=api_key,
            )
        response = llm.bind_tools(TRANSIENT_VALIDATION_TOOLS).invoke(
            [SystemMessage(content=_SYSTEM_PROMPT), *state.get("messages", [])],
            config=config,
        )
        return {"messages": [response]}

    def route_after_compare(state: TransientValidationState) -> str:
        messages = state.get("messages", [])
        if messages and getattr(messages[-1], "tool_calls", None):
            return "tools"
        return END

    workflow = StateGraph(TransientValidationState)
    workflow.add_node("compare", compare_node)
    workflow.add_node("tools", ToolNode(TRANSIENT_VALIDATION_TOOLS))
    workflow.set_entry_point("compare")
    workflow.add_conditional_edges("compare", route_after_compare, {"tools": "tools", END: END})
    workflow.add_edge("tools", "compare")
    return workflow.compile()


transient_validation_graph = build_transient_validation_graph()