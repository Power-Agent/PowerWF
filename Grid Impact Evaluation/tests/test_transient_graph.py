"""End-to-end graph tests for the transient validation workflow."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.transient_graph import build_transient_validation_graph


class FakeToolCallingModel:
    """Deterministic chat-model double that requests one real tool execution."""

    def __init__(self):
        self.calls = 0
        self.bound_tool_names = []

    def bind_tools(self, tools):
        self.bound_tool_names = [tool.name for tool in tools]
        return self

    def invoke(self, messages, config=None):
        self.calls += 1
        if self.calls == 1:
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "compare_transient_trajectories_tool",
                        "args": {
                            "reference_engine": "PowerWorld MCP",
                            "candidate_engine": "PSS/E MCP",
                            "reference_channels": {
                                "bus:1:voltage": {
                                    "time_s": [0.0, 1.0],
                                    "values": [1.0, 0.95],
                                    "unit": "pu",
                                    "base_id": "100-MVA-230-kV",
                                }
                            },
                            "candidate_channels": {
                                "bus:1:voltage": {
                                    "time_s": [0.0, 0.5, 1.0],
                                    "values": [1.0, 0.975, 0.95],
                                    "unit": "pu",
                                    "base_id": "100-MVA-230-kV",
                                }
                            },
                            "tolerances": {"bus:1:voltage": 1e-10},
                        },
                        "id": "transient-comparison-1",
                        "type": "tool_call",
                    }
                ],
            )
        assert any(isinstance(message, ToolMessage) for message in messages)
        return AIMessage(content="The supplied trajectories match within the specified tolerance.")


def test_graph_binds_and_executes_transient_tool_then_returns_summary():
    model = FakeToolCallingModel()
    graph = build_transient_validation_graph(model=model)

    result = graph.invoke(
        {"messages": [HumanMessage(content="Compare these two simulator trajectories.")]}
    )

    assert "compare_transient_trajectories_tool" in model.bound_tool_names
    assert model.calls == 2
    assert any(isinstance(message, ToolMessage) for message in result["messages"])
    assert result["messages"][-1].content.startswith("The supplied trajectories match")
