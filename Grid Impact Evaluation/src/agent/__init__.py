"""Public entry points for the PowerWF agent workflows.

Graph modules are imported lazily so numerical utilities and MCP tools can be
used without initializing unrelated simulator integrations or requiring an
OpenAI API key at import time.
"""


def __getattr__(name: str):
    if name == "graph":
        from agent.graph import graph

        return graph
    if name == "transient_validation_graph":
        from agent.transient_graph import transient_validation_graph

        return transient_validation_graph
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["graph", "transient_validation_graph"]
