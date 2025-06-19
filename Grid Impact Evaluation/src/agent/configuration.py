import os
from pydantic import BaseModel, Field
from typing import Any, Optional, List, Dict

from langchain_core.runnables import RunnableConfig


class Configuration(BaseModel):
    """The configuration for the simple chat agent with LangGraph tool support."""

    chat_model: str = Field(
        default="openai:gpt-4o",
        metadata={
            "description": "The name of the language model to use for the Grid Impact Study agent. Uses GPT-4o for enhanced power system analysis capabilities."
        },
    )

    temperature: float = Field(
        default=0.7,
        metadata={
            "description": "The temperature setting for the language model."
        },
    )

    enable_tools: bool = Field(
        default=True,
        metadata={
            "description": "Whether to enable PowerWorld analysis tools for grid impact studies."
        },
    )

    enable_tracing: bool = Field(
        default=True,
        metadata={
            "description": "Whether to enable LangSmith tracing for agent visualization."
        },
    )

    langsmith_project: str = Field(
        default="agent-datacenter-powerworld",
        metadata={
            "description": "LangSmith project name for organizing traces."
        },
    )

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}
        )

        # Get raw values from environment or config
        raw_values: dict[str, Any] = {
            name: os.environ.get(name.upper(), configurable.get(name))
            for name in cls.model_fields.keys()
        }

        # Filter out None values
        values = {k: v for k, v in raw_values.items() if v is not None}

        return cls(**values)
