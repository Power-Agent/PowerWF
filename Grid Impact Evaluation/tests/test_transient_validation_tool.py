"""Tests for the PowerWF LangChain transient-comparison tool."""

from agent.tools_transient_validation import compare_transient_trajectories_tool


def test_tool_accepts_mcp_style_payload_and_returns_report():
    result = compare_transient_trajectories_tool.invoke(
        {
            "reference_engine": "PowerWorld MCP",
            "candidate_engine": "PSS/E MCP",
            "reference_channels": {
                "bus:1:voltage": {
                    "time_s": [0.0, 1.0],
                    "values": [1.0, 0.95],
                    "unit": "pu",
                    "base_id": "system_base",
                }
            },
            "candidate_channels": {
                "bus:1:voltage": {
                    "time_s": [0.0, 0.5, 1.0],
                    "values": [1.0, 0.975, 0.95],
                    "unit": "pu",
                    "base_id": "system_base",
                }
            },
            "tolerances": {"bus:1:voltage": 1e-10},
        }
    )
    assert result["status"] == "success"
    assert result["report"]["status"] == "passed"
    assert result["report"]["channels"][0]["rmse"] == 0.0


def test_tool_returns_structured_error_for_incompatible_channels():
    result = compare_transient_trajectories_tool.invoke(
        {
            "reference_engine": "Engine A",
            "candidate_engine": "Engine B",
            "reference_channels": {
                "bus:1:voltage": {
                    "time_s": [0.0, 1.0],
                    "values": [1.0, 1.0],
                    "unit": "pu",
                }
            },
            "candidate_channels": {
                "bus:1:voltage": {
                    "time_s": [0.0, 1.0],
                    "values": [1.0, 1.0],
                    "unit": "MW",
                }
            },
        }
    )
    assert result["status"] == "error"
    assert "Incompatible units" in result["message"]



def test_tool_schema_rejects_boolean_samples_and_tolerances():
    import pytest
    from pydantic import ValidationError

    from agent.tools_transient_validation import CompareTransientTrajectoriesInput

    payload = {
        "reference_engine": "Engine A",
        "candidate_engine": "Engine B",
        "reference_channels": {
            "x": {"time_s": [0.0, 1.0], "values": [1.0, 1.0], "unit": "MW"}
        },
        "candidate_channels": {
            "x": {"time_s": [0.0, 1.0], "values": [1.0, 1.0], "unit": "MW"}
        },
    }
    for field, invalid in (
        ("time_s", [0.0, True]),
        ("values", [1.0, False]),
    ):
        invalid_payload = {
            **payload,
            "reference_channels": {
                "x": {
                    **payload["reference_channels"]["x"],
                    field: invalid,
                }
            },
        }
        with pytest.raises(ValidationError):
            CompareTransientTrajectoriesInput.model_validate(invalid_payload)

    with pytest.raises(ValidationError):
        CompareTransientTrajectoriesInput.model_validate(
            {**payload, "tolerances": {"x": True}}
        )
