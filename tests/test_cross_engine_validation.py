"""Tests for the PowerWF cross-engine transient workflow."""

import asyncio

import pytest

from cross_engine_validation import (
    ComparisonStatus,
    EngineEndpoint,
    StudySpec,
    run_cross_engine_transient_study,
)


@pytest.fixture
def study():
    return StudySpec(
        study_id="ieee39-fault-001",
        case_id="ieee39",
        parameters={"solver_step_s": 0.01, "stop_time_s": 5.0},
        disturbance={
            "type": "three_phase_fault",
            "bus": 16,
            "apply_time_s": 1.0,
            "clear_time_s": 1.1,
        },
        start_time_s=0.0,
        end_time_s=5.0,
    )


def test_runs_both_engines_with_same_study_and_compares(study):
    received = []

    async def run_a(payload):
        received.append(("A", payload))
        return {"engine": "A", "trajectory": [0.0, 1.0]}

    async def run_b(payload):
        received.append(("B", payload))
        return {"engine": "B", "trajectory": [0.0, 1.01]}

    def compare(first, second):
        assert first["trajectory"] == [0.0, 1.0]
        assert second["trajectory"] == [0.0, 1.01]
        return {"rmse": 0.01, "passed": True}

    report = asyncio.run(
        run_cross_engine_transient_study(
            study, EngineEndpoint("engine-a", run_a),
            EngineEndpoint("engine-b", run_b), compare,
        )
    )

    assert report.status is ComparisonStatus.COMPARED
    assert report.comparison == {"rmse": 0.01, "passed": True}
    assert received[0][1] == received[1][1] == study.to_payload()
    assert received[0][1] is not received[1][1]


def test_one_engine_failure_prevents_comparison(study):
    async def run_ok(payload):
        return {"trajectory": [1]}

    async def run_fail(payload):
        raise RuntimeError("simulator unavailable")

    def compare(first, second):
        pytest.fail("comparison must not run when an engine fails")

    report = asyncio.run(
        run_cross_engine_transient_study(
            study, EngineEndpoint("ok", run_ok),
            EngineEndpoint("failed", run_fail), compare,
        )
    )

    assert report.status is ComparisonStatus.ENGINE_FAILURE
    assert [run.status for run in report.runs] == ["success", "failed"]
    assert "simulator unavailable" in report.runs[1].error


def test_comparator_failure_is_reported(study):
    async def run(payload):
        return {"trajectory": [1]}

    def compare(first, second):
        raise ValueError("unit mismatch")

    report = asyncio.run(
        run_cross_engine_transient_study(
            study, EngineEndpoint("a", run),
            EngineEndpoint("b", run), compare,
        )
    )

    assert report.status is ComparisonStatus.COMPARISON_FAILURE
    assert "unit mismatch" in report.error


def test_same_engine_is_rejected_without_execution(study):
    async def run(payload):
        pytest.fail("engine should not execute")

    report = asyncio.run(
        run_cross_engine_transient_study(
            study, EngineEndpoint("PSS/E", run),
            EngineEndpoint("pss/e", run), lambda a, b: {},
        )
    )

    assert report.status is ComparisonStatus.INCOMPATIBLE_STUDY


@pytest.mark.parametrize(
    "changes",
    [
        {"study_id": ""},
        {"case_id": ""},
        {"start_time_s": -1.0},
        {"end_time_s": 0.0},
        {"disturbance": {}},
    ],
)
def test_invalid_study_definition_is_rejected(study, changes):
    values = {
        "study_id": study.study_id,
        "case_id": study.case_id,
        "parameters": study.parameters,
        "disturbance": study.disturbance,
        "start_time_s": study.start_time_s,
        "end_time_s": study.end_time_s,
        "metadata": study.metadata,
    }
    values.update(changes)
    invalid = StudySpec(**values)

    with pytest.raises(ValueError):
        invalid.to_payload()


def test_report_is_json_friendly(study):
    async def run(payload):
        return {"trajectory": [1, 2]}

    report = asyncio.run(
        run_cross_engine_transient_study(
            study, EngineEndpoint("a", run),
            EngineEndpoint("b", run), lambda a, b: {"mae": 0.0},
        )
    )

    data = report.to_dict()
    assert data["status"] == "compared"
    assert data["runs"][0]["result"] == {"trajectory": [1, 2]}
