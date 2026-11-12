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



@pytest.mark.parametrize(
    "changes",
    [
        {"start_time_s": float("nan")},
        {"end_time_s": float("inf")},
        {"parameters": None},
        {"disturbance": None},
        {"metadata": None},
        {"study_id": None},
        {"case_id": None},
    ],
)
def test_non_finite_and_malformed_study_metadata_is_rejected(study, changes):
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
    with pytest.raises((TypeError, ValueError)):
        StudySpec(**values).to_payload()


def test_sync_engine_adapter_is_reported_as_failed(study):
    def not_async(payload):
        return {"trajectory": [0, 1]}

    async def valid(payload):
        return {"trajectory": [0, 1]}

    report = asyncio.run(
        run_cross_engine_transient_study(
            study,
            EngineEndpoint("sync-engine", not_async),
            EngineEndpoint("async-engine", valid),
            lambda a, b: pytest.fail("comparison must not run"),
        )
    )
    assert report.status is ComparisonStatus.ENGINE_FAILURE
    assert report.runs[0].status == "failed"
    assert "awaitable" in report.runs[0].error


def test_adapter_mutation_does_not_modify_peer_input_or_report(study):
    seen = []

    async def mutating_adapter(payload):
        payload["parameters"]["solver_step_s"] = 9.0
        return {"trajectory": [1]}

    async def observing_adapter(payload):
        seen.append(payload["parameters"]["solver_step_s"])
        return {"trajectory": [2]}

    report = asyncio.run(
        run_cross_engine_transient_study(
            study,
            EngineEndpoint("mutator", mutating_adapter),
            EngineEndpoint("observer", observing_adapter),
            lambda a, b: {"compared": True},
        )
    )
    assert report.status is ComparisonStatus.COMPARED
    assert seen == [0.01]
    assert report.study["parameters"]["solver_step_s"] == 0.01



def test_blank_engine_name_is_rejected_before_duplicate_check(study):
    async def run(payload):
        return {"trajectory": [0, 1]}

    with pytest.raises(ValueError, match="endpoint names"):
        asyncio.run(
            run_cross_engine_transient_study(
                study,
                EngineEndpoint(" ", run),
                EngineEndpoint("engine-b", run),
                lambda a, b: {},
            )
        )


def test_async_comparator_is_supported(study):
    async def run(payload):
        return {"trajectory": [0, 1]}

    async def compare(first, second):
        return {"mae": 0.0, "source": "async-comparator"}

    report = asyncio.run(
        run_cross_engine_transient_study(
            study,
            EngineEndpoint("engine-a", run),
            EngineEndpoint("engine-b", run),
            compare,
        )
    )
    assert report.status is ComparisonStatus.COMPARED
    assert report.comparison["source"] == "async-comparator"


def test_invalid_comparator_return_type_is_reported(study):
    async def run(payload):
        return {"trajectory": [0, 1]}

    report = asyncio.run(
        run_cross_engine_transient_study(
            study,
            EngineEndpoint("engine-a", run),
            EngineEndpoint("engine-b", run),
            lambda a, b: [0.0],
        )
    )
    assert report.status is ComparisonStatus.COMPARISON_FAILURE
    assert "must be a mapping" in report.error



def test_study_time_strings_are_normalized_to_numeric_seconds(study):
    string_window = StudySpec(
        study_id=study.study_id,
        case_id=study.case_id,
        parameters=study.parameters,
        disturbance=study.disturbance,
        start_time_s="0.0",
        end_time_s="5.0",
    )
    payload = string_window.to_payload()
    assert payload["start_time_s"] == 0.0
    assert payload["end_time_s"] == 5.0
    assert isinstance(payload["start_time_s"], float)
    assert isinstance(payload["end_time_s"], float)


def test_engine_adapters_are_invoked_concurrently(study):
    import asyncio

    entered = 0
    both_entered = asyncio.Event()

    async def adapter(payload):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await asyncio.wait_for(both_entered.wait(), timeout=1.0)
        return {"trajectory": [0.0, 1.0]}

    report = asyncio.run(
        run_cross_engine_transient_study(
            study,
            EngineEndpoint("engine-a", adapter),
            EngineEndpoint("engine-b", adapter),
            lambda a, b: {"compared": True},
        )
    )
    assert report.status is ComparisonStatus.COMPARED
    assert entered == 2


def test_comparator_cannot_mutate_preserved_engine_outputs(study):
    raw_results = []

    async def adapter(payload):
        result = {"trajectory": [0.0, 1.0]}
        raw_results.append(result)
        return result

    def mutating_comparator(first, second):
        first["trajectory"][0] = 999.0
        second["injected"] = True
        return {"compared": True}

    report = asyncio.run(
        run_cross_engine_transient_study(
            study,
            EngineEndpoint("engine-a", adapter),
            EngineEndpoint("engine-b", adapter),
            mutating_comparator,
        )
    )
    assert report.status is ComparisonStatus.COMPARED
    assert report.runs[0].result["trajectory"] == [0.0, 1.0]
    assert "injected" not in report.runs[1].result
    assert raw_results[0]["trajectory"] == [0.0, 1.0]
