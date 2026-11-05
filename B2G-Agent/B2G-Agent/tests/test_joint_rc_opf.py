from __future__ import annotations

import pytest

from b2g_agent.contracts import (
    AgentRole,
    FieldStatus,
    TaskFamily,
    TaskField,
    TaskSpecification,
    TimeWindow,
)
from b2g_agent.simulation.joint_rc_opf import request_binding, window_indices


def specification(families: list[TaskFamily], windows: list[TimeWindow]) -> TaskSpecification:
    return TaskSpecification(
        episode_id="test",
        case_id="ieee33_10bem_15min",
        sender_role=AgentRole.DISTRIBUTION_GRID_OPERATOR,
        receiver_role=AgentRole.BUILDING_ENERGY_MANAGER,
        task_families=families,
        operating_day="winter_peak_01",
        windows=windows,
        target_bems=["bem_01"],
        target_buses=[3],
        fields={
            "target_power": TaskField(
                value=60.0, unit="kW", status=FieldStatus.DECLARED, confidence=1.0
            )
        },
        source_text="test request",
    )


def test_window_indices_are_end_exclusive_except_frozen_full_day() -> None:
    assert window_indices("16:00", "18:00") == list(range(64, 72))
    assert window_indices("00:00", "23:45") == list(range(96))


def test_single_renewable_family_requests_negative_service() -> None:
    task = specification(
        [TaskFamily.RENEWABLE_ABSORPTION],
        [TimeWindow(start="11:00", end="13:00")],
    )
    requested, direction = request_binding(task)
    assert requested[44:52].tolist() == [60.0] * 8
    assert direction[44:52].tolist() == [-1.0] * 8


def test_two_families_bind_positionally_to_two_windows() -> None:
    task = specification(
        [TaskFamily.RENEWABLE_ABSORPTION, TaskFamily.PEAK_SHAVING],
        [TimeWindow(start="11:00", end="13:00"), TimeWindow(start="17:00", end="19:00")],
    )
    _, direction = request_binding(task)
    assert direction[44:52].tolist() == [-1.0] * 8
    assert direction[68:76].tolist() == [1.0] * 8


def test_one_shared_window_uses_highest_priority_family_direction() -> None:
    task = specification(
        [TaskFamily.RENEWABLE_ABSORPTION, TaskFamily.PEAK_SHAVING],
        [TimeWindow(start="11:00", end="13:00")],
    )
    _, direction = request_binding(task)
    assert direction[44:52].tolist() == [-1.0] * 8

    reserve_first = specification(
        [TaskFamily.RESERVE_SERVICE, TaskFamily.RENEWABLE_ABSORPTION],
        [TimeWindow(start="11:00", end="13:00")],
    )
    _, reserve_direction = request_binding(reserve_first)
    assert reserve_direction[44:52].tolist() == [1.0] * 8


def test_mismatched_multiple_windows_are_rejected() -> None:
    task = specification(
        [
            TaskFamily.RENEWABLE_ABSORPTION,
            TaskFamily.PEAK_SHAVING,
            TaskFamily.RESERVE_SERVICE,
        ],
        [TimeWindow(start="11:00", end="13:00"), TimeWindow(start="17:00", end="19:00")],
    )
    with pytest.raises(ValueError, match="one shared window or one window per family"):
        request_binding(task)
