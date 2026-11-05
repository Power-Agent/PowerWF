from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest

from b2g_agent.simulation.energyplus_controls import (
    build_event_idfs,
    controlled_zone_names,
    flexible_electric_schedules,
    patch_runperiod,
    thermostat_schedules,
)


SIMPLE_IDF = """
RunPeriod,
  RUNPERIOD 1, !- Name
  1, !- Begin Month
  1, !- Begin Day of Month
  , !- Begin Year
  12, !- End Month
  31, !- End Day of Month
  , !- End Year
  Wednesday, !- Day of Week for Start Day
  No;
Schedule:Compact, Heat Schedule, Temperature, Through: 12/31, For: AllDays, Until: 24:00,20.0;
Schedule:Compact, Cool Schedule, Temperature, Through: 12/31, For: AllDays, Until: 24:00,24.0;
Schedule:Compact, Equipment Schedule, Fraction, Through: 12/31, For: AllDays, Until: 24:00,1.0;
ThermostatSetpoint:DualSetpoint, Zone Control, Heat Schedule, Cool Schedule;
ZoneList, Conditioned Zones, Core Zone, Perimeter Zone;
ZoneControl:Thermostat, Main Thermostat, Conditioned Zones, Control Type, ThermostatSetpoint:DualSetpoint, Zone Control;
ElectricEquipment, Plug Load, Core Zone, Equipment Schedule, Watts/Area, , 5.0;
"""


def test_thermostat_schedules_resolve_types() -> None:
    heating, cooling = thermostat_schedules(SIMPLE_IDF)
    assert heating == {"Heat Schedule": "Schedule:Compact"}
    assert cooling == {"Cool Schedule": "Schedule:Compact"}


def test_controlled_zone_names_expand_zone_lists() -> None:
    assert controlled_zone_names(SIMPLE_IDF) == {"CORE ZONE", "PERIMETER ZONE"}


def test_flexible_electric_schedules_resolve_types() -> None:
    assert flexible_electric_schedules(SIMPLE_IDF) == {
        "Equipment Schedule": "Schedule:Compact"
    }


def test_patch_runperiod_uses_previous_day_and_actual_weekday() -> None:
    result, start = patch_runperiod(SIMPLE_IDF, target_date=date(2025, 2, 6))
    assert start == date(2025, 2, 5)
    assert "  2, !- Begin Month" in result
    assert "  5, !- Begin Day of Month" in result
    assert "  2, !- End Month" in result
    assert "  6, !- End Day of Month" in result
    assert "  Wednesday, !- Day of Week for Start Day" in result


def test_event_idf_contains_setpoint_and_equipment_controls(tmp_path: Path) -> None:
    source = tmp_path / "source.idf"
    baseline = tmp_path / "out" / "baseline.idf"
    event = tmp_path / "out" / "event.idf"
    source.write_text(SIMPLE_IDF, encoding="utf-8")
    metadata = build_event_idfs(
        source_idf=source,
        baseline_idf=baseline,
        event_idf=event,
        target_date=date(2025, 2, 6),
        offsets_c=[0.5] * 96,
        load_multipliers=[0.8] * 96,
    )

    text = event.read_text(encoding="utf-8")
    assert metadata.flexible_electric_schedules == ("Equipment Schedule",)
    assert metadata.minimum_load_multiplier == 0.8
    assert "B2G Flexible Load Multiplier" in text
    assert "B2G_Apply_Load_Multiplier" in text
    assert "Equipment Schedule" in text


def test_event_idf_rejects_negative_symmetric_setpoint_offsets(tmp_path: Path) -> None:
    source = tmp_path / "source.idf"
    source.write_text(SIMPLE_IDF, encoding="utf-8")
    with pytest.raises(ValueError, match="invert a dual-setpoint thermostat deadband"):
        build_event_idfs(
            source_idf=source,
            baseline_idf=tmp_path / "out" / "baseline.idf",
            event_idf=tmp_path / "out" / "event.idf",
            target_date=date(2025, 2, 6),
            offsets_c=[-0.5] * 96,
            load_multipliers=[1.2] * 96,
        )
