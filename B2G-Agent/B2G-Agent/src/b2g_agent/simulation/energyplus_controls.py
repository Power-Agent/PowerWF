from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path


WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")


@dataclass(frozen=True)
class EventIDFMetadata:
    source_sha256: str
    baseline_sha256: str
    event_sha256: str
    target_date: str
    run_start_date: str
    heating_schedules: tuple[str, ...]
    cooling_schedules: tuple[str, ...]
    flexible_electric_schedules: tuple[str, ...]
    maximum_absolute_offset_c: float
    minimum_load_multiplier: float
    maximum_load_multiplier: float


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean_object(raw: str) -> list[str]:
    cleaned_lines = [line.split("!", 1)[0] for line in raw.splitlines()]
    cleaned = "\n".join(cleaned_lines).strip()
    return [field.strip() for field in cleaned.rstrip(";").split(",")]


def idf_objects(text: str) -> list[tuple[str, list[str]]]:
    objects: list[tuple[str, list[str]]] = []
    for raw in text.split(";"):
        fields = _clean_object(raw + ";")
        if fields and fields[0]:
            objects.append((fields[0].lower(), fields))
    return objects


def thermostat_schedules(text: str) -> tuple[dict[str, str], dict[str, str]]:
    objects = idf_objects(text)
    schedule_types: dict[str, str] = {}
    for object_type, fields in objects:
        if object_type.startswith("schedule:") and len(fields) > 1:
            schedule_types[fields[1].lower()] = fields[0]
    heating: dict[str, str] = {}
    cooling: dict[str, str] = {}
    for object_type, fields in objects:
        if object_type == "thermostatsetpoint:dualsetpoint":
            if len(fields) < 4:
                raise ValueError("ThermostatSetpoint:DualSetpoint is incomplete")
            heat_name, cool_name = fields[2], fields[3]
            if heat_name.lower() not in schedule_types or cool_name.lower() not in schedule_types:
                raise ValueError("A thermostat setpoint schedule object is missing")
            heating[heat_name] = schedule_types[heat_name.lower()]
            cooling[cool_name] = schedule_types[cool_name.lower()]
    if not heating or not cooling:
        raise ValueError("No dual thermostat schedules were found")
    overlap = set(name.lower() for name in heating) & set(name.lower() for name in cooling)
    if overlap:
        raise ValueError(f"Schedules cannot be both heating and cooling controls: {sorted(overlap)}")
    return heating, cooling


def flexible_electric_schedules(text: str) -> dict[str, str]:
    """Return schedules that directly drive indoor ElectricEquipment objects."""
    objects = idf_objects(text)
    schedule_types = {
        fields[1].lower(): fields[0]
        for object_type, fields in objects
        if object_type.startswith("schedule:") and len(fields) > 1
    }
    schedules: dict[str, str] = {}
    for object_type, fields in objects:
        if object_type != "electricequipment" or len(fields) < 4 or not fields[3]:
            continue
        schedule_name = fields[3]
        schedule_type = schedule_types.get(schedule_name.lower())
        if schedule_type is None:
            raise ValueError(
                f"ElectricEquipment schedule object is missing: {schedule_name}"
            )
        schedules[schedule_name] = schedule_type
    if not schedules:
        raise ValueError("No ElectricEquipment schedules were found")
    return schedules


def controlled_zone_names(text: str) -> set[str]:
    objects = idf_objects(text)
    zone_lists = {
        fields[1].lower(): fields[2:]
        for object_type, fields in objects
        if object_type == "zonelist" and len(fields) > 2
    }
    zones: set[str] = set()
    for object_type, fields in objects:
        if object_type != "zonecontrol:thermostat" or len(fields) < 3:
            continue
        target = fields[2]
        if target.lower() in zone_lists:
            zones.update(name.upper() for name in zone_lists[target.lower()] if name)
        else:
            zones.add(target.upper())
    if not zones:
        raise ValueError("No thermostat-controlled zones were found")
    return zones


def patch_runperiod(text: str, *, target_date: date) -> tuple[str, date]:
    start_date = target_date - timedelta(days=1)
    replacements = {
        "Begin Month": start_date.month,
        "Begin Day of Month": start_date.day,
        "End Month": target_date.month,
        "End Day of Month": target_date.day,
    }
    result = text
    for label, value in replacements.items():
        pattern = re.compile(
            rf"(?im)^(\s*)\d+(\s*,\s*!-\s*{re.escape(label)}\s*)$"
        )
        result, count = pattern.subn(rf"\g<1>{value}\g<2>", result, count=1)
        if count != 1:
            raise ValueError(f"RunPeriod field was not found exactly once: {label}")
    weekday_pattern = re.compile(
        r"(?im)^(\s*)(Sunday|Monday|Tuesday|Wednesday|Thursday|Friday|Saturday)"
        r"(\s*,\s*!-\s*Day of Week for Start Day\s*)$"
    )
    result, count = weekday_pattern.subn(
        rf"\g<1>{WEEKDAYS[start_date.weekday()]}\g<3>", result, count=1
    )
    if count != 1:
        raise ValueError("RunPeriod weekday was not found exactly once")
    return result, start_date


def _offset_schedule(target_date: date, offsets_c: list[float]) -> str:
    if len(offsets_c) != 96:
        raise ValueError("Setpoint offset profile must contain 96 intervals")
    previous = target_date - timedelta(days=1)
    aligned_offsets = offsets_c[1:] + [0.0]
    lines = [
        "ScheduleTypeLimits,",
        "  B2G Offset Limits,",
        "  -5.0,",
        "  5.0,",
        "  Continuous,",
        "  Temperature;",
        "",
        "Schedule:Compact,",
        "  B2G Setpoint Offset,",
        "  B2G Offset Limits,",
        f"  Through: {previous.month}/{previous.day},",
        "  For: AllDays,",
        "  Until: 23:45,0.0,",
        f"  Until: 24:00,{offsets_c[0]:.9f},",
        f"  Through: {target_date.month}/{target_date.day},",
        "  For: AllDays,",
    ]
    for interval, value in enumerate(aligned_offsets, start=1):
        total_minutes = interval * 15
        hour, minute = divmod(total_minutes, 60)
        ending = ";" if interval == 96 and target_date.month == 12 and target_date.day == 31 else ","
        lines.append(f"  Until: {hour:02d}:{minute:02d},{value:.9f}{ending}")
    if not (target_date.month == 12 and target_date.day == 31):
        lines.extend(
            [
                f"  Through: 12/31,",
                "  For: AllDays,",
                "  Until: 24:00,0.0;",
            ]
        )
    return "\n".join(lines)


def _load_multiplier_schedule(target_date: date, multipliers: list[float]) -> str:
    if len(multipliers) != 96:
        raise ValueError("Load multiplier profile must contain 96 intervals")
    if any(value < 0.0 for value in multipliers):
        raise ValueError("Load multipliers must be nonnegative")
    previous = target_date - timedelta(days=1)
    aligned_multipliers = multipliers[1:] + [1.0]
    lines = [
        "ScheduleTypeLimits,",
        "  B2G Multiplier Limits,",
        "  0.0,",
        "  2.0,",
        "  Continuous,",
        "  Dimensionless;",
        "",
        "Schedule:Compact,",
        "  B2G Flexible Load Multiplier,",
        "  B2G Multiplier Limits,",
        f"  Through: {previous.month}/{previous.day},",
        "  For: AllDays,",
        "  Until: 23:45,1.0,",
        f"  Until: 24:00,{multipliers[0]:.9f},",
        f"  Through: {target_date.month}/{target_date.day},",
        "  For: AllDays,",
    ]
    for interval, value in enumerate(aligned_multipliers, start=1):
        total_minutes = interval * 15
        hour, minute = divmod(total_minutes, 60)
        ending = ";" if interval == 96 and target_date.month == 12 and target_date.day == 31 else ","
        lines.append(f"  Until: {hour:02d}:{minute:02d},{value:.9f}{ending}")
    if not (target_date.month == 12 and target_date.day == 31):
        lines.extend(
            [
                "  Through: 12/31,",
                "  For: AllDays,",
                "  Until: 24:00,1.0;",
            ]
        )
    return "\n".join(lines)


def _clone_control_schedules(
    text: str, schedule_names: set[str]
) -> tuple[str, dict[str, str]]:
    wanted = {name.lower(): name for name in schedule_names}
    clones: list[str] = []
    clone_names: dict[str, str] = {}
    counter = 0
    for object_type, fields in idf_objects(text):
        if not object_type.startswith("schedule:") or len(fields) < 2:
            continue
        if fields[1].lower() not in wanted:
            continue
        counter += 1
        original = wanted[fields[1].lower()]
        clone_name = f"B2G_Base_Copy_{counter:03d}"
        clone_names[original] = clone_name
        clone_fields = [fields[0], clone_name, *fields[2:]]
        clones.append(",\n  ".join(clone_fields) + ";")
    if len(clone_names) != len(schedule_names):
        missing = sorted(schedule_names - set(clone_names), key=str.lower)
        raise ValueError(f"Could not clone thermostat schedules: {missing}")
    return "\n\n".join(clones), clone_names


def _ems_controls(
    heating: dict[str, str],
    cooling: dict[str, str],
    flexible: dict[str, str],
    clone_names: dict[str, str],
) -> str:
    blocks = [
        "EnergyManagementSystem:Sensor,",
        "  B2G_Offset_Sensor,",
        "  B2G Setpoint Offset,",
        "  Schedule Value;",
        "",
    ]
    controls: list[tuple[str, str, str]] = []
    counter = 0
    for sign, schedules in ((-1.0, heating), (1.0, cooling)):
        for schedule_name in sorted(schedules, key=str.lower):
            counter += 1
            sensor = f"B2G_Base_Schedule_{counter:03d}"
            actuator = f"B2G_Schedule_Actuator_{counter:03d}"
            blocks.extend(
                [
                    "EnergyManagementSystem:Sensor,",
                    f"  {sensor},",
                    f"  {clone_names[schedule_name]},",
                    "  Schedule Value;",
                    "",
                    "EnergyManagementSystem:Actuator,",
                    f"  {actuator},",
                    f"  {schedule_name},",
                    f"  {schedules[schedule_name]},",
                    "  Schedule Value;",
                    "",
                ]
            )
            operator = "+" if sign > 0 else "-"
            controls.append((actuator, sensor, operator))
    program_lines = [
        "EnergyManagementSystem:Program,",
        "  B2G_Apply_Setpoint_Offset,",
        "  IF B2G_Offset_Sensor == 0.0,",
    ]
    program_lines.extend(f"  SET {actuator} = NULL," for actuator, _, _ in controls)
    program_lines.append("  ELSE,")
    program_lines.extend(
        f"  SET {actuator} = {sensor} {operator} B2G_Offset_Sensor,"
        for actuator, sensor, operator in controls
    )
    program_lines.append("  ENDIF;")
    blocks.extend(program_lines)
    blocks.extend(
        [
            "",
            "EnergyManagementSystem:Sensor,",
            "  B2G_Load_Multiplier_Sensor,",
            "  B2G Flexible Load Multiplier,",
            "  Schedule Value;",
            "",
        ]
    )
    flexible_controls: list[tuple[str, str]] = []
    for load_counter, schedule_name in enumerate(
        sorted(flexible, key=str.lower), start=1
    ):
        sensor = f"B2G_Base_Load_Schedule_{load_counter:03d}"
        actuator = f"B2G_Load_Schedule_Actuator_{load_counter:03d}"
        flexible_controls.append((actuator, sensor))
        blocks.extend(
            [
                "EnergyManagementSystem:Sensor,",
                f"  {sensor},",
                f"  {clone_names[schedule_name]},",
                "  Schedule Value;",
                "",
                "EnergyManagementSystem:Actuator,",
                f"  {actuator},",
                f"  {schedule_name},",
                f"  {flexible[schedule_name]},",
                "  Schedule Value;",
                "",
            ]
        )
    load_program = [
        "EnergyManagementSystem:Program,",
        "  B2G_Apply_Load_Multiplier,",
        "  IF B2G_Load_Multiplier_Sensor == 1.0,",
    ]
    load_program.extend(
        f"  SET {actuator} = NULL," for actuator, _ in flexible_controls
    )
    load_program.append("  ELSE,")
    load_program.extend(
        f"  SET {actuator} = @MAX 0.0 (@MIN 1.0 ({sensor} * B2G_Load_Multiplier_Sensor)),"
        for actuator, sensor in flexible_controls
    )
    load_program.append("  ENDIF;")
    blocks.extend(load_program)
    blocks.extend(
        [
            "",
            "EnergyManagementSystem:ProgramCallingManager,",
            "  B2G_Setpoint_Calling_Manager,",
            "  BeginTimestepBeforePredictor,",
            "  B2G_Apply_Setpoint_Offset,",
            "  B2G_Apply_Load_Multiplier;",
        ]
    )
    return "\n".join(blocks)


def build_event_idfs(
    *,
    source_idf: Path,
    baseline_idf: Path,
    event_idf: Path,
    target_date: date,
    offsets_c: list[float],
    load_multipliers: list[float],
) -> EventIDFMetadata:
    if baseline_idf.exists() or event_idf.exists():
        raise FileExistsError("Event IDF outputs must not already exist")
    if any(value < 0.0 for value in offsets_c):
        raise ValueError(
            "Negative symmetric setpoint offsets are unsafe because they can "
            "invert a dual-setpoint thermostat deadband"
        )
    source_text = source_idf.read_text(encoding="utf-8", errors="replace")
    heating, cooling = thermostat_schedules(source_text)
    flexible = flexible_electric_schedules(source_text)
    schedule_clones, clone_names = _clone_control_schedules(
        source_text, set(heating) | set(cooling) | set(flexible)
    )
    baseline_text, start_date = patch_runperiod(source_text, target_date=target_date)
    event_text = (
        baseline_text.rstrip()
        + "\n\n!- B2G-Agent deterministic event setpoint control\n"
        + _offset_schedule(target_date, offsets_c)
        + "\n\n"
        + _load_multiplier_schedule(target_date, load_multipliers)
        + "\n\n"
        + schedule_clones
        + "\n\n"
        + _ems_controls(heating, cooling, flexible, clone_names)
        + "\n"
    )
    baseline_idf.parent.mkdir(parents=True, exist_ok=True)
    baseline_idf.write_text(baseline_text, encoding="utf-8")
    event_idf.write_text(event_text, encoding="utf-8")
    return EventIDFMetadata(
        source_sha256=sha256_file(source_idf),
        baseline_sha256=sha256_file(baseline_idf),
        event_sha256=sha256_file(event_idf),
        target_date=target_date.isoformat(),
        run_start_date=start_date.isoformat(),
        heating_schedules=tuple(sorted(heating, key=str.lower)),
        cooling_schedules=tuple(sorted(cooling, key=str.lower)),
        flexible_electric_schedules=tuple(sorted(flexible, key=str.lower)),
        maximum_absolute_offset_c=max(abs(value) for value in offsets_c),
        minimum_load_multiplier=min(load_multipliers),
        maximum_load_multiplier=max(load_multipliers),
    )
