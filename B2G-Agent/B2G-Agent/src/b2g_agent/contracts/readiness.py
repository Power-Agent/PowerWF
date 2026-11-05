"""Execution-critical input checks, independent of LLMs and ground truth."""
from __future__ import annotations

import csv
import json
import math
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from b2g_agent.contracts.models import Constraint, FieldStatus, ReviewIssue, TaskFamily, TaskSpecification


def operating_policy_issues(specification: TaskSpecification, policy: list[Constraint]) -> list[ReviewIssue]:
    """A translated contract cannot silently relax the selected case's hard bounds."""
    issues = []
    for bound in policy:
        candidates = [c for c in specification.constraints if c.metric == bound.metric]
        covered = any(c.hard and c.unit == bound.unit and c.operator == bound.operator
            and isinstance(c.value, (int, float)) and math.isfinite(c.value)
            and (c.value >= bound.value if bound.operator == "at_least" else c.value <= bound.value)
            for c in candidates)
        if not covered:
            issues.append(ReviewIssue(code="T-CASE-POLICY", responsible_agent="translator",
                message=f"The contract omits or relaxes the public operating bound {bound.metric} {bound.operator} {bound.value} {bound.unit}.",
                structured_field=f"constraints.{bound.metric}", evidence_locator=bound.evidence[0].locator,
                basis_locator=bound.evidence[0].locator,
                correction_required="Retain the case operating bound separately from sender demands. If the sender conflicts with it, clarify the conflict; never relax the safety bound."))
    for constraint in specification.constraints:
        if constraint.origin == "case_policy" and not any(
            (constraint.metric, constraint.operator, constraint.value, constraint.unit, constraint.hard)
            == (bound.metric, bound.operator, bound.value, bound.unit, bound.hard) for bound in policy
        ):
            issues.append(ReviewIssue(code="T-CONSTRAINT-PROVENANCE", responsible_agent="translator",
                message=f"{constraint.metric} is labelled case_policy but differs from the public policy.",
                structured_field=f"constraints.{constraint.metric}.origin",
                correction_required="Cite the sender or inference supporting this additional bound; preserve the public operating policy separately."))
    return issues


def execution_readiness(specification: TaskSpecification, *, case=None) -> list[ReviewIssue]:
    problems: list[tuple[str, str]] = []
    resolved = {FieldStatus.DECLARED, FieldStatus.INFERRED, FieldStatus.RESOLVED}

    def quantity(name, unit, minimum=0):
        field = specification.fields.get(name)
        if (field is None or field.status not in resolved or isinstance(field.value, bool)
                or not isinstance(field.value, (int, float)) or not math.isfinite(field.value)
                or (minimum is not None and field.value < minimum) or field.unit != unit):
            bound = f" >= {minimum}" if minimum is not None else ""
            problems.append((f"fields.{name}", f"Resolve {name} as a finite value{bound} in {unit}; do not invent it."))

    quantity("target_power", "kW")
    if TaskFamily.MARKET_TRADING in specification.task_families:
        quantity("offer_price", "USD/MWh", minimum=None)
    for index, objective in enumerate(specification.objectives):
        if objective.value is not None and not math.isfinite(objective.value):
            problems.append((f"objectives.{index}", "Resolve finite objective values before optimization."))
    for index, constraint in enumerate(specification.constraints):
        values = constraint.value if isinstance(constraint.value, list) else [constraint.value]
        if not values or any(not math.isfinite(v) for v in values) or not constraint.unit.strip():
            problems.append((f"constraints.{index}", "Resolve finite constraint bounds and explicit units."))
        elif constraint.operator == "between" and (len(values) != 2 or values[0] > values[1]):
            problems.append((f"constraints.{index}", "Resolve an ordered lower and upper constraint bound."))
    resolution = specification.fields.get("time_resolution")
    if resolution is None or resolution.value != 15 or resolution.unit != "min" or resolution.status not in resolved:
        problems.append(("fields.time_resolution", "Bind the selected case's 15-minute resolution explicitly."))
    if not specification.operating_day:
        problems.append(("operating_day", "Specify an operating day."))
    if not specification.target_bems or not specification.target_buses:
        problems.append(("target_bems/target_buses", "Resolve the building portfolios and their network buses."))
    if len(set(specification.target_bems)) != len(specification.target_bems) or len(set(specification.target_buses)) != len(specification.target_buses):
        problems.append(("target_bems/target_buses", "Remove duplicate targets without changing the intended objects."))
    if not specification.windows:
        problems.append(("windows", "Resolve an explicit service window."))
    for index, window in enumerate(specification.windows):
        start, end = (
            int(window.start[:2]) * 60 + int(window.start[3:]), int(window.end[:2]) * 60 + int(window.end[3:]))
        if start >= end or start % 15 or end % 15:
            problems.append((f"windows.{index}", "Use a nonempty within-day window aligned to the case's 15-minute intervals."))
        try:
            ZoneInfo(window.timezone)
        except (ZoneInfoNotFoundError, ValueError):
            problems.append((f"windows.{index}.timezone", "Resolve a valid IANA timezone."))
    if not specification.requested_outputs:
        problems.append(("requested_outputs", "Identify the outputs required to verify completion."))
    # Service directions are fixed by family; overlapping opposite directions
    # cannot be handed to the optimizer as a single scalar request.
    families, windows = specification.task_families, specification.windows
    if len(windows) > 1 and len(families) > 1 and len(windows) != len(families):
        problems.append(("windows", "Bind one common window or one window per task family."))
    elif windows:
        bindings = list(zip(families, windows)) if len(families) == len(windows) else [(families[0], w) for w in windows]
        occupied = {}
        for family, window in bindings:
            sign = -1 if family == TaskFamily.RENEWABLE_ABSORPTION else 1
            start = int(window.start[:2]) * 60 + int(window.start[3:])
            end = int(window.end[:2]) * 60 + int(window.end[3:])
            for minute in range(start, end, 15):
                if minute in occupied and occupied[minute] != sign:
                    problems.append(("windows", "Resolve overlapping opposite service directions."))
                    break
                occupied[minute] = sign
    if case is not None:
        if any(w.timezone != case.manifest["timezone"] for w in specification.windows):
            problems.append(("windows.timezone", "Convert the service window to the selected case timezone before execution."))
        if specification.case_id != case.case_id:
            problems.append(("case_id", "Use the selected case and its data."))
        days = json.loads((case.case_dir / "days/target_days.json").read_text(encoding="utf-8"))["target_days"]
        if specification.operating_day not in days:
            problems.append(("operating_day", "Select an available operating day or prepare its physical input data first."))
        with (case.case_dir / "network/bem_to_bus.csv").open(encoding="utf-8-sig", newline="") as handle:
            mapping = {row["bem_id"]: int(row["bus"]) for row in csv.DictReader(handle)}
        if any(mapping.get(bem) != bus for bem, bus in zip(specification.target_bems, specification.target_buses)):
            problems.append(("target_bems/target_buses", "Bind each selected portfolio to its bus in the selected case mapping."))
    issues = [ReviewIssue(code="T-EXECUTION-INPUT", message=message, responsible_agent="translator",
                        evidence_locator=f"task_specification.{field}", structured_field=field,
                        correction_required=message, basis_locator="deterministic_execution_readiness")
            for field, message in dict(problems).items()]
    if case is not None:
        from b2g_agent.case_io.operating_policy import operating_constraints
        issues.extend(operating_policy_issues(specification, operating_constraints(case)))
    return issues
