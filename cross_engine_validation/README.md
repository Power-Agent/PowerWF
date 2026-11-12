# Cross-Engine Transient Simulation Workflow

This package provides a reusable orchestration layer for a caller that has
selected two independent simulation MCP adapters. It executes one canonical
transient-stability study through both adapters and passes the returned
trajectories to a configured validation service.

## Responsibility split

- **Calling agent/application:** interprets the user's request, chooses compatible
  MCP endpoints, constructs a common study definition, and presents the report.
- **This orchestration package:** runs both injected adapters concurrently,
  preserves independent outcomes, and assembles the auditable report.
- **Validation service:** provides deterministic trajectory comparison
  (time alignment, channel/unit checks, error metrics and tolerance evaluation).
- **Simulator MCPs:** execute the actual studies and return their results.

This PR adds a callable orchestration package; it does not register simulator
adapters into the existing default PowerWorld-only graph. Deployments supply
the two configured adapters and the validation-service adapter at the call
site. The package does not reimplement simulator APIs or claim that waveform
similarity alone proves system-stability equivalence.

## Cross-engine transient workflow

Each simulator adapter must accept the same canonical study definition and
return a mapping containing raw results. The comparator adapter should call
the PowerMCP dynamic-validation API and return its structured report. The
comparator must preserve explicit validation state and identify its metric
units and tolerances; the orchestration layer does not independently interpret
arbitrary comparator payloads.

```python
from cross_engine_validation import (
    EngineEndpoint,
    StudySpec,
    run_cross_engine_transient_study,
)

study = StudySpec(
    study_id="ieee39-fault-at-bus-16",
    case_id="ieee39",
    parameters={"time_step_s": 0.01, "stop_time_s": 5.0},
    disturbance={
        "type": "three_phase_fault",
        "bus": 16,
        "apply_time_s": 1.0,
        "clear_time_s": 1.1,
    },
    start_time_s=0.0,
    end_time_s=5.0,
)

report = await run_cross_engine_transient_study(
    study=study,
    first=EngineEndpoint("engine-a", first_transient_mcp_adapter),
    second=EngineEndpoint("engine-b", second_transient_mcp_adapter),
    comparator=powermcp_comparator_adapter,
)
print(report.to_dict())
```

The transient orchestration layer remains engine-agnostic. It refuses to
compare if either engine fails and reports comparison-service failures
separately. It preserves raw results for traceability.

## Engineering safeguards

- Both endpoints receive identical canonical study definitions as detached
  copies, so one adapter cannot mutate the other engine's input.
- Distinct engine endpoints are required.
- Failed runs are explicit; no partial run is presented as a successful
  comparison.
- Numerical validation is delegated to the configured validator; no implicit
  unit conversion, channel equivalence or extrapolation is introduced here.
- The workflow's `compared` status means a comparator report was produced; it
  is not itself a tolerance pass or engineering acceptance verdict.
- Before a comparison, deployments must verify model identity, base values,
  controller/protection settings, solver configuration and event semantics.

## Status semantics

- `compared`: both runs succeeded and the comparator returned a report.
- `incompatible_study`: the endpoints are not distinct.
- `engine_failure`: at least one MCP run failed; comparison was skipped.
- `comparison_failure`: the validation adapter failed or returned an invalid
  result.

## Tests

Run from the repository root:

```bash
python -m pytest tests/test_cross_engine_validation.py -q
```

These tests use deterministic fake MCP adapters. They validate orchestration
and failure semantics, not licensed simulator execution or an external
validation-service integration, which require configured simulator MCPs and
runtime installations. The pandapower regression coverage is exercised
separately by the workflow's contingency-test job.
