# Cross-Engine Transient Simulation Workflow

This PowerWF capability lets an agent run one canonical transient-stability
study through two independent simulation MCPs and compare the returned
trajectories through an injected validation service.

## Responsibility split

- **PowerWF:** interprets the user's request, chooses two compatible MCP
  endpoints, constructs a common study definition, orchestrates both runs,
  preserves outcomes, and presents the engineering report.
- **PowerMCP:** provides the deterministic trajectory-comparison capability
  (time alignment, channel/unit checks, error metrics and tolerance evaluation).
- **Simulator MCPs:** execute the actual studies and return their results.

PowerWF does not reimplement simulator APIs or claim that waveform similarity
alone proves system stability equivalence.

## Minimal integration

Each MCP adapter must be an async callable accepting the same canonical study
payload and returning a mapping containing its raw results. The comparator
adapter should call the PowerMCP dynamic-validation API and return its
structured report.

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
    first=EngineEndpoint("psse", psse_mcp_adapter),
    second=EngineEndpoint("powerworld", powerworld_mcp_adapter),
    comparator=powermcp_comparator_adapter,
)
print(report.to_dict())
```

The adapters are deliberately injected: MCP server discovery, authentication,
tool naming and result schemas vary by deployment. The workflow refuses to
compare if either engine fails and reports comparison-service failures
separately. It preserves raw results for traceability.

## Engineering safeguards

- Both endpoints receive identical canonical study definitions (as separate
  copies, so an adapter cannot mutate the other engine's input).
- Distinct engine endpoints are required.
- Failed runs are explicit; no partial run is presented as a successful
  comparison.
- Numerical validation is delegated to PowerMCP; no implicit unit conversion,
  channel equivalence or extrapolation is introduced here.
- The study definition records case, parameters, disturbance and time window.
  Deployments should additionally verify model identity, base values,
  controller/protection settings, solver configuration and event semantics
  before execution.

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
and failure semantics, not licensed simulator execution or the external
PowerMCP integration, which require configured MCP servers and simulator
installations.
