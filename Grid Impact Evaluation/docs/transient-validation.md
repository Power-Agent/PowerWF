# Deterministic transient trajectory comparison

This module provides the engineering-grade comparison layer for PowerWF when
an agent has already obtained transient outputs from two independent simulator
MCPs. It does not replace MCP orchestration or simulator-specific adapters.
It turns returned numerical trajectories into a reproducible, auditable
comparison.

## Scope

- The agent remains responsible for selecting/invoking the two MCPs.
- Each MCP adapter extracts channel samples and supplies explicit channel IDs,
  timestamps in seconds, numerical values, and units.
- `agent.transient_validation` validates, aligns and compares the trajectories.
- The report exposes per-channel metrics and an explicit tolerance verdict.

## Input contract

Provide mappings from stable engineering channel identifiers to
`ChannelSeries(time_s, values, unit, base_id)`. Channel IDs should encode the measured
quantity and equipment identity, e.g. `bus:101:voltage_magnitude`,
`gen:1:rotor_angle`, or `branch:101-205:active_power_from`.

If the two simulators use different names, supply an explicit
`channel_map={reference_id: candidate_id}`. Do not infer that two channels
represent the same physical quantity from similar labels.

Supported unit conversions are intentionally limited to unambiguous pairs:
MW/kW/W, MVA/kVA/VA, kV/V, kA/A, Mvar/kvar/var, rad/deg, and ms/s.
Frequency (Hz), seconds, and per-unit (pu) values are compared only in the
same declared unit. Per-unit bases must be verified by the adapter; this
module does not convert between per-unit bases or infer missing bases.

## Alignment and metrics

The comparison window is the intersection of the two sampled time intervals.
The common grid is the sorted union of both engines' observed timestamps
within that interval plus its endpoints. Error integrals are evaluated exactly
for piecewise-linear error segments, so MAE/RMSE are not biased by different
export sample densities. Linear interpolation is used only
inside each engine's sampled interval. Extrapolation is never performed.
Duplicate/non-monotonic timestamps, non-finite samples, incompatible units,
missing mapped channels and empty overlap fail closed.

For each channel the report includes:
- Time-weighted MAE (mean absolute error)
- Time-weighted RMSE (root mean square error)
- Time-weighted bias (mean candidate-minus-reference error)
- Maximum absolute error
- Reference and candidate peak magnitudes, and their absolute difference
- Optional maximum-error tolerance and pass/fail result

A global `passed` status is reported only when every compared channel has a
tolerance and passes. If only some channels have tolerances, the status is
`partially_evaluated`; if no tolerances are supplied, it is
`not_evaluated`. Numerical similarity is not silently promoted to an
engineering acceptance verdict.

## Example

```python
from agent.transient_validation import ChannelSeries, compare_transient_trajectories

powerworld = {
    "bus:101:voltage_magnitude": ChannelSeries(
        time_s=[0.0, 0.01, 0.02], values=[1.0, 0.98, 0.99],
        unit="pu", base_id="100-MVA-230-kV"
    )
}
psse = {
    "bus:101:voltage_magnitude": ChannelSeries(
        time_s=[0.0, 0.005, 0.02], values=[1.0, 0.99, 0.99],
        unit="pu", base_id="100-MVA-230-kV"
    )
}
report = compare_transient_trajectories(
    powerworld,
    psse,
    reference_engine="PowerWorld",
    candidate_engine="PSS/E",
    tolerances={"bus:101:voltage_magnitude": 0.02},
)
print(report.to_dict())
```

## Tests

From this directory, run:

```bash
python -m pytest tests/test_transient_validation.py -q
```

The tests use synthetic trajectories and an injected model double to validate
both numerical comparison and the LangGraph tool-execution path. They do not
claim a licensed PowerWorld or PSS/E run.
