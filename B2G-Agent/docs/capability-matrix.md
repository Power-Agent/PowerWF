# Capability And Tool Matrix

The table distinguishes current executable behavior from future external-tool integrations.

| Capability | Current implementation | EnergyPlus-MCP | PowerMCP / OpenDSS | Status |
|---|---|---|---|---|
| Interpret free-form engineering input | OpenAI API through the official Python SDK | Not involved | Not involved | Implemented |
| Validate proposed changes | Pydantic schema plus scenario allowlist | Not involved | Not involved | Implemented |
| Residential 24-hour load response | Deterministic research equations | Planned calibrated replacement | Not involved | Research surrogate |
| Feeder voltage and loading | Deterministic radial-feeder proxy | Not involved | Planned calibrated replacement | Research surrogate |
| Cooling setpoint and retrofit | Deterministic multipliers | Planned schedule/HVAC/model changes | Not involved | Research surrogate |
| Demand-response baseline | Recent-average, weather-adjusted, or matched-day research formulation | Planned building-result source | Not involved | Research surrogate; not settlement-grade |
| DR event load reduction | Target, curtailable-load limit, delivery, and rebound equations | Planned schedule/control execution | Planned feeder-impact validation | Research surrogate |
| Connection and capacity changes | Validated shared parameters | Not involved | Planned topology/equipment edits | Research surrogate |
| Scenario report | Local JSON and Markdown artifacts | Future provenance source | Future provenance source | Implemented |

## Explicit Non-Use Statement

The current release does **not** install, import, start, or call EnergyPlus-MCP, PowerMCP, EnergyPlus, or OpenDSS. Those projects are cited as intended integration targets. No result in the current UI should be described as an EnergyPlus or OpenDSS result.

## Acceptance Rule For A Future External Capability

A capability is not production-ready until it has:

1. a typed input schema and explicit units;
2. a verified MCP tool or deterministic adapter;
3. pre-execution validation and human confirmation policy;
4. a recorded input/model diff;
5. simulator success and convergence checks;
6. result extraction with units and provenance;
7. regression tests against a known case;
8. a user-facing statement of limitations.
