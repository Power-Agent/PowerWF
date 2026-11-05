# Cross-domain mappings

## Power sign and units

Positive building load means consumption from the grid. Positive PV generation
and battery discharge reduce net load. Building outputs are kW/kvar and are
converted to MW/Mvar only at the pandapower boundary. Every interval is 15
minutes and every energy integral multiplies power by 0.25 hours.

## Common contract fields

Every executable request identifies sender and receiver roles, task family,
operating day, time window, target BEMs and buses, objective, priority,
constraints, units, evidence, and whether each field is declared, inferred,
missing, to be simulated, resolved, or unresolved.

## Clarification gate

Missing execution-critical targets, ambiguous units, conflicting priorities, or
unidentified nodes trigger a clarification. Only physically computable outputs
are sent to the Actor. The Translator never invents values that the simulator
has not produced.
