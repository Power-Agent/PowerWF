# Building-domain knowledge

## Baseline and flexibility

A Building Energy Manager reports an explicit counterfactual baseline, upward
and downward power flexibility, an availability window, ramp and duration
limits, comfort constraints, and rebound bounds. A simulation counterfactual
must not be described as a settlement-grade market baseline.

## Comfort and aggregation

Portfolio aggregation scales electric power and energy, not indoor temperature
or comfort thresholds. Comfort is evaluated in each prototype and aggregated as
violating minutes, affected-zone share, and worst-case temperature.

## Acceptance decision

The building side may accept, reject, or counteroffer. A counteroffer states a
deliverable kW trajectory, availability, confidence, comfort risk, rebound, and
price where applicable. Device-level HVAC schedules remain Actor evidence and
are not presented as a grid-side command.
