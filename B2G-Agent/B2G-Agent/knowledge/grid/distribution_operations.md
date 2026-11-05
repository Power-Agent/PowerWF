# Distribution-grid knowledge

## Network constraints

The formal case uses 0.95-1.05 p.u. service-voltage bounds, experiment-specific
line thermal ratings, and an experiment-specific upstream transformer rating.
Voltage and line loading are pandapower outputs, not input observations.

## Fifteen-minute services

Peak shaving reduces feeder demand; congestion management removes branch
overload; voltage management uses active-power flexibility and, when explicitly
enabled, inverter reactive power; reserve is a 15-minute capacity and activation
service rather than second-scale frequency regulation.

## Grid-facing commitment

A building response is translated into committed kW, node, interval, delivery
tolerance, confidence, unmet quantity, and the resulting voltage and loading
effects. If delivery is insufficient, the grid operator may redistribute the
remaining requirement.
