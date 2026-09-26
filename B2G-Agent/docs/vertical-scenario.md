# B2G-Agent Research Scenarios

The two deliberately constrained scenarios validate professional mediation before external physics stacks are connected. Both use the same four-stage interface, two engineering roles, typed case state, LLM mediation, evidence execution, decision ledger, and final review.

## Scenario 1: Distribution Grid Upgrade

A coastal community plans to add or renovate 80 homes on a radial feeder.

### Building-side decisions

- cooling setpoint;
- participating building count;
- retrofit level;
- rooftop PV per building;
- peak demand-response percentage.

### Grid-side decisions

- connection bus;
- transformer capacity;
- feeder line-capacity proxy.

### Acceptance checks

- minimum voltage at least 0.95 p.u.;
- line loading no greater than 100%;
- transformer loading no greater than 100%;
- comfort and indicative cost remain visible for human review.

## Scenario 2: Demand Response Service

A civic building is preparing to enroll in a utility demand-response service. The central boundary object is no longer only building load: it is the relationship among the counterfactual baseline, actual event-day load, delivered reduction, and rebound.

### Building-side responsibilities

- explain normal operating schedules and weather sensitivity;
- choose or challenge the baseline method;
- identify HVAC and other controllable loads;
- defend comfort constraints;
- estimate a repeatable reduction and acceptable rebound.

### Grid-side responsibilities

- reject an inflated or poorly supported baseline;
- define event timing and needed kW relief;
- verify delivery against the commitment;
- evaluate feeder value and operating limits;
- determine whether the service is reliable enough for enrollment.

### Shared parameters

| Parameter | Meaning | Current range |
|---|---|---|
| `baseline_method` | Counterfactual estimation method | recent 10-day average, weather-adjusted, or matched day |
| `baseline_adjustment_pct` | Explicit adjustment to the estimated baseline | -15% to +15% |
| `building_count` | Number of participating assets | 1 to 300 |
| `cooling_setpoint_c` | Event-day comfort-control input | 20-28 °C |
| `dr_event_start_hour` | Local event start | hour 0-23 |
| `dr_event_duration_hours` | Event duration | 1-6 hours |
| `dr_target_kw_per_building` | Committed reduction | 0-5 kW per building |
| `max_rebound_pct` | Maximum post-event rebound | 0-50% of target |
| `target_bus` | Feeder connection | bus 3-8 |

### Evidence

- baseline peak per building;
- baseline confidence score;
- event baseline energy;
- average delivered kW per building;
- percentage of commitment delivered;
- post-event rebound kW and percentage;
- voltage, feeder loading, and transformer loading.

### Acceptance checks

- baseline confidence at least 80/100;
- delivered reduction at least 90% of commitment;
- rebound no greater than the agreed limit;
- no voltage or thermal violation.

The equations are intentionally inspectable and deterministic. They are not a tariff-specific settlement calculation and do not use measured interval data.

## Promotion Path

1. Validate EnergyPlus-MCP tools for schedules, loads, thermostats, outputs, and event control.
2. Define a versioned building-load contract for baseline and event-day results.
3. Validate PowerMCP/OpenDSS tools for feeder compilation, time-series execution, and equipment changes.
4. Add real interval-meter baseline methods and program-specific settlement rules.
5. Preserve model diffs, tool versions, convergence status, units, and run provenance.
6. Compare deterministic research results with known EnergyPlus/OpenDSS cases before claiming engineering validity.
