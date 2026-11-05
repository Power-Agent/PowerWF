# Verified numerical oracles

One file named `ep-NNN.json` will be written here for every direct task only
after all of the following use frozen inputs and succeed:

1. baseline EnergyPlus simulation for the single representative IDF assigned
   to each participating BEM, followed by electrical-power-only portfolio
   scaling;
2. RC-model calibration with recorded parameters and fit errors;
3. joint RC/OPF dispatch optimization with solver status and objective value;
4. event EnergyPlus rerun with comfort and rebound validation;
5. 15-minute building-to-bus aggregation and pandapower validation;
6. deterministic metric extraction and receiver decision/completion calculation.

Each future file must include input hashes, software versions, solver settings,
random seeds, convergence status, the optimal decision variables, requested
metrics, constraint residuals, receiver feedback, sender-facing final message,
and quantified completion. A failed or non-converged run is recorded as a run
artifact, not promoted to ground truth.

This directory intentionally contains no fabricated numerical results while the
formal case reports `physics_truth_status: pending`.
