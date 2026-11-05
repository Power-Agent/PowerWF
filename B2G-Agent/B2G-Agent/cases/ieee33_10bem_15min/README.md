# IEEE 33-bus / 10-BEM / 15-minute formal case

This directory contains case definitions and small, reviewable research
artifacts. It deliberately does not contain agent implementation code or
machine-specific paths.

The committed case is sufficient to inspect and reproduce contract-level
evaluation, including the complete 200-episode corpus. Large/raw simulator
assets are acquired into the ignored `.local-data/ieee33_10bem_15min/`
directory by `python experiments/acquire_case_assets.py`. A future public
release can replace the local cache with a DOI-addressed archive without
changing case IDs or agent code.

## Fixed design choices

- Network: `pandapower.networks.case33bw()` with 32 radial branches in service
  and five normally open ties. The source network has no defensible physical
  current ratings, so all study ratings are labelled experiment-specific.
- BEM buses: 3, 5, 8, 11, 15, 18, 21, 27, 29, and 33, selected at nearest-rank
  deciles of the empirical cumulative-path-resistance distribution from the slack bus.
- Buildings: exactly one representative DOE/PNNL ASHRAE 90.1-2019
  climate-zone 5A IDF is assigned to each of the ten node-level BEMs; there is
  no multi-IDF building portfolio inside a BEM. The ten instances represent
  seven unique archetypes. BEM 03 and BEM 04 are
  byte-identical runtime replicas of the independently scaled Small Office and
  Medium Office archetypes. BEM 09 is likewise a separately scaled Stand-alone
  Retail instance at bus 29. The original school candidates, restaurant
  replacements, and annual Outpatient Healthcare candidate produced
  `CheckAirLoopFlowBalance` Severe records after official 22.1-to-23.1
  transition, so they are excluded rather than silently accepted or manually
  altering their HVAC systems. BEM 06 uses Strip Mall because the official
  prototype family has no supermarket model.
- Runtime: source IDFs are EnergyPlus 22.1 and are transitioned locally to
  EnergyPlus 23.1. Every prepared model is patched to four timesteps per hour.
  Physics readiness requires all ten instances to complete with zero Severe
  records; warning counts remain in the manifest.
- Time: all coupling and task windows use 96 intervals per day.
- Portfolio scaling: the single representative IDF's electrical power is
  multiplied by a fixed aggregation factor so that BEM peak is 40% and
  nonparticipating background peak is 60% of the original case33bw load at a
  mapped node. Temperatures and comfort metrics are never multiplied by that
  factor.

## Task corpus

The authoritative corpus is `tasks/episodes.jsonl`. Each line freezes exactly
one sender-authored natural-language message. It does not contain a prewritten
receiver reply. During a run, that message produces two communications and four
recorded information states: sender-sent, receiver-received after agent
translation, receiver-sent after domain assessment, and sender-received after
agent translation. The final state quantifies task completion.

Use `tasks/sender_messages.jsonl` to inspect the 200 starting instructions
without the internal experimental labels. Each line contains only the episode
identifier, sender role, and sender's natural-language message.

`tasks/summary.json` contains distribution checks, while `tasks/splits/` makes
the leakage-safe development, memory-seed, and test membership explicit. Exact
physical groups do not cross splits; the summary separately discloses
structural-template overlap.

`groundtruth/contract_oracles.jsonl` is the separate, private answer-key layer.
It has a source hash and perfect-execution contract for each of the 200 task
IDs. Contract labels are called `programmatically_verified` only after
deterministic schema, direction, window, source-hash, and output checks pass;
the status is stored in `groundtruth/contract_verification.jsonl` and does not
claim expert review. Direct-task numerical labels remain pending
until the frozen RC/OPF, EnergyPlus, and pandapower workflow succeeds. See
`groundtruth/README.md`; neither the task catalog nor the web API exposes these
answers.
