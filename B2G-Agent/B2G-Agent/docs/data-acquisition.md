# Data acquisition and provenance

The repository carries small definitions and hashes. Large inputs and complete
recorded outputs are distributed as checksummed packages in the Google Drive
release linked from the [main README](../../README.md#data). Restore the frozen input
package before attempting new runs; the scripts below document its preparation.

## Network

`experiments/prepare_network.py` exports `pandapower.networks.case33bw()` and
recomputes electrical-distance bus selection. The script fails if library
changes alter the selected buses or basic network counts.

## Buildings and weather

`experiments/acquire_case_assets.py` obtains ten ASHRAE 90.1-2019 commercial
prototype archives and the Buffalo TMY3 EPW from DOE. Source IDFs are version
22.1; local EnergyPlus transitions produce version 23.1 IDFs. Runtime models
are patched to four timesteps per hour and to a Wednesday January 1 so their
weekday/weekend schedules align with the case's nominal year 2025. The local
manifest records archive/member/source/runtime hashes and transition logs.

Run `python experiments/smoke_test_energyplus.py` outside a restricted sandbox
before physics evaluation. It design-day tests all ten runtime IDFs and writes
an ignored local report. A zero process exit code is insufficient: every BEM
must also contain zero EnergyPlus Severe records for physics readiness.

DOE's commercial prototype family has no supermarket model. The case therefore
uses Strip Mall for BEM 06 so all ten IDFs share one official source, standard,
software version, and climate zone.

The final ten node-level model instances contain seven unique archetypes.
BEM 03 and BEM 04 replicate the transitioned Small Office and Medium Office
runtime IDFs and receive independent portfolio scaling at buses 8 and 11.
BEM 09 is an independently scaled runtime replica of Stand-alone Retail at bus
29. This is a deliberate quality-control choice: both official school
prototypes, the tested restaurant replacements, and the Outpatient Healthcare
prototype produced nonzero `CheckAirLoopFlowBalance` Severe records under
EnergyPlus 23.1. The outpatient failure occurred only in the annual run, which
also demonstrates why the design-day smoke test is necessary but not
sufficient. Failed reports remain local evidence; no upstream HVAC object was
hand-edited to hide an incompatibility.

## Frozen full-year profile sources

- Nonparticipating node load: `experiments/prepare_background_profiles.py`
  downloads all 14 New York ComStock 2021 TMY3 class aggregates and the New
  York ResStock `Multi-Family with 5+ Units` aggregate from the public OEDI
  bucket. It checks all 35,040 source intervals, matches the ten BEM buses to
  corresponding classes, assigns a frozen heterogeneous mix to the other
  buses, and records every URL, class, allocation, byte size, and SHA-256 in
  `.local-data/ieee33_10bem_15min/profiles/background_load_manifest.json`.
  The source timestamps are interval-ending local standard time; the case file
  is explicitly interval-starting in nominal year 2025. This is a TMY shape,
  not observed 2018 demand. Methodology is documented by Wilson et al. (2022),
  DOI `10.2172/1854582`; the dataset DOI is `10.25984/1876417`.
- PV: frozen local pvlib ModelChain configuration driven by the same EPW.
- Storage: explicit synthetic assumptions in `der/storage.json`.
- Prices: the official ISO-NE `2025_smd_hourly.xlsx` annual workbook and the
  NEMA/Boston load zone are frozen in `source_registry.json`. Each final hourly
  RT LMP is repeated for four 15-minute intervals. The result is labelled
  `hourly-expanded`; it is not misrepresented as 5-minute market data. Price is
  exogenous and is not presented as causally paired with TMY weather.
- Reserve: the official ISO-NE annual `5min_reserve_price_and_designation_2025.xlsx`
  archive supplies TMSR, TMNSR, and TMOR fields for reserve zone 7002
  (NEMA/Boston). `experiments/prepare_market_profiles.py` averages each three
  consecutive five-minute price and designation values to 15 minutes. The
  output deliberately calls `desg_mw` a *designation*, not a requirement.

For both market workbooks the spring DST gap is interpolated from the adjacent
same-minute values and the repeated fall hour is averaged. The exact affected
timestamps and both source/derived hashes are written to `market_manifest.json`.

The machine-readable registry is
`cases/ieee33_10bem_15min/sources/source_registry.json`, and citable papers and
datasets are collected in `cases/ieee33_10bem_15min/sources/references.bib`.

Until these files and the derived operational limits exist, the API reports
`ready_for_physics=false`; no surrogate profile is substituted.

## RC screening models

`experiments/identify_rc_models.py` fits a seasonal scalar ARX approximation
of the paper's RC equation from the frozen EnergyPlus facility-electricity and
mean-zone-temperature series. Coefficients use development dates only and are
checked on the disjoint reference dates. The output is local-only at
`.local-data/ieee33_10bem_15min/models/rc_parameters.json` and binds all inputs
by SHA-256. Facility electricity is an observational, confounded proxy for HVAC
control, so these models can screen and optimize candidate schedules but do not
replace the mandatory EnergyPlus comfort oracle.

`experiments/prepare_grid_sensitivities.py` evaluates a +10 kW load-reduction
perturbation at every BEM bus and every interval of the ten frozen target days.
It stores deterministic NumPy arrays for the linearized bus-voltage, line-load,
and upstream-transformer constraints used by RC--OPF. The perturbation retains
the original node Q/P ratio. This is an optimization approximation only;
nonlinear pandapower validation remains mandatory.
