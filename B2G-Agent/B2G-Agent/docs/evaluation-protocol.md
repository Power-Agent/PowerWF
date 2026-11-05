# Current evaluation and private-workspace protocol

## Frozen configuration

The authoritative configuration is
`cases/ieee33_10bem_15min/evaluation/proposed_protocol.json`, protocol ID
`proposed-luna-18-v1`. The proposed method and same-LLM baseline each evaluate
**200 tasks x 3 repetitions = 600 attempts**. Each of the three final ablations
also has 600 acquired attempts, with separate frozen protocols, formal entry points,
isolated outputs, and 12-metric reports (quality 6 + corresponding agent 3 + cost 3).
See [ablation evaluation](ablation-evaluation.md) for switches and offline commands.

| Parameter | Value |
| --- | --- |
| LLM | gpt-6-luna |
| Temperature / top-p | 0 / 1.0 |
| Maximum output | 9000 tokens per individual LLM call, including repairs |
| Cumulative task token cap | Disabled |
| Reasoning effort / seed | none / 2026 when supported |
| Embedding model / dimension | all-MiniLM-L6-v2 / 384 |
| Embedding revision | 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 |
| Internal revisions / format repairs | 4 per correction loop / 2 |
| External benchmark exchange | One sender message and one receiver feedback |
| Formal online memory | Disabled; current source episode excluded from seed retrieval |

The three repetitions use the same settings and separate invocation records.
Temperature zero does not guarantee identical live responses; raw replies and
locked replay provide an exact record. Every failed scheduled attempt is retained.
The 200 original split labels remain for provenance. Pooling development and
memory-source tasks yields a fixed-corpus descriptive benchmark, not an untouched
holdout. Three repetitions are not three independent scenarios.

## Exactly 18 paper metrics

| Layer | Symbol | Definition |
| --- | --- | --- |
| Interaction quality | $Q^{\mathrm{qua}}$ | Mean quality score, formerly Q; range 0-100 |
| Interaction quality | $Q^{\mathrm{tsr}}$ | Successful attempts / all attempts, formerly Y and TSR |
| Interaction quality | $Q^{\mathrm{F}}$ | Semantic fidelity, formerly F |
| Interaction quality | $Q^{\mathrm{A}}$ | Actionability and closed-loop completeness, formerly A |
| Interaction quality | $Q^{\mathrm{V}}$ | Physical safety/feasibility, formerly V |
| Interaction quality | $Q^{\mathrm{G}}$ | Goal achievement relative to the oracle boundary, formerly G |
| Agent performance | $F^{\mathrm{tra}}_1$ | Required semantic-field weighted F1 |
| Agent performance | $F^{\mathrm{tra}}_2$ | Cross-domain mapping accuracy |
| Agent performance | $F^{\mathrm{tra}}_3$ | Clarification-decision F1 |
| Agent performance | $F^{\mathrm{act}}_1$ | Terminal plan legality after closed-loop revisions |
| Agent performance | $F^{\mathrm{act}}_2$ | Terminal tool-chain execution success after closed-loop revisions |
| Agent performance | $F^{\mathrm{act}}_3$ | Terminal numerical evidence and hard-label consistency |
| Agent performance | $F^{\mathrm{rew}}_1$ | Appropriate first-review handling over all 600 attempts |
| Agent performance | $F^{\mathrm{rew}}_2$ | Reverse-review reach over 300 source-executable attempts |
| Agent performance | $F^{\mathrm{rew}}_3$ | Final task success over the same 300 source-executable attempts |
| Computation cost | Total token consumption | Whole-framework input + output tokens across all 600 attempts |
| Computation cost | LLM inference latency | Mean per attempt of accumulated whole-framework API request time |
| Computation cost | Retrieval time | Mean per attempt of accumulated whole-framework retrieval time |

Cost includes Translator forward/reverse calls, Actor, Reviewer, all corrections
and failed calls with recorded usage. The three cost measures apply to the entire
framework, not to individual agents. Reports also show their totals, per-attempt
means and measurement coverage. Raw role-specific usage remains available for
traceability; it is not an additional paper metric. API time is measured at the
client request boundary, including network/service time, excluding local parsing,
file writes and physics. Summed parallel-call time is not batch wall time.
Retrieval timers cover Translator knowledge retrieval and Reviewer memory lookup.
Missing timing/usage is unavailable, never silently converted into a measured zero.

The current scoring protocol is `interaction-quality-f085-v2`, configured in
`cases/ieee33_10bem_15min/evaluation/scoring.json`. Success requires
F >= 0.85, A = 1, V = 1 and G >= 0.9, with
Q = 25(F+A+V+G). Unscored failures receive an explicit zero aggregation penalty;
this does not imply their missing components were observed. Numeric ground truth,
task inputs and the deterministic simulation kernel are unchanged.
This user-approved threshold revision was adopted after acquisition and applied
offline to both complete cohorts; it is not a newly run or pre-specified experiment.
The selected full-200 result is published in `outputs/comparison/final/`. Acquired worker scores and journal-embedded scores are historical evidence; current scores live in each suite's `current_scores/` folder.

Translator uses its first forward contract; corrected drafts do not replace that
initial sample. Mapping requires object, variable, direction, unit and time scale
to agree together. Reviewer starts from the first forward review and follows the
recorded downstream handling and final outcome, as defined below. Actor uses the
final result/candidate in the last
`attempt.completed` or `attempt.failed` journal event after closed-loop revisions.
It never selects the best intermediate revision or falls back to an earlier
success when the terminal result is absent. Denominators may count tasks, fields
or drafts rather than always equal 600.

For Actor, let P be all scheduled runs whose oracle requires physical execution
(100 episodes x 3 = 300 in this cohort), and O_i the required numerical fields:

- F_act_1 = valid terminal plans / |P|. Validate each terminal plan against its
  own terminal specification, retaining the existing DAG/tool/input/output checks.
- F_act_2 = terminal runs with a valid plan, all required actions completed,
  matching execution-plan hash and all oracle-required outputs present / |P|.
- F_act_3 = sum of correct required numerical fields and matching hard-constraint
  verdicts / sum_i (|O_i| + 1). Existing numeric tolerances remain unchanged.

Missing terminal plans or evidence remain in the applicable denominator and
receive zero credit. Corrected outputs are eligible; an earlier format repair,
Reviewer return or infrastructure failure does not itself invalidate the terminal
snapshot. A failed attempt's recorded terminal candidate is eligible for these
module diagnostics. Blocked candidates can have valid plans and physical evidence
without being released: interaction-quality success remains unchanged and separate.
Missing numeric references make F_act_3 unavailable, not perfect.

The Actor terminal sampling rule was applied offline to preserved interactions.
The selected Reviewer indicators were subsequently defined to measure the
consequences of removing corrective feedback, using the same 600 runs and the
independent source-executable subset of 300 runs:

- F_rew_1: appropriate first-review handling / 600 scheduled runs.
- F_rew_2: executable runs reaching reverse review / 300 source-executable runs.
- F_rew_3: executable runs satisfying the unchanged task-success condition / the same 300 runs.

The first measure uses the recorded first forward Reviewer input and an
independent structured reference. For an internal correctable error it requires
a later clean forward draft; for a clean input it requires appropriate routing
and reverse review; for a source-required clarification it requires the
appropriate clarification route and original task success. Missing review gets
no credit. These measures are workflow effects, not isolated diagnostic ability.
F_rew_3 overlaps conceptually with overall TSR, and all three Reviewer indicators
were chosen after acquisition. See `outputs/analysis/reviewer-feedback/COMPARISON.md`
for the precise audit. The other indicators and quality/cost arithmetic are unchanged.

## Formal records

Prepare without API calls or physics:

```powershell
.\.venv\Scripts\python.exe -B experiments\run_proposed_suite.py
```

Add `--execute` to perform live inference and simulation. The parent runner freezes
worker settings; personal configuration cannot override them. A completed suite
is postprocessed without further API calls. Reports reject protocol/settings
mismatches. Each `outputs/benchmark/<method>/` owns:

- `suite_manifest.json`, frozen source/configuration archive and hashes;
- `lr/<worker>/`: original requests, provider responses, parsed replies and manifests;
- `w<worker>/<run>/interactions/`: live-flushed events for each task and repetition;
- worker results/scores/errors, including every failed or withheld attempt;
- `outputs/physics/cache/<owner-hash>/`, recorded in the manifest: only this suite's
  deterministic physics cache and evidence, using compact Windows paths;
- `interaction_records/INDEX.md`: readable transcripts and links to all raw calls;
- `FINAL_METRICS.csv`, `FINAL_METRICS.json` for the selected indicator version;
- full 600-run interaction records, with the final five-method matrix in `outputs/comparison/final/`.

For each acquired ablation, phase-specific manifests, captured archives, workers
and API records live under `first50/` and `remaining150/` inside its single method
directory; consolidated interaction records and final metrics live at its top level.

No older-model physics cache or personal run is imported. A cache hit within the
current suite reuses deterministic physics only, never an LLM response. All four
message states, internal revisions, tool inputs/outputs and failures are retained
when generated. Absent messages are marked absent rather than reconstructed.
Raw API replies are journaled before separate response-file writes. Windows
sharing-lock retries affect only local writes; persistent errors still fail closed.
A verified zero-service rejection must explain its failed constraints and release
no unsafe schedule. Its scoring causes must agree with the private oracle.

## Personal interface

The web interface uses the sibling `../Personal-workspace/` by default.
`B2G_PERSONAL_ROOT` can select another private directory outside the repository or
under ignored `.personal/`; formal case/data/results folders are rejected.

- `settings.json`: independent UI settings, initially `llm_mode: offline`;
- `sessions/<id>/session.json`: arbitrary natural-language conversations and results;
- `sessions/<id>/physics/t<turn>/`: per-turn local physical runs;
- `lr/`, `interactions/`: private original API records and stage journals;
- `memory/verified_online.jsonl`: private confirmed failures.

From the repository root, start the UI using
`.venv/Scripts/python.exe -B start_personal.py --host 127.0.0.1 --port 8000`.
This launcher resolves the checkout's current source location after folder renames.
When intentionally using the API in the future, set only the private settings'
`llm_mode` to `live` and restart the UI. The API key remains in the untracked main
`.env`; it is not copied into session data. UI single-round rehearsals are private
as well. The UI does not start or populate the formal 600-attempt evaluation.

Personal settings never modify process-wide formal configuration. Frozen data and
knowledge are shared read-only, while outputs and learned memory are private.
Formal Reviewer writes are explicitly disabled. The personal folder is outside
the project and contains its own ignore-all `.gitignore`. The public repository
includes only a clean template; generated sessions and private settings are excluded.

## Retention and offline verification

The authorized legacy output/LLM/cache/pilot directories were removed. Frozen
case inputs, ground truth, knowledge, embedding assets and necessary source data
were retained. Only future runs under the current protocol enter the formal
result location. Temporary test outputs and validation files are removed after
checking. Versioned regression fixtures remain under `tests/fixtures/`; they are
never presented as case results or supplied to runtime knowledge or memory.

This preparation performs no LLM API acquisition or formal case simulation.
Offline tests block external networking and use controlled responses. Run the
full test suite and `experiments/preflight_evaluation.py --mode live` for checks;
the latter validates live-run readiness but makes no LLM API call.

Offline acceptance on 2026-10-29: **160 tests passed**; preflight **READY** with
zero errors/warnings and `llm_api_called=false`. Preparation selected exactly
600 attempts and 18 metrics with `api_calls=0` and `physics_runs=0`.

Offline repair acceptance on 2026-10-30: **193 tests passed**, including captured
Reviewer response validation, uncertainty thresholds, clarification adjudication,
forward/physical/reverse revision history, public constraint provenance, and
unchanged physical inputs when only provenance changes. Tests block external
networking and live LLM calls. This repair did not rerun the qualification or
formal suite; the earlier qualification records remain unchanged. Live behavior
of the revised prompts still requires a separately authorized qualification.
