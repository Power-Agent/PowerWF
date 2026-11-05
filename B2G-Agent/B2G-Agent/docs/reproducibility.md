# Reproducibility protocol

## LLM layer

Live evaluation freezes the model snapshot, prompts, response schema,
retrieval method, chunking, top-k, deterministic tie-break, temperature 0,
top-p 1, seed 2026 when supported, maximum tokens, and parallel-tool setting.
There is no silent model fallback.

The current configuration uses `gpt-6-luna`, temperature 0, top-p 1, and
`reasoning_effort=none`. `B2G_MAX_OUTPUT_TOKENS=9000` sets
`max_completion_tokens=9000` for **each individual LLM call**. Translator,
Actor, Reviewer, reverse translation and revision calls all receive the same
output ceiling in every repeated task execution. Input tokens and earlier
calls do not reduce this allowance. A complete task may consume more than
9000 tokens across its calls.

`B2G_MAX_TOTAL_TOKENS_PER_TASK` is empty (disabled). No cumulative task limit
or conservative input-budget reservation applies under this protocol. Actual
input, output and total usage are still retained for every response, including
replies that fail parsing. Automatic SDK retries remain disabled so invocation
records are complete. The optional cumulative-budget implementation is used
only if an explicit positive task limit is configured for a different protocol.

The Translator corpus is frozen independently as `case-derived-kb-v3` by the
source registry, per-asset revision/URL/SHA-256 lock, extractor version, output
hashes, and release hash in `knowledge/manifest.json`. Runtime retrieval uses
method `bm25+minilm-dense-embedding+weighted-rrf-v4`: metadata gating
and direction-specific weights are frozen in `knowledge/retrieval_config.json`,
followed by separate source-domain, target-domain, and mapping retrieval.

Reviewer initialization is independently reproducible from
`experiments/build_reviewer_memory.py`. Its manifest locks 150 controlled,
programmatically verified failures from only the development and memory-seed
splits; the test split and numeric ground truth are excluded. Online Reflexion
records require program or expert confirmation, and test-derived records are
write-locked until evaluation closure.

Every request is keyed by SHA-256 over model, settings, prompts, input,
retrieved evidence, and response schema. The full request, raw provider response,
and parsed object are stored immutably under
`outputs/benchmark/<suite>/lr/<worker>/<hash>/i/<invocation-id>/`. Repeated identical requests
therefore never overwrite one another. Paper
reconstruction from unchanged private records uses `B2G_LLM_MODE=replay`; live runs
quantify stochastic stability with at least three repetitions per episode/method.
Public records have declared privacy redactions; use the recorded-results
verification route in the [main README](../../README.md#reproduction) rather than
assuming exact-key public replay.

Translator, Actor, and Reviewer requests are hashed independently. The Actor's
LLM proposal is accepted only after deterministic DAG/minimal-action validation;
the Reviewer's LLM output can only return structured issues and cannot rewrite
the task contract.

The strict JSON schema is normalized by the frozen OpenAI Python SDK so every
property required by Structured Outputs is explicitly required. Each live Reviewer
semantic call returns eight categorical findings with evidence locators. Code
derives the 0/0.5/1 checklist and records both the raw audit and derived assessment
in the journal; the risk threshold is unchanged. The final agent indicators use Translator first drafts, Actor terminal candidates,
and Reviewer feedback outcomes from those same 600 interaction records. Reviewer
appropriate handling includes all attempts; reverse-review reach and final success
use the same 300 source-executable attempts in both methods. The obsolete separate
Reviewer challenge benchmark is no longer an evaluation entry point. See [evaluation protocol](evaluation-protocol.md) for label
limitations, denominators and the all-corpus scope.

Temperature zero improves stability but does not guarantee byte-identical cloud
responses. Locked replay requires unchanged matching request bytes; fresh live
runs measure the remaining stochasticity.

### Offline qualification regression coverage

`tests/fixtures/qualification_reviewer_20261030_second.json` preserves five actual
Reviewer replies from the second qualification attempt, with invocation IDs and
original parsed-object hashes. Their numeric 0.5 values were present in the raw
model replies, not introduced by parsing. Several rationales described resolved
context or pending physical evidence while the scores still withheld dispatch.
The terminal dispatch guard then reported an exception after correction exhaustion.

Tests preserve the original outcomes, reject those legacy responses against the
new live schema, and exercise bounded Reviewer-only repair, valid categorical
findings, true ambiguity, hard conflicts and controlled blocked outcomes using
explicitly synthetic offline clients. They are not new API results and do not
establish improved live task success. Tests prohibit network and real LLM calls.

Earlier 4.1-mini aggregate successes do not establish equivalence to this model
and implementation. The old 450-attempt raw artifacts were removed with user
approval, so a matched historical prompt/response comparison is unavailable.
Qualification journals are preserved; no old failure is relabelled as a success.

### Changing the OpenAI model

The formal runner takes its model and output limit from the frozen
`cases/ieee33_10bem_15min/evaluation/proposed_protocol.json`. Changing the formal
model requires a new protocol ID and matched settings in the preflight `.env`
and method registry. Existing suites are never mixed with a changed profile.
Private UI model settings live independently in `../Personal-workspace/settings.json`
and cannot alter the formal protocol. The current preflight profile is:

```dotenv
B2G_MODEL=gpt-6-luna
B2G_REASONING_EFFORT=none
B2G_MAX_TOTAL_TOKENS_PER_TASK=
B2G_MAX_OUTPUT_TOKENS=9000
B2G_MAX_AGENT_REVISIONS=4
B2G_MAX_SCHEMA_REPAIRS=2
B2G_LLM_MODE=offline
B2G_LLM_RECORD_ROOT=.local-runs/llm_records/gpt-6-luna
```

Run `experiments/preflight_evaluation.py --mode live` before acquisition. The
client uses Chat Completions Structured Outputs, `temperature=0`, `top_p=1`,
seed 2026, `reasoning_effort=none`, and `max_completion_tokens`. A model that
does not support that exact parameter profile requires a preregistered adapter
change and a new experiment series; it must not silently fall back or be mixed
with the frozen series.

The checked-in example and current local configuration remain offline until
live acquisition is authorized. The paper table should describe Maximum Tokens
as "Maximum output tokens per LLM call".
The retrieval embedding remains `sentence-transformers/all-MiniLM-L6-v2`,
384 dimensions, at the revision frozen in `knowledge/retrieval_config.json`.

## Physics layer

Each run manifest records commit/dirty state, operating system, Python,
EnergyPlus, pandapower, pvlib and solver versions, all input hashes, episode,
method, seeds, commands, convergence/severe-error status, and output hashes.
Numeric oracles additionally hash every source file on the deterministic
physics path and store one `physics_implementation_bundle` digest. Evidence is
not reused when either the joint-dispatch configuration hash or this code-bundle
hash changes.

EnergyPlus warning and severe-error counts are retained per BEM and per
baseline/event run. Severe errors fail the run; warnings remain visible in the
coupled result so prototype-conversion issues cannot be silently hidden.

Exact JSON equality is required for replayed TaskSpecification, ExecutionPlan,
ReviewerDecision, and InformationExchange. The task corpus contains only the
initial sender message; receiver feedback and completion are stored in the run
record, never backfilled into the frozen task definition. In the same software
environment, deterministic numeric outputs should match exactly. Cross-machine
acceptance tolerances are:

- voltage: `1e-6 p.u.`;
- power: `1e-3 kW`;
- temperature: `1e-3 °C`.

Wall-clock time is reported but never used as an exact reproducibility target.

The evaluation runner may reuse deterministic physics across repetitions and
methods. Cache identity includes the physics-relevant TaskSpecification
projection, the reviewed ExecutionPlan, the numerical source bundle, all
operational case/local-data inputs, and the EnergyPlus executable binary.
Retrieval citations, latency, and conversational provenance are excluded. The record is accepted only if the evidence file
still exists and its SHA-256 matches. A cache hit does not reuse Translator,
Actor, Reviewer, or same-agent outputs; those remain separate LLM calls and
separate run records. Use `--no-physics-cache` for an independent cold-run
audit.

Long evaluations use `--continue-on-error` to preserve completed work. There
is no selective whole-episode rerun: after the bounded internal corrections
(four domain revisions per loop and two format repairs per malformed response),
a failed episode gets a failed manifest, an error trace,
and an index row. `summarize_evaluation_runs.py` aggregates method-level scores,
within-episode repeat variability, failure rates, per-agent token/latency use,
retrieval time, and cache statistics to both JSON and CSV.

## Ground-truth layer

Every public task ID maps one-to-one to a private record in
`groundtruth/contract_oracles.jsonl`; source-message hashes prevent silent input
drift. A contract is marked `programmatically_verified` only when deterministic
checks confirm its source hash, schema alignment, family-to-output mapping,
service direction, time-window semantics, and execution route. The status and a
semantic fingerprint are stored separately in
`groundtruth/contract_verification.jsonl`, so verification metadata is not part
of the physics implementation digest. This status is not described as expert
truth. Likewise, a direct-task numerical oracle is not
valid until the joint RC/OPF decision, EnergyPlus rerun, and pandapower checks
converge from frozen inputs, all required outputs are defined, and the complete
provenance record is stored under `groundtruth/numeric_oracles/`.

The contract reference is derived from the visible natural-language message.
Generator-only pre-rounding values are forbidden as labels. Run
`experiments/audit_contract_translation.py` to compare all 200 translations
against their separate oracles without providing hidden episode semantics.

Every standard oracle contains both no-response and response nonlinear AC
replays. This makes feeder-peak reduction, rebound, voltage improvement, and
congestion improvement reproducible counterfactual comparisons rather than
response-only values. Renewable absorption uses the frozen request as the
exogenous energy that would otherwise be curtailed; the reported residual is
explicitly request-relative and is not presented as curtailment inferred from
an unconstrained PV power flow. Market tasks use the committed pay-as-bid rule;
the ISO-NE real-time LMP is retained as context but does not gate acceptance.

Baseline-flexibility oracles run reduction and absorption as two independently
validated closed loops. Bounds describe BEM grid-interface net demand; a
negative lower bound denotes storage export, not negative building demand.

DOE prototype zones do not share one universal occupied/unoccupied temperature
band. The nominal bands are therefore screening bands. Effective time-varying
hard bounds contain the corresponding no-response EnergyPlus baseline extreme
plus the frozen tolerance, and the validator reports both absolute nominal-band
departures and response-added violations. Run
`experiments/audit_comfort_baselines.py` to audit this distinction over all ten
target days.

`experiments/validate_case.py` checks corpus alignment by default. Its
`--require-publishable` gate additionally fails while any contract or numerical
oracle remains pending programmatic verification.
It also cross-checks the task summary, ground-truth manifest, current contract
hash, verification counts, numeric-oracle hashes, stable input hashes, and the
physics implementation bundle. Corpus regeneration is blocked while verified
numeric statuses exist unless `--reset-verified-physics` is explicitly given.

## Public release

Published records have explicit privacy redactions and original/published hash pairs. Byte-identical replay from these public copies is not guaranteed. Use the [main README's recorded-study verification workflow](../../README.md#reproduction) and consult [publication notes](publication.md). The author retains original acquisition files separately.
