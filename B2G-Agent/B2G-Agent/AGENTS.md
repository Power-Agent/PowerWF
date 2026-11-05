# AGENTS.md

B2G-Agent implements the formal Translator–Actor–Reviewer methodology for one
IEEE 33-bus, 10-BEM, 15-minute research case.

## Boundaries

- Reusable implementation belongs under `src/b2g_agent/`.
- Formal case content belongs under `cases/`; it must not import agent code.
- Experiment generation and validation entry points belong under `experiments/`.
- Stable domain knowledge belongs under `knowledge/`; Reviewer episodic memory
  is separate under the case directory.
- Large/raw inputs, transitioned local models, and machine paths belong under
  `.local-data/` or an external `B2G_DATA_ROOT`, never in committed source.
- Generated run artifacts belong under ignored `outputs/`.
- The five selected formal cohorts are Proposed, Baseline, Translator v2, Actor v2 and Reviewer v4. Each covers 200 tasks x 3 repeats. Their only visible indicator set is the final 18-row matrix in `outputs/comparison/final/metrics.csv`. Baseline has nine applicable values and each targeted ablation has twelve.
- Acquired outputs belong under exactly five `outputs/benchmark/` directories. Each ablation has one method directory with `first50/`, `remaining150/`, full-200 `interaction_records/` and `FINAL_METRICS.*`. Translator additionally has `attempt_history/` for seven superseded transport/host failures; the effective 450 slots are in `remaining150/`.
- Formal method registration is `cases/ieee33_10bem_15min/evaluation/methods.json`. Only `full_multi_agent`, `same_llm_single_agent`, `plain_translator`, `unstructured_actor`, and `reviewer_no_feedback` are runnable.
- The three ablations each change only the named agent implementation. Reviewer v4 retains full review and memory but disables corrective feedback dispatch. The author retains original acquisition bytes privately. Public records and captured archives have declared path/credential redactions; publication manifests pair original and published hashes. `experiments/retained_paths.py` resolves moved paths when reading them.
- Quality uses `interaction-quality-f085-v2`: F >= 0.85, A = 1, V = 1, G >= 0.9. Translator uses first draft, Actor terminal candidate. The final Reviewer indicators measure appropriate first handling over 600 attempts, reverse review reach over 300 source-executable attempts, and final success over the same 300. They were chosen after acquisition; consult the final comparison for interpretation.
- Final metric reports are produced offline by `experiments/finalize_selected_reports.py`; no API or physics rerun is needed. `experiments/verify_retained_layout.py` checks coverage, archives and journal hashes. Historical hash claims must not be silently rewritten to conceal source changes; publication redactions must be explicitly paired with their new hashes.
- UI sessions are private rehearsals, including UI benchmark mode. Their
  settings, records and learned memory belong in sibling `Personal-workspace/`
  (or explicitly ignored `.personal/`), never in formal run or case folders.
- Do not add demo, sample-scenario, or surrogate-physics results.

## Evidence rules

- EnergyPlus is invoked through its official CLI and pandapower through its
  Python API.
- An LLM may translate intent or propose a schema object, but deterministic code
  validates units, dependencies, time resolution, node mapping, convergence,
  and numerical outputs.
- Never substitute an LLM value for failed or unavailable physics.
- Label experiment-specific line/transformer limits as such; do not attribute
  them to the original Baran-Wu network.
- A simulation counterfactual is not a settlement-grade baseline.

## Reproducibility

- Formal intervals are exactly 15 minutes (96 per day).
- LLM live runs use temperature 0, top-p 1, seed 2026 when supported, a pinned
  model snapshot, strict structured output, and no silent model fallback.
  Free-Plan deliberately uses text generation without an API JSON Schema. Its
  adapter must not synthesize omitted actions, dependencies, parameters or outputs;
  ambiguous or incomplete plans go through the existing correction loop.
- Record full LLM request/response/parsed artifacts by content hash. Locked
  replay is the default exact reconstruction path for paper results.
- Every physics run records versions, input hashes, commands without absolute
  secret paths, convergence/error status, and output hashes.
- Unit tests must not require an API key or workstation-specific simulators.

## Completion checks

- `episodes.jsonl` validates as 200 unique episodes: 150 single-family and 50
  composite, with development/memory/test splits of 30/20/150.
- `groundtruth/contract_oracles.jsonl` validates one-to-one against task IDs and
  source-message hashes; pending expert or physics status is never presented as
  verified.
- The local web UI exposes case selection, the three-agent trace, and final
  report without exposing ground truth or API secrets. Benchmark mode enforces
  one exchange, while interactive mode supports free-form, multi-turn use.
- `pytest` passes and no populated `.env`, raw data, or local absolute path is
  tracked.
