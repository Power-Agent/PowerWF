# Five retained formal experiments

| Method | Method ID | Directory below outputs/benchmark | Tasks x repeats |
| --- | --- | --- | ---: |
| Proposed | `full_multi_agent` | `proposed/` | 200 x 3 |
| Baseline | `same_llm_single_agent` | `baseline/` | 200 x 3 |
| Translator v2 | `plain_translator` | `ablation-translator/` | 200 x 3 |
| Actor v2 | `unstructured_actor` | `ablation-actor/` | 200 x 3 |
| Reviewer v4 | `reviewer_no_feedback` | `ablation-reviewer/` | 200 x 3 |

Each ablation has one method directory. `first50/` contains the initial 150 attempts; `remaining150/` contains the next 450 effective attempts. The method root holds all 600 readable interactions and `FINAL_METRICS.*`. Translator retains seven superseded infrastructure failures under `attempt_history/`. Baseline recovery provenance is under `acquisition_history/`.

The final matrix is `outputs/comparison/final/metrics.csv`: 18 indicator names, with 18 applicable Proposed values, nine baseline values, and 12 for each targeted ablation. Failed attempts remain in applicable denominators. Translator uses first drafts and Actor uses terminal candidates. Reviewer uses appropriate first handling over 600 attempts, reverse-review reach over 300 source-executable attempts, and final success over those same 300. Reviewer definitions were selected after acquisition and describe workflow effects rather than isolated diagnostic accuracy.

Shared physical evidence is under `outputs/physics/baseline/` and `outputs/physics/cache/`. Proposed also retains its original internal evidence layout. Hash-based subdirectory names and acquisition-stage names retain provenance and keep Windows paths short. The Reviewer offline audit is under `outputs/analysis/reviewer-feedback/`.

`experiments/verify_retained_layout.py` verifies cohort coverage, 3,000 journals, eight active captured archives, referenced evidence, and the metric matrix. `experiments/finalize_selected_reports.py` validates the Reviewer audit and republishes the approved final tables; it does not reacquire results. Historical source differs from maintained source. The public archives also have explicit privacy redactions; see [publication notes](publication.md) for original/published hash mapping and replay limits.
