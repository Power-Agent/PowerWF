# Same-LLM single-agent baseline

`same_llm_single_agent` performs interpretation, tool planning, revisions, and
both return messages with one LLM agent. It has the same domain knowledge,
2+2+2 retrieval budget, public engineering context, tool catalog and deterministic
EnergyPlus/pandapower executor as the proposed method. It has no role-specific
LLM calls, inter-agent correction routing, or failure memory. Shared deterministic
plan, readiness and evidence guards return their findings to this same agent.

The baseline protocol is `cases/ieee33_10bem_15min/evaluation/baseline_protocol.json`.
It shares the proposed execution settings; its report omits the nine role-specific agent metrics:
200 episodes, three executions each, gpt-6-luna, temperature 0, top-p 1, seed 2026,
reasoning effort none, 9000 output tokens per LLM call, four bounded agent revisions
and two format repairs. Every LLM repetition is live. Only exact deterministic
physics inputs/plans can reuse the suite's own physics cache.

```powershell
.\.venv\Scripts\python.exe -B experiments\run_proposed_suite.py --method same_llm_single_agent
.\.venv\Scripts\python.exe -B experiments\run_proposed_suite.py --method same_llm_single_agent --execute
```

The first command prepares and prints settings without API calls. The second
executes the authorized formal batch in its own `outputs/benchmark/baseline/`
folder. Journals are flushed stage by stage; requests, raw responses and parsed
results are retained. Failures remain in the denominator. Personal UI sessions
and other methods never enter this cohort.

The final baseline comparison reports six interaction-quality metrics and three
whole-framework computation-cost metrics. Agent-performance metrics are omitted
because this baseline has no separate Translator, Actor, or LLM Reviewer. The
current protocol reports these nine metrics. Frozen acquisition archives preserve
the historical schema without exposing it as another current metric definition.

The selected Baseline's 600 run records are in `outputs/benchmark/baseline/`; its nine applicable final indicator values are in `FINAL_METRICS.csv` there and in the five-method comparison at `outputs/comparison/final/`. Three recorded infrastructure failures were recovered. Failed method outcomes remain in the denominator.

Both completed cohorts use scoring protocol `interaction-quality-f085-v2`: F >= 0.85, A = 1, V = 1, G >= 0.9. The shared scoring revision was offline; model outputs and physical evidence were not reacquired. Current per-run scores remain in each suite's `current_scores/`. Agent metrics use programmatic references rather than independent human labels; repeated calls on 200 tasks are not 600 independent scenarios.

The current execution still has known implementation limitations. The proposed
Translator receives a deterministic contract draft while the baseline constructs
its contract without that draft. Reference matching is sensitive to objective
names and output supersets. In the recorded baseline, 106 final blocks concern
plan-input bindings and 165 concern unresolved additional outputs. Of the former,
74 differ only by a single-object-versus-list representation; of the latter,
144 involve market-only accepted capacity requested outside the relevant context.
Changing the success threshold does not fix these interface or equivalence issues.
