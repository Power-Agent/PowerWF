# Architecture

## Pipeline

The reusable pipeline handles one turn at a time. Each turn creates four
domain-facing information states:

1. the sender's original natural-language message;
2. the agent-translated message received by the other domain;
3. the receiver's natural-language feedback after its own domain/physics assessment;
4. the agent-translated final message received by the original sender, including
   a quantified completion percentage.

Only state 1 is frozen in the 200-task benchmark corpus. Its benchmark session
policy permits one turn only. The interactive session policy accepts arbitrary
new natural-language wording, either sender role, and multiple sequential
turns; prior structured context and the full conversation history are supplied
to the Translator. Thus the experimental restriction is not hard-coded into
the agents. States 2–4 are runtime records.
For a directly executable request, the contract stage creates states 1–2 and
leaves states 3–4 pending until physical evaluation. It never fabricates a
receiver decision. If a message is ambiguous or conflicting, states 3–4 close
that turn at 0% completion and an interactive session can continue with a
clarifying turn.
The physics executor consumes the current translated `TaskSpecification` and
the Reviewer-approved `ExecutionPlan`, not the benchmark answer key. It writes
both as hashed run inputs,
executes the closed-loop or bidirectional-envelope workflow, then creates the
receiver's natural-language accept/modify/reject message and the sender-facing
completion statement from validated evidence.

1. **Translator** applies role/business/time/spatial metadata gates, retrieves
   separately from the source-domain, target-domain, and mapping stores, and
   uses direction-specific weighted fusion of BM25 and a pinned local semantic
   embedding. It produces a
   fixed `TaskSpecification` with per-field value, unit, status, confidence,
   and evidence. A configured LLM may revise the deterministic draft but must
   return the same schema.
2. **Actor** asks the configured LLM for a schema-constrained proposal drawn
   only from the finite action catalog. The deterministic compiler checks the
   DAG dependencies, typed input bindings, output contracts, minimal required
   action set, 15-minute resolution, node mapping, tools, and clarification
   gates. Offline mode uses the same compiler
   to produce the frozen reference plan.
3. **Reviewer** runs separate semantic and deterministic evidence checks over
   the specification, natural-language draft, action DAG, PhysicsEvidence,
   requested outputs, provenance hash, convergence, hard constraints, and
   execution log. It retrieves
   only verified failure memories using direction, task family, affected field,
   and failure-signature keys, then performs a structured LLM semantic review
   when enabled. It returns `pass`, `revise`, or `abstain` with semantic and
   evidence risk scores plus an explicit `clarify_sender`, `translator`,
   `actor`, `translator_then_actor`, or `release` route. A missing physical run
   is evidence uncertainty and routes to Actor; it never silently edits another
   agent's output. Its 150-record initialization library is generated only from
   development and memory-seed tasks through controlled perturbations verified
   by an independent deterministic validator. Online Reflexion writes require
   program or expert confirmation, and test failures remain locked until the
   evaluation is closed.

Both cross-domain transfers use Translator and Reviewer. The reverse transfer
starts from the receiver's natural-language response and preserves its decision,
modifications, quantities, evidence basis, and completion percentage; manual
interactive feedback and simulator-generated feedback share this path. A manual
reply that asserts simulation-dependent outputs remains a structured proposal
until the Actor supplies a plan-bound evidence package and the Reviewer releases
it.

The current `case-derived-kb-v3` release is rebuilt by
`experiments/build_knowledge_bases.py`. Raw public assets remain under ignored
`.local-data/knowledge_sources/`; committed records carry source locators and
SHA-256 hashes. The extractor has no case input and rejects formal episode IDs,
BEM IDs, and ground-truth/oracle identifiers to prevent benchmark leakage.

## Numerical boundary

Each BEM uses exactly one prepared representative EnergyPlus IDF to produce
electricity and zone-temperature time series. Only electrical power is scaled
to the BEM portfolio size; temperature and comfort variables are never scaled.
Power may be scaled from one prototype to a BEM portfolio; temperature may not.
After alignment, background load, BEM load, PV, and storage dispatch are mapped
to paper buses 1–33. The pandapower adapter converts kW/kvar to MW/Mvar only at
its boundary and runs one AC power flow per 15-minute interval.

The source `case33bw` network contains placeholder current ratings and no
upstream transformer. The formal case therefore derives line and transformer
ratings from the development baseline and reports them as experiment-specific.

## Separation of concerns

`src/` contains no episode text or node-specific case values. `cases/` contains
no agent implementation. `knowledge/` is stable Translator evidence;
`reviewer_memory/` is dynamic failure experience. `experiments/` produces and
validates artifacts but is not imported by the runtime.

Public inputs under `tasks/` are also separated from private comparison targets
under `groundtruth/`. Runtime APIs expose task metadata and status counts/hashes,
but never serialize individual ground-truth records.

## Evaluation boundary

`evaluation/scoring.json` and `b2g_agent.evaluation.InteractionScorer` implement
the paper's two-transfer semantic fidelity F, actionability A, hard-gated
physical feasibility V, business goal achievement G, and
`Q = 25(F + A + V + G)`. Scoring occurs after inference and physical execution;
oracle data is not an input to Translator, Actor, Reviewer, or the physics
executor.

`evaluation/methods.json` freezes five comparison paths: the complete
Translator--Actor--Reviewer method, Plain-Translator v2, Free-Plan Actor v2,
No-Feedback Reviewer v4, and a same-LLM tool-enabled single agent. All share the model snapshot,
sampling settings, tool catalog, physical executor, and scoring protocol; only
the registered ablation dimension changes.

Repeated evaluation uses a content-addressed physical-result cache. Its key
covers the physics-relevant projection of the candidate TaskSpecification, the
reviewed ExecutionPlan, every operational case/local input, the EnergyPlus
executable, and the numerical-kernel implementation bundle. Runtime retrieval
citations and timing never fragment an otherwise identical physical cache. Cache reuse
ends before agent-specific feedback and review, so it removes redundant
simulation without collapsing method behavior. The aggregator reports both
overall sample deviation and mean within-episode repeat deviation.

The final Reviewer indicators measure appropriate first handling over all 600
attempts, reverse-review reach over 300 source-executable attempts, and final
success over the same 300 attempts. Both methods use the same task cohorts and
independent structured references. These post-acquisition measures describe
workflow consequences of corrective feedback; they are not isolated diagnosis
scores. Reviewer v4 keeps full review and memory but disables corrective dispatch.

## Bounded corrections and durable interaction records

The batch runner and local UI use the same `execute_physics` controller. Cache
hits and misses remain inside that controller: every revised contract/plan is
checked again, and its own cache key is computed before execution. Forward
semantic failures return to Translator before Actor replanning. Reverse semantic
failures revise the sender-facing message before any required Actor retry.
Malformed Reviewer replies receive at most two format-repair attempts by Reviewer
itself. Domain correction loops and the outer physical loop are also bounded.
Exhaustion withholds release and records `workflow_status=blocked`; interactive
users may revise their request in another turn. Benchmark mode remains one exchange.
The batch runner persists and scores that controlled outcome without dispatching
physics or inventing a final message. Operational `completed` means processing
ended normally, not that an exchange was released. Reports separately show
workflow outcomes and runtime exceptions; blocked attempts remain unsuccessful
in the full benchmark denominator. Their actionability cannot satisfy final
delivery, even if an internal draft and physical results otherwise match the
reference. The quality formula and success thresholds are unchanged.

Live Reviewer output uses `SemanticAuditResult`: eight categorical findings with
expected/observed meanings and source locators. The program maps `consistent`
and `context_resolved` to 0, `material_ambiguity` to 0.5, and `contradiction` or
`omission` to 1. The LLM does not independently emit scores, weights, duplicate
issues or an overall status. This removes contradictory categorical/numeric
votes; it does not guarantee that a model's semantic judgment is correct.
A material ambiguity must identify two distinct executable interpretations and
their consequence after consulting available context. Identical expected and
observed meanings cannot establish a material difference. Legacy numeric
records remain readable under their original schema and are never rewritten.

Derived assessments separate `passed`, `uncertain`, and `failed`. A score of 1
always blocks, even below the unchanged semantic risk threshold of 0.25; scores
of 0.5 require a material unresolved ambiguity and trigger correction when the
weighted risk exceeds that threshold. Lower-risk uncertainty remains in the
checklist and raw records rather than becoming a fabricated mandatory error.
A finding explicitly requiring sender input still blocks regardless of aggregate
risk. An unsupported clarification is repaired before asking the sender.
Invalid ownership, missing citations, wrong review phase or inconsistent findings
require Reviewer response repair. Forward review checks an instruction, where
physical evidence is normally pending. Reverse review checks the receiver's
actual decision and evidence-backed completion; unverified completion claims
remain failures. Physical feasibility stays in the deterministic evidence channel.

Reviewer explicitly adjudicates a Translator clarification as `required` or
`unsupported`. Unsupported clarification returns to Translator for repair and
fresh Actor planning; it cannot directly terminate the task. Genuine material
ambiguity still returns to the sender. Forward, physical and reverse correction
loops provide the last four review snapshots, findings and a field-change summary.
These are context, not authoritative prior verdicts. Complete earlier history
remains in the durable stage journal; no history is shared between tasks.
For interactive follow-ups, Reviewer also receives the recent conversation so
it can distinguish established terms from unsupported additions.
Reviewer opinions are kept in review history rather than participant messages.
Translator receives the latest active correction separately from the participant
conversation; withdrawn corrections do not accumulate as participant requests.

Public operating constraints come from `optimization/joint_dispatch.json`, with
`origin=case_policy`, source locators and content hashes. They are distinct from
sender-declared constraints and inferred mappings. Zero comfort violation means
zero response-induced violations of the effective case bounds, not zero baseline
nominal-band violations. Missing or relaxed policy bounds block review and physical
dispatch. This adds provenance and checks without changing simulation bounds or
using ground-truth values in agent inputs.
Reviewer also receives the public timezone, time resolution, BEM-to-bus mapping,
service sign convention and window binding with file hashes. These can resolve
unstated operational details but cannot override conflicting sender preferences.

Translator can revise both structured fields and its optional
natural-language `receiver_message`; message-only changes do not change physical
cache identity. The offline formatter remains the fallback for older contracts.

Plan compilation records removal of redundant transitive edges in
`normalization_log`. It never adds missing dependencies or changes physical input
bindings. Validation reports exact missing/extra dependencies and expected/observed
inputs. A stale `valid` flag cannot authorize execution. Independent input checks
cover finite quantities, units, resolved fields, windows, direction conflicts,
case availability, and building-to-bus bindings. Numerical convergence and hard
constraint feasibility are separate evidence properties, including for rejections.

Each formal attempt writes `interactions/<run_id>.jsonl` under its run directory.
Interactive turns write `outputs/interactions/<trace_id>.jsonl`, configurable via
`B2G_JOURNAL_ROOT`. `PipelineResult.interaction_journal` and evaluation index rows
point to the record. Append order is authoritative; UTC timestamps, event IDs,
process IDs and nested span IDs provide cross-process association. Each event is
flushed and fsynced before progressing. An unavailable journal blocks further work.

Events retain stage inputs/outputs, all four message states and revisions,
Reviewer decisions/routes, normalization records, cache attempts, tool commands,
stdout/stderr and exceptions. API responses are persisted before parsing. A separate
physics-worker wrapper instruments unchanged numerical scripts and propagates the
journal to subprocesses; the numerical source fingerprint and existing ground-truth
values remain unchanged. Cache namespace v2 prevents reuse of old evidence flags
that conflated convergence with feasibility. Environment values/API keys are not
written to the event journal.

Offline verification is documented in `docs/evaluation-protocol.md`.
