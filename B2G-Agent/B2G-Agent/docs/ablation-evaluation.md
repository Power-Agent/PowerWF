# Final agent ablations

All three ablations use the same 200 tasks and three repetitions. Each replaces only the named agent; the other two implementations, prompts, model settings, and knowledge inputs remain unchanged. Their actual inputs and outputs may differ as the trajectory changes.

| Method | Removed capability | Retained capability |
| --- | --- | --- |
| `plain_translator` (v2) | Bidirectional retrieval, deterministic semantic prefilling, object pairing, and rule prefilling | Same LLM, contract interface, source message and public case rules; full Actor and Reviewer |
| `unstructured_actor` (v2) | Task-specific capability DAG, exact parameter prefilling, structured plan generation and automatic normalization | Same tools, text-plan parsing, type/input/evidence checks and Reviewer loop; full Translator and Reviewer |
| `reviewer_no_feedback` (v4) | Corrective dispatch to Translator/Actor, including contract revision, replanning after physics failures, and reverse retranslation | Full review, memory, format repair, physical checks and release gates; full Translator and Actor |

Quality thresholds are F >= 0.85, A = V = 1, and G >= 0.9. The three final Reviewer indicators measure feedback effects using existing records. See [retained experiments](retained-experiments.md), [evaluation protocol](evaluation-protocol.md), and [publication notes](publication.md).
