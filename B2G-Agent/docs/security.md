# Security And Trust Model

## API Keys

- Store `OPENAI_API_KEY` only in a local `.env` file or a deployment secret manager.
- `.env` and `.env.*` are ignored, except for the placeholder `.env.example`.
- The FastAPI service reads the key server-side. Browser JavaScript never receives it.
- Session JSON, simulation artifacts, logs, reports, and API responses must not include secrets or authorization headers.
- Interactive sessions use the tester's own key. With `B2G_REQUIRE_LLM=true`, a missing or placeholder key causes session creation to fail.
- Tests use fake mediator clients and never consume a real key.

## LLM Boundary

The LLM can interpret a message and return only validated JSON schemas. It cannot directly execute shell commands, write arbitrary paths, or calculate final engineering metrics. Unknown parameters and out-of-range values are rejected.

The result-writing pass receives computed evidence after execution. If no simulation ran, it must state that no new simulation evidence exists.

## Human Oversight

- Professional users retain approval responsibility.
- Material model changes can require explicit confirmation.
- NPC objectives and scenario assumptions must remain visible.
- Final output is a review package, not an autonomous construction or operating order.

## Simulator And MCP Isolation

Real MCP adapters should:

- restrict reads and writes to a configured scenario root;
- stage model changes in a run-specific directory;
- avoid modifying the source model in place;
- record tool name, version, inputs, output paths, and execution status;
- validate convergence and expected output fields;
- reject unsupported actions instead of approximating them silently;
- require confirmation for topology and equipment changes.

## Artifact Retention

Runtime artifacts are written under ignored `outputs/web_sessions/<session-id>/`. The current session store is in-memory and intended for local research demonstrations. Multi-user deployment requires authentication, access control, durable storage, retention policy, rate limiting, and an explicit privacy assessment.

## Pre-Push Secret Check

Before publishing:

```bash
git status --short
git ls-files .env
git grep -n "sk-" -- . ':!docs/security.md'
```

The first command should show only intended changes, the second should return nothing, and the final command should not reveal a credential.
