# B2G-Agent Architecture

## Runtime Boundary

```text
Browser
  ├─ role selection
  ├─ research-scenario selection
  ├─ mediated chat
  ├─ evidence charts and shared state
  └─ final review and report download
        |
        v
Local FastAPI service
  ├─ SessionStore
  │   ├─ selected role and scenario
  │   ├─ shared ScenarioParameters
  │   ├─ conversation and decision ledger
  │   └─ run history
  ├─ B2GMediator
  │   ├─ OpenAI planning pass -> validated MediatorPlan
  │   └─ OpenAI explanation pass -> validated MediatedReply
  ├─ ResidentialCommunitySimulator
  │   ├─ distribution-grid-upgrade equations
  │   └─ demand-response baseline/event equations
  └─ local JSON and Markdown artifacts

Future adapter boundary; not active
  ├─ EnergyPlus-MCP
  └─ PowerMCP / OpenDSS
```

The API key is read only by the local Python process. Browser JavaScript never receives it. With `B2G_REQUIRE_LLM=true`, session creation fails when a valid personal key is unavailable rather than silently continuing in fallback mode.

## Conversation Control Loop

1. The human selects a professional role and research scenario.
2. The LLM interprets one free-form message into a typed `MediatorPlan`.
3. Scenario-specific validation rejects unsupported keys or out-of-range values.
4. Material changes can require explicit confirmation.
5. Deterministic Python code executes the selected scenario; the LLM does not calculate metrics.
6. A second LLM pass explains supplied results and speaks for the counterpart.
7. B2G-Agent records the decision note, parameter state, model run, and provenance.
8. Finalization ranks candidates and produces a human-review package.

## Shared Boundary Object

`ScenarioParameters` is the cross-domain state. Each scenario exposes only a controlled subset.

Distribution Grid Upgrade includes comfort setpoint, building count, retrofit, PV, peak flexibility, connection, transformer, and line capacity.

Demand-response service includes baseline method, baseline adjustment, participating assets, event window, kW commitment, rebound limit, comfort setpoint, connection, transformer, and line capacity.

An LLM response cannot add arbitrary fields because Pydantic rejects unknown schema keys and the scenario validator rejects fields outside the selected scenario.

## Evidence Models

The distribution-grid-upgrade model produces a transparent 24-hour building and feeder profile with voltage and thermal proxies.

The demand-response model separately represents:

- the counterfactual baseline;
- event-day building load;
- target and delivered reduction;
- baseline-method confidence;
- post-event rebound;
- feeder voltage and loading consequences.

These models are intended for workflow research and interface evaluation, not professional engineering or market settlement.

## Confirmation And Safety

- Unknown parameter: rejected.
- Out-of-range parameter: rejected.
- Unsupported field for selected scenario: rejected.
- Material topology or equipment change: confirmation can be required.
- Missing personal API key: session creation rejected in the documented configuration.
- Simulator failure: no engineering metric should be asserted.
- Arbitrary shell execution or path writes from LLM output: prohibited.
- Final output: a human-review package, never an autonomous operating instruction.

## External Tool Boundary

EnergyPlus-MCP and PowerMCP are cited because they are the intended validated building and grid tool ecosystems. The current repository does not install, import, vendor, start, or call either project. A future adapter is acceptable only after typed capability discovery, run-directory isolation, model diffs, simulator success checks, provenance capture, and regression validation are implemented.
