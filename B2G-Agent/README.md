# B2G-Agent: Building--Grid Co-Design

> **Project ownership and maintenance:** B2G-Agent is independently developed and maintained by [Xueyuan Cui](https://github.com/cuixueyuan). This directory is a PowerWorkflow community entry for the collaboration. The canonical source code, releases, issues, and development history remain at [cuixueyuan/B2G-Agent](https://github.com/cuixueyuan/B2G-Agent).

B2G-Agent is an LLM-mediated research environment for collaboration between Building Engineers and Distribution Power Engineers. A human chooses one role, an LLM plays the other as an AI counterpart, and the agent translates free-form professional input into a shared, validated engineering case. It then runs transparent scenario calculations, explains the consequences to both disciplines, and records the path to a jointly reviewed plan.

![B2G-Agent four-stage interface](https://raw.githubusercontent.com/cuixueyuan/B2G-Agent/main/docs/assets/screenshots/03-codesign-room.png)

## Why This Is a PowerWorkflow Example

The workflow coordinates several responsibilities that normally sit in separate tools or professional silos:

1. interpret a participant's natural-language goals, constraints, and proposals;
2. translate the proposal into a typed, validated cross-domain case;
3. select building-side, grid-side, or coupled evidence;
4. explain technical impacts in language the other engineer can act on;
5. maintain a decision ledger and guide the dialogue toward a joint plan;
6. require human review before the final recommendation is accepted.

The current release uses an OpenAI model for intent interpretation and an AI counterpart. Engineering metrics are produced by deterministic research models rather than invented by the LLM.

## Research Scenarios

- **Distribution Grid Upgrade:** coordinate new construction and building retrofits with transformer, feeder, voltage, capacity, and flexibility decisions.
- **Demand Response Service:** agree on a defensible baseline, event window, reduction commitment, comfort constraints, delivery evidence, and rebound limits.

## Four-Stage User Journey

1. Choose the Building Engineer or Distribution Power Engineer role.
2. Choose a research scenario and review responsibilities and evidence boundaries.
3. Negotiate with the AI counterpart in a mediated natural-language dialogue.
4. Review the selected candidate, constraint checks, unresolved items, and reproducible report.

## Run the Canonical Project Locally

Each tester supplies their own OpenAI API key. No API key is stored in this PowerWorkflow repository or in the B2G-Agent source repository.

```bash
git clone https://github.com/cuixueyuan/B2G-Agent.git
cd B2G-Agent
python -m venv .venv
python -m pip install -e ".[dev]"
```

Activate the virtual environment, copy `.env.example` to `.env`, and replace the placeholder with your own key:

```dotenv
OPENAI_API_KEY=replace_with_your_actual_api_key
B2G_MODEL=gpt-4.1-mini
B2G_LLM_ENABLED=true
B2G_REQUIRE_LLM=true
```

Then run:

```text
b2g-web
```

Open <http://127.0.0.1:8000>. See the canonical repository's [complete local deployment guide](https://github.com/cuixueyuan/B2G-Agent/blob/main/docs/local-deployment.md) for platform-specific activation commands, verification, and troubleshooting.

## Tool Boundary

B2G-Agent currently executes the OpenAI Python SDK, FastAPI, Uvicorn, Pydantic, python-dotenv, and its own transparent deterministic research models. [EnergyPlus-MCP](https://github.com/LBNL-ETA/EnergyPlus-MCP) and [PowerMCP](https://github.com/Power-Agent/PowerMCP) are documented integration targets; they are not installed, imported, or called by the current release. EnergyPlus and OpenDSS are likewise not yet executed. This distinction prevents the demonstration from being mistaken for a calibrated simulation or settlement-grade study.

## Citation and License

```bibtex
@software{b2g_agent,
  title  = {B2G-Agent: An LLM-Mediated Research Environment for Building--Grid Co-Design},
  author = {Xueyuan Cui},
  year   = {2026},
  url    = {https://github.com/cuixueyuan/B2G-Agent}
}
```

B2G-Agent is released under the MIT License with copyright retained by Xueyuan Cui. See the [canonical license](https://github.com/cuixueyuan/B2G-Agent/blob/main/LICENSE). External projects retain their own licenses.
