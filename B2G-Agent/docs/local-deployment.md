# Local Deployment With Your Own API Key

This guide runs the complete four-stage B2G-Agent interface on one computer. Each tester supplies an individual OpenAI API key. The key stays in the local Python process and is never sent to the browser or committed to Git.

EnergyPlus, OpenDSS, EnergyPlus-MCP, and PowerMCP are not required because they are not integrated into the current release.

## 1. Prerequisites

- Git.
- Python 3.11 or newer.
- A modern browser.
- Your own OpenAI API key with access to the model configured in `.env`.

Verify Git and Python:

```text
git --version
python --version
```

On Windows, use `py -3.11 --version` when `python` is not on `PATH`.

## 2. Clone And Install

### Windows PowerShell

```powershell
git clone https://github.com/Power-Agent/PowerWF.git
Set-Location PowerWF\B2G-Agent
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
notepad .env
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS Or Linux

```bash
git clone https://github.com/Power-Agent/PowerWF.git
cd PowerWF/B2G-Agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
cp .env.example .env
```

To use the newest B2G-Agent version instead of the reviewed PowerWorkflow snapshot, clone the [canonical B2G-Agent repository](https://github.com/cuixueyuan/B2G-Agent) and follow the same installation steps from its root directory.

## 3. Configure Your API Key

Open `.env` in a text editor and replace the placeholder:

```dotenv
OPENAI_API_KEY=replace_with_your_actual_api_key
B2G_MODEL=gpt-4.1-mini
B2G_LLM_ENABLED=true
B2G_REQUIRE_LLM=true
```

Rules:

- Use your own key, not the repository author's key.
- Do not add quotes unless they are part of the key.
- Do not paste the key into `app.js`, README, an issue, a screenshot, or any tracked file.
- Do not rename `.env`; it is already excluded by `.gitignore`.
- Restart B2G-Agent after changing `.env`.

With `B2G_REQUIRE_LLM=true`, the API refuses to create a session if the key is empty, still contains the placeholder, or LLM access is disabled. This prevents an unnoticed offline fallback during research testing.

## 4. Start The Local Service

With the virtual environment active:

```text
b2g-web
```

Equivalent explicit command:

```text
python -m b2g_agent.web.cli --host 127.0.0.1 --port 8000
```

Keep the terminal open, then visit:

- Interface: <http://127.0.0.1:8000>
- Health check: <http://127.0.0.1:8000/api/health>
- Interactive API documentation: <http://127.0.0.1:8000/docs>

Press `Ctrl+C` in the terminal to stop the service.

## 5. Verify That Your API Is Active

1. Choose either engineer role.
2. On page 2, choose either research scenario and accept its evidence boundary.
3. Enter the co-design room.
4. Confirm the badge in the upper-right area says `LLM mediator · gpt-4.1-mini` or the model name you configured.
5. Send a natural-language engineering proposal.
6. Confirm that both a B2G-Agent explanation and an AI-counterpart response appear.
7. Use `Start over` on page 3 or `Start new session` on page 4 to return to role selection and test another role or scenario.

If the badge says `API not configured`, or session creation is rejected, stop the service and verify `.env` before continuing.

## 6. Try Both Research Scenarios

### Distribution Grid Upgrade Example

As the Building Engineer, try:

```text
Use a 23.5 °C cooling setpoint, standard retrofit, and 12% peak demand response. Test whether a 500 kVA transformer and 500 kW line are sufficient.
```

### Demand Response Example

As the Distribution Power Engineer, try:

```text
Use a weather-adjusted baseline for the civic building. Test a 16:00-19:00 event with a commitment of 0.8 kW per building and a maximum rebound of 20%.
```

The demand-response result should report baseline peak, baseline confidence, delivered reduction, delivery percentage, rebound, and grid limits.

## 7. Session Artifacts

The local service writes research artifacts under:

```text
outputs/web_sessions/<session-id>/
```

Each session contains:

- `session.json` — roles, selected scenario, parameters, messages, runs, and decision ledger;
- `final_plan.md` — the selected candidate, evidence, unresolved items, and limitations.

The entire `outputs/` directory is ignored by Git.

## 8. Run Tests

Tests use fake mediators and do not consume your API credits:

```text
python -m pytest -q
```

The current release should report:

```text
9 passed
```

The exact number may increase as the research prototype expands.

## 9. Troubleshooting

### `b2g-web` Is Not Recognized

Reactivate the environment and reinstall:

```text
python -m pip install -e ".[dev]"
python -m b2g_agent.web.cli
```

### PowerShell Cannot Activate The Environment

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Port 8000 Is Already In Use

```text
b2g-web --port 8765
```

Then open <http://127.0.0.1:8765>.

### API Key Required Or Authentication Error

- Confirm `.env` is in the repository root beside `pyproject.toml`.
- Confirm `OPENAI_API_KEY` no longer contains `replace_with_...`.
- Confirm `B2G_LLM_ENABLED=true` and `B2G_REQUIRE_LLM=true`.
- Confirm your API account can use the configured model.
- Restart `b2g-web` after every configuration change.

### The Browser Loads But Session Creation Fails

- Read the terminal error.
- Open `/api/health` and confirm it returns JSON.
- Confirm the repository directory is writable.
- Do not run the HTML file directly; always open the URL served by `b2g-web`.

## 10. Update An Existing Clone

From a clean checkout:

```text
git pull --ff-only
python -m pip install -e ".[dev]"
python -m pytest -q
```

Keep your local `.env`; Git intentionally does not manage it.
