# Personal workspace template

This is an optional clean template. The interface automatically creates its runtime folders and an offline `settings.json` beside the B2G-Agent project on first launch. No author conversations, API responses, physical runs, or learned memory are included.

- `settings.example.json`: offline defaults; copy to `settings.json` only if preconfiguring a new workspace.
- `sessions/`, `interactions/`, `lr/`, `memory/` and simulation directories: generated private state, ignored by Git.
- API keys belong in `B2G-Agent/.env` or server environment variables, never in this template.

The paper experiments do not read personal settings or learned personal memory. See the [main README](../README.md) for installation and interface usage.
