# B2G-Agent in PowerWF

This directory mirrors the complete published [B2G-Agent repository](https://github.com/cuixueyuan/B2G-Agent) at commit [`cd4343b`](https://github.com/cuixueyuan/B2G-Agent/commit/cd4343b739e1bf67b3d342cb52b3abcac9e1dcf8). B2G-Agent is independently developed and maintained by [Xueyuan Cui](https://github.com/cuixueyuan); the linked repository remains the source of ongoing development. This snapshot replaces the older two-scenario example in PowerWF.

## Layout and setup

The repository layout is preserved so the documented input restoration, experiment runners, interface, and relative links work together:

- [README.md](README.md): complete requirements, data, reproduction, results, and interface guide.
- [B2G-Agent/](B2G-Agent/): Python project, three agents, formal case, experiment scripts, tests, interface, and small result summaries.
- [Personal-workspace/](Personal-workspace/README.md): optional clean template for private interface sessions; no author session history is included.
- [SNAPSHOT.json](SNAPSHOT.json): canonical commit and SHA-256 inventory of every copied file. The three integration-only files are listed separately.

From the **PowerWF repository root**, first change to the Python project:

```powershell
cd B2G-Agent/B2G-Agent
```

Then follow the [main guide](README.md#requirements), starting with Python 3.12 and environment installation. All commands in that guide run from this Python project directory. Use a short clone path on Windows. Physical simulation requires EnergyPlus 23.1.0, the pinned Python dependencies, and the input-data package. Live experiments require the reader's own API key in an ignored `.env` file; private interface state is created in the sibling `Personal-workspace/` directory. The local interface opens at `http://127.0.0.1:8000` after launching `start_personal.py`.

## Recorded study and new runs

The case contains an IEEE 33-bus feeder, ten building energy managers, and 15-minute intervals. The final study compares Proposed, Baseline, Ablation-Translator, Ablation-Actor, and Ablation-Reviewer over 200 tasks repeated three times per method. The [Results section](README.md#results) displays the supplied figures and links their original PDFs; the [reported matrix](B2G-Agent/results/reported_metrics.csv) contains the final 18-indicator vocabulary with non-applicable entries omitted.

Large inputs and recorded outputs are distributed separately through the [Google Drive data folder](https://drive.google.com/drive/folders/1qjqdioVLz8TaIZ21DyTY-DaEg0iaApLH). The [Reproduction instructions](README.md#reproduction) distinguish offline verification of published records from fresh API/simulator runs. Restore inputs only for a new run or interface deployment; restore all archives in a separate checkout for the reported study. Records are privacy-redacted with declared original/published hash mappings. Published source includes post-acquisition maintenance; a source snapshot does not promise identical fresh cloud responses.

This contribution uses EnergyPlus and pandapower directly. It does not require PowerMCP or services from other PowerWF examples. Its application is a local research interface, with the limitations documented in the [main guide](README.md#limitations).

## Snapshot maintenance

Copied canonical files retain their exact bytes, including line endings, through the scoped `.gitattributes`. When updating this contribution, refresh the complete canonical snapshot and its inventory together; keep generated data, `.env`, virtual environments, and private sessions out of Git. Changes to other PowerWF workflows are outside this snapshot.
