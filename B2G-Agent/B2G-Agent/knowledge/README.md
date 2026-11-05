# Translator knowledge bases

`knowledge/` contains stable, source-linked evidence for the Translator. It is
separate from agent code under `src/`, the formal paper case under `cases/`, and
Reviewer episodic memory under each case.

## Release

The current release is `case-derived-kb-v3`. It combines:

- concise hand-reviewed seed notes (`*.md`);
- deterministically extracted records (`records.jsonl`);
- a versioned source registry and SHA-256 asset lock; and
- a manifest with record counts and output hashes.

The three structured stores are:

- `building/records.jsonl`: building ontology, model variables, devices,
  controls, comfort/KPI semantics, and public building-control cases;
- `grid/records.jsonl`: network-table schemas, grid concepts, topology, loads,
  profiles, transformers, and operational entities; and
- `cross_domain/records.jsonl`: co-simulation messages, demand-response
  protocol fields, meter/network relationships, and executable mapping rules.

Each store contains 180 unique records. The builder first extracts all source
candidates and then applies deterministic round-robin selection over source and
record-type strata. It never copies records to equalize counts. Record metadata
supports direction, role, business, unit, time-scale, and spatial-level gates.

Every record includes one or more `source_refs` with the source, upstream
locator, and locked raw-file hash. Raw files stay in the ignored
`.local-data/knowledge_sources/` directory and are never required at runtime.
`schemas/knowledge_record.schema.json` defines the normalized record contract and
`retrieval_config.json` freezes the weighted BM25/semantic-embedding fusion.

## Rebuild

From the repository root:

```powershell
.\.venv\Scripts\python.exe experiments\build_knowledge_bases.py --offline
```

Omit `--offline` for first acquisition. Once `asset_lock.json` exists, the
builder rejects a changed upstream file. `--update-lock` is reserved for a
deliberate, reviewed knowledge release update.

## Leakage boundary

The builder never reads `cases/`. Formal task messages, episode IDs, final
actions, numerical oracles, completion percentages, and Reviewer memory are
excluded. Public sources contribute only general concepts, schemas, interface
patterns, and analogous cases. `experiments/build_knowledge_bases.py` also
scans generated records for formal-case identifiers before writing them.

SMART-DS is registered as a catalog-only future source because bulk acquisition
is multi-terabyte. A named geography/feeder subset must be selected, licensed,
and locked before it can contribute records.
