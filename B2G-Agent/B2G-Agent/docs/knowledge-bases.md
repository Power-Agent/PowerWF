# Case-derived Translator knowledge

## Design boundary

The Translator uses three source-linked knowledge bases, not the private
benchmark answers. Public data contributes reusable vocabulary, interface
schemas, unit/time conventions, control cases, and mapping patterns. The
IEEE-33/10-BEM task messages and ground-truth outcomes remain evaluation-only.

```text
official public source slices
        |
        v
.local-data/knowledge_sources/       ignored raw assets
        |
        | deterministic extractors + leakage checks
        v
knowledge/{building,grid,cross_domain}/records.jsonl
        |
        | metadata gate + direction-aware weighted rank fusion
        v
Translator TaskSpecification + source/hash evidence
```

## Sources and roles

| Source | Extracted role |
|---|---|
| PNNL TESP | HELICS publications, subscriptions, federates, and EnergyPlus-agent interaction patterns |
| URBANopt | GeoJSON building/electrical feature schemas and district control scenarios |
| BOPTEST | Building control configurations, named operating days, measurements, and KPI mappings |
| CityLearn | Multi-building observations/actions, storage/EV fields, and electrical-service schemas |
| OCHRE | Building/device variable names, simulator aliases, units, and flexible-device defaults |
| Brick | Canonical building concepts and meter, submeter, PV, and EVSE relationships |
| ETSI SAREF | Versioned building and grid ontology concepts |
| OpenLEADR | Demand-response event, active-period, signal, target, response, and opt vocabulary |
| SimBench | Node, line, load, profile, transformer, external-network, and study-case table schemas |
| SMART-DS | Registered but intentionally not bulk-downloaded; a named feeder subset is required |

Exact URLs, revisions, licensing notes, and selected files are in
`knowledge/sources/source_registry.json`. Exact downloaded hashes and byte
counts are in `knowledge/sources/asset_lock.json`; citation-manager entries are
provided in `knowledge/sources/references.bib`.

## Runtime retrieval

The runtime reads the three structured JSONL stores. Each store contains
exactly 180 unique records selected by a deterministic source/type-stratified
policy; no duplicated padding is used. Evidence
contains the record ID, source family, record locator, excerpt, and SHA-256.

The frozen `bm25+minilm-dense-embedding+weighted-rrf-v4` method first gates
records by roles, business type, time scale, and spatial level. Transfer
direction determines the source/target partitions and their weights rather
than filtering target-domain knowledge a second time. It then retrieves separately from the source-domain,
target-domain, and cross-domain mapping stores using weighted reciprocal-rank
fusion of BM25 and the pinned 384-dimensional
`sentence-transformers/all-MiniLM-L6-v2` neural model at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. The model is downloaded once,
hashed, and then loaded with network access disabled. Direction-specific weights live in `knowledge/retrieval_config.json`
and are frozen as an author-selected methodological setting without relevance
judgments or evaluation-result tuning.

Each record includes units, time scales, spatial levels, applicable roles,
business types, directions, and review status. Source-derived cross-domain
rules are frozen as source-derived method records. This status is deliberately
not called expert verification: it records an author methodological decision
that no separate dual-domain sign-off is required.

## Updating the release

1. Review the official source, revision, licensing, and redistribution terms.
2. Add a minimal machine-readable asset to the source registry.
3. Add or update a deterministic extractor; do not use an LLM to finalize unit,
   sign, time, or topology rules.
4. Run an online build with `--update-lock` only for the intentional release.
5. Run a second `--offline` build and confirm identical output hashes.
6. Run the knowledge and full test suites.
7. Assign a new knowledge release and retrieval method ID if corpus or ranking
   semantics change, then rerun every compared experimental method.
