# Publication and provenance

The GitHub tree contains executable source, small frozen case/reference files,
tests, and an English interface guide. Large inputs and acquired outputs are
distributed separately in the linked Google Drive release. Neither distribution
contains a populated environment file or the author's private UI history.

The data packages are privacy-redacted publication copies. Full and truncated
personal filesystem prefixes are replaced by portable labels, and detected API
credentials are removed. Text files are changed only where these redactions are
needed. Captured source ZIP members receive the same treatment. Model messages,
failure records, physical numbers, and the final 18-indicator matrix are retained;
path text embedded in a prompt or an exception can differ from the acquired bytes.

`release_manifest.json` lists the downloadable ZIP hashes. After extraction,
`publication/file_hashes.jsonl` lists original and published file hashes and sizes.
`publication/redaction_map.json` provides the changed-file pairs, including
individual captured ZIP members. Original absolute prefixes are not included in
the mapping. This provides traceability without publishing those private paths.

An original hash inside a journal or frozen manifest remains a historical claim.
`recorded_hash_matches` accepts either the exact claimed bytes or an exact
original/published pair from the release manifest. It rejects unlisted changes.
This is an explicit publication transformation, not proof that the distributed
file is an untouched acquisition artifact. A checksum verifies consistency with
the manifest; it is not an independent attestation of the research results.

Current source also includes maintenance, report consolidation, Unicode escaping
of multilingual parsing literals, and portable publication readers. Changes to
public source are listed separately in
`publication/source_changes.json`. Numeric values, task definitions, and the
deterministic physics implementation are not changed for publication.

For fresh inference, use the published source and frozen input package. For the
reported results, restore the complete recorded-study package and use the
read-only cohort verifier and offline reporting commands in the
[main README](../../README.md#reproduction).
Exact-key response replay can fail after path redaction because request bytes
and captured environment paths are no longer identical. The release does not
present that replay as guaranteed numerical or bitwise reproduction.

`Personal-workspace` is runtime state, not a study input: sessions, private API
records, interaction journals, per-turn simulations, and learned failure memory.
Readers obtain a fresh workspace automatically. The public template contains only
an offline configuration example and ignore rules; no author session is needed.
