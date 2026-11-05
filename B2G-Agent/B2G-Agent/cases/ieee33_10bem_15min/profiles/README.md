# Formal profile gate

The actual full-year CSV files live under
`.local-data/ieee33_10bem_15min/profiles/` and are intentionally ignored by Git.
Their exact schemas, source selections, derivations, and quality gates are in
`profile_specifications.json`.

The case is not physics-ready until all required files contain 35,040
15-minute intervals, the development-only line/transformer limits are derived,
target/reference dates are frozen, and every file hash appears in the local
asset manifest. Missing files cause a hard refusal; the code never invents a
replacement profile.
