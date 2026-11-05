"""Canonical output locations and read-time resolution of acquisition paths."""
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"
BENCHMARK = OUTPUTS / "benchmark"
COMPARISON = OUTPUTS / "comparison" / "final"
REVIEWER_ANALYSIS = OUTPUTS / "analysis" / "reviewer-feedback"
METHOD_DIRECTORIES = {
    "full_multi_agent": "proposed",
    "same_llm_single_agent": "baseline",
    "plain_translator": "ablation-translator",
    "unstructured_actor": "ablation-actor",
    "reviewer_no_feedback": "ablation-reviewer",
}
ABLATION_SUITES = {method: BENCHMARK / name for method, name in METHOD_DIRECTORIES.items()
                   if method in {"plain_translator", "unstructured_actor", "reviewer_no_feedback"}}
BASELINE_HISTORY = BENCHMARK / "baseline" / "acquisition_history"
ACQUISITION_RELOCATIONS = {
    "20261101T233426Z-v2-t": ABLATION_SUITES["plain_translator"] / "first50",
    "20261102T181817Z-v2-tx": ABLATION_SUITES["plain_translator"] / "attempt_history",
    "20261102T035441Z-v2-a": ABLATION_SUITES["unstructured_actor"] / "first50",
    "20261102T181817Z-v2-ax": ABLATION_SUITES["unstructured_actor"] / "remaining150",
    "20261102T150426Z-v4-r": ABLATION_SUITES["reviewer_no_feedback"] / "first50",
    "20261102T181817Z-v4-rx": ABLATION_SUITES["reviewer_no_feedback"] / "remaining150",
}
OLD_DERIVED_ROOT = OUTPUTS / "campaigns/retained-extension-20261102T181817Z/serial/full200"
OLD_BASELINE_COMPARISON = OUTPUTS / "comparison/baseline-final-20261031"
RELOCATIONS = {
    **{f"outputs/benchmark/{old}": target for old, target in ACQUISITION_RELOCATIONS.items()},
    **{f"outputs/campaigns/retained-extension-20261102T181817Z/serial/full200/{method}": target
       for method, target in ABLATION_SUITES.items()},
    "outputs/comparison/baseline-final-20261031": BASELINE_HISTORY,
    "outputs/benchmark/20261030T223320Z": BENCHMARK / "proposed",
    "outputs/benchmark/20261031T075835Z-baseline": BENCHMARK / "baseline",
    "outputs/benchmark/t-v2": ABLATION_SUITES["plain_translator"],
    "outputs/benchmark/a-v2": ABLATION_SUITES["unstructured_actor"],
    "outputs/benchmark/r-v4": ABLATION_SUITES["reviewer_no_feedback"],
    "outputs/pb/20261031T075835Z": OUTPUTS / "physics/baseline",
    "outputs/pc": OUTPUTS / "physics/cache",
    "outputs/comparison/final-selected-v2-v2-v4-20261103": COMPARISON,
    "outputs/analysis/reviewer-feedback-impacts-full200-20261103": REVIEWER_ANALYSIS,
    "outputs/campaigns/retained-extension-20261102T181817Z": COMPARISON / "acquisition_campaign",
}
_ORDERED_RELOCATIONS = sorted(RELOCATIONS.items(), key=lambda item: -len(item[0]))


def resolve_recorded_path(path: str | Path, *, repo_root: Path | None = None) -> Path:
    """Resolve old/relocated checkout paths without rewriting captured records."""
    original = Path(path)
    if repo_root is None and original.exists():
        return original
    root = Path(repo_root) if repo_root is not None else ROOT
    normalized = str(path).replace("\\", "/")
    # Published repository-prefixed references can be joined to a record's
    # parent by older readers. Resolve the explicit repository anchor first.
    anchor = normalized.casefold().rfind("/b2g-agent/")
    if anchor >= 0:
        normalized = normalized[anchor + len("/B2G-Agent/"):]
    elif normalized.casefold().startswith("b2g-agent/"):
        normalized = normalized[len("B2G-Agent/"):]
    marker = normalized.casefold().find("/outputs/")
    relative = normalized[marker + 1:] if marker >= 0 else normalized
    for old, target in _ORDERED_RELOCATIONS:
        if relative.casefold() == old.casefold() or relative.casefold().startswith(old.casefold() + "/"):
            destination = root / target.relative_to(ROOT)
            suffix = relative[len(old):].lstrip("/")
            return destination / PurePosixPath(suffix)
    if relative.startswith("outputs/"):
        return root / PurePosixPath(relative)
    if normalized.startswith((".local-data/", ".local-runs/")):
        return root / PurePosixPath(normalized)
    for name in (".local-data", ".local-runs"):
        marker = "/" + name + "/"
        if marker in normalized:
            return root / name / PurePosixPath(normalized.split(marker, 1)[1])
    return original

