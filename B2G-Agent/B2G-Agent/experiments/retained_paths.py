"""Compatibility imports for canonical locations and historical path resolution."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from b2g_agent.artifact_paths import (
    ROOT, OUTPUTS, BENCHMARK, COMPARISON, REVIEWER_ANALYSIS, METHOD_DIRECTORIES,
    ABLATION_SUITES, ACQUISITION_RELOCATIONS, OLD_DERIVED_ROOT,
    OLD_BASELINE_COMPARISON, BASELINE_HISTORY, RELOCATIONS, resolve_recorded_path,
)
