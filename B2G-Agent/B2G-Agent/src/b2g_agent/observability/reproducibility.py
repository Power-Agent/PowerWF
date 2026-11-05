from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def request_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


PHYSICS_IMPLEMENTATION_SCHEMA_VERSION = "numeric-kernel-v2"

# Only files capable of changing deterministic numerical outputs belong here.
# Agent contracts, Reviewer schemas, runtime manifests, UI code, and status
# reporting are deliberately excluded so that non-numerical changes do not
# invalidate frozen physical ground truth.
PHYSICS_IMPLEMENTATION_PATHS = (
    "experiments/prepare_event_idfs.py",
    "experiments/run_closed_loop_episode.py",
    "experiments/run_energyplus_event_validation.py",
    "experiments/run_flexibility_envelope_episode.py",
    "experiments/solve_joint_dispatch.py",
    "experiments/validate_joint_dispatch.py",
    "src/b2g_agent/case_io/loader.py",
    "src/b2g_agent/simulation/adapters/energyplus.py",
    "src/b2g_agent/simulation/adapters/pandapower.py",
    "src/b2g_agent/simulation/energyplus_controls.py",
    "src/b2g_agent/simulation/joint_rc_opf.py",
)


def physics_implementation_hashes(repo_root: Path) -> dict[str, str]:
    """Hash every source file that can change a frozen physics result."""

    hashes: dict[str, str] = {}
    for relative in PHYSICS_IMPLEMENTATION_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Missing physics implementation file: {path}")
        hashes[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def physics_implementation_bundle_sha256(repo_root: Path) -> str:
    """Return one stable digest for the complete deterministic physics code path."""

    payload = canonical_json(
        {
            "schema_version": PHYSICS_IMPLEMENTATION_SCHEMA_VERSION,
            "files": physics_implementation_hashes(repo_root),
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def runtime_tool_versions() -> dict[str, str]:
    """Capture exact package/simulator identities used by an evaluation run."""

    versions: dict[str, str] = {}
    for distribution in (
        "openai",
        "pydantic",
        "numpy",
        "scipy",
        "pandapower",
        "pvlib",
        "sentence-transformers",
    ):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    executable_value = os.getenv("B2G_ENERGYPLUS_EXE", "energyplus").strip()
    executable = Path(executable_value).expanduser()
    if not executable.is_file():
        discovered = shutil.which(executable_value)
        executable = Path(discovered) if discovered else executable
    versions["energyplus"] = "unavailable"
    if executable.is_file():
        completed = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode == 0:
            output = (completed.stdout or completed.stderr).strip().splitlines()
            versions["energyplus"] = output[0] if output else "version-output-empty"
    return versions


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model: str = "gpt-6-luna"
    temperature: Literal[0.0] = 0.0
    top_p: Literal[1.0] = 1.0
    seed: int = 2026
    # Optional cumulative budget; disabled in the current per-call protocol.
    max_total_tokens_per_task: int | None = Field(default=None, ge=1)
    # Maximum output for EACH LLM call, independent of input and previous calls.
    max_output_tokens: int = Field(default=9000, ge=1)
    request_timeout_seconds: float = Field(default=120.0, gt=0.0)
    # Hidden SDK retries would bypass task accounting and invocation records.
    maximum_retries: int = Field(default=0, ge=0)
    maximum_schema_repairs: int = Field(default=2, ge=0, le=4)
    parallel_tool_calls: Literal[False] = False
    response_format: Literal["json_schema"] = "json_schema"
    reasoning_effort: Literal["none"] | None = "none"


class RunManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    run_id: str
    case_id: str
    episode_id: str
    method_id: str
    git_commit: str
    git_dirty: bool
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    python_version: str = Field(default_factory=lambda: sys.version.split()[0])
    operating_system: str = Field(default_factory=platform.platform)
    tool_versions: dict[str, str] = Field(default_factory=dict)
    seeds: dict[str, int] = Field(default_factory=lambda: {"global": 2026})
    llm_settings: LLMSettings = Field(default_factory=LLMSettings)
    llm_request_hashes: list[str] = Field(default_factory=list)
    llm_invocation_ids: list[str] = Field(default_factory=list)
    input_hashes: dict[str, str] = Field(default_factory=dict)
    output_hashes: dict[str, str] = Field(default_factory=dict)
    status: Literal["created", "running", "completed", "failed"] = "created"
    errors: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_llm_invocations(self) -> RunManifest:
        if self.llm_invocation_ids and (
            len(self.llm_invocation_ids) != len(self.llm_request_hashes)
        ):
            raise ValueError(
                "llm_invocation_ids must correspond one-to-one with request hashes"
            )
        return self

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.model_dump(mode="json"), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
