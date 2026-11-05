from __future__ import annotations

import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from b2g_agent.publication import recorded_hash_matches
from b2g_agent.artifact_paths import resolve_recorded_path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from b2g_agent.case_io import CaseBundle, sha256_file
from b2g_agent.contracts import (
    ExecutionPlan,
    PhysicsEvidence,
    ReceiverFeedback,
    TaskSpecification,
)
from b2g_agent.observability import physics_implementation_bundle_sha256


CASE_PHYSICS_INPUTS = (
    "case.json",
    "days/target_days.json",
    "der/storage.json",
    "groundtruth/contract_oracles.jsonl",
    "network/bem_to_bus.csv",
    "network/line_limits.csv",
    "network/source_case33bw.json",
    "network/transformer.json",
    "optimization/joint_dispatch.json",
)

LOCAL_PHYSICS_INPUTS = (
    "weather/USA_NY_Buffalo.Niagara.Intl_.AP_.725280_TMY3.epw",
    "profiles/background_load_15min.csv",
    "profiles/bem_baseline_15min.csv",
    "profiles/bem_thermal_15min.csv",
    "profiles/pv_15min.csv",
    "profiles/prices_15min.csv",
    "profiles/reserve_15min.csv",
    "models/rc_parameters.json",
    "models/grid_sensitivities/baseline_bus_voltage_pu.npy",
    "models/grid_sensitivities/baseline_line_loading_pct.npy",
    "models/grid_sensitivities/baseline_transformer_loading_pct.npy",
    "models/grid_sensitivities/bus_voltage_pu_per_kw_reduction.npy",
    "models/grid_sensitivities/line_loading_pct_per_kw_reduction.npy",
    "models/grid_sensitivities/transformer_loading_pct_per_kw_reduction.npy",
)


class PhysicsCacheRecord(BaseModel):
    """Reusable simulator output, excluding method-specific LLM feedback."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1.0"] = "1.0"
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    task_specification_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    execution_plan_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    physics_implementation_bundle: str = Field(pattern=r"^[0-9a-f]{64}$")
    physics_input_bundle: str = Field(pattern=r"^[0-9a-f]{64}$")
    feedback: ReceiverFeedback
    evidence: PhysicsEvidence


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def task_specification_sha256(specification: TaskSpecification) -> str:
    payload = canonical_json(task_specification_physics_payload(specification))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def task_specification_physics_payload(
    specification: TaskSpecification,
) -> dict[str, Any]:
    """Return only fields capable of changing a deterministic physics run."""

    payload = specification.model_dump(mode="json")
    for name in (
        "source_text",
        "receiver_message",
        "retrieval_evidence",
        "retrieval_trace",
        "turn_index",
        "context_origin",
    ):
        payload.pop(name, None)
    for field in payload.get("fields", {}).values():
        field.pop("confidence", None)
        field.pop("evidence", None)
        field.pop("note", None)
    for constraint in payload.get("constraints", []):
        constraint.pop("origin", None)
        constraint.pop("evidence", None)
    return payload


def execution_plan_sha256(plan: ExecutionPlan) -> str:
    payload = canonical_json(plan.model_dump(mode="json"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def physics_input_hashes(case: CaseBundle) -> dict[str, str]:
    """Hash every case/local artifact read by the deterministic physics loop."""

    local_case = case.local_data_root / case.case_id
    paths: list[tuple[str, Path]] = [
        (f"case:{relative}", case.case_dir / relative)
        for relative in CASE_PHYSICS_INPUTS
    ]
    paths.extend(
        (f"local:{relative}", local_case / relative)
        for relative in LOCAL_PHYSICS_INPUTS
    )
    paths.extend(
        (
            f"local:buildings/bem_{index:02d}/energyplus_23_1.idf",
            local_case / "buildings" / f"bem_{index:02d}" / "energyplus_23_1.idf",
        )
        for index in range(1, 11)
    )
    executable_value = os.getenv("B2G_ENERGYPLUS_EXE", "energyplus").strip()
    executable = Path(executable_value).expanduser()
    if not executable.is_file():
        discovered = shutil.which(executable_value)
        executable = Path(discovered) if discovered else executable
    paths.append(("runtime:energyplus_executable", executable))

    missing = [str(path) for _, path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Cannot fingerprint physics cache inputs; missing: " + ", ".join(missing)
        )
    return {name: sha256_file(path) for name, path in paths}


def physics_input_bundle_sha256(hashes: dict[str, str]) -> str:
    return hashlib.sha256(canonical_json(hashes).encode("utf-8")).hexdigest()


def physics_cache_key(
    specification: TaskSpecification,
    *,
    physics_implementation_bundle: str,
    physics_input_bundle: str,
    execution_plan: ExecutionPlan | None = None,
) -> str:
    payload = {
        "schema_version": "2.0",
        "evidence_interpretation": "separate_numerical_convergence_and_feasibility_v2",
        "task_specification": task_specification_physics_payload(specification),
        "physics_implementation_bundle": physics_implementation_bundle,
        "physics_input_bundle": physics_input_bundle,
        "execution_plan": (
            execution_plan.model_dump(mode="json") if execution_plan is not None else None
        ),
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


class PhysicsResultCache:
    def __init__(
        self,
        *,
        root: Path,
        repo_root: Path,
        case: CaseBundle,
    ) -> None:
        self.root = root.resolve()
        self.repo_root = repo_root.resolve()
        self.case = case
        self.implementation_bundle = physics_implementation_bundle_sha256(repo_root)
        self.input_hashes = physics_input_hashes(case)
        self.input_bundle = physics_input_bundle_sha256(self.input_hashes)

    def key(
        self, specification: TaskSpecification, execution_plan: ExecutionPlan | None = None
    ) -> str:
        return physics_cache_key(
            specification,
            physics_implementation_bundle=self.implementation_bundle,
            physics_input_bundle=self.input_bundle,
            execution_plan=execution_plan,
        )

    def load(
        self,
        specification: TaskSpecification,
        execution_plan: ExecutionPlan | None = None,
    ) -> tuple[ReceiverFeedback, PhysicsEvidence, str] | None:
        key = self.key(specification, execution_plan)
        path = self.root / "records" / f"{key}.json"
        if not path.is_file():
            return None
        record = PhysicsCacheRecord.model_validate_json(path.read_text(encoding="utf-8"))
        if record.cache_key != key:
            raise ValueError(f"Physics cache key mismatch: {path}")
        evidence_path = resolve_recorded_path(record.evidence.evidence_path)
        if not evidence_path.is_file():
            return None
        if not recorded_hash_matches(evidence_path, record.evidence.evidence_sha256):
            raise ValueError(f"Physics cache evidence hash mismatch: {evidence_path}")
        return record.feedback, record.evidence, key

    def allocate_run_root(
        self,
        specification: TaskSpecification,
        execution_plan: ExecutionPlan | None = None,
    ) -> tuple[Path, str]:
        key = self.key(specification, execution_plan)
        # Keep the simulator's deeply nested output below the traditional
        # Windows MAX_PATH boundary. The complete key remains in records/.
        run_root = self.root / "runs" / key[:2] / f"{key[:16]}-{uuid4().hex[:8]}"
        return run_root, key

    def store(
        self,
        specification: TaskSpecification,
        *,
        execution_plan: ExecutionPlan | None = None,
        feedback: ReceiverFeedback,
        evidence: PhysicsEvidence,
    ) -> str:
        key = self.key(specification, execution_plan)
        record = PhysicsCacheRecord(
            cache_key=key,
            task_specification_sha256=task_specification_sha256(specification),
            execution_plan_sha256=(
                execution_plan_sha256(execution_plan)
                if execution_plan is not None
                else None
            ),
            physics_implementation_bundle=self.implementation_bundle,
            physics_input_bundle=self.input_bundle,
            feedback=feedback,
            evidence=evidence,
        )
        path = self.root / "records" / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{uuid4().hex}.tmp")
        temporary.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return key
