from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class EvaluationMethod(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    method_id: str
    label: str
    pipeline_kind: Literal["multi_agent", "single_agent"]
    retrieval_enabled: bool
    schema_plan_enabled: bool
    reviewer_memory_enabled: bool
    implementation_status: str

    @property
    def ready(self) -> bool:
        return self.implementation_status == "ready"


class EvaluationMethodRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["1.0"] = "1.0"
    shared_invariants: dict[str, object]
    methods: list[EvaluationMethod]

    @classmethod
    def load(cls, path: Path) -> EvaluationMethodRegistry:
        return cls.model_validate(json.loads(path.read_text(encoding="utf-8")))

    def get(self, method_id: str, *, require_ready: bool = True) -> EvaluationMethod:
        matches = [method for method in self.methods if method.method_id == method_id]
        if len(matches) != 1:
            raise KeyError(f"Unknown evaluation method: {method_id}")
        method = matches[0]
        if require_ready and not method.ready:
            raise RuntimeError(
                f"Evaluation method {method_id!r} is not runnable: "
                f"{method.implementation_status}"
            )
        return method
