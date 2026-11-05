from __future__ import annotations

"""Build a verified, non-test Reviewer failure-memory seed."""

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.agents.reviewer_validation import (
    ALL_SIGNATURES,
    SEMANTIC_SIGNATURES,
    VALIDATOR_VERSION,
    validate_controlled_failure,
    validator_sha256,
)


CASE_ROOT = REPO_ROOT / "cases" / "ieee33_10bem_15min"
MEMORY_ROOT = CASE_ROOT / "reviewer_memory"
SOURCE_SPLITS = ("development", "memory_seed")
MUTATIONS_PER_EPISODE = 3


TEMPLATES: dict[str, dict[str, object]] = {
    "S-OBJECT-OMISSION": {
        "agent": "translator", "fields": ["target_bems", "target_buses"],
        "trigger": "The translation omits every controlled object named by the sender.",
        "root": "entity extraction or carry-forward failure",
        "wrong": "The receiver instruction contains no actionable building or bus target.",
        "correction": "Restore all explicitly named controlled objects and their identifiers.",
        "rule": "A cross-domain instruction must retain every explicit target object.",
        "boundary": "Applies when at least one target BEM or bus is explicit in the source message.",
    },
    "S-ACTION-DIRECTION-REVERSAL": {
        "agent": "translator", "fields": ["task_families", "target_power"],
        "trigger": "A decrease/curtailment request is translated as an increase/absorption request or vice versa.",
        "root": "sign or action-polarity normalization failure",
        "wrong": "The translated action has the opposite physical direction.",
        "correction": "Restore the source action direction and preserve the sign convention in provenance.",
        "rule": "Never infer action polarity from a positive magnitude alone.",
        "boundary": "Applies to requests whose direction affects physical dispatch.",
    },
    "S-BEM-BUS-MISMATCH": {
        "agent": "translator", "fields": ["target_bems", "target_buses"],
        "trigger": "The translated one-to-one building-to-bus mapping differs from the case mapping.",
        "root": "spatial binding failure",
        "wrong": "At least one BEM is assigned to the wrong distribution bus.",
        "correction": "Rebind each BEM through the declared case mapping and preserve pair order.",
        "rule": "Validate BEM-to-bus bindings before aggregation or execution.",
        "boundary": "Applies only where an authoritative case mapping exists.",
    },
    "S-UNIT-MISMATCH": {
        "agent": "translator", "fields": ["target_power"],
        "trigger": "The translated quantity changes unit without a valid conversion.",
        "root": "unit normalization failure",
        "wrong": "A kW request is labeled or interpreted as MW.",
        "correction": "Restore the declared unit or apply an explicit, traceable conversion.",
        "rule": "A value and unit form one semantic field and must be validated together.",
        "boundary": "Applies to all physical and market quantities with units.",
    },
    "S-TIME-WINDOW-DRIFT": {
        "agent": "translator", "fields": ["windows"],
        "trigger": "The translated event window differs from the sender's interval.",
        "root": "temporal parsing or timezone alignment failure",
        "wrong": "The translated start or end time drifts from the requested window.",
        "correction": "Restore start, end, timezone, and interval semantics from the source.",
        "rule": "Time windows must round-trip exactly unless the receiver explicitly modifies them.",
        "boundary": "Applies to interval-based requests and responses.",
    },
    "S-CONSTRAINT-OMISSION": {
        "agent": "translator", "fields": ["constraints"],
        "trigger": "A hard voltage, loading, comfort, or rebound constraint is omitted.",
        "root": "constraint extraction failure",
        "wrong": "The translated contract permits an action forbidden by the source message.",
        "correction": "Restore every explicit hard constraint with operator, value, and unit.",
        "rule": "Hard constraints cannot be summarized away during cross-domain translation.",
        "boundary": "Applies when the sender states an explicit bound or prohibition.",
    },
    "S-REQUESTED-OUTPUT-OMISSION": {
        "agent": "translator", "fields": ["requested_outputs"],
        "trigger": "A requested decision or evidence output is absent from the contract.",
        "root": "output-contract construction failure",
        "wrong": "Execution can finish without returning all requested evidence.",
        "correction": "Restore all requested business and physical outputs before planning.",
        "rule": "Every requested result must map to a typed output or a clarification.",
        "boundary": "Applies to explicit output and reporting requirements.",
    },
    "E-INVALID-DAG": {
        "agent": "actor", "fields": ["execution_plan"],
        "trigger": "An action depends on itself or a downstream action.",
        "root": "dependency-graph construction failure",
        "wrong": "The plan is cyclic and cannot be executed deterministically.",
        "correction": "Recompile a topologically ordered dependency graph.",
        "rule": "Reject plans with unknown, self, or forward dependencies.",
        "boundary": "Applies to all schema-constrained execution plans.",
    },
    "E-WRONG-TOOL": {
        "agent": "actor", "fields": ["execution_plan"],
        "trigger": "An action is bound to a tool outside its typed capability.",
        "root": "capability binding failure",
        "wrong": "A power-flow action is assigned to a non-power-flow tool.",
        "correction": "Bind the action to its registered deterministic tool.",
        "rule": "Action type and tool binding must match the capability registry.",
        "boundary": "Applies to executable typed actions.",
    },
    "E-NONCONVERGENCE": {
        "agent": "actor", "fields": ["physics_evidence"],
        "trigger": "The physical solver reports non-convergence.",
        "root": "simulation or optimization convergence failure",
        "wrong": "A non-converged run is presented as validated delivery.",
        "correction": "Replan or abstain; never release non-converged evidence.",
        "rule": "Convergence is a release gate, not a descriptive warning.",
        "boundary": "Applies to simulator- or optimizer-backed claims.",
    },
    "E-MISSING-OUTPUT": {
        "agent": "actor", "fields": ["requested_outputs"],
        "trigger": "At least one requested metric has no value in the evidence package.",
        "root": "evidence-contract completion failure",
        "wrong": "The run is marked complete with a null requested output.",
        "correction": "Produce the metric through tools or mark it unresolved and replan.",
        "rule": "Every requested output must be resolved or explicitly unresolved.",
        "boundary": "Applies after physical execution is claimed complete.",
    },
    "E-STALE-PROVENANCE": {
        "agent": "actor", "fields": ["physics_evidence"],
        "trigger": "The claimed evidence hash does not match the evidence payload.",
        "root": "stale or substituted evidence artifact",
        "wrong": "The response cites a provenance hash for different content.",
        "correction": "Recompute and verify the content hash before release.",
        "rule": "Evidence identity must be content-addressed and verified.",
        "boundary": "Applies to persisted evidence artifacts.",
    },
    "E-HARD-CONSTRAINT-VIOLATION": {
        "agent": "actor", "fields": ["constraints", "physics_evidence"],
        "trigger": "The evidence reports a hard-constraint violation.",
        "root": "infeasible action or incorrect release decision",
        "wrong": "A comfort or grid-limit violation is released as successful delivery.",
        "correction": "Reduce/modify the commitment, replan, or reject the task.",
        "rule": "No completion claim may override a failed hard constraint.",
        "boundary": "Applies whenever hard constraints are declared.",
    },
    "E-TIME-ALIGNMENT": {
        "agent": "actor", "fields": ["time_resolution", "physics_evidence"],
        "trigger": "Building and grid evidence use different interval indexes.",
        "root": "cross-simulator time alignment failure",
        "wrong": "Metrics combine unmatched EnergyPlus and grid intervals.",
        "correction": "Align timezone, interval boundaries, and 15-minute indexes before aggregation.",
        "rule": "Cross-domain metrics require an identical interval index.",
        "boundary": "Applies to coupled time-series evidence.",
    },
    "E-BEM-BUS-MAPPING": {
        "agent": "actor", "fields": ["target_bems", "target_buses", "physics_evidence"],
        "trigger": "Execution maps a BEM load to a bus inconsistent with the case mapping.",
        "root": "physical input-binding failure",
        "wrong": "Power flow evaluates flexibility at the wrong location.",
        "correction": "Rebuild nodal injections from the frozen BEM-to-bus map.",
        "rule": "Physical evidence must retain the validated spatial mapping.",
        "boundary": "Applies before every feeder power-flow call.",
    },
}


def _load_episodes() -> dict[str, dict[str, Any]]:
    return {
        item["episode_id"]: item
        for item in (
            json.loads(line)
            for line in (CASE_ROOT / "tasks" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _split_ids(name: str) -> list[str]:
    return [line.strip() for line in (CASE_ROOT / "tasks" / "splits" / f"{name}.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def _artifact(signature: str, episode: dict[str, Any]) -> dict[str, Any]:
    mappings = dict(zip(episode["target_bems"], episode["target_buses"], strict=True))
    alternate = {bem: (bus % 33) + 1 for bem, bus in mappings.items()}
    window_values = episode["windows"]
    artifacts: dict[str, dict[str, Any]] = {
        "S-OBJECT-OMISSION": {"original_objects": episode["target_bems"], "mutated_objects": []},
        "S-ACTION-DIRECTION-REVERSAL": {"original_direction": "requested", "mutated_direction": "opposite"},
        "S-BEM-BUS-MISMATCH": {"expected_mapping": mappings, "mutated_mapping": alternate},
        "S-UNIT-MISMATCH": {"source_unit": "kW", "translated_unit": "MW"},
        "S-TIME-WINDOW-DRIFT": {"source_windows": window_values, "translated_windows": []},
        "S-CONSTRAINT-OMISSION": {"source_constraints": ["voltage", "line_loading", "comfort"], "translated_constraints": ["voltage"]},
        "S-REQUESTED-OUTPUT-OMISSION": {"requested_outputs": ["business_metric", "comfort_evidence"], "translated_outputs": ["business_metric"]},
        "E-INVALID-DAG": {"action_ids": ["run_power_flow"], "dependencies": ["run_power_flow"]},
        "E-WRONG-TOOL": {"expected_tool": "pandapower", "observed_tool": "language_model"},
        "E-NONCONVERGENCE": {"converged": False},
        "E-MISSING-OUTPUT": {"requested_output_values": {"business_metric": None}},
        "E-STALE-PROVENANCE": {"claimed_sha256": "0" * 64, "evidence_payload": {"episode": episode["episode_id"]}},
        "E-HARD-CONSTRAINT-VIOLATION": {"hard_constraints_passed": False},
        "E-TIME-ALIGNMENT": {"building_intervals": [0, 15, 30], "grid_intervals": [0, 30]},
        "E-BEM-BUS-MAPPING": {"expected_mapping": mappings, "observed_mapping": alternate},
    }
    return artifacts[signature]


def build() -> dict[str, Any]:
    episodes = _load_episodes()
    sources = [(split, episode_id) for split in SOURCE_SPLITS for episode_id in _split_ids(split)]
    if any(episodes[episode_id]["split"] == "test" for _, episode_id in sources):
        raise ValueError("Reviewer memory must never be seeded from the test split")
    records: list[dict[str, Any]] = []
    for source_index, (split, episode_id) in enumerate(sources):
        episode = episodes[episode_id]
        for mutation_index in range(MUTATIONS_PER_EPISODE):
            signature = ALL_SIGNATURES[(source_index * MUTATIONS_PER_EPISODE + mutation_index) % len(ALL_SIGNATURES)]
            template = TEMPLATES[signature]
            artifact = _artifact(signature, episode)
            if not validate_controlled_failure(signature, artifact):
                raise ValueError(f"Independent validator did not detect {signature} for {episode_id}")
            channel = "semantic" if signature in SEMANTIC_SIGNATURES else "evidence"
            records.append({
                "schema_version": "2.0",
                "memory_id": f"mem-{len(records) + 1:03d}",
                "source_episode_id": episode_id,
                "source_split": split,
                "generation_method": "controlled_programmatic_perturbation",
                "failure_channel": channel,
                "failure_signature": signature,
                "responsible_agent": template["agent"],
                "direction": f"{episode['start_role']}->{_counterpart(episode['start_role'])}",
                "task_families": episode["task_families"],
                "affected_fields": template["fields"],
                "trigger": template["trigger"],
                "root_cause": template["root"],
                "wrong_output": template["wrong"],
                "correction": template["correction"],
                "generalized_rule": template["rule"],
                "applicability_boundary": template["boundary"],
                "verification_evidence": {
                    "kind": "independent_deterministic_validator",
                    "validator_version": VALIDATOR_VERSION,
                    "validator_sha256": validator_sha256(),
                    "detected_signature": signature,
                    "artifact_sha256": _sha256(artifact),
                },
                "status": "programmatically_verified_seed",
                "created_before_test": True,
            })
    MEMORY_ROOT.mkdir(parents=True, exist_ok=True)
    path = MEMORY_ROOT / "verified_seed.jsonl"
    path.write_text("".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n" for item in records), encoding="utf-8")
    signature_counts = Counter(item["failure_signature"] for item in records)
    channel_counts = Counter(item["failure_channel"] for item in records)
    manifest = {
        "schema_version": "2.0",
        "record_count": len(records),
        "source_splits": list(SOURCE_SPLITS),
        "excluded_splits": ["test"],
        "mutations_per_episode": MUTATIONS_PER_EPISODE,
        "failure_signature_counts": dict(sorted(signature_counts.items())),
        "failure_channel_counts": dict(sorted(channel_counts.items())),
        "validator_version": VALIDATOR_VERSION,
        "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (MEMORY_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _counterpart(role: str) -> str:
    return "building_energy_manager" if role == "distribution_grid_operator" else "distribution_grid_operator"


def _sha256(payload: object) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, sort_keys=True))
