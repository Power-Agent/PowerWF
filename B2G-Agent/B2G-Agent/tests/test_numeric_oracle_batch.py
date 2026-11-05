import json
from pathlib import Path

from experiments.run_numeric_oracle_batch import evidence_fingerprints, workflow_for
from experiments.run_flexibility_envelope_episode import envelope_outcome


def test_baseline_flexibility_routes_to_bidirectional_workflow() -> None:
    contract = {
        "task_specification": {
            "task_families": ["baseline_flexibility", "reserve_service"]
        }
    }
    assert workflow_for(contract) == "bidirectional_flexibility_envelope"


def test_other_families_route_to_standard_closed_loop() -> None:
    contract = {
        "task_specification": {
            "task_families": ["renewable_absorption", "peak_shaving"]
        }
    }
    assert workflow_for(contract) == "closed_loop"


def test_evidence_reuse_requires_config_and_implementation_hashes(
    tmp_path: Path,
) -> None:
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "input_hashes": {
                    "joint_dispatch_config": "config-hash",
                    "physics_implementation_bundle": "code-hash",
                }
            }
        ),
        encoding="utf-8",
    )
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            {
                "iterations": [
                    {"coupled_validation": {"path": validation.name}}
                ]
            }
        ),
        encoding="utf-8",
    )

    assert evidence_fingerprints(evidence) == {("config-hash", "code-hash")}


def test_envelope_rejects_when_either_direction_fails_hard_gates() -> None:
    outcome = envelope_outcome(
        reduction_committed_kw=120.0,
        absorption_committed_kw=80.0,
        reduction_completion=100.0,
        absorption_completion=75.0,
        reduction_release_eligible=True,
        absorption_release_eligible=False,
        reduction_full_request_satisfied=True,
        absorption_full_request_satisfied=False,
    )

    assert outcome["release_eligible"] is False
    assert outcome["receiver_decision"] == "rejected"
    assert outcome["completion_percentage"] == 0.0
    assert outcome["validated_downward_flexibility_kw"] == 0.0
    assert outcome["validated_upward_flexibility_kw"] == 0.0
    assert outcome["symmetric_envelope_capacity_kw"] == 0.0


def test_envelope_modifies_only_after_both_directional_hard_gates_pass() -> None:
    outcome = envelope_outcome(
        reduction_committed_kw=120.0,
        absorption_committed_kw=80.0,
        reduction_completion=100.0,
        absorption_completion=75.0,
        reduction_release_eligible=True,
        absorption_release_eligible=True,
        reduction_full_request_satisfied=True,
        absorption_full_request_satisfied=False,
    )

    assert outcome["release_eligible"] is True
    assert outcome["receiver_decision"] == "modified"
    assert outcome["completion_percentage"] == 75.0
    assert outcome["symmetric_envelope_capacity_kw"] == 80.0
