"""Freeze one closed-loop evidence package as a committed numeric oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_DIR = REPO_ROOT / "cases" / "ieee33_10bem_15min"
GROUNDTRUTH_DIR = CASE_DIR / "groundtruth"
CONTRACT_PATH = GROUNDTRUTH_DIR / "contract_oracles.jsonl"
MANIFEST_PATH = GROUNDTRUTH_DIR / "manifest.json"
TASK_SUMMARY_PATH = CASE_DIR / "tasks" / "summary.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-package", type=Path, required=True)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace an existing numeric oracle for the same episode after revalidation.",
    )
    return parser.parse_args()


def build_standard_numeric(
    evidence_path: Path,
    evidence: dict[str, object],
    contract: dict[str, object],
) -> dict[str, object]:
    final_iteration = evidence["iterations"][-1]
    evidence_root = evidence_path.parent
    optimizer_path = evidence_root / final_iteration["optimizer_result"]["path"]
    validation_path = evidence_root / final_iteration["coupled_validation"]["path"]
    optimizer = json.loads(optimizer_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    release_eligible = bool(validation["release_eligible"])
    requested = float(optimizer["requested_power_kw"])
    candidate_committed = float(
        validation.get(
            "candidate_committed_power_kw", validation["validated_committed_power_kw"]
        )
    )
    committed = float(validation["validated_committed_power_kw"])
    if not release_eligible:
        committed = 0.0
    completion = float(validation["validated_completion_percentage"])
    decision = validation["receiver_decision"]
    if decision == "accepted":
        completion = 100.0
    elif decision == "rejected":
        completion = 0.0
    average_delivery = float(
        validation["business_metrics"]["event_average_directed_delivery_kw"]
    )
    delivered_energy = float(validation["business_metrics"]["delivered_energy_kwh"])
    if not release_eligible:
        receiver_feedback = (
            f"After joint optimization, EnergyPlus response checking, nonlinear AC power-flow "
            f"validation, and {evidence['replan_count']} replan, we rejected the request because "
            f"one or more physical hard gates failed. No schedule or capacity is released."
        )
        sender_final = (
            f"The receiver rejected the {requested:.3f} kW request after physical validation. "
            f"No capacity is released and the task is 0.000% complete."
        )
    elif decision == "modified":
        receiver_feedback = (
            f"After joint optimization, EnergyPlus response checking, nonlinear AC power-flow "
            f"validation, and {evidence['replan_count']} replan, we modified the request. "
            f"No nonzero uniform capacity is guaranteed across every requested interval "
            f"({committed:.3f} kW sustained); the validated schedule delivers "
            f"{delivered_energy:.3f} kWh, or {average_delivery:.3f} kW on average. "
            f"Voltage, line, transformer, storage, and response-comfort constraints pass."
        )
        sender_final = (
            f"The receiver cannot guarantee a uniform {requested:.3f} kW response, but its "
            f"validated interval schedule delivers {delivered_energy:.3f} kWh "
            f"({average_delivery:.3f} kW average). The task is {completion:.3f}% complete."
        )
    else:
        receiver_feedback = (
            f"After joint optimization, EnergyPlus response checking, nonlinear AC power-flow "
            f"validation, and {evidence['replan_count']} replan, we {decision} the request. "
            f"The validated commitment is {committed:.3f} kW; voltage, line, transformer, "
            f"storage, and response-comfort constraints pass."
        )
        sender_final = (
            f"The receiver's validated commitment is {committed:.3f} kW against the "
            f"{requested:.3f} kW request. The task is {completion:.3f}% complete."
        )
    return {
        "schema_version": "1.0",
        "episode_id": evidence["episode_id"],
        "case_id": contract["case_id"],
        "physics_truth_status": "verified",
        "verification_chain": "joint RC-OPF candidate, EnergyPlus event response when required, nonlinear pandapower, deterministic replan",
        "replan_count": evidence["replan_count"],
        "final_decision": {
            "receiver_decision": decision,
            "release_eligible": release_eligible,
            "requested_power_kw": requested,
            "candidate_committed_power_kw": candidate_committed,
            "committed_power_kw": committed,
            "event_average_delivered_power_kw": average_delivery,
            "delivered_energy_kwh": delivered_energy,
            "completion_percentage": completion,
            "expected_receiver_feedback": receiver_feedback,
            "expected_sender_final": sender_final,
        },
        "validated_schedule_by_bem": optimizer["schedules"] if release_eligible else {},
        "candidate_schedule_by_bem": optimizer["schedules"],
        "validated_metrics": {
            "minimum_voltage_pu": validation["nonlinear_metrics"]["minimum_voltage_pu"],
            "maximum_voltage_pu": validation["nonlinear_metrics"]["maximum_voltage_pu"],
            "maximum_line_loading_pct": validation["nonlinear_metrics"][
                "maximum_line_loading_pct"
            ],
            "maximum_transformer_loading_pct": validation["nonlinear_metrics"][
                "maximum_transformer_loading_pct"
            ],
            "comfort_violation_intervals": validation["comfort_violation_intervals"],
            "baseline_absolute_comfort_violation_intervals": validation[
                "baseline_absolute_comfort_violation_intervals"
            ],
            "maximum_remaining_service_gap_kw": validation[
                "maximum_remaining_service_gap_kw"
            ],
        },
        "business_metrics": validation["business_metrics"],
        "grid_violations": validation["grid_violations"],
        "tool_versions": optimizer["versions"],
        "input_hashes": validation["input_hashes"],
        "artifact_hashes": {
            "closed_loop_evidence_package": sha256_file(evidence_path),
            "final_optimizer_result": sha256_file(optimizer_path),
            "final_coupled_validation": sha256_file(validation_path),
        },
        "reproduction_command": (
            f"python experiments/run_closed_loop_episode.py --episode {evidence['episode_id']}"
        ),
        "review_status": "programmatically_verified",
        "review_checks": [
            "physics_implementation_bundle",
            "stable_input_hashes",
            "required_output_definedness",
            "natural_language_completion",
        ],
    }


def build_envelope_numeric(
    evidence_path: Path,
    evidence: dict[str, object],
    contract: dict[str, object],
) -> dict[str, object]:
    direction_payloads: dict[str, dict[str, object]] = {}
    candidate_schedules: dict[str, object] = {}
    validated_schedules: dict[str, object] = {}
    validations: dict[str, dict[str, object]] = {}
    optimizers: dict[str, dict[str, object]] = {}
    artifact_hashes = {"envelope_evidence_package": sha256_file(evidence_path)}
    for direction in ("reduction", "absorption"):
        record = evidence["directions"][direction]
        child_path = evidence_path.parent / record["evidence_package"]
        child = json.loads(child_path.read_text(encoding="utf-8"))
        final_iteration = child["iterations"][-1]
        optimizer_path = child_path.parent / final_iteration["optimizer_result"]["path"]
        validation_path = child_path.parent / final_iteration["coupled_validation"]["path"]
        optimizer = json.loads(optimizer_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        direction_payloads[direction] = child
        candidate_schedules[direction] = optimizer["schedules"]
        validated_schedules[direction] = (
            optimizer["schedules"] if validation["release_eligible"] else {}
        )
        validations[direction] = validation
        optimizers[direction] = optimizer
        artifact_hashes[f"{direction}_closed_loop_evidence"] = sha256_file(child_path)
        artifact_hashes[f"{direction}_optimizer_result"] = sha256_file(optimizer_path)
        artifact_hashes[f"{direction}_coupled_validation"] = sha256_file(validation_path)

    requested = float(optimizers["reduction"]["requested_power_kw"])
    business = evidence["business_metrics"]
    committed = float(business["symmetric_envelope_capacity_kw"])
    completion = float(evidence["completion_percentage"])
    decision = evidence["receiver_decision"]
    release_eligible = bool(business["envelope_release_eligible"])
    if decision == "accepted":
        completion = 100.0
    elif decision == "rejected":
        completion = 0.0
    if not release_eligible:
        receiver_feedback = (
            "After separate reduction and absorption validation, we rejected the "
            "bidirectional offer because at least one directional physical hard gate failed. "
            "No bidirectional envelope is released."
        )
    else:
        receiver_feedback = (
            f"After separate reduction and absorption validation, we {decision} the "
            f"bidirectional offer. The validated symmetric envelope is {committed:.3f} kW, "
            f"with lower and upper net-demand bounds of "
            f"{float(business['flexibility_lower_kw']):.3f} kW and "
            f"{float(business['flexibility_upper_kw']):.3f} kW."
        )
    sender_final = (
        f"The receiver validated {committed:.3f} kW of the {requested:.3f} kW "
        f"bidirectional flexibility offer. The task is {completion:.3f}% complete."
    )
    return {
        "schema_version": "1.0",
        "episode_id": evidence["episode_id"],
        "case_id": contract["case_id"],
        "physics_truth_status": "verified",
        "verification_chain": "bidirectional joint RC-OPF, EnergyPlus response checking when required, nonlinear pandapower, deterministic replan",
        "replan_count": {
            direction: int(direction_payloads[direction]["replan_count"])
            for direction in direction_payloads
        },
        "final_decision": {
            "receiver_decision": decision,
            "release_eligible": release_eligible,
            "requested_power_kw": requested,
            "committed_power_kw": committed,
            "completion_percentage": completion,
            "expected_receiver_feedback": receiver_feedback,
            "expected_sender_final": sender_final,
        },
        "validated_schedule_by_direction_and_bem": validated_schedules,
        "candidate_schedule_by_direction_and_bem": candidate_schedules,
        "validated_metrics_by_direction": {
            direction: {
                "nonlinear_metrics": validations[direction]["nonlinear_metrics"],
                "grid_violations": validations[direction]["grid_violations"],
                "comfort_violation_intervals": validations[direction][
                    "comfort_violation_intervals"
                ],
            }
            for direction in validations
        },
        "business_metrics": business,
        "tool_versions_by_direction": {
            direction: optimizers[direction]["versions"] for direction in optimizers
        },
        "input_hashes_by_direction": {
            direction: validations[direction]["input_hashes"] for direction in validations
        },
        "artifact_hashes": artifact_hashes,
        "reproduction_command": (
            f"python experiments/run_flexibility_envelope_episode.py --episode "
            f"{evidence['episode_id']}"
        ),
        "review_status": "programmatically_verified",
        "review_checks": [
            "physics_implementation_bundle",
            "stable_input_hashes",
            "required_output_definedness",
            "natural_language_completion",
        ],
    }


def main() -> None:
    args = parse_args()
    evidence_path = args.evidence_package.resolve()
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if evidence["status"] not in {"validated_full", "validated_partial"}:
        raise ValueError("Only a physically validated evidence package can be frozen")
    episode_id = evidence["episode_id"]
    rows = [json.loads(line) for line in CONTRACT_PATH.read_text(encoding="utf-8").splitlines()]
    matching = [row for row in rows if row["episode_id"] == episode_id]
    if len(matching) != 1:
        raise ValueError(f"Expected one contract oracle for {episode_id}")
    contract = matching[0]
    if contract["physics_truth_status"] != "pending" and not args.replace:
        raise ValueError(f"{episode_id} is not pending numeric verification")
    numeric = (
        build_envelope_numeric(evidence_path, evidence, contract)
        if evidence.get("workflow") == "bidirectional_flexibility_envelope"
        else build_standard_numeric(evidence_path, evidence, contract)
    )
    output_path = GROUNDTRUTH_DIR / "numeric_oracles" / f"{episode_id}.json"
    if output_path.exists() and not args.replace:
        raise FileExistsError(output_path)
    if args.replace and not output_path.exists():
        raise FileNotFoundError(
            f"--replace was requested but no existing oracle was found: {output_path}"
        )
    output_path.write_text(json.dumps(numeric, indent=2) + "\n", encoding="utf-8")

    contract["physics_truth_status"] = "verified"
    CONTRACT_PATH.write_text(
        "\n".join(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")) for row in rows
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["physics_verified_count"] = sum(
        row["physics_truth_status"] == "verified" for row in rows
    )
    manifest["physics_pending_count"] = sum(
        row["physics_truth_status"] == "pending" for row in rows
    )
    manifest["physics_not_required_count"] = sum(
        row["physics_truth_status"] == "not_required_for_current_turn" for row in rows
    )
    manifest["programmatically_verified_numeric_count"] = manifest[
        "physics_verified_count"
    ]
    manifest["contract_oracles_sha256"] = sha256_file(CONTRACT_PATH)
    manifest["numeric_oracles"] = [
        {
            "episode_id": path.stem,
            "path": f"numeric_oracles/{path.name}",
            "sha256": sha256_file(path),
        }
        for path in sorted((GROUNDTRUTH_DIR / "numeric_oracles").glob("ep-*.json"))
    ]
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    task_summary = json.loads(TASK_SUMMARY_PATH.read_text(encoding="utf-8"))
    task_summary["groundtruth_contract_sha256"] = manifest[
        "contract_oracles_sha256"
    ]
    task_summary["physics_truth_status_counts"] = {
        "not_required_for_current_turn": manifest["physics_not_required_count"],
        "pending": manifest["physics_pending_count"],
        "verified": manifest["physics_verified_count"],
    }
    TASK_SUMMARY_PATH.write_text(
        json.dumps(task_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "numeric_oracle": str(output_path),
                "sha256": sha256_file(output_path),
                "physics_verified_count": manifest["physics_verified_count"],
                "physics_pending_count": manifest["physics_pending_count"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
