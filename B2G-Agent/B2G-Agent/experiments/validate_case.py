"""Validate committed contracts and, optionally, all local physics assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.case_io import CaseBundle  # noqa: E402
from b2g_agent.contracts import TaskFamily  # noqa: E402
from b2g_agent.observability import physics_implementation_bundle_sha256  # noqa: E402


FAMILY_OUTPUTS = {
    TaskFamily.PEAK_SHAVING: {
        "feeder_peak_kw", "delivered_reduction_kw", "comfort_violation_minutes"
    },
    TaskFamily.CONGESTION_MANAGEMENT: {
        "maximum_line_loading_pct", "overload_intervals"
    },
    TaskFamily.VOLTAGE_MANAGEMENT: {
        "minimum_voltage_pu", "maximum_voltage_pu", "violation_intervals"
    },
    TaskFamily.RESERVE_SERVICE: {
        "reserved_capacity_kw", "delivery_ratio", "rebound_kw"
    },
    TaskFamily.RENEWABLE_ABSORPTION: {
        "pv_curtailment_kwh", "absorbed_energy_kwh"
    },
    TaskFamily.BASELINE_FLEXIBILITY: {
        "counterfactual_baseline_kw", "flexibility_lower_kw", "flexibility_upper_kw"
    },
    TaskFamily.MARKET_TRADING: {
        "accepted_capacity_kw", "settlement_usd", "delivery_ratio"
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-physics", action="store_true")
    parser.add_argument(
        "--require-publishable",
        action="store_true",
        help="Fail unless all contract and numerical oracles are verified.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def interval_indices(start: str, end: str) -> set[int]:
    if start == "00:00" and end == "23:45":
        return set(range(96))
    start_hour, start_minute = (int(part) for part in start.split(":"))
    end_hour, end_minute = (int(part) for part in end.split(":"))
    first = (start_hour * 60 + start_minute) // 15
    last = (end_hour * 60 + end_minute) // 15
    return set(range(first, last))


def contract_semantic_sha256(truth: object) -> str:
    payload = truth.model_dump(mode="json")
    payload.pop("contract_truth_status", None)
    payload.pop("physics_truth_status", None)
    payload.pop("numeric_oracle_path", None)
    # Runtime retrieval diagnostics are not part of the frozen task semantics.
    payload.get("task_specification", {}).pop("retrieval_trace", None)
    canonical = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_oracle_design(episodes: list[object], truths: list[object]) -> list[str]:
    """Check independent invariants behind the generated contract oracles."""

    errors: list[str] = []
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    for truth in truths:
        episode = episode_by_id.get(truth.episode_id)
        if episode is None:
            continue
        specification = truth.task_specification
        expected_outputs: set[str] = set()
        for family in specification.task_families:
            expected_outputs.update(FAMILY_OUTPUTS[family])
        if set(specification.requested_outputs) != expected_outputs:
            errors.append(f"contract_output_mapping_mismatch:{truth.episode_id}")
        if truth.perfect_execution.route == "execute_physics_then_release":
            if set(truth.perfect_execution.required_numeric_outputs) != expected_outputs:
                errors.append(f"required_numeric_output_mismatch:{truth.episode_id}")
        elif truth.perfect_execution.required_numeric_outputs:
            errors.append(f"clarification_has_numeric_outputs:{truth.episode_id}")
        if specification.windows != episode.windows:
            errors.append(f"episode_window_mismatch:{truth.episode_id}")
        if specification.target_bems != episode.target_bems:
            errors.append(f"episode_bem_mismatch:{truth.episode_id}")
        if specification.target_buses != episode.target_buses:
            errors.append(f"episode_bus_mismatch:{truth.episode_id}")
        if "rebound_kw" in expected_outputs and any(
            window.start == "00:00" and window.end == "23:45"
            for window in specification.windows
        ):
            errors.append(f"rebound_undefined_for_full_day:{truth.episode_id}")
        has_absorption = TaskFamily.RENEWABLE_ABSORPTION in specification.task_families
        has_reduction = any(
            family != TaskFamily.RENEWABLE_ABSORPTION
            for family in specification.task_families
        )
        if has_absorption and has_reduction:
            if len(specification.windows) != len(specification.task_families):
                errors.append(f"mixed_direction_window_binding:{truth.episode_id}")
            elif len(specification.windows) > 1:
                occupied: set[int] = set()
                for window in specification.windows:
                    current = interval_indices(window.start, window.end)
                    if occupied & current:
                        errors.append(f"mixed_direction_window_overlap:{truth.episode_id}")
                        break
                    occupied.update(current)
    return errors


def main() -> int:
    args = parse_args()
    case = CaseBundle.default(REPO_ROOT)
    episodes = list(case.iter_episodes())
    truths = list(case.iter_groundtruth())
    errors: list[str] = []
    if len(episodes) != 200:
        errors.append(f"episode_count={len(episodes)}")
    if len({episode.episode_id for episode in episodes}) != 200:
        errors.append("duplicate_episode_id")
    if len(truths) != 200:
        errors.append(f"groundtruth_count={len(truths)}")
    errors.extend(case.validate_groundtruth_alignment())
    errors.extend(validate_oracle_design(episodes, truths))
    raw_episodes = [
        json.loads(line)
        for line in case.episodes_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if any("ground_truth" in record for record in raw_episodes):
        errors.append("groundtruth_leaked_into_public_task_corpus")
    if len(case.bem_mapping()) != 10:
        errors.append("bem_mapping_count")
    for required in [
        case.case_dir / "network" / "source_case33bw.json",
        case.case_dir / "network" / "study_network.json",
        case.case_dir / "tasks" / "summary.json",
        case.case_dir / "groundtruth" / "manifest.json",
        case.groundtruth_path,
        case.case_dir / "groundtruth" / "contract_verification.jsonl",
        case.case_dir / "reviewer_memory" / "verified_seed.jsonl",
        case.case_dir / "reviewer_memory" / "manifest.json",
    ]:
        if not required.is_file():
            errors.append(f"missing:{required.relative_to(REPO_ROOT)}")

    memory_path = case.case_dir / "reviewer_memory" / "verified_seed.jsonl"
    memory_manifest_path = case.case_dir / "reviewer_memory" / "manifest.json"
    if memory_path.is_file() and memory_manifest_path.is_file():
        memory_records = [
            json.loads(line)
            for line in memory_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        memory_manifest = json.loads(memory_manifest_path.read_text(encoding="utf-8"))
        if len(memory_records) != int(memory_manifest.get("record_count", -1)):
            errors.append("reviewer_memory_count_mismatch")
        if hashlib.sha256(memory_path.read_bytes()).hexdigest() != memory_manifest.get("sha256"):
            errors.append("reviewer_memory_hash_mismatch")
        if any(record.get("source_split") == "test" for record in memory_records):
            errors.append("reviewer_memory_contains_test_split")

    local_manifest = case.local_data_root / case.case_id / "local_asset_manifest.json"
    local_checks: dict[str, object] = {"manifest_exists": local_manifest.is_file()}
    if local_manifest.is_file():
        manifest = json.loads(local_manifest.read_text(encoding="utf-8"))
        records = manifest.get("assets", [])
        local_checks["asset_records"] = len(records)
        idf_count = 0
        for record in records:
            runtime_relative = record.get("runtime_path")
            if not runtime_relative:
                continue
            runtime = case.local_data_root / runtime_relative
            if not runtime.is_file():
                errors.append(f"missing_runtime_idf:{runtime_relative}")
                continue
            if sha256(runtime) != record.get("runtime_idf_sha256"):
                errors.append(f"hash_mismatch:{runtime_relative}")
            text = runtime.read_text(encoding="utf-8", errors="replace")
            if not re.search(r"(?is)Version\s*,\s*23\.1\s*;", text):
                errors.append(f"wrong_energyplus_version:{runtime_relative}")
            if not re.search(r"(?is)Timestep\s*,\s*4\s*;", text):
                errors.append(f"wrong_timestep:{runtime_relative}")
            idf_count += 1
        local_checks["metadata_validated_runtime_idfs"] = idf_count

    smoke_report_path = case.local_data_root / case.case_id / "energyplus_smoke_report.json"
    local_checks["smoke_report_exists"] = smoke_report_path.is_file()
    if smoke_report_path.is_file():
        smoke_report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
        local_checks["energyplus_smoke_status"] = smoke_report.get("status")
        local_checks["energyplus_smoke_blocked_bems"] = smoke_report.get(
            "blocked_bems", []
        )

    readiness = case.readiness()
    if args.require_physics and not readiness.ready_for_physics:
        errors.extend(f"physics_missing:{item}" for item in readiness.missing_local_assets)
    verification_path = case.case_dir / "groundtruth" / "contract_verification.jsonl"
    verification_records = (
        [
            json.loads(line)
            for line in verification_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if verification_path.is_file()
        else []
    )
    verification_by_id = {
        record.get("episode_id"): record for record in verification_records
    }
    if len(verification_by_id) != len(verification_records):
        errors.append("duplicate_contract_verification_episode_id")
    contract_programmatically_verified = 0
    for truth in truths:
        verification = verification_by_id.get(truth.episode_id)
        if verification is None:
            errors.append(f"missing_contract_verification:{truth.episode_id}")
            continue
        if verification.get("status") != "programmatically_verified":
            errors.append(f"contract_verification_pending:{truth.episode_id}")
            continue
        if verification.get("contract_semantic_sha256") != contract_semantic_sha256(
            truth
        ):
            errors.append(f"contract_verification_stale:{truth.episode_id}")
            continue
        contract_programmatically_verified += 1
    extra_verification = sorted(set(verification_by_id) - {t.episode_id for t in truths})
    errors.extend(
        f"contract_verification_without_contract:{episode_id}"
        for episode_id in extra_verification
    )
    contract_pending = len(truths) - contract_programmatically_verified
    physics_pending = sum(truth.physics_truth_status == "pending" for truth in truths)
    physics_verified = sum(truth.physics_truth_status == "verified" for truth in truths)
    numeric_directory = case.case_dir / "groundtruth" / "numeric_oracles"
    numeric_files = {path.stem: path for path in numeric_directory.glob("ep-*.json")}
    groundtruth_manifest = json.loads(
        (case.case_dir / "groundtruth" / "manifest.json").read_text(encoding="utf-8")
    )
    task_summary = json.loads(
        (case.case_dir / "tasks" / "summary.json").read_text(encoding="utf-8")
    )
    episodes_hash = sha256(case.episodes_path)
    groundtruth_hash = sha256(case.groundtruth_path)
    if groundtruth_manifest.get("episodes_sha256") != episodes_hash:
        errors.append("groundtruth_manifest_episode_hash_mismatch")
    if groundtruth_manifest.get("contract_oracles_sha256") != groundtruth_hash:
        errors.append("groundtruth_manifest_contract_hash_mismatch")
    if verification_path.is_file() and (
        groundtruth_manifest.get("contract_verification_sha256")
        != sha256(verification_path)
    ):
        errors.append("groundtruth_manifest_contract_verification_hash_mismatch")
    expected_physics_counts = {
        "not_required_for_current_turn": sum(
            truth.physics_truth_status == "not_required_for_current_turn"
            for truth in truths
        ),
        "pending": physics_pending,
        "verified": physics_verified,
    }
    if task_summary.get("groundtruth_contract_sha256") != groundtruth_hash:
        errors.append("task_summary_contract_hash_mismatch")
    if task_summary.get("physics_truth_status_counts") != expected_physics_counts:
        errors.append("task_summary_physics_counts_mismatch")
    if groundtruth_manifest.get("physics_verified_count") != physics_verified:
        errors.append("groundtruth_manifest_verified_count_mismatch")
    if groundtruth_manifest.get("physics_pending_count") != physics_pending:
        errors.append("groundtruth_manifest_pending_count_mismatch")
    if (
        groundtruth_manifest.get("programmatically_verified_numeric_count")
        != physics_verified
    ):
        errors.append("groundtruth_manifest_numeric_verification_count_mismatch")
    if (
        groundtruth_manifest.get("programmatically_verified_contract_count")
        != contract_programmatically_verified
    ):
        errors.append("groundtruth_manifest_contract_verification_count_mismatch")
    manifest_numeric = {
        record["episode_id"]: record for record in groundtruth_manifest["numeric_oracles"]
    }
    stable_inputs = {
        "background": case.local_data_root
        / case.case_id
        / "profiles"
        / "background_load_15min.csv",
        "bem_baseline": case.local_data_root
        / case.case_id
        / "profiles"
        / "bem_baseline_15min.csv",
        "bem_thermal": case.local_data_root
        / case.case_id
        / "profiles"
        / "bem_thermal_15min.csv",
        "pv": case.local_data_root / case.case_id / "profiles" / "pv_15min.csv",
        "prices": case.local_data_root
        / case.case_id
        / "profiles"
        / "prices_15min.csv",
        "target_days": case.case_dir / "days" / "target_days.json",
        "joint_dispatch_config": case.case_dir
        / "optimization"
        / "joint_dispatch.json",
    }
    current_physics_bundle = physics_implementation_bundle_sha256(REPO_ROOT)
    for truth in truths:
        if truth.physics_truth_status == "verified":
            numeric_path = numeric_files.get(truth.episode_id)
            if numeric_path is None:
                errors.append(f"missing_numeric_oracle:{truth.episode_id}")
                continue
            numeric = json.loads(numeric_path.read_text(encoding="utf-8"))
            if numeric.get("episode_id") != truth.episode_id:
                errors.append(f"numeric_oracle_id_mismatch:{truth.episode_id}")
            if (
                args.require_publishable
                and numeric.get("review_status") != "programmatically_verified"
            ):
                errors.append(f"numeric_oracle_review_pending:{truth.episode_id}")
            manifest_record = manifest_numeric.get(truth.episode_id)
            if manifest_record is None:
                errors.append(f"numeric_oracle_missing_from_manifest:{truth.episode_id}")
            elif manifest_record.get("sha256") != sha256(numeric_path):
                errors.append(f"numeric_oracle_manifest_hash_mismatch:{truth.episode_id}")
            business = numeric.get("business_metrics", {})
            requested_values = business.get("requested_output_values", {})
            for field in truth.perfect_execution.required_numeric_outputs:
                if field not in requested_values:
                    errors.append(
                        f"numeric_oracle_missing_required_output:{truth.episode_id}:{field}"
                    )
                elif requested_values[field] is None:
                    errors.append(
                        f"numeric_oracle_null_required_output:{truth.episode_id}:{field}"
                    )
            final = numeric.get("final_decision", {})
            if not final.get("expected_receiver_feedback"):
                errors.append(f"numeric_oracle_missing_receiver_feedback:{truth.episode_id}")
            if not final.get("expected_sender_final"):
                errors.append(f"numeric_oracle_missing_sender_final:{truth.episode_id}")
            decision = final.get("receiver_decision")
            completion = final.get("completion_percentage")
            if decision == "accepted" and completion != 100.0:
                errors.append(f"accepted_oracle_not_complete:{truth.episode_id}")
            if decision == "rejected" and completion != 0.0:
                errors.append(f"rejected_oracle_nonzero_completion:{truth.episode_id}")
            hash_groups = (
                list(numeric.get("input_hashes_by_direction", {}).values())
                if "input_hashes_by_direction" in numeric
                else [numeric.get("input_hashes", {})]
            )
            for input_hashes in hash_groups:
                if (
                    input_hashes.get("physics_implementation_bundle")
                    != current_physics_bundle
                ):
                    errors.append(
                        f"numeric_oracle_stale_input:{truth.episode_id}:physics_implementation_bundle"
                    )
                for key, path in stable_inputs.items():
                    if key not in input_hashes or not path.is_file():
                        continue
                    if input_hashes[key] != sha256(path):
                        errors.append(
                            f"numeric_oracle_stale_input:{truth.episode_id}:{key}"
                        )
    extra_numeric = sorted(
        set(numeric_files)
        - {truth.episode_id for truth in truths if truth.physics_truth_status == "verified"}
    )
    errors.extend(f"numeric_oracle_without_verified_contract:{item}" for item in extra_numeric)
    if args.require_publishable:
        if contract_pending:
            errors.append(f"contract_oracles_pending_verification:{contract_pending}")
        if physics_pending:
            errors.append(f"numeric_oracles_pending:{physics_pending}")
    report = {
        "case_id": case.case_id,
        "contract_status": "PASS" if not errors else "FAIL",
        "episode_count": len(episodes),
        "groundtruth_count": len(truths),
        "episode_sha256": episodes_hash,
        "groundtruth_sha256": groundtruth_hash,
        "contract_oracles_pending_verification": contract_pending,
        "contract_oracles_programmatically_verified": (
            contract_programmatically_verified
        ),
        "numeric_oracles_pending": physics_pending,
        "numeric_oracles_verified": physics_verified,
        "publishable_groundtruth": contract_pending == 0 and physics_pending == 0,
        "physics_ready": readiness.ready_for_physics,
        "missing_physics_assets": list(readiness.missing_local_assets),
        "local_checks": local_checks,
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
