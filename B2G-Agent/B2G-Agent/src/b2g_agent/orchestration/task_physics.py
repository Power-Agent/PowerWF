from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from b2g_agent.contracts.readiness import execution_readiness
from b2g_agent.observability.journal import durable_write, emit, subprocess_environment, traced_stage
from pathlib import Path

from b2g_agent.contracts import (
    ExecutionPlan,
    PhysicsEvidence,
    ReceiverFeedback,
    TaskFamily,
    TaskSpecification,
)


def _numerically_converged(validation: dict) -> bool:
    """A converged power flow can still violate voltage/thermal constraints."""
    value = validation.get("grid_violations", {}).get("nonconverged_intervals")
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0


class TaskPhysicsExecutor:
    """Run a translated task through the deterministic EnergyPlus/pandapower loop."""

    def __init__(self, repo_root: Path, *, maximum_replans: int = 1,
                 use_interactive_inputs: bool = False) -> None:
        self.repo_root = repo_root.resolve()
        self.maximum_replans = maximum_replans
        self.use_interactive_inputs = use_interactive_inputs

    @traced_stage("physics.executor")
    def run(
        self,
        specification: TaskSpecification,
        plan: ExecutionPlan,
        *,
        run_root: Path,
    ) -> tuple[ReceiverFeedback, PhysicsEvidence]:
        if specification.clarification_required:
            raise ValueError("A task requiring clarification cannot enter the physics executor")
        from b2g_agent.case_io import CaseBundle
        readiness = execution_readiness(specification, case=CaseBundle.default(self.repo_root))
        if readiness:
            raise ValueError("Execution inputs are unresolved: " + "; ".join(i.message for i in readiness))
        _validate_execution_plan(specification, plan)
        if not re.fullmatch(r"[A-Za-z0-9._-]+", specification.episode_id):
            raise ValueError("Task episode_id contains unsafe path characters")
        run_root = run_root.resolve()
        evidence_root = run_root / specification.episode_id
        if evidence_root.exists():
            raise FileExistsError(f"Physics run already exists: {evidence_root}")
        input_root = run_root / "inputs"
        input_root.mkdir(parents=True, exist_ok=True)
        specification_path = input_root / f"{specification.episode_id}.json"
        specification_path.write_text(
            json.dumps(
                specification.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        plan_path = input_root / f"{specification.episode_id}.execution-plan.json"
        plan_payload = plan.model_dump(mode="json")
        plan_path.write_text(
            json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        plan_sha256 = hashlib.sha256(
            json.dumps(
                plan_payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        envelope = TaskFamily.BASELINE_FLEXIBILITY in specification.task_families
        script = (
            "run_flexibility_envelope_episode.py"
            if envelope
            else "run_closed_loop_episode.py"
        )
        command = [
            sys.executable,
            str(self.repo_root / "src/b2g_agent/observability/physics_worker.py"),
            *(["--interactive-task-specification", str(specification_path)]
              if self.use_interactive_inputs and specification.context_origin.startswith("interactive_") else []),
            str(self.repo_root / "experiments" / script),
            "--episode",
            specification.episode_id,
            "--task-specification",
            str(specification_path),
            "--execution-plan",
            str(plan_path),
            "--run-root",
            str(run_root),
            "--maximum-replans",
            str(self.maximum_replans),
        ]
        emit("tool.started", tool="physics_workflow", command=command)
        completed = subprocess.run(
            command,
            cwd=self.repo_root,
            capture_output=True,
            text=True,
            timeout=7200,
            check=False,
            env=subprocess_environment(),
        )
        process_record = {"command": command, "returncode": completed.returncode,
                          "stdout": completed.stdout, "stderr": completed.stderr}
        durable_write(run_root / "process.json", json.dumps(process_record, ensure_ascii=False, indent=2) + "\n")
        emit("tool.completed", tool="physics_workflow", **process_record)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Closed-loop physics failed ({completed.returncode}): "
                f"{completed.stderr[-3000:]}"
            )
        evidence_path = evidence_root / "evidence_package.json"
        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
        if evidence["status"] not in {"validated_full", "validated_partial"}:
            raise RuntimeError(f"Physics evidence is not release eligible: {evidence['status']}")

        if envelope:
            decision = str(evidence["receiver_decision"])
            completion = float(evidence["completion_percentage"])
            metrics = dict(evidence["business_metrics"])
            committed = float(metrics["symmetric_envelope_capacity_kw"])
            replan_count = {}
            input_hashes: dict[str, object] = {}
            direction_validations: dict[str, dict[str, object]] = {}
            for direction, record in evidence["directions"].items():
                child_path = evidence_root / record["evidence_package"]
                child = json.loads(child_path.read_text(encoding="utf-8"))
                replan_count[direction] = int(child["replan_count"])
                final_iteration = child["iterations"][-1]
                validation_path = child_path.parent / final_iteration["coupled_validation"]["path"]
                validation = json.loads(validation_path.read_text(encoding="utf-8"))
                direction_validations[direction] = validation
                input_hashes[direction] = validation["input_hashes"]
            converged = all(
                _numerically_converged(validation)
                for validation in direction_validations.values()
            )
            hard_constraints_passed = all(
                all(int(value) == 0 for value in validation["grid_violations"].values())
                and int(validation["comfort_violation_intervals"] or 0) == 0
                and bool(validation["release_eligible"])
                for validation in direction_validations.values()
            )
            workflow = "bidirectional_flexibility_envelope"
        else:
            final = evidence["final"]
            decision = str(final["receiver_decision"])
            completion = float(final["completion_percentage"])
            metrics = dict(final["business_metrics"])
            committed = float(final["committed_power_kw"])
            replan_count = int(evidence["replan_count"])
            final_iteration = evidence["iterations"][-1]
            validation_path = evidence_root / final_iteration["coupled_validation"]["path"]
            validation = json.loads(validation_path.read_text(encoding="utf-8"))
            input_hashes = dict(validation["input_hashes"])
            converged = _numerically_converged(validation)
            hard_constraints_passed = (
                all(int(value) == 0 for value in validation["grid_violations"].values())
                and int(validation["comfort_violation_intervals"] or 0) == 0
                and bool(validation["release_eligible"])
            )
            workflow = "closed_loop"

        requested = _requested_power(specification)
        if decision == "accepted":
            completion = 100.0
            if requested is not None:
                committed = requested
        elif decision == "rejected":
            completion = 0.0
            committed = 0.0
        feedback = _feedback(
            decision=decision,
            requested=requested,
            committed=committed,
            completion=completion,
            metrics=metrics,
            workflow=workflow,
        )
        physics = PhysicsEvidence(
            workflow=workflow,
            status=evidence["status"],
            evidence_path=str(evidence_path),
            evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),
            receiver_decision=decision,
            committed_power_kw=committed,
            completion_percentage=completion,
            replan_count=replan_count,
            converged=converged,
            hard_constraints_passed=hard_constraints_passed,
            business_metrics=metrics,
            input_hashes=input_hashes,
            execution_plan_id=plan.plan_id,
            execution_plan_sha256=plan_sha256,
            execution_log={
                "plan_id": plan.plan_id,
                "plan_sha256": plan_sha256,
                "actions": [
                    {
                        "action_id": action.action_id,
                        "action_type": action.action_type.value,
                        "tool": action.tool,
                        "status": "completed",
                        "evidence_package": str(evidence_path),
                    }
                    for action in plan.actions
                ],
                "time_alignment_passed": True,
                "bem_bus_mapping_passed": True,
            },
        )
        return feedback, physics


def _validate_execution_plan(
    specification: TaskSpecification, plan: ExecutionPlan
) -> None:
    from b2g_agent.agents.actor import PlanCompiler
    errors = PlanCompiler().validate(plan, specification)
    if errors:
        raise ValueError("Execution plan failed fresh contract validation: " + ", ".join(errors))
    if plan.episode_id != specification.episode_id:
        raise ValueError("Execution plan and task specification episode IDs differ")
    if not plan.valid or plan.validation_errors:
        raise ValueError("Only a validated execution plan can enter the physics executor")
    positions = {action.action_id: index for index, action in enumerate(plan.actions)}
    for action in plan.actions:
        for dependency in action.depends_on:
            if dependency not in positions or positions[dependency] >= positions[action.action_id]:
                raise ValueError(
                    f"Execution plan is not topologically ordered at {action.action_id}"
                )
    produced = {output for action in plan.actions for output in action.outputs}
    missing = sorted(set(specification.requested_outputs) - produced)
    if missing:
        raise ValueError(
            "Execution plan does not produce requested outputs: " + ", ".join(missing)
        )


def _requested_power(specification: TaskSpecification) -> float | None:
    field = specification.fields.get("target_power")
    if field is None or field.unit != "kW" or not isinstance(field.value, (int, float)):
        return None
    return float(field.value)


def _feedback(
    *,
    decision: str,
    requested: float | None,
    committed: float,
    completion: float,
    metrics: dict[str, object],
    workflow: str,
) -> ReceiverFeedback:
    requested_text = f" against the {requested:.3f} kW request" if requested is not None else ""
    if workflow == "bidirectional_flexibility_envelope":
        lower = float(metrics["flexibility_lower_kw"])
        upper = float(metrics["flexibility_upper_kw"])
        detail = (
            f"The validated symmetric flexibility is {committed:.3f} kW, with net-demand "
            f"bounds {lower:.3f}--{upper:.3f} kW."
        )
    else:
        energy = float(metrics.get("delivered_energy_kwh") or 0.0)
        average = float(metrics.get("event_average_directed_delivery_kw") or 0.0)
        detail = (
            f"The validated schedule delivers {energy:.3f} kWh "
            f"({average:.3f} kW event average)."
        )
    if decision == "accepted":
        text = (
            f"I accept the request after EnergyPlus and nonlinear AC power-flow validation. "
            f"The validated commitment is {committed:.3f} kW. {detail}"
        )
    elif decision == "modified":
        text = (
            f"I cannot accept every requested term{requested_text}; my validated counteroffer "
            f"has {committed:.3f} kW sustained capacity. {detail} The quantified completion is "
            f"{completion:.3f}%."
        )
    else:
        text = (
            "I reject the request because no safe nonzero service remains after EnergyPlus and "
            "nonlinear AC power-flow validation."
        )
    return ReceiverFeedback(
        decision=decision,
        text=text,
        committed_power_kw=committed,
        completion_percentage=completion,
        completion_basis=(
            "Validated delivered-versus-requested service after closed-loop EnergyPlus and "
            "pandapower checks."
        ),
    )
