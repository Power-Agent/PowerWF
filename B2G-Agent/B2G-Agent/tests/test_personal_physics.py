"""Private task input regression and parity with the untouched formal IDF path."""
from datetime import datetime, timedelta
import csv
import importlib.util
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from b2g_agent.agents.actor import PlanCompiler
from b2g_agent.case_io import CaseBundle
from b2g_agent.case_io.operating_policy import operating_constraints
from b2g_agent.observability import physics_worker
from b2g_agent.orchestration.task_physics import TaskPhysicsExecutor
from b2g_agent.personal_physics import enable_native_csv, prepare_event_inputs
from b2g_agent.personal_workspace import PersonalWorkspace
from test_energyplus_controls import SIMPLE_IDF
from b2g_agent.simulation.energyplus_controls import idf_objects

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def task_assets(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    case = CaseBundle.default(ROOT)
    spec = case.get_groundtruth("ep-001").task_specification.model_copy(deep=True)
    spec.constraints = operating_constraints(case)
    for name in ["case.json", "days/target_days.json", "network/bem_to_bus.csv", "optimization/joint_dispatch.json"]:
        target = root / "cases" / case.case_id / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((case.case_dir / name).read_bytes())
    data = tmp_path / "data"
    monkeypatch.setenv("B2G_DATA_ROOT", str(data))
    local = data / case.case_id
    profile = local / "profiles/bem_baseline_15min.csv"
    profile.parent.mkdir(parents=True)
    day = json.loads((case.case_dir / "days/target_days.json").read_text())["target_days"][spec.operating_day]
    start = datetime.fromisoformat(day)
    with profile.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp", "bem_id", "portfolio_kw"])
        for bem in case.manifest["bems"]:
            source = local / "buildings" / bem["bem_id"] / "energyplus_23_1.idf"
            source.parent.mkdir(parents=True)
            source.write_text(SIMPLE_IDF, encoding="utf-8")
            for interval in range(96):
                writer.writerow([(start + timedelta(minutes=15 * interval)).isoformat(), bem["bem_id"], 0 if interval == 0 else 100])
    optimizer = {"episode_id": spec.episode_id, "schedules": {
        bem["bem_id"]: {"building_reduction_kw": [0, 10, -10, 200, -200, 0] * 16}
        for bem in case.manifest["bems"]}}
    task_path, optimizer_path = tmp_path / "task.json", tmp_path / "optimizer.json"
    task_path.write_text(spec.model_dump_json(), encoding="utf-8")
    optimizer_path.write_text(json.dumps(optimizer), encoding="utf-8")
    return root, spec, task_path, optimizer_path, optimizer


def test_dynamic_task_prepares_idfs_without_an_oracle(task_assets, tmp_path):
    root, spec, task_path, optimizer_path, optimizer = task_assets
    spec.episode_id = "bf84add80747-turn-001"
    spec.context_origin = "interactive_inferred"
    optimizer["episode_id"] = spec.episode_id
    task_path.write_text(spec.model_dump_json(), encoding="utf-8")
    optimizer_path.write_text(json.dumps(optimizer), encoding="utf-8")
    assert not (root / "cases" / spec.case_id / "groundtruth").exists()
    output = tmp_path / "private-event-idfs"
    manifest = json.loads(prepare_event_inputs(task_specification=task_path, optimizer_result=optimizer_path,
        output_root=output, repo_root=root).read_text())
    assert manifest["episode_id"] == spec.episode_id
    assert manifest["operating_day"] == "winter_peak_01" and manifest["target_date"] == "2025-02-06"
    assert len(manifest["records"]) == 10
    assert len(manifest["interactive_task_input"]["sha256"]) == 64
    assert (output / "bem_01/event.idf").is_file()


def test_private_idfs_and_control_values_are_byte_identical_to_formal(task_assets, tmp_path, monkeypatch):
    root, spec, task_path, optimizer_path, _ = task_assets
    oracle = root / "cases" / spec.case_id / "groundtruth/contract_oracles.jsonl"
    oracle.parent.mkdir(parents=True)
    oracle.write_text(json.dumps({"episode_id": spec.episode_id, "task_specification": {"operating_day": spec.operating_day}}) + "\n")
    module_spec = importlib.util.spec_from_file_location("formal_idf_parity", ROOT / "experiments/prepare_event_idfs.py")
    formal = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(formal)
    formal_output, private_output = tmp_path / "formal", tmp_path / "private"
    monkeypatch.setattr(formal, "REPO_ROOT", root)
    monkeypatch.setattr(formal, "parse_args", lambda: SimpleNamespace(optimizer_result=optimizer_path, output_root=formal_output))
    formal.main()
    prepare_event_inputs(task_specification=task_path, optimizer_result=optimizer_path, output_root=private_output, repo_root=root)
    a, b = [json.loads((folder / "manifest.json").read_text()) for folder in [formal_output, private_output]]
    b.pop("interactive_task_input")
    assert a == b  # Includes control values, dates, and all ten IDF content hashes.
    for record in a["records"]:
        for name in ["baseline.idf", "event.idf"]:
            relative = Path(record["bem_id"]) / name
            assert (formal_output / relative).read_bytes() == (private_output / relative).read_bytes()


def test_native_csv_changes_only_reporting_and_retains_actual_input_hashes(task_assets, tmp_path):
    root, _, task, optimizer, _ = task_assets
    plain, native = tmp_path / "plain", tmp_path / "native"
    prepare_event_inputs(task_specification=task, optimizer_result=optimizer, output_root=plain, repo_root=root)
    manifest = json.loads(prepare_event_inputs(task_specification=task, optimizer_result=optimizer,
        output_root=native, repo_root=root, native_csv=True).read_text())
    assert manifest["interactive_task_input"]["csv_export"] == "energyplus_native"
    for record in manifest["records"]:
        for name in ["baseline", "event"]:
            a, b = [folder / record["bem_id"] / (name + ".idf") for folder in [plain, native]]
            before, after = [idf_objects(path.read_text()) for path in [a, b]]
            assert [obj for obj in after if obj[0] != "outputcontrol:files"] == before
            assert [fields for kind, fields in after if kind == "outputcontrol:files"] == [["OutputControl:Files", "Yes"]]
            assert hashlib.sha256(b.read_bytes()).hexdigest() == record[name + "_sha256"]


def test_native_csv_preserves_other_output_choices(tmp_path):
    path = tmp_path / "controls.idf"
    path.write_text("OutputControl:Files,No,No,Yes;\n" + SIMPLE_IDF)
    enable_native_csv(path)
    enable_native_csv(path)
    objects = [fields for kind, fields in idf_objects(path.read_text()) if kind == "outputcontrol:files"]
    assert objects == [["OutputControl:Files", "Yes", "No", "Yes"]]


@pytest.mark.parametrize("invalid", ["mismatched_id", "unavailable_day", "bad_timezone", "short_schedule", "nonfinite_schedule"])
def test_invalid_private_bindings_fail_before_writing(task_assets, tmp_path, invalid):
    root, spec, task_path, optimizer_path, optimizer = task_assets
    if invalid == "mismatched_id": optimizer["episode_id"] = "another-turn"
    elif invalid == "unavailable_day": spec.operating_day = "unknown_day"
    elif invalid == "bad_timezone": spec.windows[0].timezone = "UTC+08:00"
    elif invalid == "short_schedule": optimizer["schedules"]["bem_01"]["building_reduction_kw"] = [0]
    else: optimizer["schedules"]["bem_01"]["building_reduction_kw"][0] = float("nan")
    task_path.write_text(spec.model_dump_json(), encoding="utf-8")
    optimizer_path.write_text(json.dumps(optimizer), encoding="utf-8")
    output = tmp_path / "must-not-exist"
    with pytest.raises(ValueError):
        prepare_event_inputs(task_specification=task_path, optimizer_result=optimizer_path, output_root=output, repo_root=root)
    assert not output.exists()


@pytest.mark.parametrize("enabled,origin,expected", [
    (False, "benchmark_episode", False), (False, "interactive_inferred", False),
    (True, "benchmark_episode", False), (True, "interactive_inferred", True),
    (True, "interactive_seeded", True),
])
def test_formal_command_is_unchanged_and_private_switch_is_explicit(tmp_path, monkeypatch, enabled, origin, expected):
    case = CaseBundle.default(ROOT)
    spec = case.get_groundtruth("ep-001").task_specification.model_copy(deep=True)
    spec.constraints = operating_constraints(case)
    spec.context_origin = origin
    plan = PlanCompiler().compile(spec)
    commands = []
    class CommandCaptured(Exception): pass
    def capture(command, **kwargs):
        commands.append(command)
        raise CommandCaptured
    monkeypatch.setattr("b2g_agent.orchestration.task_physics.subprocess.run", capture)
    with pytest.raises(CommandCaptured):
        TaskPhysicsExecutor(ROOT, use_interactive_inputs=enabled).run(spec, plan, run_root=tmp_path / "physics")
    command = commands[0]
    assert ("--interactive-task-specification" in command) == expected
    if not expected:
        assert command[:3] == [sys.executable, str(ROOT / "src/b2g_agent/observability/physics_worker.py"), str(ROOT / "experiments/run_closed_loop_episode.py")]
    else:
        assert Path(command[3]).is_file()
        assert json.loads(Path(command[3]).read_text())["episode_id"] == spec.episode_id


def test_private_factory_opt_in_does_not_change_default_executor(tmp_path, monkeypatch):
    monkeypatch.setenv("B2G_PERSONAL_ROOT", str(tmp_path / "private"))
    workspace = PersonalWorkspace.open(ROOT)
    pipeline = workspace.pipeline(ROOT)
    assert pipeline.physics_executor.use_interactive_inputs
    assert not TaskPhysicsExecutor(ROOT).use_interactive_inputs


@pytest.mark.parametrize("private", [False, True])
def test_worker_redirects_only_opted_in_idf_preparation(tmp_path, monkeypatch, private):
    observed = []
    monkeypatch.setattr(physics_worker.runpy, "run_path", lambda script, **kwargs: observed.append((Path(script), list(sys.argv))))
    original = ROOT / "experiments/prepare_event_idfs.py"
    task = tmp_path / "reviewed-task.json"
    physics_worker.run_script(original, ["--optimizer-result", "optimizer.json", "--output-root", "out"],
        interactive_task_specification=task if private else None)
    script, argv = observed[0]
    assert script == (ROOT / "src/b2g_agent/personal_physics.py" if private else original)
    assert ("--task-specification" in argv) == private
    if private: assert argv[-1] == str(task)


def test_private_input_propagates_to_nested_numeric_workers(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(physics_worker.subprocess, "run", lambda command, **kwargs: observed.append(command) or subprocess.CompletedProcess(command, 0, "", ""))
    child = ROOT / "experiments/run_closed_loop_episode.py"
    monkeypatch.setattr(physics_worker.runpy, "run_path", lambda *a, **k: physics_worker.subprocess.run([sys.executable, str(child), "--episode", "private-turn"]))
    task = tmp_path / "task.json"
    physics_worker.run_script(ROOT / "experiments/run_flexibility_envelope_episode.py", [], interactive_task_specification=task)
    assert observed[0][:5] == [sys.executable, str(Path(physics_worker.__file__)), "--interactive-task-specification", str(task), str(child)]
