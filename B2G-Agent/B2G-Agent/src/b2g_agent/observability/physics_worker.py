"""Run an unchanged numerical script with durable tool-stage instrumentation.

The numerical source/input fingerprint stays independent of logging. Child
experiment scripts inherit the same journal and are wrapped in the same way.
"""
from __future__ import annotations

from pathlib import Path
import runpy
import subprocess
import sys

SRC = Path(__file__).resolve().parents[2]
REPO = SRC.parent
sys.path.insert(0, str(SRC))

from b2g_agent.observability.journal import emit, traced_stage


def run_script(script: Path, arguments: list[str], *,
               interactive_task_specification: Path | None = None) -> None:
    from b2g_agent.simulation.adapters.energyplus import FormalEnergyPlusAdapter
    from b2g_agent.simulation.adapters.pandapower import FormalPandapowerAdapter

    original_run = subprocess.run
    original_energyplus = FormalEnergyPlusAdapter.run
    original_pandapower = FormalPandapowerAdapter.run
    original_argv = sys.argv
    private_arguments = (["--interactive-task-specification", str(interactive_task_specification)]
                         if interactive_task_specification is not None else [])
    if (interactive_task_specification is not None
            and script.resolve() == (REPO / "experiments/prepare_event_idfs.py").resolve()):
        # The private adapter changes metadata lookup only. Formal invocations
        # execute the original script unchanged, including its original argv.
        script = SRC / "b2g_agent/personal_physics.py"
        arguments = [*arguments, "--native-csv", "--task-specification", str(interactive_task_specification)]

    @traced_stage("tool.subprocess")
    def logged_run(command, *args, **kwargs):
        actual = command
        if isinstance(command, (list, tuple)) and len(command) > 1:
            child = Path(command[1]).resolve()
            if (Path(command[0]).resolve() == Path(sys.executable).resolve()
                    and child.suffix == ".py" and child.is_relative_to(REPO / "experiments")):
                actual = [command[0], str(Path(__file__).resolve()), *private_arguments, *command[1:]]
        emit("tool.command", original_command=command, executed_command=actual)
        completed = original_run(actual, *args, **kwargs)
        emit("tool.output", command=actual, returncode=completed.returncode,
             stdout=completed.stdout, stderr=completed.stderr)
        return completed

    @traced_stage("tool.energyplus")
    def logged_energyplus(self, *args, **kwargs):
        runner = self.runner
        if runner is original_run:
            self.runner = logged_run
        try:
            return original_energyplus(self, *args, **kwargs)
        finally:
            self.runner = runner

    subprocess.run = logged_run
    FormalEnergyPlusAdapter.run = logged_energyplus
    FormalPandapowerAdapter.run = traced_stage("tool.pandapower")(original_pandapower)
    try:
        sys.argv = [str(script), *arguments]
        emit("physics.script_started", script=script, arguments=arguments)
        try:
            runpy.run_path(str(script), run_name="__main__")
        except SystemExit as exc:
            if exc.code not in (None, 0):
                raise
        emit("physics.script_completed", script=script)
    finally:
        subprocess.run = original_run
        FormalEnergyPlusAdapter.run = original_energyplus
        FormalPandapowerAdapter.run = original_pandapower
        sys.argv = original_argv


if __name__ == "__main__":
    worker_arguments = sys.argv[1:]
    task_input = None
    if worker_arguments[:1] == ["--interactive-task-specification"]:
        if len(worker_arguments) < 3:
            raise SystemExit("An interactive task file and numerical script are required")
        task_input = Path(worker_arguments[1]).resolve(strict=True)
        worker_arguments = worker_arguments[2:]
    if not worker_arguments:
        raise SystemExit("Usage: physics_worker.py SCRIPT [ARGUMENTS...]")
    traced_stage("physics.script")(run_script)(
        Path(worker_arguments[0]).resolve(), worker_arguments[1:],
        interactive_task_specification=task_input,
    )
