from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from b2g_agent.case_io import CaseBundle


def simulator_status(case: CaseBundle) -> dict[str, Any]:
    """Return safe tool identity/readiness without exposing absolute paths."""
    executable = Path(os.getenv("B2G_ENERGYPLUS_EXE", "energyplus")).expanduser()
    energyplus_ready = executable.is_file()
    energyplus_version = "unavailable"
    if energyplus_ready:
        completed = subprocess.run(
            [str(executable), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if completed.returncode == 0:
            energyplus_version = (completed.stdout or completed.stderr).strip().splitlines()[0]
    try:
        import pandapower

        pandapower_ready = True
        pandapower_version = pandapower.__version__
    except ImportError:
        pandapower_ready = False
        pandapower_version = "unavailable"
    readiness = case.readiness()
    return {
        "energyplus": {"ready": energyplus_ready, "version": energyplus_version},
        "pandapower": {"ready": pandapower_ready, "version": pandapower_version},
        "formal_case_physics_ready": readiness.ready_for_physics,
        "missing_asset_count": len(readiness.missing_local_assets),
    }
