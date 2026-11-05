"""Run or audit EnergyPlus design-day smoke tests for all ten formal BEMs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from b2g_agent.case_io import CaseBundle  # noqa: E402
from b2g_agent.simulation.adapters.energyplus import (  # noqa: E402
    _severe_count,
    _warning_count,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        type=Path,
        default=REPO_ROOT / ".local-runs" / "energyplus-smoke",
    )
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Audit existing design-day-bem_XX directories without rerunning EnergyPlus.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def executable_path() -> Path:
    configured = os.getenv("B2G_ENERGYPLUS_EXE", "energyplus").strip()
    candidate = Path(configured).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    discovered = shutil.which(configured)
    if discovered:
        return Path(discovered).resolve()
    raise FileNotFoundError(
        "EnergyPlus executable is unavailable; configure B2G_ENERGYPLUS_EXE in .env"
    )


def main() -> int:
    load_dotenv(REPO_ROOT / ".env", override=False)
    args = parse_args()
    case = CaseBundle.default(REPO_ROOT)
    executable = executable_path()
    local_case = case.local_data_root / case.case_id
    weather = (
        local_case
        / "weather"
        / "USA_NY_Buffalo.Niagara.Intl_.AP_.725280_TMY3.epw"
    )
    if not weather.is_file():
        raise FileNotFoundError(f"Weather file is missing: {weather}")
    results_root = args.results_root.resolve()
    records: list[dict[str, object]] = []

    for index in range(1, 11):
        bem_id = f"bem_{index:02d}"
        idf = local_case / "buildings" / bem_id / "energyplus_23_1.idf"
        output_dir = results_root / f"design-day-{bem_id}"
        if not idf.is_file():
            raise FileNotFoundError(f"Runtime IDF is missing: {idf}")
        returncode: int | None = None
        stdout_tail = ""
        stderr_tail = ""
        if not args.reuse_existing:
            if output_dir.exists():
                raise FileExistsError(
                    f"Smoke output already exists: {output_dir}; choose another --results-root"
                )
            command = [
                str(executable),
                "--design-day",
                "--weather",
                str(weather),
                "--output-directory",
                str(output_dir),
                str(idf),
            ]
            completed = subprocess.run(
                command,
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=int(os.getenv("B2G_ENERGYPLUS_TIMEOUT_SECONDS", "1800")),
                check=False,
            )
            returncode = completed.returncode
            stdout_tail = completed.stdout[-2000:]
            stderr_tail = completed.stderr[-2000:]
        error_file = output_dir / "eplusout.err"
        if not error_file.is_file():
            raise FileNotFoundError(f"EnergyPlus error report is missing: {error_file}")
        error_text = error_file.read_text(encoding="utf-8", errors="replace")
        severe_count = _severe_count(error_file)
        completed_successfully = "EnergyPlus Completed Successfully" in error_text
        passed = (
            completed_successfully
            and severe_count == 0
            and returncode in {None, 0}
        )
        records.append(
            {
                "bem_id": bem_id,
                "status": "PASS" if passed else "FAIL",
                "returncode": returncode,
                "completed_successfully": completed_successfully,
                "warning_count": _warning_count(error_file),
                "severe_error_count": severe_count,
                "idf_sha256": sha256(idf),
                "error_report_sha256": sha256(error_file),
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
            }
        )

    version = subprocess.run(
        [str(executable), "--version"],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    blocked = [record["bem_id"] for record in records if record["status"] != "PASS"]
    report = {
        "schema_version": "1.0",
        "case_id": case.case_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "reuse_existing" if args.reuse_existing else "executed",
        "status": "PASS" if not blocked else "FAIL",
        "energyplus_version": (version.stdout or version.stderr).strip().splitlines()[0],
        "results_root": str(results_root),
        "blocked_bems": blocked,
        "records": records,
    }
    report_path = (
        args.report_path
        or local_case / "energyplus_smoke_report.json"
    ).resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if not blocked else 1


if __name__ == "__main__":
    raise SystemExit(main())
