from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from b2g_agent.simulation.energyplus_controls import controlled_zone_names


Runner = Callable[..., subprocess.CompletedProcess[str]]


class EnergyPlusUnavailable(RuntimeError):
    pass


class EnergyPlusExecutionError(RuntimeError):
    pass


@dataclass(frozen=True)
class FormalEnergyPlusConfig:
    executable: Path
    weather: Path
    timeout_seconds: int = 1800

    @classmethod
    def from_case(cls, *, data_root: Path, case_id: str) -> FormalEnergyPlusConfig:
        executable_value = os.getenv("B2G_ENERGYPLUS_EXE", "energyplus").strip()
        executable = Path(executable_value).expanduser()
        if not executable.is_file():
            discovered = shutil.which(executable_value)
            if not discovered:
                raise EnergyPlusUnavailable(
                    "EnergyPlus executable is unavailable; set B2G_ENERGYPLUS_EXE in the untracked .env"
                )
            executable = Path(discovered)
        weather = (
            data_root
            / case_id
            / "weather"
            / "USA_NY_Buffalo.Niagara.Intl_.AP_.725280_TMY3.epw"
        )
        if not weather.is_file():
            raise EnergyPlusUnavailable(f"Formal case weather is missing: {weather}")
        timeout = int(os.getenv("B2G_ENERGYPLUS_TIMEOUT_SECONDS", "1800"))
        return cls(executable.resolve(), weather.resolve(), timeout)


@dataclass(frozen=True)
class EnergyPlus15MinuteResult:
    bem_id: str
    date_mm_dd: str
    electricity_kw: tuple[float, ...]
    minimum_zone_temperature_c: tuple[float, ...] | None
    mean_zone_temperature_c: tuple[float, ...] | None
    maximum_zone_temperature_c: tuple[float, ...] | None
    warning_count: int
    severe_error_count: int
    version: str
    input_hashes: dict[str, str]
    artifacts: tuple[str, ...]


class FormalEnergyPlusAdapter:
    """Runs one prepared IDF through the official CLI and returns 96 intervals."""

    def __init__(self, config: FormalEnergyPlusConfig, *, runner: Runner = subprocess.run) -> None:
        self.config = config
        self.runner = runner

    def run(
        self,
        *,
        bem_id: str,
        idf_path: Path,
        date_mm_dd: str,
        run_dir: Path,
    ) -> EnergyPlus15MinuteResult:
        if not re.fullmatch(r"\d{2}/\d{2}", date_mm_dd):
            raise ValueError("date_mm_dd must use MM/DD")
        if not idf_path.is_file():
            raise EnergyPlusUnavailable(f"Prepared IDF is missing: {idf_path}")
        if run_dir.exists():
            raise FileExistsError(f"EnergyPlus run directory already exists: {run_dir}")
        output_dir = run_dir / "output"
        output_dir.mkdir(parents=True)
        command = [
            str(self.config.executable),
            "--weather",
            str(self.config.weather),
            "--output-directory",
            str(output_dir),
            "--readvars",
            str(idf_path),
        ]
        completed = self.runner(
            command,
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
            check=False,
        )
        (run_dir / "command.json").write_text(
            json.dumps(
                {
                    "command": [Path(item).name if index in {0, 2, 4, 6} else item for index, item in enumerate(command)],
                    "returncode": completed.returncode,
                    "stdout_tail": completed.stdout[-4000:],
                    "stderr_tail": completed.stderr[-4000:],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        error_path = output_dir / "eplusout.err"
        warning_count = _warning_count(error_path)
        severe_count = _severe_count(error_path)
        if completed.returncode != 0 or severe_count:
            raise EnergyPlusExecutionError(
                f"EnergyPlus failed for {bem_id}: returncode={completed.returncode}, severe_errors={severe_count}"
            )
        csv_path = output_dir / "eplusout.csv"
        controlled_zones = controlled_zone_names(
            idf_path.read_text(encoding="utf-8", errors="replace")
        )
        power, minimum_temperature, mean_temperature, maximum_temperature = (
            _read_15_minute_day_detailed(
                csv_path,
                date_mm_dd=date_mm_dd,
                zone_names=controlled_zones,
            )
        )
        return EnergyPlus15MinuteResult(
            bem_id=bem_id,
            date_mm_dd=date_mm_dd,
            electricity_kw=tuple(power),
            minimum_zone_temperature_c=(
                tuple(minimum_temperature) if minimum_temperature else None
            ),
            mean_zone_temperature_c=tuple(mean_temperature) if mean_temperature else None,
            maximum_zone_temperature_c=(
                tuple(maximum_temperature) if maximum_temperature else None
            ),
            warning_count=warning_count,
            severe_error_count=severe_count,
            version=self.version(),
            input_hashes={"idf": _sha256(idf_path), "epw": _sha256(self.config.weather)},
            artifacts=("output/eplusout.csv", "output/eplusout.err", "command.json"),
        )

    def version(self) -> str:
        completed = self.runner(
            [str(self.config.executable), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        raw = (completed.stdout or completed.stderr or "unknown").strip().splitlines()[0]
        return _stable_version(raw)


def read_15_minute_day(
    path: Path,
    *,
    date_mm_dd: str,
) -> tuple[list[float], list[float] | None]:
    power, _, _, maximum = _read_15_minute_day_detailed(path, date_mm_dd=date_mm_dd)
    return power, maximum


def _read_15_minute_day_detailed(
    path: Path,
    *,
    date_mm_dd: str,
    zone_names: set[str] | None = None,
) -> tuple[list[float], list[float] | None, list[float] | None, list[float] | None]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise EnergyPlusExecutionError("EnergyPlus CSV has no header")
        date_column = _matching_column(reader.fieldnames, "Date/Time")
        meter_column = _matching_column(reader.fieldnames, "Electricity:Facility")
        temperature_columns = [
            column
            for column in reader.fieldnames
            if "zone mean air temperature" in column.lower()
            and (
                zone_names is None
                or column.split(":", 1)[0].strip().upper() in zone_names
            )
        ]
        if zone_names is not None and not temperature_columns:
            raise EnergyPlusExecutionError(
                "No thermostat-controlled zone temperature columns were found"
            )
        electricity: list[float] = []
        minimum_temperatures: list[float] = []
        mean_temperatures: list[float] = []
        maximum_temperatures: list[float] = []
        expected_month, expected_day = (int(item) for item in date_mm_dd.split("/"))
        for row in reader:
            stamp = (row.get(date_column) or "").strip()
            match = re.match(r"0*(\d+)\s*/\s*0*(\d+)", stamp)
            if not match or (int(match.group(1)), int(match.group(2))) != (
                expected_month,
                expected_day,
            ):
                continue
            electricity.append(_interval_value_to_kw(float(row[meter_column]), meter_column, 15))
            if temperature_columns:
                values = [float(row[column]) for column in temperature_columns]
                minimum_temperatures.append(min(values))
                mean_temperatures.append(sum(values) / len(values))
                maximum_temperatures.append(max(values))
    if len(electricity) != 96:
        raise EnergyPlusExecutionError(
            f"Expected 96 timestep values for {date_mm_dd}; received {len(electricity)}"
        )
    if temperature_columns and len(maximum_temperatures) != 96:
        raise EnergyPlusExecutionError("Zone temperature output did not contain 96 values")
    return (
        electricity,
        minimum_temperatures if temperature_columns else None,
        mean_temperatures if temperature_columns else None,
        maximum_temperatures if temperature_columns else None,
    )


def _matching_column(columns: Sequence[str], token: str) -> str:
    match = next((column for column in columns if token.lower() in column.lower()), None)
    if not match:
        raise EnergyPlusExecutionError(f"EnergyPlus CSV column is missing: {token}")
    return match


def _interval_value_to_kw(value: float, column: str, minutes: int) -> float:
    unit = column.lower()
    if "[j]" in unit:
        return value / (minutes * 60.0 * 1000.0)
    if "[w]" in unit:
        return value / 1000.0
    if "[kw]" in unit:
        return value
    if "[kwh]" in unit:
        return value * 60.0 / minutes
    raise EnergyPlusExecutionError(f"Unsupported EnergyPlus electricity unit: {column}")


def _severe_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum("** Severe  **" in line or "** Severe **" in line for line in path.read_text(encoding="utf-8", errors="replace").splitlines())


def _warning_count(path: Path) -> int:
    if not path.is_file():
        return 0
    text = path.read_text(encoding="utf-8", errors="replace")
    final = re.search(r"Completed Successfully--\s*(\d+) Warning", text)
    if final:
        return int(final.group(1))
    return sum("** Warning **" in line for line in text.splitlines())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stable_version(value: str) -> str:
    return value.split(", YMD=", 1)[0].strip()
