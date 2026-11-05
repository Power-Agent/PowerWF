"""Generate the frozen 15-minute PV profile with a pinned pvlib ModelChain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "ieee33_10bem_15min"
CASE_DIR = REPO_ROOT / "cases" / CASE_ID
DEFAULT_DATA_ROOT = Path(os.getenv("B2G_DATA_ROOT", REPO_ROOT / ".local-data"))
EXPECTED_HOURS = 365 * 24
EXPECTED_QUARTERS = 365 * 96


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()

    try:
        import pandas as pd
        import pvlib
    except ImportError as exc:
        raise RuntimeError(
            "PV generation requires the optional simulation dependencies: "
            "pip install -e '.[simulation]'"
        ) from exc

    data_root = args.data_root.resolve()
    local_case = data_root / CASE_ID
    weather_path = (
        local_case
        / "weather"
        / "USA_NY_Buffalo.Niagara.Intl_.AP_.725280_TMY3.epw"
    )
    if not weather_path.is_file():
        raise FileNotFoundError(f"Weather file is missing: {weather_path}")
    configuration = json.loads(
        (CASE_DIR / "der" / "pv.json").read_text(encoding="utf-8")
    )
    weather, metadata = pvlib.iotools.read_epw(weather_path)
    if len(weather) != EXPECTED_HOURS:
        raise ValueError(f"Expected {EXPECTED_HOURS} EPW rows, found {len(weather)}")

    fixed_offset = timezone(timedelta(hours=float(metadata["TZ"])))
    hourly_index = pd.date_range(
        "2025-01-01 00:00:00", periods=EXPECTED_HOURS, freq="h", tz=fixed_offset
    )
    weather = weather.reset_index(drop=True)
    weather.index = hourly_index
    model_columns = ["ghi", "dni", "dhi", "temp_air", "wind_speed"]
    weather = weather[model_columns].astype(float)
    cyclic_endpoint = weather.iloc[[0]].copy()
    cyclic_endpoint.index = pd.DatetimeIndex([hourly_index[-1] + pd.Timedelta(hours=1)])
    quarter_weather = (
        pd.concat([weather, cyclic_endpoint])
        .resample("15min")
        .interpolate(method="time")
        .iloc[:-1]
    )
    if len(quarter_weather) != EXPECTED_QUARTERS:
        raise ValueError(
            f"Expected {EXPECTED_QUARTERS} interpolated weather rows, found {len(quarter_weather)}"
        )
    for column in ("ghi", "dni", "dhi", "wind_speed"):
        quarter_weather[column] = quarter_weather[column].clip(lower=0.0)

    mount = pvlib.pvsystem.FixedMount(
        surface_tilt=float(configuration["tilt_degrees"]),
        surface_azimuth=float(configuration["azimuth_degrees"]),
        racking_model="open_rack",
        module_height=1.0,
    )
    temperature_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
        "open_rack_glass_polymer"
    ]
    array = pvlib.pvsystem.Array(
        mount=mount,
        module_parameters={
            "pdc0": float(configuration["module_parameters"]["pdc0_w_per_kw_nameplate"]),
            "gamma_pdc": float(configuration["module_parameters"]["gamma_pdc_per_c"]),
        },
        temperature_model_parameters=temperature_parameters,
        modules_per_string=1,
        strings=1,
    )
    system = pvlib.pvsystem.PVSystem(
        arrays=[array],
        inverter_parameters={
            "pdc0": float(configuration["inverter_parameters"]["pdc0_w_per_kw_nameplate"])
        },
    )
    location = pvlib.location.Location(
        latitude=float(metadata["latitude"]),
        longitude=float(metadata["longitude"]),
        tz=int(float(metadata["TZ"])),
        altitude=float(metadata["altitude"]),
        name="Buffalo Niagara International Airport TMY3",
    )
    model = pvlib.modelchain.ModelChain(
        system,
        location,
        dc_model="pvwatts",
        ac_model="pvwatts",
        aoi_model=str(configuration["aoi_model"]),
        spectral_model=str(configuration["spectral_model"]),
        temperature_model="sapm",
    )
    model.run_model(quarter_weather)
    unit_ac_kw = model.results.ac.fillna(0.0).clip(lower=0.0) / 1000.0
    unit_ac_kw *= 1.0 - float(configuration["losses_fraction_after_inverter"])
    if any(not math.isfinite(float(value)) for value in unit_ac_kw):
        raise ValueError("PV output contains a non-finite value")

    bem_to_bus: dict[str, int] = {}
    with (CASE_DIR / "network" / "bem_to_bus.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        for row in csv.DictReader(handle):
            bem_to_bus[row["bem_id"]] = int(row["bus"])
    capacities = configuration["capacity_kw_by_bem"]
    if set(capacities) != set(bem_to_bus):
        raise ValueError("PV capacity configuration must cover the ten mapped BEMs exactly")

    output_path = local_case / "profiles" / "pv_15min.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    naive_timestamps = quarter_weather.index.tz_localize(None)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "bem_id", "paper_bus", "pv_kw"])
        for bem_id in sorted(capacities):
            capacity_kw = float(capacities[bem_id])
            for timestamp, per_kw in zip(naive_timestamps, unit_ac_kw, strict=True):
                writer.writerow(
                    [
                        timestamp.isoformat(timespec="seconds"),
                        bem_id,
                        bem_to_bus[bem_id],
                        f"{float(per_kw) * capacity_kw:.9f}",
                    ]
                )

    manifest = {
        "schema_version": "1.0",
        "case_id": CASE_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "software": {"pvlib": pvlib.__version__, "pandas": pd.__version__},
        "input_weather": {
            "path": str(weather_path.relative_to(data_root)).replace("\\", "/"),
            "sha256": sha256_file(weather_path),
            "metadata": {
                key: metadata.get(key)
                for key in ("city", "state-prov", "country", "latitude", "longitude", "TZ", "altitude")
            },
        },
        "configuration_path": "cases/ieee33_10bem_15min/der/pv.json",
        "configuration_sha256": sha256_file(CASE_DIR / "der" / "pv.json"),
        "time_semantics": "fixed local-standard-time, interval-starting, 35040 rows per BEM",
        "output": {
            "path": str(output_path.relative_to(data_root)).replace("\\", "/"),
            "row_count": EXPECTED_QUARTERS * len(capacities),
            "sha256": sha256_file(output_path),
        },
    }
    manifest_path = output_path.with_name("pv_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output_path} rows={manifest['output']['row_count']} "
        f"sha256={manifest['output']['sha256']}"
    )


if __name__ == "__main__":
    main()
