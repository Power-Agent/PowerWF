"""Prepare the frozen 15-minute nonparticipating feeder load profiles.

The source profiles are official 2021 ComStock/ResStock TMY3 New York state
aggregates from the OEDI data lake.  Their interval-ending timestamps describe
TMY shapes, not calendar-year 2018 observations.  This script converts those
shapes to interval-starting timestamps in the case's nominal year and scales
each paper-bus profile to 60% of the original ``case33bw`` node peak.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandapower as pp


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "ieee33_10bem_15min"
CASE_DIR = REPO_ROOT / "cases" / CASE_ID
DEFAULT_DATA_ROOT = Path(os.getenv("B2G_DATA_ROOT", REPO_ROOT / ".local-data"))
OEDI_BASE = (
    "https://oedi-data-lake.s3.amazonaws.com/"
    "nrel-pds-building-stock/end-use-load-profiles-for-us-building-stock/2021"
)
ELECTRICITY_COLUMN = "out.electricity.total.energy_consumption"
EXPECTED_INTERVALS = 365 * 96
NOMINAL_YEAR = 2025


@dataclass(frozen=True)
class SourceProfile:
    source_id: str
    building_class: str
    collection: str
    object_name: str

    @property
    def url(self) -> str:
        release = (
            "comstock_tmy3_release_1"
            if self.collection == "comstock"
            else "resstock_tmy3_release_1"
        )
        return (
            f"{OEDI_BASE}/{release}/timeseries_aggregates/by_state/state=NY/"
            f"{self.object_name}"
        )


SOURCES = (
    SourceProfile("comstock:fullservicerestaurant", "Full Service Restaurant", "comstock", "ny-fullservicerestaurant.csv"),
    SourceProfile("comstock:hospital", "Hospital", "comstock", "ny-hospital.csv"),
    SourceProfile("comstock:largehotel", "Large Hotel", "comstock", "ny-largehotel.csv"),
    SourceProfile("comstock:largeoffice", "Large Office", "comstock", "ny-largeoffice.csv"),
    SourceProfile("comstock:mediumoffice", "Medium Office", "comstock", "ny-mediumoffice.csv"),
    SourceProfile("comstock:outpatient", "Outpatient Healthcare", "comstock", "ny-outpatient.csv"),
    SourceProfile("comstock:primaryschool", "Primary School", "comstock", "ny-primaryschool.csv"),
    SourceProfile("comstock:quickservicerestaurant", "Quick Service Restaurant", "comstock", "ny-quickservicerestaurant.csv"),
    SourceProfile("comstock:retailstandalone", "Stand-alone Retail", "comstock", "ny-retailstandalone.csv"),
    SourceProfile("comstock:retailstripmall", "Strip Mall", "comstock", "ny-retailstripmall.csv"),
    SourceProfile("comstock:secondaryschool", "Secondary School", "comstock", "ny-secondaryschool.csv"),
    SourceProfile("comstock:smallhotel", "Small Hotel", "comstock", "ny-smallhotel.csv"),
    SourceProfile("comstock:smalloffice", "Small Office", "comstock", "ny-smalloffice.csv"),
    SourceProfile("comstock:warehouse", "Warehouse", "comstock", "ny-warehouse.csv"),
    SourceProfile(
        "resstock:multifamily5plus",
        "Multi-Family with 5+ Units",
        "resstock",
        "ny-multi-family_with_5plus_units.csv",
    ),
)
SOURCE_BY_ID = {source.source_id: source for source in SOURCES}

# BEM buses use a directly corresponding stock class.  The other buses cycle
# over the same frozen population to represent a heterogeneous nonparticipant
# mix.  The allocation itself is saved in the manifest.
MATCHED_BEM_SOURCE = {
    3: "comstock:smalloffice",
    5: "comstock:mediumoffice",
    8: "comstock:smalloffice",
    11: "comstock:mediumoffice",
    15: "comstock:retailstandalone",
    18: "comstock:retailstripmall",
    21: "comstock:smallhotel",
    27: "resstock:multifamily5plus",
    29: "comstock:retailstandalone",
    33: "comstock:warehouse",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_allocation(paper_buses: Iterable[int]) -> dict[int, SourceProfile]:
    allocation: dict[int, SourceProfile] = {}
    source_ids = tuple(source.source_id for source in SOURCES)
    for offset, bus in enumerate(sorted(paper_buses)):
        source_id = MATCHED_BEM_SOURCE.get(bus, source_ids[offset % len(source_ids)])
        allocation[bus] = SOURCE_BY_ID[source_id]
    return allocation


def read_tmy_shape(path: Path, *, expected_intervals: int = EXPECTED_INTERVALS) -> list[float]:
    values: list[float] = []
    previous_timestamp: datetime | None = None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or ELECTRICITY_COLUMN not in reader.fieldnames:
            raise ValueError(f"Missing {ELECTRICITY_COLUMN!r} in {path}")
        if "timestamp" not in reader.fieldnames:
            raise ValueError(f"Missing 'timestamp' in {path}")
        for row_number, row in enumerate(reader, 2):
            timestamp = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            if previous_timestamp is not None and timestamp - previous_timestamp != timedelta(minutes=15):
                raise ValueError(f"Non-15-minute timestamp at {path}:{row_number}")
            value = float(row[ELECTRICITY_COLUMN])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid electricity value at {path}:{row_number}: {value}")
            values.append(value)
            previous_timestamp = timestamp
    if len(values) != expected_intervals:
        raise ValueError(f"Expected {expected_intervals} intervals in {path}, found {len(values)}")
    if max(values) <= 0:
        raise ValueError(f"All-zero electricity profile: {path}")
    return values


def normalize_energy_shape_to_peak(energy_kwh: Iterable[float], target_peak_kw: float) -> list[float]:
    """Convert 15-minute energy to mean power and normalize its power peak."""
    power_kw = [4.0 * value for value in energy_kwh]
    peak_kw = max(power_kw)
    if not math.isfinite(target_peak_kw) or target_peak_kw <= 0 or peak_kw <= 0:
        raise ValueError("Both source and target peaks must be finite and positive")
    scale = target_peak_kw / peak_kw
    return [value * scale for value in power_kw]


def download(source: SourceProfile, raw_dir: Path, force: bool) -> Path:
    destination = raw_dir / source.collection / source.object_name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": "B2G-Agent reproducible case preparation/1.0"},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="redownload raw OEDI objects; the derived profile is always regenerated",
    )
    args = parser.parse_args()

    local_case = args.data_root.resolve() / CASE_ID
    raw_dir = local_case / "raw" / "oedi-eulp-2021"
    profile_dir = local_case / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)

    raw_paths: dict[str, Path] = {}
    shapes: dict[str, list[float]] = {}
    for source in SOURCES:
        path = download(source, raw_dir, args.force_download)
        raw_paths[source.source_id] = path
        shapes[source.source_id] = read_tmy_shape(path)
        print(f"validated {source.source_id}: {path.name}")

    net = pp.from_json(str(CASE_DIR / "network" / "source_case33bw.json"))
    original_loads: dict[int, tuple[float, float]] = {}
    for row in net.load.itertuples():
        paper_bus = int(row.bus) + 1
        original_loads[paper_bus] = (float(row.p_mw) * 1000.0, float(row.q_mvar) * 1000.0)
    if sorted(original_loads) != list(range(2, 34)):
        raise ValueError("case33bw must contain one original load at each paper bus 2-33")
    allocation = source_allocation(original_loads)

    output_path = profile_dir / "background_load_15min.csv"
    start = datetime(NOMINAL_YEAR, 1, 1)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "timestamp",
                "paper_bus",
                "p_kw",
                "q_kvar",
                "source_building_class",
                "source_record_id",
            ]
        )
        for paper_bus in sorted(original_loads):
            base_p_kw, base_q_kvar = original_loads[paper_bus]
            source = allocation[paper_bus]
            shape = shapes[source.source_id]
            normalized_p_kw = normalize_energy_shape_to_peak(shape, 0.60 * base_p_kw)
            q_to_p = base_q_kvar / base_p_kw
            for index, p_kw in enumerate(normalized_p_kw):
                timestamp = start + timedelta(minutes=15 * index)
                writer.writerow(
                    [
                        timestamp.isoformat(timespec="seconds"),
                        paper_bus,
                        f"{p_kw:.9f}",
                        f"{p_kw * q_to_p:.9f}",
                        source.building_class,
                        f"oedi-eulp-2021:NY:{source.source_id}",
                    ]
                )

    manifest = {
        "schema_version": "1.0",
        "case_id": CASE_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "nominal_year": NOMINAL_YEAR,
        "time_semantics": {
            "source": "15-minute interval-ending local-standard-time TMY shape",
            "output": "15-minute interval-starting naive local-standard-time",
            "dst": "not applied; exactly 35040 intervals",
        },
        "transformation": {
            "script": "experiments/prepare_background_profiles.py",
            "rule": "normalize each OEDI shape to 60% of the original case33bw bus P peak and preserve the original node Q/P ratio",
            "command": "python experiments/prepare_background_profiles.py",
        },
        "raw_sources": [
            {
                "source_id": source.source_id,
                "building_class": source.building_class,
                "url": source.url,
                "path": str(raw_paths[source.source_id].relative_to(args.data_root.resolve())).replace("\\", "/"),
                "size_bytes": raw_paths[source.source_id].stat().st_size,
                "sha256": sha256_file(raw_paths[source.source_id]),
            }
            for source in SOURCES
        ],
        "bus_allocation": {
            str(bus): allocation[bus].source_id for bus in sorted(allocation)
        },
        "derived_files": [
            {
                "path": str(output_path.relative_to(args.data_root.resolve())).replace("\\", "/"),
                "row_count": len(original_loads) * EXPECTED_INTERVALS,
                "sha256": sha256_file(output_path),
            }
        ],
    }
    manifest_path = profile_dir / "background_load_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {output_path} rows={len(original_loads) * EXPECTED_INTERVALS} "
        f"sha256={manifest['derived_files'][0]['sha256']}"
    )


if __name__ == "__main__":
    main()
