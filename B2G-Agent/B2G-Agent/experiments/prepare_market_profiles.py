"""Prepare frozen 2025 ISO-NE price and reserve profiles without Excel dependencies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import posixpath
import re
import shutil
import urllib.request
import zipfile
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Iterator
from xml.etree import ElementTree as ET


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "ieee33_10bem_15min"
DEFAULT_DATA_ROOT = Path(os.getenv("B2G_DATA_ROOT", REPO_ROOT / ".local-data"))
EXPECTED_QUARTERS = 365 * 96
NEMA_RESERVE_ZONE_ID = 7002
HOURLY_URL = "https://www.iso-ne.com/static-assets/documents/100020/2025_smd_hourly.xlsx"
RESERVE_URL = (
    "https://www.iso-ne.com/static-assets/documents/100031/"
    "5min_reserve_price_and_designation_2025.xlsx"
)
MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CELL_COLUMN = re.compile(r"([A-Z]+)")
RESERVE_PRODUCTS = {
    "TMSR": ("tmsr_dollar_per_MWh", "tmsr_desg_mw"),
    "TMNSR": ("tmnsr_dollar_per_MWh", "tmnsr_desg_mw"),
    "TMOR": ("tmor_dollar_per_MWh", "tmor_desg_mw"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, force: bool) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and not force:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        url,
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


def _shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    if "xl/sharedStrings.xml" not in workbook.namelist():
        return []
    root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
    return [
        "".join(text.text or "" for text in item.iter(f"{{{MAIN_NS}}}t"))
        for item in root.findall(f"{{{MAIN_NS}}}si")
    ]


def _sheet_path(workbook: zipfile.ZipFile, sheet_name: str) -> str:
    root = ET.fromstring(workbook.read("xl/workbook.xml"))
    relation_id: str | None = None
    for sheet in root.findall(f".//{{{MAIN_NS}}}sheet"):
        if sheet.attrib.get("name") == sheet_name:
            relation_id = sheet.attrib[f"{{{REL_NS}}}id"]
            break
    if relation_id is None:
        raise ValueError(f"Worksheet {sheet_name!r} is missing")
    relationships = ET.fromstring(workbook.read("xl/_rels/workbook.xml.rels"))
    for relation in relationships.findall(f"{{{PACKAGE_REL_NS}}}Relationship"):
        if relation.attrib.get("Id") == relation_id:
            target = relation.attrib["Target"]
            return target.lstrip("/") if target.startswith("/") else posixpath.normpath(posixpath.join("xl", target))
    raise ValueError(f"Worksheet relationship {relation_id!r} is missing")


def iter_xlsx_rows(path: Path, sheet_name: str) -> Iterator[dict[str, str]]:
    """Stream an XLSX worksheet as dictionaries using only the standard library."""
    with zipfile.ZipFile(path) as workbook:
        shared = _shared_strings(workbook)
        sheet_path = _sheet_path(workbook, sheet_name)
        headers: dict[str, str] | None = None
        with workbook.open(sheet_path) as stream:
            for _, element in ET.iterparse(stream, events=("end",)):
                if element.tag != f"{{{MAIN_NS}}}row":
                    continue
                cells: dict[str, str] = {}
                for cell in element.findall(f"{{{MAIN_NS}}}c"):
                    match = CELL_COLUMN.match(cell.attrib.get("r", ""))
                    if match is None:
                        continue
                    column = match.group(1)
                    value_node = cell.find(f"{{{MAIN_NS}}}v")
                    raw = value_node.text if value_node is not None and value_node.text is not None else ""
                    if cell.attrib.get("t") == "s" and raw:
                        raw = shared[int(raw)]
                    elif cell.attrib.get("t") == "inlineStr":
                        raw = "".join(
                            text.text or "" for text in cell.iter(f"{{{MAIN_NS}}}t")
                        )
                    cells[column] = raw
                if headers is None:
                    headers = {column: value for column, value in cells.items()}
                else:
                    yield {header: cells.get(column, "") for column, header in headers.items()}
                element.clear()


def excel_serial_to_datetime(value: str | float) -> datetime:
    seconds = round(float(value) * 24 * 60 * 60)
    return datetime(1899, 12, 30) + timedelta(seconds=seconds)


def normalize_hourly_lmp(
    records: Iterable[tuple[date, int, float]],
    *,
    expected_days: int = 365,
) -> tuple[list[tuple[datetime, float]], dict[str, object]]:
    """Normalize ISO-NE HE02/HE02X DST notation to 365 fixed 24-hour days."""
    by_day: dict[date, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for day, encoded_hour, value in records:
        if not math.isfinite(value):
            raise ValueError(f"Non-finite LMP on {day}")
        if encoded_hour == 110:  # ISO-NE workbook code rendered as HE02X.
            hour_ending = 2
        elif 21 <= encoded_hour <= 44:
            hour_ending = encoded_hour - 20
        else:
            raise ValueError(f"Unsupported ISO-NE hour-ending code: {encoded_hour}")
        by_day[day][hour_ending].append(value)

    normalized: list[tuple[datetime, float]] = []
    interpolated_days: list[str] = []
    averaged_days: list[str] = []
    for day in sorted(by_day):
        hours = by_day[day]
        for hour_ending in range(1, 25):
            values = hours.get(hour_ending, [])
            if len(values) == 1:
                value = values[0]
            elif len(values) == 2 and hour_ending == 2:
                value = sum(values) / 2.0
                averaged_days.append(day.isoformat())
            elif not values and hour_ending == 2 and hours.get(1) and hours.get(3):
                value = (hours[1][0] + hours[3][0]) / 2.0
                interpolated_days.append(day.isoformat())
            else:
                raise ValueError(
                    f"Unexpected hour multiplicity on {day} HE{hour_ending:02d}: {len(values)}"
                )
            interval_start = datetime.combine(day, datetime.min.time()) + timedelta(
                hours=hour_ending - 1
            )
            normalized.append((interval_start, value))
    expected_hours = expected_days * 24
    if len(normalized) != expected_hours:
        raise ValueError(f"Expected {expected_hours} normalized hours, found {len(normalized)}")
    return normalized, {
        "spring_forward_interpolated_days": sorted(set(interpolated_days)),
        "fall_back_he02_he02x_averaged_days": sorted(set(averaged_days)),
    }


def read_hourly_lmp(path: Path) -> tuple[list[tuple[datetime, float]], dict[str, object]]:
    records: list[tuple[date, int, float]] = []
    for row in iter_xlsx_rows(path, "NEMA"):
        hour_text = row["Hr_End"].strip().upper()
        encoded_hour = 110 if hour_text == "02X" else 20 + int(float(hour_text))
        records.append(
            (
                excel_serial_to_datetime(row["Date"]).date(),
                encoded_hour,
                float(row["RT_LMP"]),
            )
        )
    return normalize_hourly_lmp(records)


def read_reserve_5min(
    path: Path,
) -> tuple[dict[datetime, dict[str, tuple[float, float]]], dict[str, object]]:
    sums: dict[datetime, dict[str, list[float]]] = defaultdict(
        lambda: {product: [0.0, 0.0, 0.0] for product in RESERVE_PRODUCTS}
    )
    source_rows = 0
    for row in iter_xlsx_rows(path, "5min_prices_designations"):
        if int(float(row["rz_id"])) != NEMA_RESERVE_ZONE_ID:
            continue
        source_rows += 1
        timestamp = excel_serial_to_datetime(row["local_time"])
        quarter = timestamp.replace(minute=(timestamp.minute // 15) * 15, second=0, microsecond=0)
        for product, (price_column, designation_column) in RESERVE_PRODUCTS.items():
            accumulator = sums[quarter][product]
            accumulator[0] += float(row[designation_column])
            accumulator[1] += float(row[price_column])
            accumulator[2] += 1.0
    if source_rows != 365 * 24 * 12:
        raise ValueError(f"Expected 105120 NEMA five-minute rows, found {source_rows}")
    observed: dict[datetime, dict[str, tuple[float, float]]] = {}
    duplicated_quarters: list[str] = []
    for timestamp, products in sums.items():
        observed[timestamp] = {}
        for product, (designation_sum, price_sum, count) in products.items():
            if count not in {3, 6}:
                raise ValueError(
                    f"Expected three or six DST-overlap values at {timestamp} for {product}"
                )
            if count == 6:
                duplicated_quarters.append(timestamp.isoformat(timespec="seconds"))
            observed[timestamp][product] = (designation_sum / count, price_sum / count)

    result: dict[datetime, dict[str, tuple[float, float]]] = {}
    interpolated_quarters: list[str] = []
    start = datetime(2025, 1, 1)
    for index in range(EXPECTED_QUARTERS):
        timestamp = start + timedelta(minutes=15 * index)
        if timestamp in observed:
            result[timestamp] = observed[timestamp]
            continue
        before = observed.get(timestamp - timedelta(hours=1))
        after = observed.get(timestamp + timedelta(hours=1))
        if before is None or after is None:
            raise ValueError(f"Cannot normalize missing DST reserve quarter: {timestamp}")
        result[timestamp] = {
            product: (
                (before[product][0] + after[product][0]) / 2.0,
                (before[product][1] + after[product][1]) / 2.0,
            )
            for product in RESERVE_PRODUCTS
        }
        interpolated_quarters.append(timestamp.isoformat(timespec="seconds"))
    if len(result) != EXPECTED_QUARTERS:
        raise ValueError(f"Expected {EXPECTED_QUARTERS} reserve quarters, found {len(result)}")
    return result, {
        "spring_forward_interpolated_quarters": interpolated_quarters,
        "fall_back_duplicate_quarters_averaged": sorted(set(duplicated_quarters)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    local_case = data_root / CASE_ID
    raw_dir = local_case / "raw" / "iso-ne"
    profile_dir = local_case / "profiles"
    profile_dir.mkdir(parents=True, exist_ok=True)
    hourly_path = download(HOURLY_URL, raw_dir / "2025_smd_hourly.xlsx", args.force_download)
    reserve_path = download(
        RESERVE_URL,
        raw_dir / "5min_reserve_price_and_designation_2025.xlsx",
        args.force_download,
    )

    hourly, dst_normalization = read_hourly_lmp(hourly_path)
    prices_path = profile_dir / "prices_15min.csv"
    with prices_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", "location", "rt_lmp_usd_per_mwh"])
        for hour_start, lmp in hourly:
            for quarter in range(4):
                timestamp = hour_start + timedelta(minutes=15 * quarter)
                writer.writerow(
                    [timestamp.isoformat(timespec="seconds"), "NEMA/Boston", f"{lmp:.6f}"]
                )

    reserve, reserve_dst_normalization = read_reserve_5min(reserve_path)
    reserve_output_path = profile_dir / "reserve_15min.csv"
    with reserve_output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["timestamp", "reserve_product", "designation_mw", "price_usd_per_mwh"]
        )
        for timestamp in sorted(reserve):
            for product in RESERVE_PRODUCTS:
                designation, price = reserve[timestamp][product]
                writer.writerow(
                    [
                        timestamp.isoformat(timespec="seconds"),
                        product,
                        f"{designation:.6f}",
                        f"{price:.6f}",
                    ]
                )

    manifest = {
        "schema_version": "1.0",
        "case_id": CASE_ID,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "location": {"name": "NEMA/Boston", "reserve_zone_id": NEMA_RESERVE_ZONE_ID},
        "raw_sources": [
            {
                "url": HOURLY_URL,
                "path": str(hourly_path.relative_to(data_root)).replace("\\", "/"),
                "size_bytes": hourly_path.stat().st_size,
                "sha256": sha256_file(hourly_path),
            },
            {
                "url": RESERVE_URL,
                "path": str(reserve_path.relative_to(data_root)).replace("\\", "/"),
                "size_bytes": reserve_path.stat().st_size,
                "sha256": sha256_file(reserve_path),
            },
        ],
        "price_transformation": {
            "source_resolution": "hourly",
            "output_resolution": "15-minute piecewise constant",
            "rule": "repeat each final hourly RT LMP over four contained 15-minute intervals",
            "dst_normalization": dst_normalization,
        },
        "reserve_transformation": {
            "source_resolution": "5-minute",
            "output_resolution": "15-minute",
            "rule": "arithmetic mean of each three consecutive price and designation values",
            "products": list(RESERVE_PRODUCTS),
            "designation_is_not_requirement": True,
            "dst_normalization": reserve_dst_normalization,
        },
        "derived_files": [
            {
                "path": str(prices_path.relative_to(data_root)).replace("\\", "/"),
                "row_count": EXPECTED_QUARTERS,
                "sha256": sha256_file(prices_path),
            },
            {
                "path": str(reserve_output_path.relative_to(data_root)).replace("\\", "/"),
                "row_count": EXPECTED_QUARTERS * len(RESERVE_PRODUCTS),
                "sha256": sha256_file(reserve_output_path),
            },
        ],
        "causal_caveat": "Historical 2025 ISO-NE market series are exogenous and are not claimed to be causally paired with TMY weather or OEDI TMY load shapes.",
    }
    manifest_path = profile_dir / "market_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote prices={EXPECTED_QUARTERS} reserve={EXPECTED_QUARTERS * len(RESERVE_PRODUCTS)} "
        f"manifest={manifest_path}"
    )


if __name__ == "__main__":
    main()
