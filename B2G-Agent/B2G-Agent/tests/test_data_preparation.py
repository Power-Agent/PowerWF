from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from experiments.prepare_background_profiles import (
    ELECTRICITY_COLUMN,
    normalize_energy_shape_to_peak,
    read_tmy_shape,
    source_allocation,
)
from experiments.prepare_market_profiles import (
    excel_serial_to_datetime,
    normalize_hourly_lmp,
)


def _write_profile(path: Path, *, step_minutes: int = 15) -> None:
    start = datetime(2018, 1, 1, 0, 15)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timestamp", ELECTRICITY_COLUMN])
        writer.writeheader()
        for index, value in enumerate([1.0, 2.0, 3.0, 4.0]):
            writer.writerow(
                {
                    "timestamp": (start + timedelta(minutes=step_minutes * index)).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    ELECTRICITY_COLUMN: value,
                }
            )


def test_background_source_allocation_covers_all_buses_and_matches_bems() -> None:
    allocation = source_allocation(range(2, 34))
    assert sorted(allocation) == list(range(2, 34))
    assert allocation[3].source_id == "comstock:smalloffice"
    assert allocation[27].source_id == "resstock:multifamily5plus"
    assert allocation[33].source_id == "comstock:warehouse"


def test_tmy_reader_and_peak_normalization(tmp_path: Path) -> None:
    path = tmp_path / "profile.csv"
    _write_profile(path)
    energy = read_tmy_shape(path, expected_intervals=4)
    assert energy == [1.0, 2.0, 3.0, 4.0]
    power = normalize_energy_shape_to_peak(energy, 60.0)
    assert max(power) == pytest.approx(60.0)
    assert power == pytest.approx([15.0, 30.0, 45.0, 60.0])


def test_tmy_reader_rejects_non_15_minute_source(tmp_path: Path) -> None:
    path = tmp_path / "profile.csv"
    _write_profile(path, step_minutes=30)
    with pytest.raises(ValueError, match="Non-15-minute timestamp"):
        read_tmy_shape(path, expected_intervals=4)


def test_market_hour_ending_dst_normalization() -> None:
    spring_day = date(2025, 3, 9)
    fall_day = date(2025, 11, 2)
    records: list[tuple[date, int, float]] = []
    for encoded_hour in range(21, 45):
        if encoded_hour != 22:
            records.append((spring_day, encoded_hour, float(encoded_hour)))
        records.append((fall_day, encoded_hour, float(encoded_hour)))
    records.append((fall_day, 110, 100.0))

    normalized, metadata = normalize_hourly_lmp(records, expected_days=2)
    assert len(normalized) == 48
    spring_he02 = normalized[1][1]
    fall_he02 = normalized[25][1]
    assert spring_he02 == pytest.approx((21.0 + 23.0) / 2.0)
    assert fall_he02 == pytest.approx((22.0 + 100.0) / 2.0)
    assert metadata["spring_forward_interpolated_days"] == ["2025-03-09"]
    assert metadata["fall_back_he02_he02x_averaged_days"] == ["2025-11-02"]


def test_excel_serial_epoch_matches_iso_ne_workbook() -> None:
    assert excel_serial_to_datetime("45658") == datetime(2025, 1, 1)
    assert excel_serial_to_datetime("45658.003472222219") == datetime(2025, 1, 1, 0, 5)
