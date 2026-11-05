from __future__ import annotations

from pathlib import Path

import pytest

from experiments.acquire_case_assets import patch_runtime_idf


def test_runtime_patch_aligns_nominal_2025_weekday(tmp_path: Path) -> None:
    path = tmp_path / "model.idf"
    path.write_text(
        """Version,23.1;
Timestep,6;
RunPeriod,
  RUNPERIOD 1,
  1,
  1,
  ,
  12,
  31,
  ,
  Sunday, !- Day of Week for Start Day
  No;
""",
        encoding="utf-8",
    )
    patch_runtime_idf(path)
    result = path.read_text(encoding="utf-8")
    assert "Timestep,4;" in result
    assert "Wednesday, !- Day of Week for Start Day" in result
    assert "Output:Meter,Electricity:Facility,Timestep;" in result


def test_runtime_patch_rejects_missing_runperiod(tmp_path: Path) -> None:
    path = tmp_path / "model.idf"
    path.write_text("Version,23.1;\nTimestep,4;\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No RunPeriod"):
        patch_runtime_idf(path)
