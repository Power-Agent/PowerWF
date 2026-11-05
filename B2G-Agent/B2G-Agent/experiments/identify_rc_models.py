"""Identify deterministic seasonal RC/ARX screening models from frozen EnergyPlus data."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import date, timedelta
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CASE_ID = "ieee33_10bem_15min"
CASE_DIR = REPO_ROOT / "cases" / CASE_ID
DATA_ROOT = Path(os.getenv("B2G_DATA_ROOT", REPO_ROOT / ".local-data")).resolve()
LOCAL_CASE = DATA_ROOT / CASE_ID
PROFILE_DIR = LOCAL_CASE / "profiles"
EXPECTED_INTERVALS = 365 * 96


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ridge_affine_fit(
    features: np.ndarray, target: np.ndarray, ridge_lambda: float
) -> tuple[np.ndarray, float]:
    """Fit standardized ridge slopes and return coefficients in original units."""
    if features.ndim != 2 or target.ndim != 1 or len(features) != len(target):
        raise ValueError("features and target dimensions do not align")
    means = features.mean(axis=0)
    scales = features.std(axis=0)
    scales[scales < 1e-12] = 1.0
    standardized = (features - means) / scales
    design = np.column_stack([standardized, np.ones(len(target))])
    penalty = np.eye(design.shape[1]) * ridge_lambda
    penalty[-1, -1] = 0.0
    fitted = np.linalg.solve(design.T @ design + penalty, design.T @ target)
    coefficients = fitted[:-1] / scales
    intercept = float(fitted[-1] - coefficients @ means)
    return coefficients, intercept


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = actual - predicted
    rmse = float(np.sqrt(np.mean(residual**2)))
    mae = float(np.mean(np.abs(residual)))
    denominator = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else 0.0
    return {"rmse_c": rmse, "mae_c": mae, "r2": r2}


def load_profile(path: Path, value_column: str) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            bem_id = row["bem_id"]
            if bem_id not in values:
                values[bem_id] = np.zeros(EXPECTED_INTERVALS)
                counts[bem_id] = 0
            index = counts[bem_id]
            if index >= EXPECTED_INTERVALS:
                raise ValueError(f"Too many rows for {bem_id} in {path}")
            values[bem_id][index] = float(row[value_column])
            counts[bem_id] += 1
    if len(values) != 10 or any(count != EXPECTED_INTERVALS for count in counts.values()):
        raise ValueError(f"{path} must contain 35,040 rows for each of ten BEMs")
    return values


def load_weather() -> tuple[np.ndarray, np.ndarray]:
    try:
        import pvlib
    except ImportError as exc:
        raise RuntimeError("pvlib is required to read the frozen EPW") from exc
    weather_path = (
        LOCAL_CASE / "weather" / "USA_NY_Buffalo.Niagara.Intl_.AP_.725280_TMY3.epw"
    )
    weather, _ = pvlib.iotools.read_epw(weather_path)
    if len(weather) != 8760:
        raise ValueError("The frozen EPW must contain 8,760 hourly rows")
    outdoor = np.repeat(weather["temp_air"].to_numpy(dtype=float), 4)
    irradiance = np.repeat(weather["ghi"].to_numpy(dtype=float) / 1000.0, 4)
    return outdoor, irradiance


def day_labels() -> np.ndarray:
    first = date(2025, 1, 1)
    return np.array(
        [(first + timedelta(days=interval // 96)).isoformat() for interval in range(EXPECTED_INTERVALS)]
    )


def regime_masks(outdoor: np.ndarray, config: dict[str, object]) -> dict[str, np.ndarray]:
    regimes = config["regimes"]
    heating_max = float(regimes["heating"]["maximum_outdoor_temperature_c"])
    shoulder_min = float(regimes["shoulder"]["minimum_outdoor_temperature_c"])
    shoulder_max = float(regimes["shoulder"]["maximum_outdoor_temperature_c"])
    cooling_min = float(regimes["cooling"]["minimum_outdoor_temperature_c"])
    return {
        "heating": outdoor <= heating_max,
        "shoulder": (outdoor > shoulder_min) & (outdoor < shoulder_max),
        "cooling": outdoor >= cooling_min,
    }


def main() -> None:
    config_path = CASE_DIR / "optimization" / "rc_identification.json"
    split_path = CASE_DIR / "days" / "calendar_split.json"
    baseline_path = PROFILE_DIR / "bem_baseline_15min.csv"
    thermal_path = PROFILE_DIR / "bem_thermal_15min.csv"
    weather_path = (
        LOCAL_CASE / "weather" / "USA_NY_Buffalo.Niagara.Intl_.AP_.725280_TMY3.epw"
    )
    for path in (config_path, split_path, baseline_path, thermal_path, weather_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    config = json.loads(config_path.read_text(encoding="utf-8"))
    split = json.loads(split_path.read_text(encoding="utf-8"))
    training_dates = set(split["development_dates"])
    validation_dates = set(split["reference_dates"])
    electricity = load_profile(baseline_path, "single_building_kw")
    temperature = load_profile(thermal_path, "mean_zone_c")
    outdoor, irradiance = load_weather()
    dates = day_labels()
    interval = np.arange(EXPECTED_INTERVALS)
    quarter_of_day = interval % 96
    angle = 2.0 * np.pi * quarter_of_day / 96.0
    weekend = np.array(
        [date.fromisoformat(value).weekday() >= 5 for value in dates], dtype=float
    )
    feature_names = list(config["features"])
    ridge_lambda = float(config["ridge_lambda"])
    minimum_samples = int(config["minimum_training_samples_per_regime"])
    minimum_control = float(config["control_coefficient_minimum_absolute_value"])
    rmse_limit = float(config["validation_rmse_limit_c"])
    masks = regime_masks(outdoor, config)
    models: dict[str, dict[str, object]] = {}

    for bem_id in sorted(electricity):
        temp = temperature[bem_id]
        power = electricity[bem_id]
        features = np.column_stack(
            [
                temp[:-1],
                power[:-1],
                outdoor[:-1],
                irradiance[:-1],
                np.sin(angle[:-1]),
                np.cos(angle[:-1]),
                weekend[:-1],
            ]
        )
        target = temp[1:]
        same_day = dates[:-1] == dates[1:]
        bem_models: dict[str, object] = {}
        for regime, regime_mask in masks.items():
            train = (
                np.isin(dates[:-1], list(training_dates))
                & same_day
                & regime_mask[:-1]
            )
            validate = (
                np.isin(dates[:-1], list(validation_dates))
                & same_day
                & regime_mask[:-1]
            )
            if int(train.sum()) < minimum_samples or int(validate.sum()) < 1:
                raise RuntimeError(
                    f"Insufficient {regime} samples for {bem_id}: "
                    f"train={int(train.sum())} validation={int(validate.sum())}"
                )
            coefficients, intercept = ridge_affine_fit(
                features[train], target[train], ridge_lambda
            )
            train_prediction = features[train] @ coefficients + intercept
            validation_prediction = features[validate] @ coefficients + intercept
            validation_metrics = regression_metrics(target[validate], validation_prediction)
            coefficient_map = {
                name: float(value)
                for name, value in zip(feature_names, coefficients, strict=True)
            }
            control_coefficient = coefficient_map["single_building_electricity_kw"]
            bem_models[regime] = {
                "state_equation": {
                    "intercept_c": intercept,
                    "coefficients": coefficient_map,
                },
                "training_sample_count": int(train.sum()),
                "validation_sample_count": int(validate.sum()),
                "training_metrics": regression_metrics(target[train], train_prediction),
                "validation_metrics": validation_metrics,
                "observed_power_kw": {
                    "minimum": float(power[:-1][train].min()),
                    "p05": float(np.quantile(power[:-1][train], 0.05)),
                    "p95": float(np.quantile(power[:-1][train], 0.95)),
                    "maximum": float(power[:-1][train].max()),
                },
                "stable_one_step_state_coefficient": bool(
                    0.0 < coefficient_map["mean_zone_temperature_c"] < 1.0
                ),
                "control_coefficient_identified": bool(
                    abs(control_coefficient) >= minimum_control
                ),
                "validation_rmse_within_limit": bool(
                    validation_metrics["rmse_c"] <= rmse_limit
                ),
            }
        models[bem_id] = bem_models
        print(
            bem_id,
            " ".join(
                f"{name}:rmse={bem_models[name]['validation_metrics']['rmse_c']:.4f}"
                for name in ("heating", "shoulder", "cooling")
            ),
        )

    output = {
        "schema_version": "1.0",
        "case_id": CASE_ID,
        "identification_kind": "observational_energyplus_seasonal_arx",
        "time_resolution_minutes": 15,
        "feature_order": feature_names,
        "training_split": "development_dates",
        "validation_split": "reference_dates",
        "models": models,
        "limitations": [
            "Facility electricity is an observational proxy for HVAC control and is confounded by non-HVAC loads.",
            "The model is a candidate-schedule screening model; high-fidelity EnergyPlus comfort validation remains mandatory.",
            "Target/test-candidate days are excluded from coefficient fitting and validation metrics.",
        ],
        "inputs": {
            "config": {"path": str(config_path.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": sha256_file(config_path)},
            "calendar_split": {"path": str(split_path.relative_to(REPO_ROOT)).replace("\\", "/"), "sha256": sha256_file(split_path)},
            "bem_baseline": {"path": str(baseline_path.relative_to(DATA_ROOT)).replace("\\", "/"), "sha256": sha256_file(baseline_path)},
            "bem_thermal": {"path": str(thermal_path.relative_to(DATA_ROOT)).replace("\\", "/"), "sha256": sha256_file(thermal_path)},
            "weather": {"path": str(weather_path.relative_to(DATA_ROOT)).replace("\\", "/"), "sha256": sha256_file(weather_path)},
        },
    }
    output_dir = LOCAL_CASE / "models"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "rc_parameters.json"
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {output_path} sha256={sha256_file(output_path)}")


if __name__ == "__main__":
    main()
