from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "identify_rc_models.py"
SPEC = importlib.util.spec_from_file_location("identify_rc_models", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_ridge_affine_fit_recovers_original_units() -> None:
    x = np.column_stack([np.linspace(-2.0, 3.0, 100), np.linspace(4.0, 40.0, 100) ** 2])
    expected = np.array([0.7, -0.025])
    y = x @ expected + 4.2
    coefficients, intercept = MODULE.ridge_affine_fit(x, y, 1e-10)
    assert np.allclose(coefficients, expected, atol=1e-8)
    assert abs(intercept - 4.2) < 1e-8


def test_regression_metrics_are_exact_for_perfect_prediction() -> None:
    values = np.array([1.0, 2.0, 4.0])
    metrics = MODULE.regression_metrics(values, values.copy())
    assert metrics == {"rmse_c": 0.0, "mae_c": 0.0, "r2": 1.0}
