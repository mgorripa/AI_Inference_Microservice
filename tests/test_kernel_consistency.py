"""Tests comparing the selected kernel path with NumPy ReLU."""

from __future__ import annotations

import numpy as np

from service import app as app_module


def test_active_kernel_matches_numpy_relu() -> None:
    input_values = np.asarray(
        [-10.0, -1.0, 0.0, 0.5, 3.0, 100.0],
        dtype=np.float32,
    )
    expected = np.maximum(input_values, 0)

    if app_module.KERNEL_AVAILABLE:
        actual = app_module.kbinding.vec_relu(input_values)
    else:
        actual = np.maximum(input_values, 0)

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
