"""Exact parallel-work coverage for optional frequency Scan callback grains."""
import os
from pathlib import Path

import numpy as np
import pytest

from test_native_opcode_variants import scan_session


ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(os.environ.get("REUSE_GRAIN_BASELINE_LIBRARY", ROOT / "native/build-windows-opcode-scan16/Release/reuse_ort_ops.dll"))


@pytest.mark.parametrize("grain", [8, 32, 64])
@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("shape", [(2, 11, 17, 16), (5, 81, 256, 16),
                                   (2, 19, 273, 16), (2, 7, 19, 7)])
def test_grain_preserves_every_channel_bit_exactly(grain, threads, shape):
    candidate = Path(os.environ.get(f"REUSE_GRAIN_{grain}_LIBRARY",
                     ROOT / f"native/build-windows-grain{grain}/Release/reuse_ort_ops.dll"))
    if not BASELINE.is_file() or not candidate.is_file():
        pytest.skip("Build independent baseline and grain libraries first")
    batch, length, inner, rank = shape
    baseline = scan_session(BASELINE, shape, threads)
    optimized = scan_session(candidate, shape, threads)
    rng = np.random.default_rng(sum(shape) + grain + threads)
    values = {
        "u": rng.normal(0, .7, (batch, length, inner)).astype(np.float32),
        "delta": rng.uniform(1e-5, .5, (batch, length, inner)).astype(np.float32),
        "B": rng.normal(0, .5, (batch, length, rank)).astype(np.float32),
        "C": rng.normal(0, .5, (batch, length, rank)).astype(np.float32),
        "A": -np.exp(rng.uniform(-1, 5, (inner, rank))).astype(np.float32),
    }
    before = {name: value.copy() for name, value in values.items()}
    for iteration in range(3):
        # A second call exercises independent state reset; the last call must
        # cover channel tails without leaving any old output samples behind.
        if iteration == 2:
            values["u"].fill(0)
        expected = baseline.run(None, values)[0]
        actual = optimized.run(None, values)[0]
        np.testing.assert_array_equal(actual, expected)
        for name, value in values.items():
            if name != "u" or iteration != 2:
                np.testing.assert_array_equal(value, before[name])
