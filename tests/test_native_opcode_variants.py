"""Exact frequency-scan parity for the optional rank16 register specialization.

Build candidate with REUSE_SCAN_N16_REGISTERS=ON. ARM FMA has separate rounding
and must additionally pass the on-device kernel and full-model acceptance checks.
"""
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper
import pytest


ROOT = Path(__file__).resolve().parents[1]
BASELINE = Path(os.environ.get("REUSE_OPCODE_BASELINE_LIBRARY", ROOT / "native/build-windows-softplus/Release/reuse_ort_ops.dll"))
CANDIDATE = Path(os.environ.get("REUSE_OPCODE_CANDIDATE_LIBRARY", ROOT / "native/build-windows-opcode-scan16/Release/reuse_ort_ops.dll"))
pytestmark = pytest.mark.skipif(not BASELINE.is_file() or not CANDIDATE.is_file(),
                                reason="Build independent baseline and scan16 opcode libraries first")


def scan_session(library, shape, threads):
    batch, length, inner, rank = shape
    shapes = {"u": [batch, length, inner], "delta": [batch, length, inner],
              "B": [batch, length, rank], "C": [batch, length, rank], "A": [inner, rank]}
    graph = helper.make_graph(
        [helper.make_node("SelectiveScan", list(shapes), ["y"], domain="com.reuse")], "scan_opcode_parity",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, dims) for name, dims in shapes.items()],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, length, inner])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.reuse", 1)])
    model.ir_version = 10
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.register_custom_ops_library(str(library.resolve()))
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("shape", [(1, 1, 7, 16), (2, 11, 17, 16), (5, 81, 256, 16),
                                   (1, 161, 256, 16), (2, 17, 19, 7), (2, 5, 17, 64)])
@pytest.mark.parametrize("threads", [1, 4])
def test_scan_register_state_is_bit_identical_to_baseline(shape, threads):
    batch, length, inner, rank = shape
    rng = np.random.default_rng(sum(shape) + threads)
    baseline = scan_session(BASELINE, shape, threads)
    candidate = scan_session(CANDIDATE, shape, threads)
    for repetition in range(3):
        values = {
            "u": rng.normal(0, .7, (batch, length, inner)).astype(np.float32),
            "delta": rng.uniform(1e-5, .5, (batch, length, inner)).astype(np.float32),
            "B": rng.normal(0, .5, (batch, length, rank)).astype(np.float32),
            "C": rng.normal(0, .5, (batch, length, rank)).astype(np.float32),
            "A": -np.exp(rng.uniform(-1, 5, (inner, rank))).astype(np.float32),
        }
        if repetition == 1:
            values["delta"].fill(1e-6)  # Near-unit recurrent decay.
        elif repetition == 2:
            values["u"].fill(0)  # A new call must start with a cleared state.
        before = {name: value.copy() for name, value in values.items()}
        expected = baseline.run(None, values)[0]
        actual = candidate.run(None, values)[0]
        np.testing.assert_array_equal(actual, expected)
        for name, original in before.items():
            np.testing.assert_array_equal(values[name], original)
