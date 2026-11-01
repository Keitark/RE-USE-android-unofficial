"""Independent scalar/SIMD and recurrent error bounds for optional FastSoftplus."""
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper
import pytest

ROOT = Path(__file__).resolve().parents[1]
_CANDIDATE_LIBRARY = ROOT / "native/build-windows-softplus/Release/reuse_ort_ops.dll"
_DEFAULT_LIBRARY = (_CANDIDATE_LIBRARY if _CANDIDATE_LIBRARY.is_file() else
                    ROOT / "native/build-windows/Release/reuse_ort_ops.dll")
LIBRARY = Path(os.environ.get("REUSE_SOFTPLUS_LIBRARY", os.environ.get("REUSE_CUSTOM_OP_LIBRARY", _DEFAULT_LIBRARY)))
pytestmark = pytest.mark.skipif(not LIBRARY.is_file(), reason="Build optional FastSoftplus native library first")


def activation_session(shape, threads=2):
    graph = helper.make_graph(
        [helper.make_node("FastSoftplus", ["x"], ["y"], domain="com.reuse")], "fast_softplus",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.reuse", 1)])
    model.ir_version = 10
    options = ort.SessionOptions(); options.intra_op_num_threads = threads; options.inter_op_num_threads = 1
    options.register_custom_ops_library(str(LIBRARY.resolve()))
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("shape", [[], [1], [3], [4], [17], [2, 5, 256], [81, 5, 256], [0, 5]])
def test_shapes_and_accuracy_without_input_mutation(shape):
    rng = np.random.default_rng(sum(shape) + 91)
    values = rng.uniform(-80, 80, shape).astype(np.float32)
    before = values.copy()
    actual = activation_session(shape).run(None, {"x": values})[0]
    expected = np.logaddexp(0., values.astype(np.float64)).astype(np.float32)
    assert actual.shape == values.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-37)
    np.testing.assert_array_equal(values, before)


def test_dense_grid_normal_range_relative_error():
    values = np.concatenate((np.linspace(-80, 80, 20001, dtype=np.float32),
                             np.random.default_rng(123).uniform(-80, 80, 50003).astype(np.float32)))
    actual = activation_session(values.shape).run(None, {"x": values})[0]
    expected = np.logaddexp(0., values.astype(np.float64)).astype(np.float32)
    assert float(np.max(np.abs((actual-expected)/expected))) < 2e-6
    assert np.all(actual >= 0)


def test_extremes_and_nonfinite_values_keep_standard_semantics():
    values = np.asarray([-np.inf, np.inf, np.nan, -0., 0., -90., -100., -1000., 1000.], np.float32)
    actual = activation_session(values.shape).run(None, {"x": values})[0]
    with np.errstate(invalid="ignore"):
        expected = np.logaddexp(0., values.astype(np.float64)).astype(np.float32)
    assert actual[0] == 0 and np.isposinf(actual[1]) and np.isnan(actual[2])
    np.testing.assert_allclose(actual[3:], expected[3:], rtol=2e-6, atol=1e-37)


def test_pinned_output_alias_is_rejected_before_mutation():
    values = np.asarray([-10., -1., 0., 1.], np.float32); before = values.copy()
    session = activation_session(values.shape); binding = session.io_binding()
    binding.bind_cpu_input("x", values)
    binding.bind_output("y", "cpu", 0, np.float32, values.shape, values.ctypes.data)
    with pytest.raises(RuntimeError, match="distinct input and output"):
        session.run_with_iobinding(binding)
    np.testing.assert_array_equal(values, before)


def test_delta_error_stays_bounded_in_2048_near_unit_recurrent_updates():
    rng = np.random.default_rng(2048)
    shape = (2, 7)
    session = activation_session(shape, threads=1)
    state = rng.normal(0, .3, (2, 7, 16)).astype(np.float32); expected_state = state.copy()
    transition = -rng.uniform(.01, .1, (7, 16)).astype(np.float32)
    for update in range(2048):
        logits = rng.uniform(-11, -7, shape).astype(np.float32)
        delta = session.run(None, {"x": logits})[0]
        expected_delta = np.logaddexp(0., logits.astype(np.float64)).astype(np.float32)
        value = rng.normal(0, .03, shape).astype(np.float32)
        b = rng.normal(0, .2, (2, 16)).astype(np.float32)
        state = np.exp(delta[..., None]*transition)*state + (delta[..., None]*b[:, None])*value[..., None]
        expected_state = np.exp(expected_delta[..., None]*transition)*expected_state + (expected_delta[..., None]*b[:, None])*value[..., None]
        if update % 127 == 0 or update == 2047:
            np.testing.assert_allclose(state, expected_state, rtol=3e-4, atol=1e-5)
