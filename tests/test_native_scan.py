"""Numerical and contract tests for the optional fused CPU custom operator."""
import os
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto
import onnxruntime as ort
import pytest


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path(os.environ.get("REUSE_CUSTOM_OP_LIBRARY", ROOT / "native/build-windows/Release/reuse_ort_ops.dll"))
pytestmark = pytest.mark.skipif(not LIBRARY.is_file(), reason="Build the native custom operator first")


def session_for(batch, length, inner, rank, *, b_length=None, threads=2):
    shape = [batch, length, inner]
    bc_shape = [batch, length if b_length is None else b_length, rank]
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
              for name, dims in (("u", shape), ("delta", shape), ("B", bc_shape), ("C", bc_shape), ("A", [inner, rank]))]
    graph = helper.make_graph([helper.make_node("SelectiveScan", ["u", "delta", "B", "C", "A"], ["y"], domain="com.reuse")],
                              "test_scan", inputs, [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.reuse", 1)])
    model.ir_version = 10
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.register_custom_ops_library(str(LIBRARY.resolve()))
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def reference(u, delta, b, c, a):
    batch, length, inner = u.shape
    h = np.zeros((batch, inner, a.shape[1]), np.float32)
    result = np.empty_like(u)
    for t in range(length):
        h = np.exp(delta[:, t, :, None] * a) * h + delta[:, t, :, None] * b[:, t, None, :] * u[:, t, :, None]
        result[:, t] = np.sum(h * c[:, t, None, :], axis=-1)
    return result


@pytest.mark.parametrize("shape", [(1, 1, 7, 5), (2, 11, 17, 16), (1, 161, 256, 16)])
def test_native_matches_independent_recurrence(shape):
    batch, length, inner, rank = shape
    rng = np.random.default_rng(sum(shape))
    u = rng.normal(0, .7, (batch, length, inner)).astype(np.float32)
    delta = rng.uniform(1e-4, .5, u.shape).astype(np.float32)
    b = rng.normal(0, .5, (batch, length, rank)).astype(np.float32)
    c = rng.normal(0, .5, b.shape).astype(np.float32)
    a = -np.exp(rng.uniform(-1, 5, (inner, rank))).astype(np.float32)
    expected = reference(u, delta, b, c, a)
    session = session_for(*shape)
    actual = session.run(None, dict(zip(("u", "delta", "B", "C", "A"), (u, delta, b, c, a))))[0]
    np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-6)
    assert np.isfinite(actual).all()


def test_simd_exp_relative_error_below_one_part_per_million():
    # Set the first recurrent state to one, then observe its pure exponential
    # decay. N=4 exercises SIMD rather than the scalar remainder path.
    values = np.linspace(-80., 0., 10001, dtype=np.float32)
    u = np.stack((np.ones_like(values), np.zeros_like(values)))[None]
    delta = np.ones_like(u)
    b = np.ones((1, 2, 4), np.float32)
    c = np.zeros_like(b)
    c[0, 1, 0] = 1.
    a = np.repeat(values[:, None], 4, axis=1)
    actual = session_for(1, 2, len(values), 4).run(None, {"u": u, "delta": delta, "B": b, "C": c, "A": a})[0][0, 1]
    expected = np.exp(values)
    relative = np.abs(actual - expected) / expected
    assert float(relative.max()) < 1e-6


def test_invalid_shapes_return_runtime_error():
    session = session_for(1, 3, 4, 4, b_length=2)
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidArgument, match="SelectiveScan requires"):
        session.run(None, {"u": np.zeros((1, 3, 4), np.float32), "delta": np.ones((1, 3, 4), np.float32),
                           "B": np.ones((1, 2, 4), np.float32), "C": np.ones((1, 2, 4), np.float32),
                           "A": -np.ones((4, 4), np.float32)})


def step_session(batch, inner, rank, *, state_rank=None, threads=2):
    shape = [batch, inner]
    state_shape = [batch, inner, rank if state_rank is None else state_rank]
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, dims)
              for name, dims in (("u", shape), ("delta", shape), ("B", [batch, rank]),
                                 ("C", [batch, rank]), ("A", [inner, rank]), ("state", state_shape))]
    graph = helper.make_graph([helper.make_node("SelectiveStep", ["u", "delta", "B", "C", "A", "state"],
                                               ["y", "next_state"], domain="com.reuse")],
                              "test_step", inputs,
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape),
                               helper.make_tensor_value_info("next_state", TensorProto.FLOAT, [batch, inner, rank])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.reuse", 1)])
    model.ir_version = 10
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.register_custom_ops_library(str(LIBRARY.resolve()))
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def step_reference(u, delta, b, c, a, state):
    next_state = np.exp(delta[..., None] * a) * state + delta[..., None] * b[:, None, :] * u[..., None]
    return np.sum(next_state * c[:, None, :], axis=-1), next_state


@pytest.mark.parametrize("shape", [(1, 7, 5), (2, 17, 16), (81, 256, 16), (161, 256, 16)])
def test_native_step_matches_reference_without_mutating_input(shape):
    batch, inner, rank = shape
    rng = np.random.default_rng(sum(shape) + 17)
    values = (rng.normal(0, .7, (batch, inner)).astype(np.float32),
              rng.uniform(1e-5, .5, (batch, inner)).astype(np.float32),
              rng.normal(0, .5, (batch, rank)).astype(np.float32),
              rng.normal(0, .5, (batch, rank)).astype(np.float32),
              -np.exp(rng.uniform(-1, 5, (inner, rank))).astype(np.float32),
              rng.normal(0, .3, (batch, inner, rank)).astype(np.float32))
    state_before = values[-1].copy()
    expected = step_reference(*values)
    actual = step_session(*shape).run(None, dict(zip(("u", "delta", "B", "C", "A", "state"), values)))
    for a, e in zip(actual, expected):
        np.testing.assert_allclose(a, e, rtol=3e-5, atol=3e-6)
        assert np.isfinite(a).all()
    np.testing.assert_array_equal(values[-1], state_before)
    assert not np.shares_memory(actual[1], values[-1])


def test_temporal_state_near_unit_decay_remains_bounded_for_2048_updates():
    rng = np.random.default_rng(2048)
    batch, inner, rank = 2, 7, 16
    u = rng.normal(0, .03, (batch, inner)).astype(np.float32)
    b = rng.normal(0, .2, (batch, rank)).astype(np.float32)
    c = rng.normal(0, .2, b.shape).astype(np.float32)
    a = -rng.uniform(.01, .1, (inner, rank)).astype(np.float32)
    actual_state = rng.normal(0, .3, (batch, inner, rank)).astype(np.float32)
    expected_state = actual_state.copy()
    session = step_session(batch, inner, rank, threads=1)
    for step in range(2048):
        delta = rng.uniform(1e-5, 1e-3, u.shape).astype(np.float32)
        expected, expected_state = step_reference(u, delta, b, c, a, expected_state)
        actual, actual_state = session.run(None, {"u": u, "delta": delta, "B": b, "C": c,
                                                  "A": a, "state": actual_state})
        if step % 127 == 0 or step == 2047:
            # Persistent state must not accumulate an unbounded approximation
            # error. This bound is tighter than full-network export acceptance.
            np.testing.assert_allclose(actual_state, expected_state, rtol=3e-4, atol=1e-5)
            np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=1e-5)


def test_native_step_rejects_invalid_state_shape():
    session = step_session(81, 4, 4, state_rank=3)
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidArgument, match="SelectiveStep requires"):
        session.run(None, {"u": np.zeros((81, 4), np.float32), "delta": np.ones((81, 4), np.float32),
                           "B": np.ones((81, 4), np.float32), "C": np.ones((81, 4), np.float32),
                           "A": -np.ones((4, 4), np.float32), "state": np.ones((81, 4, 3), np.float32)})
