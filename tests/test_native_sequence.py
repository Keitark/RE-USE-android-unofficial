"""Independent numerical and memory-contract checks for fused temporal chunks."""
import os
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper
import pytest


ROOT = Path(__file__).resolve().parents[1]
LIBRARY = Path(os.environ.get("REUSE_CUSTOM_OP_LIBRARY", ROOT / "native/build-windows/Release/reuse_ort_ops.dll"))
pytestmark = pytest.mark.skipif(not LIBRARY.is_file(), reason="Build the native custom operator first")
NAMES = ("u", "delta", "B", "C", "A", "state")


def sequence_session(batch, length, inner, rank, *, overrides=None, threads=2):
    shape = [batch, length, inner]
    shapes = {"u": shape, "delta": shape, "B": [batch, length, rank],
              "C": [batch, length, rank], "A": [inner, rank], "state": [batch, inner, rank]}
    shapes.update(overrides or {})
    graph = helper.make_graph(
        [helper.make_node("SelectiveSequence", list(NAMES), ["y", "next_state"], domain="com.reuse")],
        "test_temporal_sequence",
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, shapes[name]) for name in NAMES],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, shape),
         helper.make_tensor_value_info("next_state", TensorProto.FLOAT, [batch, inner, rank])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.reuse", 1)])
    model.ir_version = 10
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.register_custom_ops_library(str(LIBRARY.resolve()))
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def independent_reference(u, delta, b, c, a, state):
    state = state.copy()
    observed = np.empty_like(u)
    for frame in range(u.shape[1]):
        dt = delta[:, frame, :, None]
        state = np.exp(dt * a) * state + (dt * b[:, frame, None, :]) * u[:, frame, :, None]
        observed[:, frame] = np.sum(state * c[:, frame, None, :], axis=-1)
    return observed, state


def values_for(batch, length, inner, rank, seed=9):
    rng = np.random.default_rng(seed)
    return (rng.normal(0, .7, (batch, length, inner)).astype(np.float32),
            rng.uniform(1e-5, .5, (batch, length, inner)).astype(np.float32),
            rng.normal(0, .5, (batch, length, rank)).astype(np.float32),
            rng.normal(0, .5, (batch, length, rank)).astype(np.float32),
            -np.exp(rng.uniform(-1, 5, (inner, rank))).astype(np.float32),
            rng.normal(0, .3, (batch, inner, rank)).astype(np.float32))


@pytest.mark.parametrize("shape", [(1, 1, 7, 5), (2, 5, 17, 7), (2, 5, 17, 16),
                                   (81, 1, 256, 16), (81, 5, 256, 16), (161, 5, 256, 16)])
def test_sequence_matches_independent_recurrence_and_preserves_inputs(shape):
    values = values_for(*shape, seed=sum(shape))
    originals = [value.copy() for value in values]
    expected = independent_reference(*values)
    actual = sequence_session(*shape).run(None, dict(zip(NAMES, values)))
    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference, rtol=3e-5, atol=3e-6)
        assert np.isfinite(value).all()
    for value, before in zip(values, originals):
        np.testing.assert_array_equal(value, before)
    assert not np.shares_memory(actual[1], values[-1])


@pytest.mark.parametrize("length", [1, 5])
def test_sequence_is_bit_identical_to_chronological_native_steps(length):
    from test_native_scan import step_session
    values = values_for(2, length, 17, 16, seed=length + 1024)
    u, delta, b, c, a, state = values
    step = step_session(2, 17, 16)
    rows = []
    for frame in range(length):
        observed, state = step.run(None, dict(zip(NAMES, (
            u[:, frame], delta[:, frame], b[:, frame], c[:, frame], a, state))))
        rows.append(observed)
    actual = sequence_session(2, length, 17, 16).run(None, dict(zip(NAMES, values)))
    np.testing.assert_array_equal(actual[0], np.stack(rows, axis=1))
    np.testing.assert_array_equal(actual[1], state)


@pytest.mark.parametrize("overrides", [{"delta": [1, 5, 3]}, {"B": [1, 4, 4]},
                                       {"state": [1, 4, 3]}])
def test_sequence_rejects_inconsistent_shapes(overrides):
    session = sequence_session(1, 5, 4, 4, overrides=overrides)
    values = dict(zip(NAMES, values_for(1, 5, 4, 4)))
    for name, shape in overrides.items():
        values[name] = np.zeros(shape, np.float32)
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidArgument, match="SelectiveSequence requires"):
        session.run(None, values)


def test_sequence_rejects_empty_time_dimension():
    session = sequence_session(1, 0, 4, 4)
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidArgument, match="nonempty"):
        session.run(None, dict(zip(NAMES, values_for(1, 0, 4, 4))))


def test_sequence_rejects_excessive_state_rank_without_large_allocation():
    session = sequence_session(1, 1, 1, 1025)
    with pytest.raises(ort.capi.onnxruntime_pybind11_state.InvalidArgument, match="supported tensor size"):
        session.run(None, dict(zip(NAMES, values_for(1, 1, 1, 1025))))


def test_sequence_rejects_pinned_output_state_alias_before_input_mutation():
    values = values_for(1, 5, 7, 16)
    state_before = values[-1].copy()
    session = sequence_session(1, 5, 7, 16)
    binding = session.io_binding()
    for name, value in zip(NAMES, values):
        binding.bind_cpu_input(name, value)
    binding.bind_output("y", "cpu")
    state = values[-1]
    binding.bind_output("next_state", "cpu", 0, np.float32, state.shape, state.ctypes.data)
    with pytest.raises(RuntimeError, match="distinct input and output buffers"):
        session.run_with_iobinding(binding)
    np.testing.assert_array_equal(state, state_before)


def test_near_unit_decay_remains_bounded_over_2048_hops_in_chunks():
    rng = np.random.default_rng(2048)
    batch, inner, rank, total_hops = 2, 7, 16, 2048
    state = rng.normal(0, .3, (batch, inner, rank)).astype(np.float32)
    expected_state = state.copy()
    a = -rng.uniform(.01, .1, (inner, rank)).astype(np.float32)
    sessions = {length: sequence_session(batch, length, inner, rank, threads=1) for length in (3, 5)}
    for offset in range(0, total_hops, 5):
        length = min(5, total_hops - offset)
        u = rng.normal(0, .03, (batch, length, inner)).astype(np.float32)
        delta = rng.uniform(1e-5, 1e-3, u.shape).astype(np.float32)
        b = rng.normal(0, .2, (batch, length, rank)).astype(np.float32)
        c = rng.normal(0, .2, b.shape).astype(np.float32)
        expected, expected_state = independent_reference(u, delta, b, c, a, expected_state)
        actual, state = sessions[length].run(None, dict(zip(NAMES, (u, delta, b, c, a, state))))
        np.testing.assert_allclose(state, expected_state, rtol=3e-4, atol=1e-5)
        np.testing.assert_allclose(actual, expected, rtol=3e-4, atol=1e-5)
