"""Independent stateless causal frequency Conv/SiLU and strict graph proofs."""
import copy
import os
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper
import onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import InvalidArgument
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from optimize_frequency_conv import optimize_graph
LIBRARY = Path(os.environ.get("REUSE_FREQUENCY_CONV_LIBRARY", ROOT / "native/build-windows-frequency-conv/Release/reuse_ort_ops.dll"))
SOURCE = ROOT / "models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal-channelnorm/model.onnx"


def session(batch, time, channels, threads=2, weight_shape=None):
    if not LIBRARY.is_file():
        pytest.skip("Build optional FrequencyConvSilu library first")
    graph = helper.make_graph([helper.make_node("FrequencyConvSilu", ["u", "weight", "bias"], ["y"], domain="com.reuse")], "frequency",
        [helper.make_tensor_value_info("u", TensorProto.FLOAT, [batch, time, channels]),
         helper.make_tensor_value_info("weight", TensorProto.FLOAT, weight_shape or [channels, 1, 4]),
         helper.make_tensor_value_info("bias", TensorProto.FLOAT, [channels])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [batch, time, channels])])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.reuse", 1)])
    model.ir_version = 10
    options = ort.SessionOptions(); options.intra_op_num_threads = threads; options.inter_op_num_threads = 1
    options.register_custom_ops_library(str(LIBRARY.resolve()))
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


def reference(u, weight, bias):
    values = np.zeros(u.shape, np.float64)
    for t in range(u.shape[1]):
        for tap in range(4):
            if t + tap >= 3:
                values[:, t, :] += u[:, t + tap - 3, :].astype(np.float64) * weight[:, 0, tap]
        values[:, t, :] += bias
    with np.errstate(invalid="ignore", under="ignore"):
        e = np.exp(-np.abs(values))
        result = values * np.where(values >= 0, 1 / (1 + e), e / (1 + e))
    return result.astype(np.float32)


@pytest.mark.parametrize("shape", [(1, 1, 1), (2, 2, 7), (1, 3, 4), (2, 5, 65), (3, 8, 17), (5, 81, 256)])
def test_independent_double_convolution_silu_and_input_immutability(shape):
    b, t, c = shape; rng = np.random.default_rng(b + t + c)
    feed = {"u": rng.normal(0, .5, shape).astype(np.float32),
            "weight": rng.normal(0, .3, (c, 1, 4)).astype(np.float32), "bias": rng.normal(0, .2, c).astype(np.float32)}
    before = {k: v.copy() for k, v in feed.items()}
    np.testing.assert_allclose(session(*shape).run(None, feed)[0], reference(**feed), rtol=3e-6, atol=2e-7)
    for key in feed:
        np.testing.assert_array_equal(feed[key], before[key])


@pytest.mark.parametrize("reverse", [False, True])
def test_padding_edges_impulses_and_direction_chronology(reverse):
    u = np.zeros((2, 9, 7), np.float32); u[:, 0, :] = 1; u[:, -1, :] = -2
    if reverse:
        u = np.ascontiguousarray(u[:, ::-1, :])
    weight = np.tile(np.asarray([-.2, .4, -.6, .8], np.float32), (7, 1)).reshape(7, 1, 4)
    feed = {"u": u, "weight": weight, "bias": np.linspace(-.3, .3, 7, dtype=np.float32)}
    np.testing.assert_allclose(session(2, 9, 7).run(None, feed)[0], reference(**feed), rtol=3e-6, atol=2e-7)


def test_zero_input_and_stateless_repeated_calls():
    s = session(1, 81, 7); rng = np.random.default_rng(23)
    weight = rng.normal(size=(7, 1, 4)).astype(np.float32); bias = np.zeros(7, np.float32)
    impulse = np.zeros((1, 81, 7), np.float32); impulse[:, -1, :] = 1
    s.run(None, {"u": impulse, "weight": weight, "bias": bias})
    zero = s.run(None, {"u": np.zeros_like(impulse), "weight": weight, "bias": bias})[0]
    np.testing.assert_array_equal(zero, np.zeros_like(impulse))


def test_serial_and_parallel_outputs_are_identical():
    rng = np.random.default_rng(73); shape = (5, 81, 256)
    feed = {"u": rng.normal(size=shape).astype(np.float32), "weight": rng.normal(size=(256, 1, 4)).astype(np.float32),
            "bias": rng.normal(size=256).astype(np.float32)}
    np.testing.assert_array_equal(session(*shape, threads=1).run(None, feed)[0], session(*shape, threads=4).run(None, feed)[0])


def test_silu_negative_range_subnormals_and_nonfinite_semantics():
    bias = np.concatenate((np.linspace(-80, 80, 4097, dtype=np.float32), np.asarray([-100, -np.inf, np.inf, np.nan], np.float32)))
    c = len(bias); feed = {"u": np.zeros((1, 1, c), np.float32), "weight": np.zeros((c, 1, 4), np.float32), "bias": bias}
    actual = session(1, 1, c).run(None, feed)[0][0, 0]
    expected = reference(**feed)[0, 0]
    np.testing.assert_allclose(actual[:4097], expected[:4097], rtol=2e-6, atol=1e-37)
    assert abs(float(actual[-4]) - float(expected[-4])) < 1e-37
    assert np.isnan(actual[-3]) and np.isposinf(actual[-2]) and np.isnan(actual[-1])


@pytest.mark.parametrize("alias", ["u", "weight", "bias"])
def test_pinned_input_alias_is_rejected_before_mutation(alias):
    shape = (1, 5, 4); rng = np.random.default_rng(47)
    feed = {"u": rng.normal(size=shape).astype(np.float32), "weight": rng.normal(size=(4, 1, 4)).astype(np.float32), "bias": rng.normal(size=4).astype(np.float32)}
    before = {k: v.copy() for k, v in feed.items()}; s = session(*shape); binding = s.io_binding()
    for k, v in feed.items():
        binding.bind_cpu_input(k, v)
    binding.bind_output("y", "cpu", 0, np.float32, shape, feed[alias].ctypes.data)
    with pytest.raises(RuntimeError, match="distinct input and output"):
        s.run_with_iobinding(binding)
    for key in feed:
        np.testing.assert_array_equal(feed[key], before[key])


@pytest.mark.parametrize("shape", [(4, 1, 3), (3, 1, 4), (4, 4, 1)])
def test_bad_weight_shape_rejected(shape):
    s = session(1, 5, 4, weight_shape=shape)
    with pytest.raises(InvalidArgument, match="requires u"):
        s.run(None, {"u": np.zeros((1, 5, 4), np.float32), "weight": np.zeros(shape, np.float32), "bias": np.zeros(4, np.float32)})


def source_model(path=SOURCE):
    if not path.is_file():
        pytest.skip("Frozen source model is unavailable")
    return onnx.load(path)


@pytest.mark.parametrize("channelnorm", [False, True])
def test_strict_six_region_rewrite_and_unchanged_public_contract(channelnorm):
    source = source_model() if channelnorm else source_model(ROOT / "models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal/model.onnx")
    before = source.SerializeToString(); candidate, proofs = optimize_graph(source)
    assert len(proofs) == 6
    assert sum(n.op_type == "FrequencyConvSilu" for n in candidate.graph.node) == 6
    assert sum(n.op_type == "Conv" for n in source.graph.node) - sum(n.op_type == "Conv" for n in candidate.graph.node) == 6
    assert source.graph.input == candidate.graph.input and source.graph.output == candidate.graph.output
    assert source.SerializeToString() == before
    original = {x.name: x.SerializeToString() for x in source.graph.initializer}
    for value in candidate.graph.initializer:
        assert value.SerializeToString() == original[value.name]


@pytest.mark.parametrize("mutation", ["padding", "group", "dilation", "weight_shape", "pre_transpose", "post_transpose",
    "extra_conv_consumer", "extra_activation_consumer", "public_internal", "slice_end", "slice_axis", "slice_step", "silu_operand", "nonfinite_weight"])
def test_rewrite_rejects_changed_contract(mutation):
    source = source_model(); model = copy.deepcopy(source); nodes = {n.name: n for n in model.graph.node}
    conv = nodes["/freq_mamba/forward_blocks/conv1d/Conv"]
    trim = nodes["/freq_mamba/forward_blocks/Slice_2"]
    pre = nodes["/freq_mamba/forward_blocks/Transpose"]; post = nodes["/freq_mamba/forward_blocks/Transpose_1"]
    if mutation in ("padding", "group", "dilation"):
        name, value = {"padding": ("pads", [2, 3]), "group": ("group", 128), "dilation": ("dilations", [2])}[mutation]
        a = next(a for a in conv.attribute if a.name == name)
        if isinstance(value, list):
            a.ints[:] = value
        else:
            a.i = value
    elif mutation == "weight_shape":
        next(v for v in model.graph.initializer if v.name == conv.input[1]).dims[:] = [256, 4, 1]
    elif mutation == "nonfinite_weight":
        value = next(v for v in model.graph.initializer if v.name == conv.input[1])
        data = onnx.numpy_helper.to_array(value).copy(); data.flat[0] = np.nan
        value.CopyFrom(onnx.numpy_helper.from_array(data, value.name))
    elif mutation in ("pre_transpose", "post_transpose"):
        next(a for a in (pre if mutation == "pre_transpose" else post).attribute if a.name == "perm").ints[:] = [0, 1, 2]
    elif mutation in ("extra_conv_consumer", "extra_activation_consumer"):
        value = conv.output[0] if mutation == "extra_conv_consumer" else post.output[0]
        model.graph.node.append(helper.make_node("Identity", [value], ["unexpected"], name="extra"))
    elif mutation == "public_internal":
        model.graph.output.append(helper.make_tensor_value_info(conv.output[0], TensorProto.FLOAT, [5, 256, 84]))
    elif mutation.startswith("slice_"):
        slot, value = {"slice_end": (2, 80), "slice_axis": (3, 1), "slice_step": (4, 2)}[mutation]
        model.graph.node.insert(0, helper.make_node("Constant", [], ["wrong_bound"], name="wrong_bound_constant", value=helper.make_tensor("v", TensorProto.INT64, [1], [value])))
        trim.input[slot] = "wrong_bound"
    else:
        nodes["/freq_mamba/forward_blocks/Mul_2"].input[1] = post.output[0]
    with pytest.raises((ValueError, onnx.shape_inference.InferenceError)):
        optimize_graph(model)
