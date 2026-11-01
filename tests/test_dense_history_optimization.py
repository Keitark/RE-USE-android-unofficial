"""Prove sparse history selection and conservative graph rewrite boundaries."""
from pathlib import Path
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
import onnxruntime as ort
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from optimize_dense_history import optimize_graph, static_shapes


def convolution_model(dilation, *, pad_value=0., share_pad=False, current_frames=1):
    rng = np.random.default_rng(dilation)
    inputs = [helper.make_tensor_value_info("cache", TensorProto.FLOAT, [1, 2, 2*dilation, 7]),
              helper.make_tensor_value_info("current", TensorProto.FLOAT, [1, 2, current_frames, 7])]
    weights = rng.normal(size=(3, 2, 3, 3)).astype(np.float32)
    tensors = [numpy_helper.from_array(weights, "weights"),
               numpy_helper.from_array(np.asarray([0, 0, 0, 1, 0, 0, 0, 1], np.int64), "padding"),
               numpy_helper.from_array(np.asarray(pad_value, np.float32), "pad_value"),
               numpy_helper.from_array(np.asarray([current_frames], np.int64), "start"),
               numpy_helper.from_array(np.asarray([2*dilation+current_frames], np.int64), "end"),
               numpy_helper.from_array(np.asarray([2], np.int64), "axis"),
               numpy_helper.from_array(np.asarray([1], np.int64), "step")]
    nodes = [helper.make_node("Concat", ["cache", "current"], ["history"], name="history", axis=2),
             helper.make_node("Slice", ["history", "start", "end", "axis", "step"], ["next_cache"], name="update"),
             helper.make_node("Pad", ["history", "padding", "pad_value"], ["padded"], name="pad", mode="constant"),
             helper.make_node("Conv", ["padded", "weights"], ["output"], name="conv",
                              kernel_shape=[3, 3], strides=[1, 1], pads=[0, 0, 0, 0], dilations=[dilation, 1])]
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, current_frames, 7]),
               helper.make_tensor_value_info("next_cache", TensorProto.FLOAT, [1, 2, 2*dilation, 7])]
    if share_pad:
        nodes.append(helper.make_node("Identity", ["padded"], ["shared"], name="other_pad_consumer"))
        outputs.append(helper.make_tensor_value_info("shared", TensorProto.FLOAT,
                                                   [1, 2, 2*dilation+current_frames, 9]))
    model = helper.make_model(helper.make_graph(nodes, "history_test", inputs, outputs, initializer=tensors),
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model, full_check=True)
    return model


def session(model):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    return ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])


@pytest.mark.parametrize("dilation", [1, 2, 4, 8])
def test_temporal_taps_and_next_cache_match_over_recurrent_updates(dilation):
    original = convolution_model(dilation)
    frozen_source = original.SerializeToString()
    optimized, report = optimize_graph(original)
    assert original.SerializeToString() == frozen_source
    assert len(report["folded_spatial_pads"]) == 1
    assert len(report["compacted_temporal_convolutions"]) == int(dilation > 1)
    original_weights = next(value for value in original.graph.initializer if value.name == "weights")
    rewritten_weights = next(value for value in optimized.graph.initializer if value.name == "weights")
    assert original_weights.SerializeToString() == rewritten_weights.SerializeToString()
    sessions = [session(model) for model in (original, optimized)]
    rng = np.random.default_rng(1000+dilation)
    caches = [rng.normal(size=(1, 2, 2*dilation, 7)).astype(np.float32)]
    caches.append(caches[0].copy())
    for frame in range(7):
        current = rng.normal(size=(1, 2, 1, 7)).astype(np.float32)
        result = [runner.run(None, {"cache": cache, "current": current})
                  for runner, cache in zip(sessions, caches)]
        np.testing.assert_allclose(result[1][0], result[0][0], rtol=1e-6, atol=2e-6,
                                   err_msg=f"dilation {dilation}, frame {frame}")
        np.testing.assert_array_equal(result[1][1], result[0][1])
        caches = [values[1] for values in result]


@pytest.mark.parametrize("kwargs", [{"pad_value": 1.}, {"share_pad": True}, {"current_frames": 4}])
def test_unsafe_or_overlapping_frame_patterns_are_not_compacted(kwargs):
    original = convolution_model(4, **kwargs)
    optimized, report = optimize_graph(original)
    assert not report["compacted_temporal_convolutions"]
    if "current_frames" not in kwargs:
        assert not report["folded_spatial_pads"]
    runners = [session(model) for model in (original, optimized)]
    rng = np.random.default_rng(44)
    values = {"cache": rng.normal(size=(1, 2, 8, 7)).astype(np.float32),
              "current": rng.normal(size=(1, 2, kwargs.get("current_frames", 1), 7)).astype(np.float32)}
    for actual, expected in zip(runners[1].run(None, values), runners[0].run(None, values)):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-6)


@pytest.mark.parametrize("mode", ["pads", "compact"])
def test_five_frame_dilation_eight_preserves_all_outputs_and_recurrent_cache(mode):
    original = convolution_model(8, current_frames=5)
    source = original.SerializeToString()
    optimized, report = optimize_graph(original, mode)
    assert original.SerializeToString() == source
    assert report["mode"] == mode
    assert len(report["folded_spatial_pads"]) == 1
    assert len(report["compacted_temporal_convolutions"]) == int(mode == "compact")
    conv = next(node for node in optimized.graph.node if node.op_type == "Conv")
    dilation = next(helper.get_attribute_value(value) for value in conv.attribute if value.name == "dilations")
    assert dilation == ([5, 1] if mode == "compact" else [8, 1])
    if mode == "compact":
        compact = report["compacted_temporal_convolutions"][0]
        assert compact["selected_frames"] == list(range(5)) + list(range(8, 13)) + list(range(16, 21))
        assert compact["original_history_frames"] == 21 and compact["compact_history_frames"] == 15
    else:
        assert not any(node.op_type == "Gather" for node in optimized.graph.node)
    runners = [session(model) for model in (original, optimized)]
    rng = np.random.default_rng(508)
    caches = [rng.normal(size=(1, 2, 16, 7)).astype(np.float32)]
    caches.append(caches[0].copy())
    for index in range(8):
        current = rng.normal(size=(1, 2, 5, 7)).astype(np.float32)
        values = [runner.run(None, {"cache": cache, "current": current})
                  for runner, cache in zip(runners, caches)]
        np.testing.assert_allclose(values[1][0], values[0][0], rtol=1e-6, atol=2e-6,
                                   err_msg=f"five-frame recurrent call {index}")
        np.testing.assert_array_equal(values[1][1], values[0][1])
        caches = [value[1] for value in values]


@pytest.mark.parametrize("dilation", [1, 2, 4])
def test_five_frame_overlapping_windows_do_not_add_gather(dilation):
    original = convolution_model(dilation, current_frames=5)
    optimized, report = optimize_graph(original)
    assert len(report["folded_spatial_pads"]) == 1
    assert not report["compacted_temporal_convolutions"]
    assert not any(node.op_type == "Gather" for node in optimized.graph.node)


@pytest.mark.parametrize("mode", ["pads", "compact"])
@pytest.mark.parametrize("kwargs", [{"pad_value": 1.}, {"share_pad": True}])
def test_five_frame_unsafe_padding_remains_unchanged(mode, kwargs):
    original = convolution_model(8, current_frames=5, **kwargs)
    optimized, report = optimize_graph(original, mode)
    assert not report["folded_spatial_pads"]
    assert not report["compacted_temporal_convolutions"]
    assert optimized.SerializeToString() == original.SerializeToString()


def test_invalid_optimization_mode_is_rejected_before_mutating_model():
    source = convolution_model(8, current_frames=5)
    before = source.SerializeToString()
    with pytest.raises(ValueError, match="mode"):
        optimize_graph(source, "unknown")
    assert source.SerializeToString() == before


def test_metadata_only_shape_expression_does_not_hide_five_frame_compaction():
    original = convolution_model(8, current_frames=5)
    original.graph.initializer.append(numpy_helper.from_array(np.zeros(4, np.int64), "shape_zero"))
    original.graph.node.insert(0, helper.make_node("Shape", ["current"], ["current_shape"], name="shape"))
    original.graph.node.insert(1, helper.make_node("Add", ["current_shape", "shape_zero"],
                                                  ["derived_shape"], name="shape_arithmetic"))
    original.graph.node.insert(2, helper.make_node("Reshape", ["current", "derived_shape"],
                                                  ["current_view"], name="reshape"))
    next(node for node in original.graph.node if node.name == "history").input[1] = "current_view"
    frozen = original.SerializeToString()
    optimized, report = optimize_graph(original)
    assert original.SerializeToString() == frozen
    assert len(report["compacted_temporal_convolutions"]) == 1
    # Shape evaluation belongs only to the proof copy; ordinary deployment
    # operators are not replaced outside the requested padding/tap rewrite.
    assert any(node.name == "shape" and node.op_type == "Shape" for node in optimized.graph.node)
    rng = np.random.default_rng(1005)
    inputs = {"cache": rng.normal(size=(1, 2, 16, 7)).astype(np.float32),
              "current": rng.normal(size=(1, 2, 5, 7)).astype(np.float32)}
    for actual, expected in zip(session(optimized).run(None, inputs), session(original).run(None, inputs)):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=2e-6)


def test_registered_sequence_shape_contract_is_used_only_for_analysis():
    shape = [7, 5, 2]
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, dims) for name, dims in (
        ("u", shape), ("delta", shape), ("B", [7, 5, 4]), ("C", [7, 5, 4]),
        ("A", [2, 4]), ("state", [7, 2, 4]))]
    nodes = [helper.make_node("SelectiveSequence", ["u", "delta", "B", "C", "A", "state"],
                              ["observed", "next_state"], domain="com.reuse")]
    graph = helper.make_graph(nodes, "custom_shape", inputs,
                              [helper.make_empty_tensor_value_info("observed"),
                               helper.make_empty_tensor_value_info("next_state")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17),
                                                   helper.make_opsetid("com.reuse", 1)])
    frozen = model.SerializeToString()
    known = static_shapes(model)
    assert known["observed"] == shape and known["next_state"] == [7, 2, 4]
    assert model.SerializeToString() == frozen
