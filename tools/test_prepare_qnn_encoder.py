# SPDX-License-Identifier: MIT
"""Independent host contracts for the selective encoder experiment."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import prepare_qnn_encoder as encoder


def toy_source():
    """A small stateful graph with the same six-node channel64 region contract."""
    rng = np.random.default_rng(123)
    params = {}
    nodes = [helper.make_node("Add", ["feature", "state_000"], [encoder.REGION_INPUT], name="stateful_input")]
    previous = encoder.REGION_INPUT
    for stage in range(2):
        inputs = 2 if stage == 0 else 64
        params[f"w{stage}"] = (rng.normal(size=(64, inputs, 1, 1)) * .15).astype(np.float32)
        params[f"b{stage}"] = (rng.normal(size=64) * .1).astype(np.float32)
        params[f"gamma{stage}"] = rng.uniform(.5, 1.5, 64).astype(np.float32)
        params[f"beta{stage}"] = rng.uniform(-.3, .3, 64).astype(np.float32)
        params[f"slope{stage}"] = np.linspace(-.5, 1.2, 64, dtype=np.float32).reshape(64, 1, 1)
        conv, transpose, norm = encoder.REGION_NAMES[stage * 3:stage * 3 + 3]
        out = encoder.REGION_OUTPUT if stage == 1 else "source_stage0_output"
        nodes.extend([
            helper.make_node("Conv", [previous, f"w{stage}", f"b{stage}"], [f"source_conv{stage}"],
                             name=conv, kernel_shape=[1, 1], strides=[1, 1], pads=[0, 0, 0, 0]),
            helper.make_node("Transpose", [f"source_conv{stage}"], [f"source_nhwc{stage}"],
                             name=transpose, perm=[0, 2, 3, 1]),
            helper.make_node("ChannelNormPRelu", [f"source_nhwc{stage}", f"gamma{stage}", f"beta{stage}", f"slope{stage}"],
                             [out], name=norm, domain="com.reuse", epsilon=1e-5),
        ])
        previous = out
    nodes.extend([
        helper.make_node("ReduceMean", [encoder.REGION_OUTPUT], ["output_mean"], name="update_mean", keepdims=0),
        helper.make_node("Add", ["state_000", "output_mean"], ["next_state_000"], name="update_state0"),
    ])
    nodes.extend(helper.make_node("Identity", [f"state_{i:03d}"], [f"next_state_{i:03d}"], name=f"retain_state{i}")
                 for i in range(1, 21))
    graph = helper.make_graph(nodes, "small_stateful_encoder_contract",
                              [helper.make_tensor_value_info("feature", TensorProto.FLOAT, [1, 2, 3, 5]),
                               *[helper.make_tensor_value_info(f"state_{i:03d}", TensorProto.FLOAT,
                                                               [1, 2, 3, 5] if i == 0 else [1]) for i in range(21)]],
                              [helper.make_tensor_value_info(encoder.REGION_OUTPUT, TensorProto.FLOAT, [1, 64, 3, 5]),
                               *[helper.make_tensor_value_info(f"next_state_{i:03d}", TensorProto.FLOAT,
                                                               [1, 2, 3, 5] if i == 0 else [1]) for i in range(21)]],
                              initializer=[numpy_helper.from_array(value, name) for name, value in params.items()])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.reuse", 1)], ir_version=10), params


def numpy_region(value, params):
    current = np.asarray(value, np.float64)
    for stage in range(2):
        current = np.einsum("nchw,oc->nohw", current, params[f"w{stage}"][:, :, 0, 0].astype(np.float64))
        current += params[f"b{stage}"][None, :, None, None]
        nhwc = current.transpose(0, 2, 3, 1)
        mean = nhwc.mean(axis=-1, keepdims=True)
        variance = ((nhwc - mean) ** 2).mean(axis=-1, keepdims=True)
        normalized = (nhwc - mean) / np.sqrt(variance + 1e-5)
        normalized = normalized * params[f"gamma{stage}"] + params[f"beta{stage}"]
        signed_slopes = params[f"slope{stage}"].reshape(64)
        nhwc = np.where(normalized < 0, normalized * signed_slopes, normalized)
        current = nhwc.transpose(0, 3, 1, 2)
    return current.astype(np.float32)


class EncoderPreparationTest(unittest.TestCase):
    def test_unfused_norm_and_signed_slope_math_matches_independent_numpy(self):
        source, params = toy_source()
        region = encoder.portable_region(source, [1, 2, 3, 5])
        value = np.linspace(-2, 3, 30, dtype=np.float32).reshape(1, 2, 3, 5)
        actual = encoder.session(region, threads=1, optimize=False).run(["output"], {"input": value})[0]
        np.testing.assert_allclose(actual, numpy_region(value, params), atol=2e-6, rtol=2e-5)
        self.assertEqual([n.op_type for n in region.graph.node].count("LayerNormalization"), 2)
        self.assertTrue(all(not n.domain for n in region.graph.node))
        slopes = [numpy_helper.to_array(x) for x in region.graph.initializer if "slope" in x.name]
        self.assertTrue(all(x.shape == (64,) and x.min() < 0 and x.max() > 0 for x in slopes))

    def test_fp32_graft_preserves_public_abi_and_chronological_state_math(self):
        source, params = toy_source()
        region = encoder.portable_region(source, [1, 2, 3, 5])
        graft = encoder.graft_region(source, region)
        self.assertEqual(list(graft.graph.input), list(source.graph.input))
        self.assertEqual(list(graft.graph.output), list(source.graph.output))
        information = {x.name: x for x in graft.graph.value_info}
        self.assertEqual([x.dim_value for x in information[encoder.REGION_INPUT].type.tensor_type.shape.dim], [1, 2, 3, 5])
        self.assertTrue(any(name.startswith("/qnn_encoder_tail/") for name in information))
        runtime = encoder.session(graft, threads=1)
        states = [np.zeros(x.shape, np.float32) for x in runtime.get_inputs()[1:]]
        expected_state = np.zeros((1, 2, 3, 5), np.float32)
        for call in range(5):
            value = (np.linspace(-1, 1, 30).reshape(1, 2, 3, 5) * (call + 1)).astype(np.float32)
            expected = numpy_region(value + expected_state, params)
            result = runtime.run(None, dict(zip([x.name for x in runtime.get_inputs()], [value, *states])))
            np.testing.assert_allclose(result[0], expected, atol=1e-5, rtol=1e-4)
            expected_state += expected.mean(dtype=np.float32)
            np.testing.assert_allclose(result[1], expected_state, atol=1e-5, rtol=1e-4)
            states = result[1:]
        self.assertEqual(len(states), 21)
        self.assertFalse(any(n.domain == "com.reuse" for n in graft.graph.node))

    def test_qdq_norm_bias_and_signed_slopes_follow_integer_contract(self):
        source, _ = toy_source()
        region = encoder.portable_region(source, [1, 2, 3, 5])
        ranges = {name: [-8, 8] for n in region.graph.node for name in n.output}
        ranges["input"] = [-3, 3]
        quantized, params = encoder.quantized_region(region, ranges)
        onnx.checker.check_model(quantized, full_check=True)
        tensors = {x.name: numpy_helper.to_array(x) for x in quantized.graph.initializer}
        producers = {name: n for n in quantized.graph.node for name in n.output}
        for index in (2, 7):
            norm = next(n for n in quantized.graph.node if n.name == f"encoder_q_node_{index}")
            self.assertEqual(norm.op_type, "LayerNormalization")
            self.assertEqual(encoder.attributes(norm)["axis"], -1)
            self.assertEqual(len(norm.output), 1)
            gamma_dq, bias_dq = producers[norm.input[1]], producers[norm.input[2]]
            self.assertEqual(tensors[gamma_dq.input[0]].dtype, np.dtype("uint8"))
            self.assertEqual(tensors[bias_dq.input[0]].dtype, np.dtype("int32"))
            self.assertEqual(int(tensors[bias_dq.input[2]]), 0)
            a_dq = producers[norm.input[0]]
            expected_scale = np.float32(tensors[a_dq.input[1]] * tensors[gamma_dq.input[1]])
            self.assertEqual(float(tensors[bias_dq.input[1]]), float(expected_scale))
        self.assertGreater(params["slope0_nhwc"]["zero_point"], 0)
        value = np.linspace(-1, 1, 30, dtype=np.float32).reshape(1, 2, 3, 5)
        result = encoder.session(quantized, threads=1).run(["output"], {"input": value})[0]
        self.assertEqual(result.shape, (1, 64, 3, 5))
        self.assertTrue(np.isfinite(result).all())

    def test_u16_activations_use_contrib_domain_static_shapes_and_preserve_full_opset17(self):
        source, _ = toy_source()
        region = encoder.portable_region(source, [1, 2, 3, 5])
        ranges = {name: [-8, 8] for n in region.graph.node for name in n.output}
        ranges["input"] = [-3, 3]
        quantized, params = encoder.quantized_region(region, ranges, "uint16")
        self.assertEqual({x.domain: x.version for x in quantized.opset_import}, {"": 17, "com.microsoft": 1})
        tensors = {x.name: numpy_helper.to_array(x) for x in quantized.graph.initializer}
        infos = {x.name: x for x in quantized.graph.value_info}
        q_nodes = [node for node in quantized.graph.node if node.op_type == "QuantizeLinear"]
        self.assertEqual(len(q_nodes), 11)
        for node in q_nodes:
            self.assertEqual(node.domain, "com.microsoft")
            self.assertEqual(tensors[node.input[2]].dtype, np.dtype("uint16"))
            self.assertEqual(infos[node.output[0]].type.tensor_type.elem_type, TensorProto.UINT16)
            self.assertTrue(all(d.dim_value > 0 for d in infos[node.output[0]].type.tensor_type.shape.dim))
        self.assertEqual(params["input"]["dtype"], "uint16")
        self.assertEqual(params["w0"]["dtype"], "uint8")
        self.assertEqual(params["gamma0"]["dtype"], "uint8")
        self.assertEqual(params["slope0_nhwc"]["dtype"], "uint8")
        graft = encoder.graft_region(source, quantized)
        self.assertEqual({x.domain: x.version for x in graft.opset_import}, {"": 17, "com.reuse": 1, "com.microsoft": 1})
        self.assertEqual(list(graft.graph.input), list(source.graph.input))
        self.assertEqual(list(graft.graph.output), list(source.graph.output))
        value = np.linspace(-1, 1, 30, dtype=np.float32).reshape(1, 2, 3, 5)
        runtime = encoder.session(graft, threads=1)
        feed = {"feature": value, **{x.name: np.zeros(x.shape, np.float32) for x in runtime.get_inputs()[1:]}}
        values = runtime.run(None, feed)
        self.assertEqual(len(values), 22)
        self.assertTrue(all(np.isfinite(x).all() for x in values))

    def test_u16_parameter_guards_and_cpu_roundtrip(self):
        q = encoder.activation_parameters(-4, 5, "uint16")
        self.assertGreater(q["zero_point"], 255)
        self.assertAlmostEqual(q["scale"], 9 / 65535, delta=1e-10)
        constants = [numpy_helper.from_array(np.array(q["scale"], np.float32), "scale"),
                     numpy_helper.from_array(np.array(q["zero_point"], np.uint16), "zero")]
        graph = helper.make_graph([
            helper.make_node("QuantizeLinear", ["input", "scale", "zero"], ["integer"], domain="com.microsoft"),
            helper.make_node("DequantizeLinear", ["integer", "scale", "zero"], ["output"], domain="com.microsoft")],
            "u16_roundtrip", [helper.make_tensor_value_info("input", TensorProto.FLOAT, [6])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [6])], initializer=constants)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)], ir_version=10)
        value = np.array([-4, -2.25, -.01, 0, 1.125, 5], np.float32)
        actual = encoder.session(model, threads=1).run(["output"], {"input": value})[0]
        np.testing.assert_allclose(actual, value, atol=q["scale"] * 1.01, rtol=0)
        for a, b, kind in ((np.nan, 1, "uint16"), (1, -1, "uint16"), (-1, 1, "int16")):
            with self.assertRaises(ValueError): encoder.activation_parameters(a, b, kind)

    def test_u8_option_remains_available_without_contrib_domain(self):
        source, _ = toy_source()
        region = encoder.portable_region(source, [1, 2, 3, 5])
        ranges = {name: [-8, 8] for n in region.graph.node for name in n.output}
        ranges["input"] = [-3, 3]
        quantized, params = encoder.quantized_region(region, ranges, "uint8")
        self.assertEqual({x.domain: x.version for x in quantized.opset_import}, {"": 17})
        self.assertEqual(params["input"]["dtype"], "uint8")
        self.assertTrue(all(not node.domain for node in quantized.graph.node))
        with self.assertRaises(ValueError): encoder.quantized_region(region, ranges, "float16")

    def test_graft_cannot_silently_change_existing_operator_semantics(self):
        source, _ = toy_source()
        region = encoder.portable_region(source, [1, 2, 3, 5])
        region.opset_import[0].version = 21
        with self.assertRaisesRegex(ValueError, "operator-set version"):
            encoder.graft_region(source, region)

    def test_graft_rejects_noncontiguous_replacement(self):
        source, _ = toy_source()
        region = encoder.portable_region(source, [1, 2, 3, 5])
        source.graph.node.insert(3, helper.make_node("Identity", ["feature"], ["interloper"], name="unrelated"))
        with self.assertRaisesRegex(ValueError, "contiguous"):
            encoder.graft_region(source, region)

    def test_quantization_metrics_have_known_reference(self):
        actual = encoder.metric(np.array([1, 2, -1]), np.array([1, 3, -5]))
        self.assertEqual(actual["max_abs"], 4)
        self.assertAlmostEqual(actual["rmse"], np.sqrt(17 / 3))
        self.assertAlmostEqual(actual["normalized_rmse"], np.sqrt(17 / 6))

    def test_stream_comparison_uses_each_models_own_states_and_trims_flush(self):
        audio = np.sin(np.arange(2240, dtype=np.float32) / 31) * .1
        features = encoder.feature_calls(audio)
        self.assertEqual(len(features), 3)

        class FakeSession:
            def __init__(self, gain): self.gain, self.calls = gain, 0
            def get_inputs(self):
                return [SimpleNamespace(name="magnitude", shape=[5, 161], type="tensor(float)"),
                        SimpleNamespace(name="phase", shape=[5, 161], type="tensor(float)"),
                        *[SimpleNamespace(name=f"state_{i:03d}", shape=[1], type="tensor(float)") for i in range(21)]]
            def run(self, _, feed):
                for i in range(21):
                    np.testing.assert_array_equal(feed[f"state_{i:03d}"], np.array([self.calls * self.gain], np.float32))
                amplitude = np.expm1(feed["magnitude"]) * (1 + .01 * self.calls * self.gain)
                real, imag = amplitude * np.cos(feed["phase"]), amplitude * np.sin(feed["phase"])
                self.calls += 1
                return [real, imag, *[np.array([self.calls * self.gain], np.float32) for _ in range(21)]]

        with patch.object(encoder, "session", side_effect=lambda model, *a, **k: FakeSession(2 if model == "candidate" else 1)):
            report, waveforms = encoder.stream_comparison("baseline", "candidate", features, len(audio), None)
        self.assertEqual([len(x) for x in waveforms], [2240, 2240])
        self.assertEqual(report["state_max_abs"], 3)
        self.assertTrue(report["independent_state_feedback"])
        self.assertTrue(report["all_21_states_checked"])
        self.assertGreater(report["waveform"]["rmse"], 0)
        self.assertFalse(report["audio_quality_verified"])

    def test_untrusted_source_hash_fails_before_writing_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = root / "wrong.onnx"
            graph.write_bytes(b"wrong graph")
            with self.assertRaisesRegex(ValueError, "source graph SHA-256 mismatch"):
                encoder.prepare(graph, root / "missing.wav", root / "missing.dll", root / "assets")
            self.assertFalse((root / "assets").exists())

    def test_untrusted_public_audio_hash_fails_before_native_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph, clip = root / "graph.onnx", root / "wrong.wav"
            graph.write_bytes(b"test graph"); clip.write_bytes(b"not the pinned public sample")
            with patch.object(encoder, "MODEL_SHA256", hashlib.sha256(graph.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(ValueError, "public NVIDIA sample SHA-256 mismatch"):
                    encoder.prepare(graph, clip, root / "missing.dll", root / "assets")
            self.assertFalse((root / "assets").exists())


if __name__ == "__main__": unittest.main()
