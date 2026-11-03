# SPDX-License-Identifier: MIT
"""Contracts for a single-Conv stateful graft; no phone performance claim."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

import prepare_qnn_conv_hybrid as fallback
from prepare_qnn_probe import qdq_conv, standalone_fp32
from prepare_qnn_encoder import session


def toy_models():
    rng = np.random.default_rng(52)
    weights = rng.normal(0, .12, (3, 2, 3, 3)).astype(np.float32)
    bias = np.array([.125, -.25, .0625], np.float32)
    attributes = {"kernel_shape": [3, 3], "strides": [1, 1], "dilations": [2, 1], "pads": [0, 1, 0, 1], "group": 1}
    nodes = [helper.make_node("Add", ["feature", "state_000"], [fallback.REGION_INPUT], name="before_conv"),
             helper.make_node("Conv", [fallback.REGION_INPUT, "original_weight", "original_bias"], [fallback.REGION_OUTPUT],
                              name=fallback.CONV_NAME, **attributes),
             helper.make_node("Identity", [fallback.REGION_OUTPUT], ["real"], name="after_conv"),
             helper.make_node("Neg", [fallback.REGION_OUTPUT], ["imag"], name="other_output"),
             helper.make_node("ReduceMean", [fallback.REGION_OUTPUT], ["mean"], name="state_mean", keepdims=0),
             helper.make_node("Add", ["state_000", "mean"], ["next_state_000"], name="state_update")]
    nodes.extend(helper.make_node("Identity", [f"state_{i:03d}"], [f"next_state_{i:03d}"], name=f"preserved_state_{i}")
                 for i in range(1, 21))
    inputs = [helper.make_tensor_value_info("feature", TensorProto.FLOAT, [1, 2, 7, 5]),
              helper.make_tensor_value_info("phase", TensorProto.FLOAT, [1]),
              *[helper.make_tensor_value_info(f"state_{i:03d}", TensorProto.FLOAT, [1]) for i in range(21)]]
    outputs = [helper.make_tensor_value_info("real", TensorProto.FLOAT, [1, 3, 3, 5]),
               helper.make_tensor_value_info("imag", TensorProto.FLOAT, [1, 3, 3, 5]),
               *[helper.make_tensor_value_info(f"next_state_{i:03d}", TensorProto.FLOAT, [1]) for i in range(21)]]
    graph = helper.make_graph(nodes, "stateful_single_conv", inputs, outputs,
                              initializer=[numpy_helper.from_array(weights, "original_weight"),
                                           numpy_helper.from_array(bias, "original_bias")])
    source = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    fp32 = standalone_fp32("Conv", (1, 2, 7, 5), weights, bias, attributes)
    quantized, _ = qdq_conv((1, 2, 7, 5), weights, bias, attributes, (-5, 5), (-10, 10))
    return source, fp32, quantized


class SingleConvFallbackTest(unittest.TestCase):
    def test_graft_retains_all_other_nodes_original_weights_and_21_state_abi(self):
        source, fp32, quantized = toy_models()
        graft = fallback.graft_single_conv(source, quantized, fp32)
        originals = {n.name: n.SerializeToString() for n in source.graph.node if n.name != fallback.CONV_NAME}
        actual = {n.name: n.SerializeToString() for n in graft.graph.node if not n.name.startswith(fallback.PREFIX)}
        self.assertEqual(actual, originals)
        tensors = {x.name: x.SerializeToString() for x in graft.graph.initializer}
        for item in source.graph.initializer:
            self.assertEqual(tensors[item.name], item.SerializeToString())
        self.assertEqual(list(graft.graph.input), list(source.graph.input))
        self.assertEqual(list(graft.graph.output), list(source.graph.output))
        self.assertEqual(len(graft.graph.input), 23)
        self.assertEqual(len(graft.graph.output), 23)
        self.assertEqual(list(graft.opset_import), list(source.opset_import))
        info = {x.name: x for x in graft.graph.value_info}
        self.assertEqual([x.dim_value for x in info[fallback.REGION_INPUT].type.tensor_type.shape.dim], [1, 2, 7, 5])
        self.assertEqual([x.dim_value for x in info[fallback.REGION_OUTPUT].type.tensor_type.shape.dim], [1, 3, 3, 5])
        self.assertTrue(any(name.startswith(fallback.PREFIX) for name in info))

    def test_chronological_graft_matches_the_standalone_qdq_equations_with_independent_state(self):
        source, fp32, quantized = toy_models()
        graft = fallback.graft_single_conv(source, quantized, fp32)
        runtime, region = session(graft, threads=1), session(quantized, threads=1, optimize=False)
        states = [np.zeros(1, np.float32) for _ in range(21)]
        original_states = [np.zeros(1, np.float32) for _ in range(21)]
        original = session(source, threads=1)
        for call in range(6):
            value = (np.sin(np.arange(70) / 9).reshape(1, 2, 7, 5) * (call + 1) / 4).astype(np.float32)
            expected = region.run(["output"], {"input": value + states[0]})[0]
            result = runtime.run(None, dict(zip([x.name for x in runtime.get_inputs()], [value, np.zeros(1, np.float32), *states])))
            np.testing.assert_allclose(result[0], expected, atol=2e-6, rtol=1e-5)
            np.testing.assert_array_equal(result[1], -result[0])
            np.testing.assert_allclose(result[2], states[0] + result[0].mean(dtype=np.float32), atol=2e-6)
            states = result[2:]
            baseline = original.run(None, dict(zip([x.name for x in original.get_inputs()],
                                                   [value, np.zeros(1, np.float32), *original_states])))
            original_states = baseline[2:]
            self.assertTrue(all(np.isfinite(x).all() for x in result))
        self.assertFalse(np.array_equal(states[0], original_states[0]))

    def test_fp32_replacement_retains_unquantized_behavior(self):
        source, fp32, _ = toy_models()
        graft = fallback.graft_single_conv(source, fp32, fp32)
        value = np.linspace(-1, 1, 70, dtype=np.float32).reshape(1, 2, 7, 5)
        feed = {"feature": value, "phase": np.zeros(1, np.float32),
                **{f"state_{i:03d}": np.array([i / 10], np.float32) for i in range(21)}}
        for baseline, actual in zip(session(source, threads=1).run(None, feed), session(graft, threads=1).run(None, feed)):
            np.testing.assert_allclose(actual, baseline, atol=1e-6, rtol=1e-6)

    def test_wrong_original_parameters_and_namespace_collisions_rejected(self):
        source, fp32, quantized = toy_models()
        changed = onnx.ModelProto(); changed.CopyFrom(fp32)
        array = numpy_helper.to_array(changed.graph.initializer[0]).copy()
        array.flat[0] += 1
        changed.graph.initializer[0].CopyFrom(numpy_helper.from_array(array, changed.graph.initializer[0].name))
        with self.assertRaises(AssertionError):
            fallback.graft_single_conv(source, quantized, changed)
        source.graph.initializer.append(numpy_helper.from_array(np.array(1, np.float32), fallback.PREFIX + "activation_scale"))
        with self.assertRaisesRegex(ValueError, "namespace collides"):
            fallback.graft_single_conv(source, quantized, fp32)

    def test_untrusted_source_rejects_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "wrong.onnx"
            source.write_bytes(b"not the pinned F950 graph")
            output = root / "output"
            with self.assertRaisesRegex(ValueError, "source graph SHA-256 mismatch"):
                fallback.prepare(source, root / "missing.wav", root / "missing.dll", root / "fixtures", output)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
