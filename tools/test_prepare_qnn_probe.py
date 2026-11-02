# SPDX-License-Identifier: MIT
"""Host contracts for the portable QNN smoke assets; no hardware claim."""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import onnx
from onnx import numpy_helper

import prepare_qnn_probe as probe


class QnnProbeAssetsTest(unittest.TestCase):
    def test_static_qdq_graph_has_integer_weights_and_no_custom_operator(self):
        value = np.arange(12, dtype=np.float32).reshape(3, 4) / 8 - 0.5
        weights = np.arange(20, dtype=np.float32).reshape(4, 5) / 16 - 0.75
        model, _ = probe.qdq_matmul(value.shape, weights, (-1, 2), (-4, 4))
        onnx.checker.check_model(model, full_check=True)
        self.assertEqual([(x.domain, x.version) for x in model.opset_import], [("", 13)])
        self.assertTrue(all(not node.domain for node in model.graph.node))
        self.assertEqual([x.dim_value for x in model.graph.input[0].type.tensor_type.shape.dim], [3, 4])
        self.assertEqual([x.dim_value for x in model.graph.output[0].type.tensor_type.shape.dim], [3, 5])
        initializers = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
        self.assertEqual(initializers["weight_u8"].dtype, np.dtype("uint8"))
        self.assertFalse(any(x.shape == weights.shape and x.dtype == np.float32 for x in initializers.values()))
        matmul = next(x for x in model.graph.node if x.op_type == "MatMul")
        producers = {name: n for n in model.graph.node for name in n.output}
        self.assertTrue(all(producers[x].op_type == "DequantizeLinear" for x in matmul.input))

    def test_cpu_qdq_reference_agrees_with_independent_integer_accumulation(self):
        value = np.array([[-1.0, .25, .8], [.125, -.7, 1.2]], np.float32)
        weights = np.array([[-.4, .6], [.9, -.2], [.3, -.8]], np.float32)
        model, _ = probe.qdq_matmul(value.shape, weights, (-1.0, 1.2), (-2, 2))
        tensors = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
        # Compute the integer MAC independently, using only serialized constants.
        a_scale, a_zero = tensors["activation_scale"], int(tensors["activation_zero"])
        w_scale, w_zero = tensors["weight_scale"], int(tensors["weight_zero"])
        y_scale, y_zero = tensors["output_scale"], int(tensors["output_zero"])
        a_int = np.clip(np.rint(value / a_scale) + a_zero, 0, 255).astype(np.int64) - a_zero
        w_int = tensors["weight_u8"].astype(np.int64) - w_zero
        accumulators = a_int @ w_int
        quantized = np.clip(np.rint(accumulators * float(a_scale) * float(w_scale) / float(y_scale)) + y_zero, 0, 255)
        expected = (quantized - y_zero) * float(y_scale)
        actual = probe.cpu_reference(model, value)
        np.testing.assert_allclose(actual, expected, atol=float(y_scale) * 1.01, rtol=0)
        self.assertGreater(float(np.ptp(actual)), 0.5)

    def test_fixtures_are_little_endian_nonconstant_and_reproducible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            a, b = root / "a", root / "b"
            a.mkdir(); b.mkdir()
            first, second = probe.synthetic_case(a), probe.synthetic_case(b)
            self.assertEqual(first, second)
            for name in (first["model"], first["input"]["file"], first["output"]["file"]):
                self.assertEqual((a / name).read_bytes(), (b / name).read_bytes())
            for side in ("input", "output"):
                metadata = first[side]
                data = (a / metadata["file"]).read_bytes()
                self.assertEqual(hashlib.sha256(data).hexdigest(), metadata["sha256"])
                values = np.frombuffer(data, dtype="<f4").reshape(metadata["shape"])
                self.assertTrue(np.isfinite(values).all())
                self.assertGreater(float(np.ptp(values)), 0.5)
            graph = onnx.load(a / first["model"])
            inputs = np.fromfile(a / first["input"]["file"], dtype="<f4").reshape(16, 16)
            outputs = np.fromfile(a / first["output"]["file"], dtype="<f4").reshape(16, 16)
            np.testing.assert_array_equal(outputs, probe.cpu_reference(graph, inputs))
            self.assertLess(first["fp32_error"]["rmse"], 0.03)
            self.assertLess(first["max_abs_error"], 0.2)

    def test_wrong_source_graph_hash_rejects_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = root / "not-pinned.onnx"
            graph.write_bytes(b"not the verified source")
            output = root / "assets"
            with self.assertRaisesRegex(ValueError, "source graph SHA-256 mismatch"):
                probe.prepare(graph, root / "missing.wav", root / "missing.dll", output)
            self.assertFalse(output.exists())

    def test_wrong_public_sample_hash_rejects_before_loading_source_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph, sample = root / "source.onnx", root / "sample.wav"
            graph.write_bytes(b"test graph placeholder")
            sample.write_bytes(b"not the verified public audio")
            output = root / "assets"
            with patch.object(probe, "MODEL_SHA256", hashlib.sha256(graph.read_bytes()).hexdigest()):
                with self.assertRaisesRegex(ValueError, "public NVIDIA sample SHA-256 mismatch"):
                    probe.prepare(graph, sample, root / "missing.dll", output)
            self.assertFalse(output.exists())

    def test_nonfinite_or_reversed_ranges_rejected(self):
        for minimum, maximum in ((np.nan, 1), (-1, np.inf), (1, -1)):
            with self.subTest(minimum=minimum, maximum=maximum):
                with self.assertRaises(ValueError):
                    probe.quantization_parameters(minimum, maximum)


if __name__ == "__main__":
    unittest.main()
