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
    @staticmethod
    def manual_conv(value, weights, bias, attributes):
        """Straight coordinate MAC; no ONNX or generator shape helper."""
        sy, sx = attributes.get("strides", [1, 1])
        dy, dx = attributes.get("dilations", [1, 1])
        top, left, bottom, right = attributes.get("pads", [0, 0, 0, 0])
        n, channels, height, width = value.shape
        outputs, _, ky, kx = weights.shape
        oh = (height + top + bottom - (ky - 1) * dy - 1) // sy + 1
        ow = (width + left + right - (kx - 1) * dx - 1) // sx + 1
        result = np.zeros((n, outputs, oh, ow), dtype=np.float64)
        for batch in range(n):
            for out in range(outputs):
                for y in range(oh):
                    for x in range(ow):
                        total = float(bias[out])
                        for channel in range(channels):
                            for j in range(ky):
                                iy = y * sy + j * dy - top
                                for k in range(kx):
                                    ix = x * sx + k * dx - left
                                    if 0 <= iy < height and 0 <= ix < width:
                                        total += float(value[batch, channel, iy, ix]) * float(weights[out, channel, j, k])
                        result[batch, out, y, x] = total
        return result

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
            for name in (first["model"], first["input"]["file"], first["output"]["file"],
                         first["cpu_fp32"]["model"], first["cpu_fp32"]["output"]["file"]):
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
            fp32 = first["cpu_fp32"]
            baseline = onnx.load(a / fp32["model"])
            original = np.fromfile(a / fp32["output"]["file"], dtype="<f4").reshape(16, 16)
            np.testing.assert_array_equal(original, probe.cpu_reference(baseline, inputs))
            self.assertEqual([node.op_type for node in baseline.graph.node], ["MatMul"])
            self.assertEqual(fp32["max_abs_error"], 1e-4)
            self.assertEqual(fp32["relative_tolerance"], 1e-4)

    def test_quantized_conv_matches_independent_integer_mac_with_stride_dilation_and_padding(self):
        for attrs, shape, kernel in (
                ({"strides": [1, 2], "dilations": [1, 1], "pads": [0, 0, 0, 0]}, (1, 2, 3, 6), (2, 2, 1, 3)),
                ({"strides": [1, 1], "dilations": [2, 1], "pads": [0, 1, 0, 1]}, (1, 2, 7, 5), (2, 2, 3, 3))):
            with self.subTest(attributes=attrs):
                value = (((np.arange(np.prod(shape)) * 17 + 3) % 61 - 30) / 32).astype(np.float32).reshape(shape)
                weights = (((np.arange(np.prod(kernel)) * 7 + 2) % 29 - 14) / 32).astype(np.float32).reshape(kernel)
                bias = np.array([.234, -.178], np.float32)
                model, parameters = probe.qdq_conv(shape, weights, bias, attrs, (-1, 1), (-5, 5))
                tensors = {x.name: numpy_helper.to_array(x) for x in model.graph.initializer}
                self.assertEqual(tensors["bias_i32"].dtype, np.dtype("int32"))
                self.assertEqual(tensors["bias_zero"].dtype, np.dtype("int32"))
                self.assertEqual(int(tensors["bias_zero"]), 0)
                self.assertEqual(float(tensors["bias_scale"]),
                                 float(np.float32(tensors["activation_scale"] * tensors["weight_scale"])))
                a_scale, a_zero = float(tensors["activation_scale"]), int(tensors["activation_zero"])
                w_scale, w_zero = float(tensors["weight_scale"]), int(tensors["weight_zero"])
                y_scale, y_zero = float(tensors["output_scale"]), int(tensors["output_zero"])
                a_center = np.clip(np.rint(value / np.float32(a_scale)) + a_zero, 0, 255).astype(np.int64) - a_zero
                w_center = tensors["weight_u8"].astype(np.int64) - w_zero
                # Bias enters the same int32 accumulator scale as the integer convolution.
                mac = self.manual_conv(a_center, w_center, tensors["bias_i32"], attrs)
                codes = np.clip(np.rint(mac * a_scale * w_scale / y_scale) + y_zero, 0, 255)
                expected = (codes - y_zero) * y_scale
                actual = probe.cpu_reference(model, value)
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_allclose(actual, expected, atol=y_scale * 1.01, rtol=0)
                self.assertGreater(float(np.ptp(actual)), 0.5)
                self.assertEqual(parameters["bias"]["zero_point"], 0)

    def test_fp32_conv_baseline_preserves_original_weights_bias_and_source_math(self):
        value = (np.arange(40, dtype=np.float32).reshape(1, 2, 4, 5) - 16) / 16
        weights = (np.arange(24, dtype=np.float32).reshape(2, 2, 2, 3) - 11) / 32
        bias = np.array([.125, -.25], np.float32)
        attrs = {"strides": [1, 2], "dilations": [2, 1], "pads": [0, 1, 0, 1]}
        original = self.manual_conv(value, weights, bias, attrs).astype(np.float32)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            metadata = probe.write_case(root, "conv_test", "Conv fixture", value, weights, (-2, 2), (-6, 6),
                                        {"kind": "test"}, original, operator="Conv", bias=bias, attributes=attrs)
            fp32 = metadata["cpu_fp32"]
            graph = onnx.load(root / fp32["model"])
            tensors = {x.name: numpy_helper.to_array(x) for x in graph.graph.initializer}
            np.testing.assert_array_equal(tensors["weight"], weights)
            np.testing.assert_array_equal(tensors["bias"], bias)
            saved = np.fromfile(root / fp32["output"]["file"], dtype="<f4").reshape(original.shape)
            np.testing.assert_allclose(saved, original, atol=1e-6, rtol=1e-6)
            self.assertEqual(hashlib.sha256((root / fp32["model"]).read_bytes()).hexdigest(), fp32["model_sha256"])

    def test_bias_overflow_is_rejected_instead_of_silently_clamped(self):
        with self.assertRaisesRegex(ValueError, "bias overflows int32"):
            probe.qdq_conv((1, 1, 1, 1), np.array([[[[.001]]]], np.float32),
                           np.array([1e20], np.float32), {}, (-.001, .001), (-1, 1))

    def test_output_clipping_is_reported_without_relaxing_tolerance(self):
        model, parameters = probe.qdq_matmul((2, 2), np.ones((2, 2), np.float32), (-1, 1), (-.01, .01))
        saturation = probe.saturation_report(model, [np.ones((2, 2), np.float32)], parameters, "MatMul")
        self.assertTrue(saturation["significant"])
        self.assertEqual(saturation["output"]["clipped_fraction"], 1.0)
        self.assertEqual(saturation["activation"]["clipped_fraction"], 0)
        self.assertFalse(saturation["quality_claim"])

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
