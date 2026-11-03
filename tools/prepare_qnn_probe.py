# SPDX-License-Identifier: LicenseRef-NVIDIA-NSCLv1
# Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
# Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
"""Prepare small, fixed-shape QNN HTP smoke tests, not an audio-quality model.

The synthetic case has no NVIDIA weights. The model-derived cases extract the
first temporal input projection and two encoder convolutions from the pinned
Android RE-USE graph. Activation ranges come from the pinned, public NVIDIA sample, with all source
recurrent states advanced chronologically. One sample is not general-purpose
quantization calibration. These graphs do not enhance audio or establish
real-time performance.

The model-derived output remains subject to NVIDIA NSCLv1. See the copied
LICENSE-NVIDIA.txt and source_provenance in manifest.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "f950dcfb9321191b2bcb3ab9afa3e100c40e5c64dd1de1d30849be4c1aacce32"
MODEL_DIRECTORY = ROOT / "models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal-channelnorm-frequency"
PROJECTION_NODE = "/in_proj/MatMul"
PROJECTION_INPUT = "/Reshape_2_output_0"
PROJECTION_WEIGHT = "onnx::MatMul_1860"
PROJECTION_OUTPUT = "/in_proj/MatMul_output_0"
SOURCE_CASES = [
    {"id": "reuse_in_proj_u8", "title": "RE-USE input projection (405 x 64 x 512, uint8)",
     "node": PROJECTION_NODE, "operator": "MatMul", "input": PROJECTION_INPUT,
     "weight": PROJECTION_WEIGHT, "output": PROJECTION_OUTPUT,
     "input_shape": [81, 5, 64], "weight_shape": [64, 512], "output_shape": [81, 5, 512]},
    {"id": "reuse_encoder_stride2_conv_u8", "title": "RE-USE encoder stride-2 convolution (uint8)",
     "node": "/dense_encoder/dense_conv_2/dense_conv_2.0/Conv", "operator": "Conv",
     "input": "/dense_encoder/dense_block/dense_block.3.3/PRelu_output_0",
     "weight": "dense_encoder.dense_conv_2.0.weight", "bias": "dense_encoder.dense_conv_2.0.bias",
     "output": "/dense_encoder/dense_conv_2/dense_conv_2.0/Conv_output_0",
     "input_shape": [1, 64, 5, 163], "weight_shape": [64, 64, 1, 3], "output_shape": [1, 64, 5, 81],
     "attributes": {"dilations": [1, 1], "group": 1, "kernel_shape": [1, 3],
                    "pads": [0, 0, 0, 0], "strides": [1, 2]}},
    {"id": "reuse_encoder_dilated_conv_u8", "title": "RE-USE encoder dilated convolution (uint8)",
     "node": "/dense_encoder/dense_block/dense_block.3.1/Conv", "operator": "Conv",
     "input": "/dense_encoder/dense_block/Concat_9_output_0",
     "weight": "dense_encoder.dense_block.dense_block.3.1.weight",
     "bias": "dense_encoder.dense_block.dense_block.3.1.bias",
     "output": "/dense_encoder/dense_block/dense_block.3.1/Conv_output_0",
     "input_shape": [1, 256, 21, 163], "weight_shape": [64, 256, 3, 3], "output_shape": [1, 64, 5, 163],
     "attributes": {"dilations": [8, 1], "group": 1, "kernel_shape": [3, 3],
                    "pads": [0, 1, 0, 1], "strides": [1, 1]}},
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_hash(path: Path, expected: str, description: str) -> None:
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Pinned {description} SHA-256 mismatch: {path}")


def quantization_parameters(minimum: float, maximum: float) -> dict:
    """Per-tensor asymmetric uint8 quantization, with exactly representable zero."""
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
        raise ValueError("Quantization range must be finite and ordered")
    minimum, maximum = min(float(minimum), 0.0), max(float(maximum), 0.0)
    scale = np.float32((maximum - minimum) / 255.0) if maximum > minimum else np.float32(1.0)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Quantization scale must be finite and positive")
    zero_point = int(np.clip(np.rint(-minimum / float(scale)), 0, 255))
    return {"dtype": "uint8", "scale": float(scale), "zero_point": zero_point,
            "observed_min": minimum, "observed_max": maximum, "per_tensor": True}


def quantize_uint8(value: np.ndarray, parameters: dict) -> np.ndarray:
    scaled = np.asarray(value, np.float32) / np.float32(parameters["scale"])
    return np.clip(np.rint(scaled) + parameters["zero_point"], 0, 255).astype(np.uint8)


def qdq_matmul(input_shape: tuple[int, int], weights: np.ndarray,
               activation_range: tuple[float, float],
               output_range: tuple[float, float]) -> tuple[onnx.ModelProto, dict]:
    """A portable float-I/O QDQ graph whose MatMul has uint8/uint8 boundaries."""
    weights = np.asarray(weights, np.float32)
    if (len(input_shape) != 2 or weights.ndim != 2 or input_shape[1] != weights.shape[0]
            or min(*input_shape, *weights.shape) <= 0 or not np.isfinite(weights).all()):
        raise ValueError("Expected finite compatible two-dimensional MatMul operands")
    parameters = {
        "activation": quantization_parameters(*activation_range),
        "weight": quantization_parameters(float(weights.min()), float(weights.max())),
        "output": quantization_parameters(*output_range),
    }
    initializers = []
    for prefix, values in parameters.items():
        initializers.extend([
            numpy_helper.from_array(np.array(values["scale"], np.float32), prefix + "_scale"),
            numpy_helper.from_array(np.array(values["zero_point"], np.uint8), prefix + "_zero"),
        ])
    initializers.append(numpy_helper.from_array(quantize_uint8(weights, parameters["weight"]), "weight_u8"))
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "activation_scale", "activation_zero"],
                         ["input_u8"], name="input_quantize"),
        helper.make_node("DequantizeLinear", ["input_u8", "activation_scale", "activation_zero"],
                         ["input_dequantized"], name="input_dequantize"),
        helper.make_node("DequantizeLinear", ["weight_u8", "weight_scale", "weight_zero"],
                         ["weight_dequantized"], name="weight_dequantize"),
        helper.make_node("MatMul", ["input_dequantized", "weight_dequantized"],
                         ["projection"], name="probe_matmul"),
        helper.make_node("QuantizeLinear", ["projection", "output_scale", "output_zero"],
                         ["output_u8"], name="output_quantize"),
        helper.make_node("DequantizeLinear", ["output_u8", "output_scale", "output_zero"],
                         ["output"], name="output_dequantize"),
    ]
    output_shape = (input_shape[0], weights.shape[1])
    graph = helper.make_graph(nodes, "qnn_htp_matmul_smoke",
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
                              initializer=initializers)
    model = helper.make_model(graph, producer_name="RE-USE unofficial QNN smoke generator",
                              opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
    model.doc_string = "Hardware smoke test only; not an audio enhancement model or quality-calibrated export."
    onnx.checker.check_model(model, full_check=True)
    return model, parameters


def conv_output_shape(input_shape: tuple, weights: np.ndarray, attributes: dict) -> tuple:
    if (len(input_shape) != 4 or weights.ndim != 4 or attributes.get("group", 1) != 1
            or input_shape[1] != weights.shape[1] or min(*input_shape, *weights.shape) <= 0
            or not np.isfinite(weights).all()):
        raise ValueError("Expected finite compatible NCHW, group-1 Conv operands")
    strides = attributes.get("strides", [1, 1])
    dilations = attributes.get("dilations", [1, 1])
    pads = attributes.get("pads", [0, 0, 0, 0])
    if (len(strides) != 2 or len(dilations) != 2 or len(pads) != 4
            or min(*strides, *dilations) <= 0 or min(pads) < 0):
        raise ValueError("Invalid Conv strides, dilations, or pads")
    spatial = [((input_shape[i + 2] + pads[i] + pads[i + 2]
                 - dilations[i] * (weights.shape[i + 2] - 1) - 1) // strides[i]) + 1
               for i in range(2)]
    if min(spatial) <= 0:
        raise ValueError("Conv output shape must be positive")
    return (input_shape[0], weights.shape[0], *spatial)


def standalone_fp32(operator: str, input_shape: tuple, weights: np.ndarray,
                    bias: np.ndarray | None = None, attributes: dict | None = None) -> onnx.ModelProto:
    weights = np.asarray(weights, np.float32)
    initializers = [numpy_helper.from_array(weights, "weight")]
    inputs = ["input", "weight"]
    attributes = attributes or {}
    if operator == "MatMul":
        if len(input_shape) != 2 or weights.ndim != 2 or input_shape[1] != weights.shape[0]:
            raise ValueError("Incompatible FP32 MatMul operands")
        output_shape = (input_shape[0], weights.shape[1])
    elif operator == "Conv":
        output_shape = conv_output_shape(input_shape, weights, attributes)
        if bias is not None:
            bias = np.asarray(bias, np.float32)
            if bias.shape != (weights.shape[0],) or not np.isfinite(bias).all():
                raise ValueError("Conv bias must have one finite value per output channel")
            initializers.append(numpy_helper.from_array(bias, "bias"))
            inputs.append("bias")
    else:
        raise ValueError("Unsupported standalone operator")
    node = helper.make_node(operator, inputs, ["output"], name="fp32_baseline", **attributes)
    graph = helper.make_graph([node], "original_fp32_isolated_operator",
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
                              initializer=initializers)
    model = helper.make_model(graph, producer_name="RE-USE unofficial QNN smoke generator",
                              opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
    model.doc_string = "Original FP32 standalone operator; not a complete enhancement model."
    onnx.checker.check_model(model, full_check=True)
    return model


def qdq_conv(input_shape: tuple, weights: np.ndarray, bias: np.ndarray | None,
             attributes: dict, activation_range: tuple, output_range: tuple) -> tuple[onnx.ModelProto, dict]:
    """UINT8 Conv with int32 bias quantized at activation_scale * weight_scale."""
    weights = np.asarray(weights, np.float32)
    output_shape = conv_output_shape(input_shape, weights, attributes)
    parameters = {"activation": quantization_parameters(*activation_range),
                  "weight": quantization_parameters(float(weights.min()), float(weights.max())),
                  "output": quantization_parameters(*output_range)}
    initializers = []
    for prefix, values in parameters.items():
        initializers.extend([
            numpy_helper.from_array(np.array(values["scale"], np.float32), prefix + "_scale"),
            numpy_helper.from_array(np.array(values["zero_point"], np.uint8), prefix + "_zero")])
    initializers.append(numpy_helper.from_array(quantize_uint8(weights, parameters["weight"]), "weight_u8"))
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "activation_scale", "activation_zero"],
                         ["input_u8"], name="input_quantize"),
        helper.make_node("DequantizeLinear", ["input_u8", "activation_scale", "activation_zero"],
                         ["input_dequantized"], name="input_dequantize"),
        helper.make_node("DequantizeLinear", ["weight_u8", "weight_scale", "weight_zero"],
                         ["weight_dequantized"], name="weight_dequantize")]
    conv_inputs = ["input_dequantized", "weight_dequantized"]
    if bias is not None:
        bias = np.asarray(bias, np.float32)
        if bias.shape != (weights.shape[0],) or not np.isfinite(bias).all():
            raise ValueError("Conv bias must have one finite value per output channel")
        scale = np.float32(parameters["activation"]["scale"]) * np.float32(parameters["weight"]["scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Bias scale must be finite and positive")
        quantized = np.rint(bias.astype(np.float64) / float(scale))
        limits = np.iinfo(np.int32)
        if np.any(quantized < limits.min) or np.any(quantized > limits.max):
            raise ValueError("Quantized Conv bias overflows int32")
        initializers.extend([
            numpy_helper.from_array(quantized.astype(np.int32), "bias_i32"),
            numpy_helper.from_array(np.array(scale, np.float32), "bias_scale"),
            numpy_helper.from_array(np.array(0, np.int32), "bias_zero")])
        nodes.append(helper.make_node("DequantizeLinear", ["bias_i32", "bias_scale", "bias_zero"],
                                      ["bias_dequantized"], name="bias_dequantize"))
        conv_inputs.append("bias_dequantized")
        parameters["bias"] = {"dtype": "int32", "scale": float(scale), "zero_point": 0,
                              "per_tensor": True, "policy": "round(original_FP32_bias / (activation_scale * weight_scale))"}
    nodes.extend([
        helper.make_node("Conv", conv_inputs, ["convolution"], name="probe_conv", **attributes),
        helper.make_node("QuantizeLinear", ["convolution", "output_scale", "output_zero"],
                         ["output_u8"], name="output_quantize"),
        helper.make_node("DequantizeLinear", ["output_u8", "output_scale", "output_zero"],
                         ["output"], name="output_dequantize")])
    graph = helper.make_graph(nodes, "qnn_htp_conv_smoke",
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
                              [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
                              initializer=initializers)
    model = helper.make_model(graph, producer_name="RE-USE unofficial QNN smoke generator",
                              opset_imports=[helper.make_opsetid("", 13)], ir_version=10)
    model.doc_string = "Isolated QDQ Conv smoke test; not a complete enhancement or quality-calibrated model."
    onnx.checker.check_model(model, full_check=True)
    return model, parameters


def cpu_reference(model: onnx.ModelProto, value: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # Keep the float-I/O QDQ equations explicit for the published reference.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    return session.run(["output"], {"input": np.asarray(value, np.float32)})[0]


def saturation_report(model: onnx.ModelProto, inputs: list[np.ndarray],
                       parameters: dict, operator: str) -> dict:
    """Measure actual rounding/clipping, without widening the calibrated range."""
    raw_name = "convolution" if operator == "Conv" else "projection"
    raw_model = onnx.ModelProto()
    raw_model.CopyFrom(model)
    shape = [d.dim_value for d in model.graph.output[0].type.tensor_type.shape.dim]
    raw_model.graph.output.append(helper.make_tensor_value_info(raw_name, TensorProto.FLOAT, shape))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(raw_model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    totals = {side: {"elements": 0, "below_range": 0, "above_range": 0, "maximum_excess_quantized_codes": 0.0}
              for side in ("activation", "output")}
    for value in inputs:
        raw = session.run([raw_name], {"input": np.asarray(value, np.float32)})[0]
        for side, array in (("activation", value), ("output", raw)):
            q = parameters[side]
            rounded = np.rint(np.asarray(array, np.float32) / np.float32(q["scale"])) + q["zero_point"]
            totals[side]["elements"] += int(rounded.size)
            totals[side]["below_range"] += int(np.count_nonzero(rounded < 0))
            totals[side]["above_range"] += int(np.count_nonzero(rounded > 255))
            totals[side]["maximum_excess_quantized_codes"] = max(
                totals[side]["maximum_excess_quantized_codes"], max(0.0, -float(rounded.min()), float(rounded.max()) - 255))
    for result in totals.values():
        result["clipped_fraction"] = (result["below_range"] + result["above_range"]) / result["elements"]
    return {"scope": "All supplied calibration calls (public NVIDIA clip for model-derived cases); values whose rounded quantized code falls outside uint8 before clamping.",
            "calls": len(inputs), **totals, "significant_fraction_threshold": 0.001,
            "significant": any(x["clipped_fraction"] > 0.001 for x in totals.values()),
            "quality_claim": False}


def write_case(output: Path, case_id: str, title: str, value: np.ndarray, weights: np.ndarray,
               activation_range: tuple[float, float], output_range: tuple[float, float],
               provenance: dict, fp32_reference: np.ndarray | None = None, *,
               operator: str = "MatMul", bias: np.ndarray | None = None,
               attributes: dict | None = None, calibration_inputs: list[np.ndarray] | None = None) -> dict:
    value = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Probe input contains nonfinite values")
    attributes = attributes or {}
    if operator == "MatMul":
        model, parameters = qdq_matmul(tuple(value.shape), weights, activation_range, output_range)
    elif operator == "Conv":
        model, parameters = qdq_conv(tuple(value.shape), weights, bias, attributes, activation_range, output_range)
    else:
        raise ValueError("Unsupported probe operator")
    fp32_model = standalone_fp32(operator, tuple(value.shape), weights, bias, attributes)
    fp32_output = cpu_reference(fp32_model, value)
    actual = cpu_reference(model, value)
    original = np.asarray(fp32_output if fp32_reference is None else fp32_reference, np.float32)
    if actual.shape != original.shape or not np.isfinite(actual).all() or not np.isfinite(original).all():
        raise ValueError("Probe reference has unexpected shape or nonfinite values")
    difference = actual.astype(np.float64) - original.astype(np.float64)
    np.testing.assert_allclose(fp32_output, original, rtol=1e-4, atol=1e-4,
                               err_msg="Standalone FP32 operator differs from its captured original trained source")
    fp32_difference = fp32_output.astype(np.float64) - original.astype(np.float64)
    model_path = output / (case_id + ".onnx")
    onnx.save(model, model_path)
    fixture_specs = {}
    for name, array in (("input", value), ("output", actual)):
        path = output / (case_id + "." + name + ".f32")
        np.asarray(array, dtype="<f4").tofile(path)
        fixture_specs[name] = {"name": name, "shape": list(array.shape),
                               "file": path.name, "sha256": sha256(path)}
    fp32_path = output / (case_id + ".fp32.onnx")
    onnx.save(fp32_model, fp32_path)
    fp32_output_path = output / (case_id + ".fp32.output.f32")
    np.asarray(fp32_output, dtype="<f4").tofile(fp32_output_path)
    saturation = saturation_report(model, calibration_inputs or [value], parameters, operator)
    return {"id": case_id, "title": title, "model": model_path.name,
            "model_sha256": sha256(model_path), **fixture_specs,
            "max_abs_error": max(1e-5, 2.1 * parameters["output"]["scale"]),
            "tolerance_policy": "At most 2 output quantization steps plus 0.1 step margin; compares hardware with CPU QDQ reference only.",
            "fp32_error": {"max_abs": float(np.max(np.abs(difference))),
                           "rmse": float(np.sqrt(np.mean(difference ** 2)))},
            "cpu_fp32": {"model": fp32_path.name, "model_sha256": sha256(fp32_path),
                         "output": {"name": "output", "shape": list(fp32_output.shape),
                                    "file": fp32_output_path.name, "sha256": sha256(fp32_output_path)},
                         "max_abs_error": 1e-4, "relative_tolerance": 1e-4,
                         "source_parity": {"max_abs": float(np.max(np.abs(fp32_difference))),
                                           "rmse": float(np.sqrt(np.mean(fp32_difference ** 2)))},
                         "weight_precision": "Original trained FP32 weights and bias; no quantization"},
            "operator": operator, "attributes": attributes,
            "calibration_saturation": saturation,
            "quantization": parameters, "source_provenance": provenance}


def synthetic_case(output: Path) -> dict:
    indices = np.arange(16 * 16, dtype=np.int32).reshape(16, 16)
    value = (((indices * 37 + 11) % 127 - 63) / 64.0).astype(np.float32)
    weights = (((indices * 19 + 7) % 61 - 30) / 64.0).astype(np.float32)
    calibration = [value, -value, np.roll(value, 3, axis=1), np.flip(value, axis=1).copy()]
    products = [item @ weights for item in calibration]
    return write_case(output, "synthetic_matmul_u8", "Synthetic uint8 MatMul (16 x 16)", value, weights,
                      (min(float(x.min()) for x in calibration), max(float(x.max()) for x in calibration)),
                      (min(float(x.min()) for x in products), max(float(x.max()) for x in products)),
                      {"kind": "deterministic_integer_patterns", "license": "MIT",
                       "calibration_cases": len(calibration), "contains_model_weights": False},
                      calibration_inputs=calibration)


def capture_operators(model_path: Path, sample_path: Path, library_path: Path,
                      threads: int) -> tuple[list[dict], dict]:
    """Capture three operators in one chronological public-clip reference pass."""
    import soundfile as sf
    from scipy.signal import resample_poly
    from audio_stream import StreamingAudio

    model = onnx.load(model_path)
    initializers = {x.name: x for x in model.graph.initializer}
    captures = []
    original_outputs = [x.name for x in model.graph.output]
    extra_outputs = []
    for spec in SOURCE_CASES:
        selected = [n for n in model.graph.node if n.name == spec["node"]]
        expected_inputs = [spec["input"], spec["weight"]] + ([spec["bias"]] if "bias" in spec else [])
        if (len(selected) != 1 or selected[0].op_type != spec["operator"]
                or list(selected[0].input) != expected_inputs or list(selected[0].output) != [spec["output"]]):
            raise ValueError("Pinned operator contract differs: " + spec["node"])
        attrs = {a.name: helper.get_attribute_value(a) for a in selected[0].attribute}
        if attrs != spec.get("attributes", {}):
            raise ValueError("Pinned operator attributes differ: " + spec["node"])
        weights = numpy_helper.to_array(initializers[spec["weight"]]).copy()
        bias = numpy_helper.to_array(initializers[spec["bias"]]).copy() if "bias" in spec else None
        if list(weights.shape) != spec["weight_shape"] or (bias is not None and bias.shape != (weights.shape[0],)):
            raise ValueError("Pinned operator weight/bias shape differs: " + spec["node"])
        captures.append({"spec": spec, "weights": weights, "bias": bias, "activation_min": math.inf,
                         "activation_max": -math.inf, "output_min": math.inf, "output_max": -math.inf,
                         "calibration_inputs": []})
        for side in ("input", "output"):
            name = spec[side]
            if name not in extra_outputs:
                extra_outputs.append(name)
                model.graph.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, spec[side + "_shape"]))
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.register_custom_ops_library(str(library_path.resolve()))
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    input_names = [x.name for x in session.get_inputs()]
    if input_names[:2] != ["magnitude", "phase"] or len(input_names) != 23 or len(original_outputs) != 23:
        raise ValueError("Pinned streaming state ABI differs")
    states = [np.zeros(x.shape, np.float32) for x in session.get_inputs()[2:]]
    samples, source_rate = sf.read(sample_path, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1, dtype=np.float32)
    divisor = math.gcd(source_rate, 8000)
    audio = resample_poly(samples, 8000 // divisor, source_rate // divisor).astype(np.float32)
    frontend = StreamingAudio(8000)
    calls = math.ceil(len(audio) / 800)
    if calls < 4:
        raise ValueError("Public calibration sample must contain several chronological calls")
    chosen_call = min(12, calls - 1)
    for call in range(calls):
        chunk = audio[call * 800:(call + 1) * 800]
        chunk = np.pad(chunk, (0, 800 - len(chunk)))
        frames = [frontend.analyze(chunk[i * 160:(i + 1) * 160]) for i in range(5)]
        magnitude = np.concatenate([item[0] for item in frames], axis=0)
        phase = np.concatenate([item[1] for item in frames], axis=0)
        feed = dict(zip(input_names, [magnitude, phase] + states))
        values = session.run(original_outputs + extra_outputs, feed)
        states = values[2:len(original_outputs)]
        intermediates = dict(zip(extra_outputs, values[len(original_outputs):]))
        for capture in captures:
            spec = capture["spec"]
            activation, result = intermediates[spec["input"]], intermediates[spec["output"]]
            if list(activation.shape) != spec["input_shape"] or list(result.shape) != spec["output_shape"]:
                raise ValueError("Captured operator dimensions differ: " + spec["node"])
            if not np.isfinite(activation).all() or not np.isfinite(result).all():
                raise ValueError("Source operator has nonfinite values: " + spec["node"])
            capture["activation_min"] = min(capture["activation_min"], float(activation.min()))
            capture["activation_max"] = max(capture["activation_max"], float(activation.max()))
            capture["output_min"] = min(capture["output_min"], float(result.min()))
            capture["output_max"] = max(capture["output_max"], float(result.max()))
            if spec["operator"] == "MatMul":
                activation, result = activation.reshape(405, 64), result.reshape(405, 512)
            capture["calibration_inputs"].append(activation.copy())
            if call == chosen_call:
                capture["value"] = activation.copy()
                capture["original"] = result.copy()
    for capture in captures:
        if "value" not in capture or not np.any(capture["value"]):
            raise ValueError("Selected public-sample operator input is zero")
    return captures, {
        "calibration_calls": calls, "calibration_audio_samples_8khz": len(audio),
        "calibration_source_rate": source_rate, "state_feedback": "all 21 FP32 recurrent states advanced chronologically",
        "test_call_index": chosen_call, "test_audio_start_seconds": chosen_call * 0.1,
        "final_call_zero_padded": len(audio) % 800 != 0,
    }


def prepare(model_path: Path, sample_path: Path, library_path: Path, output: Path,
            threads: int = 2) -> dict:
    """Verify every pinned source before writing any output assets."""
    lock = json.loads((ROOT / "tools/source-lock.json").read_text(encoding="utf-8"))
    clip_hash = lock["files"]["noisy_audio/mic_test.wav"]
    verify_hash(model_path, MODEL_SHA256, "source graph")
    verify_hash(sample_path, clip_hash, "public NVIDIA sample")
    if not library_path.is_file():
        raise ValueError("The source graph requires an existing host reuse_ort_ops library")
    if threads < 1:
        raise ValueError("threads must be positive")
    license_path = model_path.parent / "LICENSE-NVIDIA.txt"
    if not license_path.is_file() or not license_path.read_text(encoding="utf-8").strip():
        raise ValueError("Missing NVIDIA model license")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a new or empty output directory; existing probe assets are preserved")
    captures, calibration = capture_operators(model_path, sample_path, library_path, threads)
    provenance = {"kind": "extracted_real_time_reuse_operator", "source_graph_sha256": MODEL_SHA256,
                  "source_repository": lock["repository"], "source_revision": lock["revision"],
                  "source_checkpoint_sha256": lock["files"]["model.safetensors"],
                  "public_sample": "noisy_audio/mic_test.wav", "public_sample_sha256": clip_hash,
                  "license": "NVIDIA NSCLv1", "commercial_use": False, **calibration}
    output.mkdir(parents=True, exist_ok=True)
    cases = [synthetic_case(output)]
    for capture in captures:
        spec = capture["spec"]
        case_provenance = {**provenance, "source_node": spec["node"], "source_weight": spec["weight"]}
        if "bias" in spec:
            case_provenance["source_bias"] = spec["bias"]
        cases.append(write_case(output, spec["id"], spec["title"], capture["value"], capture["weights"],
                                (capture["activation_min"], capture["activation_max"]),
                                (capture["output_min"], capture["output_max"]), case_provenance,
                                capture["original"], operator=spec["operator"], bias=capture["bias"],
                                attributes=spec.get("attributes", {}), calibration_inputs=capture["calibration_inputs"]))
    (output / "LICENSE-NVIDIA.txt").write_bytes(license_path.read_bytes())
    manifest = {"schema_version": 2,
                "scope": "QNN HTP smoke tests only; no microphone, complete enhancement, general audio-quality calibration, or real-time performance claim.",
                "fixture_format": "IEEE754 float32 little-endian, contiguous row-major",
                "reference_backend": "ONNX Runtime CPUExecutionProvider; QDQ graph optimization disabled",
                "reference_onnxruntime_version": ort.__version__,
                "host_custom_op_library_sha256": sha256(library_path),
                "model_derived_license": "LICENSE-NVIDIA.txt", "cases": cases}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_DIRECTORY / "model.onnx")
    parser.add_argument("--sample", type=Path, default=ROOT / ".research/upstream/noisy_audio/mic_test.wav")
    parser.add_argument("--custom-op-library", type=Path,
                        default=ROOT / "native/build-windows-frequency-conv/Release/reuse_ort_ops.dll")
    parser.add_argument("--output", type=Path, default=ROOT / ".research/qnn-probe2-assets")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        manifest = prepare(args.model, args.sample, args.custom_op_library, args.output, args.threads)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps({"output": str(args.output), "cases": [
        {"id": item["id"], "model_sha256": item["model_sha256"], "fp32_error": item["fp32_error"]}
        for item in manifest["cases"]]}, indent=2))


if __name__ == "__main__":
    main()
