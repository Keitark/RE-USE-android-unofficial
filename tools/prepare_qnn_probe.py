# SPDX-License-Identifier: LicenseRef-NVIDIA-NSCLv1
# Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
# Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
"""Prepare small, fixed-shape QNN HTP smoke tests, not an audio-quality model.

The synthetic case has no NVIDIA weights. The projection case extracts the
first temporal input projection from the pinned Android RE-USE graph. Its
activation ranges come from the pinned, public NVIDIA sample, with all source
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


def cpu_reference(model: onnx.ModelProto, value: np.ndarray) -> np.ndarray:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    # Keep the float-I/O QDQ equations explicit for the published reference.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(model.SerializeToString(), options, providers=["CPUExecutionProvider"])
    return session.run(["output"], {"input": np.asarray(value, np.float32)})[0]


def write_case(output: Path, case_id: str, title: str, value: np.ndarray, weights: np.ndarray,
               activation_range: tuple[float, float], output_range: tuple[float, float],
               provenance: dict, fp32_reference: np.ndarray | None = None) -> dict:
    value = np.ascontiguousarray(value, dtype=np.float32)
    if not np.isfinite(value).all():
        raise ValueError("Probe input contains nonfinite values")
    model, parameters = qdq_matmul(tuple(value.shape), weights, activation_range, output_range)
    actual = cpu_reference(model, value)
    original = np.asarray(value @ weights if fp32_reference is None else fp32_reference, np.float32)
    if actual.shape != original.shape or not np.isfinite(actual).all() or not np.isfinite(original).all():
        raise ValueError("Probe reference has unexpected shape or nonfinite values")
    difference = actual.astype(np.float64) - original.astype(np.float64)
    model_path = output / (case_id + ".onnx")
    onnx.save(model, model_path)
    fixture_specs = {}
    for name, array in (("input", value), ("output", actual)):
        path = output / (case_id + "." + name + ".f32")
        np.asarray(array, dtype="<f4").tofile(path)
        fixture_specs[name] = {"name": name, "shape": list(array.shape),
                               "file": path.name, "sha256": sha256(path)}
    return {"id": case_id, "title": title, "model": model_path.name,
            "model_sha256": sha256(model_path), **fixture_specs,
            "max_abs_error": max(1e-5, 2.1 * parameters["output"]["scale"]),
            "tolerance_policy": "At most 2 output quantization steps plus 0.1 step margin; compares hardware with CPU QDQ reference only.",
            "fp32_error": {"max_abs": float(np.max(np.abs(difference))),
                           "rmse": float(np.sqrt(np.mean(difference ** 2)))},
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
                       "calibration_cases": len(calibration), "contains_model_weights": False})


def capture_projection(model_path: Path, sample_path: Path, library_path: Path,
                       threads: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple, tuple, dict]:
    """Capture all public-clip activation ranges while feeding source states back."""
    import soundfile as sf
    from scipy.signal import resample_poly
    from audio_stream import StreamingAudio

    model = onnx.load(model_path)
    selected = [n for n in model.graph.node if n.name == PROJECTION_NODE]
    if (len(selected) != 1 or selected[0].op_type != "MatMul"
            or list(selected[0].input) != [PROJECTION_INPUT, PROJECTION_WEIGHT]
            or list(selected[0].output) != [PROJECTION_OUTPUT]):
        raise ValueError("Pinned input projection contract differs")
    weights = numpy_helper.to_array(next(x for x in model.graph.initializer if x.name == PROJECTION_WEIGHT)).copy()
    if weights.shape != (64, 512):
        raise ValueError("Pinned projection weight shape differs")
    original_outputs = [x.name for x in model.graph.output]
    model.graph.output.extend([
        helper.make_tensor_value_info(PROJECTION_INPUT, TensorProto.FLOAT, [81, 5, 64]),
        helper.make_tensor_value_info(PROJECTION_OUTPUT, TensorProto.FLOAT, [81, 5, 512]),
    ])
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
    chosen_input = chosen_output = None
    a_min = y_min = math.inf
    a_max = y_max = -math.inf
    for call in range(calls):
        chunk = audio[call * 800:(call + 1) * 800]
        chunk = np.pad(chunk, (0, 800 - len(chunk)))
        frames = [frontend.analyze(chunk[i * 160:(i + 1) * 160]) for i in range(5)]
        magnitude = np.concatenate([item[0] for item in frames], axis=0)
        phase = np.concatenate([item[1] for item in frames], axis=0)
        feed = dict(zip(input_names, [magnitude, phase] + states))
        values = session.run(original_outputs + [PROJECTION_INPUT, PROJECTION_OUTPUT], feed)
        states = values[2:len(original_outputs)]
        activation, projection = values[-2:]
        if not np.isfinite(activation).all() or not np.isfinite(projection).all():
            raise ValueError("Source projection is nonfinite")
        a_min, a_max = min(a_min, float(activation.min())), max(a_max, float(activation.max()))
        y_min, y_max = min(y_min, float(projection.min())), max(y_max, float(projection.max()))
        if call == chosen_call:
            chosen_input = activation.reshape(405, 64).copy()
            chosen_output = projection.reshape(405, 512).copy()
    if chosen_input is None or not np.any(chosen_input):
        raise ValueError("Selected public-sample projection input is zero")
    return chosen_input, weights, chosen_output, (a_min, a_max), (y_min, y_max), {
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
    value, weights, original, a_range, y_range, calibration = capture_projection(
        model_path, sample_path, library_path, threads)
    provenance = {"kind": "extracted_real_time_reuse_projection", "source_graph_sha256": MODEL_SHA256,
                  "source_repository": lock["repository"], "source_revision": lock["revision"],
                  "source_checkpoint_sha256": lock["files"]["model.safetensors"],
                  "public_sample": "noisy_audio/mic_test.wav", "public_sample_sha256": clip_hash,
                  "source_node": PROJECTION_NODE, "source_weight": PROJECTION_WEIGHT,
                  "license": "NVIDIA NSCLv1", "commercial_use": False, **calibration}
    output.mkdir(parents=True, exist_ok=True)
    cases = [synthetic_case(output), write_case(output, "reuse_in_proj_u8",
             "RE-USE input projection (405 x 64 x 512, uint8)", value, weights, a_range, y_range,
             provenance, original)]
    (output / "LICENSE-NVIDIA.txt").write_bytes(license_path.read_bytes())
    manifest = {"schema_version": 1,
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
    parser.add_argument("--output", type=Path, default=ROOT / ".research/qnn-probe-assets")
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
