# SPDX-License-Identifier: LicenseRef-NVIDIA-NSCLv1
# Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
"""Stage a standard-ONNX FP32 reconstruction for experimental GPU testing.

This validates host CPU consistency, never GPU support, phone throughput or
perceptual quality. Inputs are preserved, and an occupied output is refused.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from optimize_dense_history import Constants, remove_dead_nodes, static_shapes
from prepare_qnn_encoder import feature_calls, load_public_audio, stream_comparison
from prepare_qnn_probe import MODEL_SHA256, ROOT, sha256, verify_hash

CHECKPOINT_SHA256 = "6eac4e8cc721f99a22cde7eac3c9f8e2826ced5dc8a4234fd1977de1ec0d3ade"
LICENSE_SHA256 = "6d1daa9c89ec421ba80fe051ac9f0dd46010d4fa4e8f4af654eb8e62f64e2ad5"
FIXTURE_MANIFEST_SHA256 = "3b4c54edae099e28df99b75285657f65104400ee70741fcd718ba895581aa51e"
PUBLIC_INPUT_SHA256 = "e6af48deb208432e7eeb624058b98ceb4b4542ab1a206c04fa621f3684ac33b5"
PUBLIC_SAMPLE_SHA256 = "dd54fe1decefe2e5c11c489f3d3d5b0e567bdd840f71bf4007cbe6349194f794"
HOST_LIBRARY_SHA256 = "f0cdf7db6e3ac18f39e9eee7c87bcf4d283e39ac3b092d4576883c81b16b6807"
REVISION = "4c0c186272ca853cd18d832fc5c28777b62f32d1"
MAX_ABS = .001
MAX_NORMALIZED_RMSE = .005
SETTINGS = {"sample_rate": 8000, "fft_size": 320, "hop_size": 160,
            "frames_per_call": 5, "chunk_size": 800, "exit_layers": 3,
            "look_ahead_frames": 0, "processing_interval_ms": 100,
            "maximum_collection_delay_ms": 80}
QUANTIZED_OPS = {"QuantizeLinear", "DequantizeLinear", "DynamicQuantizeLinear",
                 "MatMulInteger", "ConvInteger", "QLinearConv", "QLinearMatMul"}
NONCONSTANT_OPS = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike",
                   "Multinomial", "Bernoulli", "Dropout"}


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def static_float_io(model):
    """Return the ordered ABI; symbolic, empty, zero-size or non-FP32 IO fails."""
    result = []
    for side in (model.graph.input, model.graph.output):
        values = []
        for item in side:
            tensor = item.type.tensor_type
            if (not item.type.HasField("tensor_type") or tensor.elem_type != TensorProto.FLOAT
                    or not tensor.HasField("shape") or not tensor.shape.dim
                    or any(not d.HasField("dim_value") or d.dim_value <= 0 for d in tensor.shape.dim)):
                raise ValueError("Expected positive static FP32 public IO: " + item.name)
            values.append((item.name, tuple(d.dim_value for d in tensor.shape.dim)))
        if len({x[0] for x in values}) != len(values):
            raise ValueError("Duplicate public IO names")
        result.append(values)
    return result


def validate_standard_graph(model):
    if model.functions or model.graph.sparse_initializer:
        raise ValueError("Local functions and sparse initializers are outside the standard graph contract")
    if any(x.domain not in ("", "ai.onnx") for x in model.opset_import):
        raise ValueError("Custom-domain opset imports are forbidden")
    tensors = list(model.graph.initializer)
    if len({x.name for x in tensors}) != len(tensors):
        raise ValueError("Duplicate initializer names")
    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx"):
            raise ValueError("Custom-domain nodes are forbidden: " + node.name)
        if node.op_type in QUANTIZED_OPS or node.op_type.startswith("QLinear"):
            raise ValueError("Quantization is forbidden")
        for attribute in node.attribute:
            if attribute.type in (onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS):
                raise ValueError("Subgraphs are forbidden")
            if attribute.type in (onnx.AttributeProto.SPARSE_TENSOR, onnx.AttributeProto.SPARSE_TENSORS):
                raise ValueError("Sparse tensor attributes are forbidden")
            if attribute.type == onnx.AttributeProto.TENSOR:
                tensors.append(attribute.t)
            elif attribute.type == onnx.AttributeProto.TENSORS:
                tensors.extend(attribute.tensors)
            if node.op_type == "Cast" and attribute.name == "to" and attribute.i in (
                    TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE):
                raise ValueError("Reduced/double precision casts are forbidden")
    for tensor in tensors:
        if tensor.data_location == TensorProto.EXTERNAL or tensor.external_data:
            raise ValueError("External tensor data is forbidden")
        if tensor.data_type in (TensorProto.FLOAT16, TensorProto.BFLOAT16, TensorProto.DOUBLE):
            raise ValueError("Non-FP32 floating tensors are forbidden")
        value = numpy_helper.to_array(tensor)
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError("Nonfinite constant/initializer")
    static_float_io(model)
    onnx.checker.check_model(model, full_check=True)


def validate_manifest(manifest, model, baseline=None):
    for name, value in SETTINGS.items():
        if manifest.get(name) != value:
            raise ValueError("Pinned configuration differs: " + name)
    if (manifest.get("source_repository") != "nvidia/Real-time_RE-USE"
            or manifest.get("source_revision") != REVISION
            or manifest.get("source_checkpoint_sha256") != CHECKPOINT_SHA256
            or manifest.get("commercial_use") is not False):
        raise ValueError("Pinned checkpoint/repository/license provenance differs")
    states = manifest.get("states", [])
    if len(states) != 21 or any(s.get("input") != f"state_{i:03d}" or
                              s.get("output") != f"next_state_{i:03d}" or
                              s.get("dtype") != "float32" for i, s in enumerate(states)):
        raise ValueError("Expected all 21 ordered FP32 recurrent states")
    inputs, outputs = static_float_io(model)
    wanted_inputs = [("magnitude", (5, 161)), ("phase", (5, 161))]
    wanted_outputs = [("spectrum_real", (5, 161)), ("spectrum_imag", (5, 161))]
    wanted_inputs += [(s["input"], tuple(s["shape"])) for s in states]
    wanted_outputs += [(s["output"], tuple(s["shape"])) for s in states]
    if inputs != wanted_inputs or outputs != wanted_outputs:
        raise ValueError("Graph IO differs from the 23-input/output state ABI")
    if baseline is not None and states != baseline["states"]:
        raise ValueError("State ABI differs from pinned baseline")


def initializer_identity(reference, candidate):
    """Exporter-generated MatMul names may differ; dtype/shape/value bytes may not."""
    def records(model):
        values = []
        for item in model.graph.initializer:
            array = numpy_helper.to_array(item)
            values.append((int(item.data_type), tuple(item.dims), hashlib.sha256(array.tobytes(order="C")).hexdigest()))
        return Counter(values)
    left, right = records(reference), records(candidate)
    if left != right:
        raise ValueError("Exported initializer bytes differ from the pinned FP32 weights")
    left_named = {x.name: x for x in reference.graph.initializer}
    right_named = {x.name: x for x in candidate.graph.initializer}
    shared = set(left_named).intersection(right_named)
    if any(left_named[n] != right_named[n] for n in shared):
        raise ValueError("A same-named original trained initializer changed")
    canonical = json_bytes(sorted((dtype, list(shape), digest, count)
                                  for (dtype, shape, digest), count in left.items()))
    return {"passed": True, "initializer_count": sum(left.values()),
            "same_named_initializers_byte_identical": len(shared),
            "dtype_shape_value_multiset_sha256": hashlib.sha256(canonical).hexdigest(),
            "scope": "All exported initializers have byte-identical dtype, shape and values to the pinned fused baseline; exporter-generated names may change"}


def fold_static_constants(source):
    """Only evaluate closed standard expressions and metadata-proven Shape ops."""
    validate_standard_graph(source)
    model = copy.deepcopy(source)
    shape_replacements = 0
    total_bytes = 0
    folded_count = 0
    # Pad/Reshape/Slice controls can themselves be closed expressions. Fold
    # those first; only then can shape inference prove later runtime shapes.
    # Every productive iteration removes a node or replaces a Shape, so this
    # structural bound cannot freeze an endless metadata rewrite.
    for iteration in range(2 * len(source.graph.node) + 1):
        constants = Constants(model)
        available = {x.name for x in model.graph.initializer}
        folded, folded_nodes = {}, set()
        for index, node in enumerate(model.graph.node):
            if node.op_type in NONCONSTANT_OPS:
                continue
            if node.op_type != "Constant" and any(name and name not in available for name in node.input):
                continue
            values = {name: np.asarray(constants.get(name)) for name in node.output if name}
            if any(value.nbytes > 16 * 1024 * 1024 for value in values.values()):
                raise ValueError("Closed constant exceeds the bounded fold size")
            for name, value in values.items():
                if name in available:
                    raise ValueError("Constant folding would overwrite an existing tensor")
                total_bytes += value.nbytes
                if total_bytes > 64 * 1024 * 1024:
                    raise ValueError("Closed constants exceed the bounded total fold size")
                folded[name] = numpy_helper.from_array(value.copy(), name)
                available.add(name)
            folded_nodes.add(index)
        retained = [n for i, n in enumerate(model.graph.node) if i not in folded_nodes]
        del model.graph.node[:]; model.graph.node.extend(retained)
        model.graph.initializer.extend(folded.values())
        folded_count += len(folded_nodes)
        remove_dead_nodes(model)
        # Standard schemas can recreate this metadata from the fixed public
        # inputs. Discard stale exporter symbols instead of treating them as
        # proof of a runtime dimension.
        del model.graph.value_info[:]
        model = onnx.shape_inference.infer_shapes(model, check_type=True, strict_mode=True, data_prop=True)
        shapes = static_shapes(model)
        replaced = 0
        for node in model.graph.node:
            if node.op_type != "Shape":
                continue
            dimensions = shapes.get(node.input[0])
            if dimensions is None or any(not isinstance(x, int) or x < 0 for x in dimensions):
                continue
            attributes = {a.name: helper.get_attribute_value(a) for a in node.attribute}
            dimensions = dimensions[attributes.get("start", 0):attributes.get("end", len(dimensions))]
            node.op_type = "Constant"
            del node.input[:]; del node.attribute[:]
            node.attribute.append(helper.make_attribute("value", numpy_helper.from_array(np.asarray(dimensions, np.int64))))
            replaced += 1
        shape_replacements += replaced
        if not folded_nodes and not replaced:
            break
    else:
        raise ValueError("Static folding did not converge")
    information = {v.name: v for v in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output)}
    runtime_values = {x.name for x in model.graph.input} | {name for n in model.graph.node for name in n.output if name}
    for name in runtime_values:
        item = information.get(name)
        if item is None or not item.type.HasField("tensor_type"):
            raise ValueError("Runtime tensor lacks inferred type: " + name)
        tensor = item.type.tensor_type
        if not tensor.HasField("shape") or any(not d.HasField("dim_value") or d.dim_value <= 0 for d in tensor.shape.dim):
            raise ValueError("Runtime tensor is not static after closed folding: " + name)
    constants = {x.name for x in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type == "Shape":
            raise ValueError("Runtime Shape survived static folding")
        if node.op_type == "Slice" and any(name and name not in constants for name in node.input[1:]):
            raise ValueError("Slice controls are not constants")
    validate_standard_graph(model)
    if static_float_io(model) != static_float_io(source):
        raise ValueError("Static folding changed public IO")
    return model, {"kind": "proven_static_shapes_and_closed_standard_constant_expressions",
                   "shape_nodes_replaced": shape_replacements, "closed_nodes_folded": folded_count,
                   "folding_iterations": iteration + 1,
                   "folded_constant_bytes": total_bytes, "runtime_tensor_count": len(runtime_values),
                   "all_runtime_tensor_dimensions_static": True, "slice_controls_constant": True,
                   "no_weight_quantization_or_approximation": True}


def waveform_gate(comparison):
    metrics = copy.deepcopy(comparison["waveform"])
    maximum, normalized = metrics.get("max_abs"), metrics.get("normalized_rmse")
    passed = (isinstance(maximum, (int, float)) and np.isfinite(maximum) and 0 <= maximum <= MAX_ABS
              and isinstance(normalized, (int, float)) and np.isfinite(normalized)
              and 0 <= normalized <= MAX_NORMALIZED_RMSE)
    metrics.update(max_abs_limit=MAX_ABS, normalized_rmse_limit=MAX_NORMALIZED_RMSE, passed=bool(passed))
    if (not passed or not comparison.get("independent_state_feedback") or
            not comparison.get("all_21_states_checked") or not comparison.get("all_values_finite")
            or comparison.get("samples") != 51840 or comparison.get("calls") != 66):
        raise ValueError("Full CPU reconstruction failed fixed waveform/state/sample gates")
    return metrics


def fresh_output(output, inputs):
    output = Path(output).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if output == source or output in source.parents or source in output.parents:
            raise ValueError("Output must be separate from every input directory")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use a fresh or empty output directory; existing inputs/assets are preserved")
    return output


def prepare(source, fixtures, output, custom_op_library, threads=2):
    source, fixtures, custom_op_library = map(Path, (source, fixtures, custom_op_library))
    output = fresh_output(output, (source, fixtures))
    if threads < 1 or threads > 8:
        raise ValueError("Host CPU thread count must be one to eight")
    verify_hash(fixtures / "manifest.json", FIXTURE_MANIFEST_SHA256, "original fixture manifest")
    verify_hash(fixtures / "full-baseline/model.onnx", MODEL_SHA256, "source graph")
    verify_hash(fixtures / "full-baseline/LICENSE-NVIDIA.txt", LICENSE_SHA256, "NVIDIA license")
    verify_hash(source / "LICENSE-NVIDIA.txt", LICENSE_SHA256, "exported NVIDIA license")
    verify_hash(custom_op_library, HOST_LIBRARY_SHA256, "host custom-op library")
    lock = json.loads((ROOT / "tools/source-lock.json").read_text(encoding="utf-8"))
    if lock["files"]["model.safetensors"] != CHECKPOINT_SHA256 or lock["revision"] != REVISION:
        raise ValueError("Source lock differs from the pinned checkpoint")
    verify_hash(ROOT / ".research/upstream/model.safetensors", CHECKPOINT_SHA256, "source checkpoint")
    sample = ROOT / ".research/upstream/noisy_audio/mic_test.wav"
    verify_hash(sample, PUBLIC_SAMPLE_SHA256, "public NVIDIA sample")
    verify_hash(fixtures / "public-input-8k.f32", PUBLIC_INPUT_SHA256, "public 8 kHz input")
    original_manifest = json.loads((fixtures / "manifest.json").read_text(encoding="utf-8"))
    verify_hash(fixtures / "full-baseline/manifest.json",
                original_manifest["full_stream"]["baseline"]["manifest_sha256"], "baseline model manifest")
    if any(p.is_symlink() for p in fixtures.rglob("*")):
        raise ValueError("Fixture symlinks are forbidden")
    baseline_manifest = json.loads((fixtures / "full-baseline/manifest.json").read_text(encoding="utf-8"))
    exported_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    verify_hash(source / "model.onnx", exported_manifest["model_sha256"], "exported graph")
    if (source / "model.onnx").stat().st_size != exported_manifest.get("model_bytes"):
        raise ValueError("Exported graph byte count differs from manifest")
    baseline = onnx.load(fixtures / "full-baseline/model.onnx", load_external_data=False)
    exported = onnx.load(source / "model.onnx", load_external_data=False)
    validate_manifest(baseline_manifest, baseline)
    validate_manifest(exported_manifest, exported, baseline_manifest)
    if exported_manifest.get("custom_ops") != [] or exported_manifest.get("scan_mode") != "parallel":
        raise ValueError("Expected an explicitly standard parallel-scan export without custom ops")
    validate_standard_graph(exported)
    identity = initializer_identity(baseline, exported)
    candidate, folding = fold_static_constants(exported)
    audio = load_public_audio(sample)
    if audio.shape != (51840,) or not np.isfinite(audio).all():
        raise ValueError("Pinned public audio sample count/finiteness differs")
    if hashlib.sha256(np.asarray(audio, dtype="<f4").tobytes()).hexdigest() != PUBLIC_INPUT_SHA256:
        raise ValueError("Public resampling no longer reproduces the packaged FP32 input")
    # The existing loader already pads the last 80-ms chunk by one hop. One
    # further zero 800-sample block exercises Android's additional flush call.
    features = feature_calls(np.pad(audio, (0, 800)))
    if len(features) != 66:
        raise ValueError("Expected 65 public process calls plus one five-frame flush")
    print("Checking 65 public calls plus flush with independent 21-state feedback...", flush=True)
    comparison, waveforms = stream_comparison(baseline, candidate, features, len(audio),
                                             custom_op_library, threads, require_close=True)
    metrics = waveform_gate(comparison)
    if any(v.shape != (51840,) or not np.isfinite(v).all() for v in waveforms):
        raise ValueError("Reconstructed/reference waveform sample accounting failed")
    payload = candidate.SerializeToString()
    digest = hashlib.sha256(payload).hexdigest()
    manifest = copy.deepcopy(exported_manifest)
    manifest.update(model_sha256=digest, model_bytes=len(payload))
    manifest["graph_optimizations"] = {**folding, "source_export_model_sha256": exported_manifest["model_sha256"],
                                       "source_fused_model_sha256": MODEL_SHA256}
    manifest["validation"] = {"host_numerical_consistency_verified": True,
                              "phone_realtime_verified": False, "gpu_execution_verified": False,
                              "audio_quality_verified": False,
                              "scope": "See validation.json; host CPU numerical consistency only"}
    validation = {"schema_version": 1, "source": "full_gpu_standard_host_preparation",
                  "scope": "Standard FP32 reconstruction and host CPU numerical checks on one pinned public clip; no GPU, phone-throughput or perceptual-quality acceptance",
                  "host_validation_passed": True, "custom_node_count": 0, "standard_domain_only": True,
                  "static_float32_io": True, "state_abi_matches_baseline": True, "quantization_present": False,
                  "source_graph_sha256": MODEL_SHA256, "source_checkpoint_sha256": CHECKPOINT_SHA256,
                  "source_export_model_sha256": exported_manifest["model_sha256"], "model_sha256": digest,
                  "license_sha256": LICENSE_SHA256, "original_fixture_manifest_sha256": FIXTURE_MANIFEST_SHA256,
                  "public_input_sha256": PUBLIC_INPUT_SHA256, "public_sample_sha256": PUBLIC_SAMPLE_SHA256,
                  "host_custom_op_library_sha256": HOST_LIBRARY_SHA256, "initializer_identity": identity,
                  "static_folding": folding, "input_samples": 51840, "sample_rate": 8000,
                  "audio_calls": 65, "flush_calls": 1, "total_model_calls": 66, "frames_per_call": 5,
                  "state_count": 21, "waveform_comparison": metrics, "sequence_comparison": comparison,
                  "reference_waveform_sha256": hashlib.sha256(np.asarray(waveforms[0], dtype="<f4").tobytes()).hexdigest(),
                  "candidate_waveform_sha256": hashlib.sha256(np.asarray(waveforms[1], dtype="<f4").tobytes()).hexdigest(),
                  "host_backend": "CPUExecutionProvider", "host_threads": threads,
                  "versions": {"onnx": onnx.__version__, "onnxruntime": ort.__version__, "numpy": np.__version__},
                  "operator_counts": dict(sorted(Counter(n.op_type for n in candidate.graph.node).items())),
                  "node_count": len(candidate.graph.node), "device_tests_run": False,
                  "gpu_execution_verified": False, "phone_realtime_verified": False,
                  "microphone_started": False, "audio_captured": False, "audio_quality_verified": False}
    manifest_payload, validation_payload = json_bytes(manifest), json_bytes(validation)
    name = "full-gpu-standard"
    license_data = (source / "LICENSE-NVIDIA.txt").read_bytes()
    spec = {"model": name + "/model.onnx", "manifest": name + "/manifest.json",
            "validation": name + "/validation.json", "license": name + "/LICENSE-NVIDIA.txt",
            "model_sha256": digest, "manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
            "validation_sha256": hashlib.sha256(validation_payload).hexdigest(), "license_sha256": LICENSE_SHA256}
    original_manifest["full_gpu_standard"] = spec
    # Only write after every numerical/provenance guard passes. No existing
    # output files are replaced, and no recursive deletion is used.
    output.mkdir(parents=True, exist_ok=True)
    for p in sorted(fixtures.rglob("*")):
        if p.is_symlink():
            raise ValueError("Fixture symlinks are forbidden")
        target = output / p.relative_to(fixtures)
        if p.is_dir():
            target.mkdir(exist_ok=False)
        elif p.is_file():
            shutil.copyfile(p, target)
    target = output / name
    target.mkdir(exist_ok=False)
    for filename, value in (("model.onnx", payload), ("manifest.json", manifest_payload),
                            ("validation.json", validation_payload), ("LICENSE-NVIDIA.txt", license_data)):
        (target / filename).write_bytes(value)
    (output / "manifest.json").write_bytes(json_bytes(original_manifest))
    return {"full_gpu_standard": spec, "node_count": len(candidate.graph.node),
            "waveform_comparison": metrics, "host_validation_passed": True,
            "gpu_execution_verified": False, "audio_quality_verified": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".research/gpu-full-standard-export")
    parser.add_argument("--fixtures", type=Path, default=ROOT / ".research/qnn-eightconv-assets")
    parser.add_argument("--output", type=Path, default=ROOT / ".research/full-gpu-assets")
    parser.add_argument("--custom-op-library", type=Path,
                        default=ROOT / "native/build-windows-frequency-conv/Release/reuse_ort_ops.dll")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        result = prepare(args.source, args.fixtures, args.output, args.custom_op_library, args.threads)
    except (OSError, ValueError, RuntimeError, AssertionError, KeyError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
