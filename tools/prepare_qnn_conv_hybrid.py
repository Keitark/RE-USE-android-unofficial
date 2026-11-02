# SPDX-License-Identifier: LicenseRef-NVIDIA-NSCLv1
# Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
# Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
"""Graft one previously verified uint8 Conv into the complete causal F950 model.

This deliberately retains every original normalization/PReLU custom operation,
Mamba operation and FP32 state tensor. The probe2 isolated Conv/FP32 models and
fixtures are copied byte-for-byte. One public NVIDIA clip supplies the host
sequence check; it establishes neither general audio quality nor phone speed.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper

from prepare_qnn_probe import MODEL_DIRECTORY, MODEL_SHA256, ROOT, sha256, verify_hash
from prepare_qnn_encoder import feature_calls, load_public_audio, stream_comparison, write_float

CASE_ID = "reuse_encoder_dilated_conv_u8"
CONV_NAME = "/dense_encoder/dense_block/dense_block.3.1/Conv"
REGION_INPUT = "/dense_encoder/dense_block/Concat_9_output_0"
REGION_OUTPUT = "/dense_encoder/dense_block/dense_block.3.1/Conv_output_0"
PREFIX = "/qnn_single_heavy_conv/"
REGION_SHA256 = "7b3bca87e4e78674cbd8fb37724dae7977b79ba18e6a220e8a3399e0f948d713"
FP32_REGION_SHA256 = "59fa2612f7a6f160615eb51325cf83d0ce00e378cdef01d55a4169e3986ba2b3"


def attrs(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def graft_single_conv(source: onnx.ModelProto, region: onnx.ModelProto,
                      fp32_region: onnx.ModelProto) -> onnx.ModelProto:
    """Replace one Conv; keep every other node and original initializer byte-exact."""
    indices = [i for i, node in enumerate(source.graph.node) if node.name == CONV_NAME]
    if len(indices) != 1:
        raise ValueError("Expected exactly one selected source Conv")
    selected = source.graph.node[indices[0]]
    if (selected.op_type != "Conv" or selected.domain or selected.input[0] != REGION_INPUT
            or list(selected.output) != [REGION_OUTPUT] or len(selected.input) != 3):
        raise ValueError("Selected Conv contract differs")
    if len(fp32_region.graph.node) != 1 or fp32_region.graph.node[0].op_type != "Conv":
        raise ValueError("The original FP32 fixture must contain exactly one Conv")
    baseline = fp32_region.graph.node[0]
    if attrs(baseline) != attrs(selected):
        raise ValueError("Fixture and source Conv attributes differ")
    source_weights = {x.name: x for x in source.graph.initializer}
    fixture_weights = {x.name: x for x in fp32_region.graph.initializer}
    for original, fixture in zip(selected.input[1:], baseline.input[1:]):
        np.testing.assert_array_equal(numpy_helper.to_array(source_weights[original]),
                                      numpy_helper.to_array(fixture_weights[fixture]),
                                      err_msg="Fixture does not preserve original Conv parameters")
    if (len(region.graph.input) != 1 or len(region.graph.output) != 1
            or region.graph.input[0].name != "input" or region.graph.output[0].name != "output"):
        raise ValueError("Unexpected isolated region IO contract")
    imports = {x.domain: x.version for x in source.opset_import}
    if any(x.domain or imports.get("") < x.version for x in region.opset_import):
        raise ValueError("The Conv graft cannot change existing operator-set semantics")
    if any(n.domain or n.op_type not in {"QuantizeLinear", "DequantizeLinear", "Conv"} for n in region.graph.node):
        raise ValueError("Only the pinned standard Conv/QDQ graph can be grafted")
    if sum(n.op_type == "Conv" for n in region.graph.node) != 1:
        raise ValueError("Replacement must contain exactly one Conv")
    replacement = onnx.shape_inference.infer_shapes(copy.deepcopy(region))
    rename = {"input": REGION_INPUT, "output": REGION_OUTPUT}
    names = {x.name for x in replacement.graph.initializer}
    names.update(name for node in replacement.graph.node for name in [*node.input, *node.output] if name)
    rename.update({name: PREFIX + name for name in names if name not in rename})
    existing_names = {name for n in source.graph.node for name in [*n.input, *n.output]}
    existing_names.update(source_weights)
    if any(name in existing_names for key, name in rename.items() if key not in {"input", "output"}):
        raise ValueError("Replacement tensor namespace collides with the original model")
    original_node_names = {n.name for n in source.graph.node}
    for node in replacement.graph.node:
        node.name = PREFIX + node.name
        if node.name in original_node_names:
            raise ValueError("Replacement node namespace collides with the original model")
        for i, name in enumerate(node.input): node.input[i] = rename[name]
        for i, name in enumerate(node.output): node.output[i] = rename[name]
    for item in replacement.graph.initializer: item.name = rename[item.name]
    result = copy.deepcopy(source)
    retained = [*result.graph.node[:indices[0]], *replacement.graph.node, *result.graph.node[indices[0] + 1:]]
    del result.graph.node[:]
    result.graph.node.extend(retained)
    # Retain even the now-unused original Conv parameters, rather than pruning
    # or reserializing any existing trained tensor during this bounded graft.
    result.graph.initializer.extend(replacement.graph.initializer)
    information = {x.name: x for x in result.graph.value_info}
    public_names = {x.name for x in [*result.graph.input, *result.graph.output]}
    for item in [*region.graph.input, *region.graph.output,
                 *onnx.shape_inference.infer_shapes(region).graph.value_info]:
        value = copy.deepcopy(item)
        value.name = rename[value.name]
        if value.name in public_names:
            continue
        if value.name in information:
            information[value.name].CopyFrom(value)
        else:
            result.graph.value_info.append(value)
            information[value.name] = result.graph.value_info[-1]
    if list(result.graph.input) != list(source.graph.input) or list(result.graph.output) != list(source.graph.output):
        raise AssertionError("Single Conv graft changed the public stateful ABI")
    if len({n.name for n in result.graph.node}) != len(result.graph.node):
        raise ValueError("Candidate has duplicate node names")
    onnx.checker.check_model(result, full_check=True)
    return result


def safe_file(name):
    value = Path(name)
    if value.name != name or not name or ".." in name or "/" in name or "\\" in name:
        raise ValueError("Unsafe fixture filename")
    return name


def copy_case(source_dir, output, case):
    files = {case["model"]: case["model_sha256"],
             case["cpu_fp32"]["model"]: case["cpu_fp32"]["model_sha256"]}
    for spec in (case["input"], case["output"], case["cpu_fp32"]["output"]):
        files[spec["file"]] = spec["sha256"]
    for name, expected in files.items():
        source = source_dir / safe_file(name)
        verify_hash(source, expected, "probe2 Conv fixture")
        (output / name).write_bytes(source.read_bytes())


def write_model(output, directory, model, source_manifest, license_data, scope):
    target = output / directory
    target.mkdir()
    path = target / "model.onnx"
    onnx.save(model, path)
    manifest = copy.deepcopy(source_manifest)
    manifest["model_sha256"], manifest["model_bytes"] = sha256(path), path.stat().st_size
    manifest["graph_optimizations"] = {
        "name": "qnn_single_heavy_conv_experiment", "source_model_sha256": MODEL_SHA256,
        "scope": scope, "replaced_source_nodes": [CONV_NAME] if directory == "full-candidate" else [],
        "public_state_abi_unchanged": True, "original_initializers_retained_byte_exact": True,
        "previous_optimizations": manifest.get("graph_optimizations")}
    manifest["validation"] = {"phone_realtime_verified": False, "audio_quality_verified": False,
                              "scope": "Only one Conv is quantized; see host-stream-comparison.json"}
    metadata = target / "manifest.json"
    metadata.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    license_path = target / "LICENSE-NVIDIA.txt"
    license_path.write_bytes(license_data)
    return {"model": directory + "/model.onnx", "manifest": directory + "/manifest.json",
            "model_sha256": sha256(path), "manifest_sha256": sha256(metadata),
            "license_sha256": sha256(license_path)}


def prepare(model_path, sample_path, library_path, fixtures, output, threads=2):
    lock = json.loads((ROOT / "tools/source-lock.json").read_text(encoding="utf-8"))
    verify_hash(model_path, MODEL_SHA256, "source graph")
    verify_hash(sample_path, lock["files"]["noisy_audio/mic_test.wav"], "public NVIDIA sample")
    if not library_path.is_file() or threads < 1:
        raise ValueError("An existing host custom-op library and positive thread count are required")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh or empty fallback output directory; previous artifacts are preserved")
    original_manifest = json.loads((fixtures / "manifest.json").read_text(encoding="utf-8"))
    cases = [item for item in original_manifest["cases"] if item["id"] == CASE_ID]
    if len(cases) != 1:
        raise ValueError("Missing unique probe2 heavy Conv fixture")
    case = copy.deepcopy(cases[0])
    if (case["model_sha256"] != REGION_SHA256 or case["cpu_fp32"]["model_sha256"] != FP32_REGION_SHA256
            or case["source_provenance"]["source_graph_sha256"] != MODEL_SHA256
            or case["source_provenance"]["public_sample_sha256"] != lock["files"]["noisy_audio/mic_test.wav"]):
        raise ValueError("Fallback requires the original pinned probe2 heavy Conv models and public sample")
    verify_hash(fixtures / case["model"], REGION_SHA256, "probe2 quantized Conv")
    verify_hash(fixtures / case["cpu_fp32"]["model"], FP32_REGION_SHA256, "probe2 FP32 Conv")
    if case["max_abs_error"] != 2.1 * case["quantization"]["output"]["scale"]:
        raise ValueError("The original 2.1-step fixture tolerance changed")
    source = onnx.load(model_path)
    source_manifest = json.loads((model_path.parent / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest["model_sha256"] != MODEL_SHA256:
        raise ValueError("Pinned graph manifest hash differs")
    license_data = (model_path.parent / "LICENSE-NVIDIA.txt").read_bytes()
    if not license_data.strip():
        raise ValueError("Missing NVIDIA license")
    region = onnx.load(fixtures / case["model"])
    fp32_region = onnx.load(fixtures / case["cpu_fp32"]["model"])
    candidate = graft_single_conv(source, region, fp32_region)
    fp32_graft = graft_single_conv(source, fp32_region, fp32_region)
    audio = load_public_audio(sample_path)
    features = feature_calls(audio)
    if len(features) != 65 or len(audio) != 51840:
        raise ValueError("Pinned public clip no longer matches the bounded 65-call protocol")
    output.mkdir(parents=True, exist_ok=True)
    copy_case(fixtures, output, case)
    (output / "LICENSE-NVIDIA.txt").write_bytes(license_data)
    baseline = write_model(output, "full-baseline", source, source_manifest, license_data, "Original full F950 FP32 baseline")
    candidate_meta = write_model(output, "full-candidate", candidate, source_manifest, license_data,
                                 "Only the last dilated encoder Conv uses the byte-exact probe2 uint8 QDQ; every original normalization, Mamba operation and state remains FP32")
    manifest = {"schema_version": 2,
                "scope": "Experimental single-Conv hybrid full-model candidate and original isolated prerequisite; no general audio-quality or phone-speed claim",
                "fixture_format": "IEEE754 float32 little-endian, contiguous row-major",
                "cases": [case], "model_derived_license": "LICENSE-NVIDIA.txt",
                "host_custom_op_library_sha256": sha256(library_path),
                "source_fixture_manifest_sha256": sha256(fixtures / "manifest.json"),
                "full_stream": {"input": {**write_float(output, "public-input-8k.f32", audio), "sample_rate": 8000},
                                "baseline": baseline, "candidate": candidate_meta, "calls": len(features), "frames_per_call": 5,
                                "public_sample_sha256": sha256(sample_path), "host_comparison": "host-stream-comparison.json",
                                "host_comparison_complete": False, "audio_quality_verified": False, "phone_performance_verified": False},
                "preservation": {"normalization_and_prelu": "All original native FP32 custom operations retained byte-exact",
                                 "mamba_and_states": "All original operations and 21 FP32 state input/output definitions retained byte-exact",
                                 "original_initializers_retained": len(source.graph.initializer)}}
    destination = output / "manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Single-Conv assets and full models written; checking chronological FP32 graft and quantized candidate...", flush=True)
    fp32_check, _ = stream_comparison(source, fp32_graft, features, len(audio), library_path, threads, require_close=True)
    qdq_check, waveforms = stream_comparison(source, candidate, features, len(audio), library_path, threads)
    comparison = {"scope": "Host CPU comparison on one pinned public clip; no physical-phone speed or general quality acceptance",
                  "fp32_conv_graft": fp32_check, "single_conv_qdq_candidate": qdq_check,
                  "source_graph_sha256": MODEL_SHA256, "candidate_graph_sha256": candidate_meta["model_sha256"]}
    (output / "host-stream-comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest["full_stream"].update({"baseline_output": write_float(output, "full-baseline-output.f32", waveforms[0]),
                                    "candidate_cpu_output": write_float(output, "full-candidate-cpu-output.f32", waveforms[1]),
                                    "host_comparison_complete": True})
    destination.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_DIRECTORY / "model.onnx")
    parser.add_argument("--sample", type=Path, default=ROOT / ".research/upstream/noisy_audio/mic_test.wav")
    parser.add_argument("--custom-op-library", type=Path,
                        default=ROOT / "native/build-windows-frequency-conv/Release/reuse_ort_ops.dll")
    parser.add_argument("--fixtures", type=Path, default=ROOT / ".research/qnn-probe2-assets")
    parser.add_argument("--output", type=Path, default=ROOT / ".research/qnn-conv-hybrid-assets")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    try:
        manifest = prepare(args.model, args.sample, args.custom_op_library, args.fixtures, args.output, args.threads)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps({"output": str(args.output), "manifest_sha256": sha256(args.output / "manifest.json"),
                      "candidate": manifest["full_stream"]["candidate"]}, indent=2))


if __name__ == "__main__":
    main()
