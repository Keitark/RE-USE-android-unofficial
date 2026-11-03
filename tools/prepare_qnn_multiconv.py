# SPDX-License-Identifier: LicenseRef-NVIDIA-NSCLv1
# Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
# Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
"""Stage nine Conv-only QDQ experiments; preserve the selected hybrid bundle.

Only three dilated dense convolutions in each encoder/mask/phase branch are
replaced. Every native FP32 normalization, PReLU, Mamba node, original
initializer and public state definition remains unchanged. Calibration and
checks use one pinned public clip, never microphone audio. Finite outputs or
waveform RMSE do not establish general speech quality or NPU compatibility.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from prepare_qnn_encoder import (MODEL_DIRECTORY, MODEL_SHA256, ROOT, attributes,
                                feature_calls, load_public_audio, session,
                                stream_comparison, write_float)
from prepare_qnn_probe import sha256, verify_hash, write_case

SINGLE_CONV_SHA256 = "605740abc6dc59d7c77776a12ee9ef6534ea58ea9936c7f3970a1816c93a9d9f"
ORIGINAL_HEAVY_CASE_SHA256 = "7b3bca87e4e78674cbd8fb37724dae7977b79ba18e6a220e8a3399e0f948d713"


def target_specs():
    result = []
    for branch, label, bins in (("dense_encoder", "encoder", 163),
                                 ("mask_decoder_list.2", "mask", 81),
                                 ("phase_decoder_list.2", "phase", 81)):
        for block in (1, 2, 3):
            name = f"/{branch}/dense_block/dense_block.{block}.1/Conv"
            channels, dilation = 64 * (block + 1), 2 ** block
            case_id = f"reuse_{label}_dense{block}_conv_u8"
            if label == "encoder" and block == 3:
                case_id = "reuse_encoder_dilated_conv_u8"
            result.append({"id": case_id, "title": f"RE-USE {label} dense Conv {block} (uint8)",
                           "node": name, "input": f"/{branch}/dense_block/Concat_{block * 3}_output_0",
                           "output": name + "_output_0",
                           "weight": f"{branch}.dense_block.dense_block.{block}.1.weight",
                           "bias": f"{branch}.dense_block.dense_block.{block}.1.bias",
                           "input_shape": [1, channels, 5 + 2 * dilation, bins],
                           "output_shape": [1, 64, 5, bins], "weight_shape": [64, channels, 3, 3],
                           "attributes": {"dilations": [dilation, 1], "group": 1, "kernel_shape": [3, 3],
                                          "pads": [0, 1, 0, 1], "strides": [1, 1]}})
    return result


TARGETS = target_specs()


def select_targets(exclude_nodes=None):
    excluded = list(exclude_nodes or [])
    if len(set(excluded)) != len(excluded):
        raise ValueError("Duplicate excluded Conv node")
    unknown = set(excluded) - {x["node"] for x in TARGETS}
    if unknown:
        raise ValueError("Excluded node is outside the known nine Convs: " + ", ".join(sorted(unknown)))
    targets = [x for x in TARGETS if x["node"] not in excluded]
    if not targets:
        raise ValueError("At least one Conv target must remain")
    return targets


def capture_calibration(source, features, library, threads, specs=TARGETS):
    augmented = copy.deepcopy(source)
    weights = {x.name: x for x in source.graph.initializer}
    nodes = {x.name: x for x in source.graph.node}
    original_outputs = [x.name for x in source.graph.output]
    if len(source.graph.input) != 23 or len(original_outputs) != 23:
        raise ValueError("Expected the pinned 23-input/output streaming ABI")
    captures, extra = [], []
    for spec in specs:
        node = nodes.get(spec["node"])
        if (node is None or node.domain or node.op_type != "Conv"
                or list(node.input) != [spec["input"], spec["weight"], spec["bias"]]
                or list(node.output) != [spec["output"]] or attributes(node) != spec["attributes"]):
            raise ValueError("Pinned Conv contract differs: " + spec["node"])
        weight, bias = [numpy_helper.to_array(weights[spec[key]]).copy() for key in ("weight", "bias")]
        if list(weight.shape) != spec["weight_shape"] or bias.shape != (64,):
            raise ValueError("Pinned Conv parameter shape differs")
        captures.append({"spec": spec, "weights": weight, "bias": bias, "inputs": [],
                         "input_min": float("inf"), "input_max": -float("inf"),
                         "output_min": float("inf"), "output_max": -float("inf")})
        for side in ("input", "output"):
            extra.append(spec[side])
            augmented.graph.output.append(helper.make_tensor_value_info(spec[side], TensorProto.FLOAT, spec[side + "_shape"]))
    runtime = session(augmented, library, threads)
    input_names = [x.name for x in runtime.get_inputs()]
    states = [np.zeros(x.shape, np.float32) for x in runtime.get_inputs()[2:]]
    chosen = min(12, len(features) - 1)
    for call, (magnitude, phase) in enumerate(features):
        values = runtime.run(original_outputs + extra, dict(zip(input_names, [magnitude, phase, *states])))
        states = values[2:len(original_outputs)]
        intermediates = dict(zip(extra, values[len(original_outputs):]))
        for capture in captures:
            spec = capture["spec"]
            value, result = intermediates[spec["input"]], intermediates[spec["output"]]
            if list(value.shape) != spec["input_shape"] or list(result.shape) != spec["output_shape"]:
                raise ValueError("Captured Conv shape differs")
            if not np.isfinite(value).all() or not np.isfinite(result).all():
                raise ValueError("Nonfinite source Conv calibration")
            capture["input_min"] = min(capture["input_min"], float(value.min()))
            capture["input_max"] = max(capture["input_max"], float(value.max()))
            capture["output_min"] = min(capture["output_min"], float(result.min()))
            capture["output_max"] = max(capture["output_max"], float(result.max()))
            capture["inputs"].append(value.copy())
            if call == chosen:
                capture["reference"] = result.copy()
    return captures, chosen


def graft_convolutions(source, replacements, specs=TARGETS):
    """Replace only named Conv nodes; retain every original initializer/node."""
    model = copy.deepcopy(source)
    standard_opset = next(x.version for x in source.opset_import if x.domain == "")
    if standard_opset != 17:
        raise ValueError("The full source must retain standard opset17 semantics")
    by_name = {x.name: x for x in source.graph.node}
    inserted, new_initializers, new_information = {}, [], {}
    for spec in specs:
        original = by_name.get(spec["node"])
        if (original is None or original.domain or original.op_type != "Conv"
                or list(original.input) != [spec["input"], spec["weight"], spec["bias"]]
                or list(original.output) != [spec["output"]] or attributes(original) != spec["attributes"]):
            raise ValueError("Source replacement target differs")
        replacement = copy.deepcopy(replacements[spec["id"]])
        if len(replacement.graph.input) != 1 or len(replacement.graph.output) != 1:
            raise ValueError("Replacement must have one input and one output")
        for side in ("input", "output"):
            value = getattr(replacement.graph, side)[0]
            if value.type.tensor_type.elem_type != TensorProto.FLOAT or [d.dim_value for d in value.type.tensor_type.shape.dim] != spec[side + "_shape"]:
                raise ValueError("Replacement float IO shape differs")
        replacement_opset = next(x.version for x in replacement.opset_import if x.domain == "")
        for node in replacement.graph.node:
            if node.domain or node.op_type not in ("Conv", "QuantizeLinear", "DequantizeLinear"):
                raise ValueError("Replacement contains an operation outside Conv/QDQ")
            if onnx.defs.get_schema(node.op_type, replacement_opset).since_version != onnx.defs.get_schema(node.op_type, 17).since_version:
                raise ValueError("Replacement operator semantics differ from full opset17")
        if sum(x.op_type == "Conv" for x in replacement.graph.node) != 1:
            raise ValueError("Replacement must contain exactly one Conv")
        rename = {replacement.graph.input[0].name: spec["input"], replacement.graph.output[0].name: spec["output"]}
        prefix = "/qnn_multiconv/" + spec["id"] + "/"
        for node in replacement.graph.node:
            for name in list(node.input) + list(node.output):
                rename.setdefault(name, prefix + name)
        for value in replacement.graph.initializer:
            value.name = rename[value.name]
        for node in replacement.graph.node:
            node.name = prefix + node.name
            for i, name in enumerate(node.input): node.input[i] = rename[name]
            for i, name in enumerate(node.output): node.output[i] = rename[name]
        inserted[spec["node"]] = list(replacement.graph.node)
        new_initializers.extend(replacement.graph.initializer)
        inferred = onnx.shape_inference.infer_shapes(replacements[spec["id"]])
        for value in list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output):
            value = copy.deepcopy(value); value.name = rename[value.name]
            new_information[value.name] = value
    nodes = []
    for node in model.graph.node:
        nodes.extend(inserted.get(node.name, [node]))
    del model.graph.node[:]; model.graph.node.extend(nodes)
    model.graph.initializer.extend(new_initializers)
    existing_info = {x.name: x for x in model.graph.value_info}
    public = {x.name for x in list(model.graph.input) + list(model.graph.output)}
    for name, value in new_information.items():
        if name in public: continue
        if name in existing_info: existing_info[name].CopyFrom(value)
        else: model.graph.value_info.append(value)
    if list(model.graph.input) != list(source.graph.input) or list(model.graph.output) != list(source.graph.output):
        raise AssertionError("Multiconv graft changed public IO/state definitions")
    if len({x.name for x in model.graph.node}) != len(model.graph.node):
        raise ValueError("Multiconv graft has duplicate node names")
    onnx.checker.check_model(model, full_check=True)
    return model


def clipping_statistics(value, quantization):
    if not np.isfinite(value).all(): raise ValueError("Nonfinite candidate Conv activation")
    codes = np.rint(np.asarray(value, np.float32) / np.float32(quantization["scale"])) + quantization["zero_point"]
    below, above = int(np.count_nonzero(codes < 0)), int(np.count_nonzero(codes > 255))
    return {"elements": int(codes.size), "below_range": below, "above_range": above,
            "clipped_fraction": (below + above) / codes.size,
            "maximum_excess_quantized_codes": max(0.0, -float(codes.min()), float(codes.max()) - 255)}


def candidate_clipping(model, cases, features, library, threads, specs=TARGETS):
    augmented = copy.deepcopy(model)
    original_names = [x.name for x in model.graph.output]
    extra = []
    for spec in specs:
        extra.append(spec["input"])
        augmented.graph.output.append(helper.make_tensor_value_info(spec["input"], TensorProto.FLOAT, spec["input_shape"]))
    runtime = session(augmented, library, threads)
    inputs = runtime.get_inputs()
    states = [np.zeros(x.shape, np.float32) for x in inputs[2:]]
    params = {x["id"]: x["quantization"]["activation"] for x in cases}
    totals = {spec["id"]: {"elements": 0, "below_range": 0, "above_range": 0, "maximum_excess_quantized_codes": 0.0} for spec in specs}
    for magnitude, phase in features:
        values = runtime.run(original_names + extra, dict(zip([x.name for x in inputs], [magnitude, phase, *states])))
        states = values[2:len(original_names)]
        for spec, value in zip(specs, values[len(original_names):]):
            item = clipping_statistics(value, params[spec["id"]]); total = totals[spec["id"]]
            for field in ("elements", "below_range", "above_range"): total[field] += item[field]
            total["maximum_excess_quantized_codes"] = max(total["maximum_excess_quantized_codes"], item["maximum_excess_quantized_codes"])
    for total in totals.values():
        total["clipped_fraction"] = (total["below_range"] + total["above_range"]) / total["elements"]
        total["significant"] = total["clipped_fraction"] > .001
    return {"scope": "Actual candidate's pre-Conv activations during 65 CPU chronological calls, each with its own recurrent feedback; not perceptual quality.",
            "calls": len(features), "significant_fraction_threshold": .001, "cases": totals}


def write_model(output, folder, model, source_manifest, license_data, scope, specs=TARGETS, exclude_nodes=None):
    directory = output / folder; directory.mkdir()
    path = directory / "model.onnx"; onnx.save(model, path)
    manifest = copy.deepcopy(source_manifest)
    manifest.update(model_sha256=sha256(path), model_bytes=path.stat().st_size)
    manifest["graph_optimizations"] = {"name": "qnn_multiconv_experiment", "scope": scope, "source_model_sha256": MODEL_SHA256,
                                       "target_nodes": [x["node"] for x in specs], "excluded_known_nodes": list(exclude_nodes or []),
                                       "previous_optimizations": source_manifest.get("graph_optimizations")}
    manifest["validation"] = {"phone_realtime_verified": False, "audio_quality_verified": False,
                              "scope": "Host experiment only; see host-stream-comparison.json"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (directory / "LICENSE-NVIDIA.txt").write_bytes(license_data)
    return {"model": folder + "/model.onnx", "model_sha256": sha256(path),
            "manifest": folder + "/manifest.json", "manifest_sha256": sha256(directory / "manifest.json"),
            "license": folder + "/LICENSE-NVIDIA.txt", "license_sha256": sha256(directory / "LICENSE-NVIDIA.txt")}


def selected_model(assets):
    manifest = json.loads((assets / "manifest.json").read_text(encoding="utf-8"))
    meta = manifest["full_stream"]["candidate"]
    path = (assets / meta["model"]).resolve()
    if not path.is_relative_to(assets.resolve()): raise ValueError("Unsafe selected model path")
    if meta["model_sha256"] != SINGLE_CONV_SHA256: raise ValueError("Selected single-Conv model differs from pinned comparison")
    verify_hash(path, SINGLE_CONV_SHA256, "selected single-Conv model")
    return onnx.load(path)


def prepare(model_path, sample_path, library_path, output, selected_assets, threads=2, exclude_nodes=None):
    lock = json.loads((ROOT / "tools/source-lock.json").read_text(encoding="utf-8"))
    verify_hash(model_path, MODEL_SHA256, "source graph")
    verify_hash(sample_path, lock["files"]["noisy_audio/mic_test.wav"], "public NVIDIA sample")
    specs = select_targets(exclude_nodes)
    if output.exists() and any(output.iterdir()): raise ValueError("Use a fresh or empty multiconv output directory")
    if not library_path.is_file() or threads < 1: raise ValueError("Existing host custom-op library and positive threads required")
    selected = selected_model(selected_assets)
    source = onnx.load(model_path)
    source_manifest = json.loads((model_path.parent / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest["model_sha256"] != MODEL_SHA256: raise ValueError("Source manifest differs from pinned graph")
    license_data = (model_path.parent / "LICENSE-NVIDIA.txt").read_bytes()
    if not license_data.strip(): raise ValueError("Missing complete NVIDIA license")
    audio = load_public_audio(sample_path); features = feature_calls(audio)
    if len(features) != 65 or len(audio) != 51840: raise ValueError("Pinned public clip no longer matches bounded protocol")
    print(f"Capturing {len(specs)} Conv activations with all 21 source states advanced chronologically...", flush=True)
    captures, chosen = capture_calibration(source, features, library_path, threads, specs)
    output.mkdir(parents=True, exist_ok=True)
    cases, qdq, fp32 = [], {}, {}
    for capture in captures:
        spec = capture["spec"]
        print("Preparing isolated fixture: " + spec["id"], flush=True)
        provenance = {"kind": "extracted_real_time_reuse_conv", "source_graph_sha256": MODEL_SHA256,
                      "source_repository": lock["repository"], "source_revision": lock["revision"],
                      "public_sample_sha256": sha256(sample_path), "source_node": spec["node"],
                      "calibration_calls": len(features), "test_call_index": chosen,
                      "state_feedback": "All 21 source states advanced chronologically", "license": "NVIDIA NSCLv1", "commercial_use": False}
        case = write_case(output, spec["id"], spec["title"], capture["inputs"][chosen], capture["weights"],
                          (capture["input_min"], capture["input_max"]), (capture["output_min"], capture["output_max"]),
                          provenance, capture["reference"], operator="Conv", bias=capture["bias"],
                          attributes=spec["attributes"], calibration_inputs=capture["inputs"])
        if spec["id"] == "reuse_encoder_dilated_conv_u8" and case["model_sha256"] != ORIGINAL_HEAVY_CASE_SHA256:
            raise ValueError("Original heavy-Conv prerequisite no longer reproduces probe2 bytes")
        cases.append(case)
        qdq[spec["id"]] = onnx.load(output / case["model"])
        fp32[spec["id"]] = onnx.load(output / case["cpu_fp32"]["model"])
        # The full source-calibration arrays are no longer needed after its case.
        capture["inputs"].clear()
    candidate = graft_convolutions(source, qdq, specs)
    fp32_control = graft_convolutions(source, fp32, specs)
    baseline_meta = write_model(output, "full-baseline", source, source_manifest, license_data, "Pinned FP32 baseline", specs, exclude_nodes)
    candidate_meta = write_model(output, "full-candidate", candidate, source_manifest, license_data,
                                 f"{len(specs)} Conv-only UINT8 QDQ replacements; all excluded Convs, native normalization, Mamba and public states unchanged", specs, exclude_nodes)
    (output / "LICENSE-NVIDIA.txt").write_bytes(license_data)
    manifest = {"schema_version": 2, "scope": f"{len(specs)} Conv-only host experiment; every isolated numerical gate remains 2.1 output steps. Hardware compatibility and general audio quality unverified.",
                "fixture_format": "IEEE754 float32 little-endian, contiguous row-major", "cases": cases,
                "model_derived_license": "LICENSE-NVIDIA.txt", "host_custom_op_library_sha256": sha256(library_path),
                "preservation": {"changed_nodes": [x["node"] for x in specs], "excluded_known_nodes": list(exclude_nodes or []),
                                 "all_original_initializers_retained": True,
                                 "all_other_original_nodes_byte_identical": True, "all_21_fp32_state_definitions_unchanged": True},
                "full_stream": {"input": {**write_float(output, "public-input-8k.f32", audio), "sample_rate": 8000},
                                "baseline": baseline_meta, "candidate": candidate_meta, "calls": 65, "frames_per_call": 5,
                                "public_sample_sha256": sha256(sample_path), "host_comparison": "host-stream-comparison.json",
                                "host_comparison_complete": False, "audio_quality_verified": False, "phone_performance_verified": False}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{len(specs)} fixtures and complete candidate staged; checking full recurrent sequences...", flush=True)
    fp32_check, _ = stream_comparison(source, fp32_control, features, len(audio), library_path, threads, require_close=True)
    multi_check, waveforms = stream_comparison(source, candidate, features, len(audio), library_path, threads)
    selected_check, _ = stream_comparison(selected, candidate, features, len(audio), library_path, threads)
    clipping = candidate_clipping(candidate, cases, features, library_path, threads, specs)
    comparison = {"scope": "Host CPU numerical checks on one pinned public clip; no perceptual quality acceptance or physical-phone timing claim",
                  "fp32_graft_control": fp32_check, "multiconv_vs_fp32": multi_check,
                  "multiconv_vs_selected_single_conv": selected_check, "candidate_input_clipping": clipping,
                  "selected_single_conv_model_sha256": SINGLE_CONV_SHA256}
    (output / "host-stream-comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest["full_stream"].update(host_comparison_complete=True,
                                   baseline_output=write_float(output, "full-baseline-output.f32", waveforms[0]),
                                   candidate_cpu_output=write_float(output, "full-candidate-cpu-output.f32", waveforms[1]))
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_DIRECTORY / "model.onnx")
    parser.add_argument("--sample", type=Path, default=ROOT / ".research/upstream/noisy_audio/mic_test.wav")
    parser.add_argument("--custom-op-library", type=Path, default=ROOT / "native/build-windows-frequency-conv/Release/reuse_ort_ops.dll")
    parser.add_argument("--selected-assets", type=Path, default=ROOT / ".research/qnn-conv-hybrid-assets")
    parser.add_argument("--output", type=Path, default=ROOT / ".research/qnn-multiconv-assets")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--exclude-node", action="append", default=[],
                        help="Leave this exact known dense Conv in original FP32; repeat for distinct known targets")
    args = parser.parse_args()
    manifest = prepare(args.model, args.sample, args.custom_op_library, args.output, args.selected_assets, args.threads, args.exclude_node)
    print(json.dumps({"output": str(args.output), "manifest_sha256": sha256(args.output / "manifest.json"),
                      "cases": len(manifest["cases"]), "full_candidate": manifest["full_stream"]["candidate"]}, indent=2))


if __name__ == "__main__": main()
