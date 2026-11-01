# Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
# Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
"""Opt-in FastSoftplus rewrite for NVIDIA NSCLv1 explicit-state RE-USE graphs.

Only proven float32 standard Softplus nodes are replaced. The model weights,
state ABI and graph input/output metadata are preserved; validate recurrent
accuracy with the newly built custom library before using the candidate.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import onnx
from onnx import TensorProto, helper

from optimize_dense_history import validate


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def optimize(source_dir: Path, output_dir: Path):
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    manifest = json.loads((source_dir / "manifest.json").read_text())
    if any(item["name"] == "FastSoftplus" for item in manifest.get("custom_ops", [])):
        raise ValueError("Source manifest already requests FastSoftplus")
    source = source_dir / "model.onnx"
    if sha256(source) != manifest["model_sha256"]:
        raise ValueError("Source model SHA-256 differs from its manifest")
    model = onnx.load(source)
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {value.name: value.type.tensor_type.elem_type for value in
             list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)}
    types.update({value.name: value.data_type for value in inferred.graph.initializer})
    converted = []
    for node in model.graph.node:
        if node.domain not in ("", "ai.onnx") or node.op_type != "Softplus":
            continue
        if len(node.input) != 1 or len(node.output) != 1 or types.get(node.input[0]) != TensorProto.FLOAT:
            raise ValueError(f"Cannot prove a unary float32 Softplus contract: {node.name}")
        if node.attribute:
            raise ValueError(f"Unexpected standard Softplus attributes: {node.name}")
        converted.append(node.name)
        node.domain = "com.reuse"; node.op_type = "FastSoftplus"
    if not converted:
        raise ValueError("No eligible float32 Softplus node found")
    if not any(item.domain == "com.reuse" for item in model.opset_import):
        model.opset_import.append(helper.make_opsetid("com.reuse", 1))
    onnx.checker.check_model(model, full_check=True)
    output_dir.mkdir(parents=True)
    graph = output_dir / "model.onnx"; onnx.save(model, graph)
    manifest = copy.deepcopy(manifest)
    previous = manifest.get("graph_optimizations")
    manifest["graph_optimizations"] = {"name": "fast_softplus_v1", "source_model_sha256": sha256(source),
                                      "replaced_nodes": converted, "previous_optimizations": previous}
    manifest["model_sha256"] = sha256(graph); manifest["model_bytes"] = graph.stat().st_size
    operations = manifest.setdefault("custom_ops", [])
    operations.append({"domain": "com.reuse", "name": "FastSoftplus", "version": 1, "library": "reuse_ort_ops"})
    manifest["validation"] = {"phone_realtime_verified": False, "optimization_parity_verified": False,
                              "scope": "FastSoftplus candidate; see optimization-parity.json for fresh validation"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(source_dir / "LICENSE-NVIDIA.txt", output_dir / "LICENSE-NVIDIA.txt")
    return {"source_model_sha256": sha256(source), "optimized_model_sha256": sha256(graph),
            "replaced_softplus_nodes": len(converted), "nodes": converted}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--custom-op-library", type=Path)
    parser.add_argument("--validation-frames", type=int, default=40)
    parser.add_argument("--speech-pcm", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validation_frames < 2:
        parser.error("At least two recurrent frames are required")
    if not args.validate_only:
        print(json.dumps(optimize(args.source_dir, args.output_dir), indent=2))
    if args.custom_op_library:
        print(json.dumps(validate(args.source_dir, args.output_dir, args.custom_op_library,
                                  args.validation_frames, args.speech_pcm), indent=2))
    elif args.validate_only:
        parser.error("--validate-only requires --custom-op-library")
