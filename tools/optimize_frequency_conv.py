# Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
# Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
"""Opt-in NVIDIA NSCLv1 six-region causal frequency Conv/SiLU/layout fusion."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from optimize_channel_norm import proof_shapes
from optimize_dense_history import Constants, attrs, remove_dead_nodes, validate


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analysis_contract(model):
    """Bridge existing custom layouts only in a private shape/constant proof."""
    analysis = copy.deepcopy(model)
    for node in analysis.graph.node:
        if node.domain == "com.reuse" and node.op_type == "ChannelNormPRelu":
            if len(node.input) != 4 or len(node.output) != 1 or set(attrs(node)) != {"epsilon"}:
                raise ValueError("Invalid ChannelNormPRelu shape contract")
            node.domain = ""; node.op_type = "Transpose"
            del node.input[1:]; del node.attribute[:]
            node.attribute.append(helper.make_attribute("perm", [0, 3, 1, 2]))
    shapes = proof_shapes(analysis)
    # A Slice endpoint may be expressed through Shape(current). Resolve only
    # proven static metadata; never evaluate feature/state data as a constant.
    for node in analysis.graph.node:
        if node.domain not in ("", "ai.onnx") or node.op_type != "Shape":
            continue
        dimensions = shapes.get(node.input[0])
        if dimensions is None or not all(isinstance(x, int) for x in dimensions):
            continue
        a = attrs(node)
        if set(a) - {"start", "end"}:
            raise ValueError("Unexpected Shape metadata attributes")
        selected = dimensions[a.get("start", 0):a.get("end", len(dimensions))]
        node.op_type = "Constant"; del node.input[:]; del node.attribute[:]
        node.attribute.append(helper.make_attribute("value", numpy_helper.from_array(np.asarray(selected, np.int64))))
    return shapes, Constants(analysis)


def optimize_graph(source):
    model = copy.deepcopy(source)
    if len({n.name for n in model.graph.node}) != len(model.graph.node):
        raise ValueError("Graph node names must be unique for strict fusion")
    if any(n.domain == "com.reuse" and n.op_type == "FrequencyConvSilu" for n in model.graph.node):
        raise ValueError("Source already has FrequencyConvSilu")
    shapes, constants = analysis_contract(model)
    named = {n.name: n for n in model.graph.node}
    producers = {out: n for n in model.graph.node for out in n.output}
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    initializers = {v.name: v for v in model.graph.initializer}
    public = {v.name for v in model.graph.output}
    removed = set(); replacements = {}; proofs = []

    def standard(node, kind):
        if node is None or node.domain not in ("", "ai.onnx") or node.op_type != kind or len(node.output) != 1:
            raise ValueError(f"Cannot prove standard {kind} node")
        return node

    def parent(name, kind):
        return standard(producers.get(name), kind)

    def only(name, expected):
        actual = sorted(n.name for n in consumers.get(name, []))
        if actual != sorted(expected):
            raise ValueError(f"Unexpected consumers of {name}: {actual}")

    for layer in range(3):
        for direction in ("forward", "backward"):
            suffix = f"_{layer}" if layer else ""
            conv_name = f"/freq_mamba/{direction}_blocks/conv1d{suffix}/Conv"
            conv = standard(named.get(conv_name), "Conv")
            if len(conv.input) != 3 or attrs(conv) != {"dilations": [1], "group": 256, "kernel_shape": [4], "pads": [3, 3], "strides": [1]}:
                raise ValueError(f"Unexpected frequency Conv attributes: {conv_name}")
            for name, shape in zip(conv.input[1:], ([256, 1, 4], [256])):
                value = initializers.get(name)
                if value is None or value.data_type != TensorProto.FLOAT or list(value.dims) != shape or not np.isfinite(numpy_helper.to_array(value)).all():
                    raise ValueError("Expected fixed finite float32 frequency weights/bias")
            pre = parent(conv.input[0], "Transpose")
            if len(pre.input) != 1 or attrs(pre) != {"perm": [0, 2, 1]}:
                raise ValueError("Unexpected frequency input transpose")
            current = pre.input[0]
            if shapes.get(current) != [5, 81, 256] or shapes.get(pre.output[0]) != [5, 256, 81] or shapes.get(conv.output[0]) != [5, 256, 84]:
                raise ValueError("Cannot prove fixed B5/F81/C256 frequency shapes")
            conv_users = consumers.get(conv.output[0], [])
            trim = standard(conv_users[0] if len(conv_users) == 1 else None, "Slice")
            if trim.attribute or len(trim.input) not in (4, 5):
                raise ValueError("Unexpected frequency Slice contract")
            values = [np.asarray(constants.get(name)) for name in trim.input[1:]]
            if any(v.dtype != np.int64 or v.ndim != 1 or v.size != 1 for v in values):
                raise ValueError("Frequency Slice bounds must be scalar int64 vectors")
            start, end, axis = (int(v[0]) for v in values[:3])
            step = int(values[3][0]) if len(values) == 4 else 1
            if axis != 2 or step != 1 or list(range(84))[slice(start, end, step)] != list(range(81)):
                raise ValueError("Frequency Conv must retain exactly first81 positions")
            trim_users = consumers.get(trim.output[0], [])
            post = standard(trim_users[0] if len(trim_users) == 1 else None, "Transpose")
            if len(post.input) != 1 or attrs(post) != {"perm": [0, 2, 1]} or shapes.get(post.output[0]) != [5, 81, 256]:
                raise ValueError("Unexpected frequency output transpose")
            activation = post.output[0]
            sigmoid = standard(next((n for n in consumers.get(activation, []) if n.op_type == "Sigmoid"), None), "Sigmoid")
            mul = standard(next((n for n in consumers.get(activation, []) if n.op_type == "Mul"), None), "Mul")
            if sigmoid.attribute or list(sigmoid.input) != [activation] or mul.attribute or sorted(mul.input) != sorted([activation, sigmoid.output[0]]):
                raise ValueError("Cannot prove exact frequency SiLU expression")
            only(pre.output[0], [conv.name]); only(conv.output[0], [trim.name])
            only(trim.output[0], [post.name]); only(activation, [sigmoid.name, mul.name]); only(sigmoid.output[0], [mul.name])
            selected = [pre, conv, trim, post, sigmoid, mul]
            names = {n.name for n in selected}
            if len(names) != 6 or names & removed or any(not n for n in names):
                raise ValueError("Overlapping or unnamed frequency fusion")
            if {out for n in selected for out in n.output if out != mul.output[0]} & public:
                raise ValueError("Fusion would remove public intermediate output")
            removed.update(names)
            replacements[pre.name] = helper.make_node("FrequencyConvSilu", [current, conv.input[1], conv.input[2]], list(mul.output),
                name=f"/FrequencyConvSilu_{layer}_{direction}", domain="com.reuse")
            proofs.append({"layer": layer, "direction": direction, "input_shape": [5, 81, 256], "output_shape": [5, 81, 256],
                           "left_padding": 3, "retained_positions": 81, "independent_zero_history": True, "removed_nodes": sorted(names)})
    nodes = []
    for node in model.graph.node:
        if node.name in replacements:
            nodes.append(replacements[node.name])
        if node.name not in removed:
            nodes.append(node)
    del model.graph.node[:]; model.graph.node.extend(nodes)
    remove_dead_nodes(model)
    onnx.checker.check_model(model, full_check=True)
    if list(model.graph.input) != list(source.graph.input) or list(model.graph.output) != list(source.graph.output):
        raise ValueError("Frequency fusion changed public graph metadata")
    return model, proofs


def optimize(source_dir, output_dir):
    if output_dir.exists():
        raise FileExistsError(f"Output exists: {output_dir}")
    manifest = json.loads((source_dir / "manifest.json").read_text())
    source = source_dir / "model.onnx"
    if sha256(source) != manifest["model_sha256"]:
        raise ValueError("Source graph hash mismatch")
    if manifest.get("sample_rate") != 8000 or manifest.get("frames_per_call") != 5 or manifest.get("exit_layers") != 3:
        raise ValueError("Only fixed8k/exit3/chunk5 graphs are supported")
    if any(c["name"] == "FrequencyConvSilu" for c in manifest["custom_ops"]):
        raise ValueError("Source already requests frequency fusion")
    model, proofs = optimize_graph(onnx.load(source))
    output_dir.mkdir(parents=True)
    graph = output_dir / "model.onnx"; onnx.save(model, graph)
    previous = manifest.get("graph_optimizations")
    manifest["graph_optimizations"] = {"name": "frequency_conv_silu_v1", "source_model_sha256": sha256(source), "fused_regions": proofs, "previous_optimizations": previous}
    manifest["custom_ops"].append({"domain": "com.reuse", "name": "FrequencyConvSilu", "version": 1, "library": "reuse_ort_ops"})
    manifest["model_sha256"] = sha256(graph); manifest["model_bytes"] = graph.stat().st_size
    manifest["validation"] = {"phone_realtime_verified": False, "optimization_parity_verified": False,
                              "scope": "Optional FrequencyConvSilu candidate; fresh recurrent/phone parity required"}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    shutil.copy2(source_dir / "LICENSE-NVIDIA.txt", output_dir / "LICENSE-NVIDIA.txt")
    report = {"source_model_sha256": sha256(source), "optimized_model_sha256": sha256(graph), "fused_regions": proofs}
    (output_dir / "optimization.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--custom-op-library", type=Path); parser.add_argument("--speech-pcm", type=Path)
    parser.add_argument("--validation-frames", type=int, default=40); parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.validation_frames < 2:
        parser.error("At least two recurrent hops are required")
    if not args.validate_only:
        print(json.dumps(optimize(args.source_dir, args.output_dir), indent=2))
    if args.custom_op_library:
        print(json.dumps(validate(args.source_dir, args.output_dir, args.custom_op_library, args.validation_frames, args.speech_pcm), indent=2))
    elif args.validate_only:
        parser.error("--validate-only requires --custom-op-library")
