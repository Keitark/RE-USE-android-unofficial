# Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
# Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
"""Compact streaming convolution history without changing weights or state ABI.

This is an optional post-export optimization for the NVIDIA NSCLv1 RE-USE
graphs. It folds constant zero spatial Pad into Conv and can compact a dilated
three-tap temporal history to the exact frame intervals the convolution reads.
The original next-state computation remains unchanged.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto
from onnx.reference import ReferenceEvaluator


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def attrs(node):
    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def set_attr(node, name, value):
    retained = [item for item in node.attribute if item.name != name]
    del node.attribute[:]
    node.attribute.extend(retained)
    node.attribute.append(helper.make_attribute(name, value))


class Constants:
    """Evaluate only a closed, input-free dependency graph for a Pad constant."""
    def __init__(self, model):
        self.model = model
        self.initializers = {item.name: item for item in model.graph.initializer}
        self.producers = {name: node for node in model.graph.node for name in node.output}
        self.values = {}

    def get(self, name):
        if name in self.values:
            return self.values[name]
        if name in self.initializers:
            result = numpy_helper.to_array(self.initializers[name])
        else:
            tensors = set()

            def visit(value):
                if not value or value in tensors:
                    return
                tensors.add(value)
                if value in self.initializers:
                    return
                producer = self.producers.get(value)
                if producer is None or producer.domain not in ("", "ai.onnx"):
                    raise ValueError(f"Not a closed standard-ONNX constant: {value}")
                for dependency in producer.input:
                    visit(dependency)

            visit(name)
            nodes = [copy.deepcopy(node) for node in self.model.graph.node
                     if any(output in tensors for output in node.output)]
            graph = helper.make_graph(nodes, "pad_constant", [], [helper.make_empty_tensor_value_info(name)],
                                      initializer=[copy.deepcopy(item) for key, item in self.initializers.items() if key in tensors])
            model = helper.make_model(graph, opset_imports=list(self.model.opset_import))
            model.ir_version = self.model.ir_version
            result = ReferenceEvaluator(model).run(None, {})[0]
        self.values[name] = np.asarray(result)
        return self.values[name]


def static_shapes(model):
    # Resolve metadata-only Shape expressions and the fixed shape contracts of
    # our three registered operations. Standard ONNX inference deliberately has
    # no schema for these custom ops, and otherwise leaves downstream decoder
    # T/C dimensions unknown. This private copy is never a deployment rewrite.
    analysis = copy.deepcopy(model)
    contracts = {"SelectiveScan": ((0, 0),),
                 "SelectiveStep": ((0, 0), (1, 5)),
                 "SelectiveSequence": ((0, 0), (1, 5))}
    for _ in range(len(analysis.graph.node)+1):
        inferred = onnx.shape_inference.infer_shapes(analysis)
        information = {item.name: item for item in
                       list(inferred.graph.input) + list(inferred.graph.value_info) + list(inferred.graph.output)}
        shapes = {name: [dim.dim_value if dim.HasField("dim_value") else None
                         for dim in item.type.tensor_type.shape.dim]
                  for name, item in information.items() if item.type.tensor_type.HasField("shape")}
        changed = False
        for node in analysis.graph.node:
            if node.op_type == "Shape" and node.domain in ("", "ai.onnx"):
                dimensions = shapes.get(node.input[0])
                if dimensions is not None and all(isinstance(value, int) for value in dimensions):
                    attributes = attrs(node)
                    selected = dimensions[attributes.get("start", 0):attributes.get("end", len(dimensions))]
                    node.op_type = "Constant"
                    del node.input[:]
                    del node.attribute[:]
                    node.attribute.append(helper.make_attribute("value", numpy_helper.from_array(
                        np.asarray(selected, np.int64))))
                    changed = True
            elif node.domain == "com.reuse" and node.op_type in contracts:
                for output_index, input_index in contracts[node.op_type]:
                    if output_index >= len(node.output) or input_index >= len(node.input):
                        continue
                    input_name, output_name = node.input[input_index], node.output[output_index]
                    dimensions = shapes.get(input_name)
                    if (dimensions is None or not all(isinstance(value, int) for value in dimensions) or
                            information[input_name].type.tensor_type.elem_type != TensorProto.FLOAT or
                            shapes.get(output_name) == dimensions):
                        continue
                    item = next((item for item in list(analysis.graph.output) + list(analysis.graph.value_info)
                                 if item.name == output_name), None)
                    declared = helper.make_tensor_value_info(output_name, TensorProto.FLOAT, dimensions)
                    if item is None:
                        analysis.graph.value_info.append(declared)
                    else:
                        item.CopyFrom(declared)
                    changed = True
        # PyTorch expresses reshape/slice parameters through small Shape,
        # Gather, arithmetic and Concat subgraphs. Once Shape is known these
        # are closed integer expressions. Evaluate only those parameter inputs,
        # never model data, and keep all replacements in the proof copy.
        constants = Constants(analysis)
        constant_names = {item.name for item in analysis.graph.initializer}
        parameters = {"Reshape": (1,), "Expand": (1,), "Slice": (1, 2, 3, 4),
                      "Squeeze": (1,), "Unsqueeze": (1,), "Tile": (1,),
                      "ConstantOfShape": (0,), "Split": (1,)}
        for index, node in enumerate(analysis.graph.node):
            if node.domain not in ("", "ai.onnx"):
                continue
            for slot in parameters.get(node.op_type, ()):
                if slot >= len(node.input) or not node.input[slot] or node.input[slot] in constant_names:
                    continue
                try:
                    value = constants.get(node.input[slot])
                except (ValueError, NotImplementedError):
                    continue
                if value.dtype != np.int64 or value.ndim > 1 or value.size > 16:
                    continue
                name = f"__shape_analysis_value_{index}_{slot}"
                if name in constant_names:
                    raise ValueError("Reserved static shape analysis name already exists")
                analysis.graph.initializer.append(numpy_helper.from_array(value, name))
                constant_names.add(name)
                node.input[slot] = name
                changed = True
        if not changed:
            return shapes
    raise ValueError("Static shape analysis did not converge")


def remove_dead_nodes(model):
    needed = {item.name for item in model.graph.output}
    retained = []
    for node in reversed(model.graph.node):
        if any(output in needed for output in node.output):
            retained.append(node)
            needed.update(name for name in node.input if name)
    del model.graph.node[:]
    model.graph.node.extend(reversed(retained))
    initializers = [item for item in model.graph.initializer if item.name in needed]
    del model.graph.initializer[:]
    model.graph.initializer.extend(initializers)
    information = [item for item in model.graph.value_info if item.name in needed]
    del model.graph.value_info[:]
    model.graph.value_info.extend(information)


def optimize_graph(source: onnx.ModelProto, mode: str = "compact") -> tuple[onnx.ModelProto, dict]:
    if mode not in ("pads", "compact"):
        raise ValueError("Optimization mode must be pads or compact")
    model = copy.deepcopy(source)
    producers = {name: node for node in model.graph.node for name in node.output}
    consumers = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    outputs = {item.name for item in model.graph.output}
    constants = Constants(model)
    folded = []
    for node in model.graph.node:
        if node.op_type != "Conv" or node.domain not in ("", "ai.onnx"):
            continue
        pad = producers.get(node.input[0])
        if pad is None or pad.op_type != "Pad" or pad.domain not in ("", "ai.onnx"):
            continue
        if len(consumers.get(pad.output[0], [])) != 1 or pad.output[0] in outputs:
            continue
        attributes = attrs(node)
        if attributes.get("auto_pad", b"NOTSET") not in (b"NOTSET", b""):
            continue
        if (attrs(pad).get("mode", b"constant") != b"constant" or len(pad.input) < 2 or
                len(pad.input) > 3 and pad.input[3]):
            continue
        try:
            padding = constants.get(pad.input[1])
            value = constants.get(pad.input[2]) if len(pad.input) > 2 and pad.input[2] else np.asarray(0.)
        except (ValueError, NotImplementedError):
            continue
        if padding.dtype != np.int64 or padding.shape != (8,) or np.any(padding < 0):
            continue
        if np.any(padding[[0, 1, 4, 5]]) or value.size != 1 or float(value.reshape(-1)[0]) != 0.:
            continue
        existing = attributes.get("pads", [0, 0, 0, 0])
        if len(existing) != 4:
            continue
        combined = [int(existing[index]) + int(padding[axis]) for index, axis in enumerate((2, 3, 6, 7))]
        node.input[0] = pad.input[0]
        set_attr(node, "pads", combined)
        folded.append({"conv": node.name, "pad": pad.name, "pads": combined})

    # With Pad made explicit in Conv, ordinary shape inference proves the
    # fixed current/history lengths. This analysis does not alter the graph.
    shape_model = copy.deepcopy(model)
    for index, node in enumerate(shape_model.graph.node):
        if node.op_type == "Pad" and len(node.input) > 1:
            try:
                padding = constants.get(node.input[1])
            except (ValueError, NotImplementedError):
                continue
            if padding.dtype == np.int64 and padding.ndim == 1:
                name = f"__shape_analysis_pad_{index}"
                shape_model.graph.initializer.append(numpy_helper.from_array(padding, name))
                node.input[1] = name
    shapes = static_shapes(shape_model)
    producers = {name: node for node in model.graph.node for name in node.output}
    initializers = {item.name: item for item in model.graph.initializer}
    state_inputs = {item.name: item for item in model.graph.input}
    additions = {}
    gathered = []
    for node in model.graph.node:
        if mode == "pads":
            break
        attributes = attrs(node)
        if node.op_type != "Conv" or node.domain not in ("", "ai.onnx"):
            continue
        dilation = attributes.get("dilations", [1, 1])
        if len(dilation) != 2 or dilation[0] <= 1 or dilation[1] != 1:
            continue
        history = producers.get(node.input[0])
        if history is None or history.op_type != "Concat" or attrs(history).get("axis") != 2 or len(history.input) != 2:
            continue
        # A cache of 2*d frames followed by T current frames yields T outputs.
        # When d>T, each output reads the three disjoint intervals beginning
        # at 0,d,2*d. Gather those intervals and use dilation T instead.
        cache_name, current_name = history.input
        cache_shape = shapes.get(cache_name, [])
        current_shape = shapes.get(current_name, [])
        history_shape = shapes.get(history.output[0], [])
        weight = initializers.get(node.input[1])
        d = int(dilation[0])
        if cache_name not in state_inputs or len(cache_shape) != 4 or len(current_shape) != 4 or len(history_shape) != 4:
            continue
        frames = current_shape[2]
        if (not isinstance(frames, int) or frames <= 0 or d <= frames or
                cache_shape[2] != 2*d or history_shape[2] != 2*d+frames):
            continue
        if any(cache_shape[axis] != current_shape[axis] or
               cache_shape[axis] != history_shape[axis] or cache_shape[axis] is None
               for axis in (0, 1, 3)):
            continue
        if weight is None or list(weight.dims)[2:] != [3, 3] or attributes.get("strides", [1, 1]) != [1, 1]:
            continue
        padding = attributes.get("pads", [0, 0, 0, 0])
        if len(padding) != 4 or padding[0] or padding[2]:
            continue
        # Selecting the original exact frames leaves kernel arithmetic order,
        # weights, frequency padding and the separate next-state path intact.
        indices_name = node.name + "/compact_time_indices"
        selected_name = node.name + "/compact_time_history"
        if indices_name in initializers or selected_name in producers:
            raise ValueError("Optimizer output names already exist")
        indices = [start+frame for start in (0, d, 2*d) for frame in range(frames)]
        model.graph.initializer.append(numpy_helper.from_array(np.asarray(indices, np.int64), indices_name))
        gather = helper.make_node("Gather", [node.input[0], indices_name], [selected_name],
                                  name=node.name + "/GatherTimeTaps", axis=2)
        additions[node.name] = gather
        node.input[0] = selected_name
        set_attr(node, "dilations", [frames, 1])
        gathered.append({"conv": node.name, "dilation": d, "output_frames": frames,
                         "optimized_dilation": frames, "selected_frames": indices,
                         "original_history_frames": 2*d+frames, "compact_history_frames": 3*frames,
                         "history_shape": history_shape})
    original_nodes = list(model.graph.node)
    del model.graph.node[:]
    for node in original_nodes:
        if node.name in additions:
            model.graph.node.append(additions[node.name])
        model.graph.node.append(node)
    remove_dead_nodes(model)
    onnx.checker.check_model(model, full_check=True)
    return model, {"mode": mode, "folded_spatial_pads": folded, "compacted_temporal_convolutions": gathered,
                   "original_nodes": len(source.graph.node), "optimized_nodes": len(model.graph.node)}


def optimize(source_dir: Path, output_dir: Path, mode: str = "compact") -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    source_graph = source_dir / "model.onnx"
    if sha256(source_graph) != manifest["model_sha256"]:
        raise ValueError("Source graph SHA-256 disagrees with its manifest")
    model, report = optimize_graph(onnx.load(source_graph), mode)
    if mode == "compact" and not report["compacted_temporal_convolutions"]:
        raise ValueError("No provably equivalent streaming temporal convolution was found")
    if not report["folded_spatial_pads"] and not report["compacted_temporal_convolutions"]:
        raise ValueError("No provably equivalent convolution optimization was found")
    output_dir.mkdir(parents=True)
    graph = output_dir / "model.onnx"
    onnx.save(model, graph)
    manifest["model_sha256"] = sha256(graph)
    manifest["model_bytes"] = graph.stat().st_size
    manifest["graph_optimizations"] = {"name": "dense_history_layout_v2", "mode": mode,
                                       "source_model_sha256": sha256(source_graph),
                                       "folded_spatial_pads": len(report["folded_spatial_pads"]),
                                       "compacted_temporal_convolutions": len(report["compacted_temporal_convolutions"]),
                                       "weights_and_state_abi_unchanged": True}
    # Source-graph runtime measurements are not measurements of this rewrite.
    manifest["validation"] = {key: value for key, value in manifest.get("validation", {}).items()
                              if not key.startswith("host_")}
    manifest["validation"]["phone_realtime_verified"] = False
    manifest["validation"]["optimization_parity_verified"] = False
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copy2(source_dir / "LICENSE-NVIDIA.txt", output_dir / "LICENSE-NVIDIA.txt")
    report.update(source_model_sha256=sha256(source_graph), optimized_model_sha256=sha256(graph))
    (output_dir / "optimization.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def validate(source_dir: Path, output_dir: Path, library: Path, frames: int = 40,
             speech_pcm: Path | None = None) -> dict:
    """Feed each model its own recurrent states, comparing every output."""
    import onnxruntime as ort
    from audio_stream import StreamingAudio
    from validate_even_probe import filter95

    manifest = json.loads((source_dir / "manifest.json").read_text(encoding="utf-8"))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.register_custom_ops_library(str(library.resolve()))
    sessions = [ort.InferenceSession(str(directory / "model.onnx"), options, providers=["CPUExecutionProvider"])
                for directory in (source_dir, output_dir)]
    for get in ("get_inputs", "get_outputs"):
        metadata = [[(value.name, value.shape, value.type) for value in getattr(session, get)()] for session in sessions]
        if metadata[0] != metadata[1]:
            raise AssertionError("Optimized graph changed its static deployment ABI")
    input_names = [value.name for value in sessions[0].get_inputs()]
    output_names = [value.name for value in sessions[0].get_outputs()]
    rng = np.random.default_rng(160320)
    bins = manifest["hop_size"] + 1
    frames_per_call = manifest.get("frames_per_call", 1)
    calls = (frames+frames_per_call-1)//frames_per_call
    total_frames = calls*frames_per_call
    datasets = {"random": [(rng.uniform(0, 3, (frames_per_call, bins)).astype(np.float32),
                             rng.uniform(-3.14, 3.14, (frames_per_call, bins)).astype(np.float32))
                            for _ in range(calls)]}
    if speech_pcm is not None:
        audio = np.fromfile(speech_pcm, dtype="<i2").astype(np.float32) / 32768
        if manifest["sample_rate"] == 8000:
            audio = filter95(audio)[1::2]
        frontend = StreamingAudio(manifest["sample_rate"])
        hop = manifest["hop_size"]
        if len(audio) < total_frames * hop:
            raise ValueError("Speech fixture is shorter than the requested validation")
        speech_frames = [frontend.analyze(audio[index*hop:(index+1)*hop]) for index in range(total_frames)]
        datasets["speech"] = [(np.concatenate([value[0] for value in speech_frames[start:start+frames_per_call]]),
                                np.concatenate([value[1] for value in speech_frames[start:start+frames_per_call]]))
                               for start in range(0, total_frames, frames_per_call)]
    results = {}
    for label, features in datasets.items():
        states = [[np.zeros(spec["shape"], np.float32) for spec in manifest["states"]] for _ in sessions]
        max_spectrum = max_state = 0.
        for index, (magnitude, phase) in enumerate(features):
            values = [session.run(None, dict(zip(input_names, [magnitude, phase] + state)))
                      for session, state in zip(sessions, states)]
            for output, expected, actual in zip(output_names, values[0], values[1]):
                np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=3e-4,
                                           err_msg=f"{label}, frame {index}, {output}")
                if not np.isfinite(actual).all():
                    raise AssertionError(f"Nonfinite optimized output: {output}")
                error = float(np.max(np.abs(actual-expected)))
                if output in ("spectrum_real", "spectrum_imag"):
                    max_spectrum = max(max_spectrum, error)
                else:
                    max_state = max(max_state, error)
            states = [value[2:] for value in values]
        results[label] = {"frames": len(features)*frames_per_call, "calls": len(features),
                          "frames_per_call": frames_per_call, "max_abs_spectrum_error": max_spectrum,
                          "max_abs_state_error": max_state, "passed": True}
    report = {"source_model_sha256": sha256(source_dir / "model.onnx"),
              "optimized_model_sha256": sha256(output_dir / "model.onnx"),
              "custom_op_library_sha256": sha256(library), "rtol": 2e-3, "atol": 3e-4,
              "all_output_tensors_checked": len(output_names), "separate_recurrent_states": True,
              "static_io_metadata_identical": True, "datasets": results, "passed": True,
              "phone_realtime_verified": False}
    (output_dir / "optimization-parity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    optimized_manifest_path = output_dir / "manifest.json"
    optimized_manifest = json.loads(optimized_manifest_path.read_text(encoding="utf-8"))
    optimized_manifest["validation"]["optimization_parity_verified"] = True
    optimized_manifest_path.write_text(json.dumps(optimized_manifest, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("pads", "compact"), default="compact",
                        help="Fold zero padding only, or also compact unused temporal history")
    parser.add_argument("--custom-op-library", type=Path, help="Also run recurrent all-output parity with this library")
    parser.add_argument("--validation-frames", type=int, default=40)
    parser.add_argument("--speech-pcm", type=Path, help="Optional actual 16 kHz mono PCM16LE fixture")
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.validation_frames < 2:
        parser.error("At least two recurrent frames are required")
    if not arguments.validate_only:
        print(json.dumps(optimize(arguments.source_dir, arguments.output_dir, arguments.mode), indent=2))
    if arguments.custom_op_library:
        print(json.dumps(validate(arguments.source_dir, arguments.output_dir, arguments.custom_op_library,
                                  arguments.validation_frames, arguments.speech_pcm), indent=2))
    elif arguments.validate_only:
        parser.error("--validate-only requires --custom-op-library")
