# SPDX-License-Identifier: LicenseRef-NVIDIA-NSCLv1
# Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
# Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
"""Prepare a selective QNN encoder experiment; never a production-quality claim.

Only the last dense encoder convolution, its normalization/PReLU, the stride-2
convolution, and its normalization/PReLU are quantized. The rest of the pinned
F950 streaming graph, its 21 FP32 recurrent states, exit, rate and look-ahead
remain unchanged. Calibration and numerical checks use one pinned public
NVIDIA sample; they do not establish general audio quality or phone speed.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from prepare_qnn_probe import MODEL_DIRECTORY, MODEL_SHA256, ROOT, quantization_parameters, quantize_uint8, sha256, verify_hash

REGION_INPUT = "/dense_encoder/dense_block/Concat_9_output_0"
REGION_OUTPUT = "/dense_encoder/dense_conv_2/dense_conv_2.2/PRelu_output_0"
INPUT_SHAPE = [1, 256, 21, 163]
OUTPUT_SHAPE = [1, 64, 5, 81]
REGION_NAMES = [
    "/dense_encoder/dense_block/dense_block.3.1/Conv",
    "/dense_encoder/dense_block/dense_block.3.2/Transpose",
    "/ChannelNormPRelu_4",
    "/dense_encoder/dense_conv_2/dense_conv_2.0/Conv",
    "/dense_encoder/dense_conv_2/dense_conv_2.1/Transpose",
    "/ChannelNormPRelu_5",
]
CASE_ID = "reuse_encoder_tail_u16"


def attributes(node):
    return {item.name: helper.get_attribute_value(item) for item in node.attribute}


def portable_region(source: onnx.ModelProto, input_shape=INPUT_SHAPE) -> onnx.ModelProto:
    """Unfuse both proven channel64 norms and retain all trained parameters.

    PReLU is applied in NHWC with the original slopes reshaped to [64], before
    the return transpose. This is the same channelwise operation as F950's
    NCHW PReLU with slopes [64,1,1], including negative trained slopes.
    """
    by_name = {node.name: node for node in source.graph.node}
    initializers = {item.name: item for item in source.graph.initializer}
    selected = [by_name[name] for name in REGION_NAMES]
    if [node.op_type for node in selected] != ["Conv", "Transpose", "ChannelNormPRelu", "Conv", "Transpose", "ChannelNormPRelu"]:
        raise ValueError("Encoder region operations differ from the fixed contract")
    if selected[0].input[0] != REGION_INPUT or selected[-1].output[0] != REGION_OUTPUT:
        raise ValueError("Encoder region boundaries differ")
    nodes, weights = [], {}
    previous = "input"
    for stage in range(2):
        conv, transpose, norm = selected[stage * 3:stage * 3 + 3]
        if (conv.domain or transpose.domain or norm.domain != "com.reuse"
                or attributes(transpose) != {"perm": [0, 2, 3, 1]}
                or transpose.input[0] != conv.output[0] or norm.input[0] != transpose.output[0]):
            raise ValueError("Encoder Conv/transpose/norm linkage differs")
        if stage and conv.input[0] != selected[2].output[0]:
            raise ValueError("Encoder convolutions are not contiguous")
        for name in list(conv.input[1:]) + list(norm.input[1:]):
            if name not in initializers:
                raise ValueError("Encoder parameter is not a static initializer")
        for name in conv.input[1:]:
            weights[name] = copy.deepcopy(initializers[name])
        gamma, beta, slope = [numpy_helper.to_array(initializers[name]) for name in norm.input[1:]]
        if gamma.shape != (64,) or beta.shape != (64,) or slope.shape != (64, 1, 1):
            raise ValueError("Encoder normalization parameter shape differs")
        for name in norm.input[1:3]:
            weights[name] = copy.deepcopy(initializers[name])
        slope_name = norm.input[3] + "_nhwc"
        weights[slope_name] = numpy_helper.from_array(slope.reshape(64).copy(), slope_name)
        conv_out, nhwc, normalized, activated = [f"stage{stage}_{suffix}" for suffix in ("conv", "nhwc", "norm", "prelu")]
        next_value = "output" if stage == 1 else "stage0_output"
        nodes.extend([
            helper.make_node("Conv", [previous, *conv.input[1:]], [conv_out], name=f"encoder_tail/conv{stage}", **attributes(conv)),
            helper.make_node("Transpose", [conv_out], [nhwc], name=f"encoder_tail/to_nhwc{stage}", perm=[0, 2, 3, 1]),
            helper.make_node("LayerNormalization", [nhwc, norm.input[1], norm.input[2]], [normalized],
                             name=f"encoder_tail/norm{stage}", axis=-1, epsilon=attributes(norm)["epsilon"]),
            helper.make_node("PRelu", [normalized, slope_name], [activated], name=f"encoder_tail/prelu{stage}"),
            helper.make_node("Transpose", [activated], [next_value], name=f"encoder_tail/to_nchw{stage}", perm=[0, 3, 1, 2]),
        ])
        previous = next_value
    graph = helper.make_graph(nodes, "reuse_encoder_tail_fp32",
                              [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
                              [helper.make_empty_tensor_value_info("output")], initializer=list(weights.values()))
    model = helper.make_model(graph, producer_name="RE-USE unofficial selective QNN experiment",
                              opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model


def session(model, library=None, threads=2, optimize=True):
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL if optimize else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    if library is not None:
        options.register_custom_ops_library(str(Path(library).resolve()))
    data = model.SerializeToString() if isinstance(model, onnx.ModelProto) else str(model)
    return ort.InferenceSession(data, options, providers=["CPUExecutionProvider"])


def load_public_audio(path):
    import soundfile as sf
    from scipy.signal import resample_poly
    samples, rate = sf.read(path, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1, dtype=np.float32)
    divisor = math.gcd(rate, 8000)
    return resample_poly(samples, 8000 // divisor, rate // divisor).astype(np.float32)


def feature_calls(audio):
    from audio_stream import StreamingAudio
    frontend = StreamingAudio(8000)
    logical_hops = math.ceil(len(audio) / 160) + 1
    calls = math.ceil(logical_hops / 5)
    padded = np.pad(audio, (0, calls * 800 - len(audio)))
    values = []
    for call in range(calls):
        frames = [frontend.analyze(padded[(call * 5 + i) * 160:(call * 5 + i + 1) * 160]) for i in range(5)]
        values.append((np.concatenate([x[0] for x in frames]), np.concatenate([x[1] for x in frames])))
    return values


def capture_source(source, features, library, threads):
    extended = copy.deepcopy(source)
    original_names = [item.name for item in source.graph.output]
    extended.graph.output.extend([helper.make_tensor_value_info(REGION_INPUT, TensorProto.FLOAT, INPUT_SHAPE),
                                  helper.make_tensor_value_info(REGION_OUTPUT, TensorProto.FLOAT, OUTPUT_SHAPE)])
    runtime = session(extended, library, threads)
    inputs = runtime.get_inputs()
    states = [np.zeros(item.shape, np.float32) for item in inputs[2:]]
    captured_inputs, captured_outputs = [], []
    for magnitude, phase in features:
        values = runtime.run(original_names + [REGION_INPUT, REGION_OUTPUT],
                             dict(zip([x.name for x in inputs], [magnitude, phase] + states)))
        states = values[2:len(original_names)]
        captured_inputs.append(values[-2].copy())
        captured_outputs.append(values[-1].copy())
    return captured_inputs, captured_outputs


def calibrate_region(model, inputs, expected):
    extended = copy.deepcopy(model)
    names = [item.output[0] for item in model.graph.node]
    existing = {item.name for item in extended.graph.output}
    information = {item.name: item for item in list(model.graph.value_info) + list(model.graph.output)}
    for name in names:
        if name not in existing:
            extended.graph.output.append(copy.deepcopy(information[name]))
    runtime = session(extended, threads=1, optimize=False)
    ranges = {name: [math.inf, -math.inf] for name in ["input", *names]}
    max_error, squares, count = 0.0, 0.0, 0
    for value, reference in zip(inputs, expected):
        values = runtime.run(names, {"input": value})
        for name, array in zip(["input", *names], [value, *values]):
            if not np.isfinite(array).all():
                raise ValueError("Nonfinite region calibration values")
            ranges[name][0] = min(ranges[name][0], float(array.min()))
            ranges[name][1] = max(ranges[name][1], float(array.max()))
        result = values[names.index("output")]
        np.testing.assert_allclose(result, reference, rtol=1e-4, atol=3e-5,
                                   err_msg="Portable FP32 encoder tail differs from original F950")
        delta = result.astype(np.float64) - reference
        max_error = max(max_error, float(np.abs(delta).max()))
        squares += float(np.sum(delta ** 2)); count += delta.size
    return ranges, {"max_abs": max_error, "rmse": math.sqrt(squares / count), "calls": len(inputs),
                    "absolute_tolerance": 3e-5, "relative_tolerance": 1e-4, "passed": True}


def activation_parameters(minimum, maximum, dtype):
    if dtype == "uint8":
        return quantization_parameters(minimum, maximum)
    if dtype != "uint16":
        raise ValueError("Activation dtype must be uint8 or uint16")
    if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
        raise ValueError("Quantization range must be finite and ordered")
    minimum, maximum = min(float(minimum), 0.0), max(float(maximum), 0.0)
    scale = np.float32((maximum - minimum) / 65535.0) if maximum > minimum else np.float32(1.0)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Quantization scale must be finite and positive")
    zero_point = int(np.clip(np.rint(-minimum / float(scale)), 0, 65535))
    return {"dtype": "uint16", "scale": float(scale), "zero_point": zero_point,
            "observed_min": minimum, "observed_max": maximum, "per_tensor": True}


def quantized_region(fp32, ranges, activation_dtype="uint16"):
    """Norm-aware A8/A16-W8 QDQ, int32 Conv/LN bias at A_scale*W_scale.

    U16 activations use the tested com.microsoft Q/DQ v1 operators while the
    standard opset stays 17. This avoids changing F950's existing operator
    semantics merely to obtain standard-opset21 UINT16 Q/DQ support.
    """
    if activation_dtype not in ("uint8", "uint16"):
        raise ValueError("Activation dtype must be uint8 or uint16")
    activation_numpy_type = np.uint16 if activation_dtype == "uint16" else np.uint8
    activation_tensor_type = TensorProto.UINT16 if activation_dtype == "uint16" else TensorProto.UINT8
    activation_domain = "com.microsoft" if activation_dtype == "uint16" else ""
    weights = {item.name: numpy_helper.to_array(item) for item in fp32.graph.initializer}
    nodes, initializers, acts, parameters = [], [], {}, {}
    shapes = {item.name: [d.dim_value for d in item.type.tensor_type.shape.dim]
              for item in list(fp32.graph.input) + list(fp32.graph.value_info) + list(fp32.graph.output)}
    information = {}
    index = 0

    def constants(prefix, q, zero_dtype=np.uint8):
        initializers.extend([numpy_helper.from_array(np.array(q["scale"], np.float32), prefix + "_scale"),
                             numpy_helper.from_array(np.array(q["zero_point"], zero_dtype), prefix + "_zero")])

    def activation(name, raw, q=None):
        nonlocal index
        prefix = f"encoder_q_act_{index}"; index += 1
        q = dict(q or activation_parameters(*ranges[name], activation_dtype))
        constants(prefix, q, activation_numpy_type)
        dq_name = "output" if name == "output" else prefix + "_dq"
        quant_name = prefix + ("_u16" if activation_dtype == "uint16" else "_u8")
        domain_options = {"domain": activation_domain} if activation_domain else {}
        nodes.extend([
            helper.make_node("QuantizeLinear", [raw, prefix + "_scale", prefix + "_zero"], [quant_name],
                             name=prefix + "_q", **domain_options),
            helper.make_node("DequantizeLinear", [quant_name, prefix + "_scale", prefix + "_zero"], [dq_name],
                             name=prefix + "_dequantize", **domain_options)])
        information[quant_name] = helper.make_tensor_value_info(quant_name, activation_tensor_type, shapes[name])
        for float_name in (raw, dq_name):
            if float_name not in ("input", "output"):
                information[float_name] = helper.make_tensor_value_info(float_name, TensorProto.FLOAT, shapes[name])
        acts[name], parameters[name] = dq_name, q
        return dq_name

    def weight(name, prefix):
        value = weights[name]
        q = quantization_parameters(float(value.min()), float(value.max()))
        constants(prefix, q)
        initializers.append(numpy_helper.from_array(quantize_uint8(value, q), prefix + "_u8"))
        nodes.append(helper.make_node("DequantizeLinear", [prefix + "_u8", prefix + "_scale", prefix + "_zero"],
                                      [prefix + "_dq"], name=prefix + "_dequantize"))
        parameters[name] = q
        return prefix + "_dq", q

    def bias(name, prefix, a_q, w_q):
        scale = np.float32(a_q["scale"]) * np.float32(w_q["scale"])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid encoder bias scale")
        values = np.rint(weights[name].astype(np.float64) / float(scale))
        if np.any(values < np.iinfo(np.int32).min) or np.any(values > np.iinfo(np.int32).max):
            raise ValueError("Encoder bias overflows int32")
        q = {"dtype": "int32", "scale": float(scale), "zero_point": 0, "per_tensor": True}
        constants(prefix, q, np.int32)
        initializers.append(numpy_helper.from_array(values.astype(np.int32), prefix + "_i32"))
        nodes.append(helper.make_node("DequantizeLinear", [prefix + "_i32", prefix + "_scale", prefix + "_zero"],
                                      [prefix + "_dq"], name=prefix + "_dequantize"))
        parameters[name] = q
        return prefix + "_dq"

    activation("input", "input")
    for i, node in enumerate(fp32.graph.node):
        prefix = f"encoder_q_node_{i}"
        data_input = acts[node.input[0]]
        inputs = [data_input]
        if node.op_type in ("Conv", "LayerNormalization", "PRelu"):
            w_name, w_q = weight(node.input[1], prefix + "_weight")
            inputs.append(w_name)
            if len(node.input) == 3:
                inputs.append(bias(node.input[2], prefix + "_bias", parameters[node.input[0]], w_q))
        elif node.op_type != "Transpose":
            raise ValueError("Unexpected encoder region operation")
        raw = prefix + "_raw"
        nodes.append(helper.make_node(node.op_type, inputs, [raw], name=prefix, **attributes(node)))
        propagated = parameters[node.input[0]] if node.op_type == "Transpose" else None
        activation(node.output[0], raw, propagated)
    graph = helper.make_graph(nodes, "reuse_encoder_tail_" + ("u16" if activation_dtype == "uint16" else "u8") + "_qdq",
                              copy.deepcopy(list(fp32.graph.input)), copy.deepcopy(list(fp32.graph.output)),
                              initializer=initializers, value_info=list(information.values()) if activation_domain else [])
    imports = [helper.make_opsetid("", 17)]
    if activation_domain:
        imports.append(helper.make_opsetid(activation_domain, 1))
    model = helper.make_model(graph, producer_name="RE-USE unofficial selective QNN experiment",
                              opset_imports=imports, ir_version=10)
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model, full_check=True)
    return model, parameters


def graft_region(source, region):
    """Replace exactly six encoder nodes, preserving the public stateful ABI."""
    model = copy.deepcopy(source)
    imports = {item.domain: item.version for item in model.opset_import}
    for item in region.opset_import:
        if item.domain in imports and imports[item.domain] != item.version:
            raise ValueError("Region cannot change an existing full-model operator-set version")
        if item.domain not in imports:
            model.opset_import.append(copy.deepcopy(item))
            imports[item.domain] = item.version
    indices = [i for i, node in enumerate(model.graph.node) if node.name in REGION_NAMES]
    if len(indices) != 6 or indices != list(range(indices[0], indices[0] + 6)):
        raise ValueError("Source encoder region is not the expected contiguous six-node region")
    region = copy.deepcopy(region)
    rename = {"input": REGION_INPUT, "output": REGION_OUTPUT}
    for node in region.graph.node:
        for name in list(node.input) + list(node.output):
            if name not in rename:
                rename[name] = "/qnn_encoder_tail/" + name
    for item in region.graph.initializer:
        item.name = rename[item.name]
    for node in region.graph.node:
        node.name = "/qnn_encoder_tail/" + node.name
        for i, name in enumerate(node.input): node.input[i] = rename[name]
        for i, name in enumerate(node.output): node.output[i] = rename[name]
    retained = list(model.graph.node[:indices[0]]) + list(region.graph.node) + list(model.graph.node[indices[-1] + 1:])
    del model.graph.node[:]; model.graph.node.extend(retained)
    model.graph.initializer.extend(region.graph.initializer)
    used = {name for node in model.graph.node for name in node.input}
    kept = [item for item in model.graph.initializer if item.name in used]
    del model.graph.initializer[:]; model.graph.initializer.extend(kept)
    # Drop stale metadata for removed tensors; public IO is copied verbatim.
    produced = {name for node in model.graph.node for name in node.output}
    kept_info = [item for item in model.graph.value_info if item.name in produced]
    del model.graph.value_info[:]; model.graph.value_info.extend(kept_info)
    # The CPU custom op before this region has no standard ONNX shape schema.
    # Publish all independently proven region ranks/types, including its cut
    # input and output, so QNN capability inspection sees static dimensions.
    public = {item.name for item in list(model.graph.input) + list(model.graph.output)}
    information = {item.name: item for item in model.graph.value_info}
    for item in list(region.graph.input) + list(region.graph.value_info) + list(region.graph.output):
        value = copy.deepcopy(item)
        value.name = rename[value.name]
        if value.name in public:
            continue
        if value.name in information:
            information[value.name].CopyFrom(value)
        else:
            model.graph.value_info.append(value)
            information[value.name] = model.graph.value_info[-1]
    if list(model.graph.input) != list(source.graph.input) or list(model.graph.output) != list(source.graph.output):
        raise AssertionError("Encoder graft changed public inputs/outputs")
    if len({node.name for node in model.graph.node}) != len(model.graph.node):
        raise ValueError("Encoder graft has duplicate node names")
    onnx.checker.check_model(model, full_check=True)
    return model


def metric(reference, actual):
    ref, value = np.asarray(reference, np.float64), np.asarray(actual, np.float64)
    if ref.shape != value.shape or not np.isfinite(ref).all() or not np.isfinite(value).all():
        raise ValueError("Numerical comparison shape or finiteness failure")
    delta = value - ref
    rmse = float(np.sqrt(np.mean(delta ** 2)))
    rms = float(np.sqrt(np.mean(ref ** 2)))
    return {"max_abs": float(np.max(np.abs(delta))), "rmse": rmse, "reference_rms": rms,
            "normalized_rmse": rmse / rms if rms else None,
            "snr_db": float(20 * np.log10(rms / rmse)) if rms and rmse else None}


def stream_comparison(source, candidate, features, samples, library, threads=2, require_close=False):
    from audio_stream import StreamingAudio
    runtimes = [session(model, library, threads) for model in (source, candidate)]
    names = [x.name for x in runtimes[0].get_inputs()]
    if any([(x.name, x.shape, x.type) for x in runtime.get_inputs()] !=
           [(x.name, x.shape, x.type) for x in runtimes[0].get_inputs()] for runtime in runtimes):
        raise ValueError("Candidate input ABI differs")
    states = [[np.zeros(x.shape, np.float32) for x in runtime.get_inputs()[2:]] for runtime in runtimes]
    synthesis = [StreamingAudio(8000), StreamingAudio(8000)]
    blocks = [[], []]
    state_max, spectrum_max = 0.0, 0.0
    per_state = {name: 0.0 for name in names[2:]}
    for call, (magnitude, phase) in enumerate(features):
        values = [runtime.run(None, dict(zip(names, [magnitude, phase] + state)))
                  for runtime, state in zip(runtimes, states)]
        for index, (a, b) in enumerate(zip(*values)):
            if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError("Full model has nonfinite/mismatched outputs")
            error = float(np.max(np.abs(a.astype(np.float64) - b)))
            if index < 2: spectrum_max = max(spectrum_max, error)
            else:
                state_max = max(state_max, error)
                per_state[names[index]] = max(per_state[names[index]], error)
            if require_close:
                np.testing.assert_allclose(b, a, rtol=2e-3, atol=3e-4,
                                           err_msg=f"FP32 region graft parity call {call}, output {index}")
        states = [item[2:] for item in values]
        for run, (real, imag, *_) in enumerate(values):
            for frame in range(5):
                block = synthesis[run].synthesize(real[frame], imag[frame])
                if block is not None: blocks[run].append(block)
    audio = [np.concatenate(item)[:samples] for item in blocks]
    if any(len(item) != samples for item in audio):
        raise ValueError("Full-model synthesis did not return all source samples")
    return {"calls": len(features), "samples": samples, "independent_state_feedback": True,
            "all_21_states_checked": len(states[0]) == 21, "all_values_finite": True,
            "spectrum_max_abs": spectrum_max, "state_max_abs": state_max,
            "per_state_max_abs": per_state, "waveform": metric(*audio),
            "require_close": require_close, "audio_quality_verified": False}, audio


def write_float(output, name, array):
    path = output / name
    np.asarray(array, dtype="<f4").tofile(path)
    return {"file": name, "sha256": sha256(path), "shape": list(array.shape)}


def write_full_model(output, directory, model, source_manifest, license_data, scope):
    target = output / directory
    target.mkdir()
    path = target / "model.onnx"
    onnx.save(model, path)
    manifest = copy.deepcopy(source_manifest)
    manifest["model_sha256"], manifest["model_bytes"] = sha256(path), path.stat().st_size
    manifest["graph_optimizations"] = {"name": "qnn_encoder_tail_experiment", "source_model_sha256": MODEL_SHA256,
                                       "scope": scope, "public_state_abi_unchanged": True,
                                       "previous_optimizations": manifest.get("graph_optimizations")}
    manifest["validation"] = {"phone_realtime_verified": False, "audio_quality_verified": False,
                              "scope": "Experimental selective encoder graph; see host-stream-comparison.json"}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (target / "LICENSE-NVIDIA.txt").write_bytes(license_data)
    return {"model": directory + "/model.onnx", "manifest": directory + "/manifest.json",
            "model_sha256": sha256(path), "manifest_sha256": sha256(target / "manifest.json"),
            "license": directory + "/LICENSE-NVIDIA.txt", "license_sha256": sha256(target / "LICENSE-NVIDIA.txt")}


def prepare(model_path, sample_path, library_path, output, threads=2, activation_dtype="uint16"):
    lock = json.loads((ROOT / "tools/source-lock.json").read_text(encoding="utf-8"))
    verify_hash(model_path, MODEL_SHA256, "source graph")
    verify_hash(sample_path, lock["files"]["noisy_audio/mic_test.wav"], "public NVIDIA sample")
    if activation_dtype not in ("uint8", "uint16"):
        raise ValueError("Activation dtype must be uint8 or uint16")
    case_id = "reuse_encoder_tail_u16" if activation_dtype == "uint16" else "reuse_encoder_tail_u8"
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh or empty encoder output directory")
    if not library_path.is_file() or threads < 1:
        raise ValueError("An existing host custom-op library and positive thread count are required")
    license_data = (model_path.parent / "LICENSE-NVIDIA.txt").read_bytes()
    if not license_data.strip(): raise ValueError("Missing complete NVIDIA license")
    source = onnx.load(model_path)
    source_manifest = json.loads((model_path.parent / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest["model_sha256"] != MODEL_SHA256:
        raise ValueError("Source model manifest differs from pinned graph")
    audio = load_public_audio(sample_path)
    features = feature_calls(audio)
    region = portable_region(source)
    output.mkdir(parents=True, exist_ok=True)
    fp32_path = output / (case_id + ".fp32.onnx")
    onnx.save(region, fp32_path)
    print("Capturing chronological encoder inputs from the verified public sample...", flush=True)
    captured_inputs, captured_outputs = capture_source(source, features, library_path, threads)
    ranges, parity = calibrate_region(region, captured_inputs, captured_outputs)
    (output / "region-fp32-parity.json").write_text(json.dumps(parity, indent=2) + "\n", encoding="utf-8")
    print("Portable FP32 region parity passed; preparing norm-aware " + activation_dtype + " graph...", flush=True)
    quantized, quantization = quantized_region(region, ranges, activation_dtype)
    qdq_path = output / (case_id + ".onnx")
    onnx.save(quantized, qdq_path)
    qruntime = session(quantized, threads=1, optimize=False)
    fruntime = session(region, threads=1, optimize=False)
    chosen = min(12, len(captured_inputs) - 1)
    value = captured_inputs[chosen]
    qvalue = qruntime.run(["output"], {"input": value})[0]
    fvalue = fruntime.run(["output"], {"input": value})[0]
    input_meta = {"name": "input", **write_float(output, case_id + ".input.f32", value)}
    qoutput = {"name": "output", **write_float(output, case_id + ".output.f32", qvalue)}
    foutput = {"name": "output", **write_float(output, case_id + ".fp32.output.f32", fvalue)}
    case = {"id": case_id, "title": "RE-USE grouped encoder tail (two Conv + LayerNorm/PReLU, " + activation_dtype + " activations)",
            "model": qdq_path.name, "model_sha256": sha256(qdq_path), "input": input_meta, "output": qoutput,
            "max_abs_error": max(1e-5, 2.1 * quantization["output"]["scale"]),
            "relative_tolerance": 0, "quantization": quantization, "activation_dtype": activation_dtype,
            "weight_dtype": "uint8", "bias_dtype": "int32",
            "fp32_error": metric(fvalue, qvalue), "source_fp32_parity": parity,
            "cpu_fp32": {"model": fp32_path.name, "model_sha256": sha256(fp32_path), "output": foutput,
                         "max_abs_error": 1e-4, "relative_tolerance": 1e-4},
            "source_provenance": {"source_graph_sha256": MODEL_SHA256, "source_repository": lock["repository"],
                                  "source_revision": lock["revision"], "source_nodes": REGION_NAMES,
                                  "public_sample_sha256": sha256(sample_path), "calibration_calls": len(features),
                                  "test_call_index": chosen, "license": "NVIDIA NSCLv1", "commercial_use": False}}
    manifest = {"schema_version": 2, "scope": "Experimental isolated encoder region and selective full-model candidate; not general audio-quality calibration.",
                "fixture_format": "IEEE754 float32 little-endian, contiguous row-major", "cases": [case],
                "model_derived_license": "LICENSE-NVIDIA.txt", "host_custom_op_library_sha256": sha256(library_path)}
    (output / "LICENSE-NVIDIA.txt").write_bytes(license_data)
    # Publish the isolated case before the slower complete streaming checks.
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    candidate = graft_region(source, quantized)
    fp32_graft = graft_region(source, region)
    baseline_meta = write_full_model(output, "full-baseline", source, source_manifest, license_data, "Pinned full FP32 reference")
    candidate_meta = write_full_model(output, "full-candidate", candidate, source_manifest, license_data,
                                      "Only encoder tail QDQ with " + activation_dtype + " activations and uint8 weights; remaining operations and 21 states FP32")
    print("Grouped graph and full candidate written; checking complete chronological state feedback...", flush=True)
    fp32_check, _ = stream_comparison(source, fp32_graft, features, len(audio), library_path, threads, require_close=True)
    qdq_check, waveforms = stream_comparison(source, candidate, features, len(audio), library_path, threads)
    comparison = {"scope": "Host CPU numerical comparison on one public clip; not physical-phone speed or general audio quality.",
                  "fp32_region_graft": fp32_check, "qdq_candidate": qdq_check}
    (output / "host-stream-comparison.json").write_text(json.dumps(comparison, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest["full_stream"] = {
        "input": {**write_float(output, "public-input-8k.f32", audio), "sample_rate": 8000},
        "features": {"magnitude": write_float(output, "public-magnitude.f32", np.stack([x[0] for x in features])),
                     "phase": write_float(output, "public-phase.f32", np.stack([x[1] for x in features]))},
        "baseline": baseline_meta, "candidate": candidate_meta, "calls": len(features), "frames_per_call": 5,
        "baseline_output": write_float(output, "full-baseline-output.f32", waveforms[0]),
        "candidate_cpu_output": write_float(output, "full-candidate-cpu-output.f32", waveforms[1]),
        "host_comparison": "host-stream-comparison.json", "public_sample_sha256": sha256(sample_path),
        "audio_quality_verified": False, "phone_performance_verified": False}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_DIRECTORY / "model.onnx")
    parser.add_argument("--sample", type=Path, default=ROOT / ".research/upstream/noisy_audio/mic_test.wav")
    parser.add_argument("--custom-op-library", type=Path, default=ROOT / "native/build-windows-frequency-conv/Release/reuse_ort_ops.dll")
    parser.add_argument("--output", type=Path, default=ROOT / ".research/qnn-encoder16-assets")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--activation-dtype", choices=("uint8", "uint16"), default="uint16")
    args = parser.parse_args()
    result = prepare(args.model, args.sample, args.custom_op_library, args.output, args.threads, args.activation_dtype)
    print(json.dumps({"output": str(args.output), "manifest_sha256": sha256(args.output / "manifest.json"),
                      "case": result["cases"][0]["id"], "full_candidate": result["full_stream"]["candidate"]}, indent=2))


if __name__ == "__main__": main()
