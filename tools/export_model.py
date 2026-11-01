# Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
# Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
"""Export a fixed-shape, stateful, CPU-compatible streaming model for Android."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from fetch_model import ROOT, fetch, sha256
from reuse_port import StreamingReuse
from reuse_chunk import ChunkedStreamingReuse


def enforce_fixed_io_shapes(graph: onnx.ModelProto, input_shapes: dict, output_shapes: dict) -> None:
    """Publish the deployment ABI even when a custom op has no shape schema.

    These exports intentionally have no dynamic axes. The legacy exporter may
    replace custom-op output sizes with symbols, which makes an otherwise
    numerically correct graph unusable by strict native-state allocators.
    Callers verify the exact same shapes against eager and ORT execution.
    """
    for values, expected in ((graph.graph.input, input_shapes), (graph.graph.output, output_shapes)):
        if {value.name for value in values} != set(expected):
            raise RuntimeError("ONNX input/output names differ from the fixed deployment contract")
        for value in values:
            tensor = value.type.tensor_type
            if tensor.elem_type != onnx.TensorProto.FLOAT:
                raise RuntimeError(f"Expected float32 deployment tensor: {value.name}")
            tensor.shape.ClearField("dim")
            for size in expected[value.name]:
                tensor.shape.dim.add().dim_value = int(size)


def export(args: argparse.Namespace) -> dict:
    scan_mode = getattr(args, "scan_mode", "parallel")
    fuse_time_step = getattr(args, "fuse_time_step", False)
    fuse_time_sequence = getattr(args, "fuse_time_sequence", False)
    frames_per_call = getattr(args, "frames_per_call", 1)
    if frames_per_call not in (1, 5):
        raise ValueError("frames_per_call must be 1 or 5")
    if fuse_time_step and scan_mode != "native":
        raise ValueError("--fuse-time-step requires --scan-mode native")
    if fuse_time_sequence and (frames_per_call != 5 or scan_mode != "native" or not fuse_time_step):
        raise ValueError("--fuse-time-sequence requires --frames-per-call 5 --scan-mode native --fuse-time-step")
    custom_op_library = getattr(args, "custom_op_library", None)
    if scan_mode == "native" and (custom_op_library is None or not Path(custom_op_library).is_file()):
        raise ValueError("--scan-mode native requires --custom-op-library pointing to a built reuse_ort_ops library")
    torch.set_num_threads(args.threads)
    lock = fetch(args.source)
    model_type = StreamingReuse if frames_per_call == 1 else ChunkedStreamingReuse
    chunk_kwargs = {} if frames_per_call == 1 else {"frames_per_call": frames_per_call}
    if fuse_time_sequence:
        chunk_kwargs["fuse_time_sequence"] = True
    model = model_type(checkpoint=args.source / "model.safetensors",
                       sample_rate=args.sample_rate, exit_layer=args.exit_layer,
                       scan_mode=scan_mode, fuse_time_step=fuse_time_step, **chunk_kwargs).eval()
    serial_model = (StreamingReuse(model.state_dict(), sample_rate=args.sample_rate,
                                  exit_layer=args.exit_layer, scan_mode=scan_mode,
                                  fuse_time_step=fuse_time_step).eval()
                    if frames_per_call > 1 else None)
    args.output.mkdir(parents=True, exist_ok=True)
    graph = args.output / "model.onnx"
    specs = model.state_specs
    states = model.initial_states()
    rng = torch.Generator().manual_seed(160320)
    features = (torch.rand(frames_per_call, args.sample_rate // 50 + 1, generator=rng),
                (torch.rand(frames_per_call, args.sample_rate // 50 + 1, generator=rng) - .5) * 6)
    input_names = ["magnitude", "phase"] + [s["input"] for s in specs]
    output_names = ["spectrum_real", "spectrum_imag"] + [s["output"] for s in specs]
    feature_shape = [frames_per_call, model.frequency_bins]
    input_shapes = {"magnitude": feature_shape, "phase": feature_shape,
                    **{spec["input"]: spec["shape"] for spec in specs}}
    output_shapes = {"spectrum_real": feature_shape, "spectrum_imag": feature_shape,
                     **{spec["output"]: spec["shape"] for spec in specs}}
    with torch.inference_mode():
        for name, value in zip(output_names, model(*(features + states))):
            if list(value.shape) != output_shapes[name]:
                raise RuntimeError(f"Eager output shape differs from deployment contract: {name}")
    print(f"Exporting streaming graph (scan_mode={scan_mode}, no CUDA)...", flush=True)
    # ONNX has Exp, but no Expm1 in opset 17. The resulting graph contains
    # ordinary Exp/Sub operators; this is an exporter mapping, not a custom
    # operator or runtime extension. Small-amplitude error is checked below.
    def export_expm1(g, value):
        return g.op("Sub", g.op("Exp", value),
                    g.op("Constant", value_t=torch.tensor(1., dtype=torch.float32)))
    torch.onnx.register_custom_op_symbolic("aten::expm1", export_expm1, 17)
    with torch.inference_mode():
        torch.onnx.export(model, features + states, str(graph), input_names=input_names,
                          output_names=output_names, opset_version=17, dynamo=False,
                          do_constant_folding=True,
                          custom_opsets={"com.reuse": 1} if scan_mode == "native" else None)
    checked = onnx.load(str(graph))
    enforce_fixed_io_shapes(checked, input_shapes, output_shapes)
    onnx.checker.check_model(checked, full_check=True)
    onnx.save(checked, str(graph))
    domains = sorted({n.domain for n in checked.graph.node})
    allowed_domains = ("", "ai.onnx", "com.reuse") if scan_mode == "native" else ("", "ai.onnx")
    if any(d not in allowed_domains for d in domains):
        raise RuntimeError(f"Nonportable ONNX operators: {domains}")
    custom_nodes = [node for node in checked.graph.node if node.domain == "com.reuse"]
    if scan_mode == "native":
        expected_counts = {"SelectiveScan": 2 * args.exit_layer}
        if fuse_time_sequence:
            expected_counts["SelectiveSequence"] = args.exit_layer
        elif fuse_time_step:
            expected_counts["SelectiveStep"] = args.exit_layer * frames_per_call
        actual_counts = {name: sum(node.op_type == name for node in custom_nodes) for name in expected_counts}
        if actual_counts != expected_counts or len(custom_nodes) != sum(expected_counts.values()):
            raise RuntimeError(f"Native graph custom nodes differ from expected {expected_counts}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if scan_mode == "native":
        opts.register_custom_ops_library(str(Path(custom_op_library).resolve()))
    session = ort.InferenceSession(str(graph), sess_options=opts, providers=["CPUExecutionProvider"])
    for values, shapes in ((session.get_inputs(), input_shapes), (session.get_outputs(), output_shapes)):
        if {value.name for value in values} != set(shapes):
            raise RuntimeError("ORT metadata input/output names differ from the deployment contract")
        for value in values:
            if value.shape != shapes[value.name] or value.type != "tensor(float)":
                raise RuntimeError(f"ORT metadata must expose an exact static float32 shape: {value.name}: {value.shape}")
    state_np = [t.numpy().copy() for t in states]
    state_pt = states
    max_abs, max_relative, state_abs = 0., 0., 0.
    serial_spectrum_abs, serial_state_abs = 0., 0.
    serial_states = states
    validation_calls = math.ceil(args.validation_frames / frames_per_call)
    timings = []
    # Feed updated state back for every frame: single zero-state comparisons can
    # miss incorrect temporal caches, axis order, and recurrent updates.
    with torch.inference_mode():
        for frame in range(validation_calls):
            mag = torch.rand(features[0].shape, generator=rng) * (1. + frame % 3)
            pha = (torch.rand(features[1].shape, generator=rng) - .5) * 6
            expected = model(mag, pha, *state_pt)
            serial_expected = None
            if serial_model is not None:
                real, imag = [], []
                for index in range(frames_per_call):
                    value = serial_model(mag[index:index+1], pha[index:index+1], *serial_states)
                    real.append(value[0]);imag.append(value[1]);serial_states = tuple(value[2:])
                serial_expected = (torch.cat(real), torch.cat(imag), *serial_states)
                for index, (chunk_value, serial_value) in enumerate(zip(expected, serial_expected)):
                    torch.testing.assert_close(chunk_value, serial_value, rtol=2e-3, atol=3e-4,
                                               msg=f"Chunk vs serial parity, call {frame}, output {index}")
            inputs = dict(zip(input_names, [mag.numpy(), pha.numpy()] + state_np))
            start = time.perf_counter()
            actual = session.run(None, inputs)
            timings.append((time.perf_counter() - start) * 1000)
            for i, (a, e) in enumerate(zip(actual, expected)):
                e = e.numpy()
                if list(a.shape) != output_shapes[output_names[i]]:
                    raise AssertionError(f"Actual ORT output shape differs: {output_names[i]}: {a.shape}")
                err = float(np.max(np.abs(a - e)))
                if i < 2:
                    max_abs = max(max_abs, err)
                    max_relative = max(max_relative, err / max(float(np.max(np.abs(e))), 1e-5))
                else:
                    state_abs = max(state_abs, err)
                np.testing.assert_allclose(a, e, rtol=2e-3, atol=3e-4,
                                           err_msg=f"ONNX parity, frame {frame}, {output_names[i]}")
                if not np.isfinite(a).all():
                    raise AssertionError(f"Nonfinite output {output_names[i]}")
                if serial_expected is not None:
                    serial_value = serial_expected[i].numpy()
                    err = float(np.max(np.abs(a - serial_value)))
                    if i < 2: serial_spectrum_abs = max(serial_spectrum_abs, err)
                    else: serial_state_abs = max(serial_state_abs, err)
                    np.testing.assert_allclose(a, serial_value, rtol=2e-3, atol=3e-4,
                                               err_msg=f"ORT chunk vs serial, call {frame}, {output_names[i]}")
            state_pt = tuple(expected[2:])
            state_np = actual[2:]
    manifest = {
        "format_version": 1,
        "sample_rate": args.sample_rate,
        "fft_size": args.sample_rate // 25,
        "hop_size": args.sample_rate // 50,
        "frames_per_call": frames_per_call,
        "chunk_size": args.sample_rate // 50 * frames_per_call,
        "processing_interval_ms": 20 * frames_per_call,
        "maximum_collection_delay_ms": (frames_per_call - 1) * 20,
        "exit_layers": args.exit_layer,
        "look_ahead_frames": 0,
        "scan_mode": scan_mode,
        "fuse_time_step": fuse_time_step,
        "custom_ops": ([{"domain": "com.reuse", "name": "SelectiveScan", "version": 1,
                         "library": "reuse_ort_ops"}] +
                       ([{"domain": "com.reuse", "name": "SelectiveSequence", "version": 1,
                          "library": "reuse_ort_ops"}] if fuse_time_sequence else
                        [{"domain": "com.reuse", "name": "SelectiveStep", "version": 1,
                          "library": "reuse_ort_ops"}] if fuse_time_step else [])
                       if scan_mode == "native" else []),
        "model_sha256": sha256(graph),
        "model_bytes": graph.stat().st_size,
        "source_repository": lock["repository"],
        "source_revision": lock["revision"],
        "source_checkpoint_sha256": lock["files"]["model.safetensors"],
        "license": lock["license"],
        "commercial_use": False,
        "states": specs,
        "state_bytes": sum(t.numel() * t.element_size() for t in states),
        "input_compression": "log1p(abs(STFT))",
        "window": "periodic_hann",
        "analysis_alignment": "causal_previous_hop_plus_current_hop_zero_padded_start",
        "output_alignment": ("first_process_returns_null_then_one_hop_delayed_flush_final_hop"
                             if frames_per_call == 1 else
                             "collect_chunk_then_first_spectrum_has_no_PCM_one_hop_delayed_synthesis_flush_final_hop"),
        "architecture": model.architecture_manifest(),
        "validation": {
            "frames": validation_calls * frames_per_call,
            "calls": validation_calls,
            "torch_onnx_max_abs_spectrum_error": max_abs,
            "torch_onnx_max_relative_spectrum_error": max_relative,
            "torch_onnx_max_abs_state_error": state_abs,
            "host_inference_ms_median": float(np.median(timings[1:] or timings)),
            "host_ms_per_hop_median": float(np.median(timings[1:] or timings)) / frames_per_call,
            "serial_baseline_parity_verified": serial_model is not None,
            "ort_serial_max_abs_spectrum_error": serial_spectrum_abs if serial_model is not None else None,
            "ort_serial_max_abs_state_error": serial_state_abs if serial_model is not None else None,
            "host_threads": args.threads,
            "phone_realtime_verified": False,
            "fixed_io_metadata_verified": True,
        },
        "versions": {"torch": torch.__version__, "onnx": onnx.__version__, "onnxruntime": ort.__version__},
    }
    if fuse_time_sequence:
        manifest["fuse_time_sequence"] = True
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(ROOT / "licenses/NVIDIA-NSCLv1.txt", args.output / "LICENSE-NVIDIA.txt")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("states", "architecture")}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".research/upstream")
    parser.add_argument("--output", type=Path, default=ROOT / "models/reuse-16k-exit3")
    parser.add_argument("--sample-rate", type=int, choices=(8000, 16000), default=16000)
    parser.add_argument("--exit-layer", type=int, choices=range(3, 13), default=3)
    parser.add_argument("--scan-mode", choices=("parallel", "native"), default="parallel",
                        help="Portable ONNX tree scan, or optional reuse_ort_ops native scan")
    parser.add_argument("--custom-op-library", type=Path,
                        help="Built native host library (.dll/.so) required to parity-check a native export")
    parser.add_argument("--fuse-time-step", action="store_true",
                        help="Also fuse each temporal SSM update with optional native SelectiveStep")
    parser.add_argument("--fuse-time-sequence", action="store_true",
                        help="Fuse five chronological temporal updates into SelectiveSequence")
    parser.add_argument("--frames-per-call", type=int, choices=(1, 5), default=1,
                        help="Vectorize consecutive 20 ms hops; 5 collects up to 80 ms extra input")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--validation-frames", type=int, default=8)
    args = parser.parse_args()
    if args.validation_frames < 2 or args.threads < 1:
        parser.error("At least two validation frames and one CPU thread are required")
    if args.scan_mode == "native" and (args.custom_op_library is None or not args.custom_op_library.is_file()):
        parser.error("--scan-mode native requires an existing --custom-op-library")
    if args.fuse_time_step and args.scan_mode != "native":
        parser.error("--fuse-time-step requires --scan-mode native")
    if args.fuse_time_sequence and (args.frames_per_call != 5 or args.scan_mode != "native"
                                   or not args.fuse_time_step):
        parser.error("--fuse-time-sequence requires --frames-per-call 5 --scan-mode native --fuse-time-step")
    export(args)
