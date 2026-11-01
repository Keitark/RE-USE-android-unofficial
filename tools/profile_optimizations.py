"""Freeze and profile a streaming graph on the host; never phone evidence."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import time

import numpy as np
import onnxruntime as ort


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def benchmark(model_dir: Path, library: Path, output: Path, threads: int,
              warmup: int, frames: int, profile: bool, entries: dict[str, str]):
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if sha256(model_dir / "model.onnx") != manifest["model_sha256"]:
        raise ValueError("Graph SHA256 differs from frozen manifest")
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.enable_profiling = profile
    options.profile_file_prefix = str(output / f"ort-{threads}t")
    if profile:
        options.optimized_model_filepath = str(output / f"optimized-{threads}t.onnx")
    for key, value in entries.items():
        options.add_session_config_entry(key, value)
    options.register_custom_ops_library(str(library.resolve()))
    session = ort.InferenceSession(str(model_dir / "model.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    bins = manifest["fft_size"] // 2 + 1
    frames_per_call = manifest.get("frames_per_call", 1)
    warmup_calls = math.ceil(warmup / frames_per_call)
    measured_calls = math.ceil(frames / frames_per_call)
    feature_shape = (frames_per_call, bins)
    rng = np.random.default_rng(234)
    features = [rng.uniform(0, 2, feature_shape).astype(np.float32),
                rng.uniform(-3, 3, feature_shape).astype(np.float32)]
    states = [[np.zeros(spec["shape"], np.float32) for spec in manifest["states"]]
              for _ in range(2)]
    spectra = [np.empty(feature_shape, np.float32), np.empty(feature_shape, np.float32)]
    bindings = []
    for bank in range(2):
        binding = session.io_binding()
        for name, value in zip(("magnitude", "phase"), features):
            binding.bind_cpu_input(name, value)
        for spec, value in zip(manifest["states"], states[bank]):
            binding.bind_cpu_input(spec["input"], value)
        for name, value in zip(("spectrum_real", "spectrum_imag"), spectra):
            binding.bind_output(name, "cpu", 0, np.float32, value.shape, value.ctypes.data)
        for spec, value in zip(manifest["states"], states[1-bank]):
            binding.bind_output(spec["output"], "cpu", 0, np.float32, value.shape, value.ctypes.data)
        bindings.append(binding)
    times, call_times = [], []
    bank = 0
    for index in range(warmup_calls + measured_calls):
        start = time.perf_counter_ns()
        session.run_with_iobinding(bindings[bank])
        elapsed = (time.perf_counter_ns() - start) / 1e6
        bank = 1 - bank
        if index >= warmup_calls:
            times.append(elapsed / frames_per_call)
            call_times.append(elapsed)
    result = {"threads": threads, "profiled": profile, "warmup_frames": warmup_calls * frames_per_call,
              "measured_frames": measured_calls * frames_per_call, "frames_per_call": frames_per_call,
              "measured_calls": measured_calls, "timing_unit": "amortized_ms_per_20ms_hop",
              "call_times_ms": call_times, "mean_call_ms": float(np.mean(call_times)),
              "p95_call_ms": float(np.percentile(call_times, 95)),
              "session_config": entries, "times_ms": times,
              "mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
              "p95_ms": float(np.percentile(times, 95)), "max_ms": max(times),
              "rtf": float(np.mean(times) / 20)}
    if profile:
        path = Path(session.end_profiling())
        events = json.loads(path.read_text(encoding="utf-8"))
        by_node = defaultdict(list)
        for event in events:
            if event.get("cat") == "Node" and event["name"].endswith("_kernel_time"):
                by_node[event["name"]].append(event)
        operators = defaultdict(float)
        counts = defaultdict(int)
        nodes = []
        for name, all_events in by_node.items():
            selected = all_events[warmup_calls:warmup_calls + measured_calls]
            if not selected:
                continue
            op = selected[0].get("args", {}).get("op_name", "unknown")
            duration = sum(e["dur"] for e in selected)
            operators[op] += duration
            counts[op] += len(selected)
            nodes.append({"name": name, "op": op, "us_per_frame": duration / (measured_calls * frames_per_call),
                          "inputs": selected[0].get("args", {}).get("input_type_shape"),
                          "outputs": selected[0].get("args", {}).get("output_type_shape")})
        total = sum(operators.values())
        result["profile_path"] = str(path)
        result["kernel_us_per_frame"] = total / (measured_calls * frames_per_call)
        result["operators"] = [{"op": op, "percent": 100 * duration / total,
                                "us_per_frame": duration / (measured_calls * frames_per_call),
                                "count_per_frame": counts[op] / (measured_calls * frames_per_call)}
                               for op, duration in sorted(operators.items(), key=lambda p: -p[1])]
        result["top_nodes"] = sorted(nodes, key=lambda n: -n["us_per_frame"])[:50]
    return result


def main(args):
    if args.output.exists():
        raise ValueError("Use a new output directory so historical evidence stays frozen")
    args.output.mkdir(parents=True)
    snapshot = args.output / "snapshot"
    snapshot.mkdir()
    sources = [args.model_dir / "model.onnx", args.model_dir / "manifest.json", args.custom_op_library]
    frozen = {}
    for source in sources:
        before = sha256(source)
        target = snapshot / source.name
        shutil.copy2(source, target)
        if sha256(target) != before or sha256(source) != before:
            raise RuntimeError(f"Input changed while freezing: {source}")
        frozen[source.name] = {"bytes": target.stat().st_size, "sha256": before}
    entries = dict(entry.split("=", 1) for entry in args.session_config)
    report = {"scope": "Host Windows CPU only; not physical-phone performance",
              "platform": platform.platform(), "processor": platform.processor(),
              "onnxruntime": ort.__version__, "frozen_inputs": frozen, "runs": []}
    if args.compare_model_dir:
        candidate = args.output / "candidate"
        candidate.mkdir()
        candidate_frozen = {}
        for filename in ("model.onnx", "manifest.json"):
            source = args.compare_model_dir / filename
            before = sha256(source)
            shutil.copy2(source, candidate / filename)
            if sha256(candidate / filename) != before or sha256(source) != before:
                raise RuntimeError(f"Candidate changed while freezing: {source}")
            candidate_frozen[filename] = {"bytes": source.stat().st_size, "sha256": before}
        report["candidate_frozen_inputs"] = candidate_frozen
        # ABBA order counterbalances CPU frequency and background load. Both
        # graphs use the same frozen native library and fresh prewarmed sessions.
        for threads in args.threads:
            for index, (label, directory) in enumerate((
                    ("baseline", snapshot), ("candidate", candidate),
                    ("candidate", candidate), ("baseline", snapshot))):
                result = benchmark(directory, snapshot / args.custom_op_library.name,
                                   args.output, threads, args.warmup, args.frames, False, entries)
                result.update(graph=label, order=index)
                report["runs"].append(result)
                print(json.dumps({k: v for k, v in result.items() if k not in ("times_ms", "call_times_ms")}), flush=True)
        summaries = []
        for threads in args.threads:
            values = {}
            for label in ("baseline", "candidate"):
                samples = [t for r in report["runs"] if r["threads"] == threads and r["graph"] == label
                           for t in r["times_ms"]]
                values[label] = {"mean_ms": float(np.mean(samples)), "median_ms": float(np.median(samples)),
                                 "p95_ms": float(np.percentile(samples, 95)), "measured_calls": len(samples),
                                 "measured_hops": sum(r["measured_frames"] for r in report["runs"] if r["threads"] == threads and r["graph"] == label)}
            summaries.append({"threads": threads, **values,
                              "mean_reduction_percent": 100 * (1 - values["candidate"]["mean_ms"] / values["baseline"]["mean_ms"]),
                              "median_reduction_percent": 100 * (1 - values["candidate"]["median_ms"] / values["baseline"]["median_ms"])})
        report["comparison"] = summaries
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summaries, indent=2))
        return
    for threads in args.threads:
        result = benchmark(snapshot, snapshot / args.custom_op_library.name, args.output,
                           threads, args.warmup, args.frames, False, entries)
        report["runs"].append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in ("times_ms", "call_times_ms")}), flush=True)
    report["runs"].append(benchmark(snapshot, snapshot / args.custom_op_library.name,
                                    args.output, args.profile_threads, 3, 8, True, entries))
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["runs"][-1]["operators"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/reuse-8k-exit3-native-step"))
    parser.add_argument("--custom-op-library", type=Path, default=Path("native/build-windows/Release/reuse_ort_ops.dll"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--profile-threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--session-config", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--compare-model-dir", type=Path,
                        help="Measure baseline/candidate/candidate/baseline order with one frozen DLL")
    main(parser.parse_args())
