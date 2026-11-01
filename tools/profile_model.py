"""Profile the exported graph; measurements are host CPU, not phone results."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort


def run(model_dir: Path, report_dir: Path, threads: int, profile: bool = False,
        custom_op_library: Path | None = None):
    manifest = json.loads((model_dir / "manifest.json").read_text())
    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.enable_profiling = profile
    options.profile_file_prefix = str(report_dir / f"ort-{threads}t")
    options.optimized_model_filepath = str(report_dir / f"optimized-{threads}t.onnx") if profile else ""
    if custom_op_library is not None:
        options.register_custom_ops_library(str(custom_op_library.resolve()))
    session = ort.InferenceSession(str(model_dir / "model.onnx"), options,
                                   providers=["CPUExecutionProvider"])
    bins = manifest["fft_size"] // 2 + 1
    rng = np.random.default_rng(234)
    features = [rng.uniform(0, 2, (1, bins)).astype(np.float32),
                rng.uniform(-3, 3, (1, bins)).astype(np.float32)]
    states = [[np.zeros(spec["shape"], np.float32) for spec in manifest["states"]] for _ in range(2)]
    spectra = [np.empty((1, bins), np.float32), np.empty((1, bins), np.float32)]
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
    bank = 0
    times = []
    for index in range(8):
        start = time.perf_counter()
        session.run_with_iobinding(bindings[bank])
        elapsed = (time.perf_counter() - start) * 1000
        bank = 1-bank
        if index >= 3:
            times.append(elapsed)
    result = {"threads": threads, "profiled": profile, "times_ms": times,
              "median_ms": float(np.median(times)), "p95_ms": float(np.percentile(times, 95))}
    if profile:
        path = Path(session.end_profiling())
        events = json.loads(path.read_text())
        ops = defaultdict(float)
        nodes = defaultdict(float)
        counts = defaultdict(int)
        for event in events:
            if event.get("cat") != "Node" or not event["name"].endswith("_kernel_time"):
                continue
            op = event.get("args", {}).get("op_name", "unknown")
            ops[op] += event["dur"]
            counts[op] += 1
            nodes[event["name"]] += event["dur"]
        total = sum(ops.values())
        result["profile_path"] = str(path)
        result["operators"] = [{"op": op, "percent": 100*duration/total,
                                "us_per_frame": duration/8, "count_per_frame": counts[op]/8}
                               for op, duration in sorted(ops.items(), key=lambda p: -p[1])]
        result["top_nodes"] = [{"name": name, "us_per_frame": duration/8}
                               for name, duration in sorted(nodes.items(), key=lambda p: -p[1])[:35]]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models/reuse-16k-exit3"))
    parser.add_argument("--output", type=Path, default=Path(".research/profile"))
    parser.add_argument("--custom-op-library", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for count in (1, 2, 4):
        result = run(args.model_dir, args.output, count, custom_op_library=args.custom_op_library)
        results.append(result)
        print(json.dumps(result), flush=True)
    results.append(run(args.model_dir, args.output, 2, True, args.custom_op_library))
    (args.output / "report.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results[-1], indent=2))
