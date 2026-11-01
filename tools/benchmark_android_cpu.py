"""Counterbalanced native ARM model diagnostic; never paced-audio acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(args):
    if args.output.exists():
        raise ValueError("Use a fresh output directory to preserve evidence")
    args.output.mkdir(parents=True)
    frozen = args.output / "snapshot"
    frozen.mkdir()
    files = {}

    def freeze(source, name):
        before = sha(source)
        target = frozen / name
        shutil.copy2(source, target)
        if sha(target) != before or sha(source) != before:
            raise RuntimeError("Input changed while freezing")
        files[name] = {"sha256": before, "bytes": target.stat().st_size}
        return target

    baseline = json.loads((args.model_dir / "manifest.json").read_text())
    candidate_dir = args.candidate_model_dir or args.model_dir
    candidate = json.loads((candidate_dir / "manifest.json").read_text())
    for path, manifest in ((args.model_dir, baseline), (candidate_dir, candidate)):
        if sha(path / "model.onnx") != manifest["model_sha256"]:
            raise ValueError("Graph hash differs from manifest")
    for key in ("fft_size", "frames_per_call", "states"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError("Comparison requires identical external state/features")
    freeze(args.runner, "runner")
    freeze(args.ort, "libonnxruntime.so")
    freeze(args.baseline_library, "baseline.so")
    freeze(args.candidate_library, "candidate.so")
    freeze(args.model_dir / "model.onnx", "baseline.onnx")
    freeze(candidate_dir / "model.onnx", "candidate.onnx")
    freeze(args.model_dir / "manifest.json", "baseline-manifest.json")
    freeze(candidate_dir / "manifest.json", "candidate-manifest.json")
    # Revalidate the pairs actually timed, not only their live producer inputs.
    baseline=json.loads((frozen/"baseline-manifest.json").read_text())
    candidate=json.loads((frozen/"candidate-manifest.json").read_text())
    for label, manifest in (("baseline",baseline),("candidate",candidate)):
        if files[label+".onnx"]["sha256"]!=manifest["model_sha256"]:
            raise RuntimeError("Frozen graph and manifest are from different revisions")
    for key in ("fft_size","frames_per_call","states"):
        if baseline.get(key)!=candidate.get(key):
            raise RuntimeError("Frozen model contracts differ")
    shape = (baseline.get("frames_per_call", 1), baseline["fft_size"] // 2 + 1)
    rng = np.random.default_rng(234)
    features = np.concatenate([rng.uniform(0, 2, shape).astype(np.float32).ravel(),
                               rng.uniform(-3, 3, shape).astype(np.float32).ravel()])
    (frozen / "features.f32").write_bytes(features.tobytes())
    files["features.f32"] = {"sha256": sha(frozen / "features.f32"), "bytes": features.nbytes}
    prefix = [str(args.adb), "-s", args.serial]

    def adb(*parts):
        return subprocess.run(prefix + list(parts), check=True, capture_output=True, timeout=120)

    def screen_state():
        power = adb("shell", "dumpsys", "power").stdout.decode()
        policy = adb("shell", "dumpsys", "window", "policy").stdout.decode()
        wake = re.search(r"\bmWakefulness=(\w+)", power)
        keyguard = re.search(r"KeyguardServiceDelegate[\s\S]{0,2500}?\bshowing=(true|false)", policy)
        state = {"wakefulness": wake.group(1) if wake else "unknown",
                 "keyguard_showing": keyguard.group(1) == "true" if keyguard else None}
        if state["wakefulness"] != "Awake" or state["keyguard_showing"] is not False:
            raise RuntimeError("Unlock the phone and keep the screen on before CPU timing")
        return state

    before = screen_state()
    if adb("shell", "getprop", "ro.kernel.qemu").stdout.decode().strip() == "1":
        raise RuntimeError("Use a physical phone for ARM CPU performance comparisons")
    if adb("shell", "getprop", "ro.product.cpu.abi").stdout.decode().strip() != "arm64-v8a":
        raise RuntimeError("The opcode comparison requires native ARM64")
    (args.output/"thermal-before.txt").write_bytes(adb("shell","dumpsys","thermalservice").stdout)
    suffix = re.sub(r"[^a-zA-Z0-9_-]", "_", args.output.name)
    remote = "/data/local/tmp/reuse-opcode-" + suffix
    adb("shell", "mkdir", "-p", remote)
    for item in frozen.iterdir():
        if item.name.endswith(".json"):
            continue
        adb("push", str(item), remote + "/" + item.name)
    adb("shell", "chmod", "755", remote + "/runner")
    runs = []
    order=("candidate","baseline","baseline","candidate") if args.reverse_order else ("baseline","candidate","candidate","baseline")
    for index, label in enumerate(order):
        run_screen_before=screen_state()
        battery_before=adb("shell","dumpsys","battery").stdout.decode()
        result = adb("shell", remote + "/runner", remote + "/" + label + ".onnx",
                     remote + "/libonnxruntime.so", remote + "/" + label + ".so",
                     remote + "/features.f32", str(args.threads), str(args.warmup_calls),
                     str(args.calls), "-", remote + "/" + label + "-outputs" if index < 2 else "-")
        data = json.loads(result.stdout)
        battery_after=adb("shell","dumpsys","battery").stdout.decode()
        temperature=lambda text: float(re.search(r"\btemperature:\s*(-?\d+)",text).group(1))/10 if re.search(r"\btemperature:\s*(-?\d+)",text) else None
        data.update(label=label, order=index,screen_before=run_screen_before,screen_after=screen_state(),
                    battery_temperature_c_before=temperature(battery_before),battery_temperature_c_after=temperature(battery_after))
        runs.append(data)
        (args.output / f"run-{index}-{label}.json").write_text(json.dumps(data, indent=2) + "\n")
        if index < 2:
            adb("pull", remote + "/" + label + "-outputs.f32", str(args.output / (label + "-outputs.f32")))
        print(json.dumps({"order": index, "graph": label,
                          "mean_call_ms": float(np.mean(data["wall_ms"])),
                          "mean_process_cpu_ms": float(np.mean(data["process_cpu_ms"]))}), flush=True)
    summaries = {}
    for label in ("baseline", "candidate"):
        wall = [value for run in runs if run["label"] == label for value in run["wall_ms"]]
        cpu = [value for run in runs if run["label"] == label for value in run["process_cpu_ms"]]
        summaries[label] = {"mean_call_ms": float(np.mean(wall)), "median_call_ms": float(np.median(wall)),
                            "p95_call_ms": float(np.percentile(wall, 95)), "max_call_ms": float(np.max(wall)),
                            "mean_process_cpu_ms": float(np.mean(cpu)), "measured_calls": len(wall)}
    arrays = [np.fromfile(args.output / (label + "-outputs.f32"), np.float32) for label in ("baseline", "candidate")]
    spectral_elements = args.calls * 2 * int(np.prod(shape))
    expected = spectral_elements + sum(int(np.prod(s["shape"])) for s in baseline["states"])
    if any(value.size != expected for value in arrays):
        raise RuntimeError("Output tensor dump size differs from graph contract")
    parity = {"rtol": 2e-3, "atol": 3e-4, "spectra_and_final_states": True,
              "all_finite": all(bool(np.all(np.isfinite(x))) for x in arrays),
              "max_spectrum_abs_error": float(np.max(np.abs(arrays[0][:spectral_elements] - arrays[1][:spectral_elements]))),
              "max_final_state_abs_error": float(np.max(np.abs(arrays[0][spectral_elements:] - arrays[1][spectral_elements:])))}
    parity["passed"] = parity["all_finite"] and bool(np.allclose(*arrays, rtol=parity["rtol"], atol=parity["atol"]))
    report = {"scope": "Physical Android native-shell model ABBA; not paced/app/microphone acceptance",
              "frozen_inputs": files, "screen_before": before, "screen_after": screen_state(),
              "system_abi": adb("shell", "getprop", "ro.product.cpu.abi").stdout.decode().strip(),
              "device_model": adb("shell", "getprop", "ro.product.model").stdout.decode().strip(),
              "soc": adb("shell", "getprop", "ro.soc.model").stdout.decode().strip(),
              "runs": runs, "summary": summaries, "parity": parity,
              "mean_reduction_percent": 100 * (1 - summaries["candidate"]["mean_call_ms"] / summaries["baseline"]["mean_call_ms"])}
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output / "thermal-after.txt").write_bytes(adb("shell", "dumpsys", "thermalservice").stdout)
    print(json.dumps({key: report[key] for key in ("summary", "parity", "mean_reduction_percent")}, indent=2))
    if not parity["passed"]:
        raise RuntimeError("Numerical parity failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("adb", "runner", "ort", "baseline-library", "candidate-library", "model-dir", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--candidate-model-dir", type=Path)
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 9))
    parser.add_argument("--warmup-calls", type=int, default=8)
    parser.add_argument("--calls", type=int, default=40)
    parser.add_argument("--reverse-order",action="store_true",help="Use candidate/baseline/baseline/candidate order")
    main(parser.parse_args())
