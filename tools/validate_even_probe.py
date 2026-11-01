"""Compare an ARM LC3/interposer direct replay with an independent host reference.

The paced live-burst counters remain separate: a correct offline replay is not
proof that the phone keeps up with incoming microphone audio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audio_stream import StreamingAudio
from enhance_audio import OnnxStream


def filter95(samples: np.ndarray) -> np.ndarray:
    index = np.arange(95, dtype=np.float64)
    x = index - 47
    cutoff = 3400. / 16000.
    coefficients = (2 * cutoff * np.sinc(2 * cutoff * x) * (.54-.46*np.cos(2*np.pi*index/94))).astype(np.float32)
    coefficients /= np.float32(coefficients.astype(np.float64).sum())
    # Accumulate the FIR in float64 before rounding its output. A float32
    # running sum leaves platform-dependent residue in the stopband; the model
    # consumes phase even in those low-energy bins, making that residue matter.
    result = np.zeros(len(samples), dtype=np.float64)
    accurate_input = samples.astype(np.float64)
    for k, coefficient in enumerate(coefficients):
        if k < len(samples):
            result[k:] += float(coefficient) * accurate_input[:len(samples)-k]
    return result.astype(np.float32)


def reference(baseline: np.ndarray, stream: OnnxStream) -> np.ndarray:
    rate = stream.manifest["sample_rate"]
    # Complete 20 ms G2 hops and one flush hop. The decimator continues through
    # the flush, exactly as the native AudioEngine does.
    frames_per_call = stream.frames_per_call
    logical_hops = (len(baseline) + 319) // 320 + 1
    processed_hops = ((logical_hops + frames_per_call - 1) // frames_per_call) * frames_per_call
    padded = np.pad(baseline, (0, processed_hops * 320 - len(baseline))).astype(np.float32)
    model_input = filter95(padded)[1::2] if rate == 8000 else padded
    frontend = StreamingAudio(rate)
    stream.reset()
    blocks = []
    hops = model_input.reshape(-1, rate//50)
    for start in range(0, len(hops), frames_per_call):
        features = [frontend.analyze(frame) for frame in hops[start:start+frames_per_call]]
        mag = np.concatenate([value[0] for value in features], axis=0)
        phase = np.concatenate([value[1] for value in features], axis=0)
        real, imag = stream.frame(mag, phase)
        for index in range(frames_per_call):
            block = frontend.synthesize(real[index], imag[index])
            if block is not None:
                blocks.append(block)
    output = np.concatenate(blocks)
    if rate == 8000:
        upsampled = np.zeros(len(output)*2, np.float32)
        upsampled[::2] = output
        output = 2 * filter95(upsampled)
    return np.clip(np.rint(output[:len(baseline)]*32768), -32768, 32767).astype("<i2")


def validate(args):
    baseline = np.fromfile(args.baseline, dtype="<i2")
    actual = np.fromfile(args.actual, dtype="<i2")
    if len(actual) != len(baseline) or not len(actual):
        raise AssertionError("PCM sample count mismatch or empty probe")
    expected = reference(baseline.astype(np.float32)/32768,
                         OnnxStream(args.model_dir, args.threads, args.custom_op_library))
    error = (actual.astype(np.float64)-expected.astype(np.float64))/32768
    maximum, rmse = float(np.max(np.abs(error))), float(np.sqrt(np.mean(error**2)))
    power = np.mean((expected.astype(np.float64)/32768)**2)
    snr = float(10*np.log10(max(power, 1e-30)/max(rmse**2, 1e-30)))
    report = {
        "reference": "Independent host FIR + NumPy STFT/OLA + CPU ORT; original LC3 decoded baseline",
        "model_sha256": json.loads((args.model_dir/"manifest.json").read_text())["model_sha256"],
        "baseline_sha256": hashlib.sha256(args.baseline.read_bytes()).hexdigest(),
        "actual_sha256": hashlib.sha256(args.actual.read_bytes()).hexdigest(),
        "reference_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "numpy_version": np.__version__,
        "samples": len(actual), "pcm_sample_rate": 16000,
        "max_abs_error": maximum, "rmse": rmse, "snr_db": snr,
        "max_abs_tolerance": args.max_abs, "rmse_tolerance": args.rmse,
        "passed": maximum <= args.max_abs and rmse <= args.rmse,
        "scope": "Offline ARM native codec/model/DSP correctness; live burst deadlines remain separate",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    expected.tofile(args.output.with_suffix(".expected.pcm16le"))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise AssertionError("ARM native DSP/model output differs from independent reference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--actual", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--custom-op-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-abs", type=float, default=5e-4)
    parser.add_argument("--rmse", type=float, default=1e-4)
    validate(parser.parse_args())
