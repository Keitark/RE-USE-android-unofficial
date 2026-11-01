"""Run true streaming WAV enhancement and generate Android parity fixtures."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import platform
import time

import numpy as np
import onnxruntime as ort
import soundfile as sf
from scipy.signal import resample_poly

from audio_stream import StreamingAudio
from fetch_model import sha256


class OnnxStream:
    def __init__(self, model_dir: Path, threads: int, custom_op_library: Path | None = None):
        self.manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
        self.frames_per_call = self.manifest.get("frames_per_call", 1)
        if type(self.frames_per_call) is not int or self.frames_per_call not in (1, 5):
            raise ValueError("frames_per_call must be 1 or 5")
        graph = model_dir / "model.onnx"
        if sha256(graph) != self.manifest["model_sha256"]:
            raise ValueError("Model SHA256 does not match manifest")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        custom_ops = self.manifest.get("custom_ops", [])
        if custom_ops:
            if custom_op_library is None:
                raise ValueError("This graph requires --custom-op-library pointing to reuse_ort_ops.dll/.so")
            options.register_custom_ops_library(str(custom_op_library.resolve()))
        self.session = ort.InferenceSession(str(graph), sess_options=options, providers=["CPUExecutionProvider"])
        self.states = [[np.zeros(s["shape"], np.float32) for s in self.manifest["states"]] for _ in range(2)]
        bins = self.manifest["fft_size"] // 2 + 1
        shape = (self.frames_per_call, bins)
        for values, names in ((self.session.get_inputs(), ("magnitude", "phase")),
                              (self.session.get_outputs(), ("spectrum_real", "spectrum_imag"))):
            tensors = {value.name: value for value in values}
            for name in names:
                if name not in tensors or tensors[name].shape != list(shape) or tensors[name].type != "tensor(float)":
                    raise ValueError(f"Graph feature contract differs from manifest: {name}, expected {shape} float32")
        self.magnitude = np.zeros(shape, np.float32)
        self.phase = np.zeros(shape, np.float32)
        self.outputs = [np.zeros(shape, np.float32), np.zeros(shape, np.float32)]
        self.bindings = []
        # All tensors are bound once, with recurrent state ping-ponged in place.
        for bank in range(2):
            binding = self.session.io_binding()
            binding.bind_cpu_input("magnitude", self.magnitude)
            binding.bind_cpu_input("phase", self.phase)
            for spec, value in zip(self.manifest["states"], self.states[bank]):
                binding.bind_cpu_input(spec["input"], value)
            for name, value in zip(("spectrum_real", "spectrum_imag"), self.outputs):
                binding.bind_output(name, "cpu", 0, np.float32, value.shape, value.ctypes.data)
            for spec, value in zip(self.manifest["states"], self.states[1-bank]):
                binding.bind_output(spec["output"], "cpu", 0, np.float32, value.shape, value.ctypes.data)
            self.bindings.append(binding)
        self.reset()

    def reset(self):
        for bank in self.states:
            for state in bank:
                state.fill(0)
        self.bank = 0

    def frame(self, magnitude, phase):
        # np.copyto would otherwise silently broadcast [1,bins] into [5,bins],
        # incorrectly repeating one frame through the temporal model.
        magnitude, phase = np.asarray(magnitude, dtype=np.float32), np.asarray(phase, dtype=np.float32)
        if magnitude.shape != self.magnitude.shape or phase.shape != self.phase.shape:
            raise ValueError(f"Expected exact feature shape {self.magnitude.shape} for magnitude and phase")
        if not np.isfinite(magnitude).all() or not np.isfinite(phase).all():
            raise ValueError("Features must be finite")
        np.copyto(self.magnitude, magnitude)
        np.copyto(self.phase, phase)
        self.session.run_with_iobinding(self.bindings[self.bank])
        self.bank = 1 - self.bank
        return self.outputs


def process_hops(stream, samples: np.ndarray, rate: int):
    """Analyze chronological hops, infer whole calls and synthesize chronologically.

    ``samples`` contains complete input hops. The caller resets neural state.
    One real flush hop and any call-completion padding pass through the normal
    frontend; output is trimmed to the original complete-hop sample count.
    """
    hop = rate // 50
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not len(samples) or len(samples) % hop or not np.isfinite(samples).all():
        raise ValueError("Expected nonempty finite mono PCM containing complete hops")
    frames_per_call = stream.frames_per_call
    input_hops = len(samples) // hop
    logical_hops = input_hops + 1  # One zero hop recovers the final OLA block.
    processed_hops = math.ceil(logical_hops / frames_per_call) * frames_per_call
    padded = np.pad(samples, (0, processed_hops * hop - len(samples)))
    frontend = StreamingAudio(rate)
    timings, model_timings, blocks = [], [], []
    for offset in range(0, len(padded), frames_per_call * hop):
        start = time.perf_counter()
        features = [frontend.analyze(padded[position:position+hop])
                    for position in range(offset, offset + frames_per_call * hop, hop)]
        magnitude = np.concatenate([value[0] for value in features], axis=0)
        phase = np.concatenate([value[1] for value in features], axis=0)
        model_start = time.perf_counter()
        real, imag = stream.frame(magnitude, phase)
        model_timings.append((time.perf_counter() - model_start) * 1000)
        for index in range(frames_per_call):
            output = frontend.synthesize(real[index], imag[index])
            if output is not None:
                blocks.append(output)
        timings.append((time.perf_counter() - start) * 1000)
    enhanced = np.concatenate(blocks)[:len(samples)]
    if len(enhanced) != len(samples) or not np.isfinite(enhanced).all():
        raise AssertionError("Invalid output length or nonfinite enhanced PCM")
    return enhanced, timings, model_timings, logical_hops, processed_hops


def enhance(args):
    stream = OnnxStream(args.model_dir, args.threads, args.custom_op_library)
    rate = stream.manifest["sample_rate"]
    samples, source_rate = sf.read(args.input, dtype="float32", always_2d=True)
    samples = samples.mean(axis=1)
    if source_rate != rate:
        divisor = math.gcd(source_rate, rate)
        samples = resample_poly(samples, rate // divisor, source_rate // divisor).astype(np.float32)
    samples = samples[round(args.start_seconds * rate):]
    if args.frames:
        samples = samples[:args.frames * (rate // 50)]
    original_length = len(samples)
    hop = rate // 50
    if original_length == 0:
        raise ValueError("Input is empty")
    samples = np.pad(samples, (0, (-len(samples)) % hop))
    silence = np.zeros((stream.frames_per_call, hop + 1), np.float32)
    # Warm code paths, then reset ALL audio and neural state for the real clip.
    for _ in range(args.warmup):
        stream.frame(silence, silence)
    stream.reset()
    complete_output, timings, model_timings, logical_hops, processed_hops = process_hops(stream, samples, rate)
    enhanced = complete_output[:original_length]
    assert len(enhanced) == original_length and np.isfinite(enhanced).all()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, enhanced, rate, subtype="FLOAT")
    if args.fixture:
        args.fixture.mkdir(parents=True, exist_ok=True)
        samples.astype("<f4").tofile(args.fixture / "input.f32")
        complete_output.astype("<f4").tofile(args.fixture / "expected.f32")
        sf.write(args.fixture / "input.wav", samples, rate, subtype="FLOAT")
        sf.write(args.fixture / "expected.wav", complete_output, rate, subtype="FLOAT")
    amortized = np.asarray(timings) / stream.frames_per_call
    deadline = 20 * stream.frames_per_call
    report = {
        "platform": platform.platform(), "processor": platform.processor(),
        "runtime": f"ONNX Runtime {ort.__version__} CPU", "threads": args.threads,
        "model_sha256": stream.manifest["model_sha256"], "sample_rate": rate,
        "samples": original_length, "audio_seconds": original_length / rate,
        "source_start_seconds": args.start_seconds,
        "frames_per_call": stream.frames_per_call,
        "maximum_collection_delay_ms": (stream.frames_per_call - 1) * 20,
        "frames_including_flush": logical_hops, "processed_hops_including_padding": processed_hops,
        "tail_padding_hops": processed_hops - logical_hops,
        "model_calls": len(timings), "warmup_calls": args.warmup,
        "warmup_frames": args.warmup * stream.frames_per_call,
        "end_to_end_call_ms": {"median": float(np.median(timings)), "p95": float(np.percentile(timings, 95)),
                                "max": float(max(timings))},
        "end_to_end_hop_ms": {"median": float(np.median(amortized)), "p95": float(np.percentile(amortized, 95)),
                               "max": float(max(amortized))},
        "mean_amortized_hop_ms": float(np.mean(amortized)),
        "model_call_ms_median": float(np.median(model_timings)),
        "model_hop_ms_median": float(np.median(model_timings)) / stream.frames_per_call,
        "deadline_ms": deadline, "deadline_misses": sum(t > deadline for t in timings),
        "rtf": sum(timings) / (original_length / rate * 1000),
        "processing_rtf_including_padding": sum(timings) / (processed_hops * 20),
        "output_peak": float(np.max(np.abs(enhanced))),
        "phone_realtime_verified": False,
    }
    report_file = args.output.with_suffix(".json")
    report_file.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--frames", type=int, default=0, help="Limit to N input hops, 0 means full clip")
    parser.add_argument("--start-seconds", type=float, default=0, help="Offset into the input recording before processing")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--custom-op-library", type=Path)
    args = parser.parse_args()
    if args.threads < 1 or args.frames < 0 or args.warmup < 0 or args.start_seconds < 0:
        parser.error("Invalid thread/frame count")
    enhance(args)
