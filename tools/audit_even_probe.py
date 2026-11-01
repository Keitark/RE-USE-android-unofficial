"""Audit complete work and continuity in a physical recorded-audio probe.

This does not establish live G2 transport, microphone-stop tail handling, or
microphone-to-ASR latency. Detailed device identifiers stay in private inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def audit(directory: Path, expected_model: str | None = None) -> dict:
    report = json.loads((directory / "probe-result.json").read_text())
    execution = json.loads((directory / "execution.json").read_text())
    stream, counters = report["self_test"], report["diagnostics"]
    frames = stream.get("frames_per_call", 1)
    samples_per_hop = stream["pcm_sample_rate"] // 50
    hops = stream["input_samples"] // samples_per_hop
    calls = hops // frames
    reported_calls = stream.get("inference_calls", stream["inference_hops"] if frames == 1 else -1)
    checks = {
        "self_test_pass": stream["ok"],
        "complete_chunks": stream["input_samples"] % (samples_per_hop * frames) == 0,
        "exact_hop_count": stream["inference_hops"] == hops,
        "exact_call_count": reported_calls == calls,
        "recorded_sample_count_matches": stream["output_samples"] == stream["input_samples"],
        "no_output_underflows": stream["underflow_zero_frames"] == 0,
        "no_backlog_resets": stream["backlog_resets"] == 0,
        "no_failures_or_fallback": stream["failures"] == stream["fallback_frames"] == 0,
        "no_idle_explicit_or_unsupported": all(counters[key] == 0 for key in (
            "idle_resets", "explicit_resets", "unsupported_frames")),
        "average_keeps_up": 0 < stream["rtf"] < 1,
        "exact_apk": execution["exact_apk_match"],
        "native_arm64_phone": not execution["device_is_emulator"] and execution["system_abi"] == "arm64-v8a",
        "awake_unlocked_endpoints": all(report[key] for key in (
            "interactive_at_start", "interactive_at_finish")) and not any(report[key] for key in (
            "keyguard_locked_at_start", "keyguard_locked_at_finish")),
    }
    if execution["warm_requested"]:
        checks["same_process_warm_launch"] = execution["same_process_warm_launch"]
    requested_profile = execution.get("requested_runtime_profile")
    if requested_profile is not None:
        checks["requested_runtime_profile_matches"] = (
            execution.get("runtime_profile_matches_requested") is True
            and report.get("effective_runtime_profile") == requested_profile
            and counters.get("runtime_profile") == requested_profile)
    if expected_model is not None:
        checks["expected_model_sha256"] = counters["model_sha256"] == expected_model
    startup = stream["codec_frames"] - stream["model_applied_frames"] - stream["underflow_zero_frames"]
    continuous = checks["no_output_underflows"] and checks["no_backlog_resets"] and checks["no_failures_or_fallback"]
    result = {
        "scope": "Bounded recorded five-LC3-frames/50ms continuous probe; not live G2 acceptance",
        "checks": checks, "passed": all(checks.values()),
        "input_audio_seconds": stream["input_samples"] / stream["pcm_sample_rate"],
        "expected_calls": calls, "expected_hops": hops,
        "actual_model_sha256": counters["model_sha256"],
        "startup_silence_frames_this_run": startup,
        "nominal_source_delay_ms_this_run": startup * 10 if continuous else None,
        "mean_call_ms": stream["rtf"] * frames * 20,
        "amortized_hop_ms": stream["rtf"] * 20,
        "session_reused": report.get("session_reused", False),
        "session_reconfigured": report.get("session_reconfigured", False),
        "requested_runtime_profile": requested_profile,
        "effective_runtime_profile": counters.get("runtime_profile"),
        "runtime_options": counters.get("runtime_options", {}),
        "available_providers": counters.get("available_providers", []),
        "paced_tail_flushed": stream["tail_flushed"],
        "input_work_complete": checks["exact_hop_count"] and checks["exact_call_count"],
        "paced_timing": {key: stream[key] for key in (
            "paced_wall_us", "max_burst_lateness_us", "late_bursts_over_10ms",
            "max_decode_call_us", "recording_file_io_in_paced_loop") if key in stream},
        "source_receipt_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                  for name in ("probe-result.json", "execution.json")},
        "limitations": [
            "Delay is nominal source-frame age, not wall-clock microphone-to-ASR latency.",
            "Recorded output includes startup zeros; the paced native hook does not deliver its delayed end-of-stream tail.",
            "Whole self-test time and post-test thermal readings also include separate offline replay.",
        ],
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--expected-model-sha256")
    args = parser.parse_args()
    result = audit(args.directory, args.expected_model_sha256)
    (args.directory / "continuous-acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "passed", "input_audio_seconds", "expected_calls", "input_work_complete",
        "mean_call_ms", "nominal_source_delay_ms_this_run")}))
    raise SystemExit(0 if result["passed"] else 2)
