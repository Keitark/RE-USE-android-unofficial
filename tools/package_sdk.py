"""Collect the built AAR, local Maven metadata, model packs and validation records."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import os
import argparse

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_PROFILES = {"default", "no_spin", "bounded_spin", "dynamic4"}
RUNTIME_OPTIONS = {
    "session.intra_op.allow_spinning": {"0", "1"},
    "session.intra_op.spin_duration_us": {"1000"},
    "session.intra_op.spin_backoff_max": {"4"},
    "session.force_spinning_stop": {"1"},
    "session.dynamic_block_base": {"4"},
}


def runtime_settings(values):
    """Project only recognized per-session settings and provider identities."""
    result = {}
    for key in ("runtime_profile", "requested_runtime_profile", "effective_runtime_profile",
                "native_runtime_profile"):
        if isinstance(values.get(key), str) and values[key] in RUNTIME_PROFILES:
            result[key] = values[key]
    options = values.get("runtime_options")
    if isinstance(options, dict):
        result["runtime_options"] = {key: value for key, value in options.items()
                                     if key in RUNTIME_OPTIONS and isinstance(value, str)
                                     and value in RUNTIME_OPTIONS[key]}
    providers = values.get("available_providers")
    if isinstance(providers, list):
        result["available_providers"] = [value for value in providers
                                         if value in ("CPUExecutionProvider", "NnapiExecutionProvider")]
    return result


def runtime_receipt(directory: Path, scope: str):
    """Retain relevant evidence without a device serial or private path fields."""
    directory = directory.resolve()
    report_path = directory / "probe-result.json"
    execution_path = directory / "execution.json"
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    execution = json.loads(execution_path.read_text(encoding="utf-8-sig"))
    result = {
        "run": directory.name,
        "source_receipt_sha256": {
            "probe-result.json": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            "execution.json": hashlib.sha256(execution_path.read_bytes()).hexdigest(),
        },
        "probe": {key: report[key] for key in (
            "schema_version", "device", "abi", "api_level", "started_at_epoch_ms",
            "finished_at_epoch_ms", "interactive_at_start", "interactive_at_finish",
            "keyguard_locked_at_start", "keyguard_locked_at_finish", "hold_screen",
            "requested_model_rate", "requested_threads", "effective_model_rate",
            "effective_threads", "warm_launch_requested", "session_reconfigured",
            "session_reused", "self_test_wall_ms", "ok", "input_bytes", "output_bytes"
        ) if key in report},
        "execution": {key: execution[key] for key in (
            "package", "apk_sha256", "installed_apk_sha256", "exact_apk_match",
            "system_abi", "packaged_native_abi", "device_is_emulator", "warm_requested",
            "foreground_vs_doze", "physical_phone_probe_verified", "device_model",
            "android_version", "screen_before", "screen_after"
        ) if key in execution},
        "scope": scope,
    }
    result["probe"].update(runtime_settings(report))
    result["execution"].update(runtime_settings(execution))
    for key in ("runtime_profile_matches_requested", "same_process_warm_launch"):
        if isinstance(execution.get(key), bool):
            result["execution"][key] = execution[key]
    # Numeric counters, flags and fixed identities are sufficient to interpret
    # the run. In particular, never copy errors that can embed private paths.
    for section in ("self_test", "diagnostics"):
        values = report.get(section, {})
        if isinstance(values, dict):
            result[section] = {key: value for key, value in values.items()
                               if isinstance(value, (bool, int, float)) or
                               key in ("ort_version", "model_sha256")}
            result[section].update(runtime_settings(values))
    parity_path = directory / "direct-parity.json"
    if parity_path.exists():
        parity = json.loads(parity_path.read_text(encoding="utf-8-sig"))
        result["direct_parity"] = {key: value for key, value in parity.items()
                                   if isinstance(value, (bool, int, float)) or key.endswith("sha256")}
        if "model_sha256" in result["direct_parity"]:
            result["direct_parity"]["reference_model_sha256"] = result["direct_parity"].pop("model_sha256")
        result["source_receipt_sha256"]["direct-parity.json"] = hashlib.sha256(parity_path.read_bytes()).hexdigest()
    acceptance_path = directory / "continuous-acceptance.json"
    if acceptance_path.exists():
        acceptance = json.loads(acceptance_path.read_text(encoding="utf-8-sig"))
        fields = ("scope", "passed", "input_audio_seconds", "expected_calls", "expected_hops",
                  "actual_model_sha256", "startup_silence_frames_this_run",
                  "nominal_source_delay_ms_this_run", "mean_call_ms", "amortized_hop_ms",
                  "session_reused", "session_reconfigured", "paced_timing",
                  "paced_tail_flushed", "input_work_complete", "source_receipt_sha256", "limitations")
        result["continuous_acceptance"] = {key: acceptance[key] for key in fields if key in acceptance}
        result["continuous_acceptance"].update(runtime_settings(acceptance))
        result["continuous_acceptance"]["checks"] = {
            key: value for key, value in acceptance.get("checks", {}).items() if isinstance(value, bool)}
        result["source_receipt_sha256"]["continuous-acceptance.json"] = hashlib.sha256(acceptance_path.read_bytes()).hexdigest()
    continuity_path = directory / "pipeline-continuity-parity.json"
    if continuity_path.exists():
        continuity = json.loads(continuity_path.read_text(encoding="utf-8-sig"))
        fields = ("scope", "startup_silence_samples", "nominal_delay_ms", "compared_samples",
                  "maximum_pcm16_lsb_error", "rmse_pcm16_lsb", "startup_is_zero", "paced_sha256",
                  "direct_sha256", "passed")
        result["pipeline_continuity"] = {key: continuity[key] for key in fields if key in continuity}
        result["source_receipt_sha256"]["pipeline-continuity-parity.json"] = hashlib.sha256(continuity_path.read_bytes()).hexdigest()
    return result


def physical_receipt(directory: Path):
    result = runtime_receipt(directory,
            "Recorded-audio historical one-hop physical ARM64 codec/model probe; not live G2 acceptance")
    if result.get("self_test", {}).get("frames_per_call") == 5:
        result["scope"] = "Recorded-audio optimized five-hop physical ARM64 codec/model probe; not live G2 or microphone-to-ASR latency acceptance"
    if directory.name == "xperia-opcode-setup-2s":
        result["scope"] = "Historical opcode setup check with an earlier probe binary; not the final opcode APK or live G2 acceptance"
    if directory.name == "next-compute-setup-2s":
        result["scope"] = "Failed cold setup with the earlier opcode APK; not a ChannelNormPRelu pipeline test or live G2 acceptance"
    return result


def host_comparison_receipt(path: Path):
    """Project reproducible timing evidence without profiler/host file paths."""
    report = json.loads(path.read_text(encoding="utf-8"))
    result = {
        "scope": "Unprofiled host CPU ABBA comparison; not physical-phone timing",
        "source_receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "onnxruntime": report["onnxruntime"],
        "timing_unit": report.get("timing_unit", report["runs"][0].get("timing_unit", "ms_per_one_hop")),
        "comparison": report["comparison"],
    }
    for key in ("frozen_inputs", "candidate_frozen_inputs"):
        result[key] = {name: {field: value[field] for field in ("bytes", "sha256")}
                       for name, value in report.get(key, {}).items()
                       if name in ("model.onnx", "manifest.json", "reuse_ort_ops.dll")}
    fields = ("threads", "profiled", "warmup_frames", "measured_frames", "frames_per_call",
              "measured_calls", "timing_unit", "mean_ms", "median_ms", "p95_ms", "max_ms",
              "rtf", "graph", "order", "mean_call_ms", "p95_call_ms")
    result["runs"] = [{key: run[key] for key in fields if key in run}
                      for run in report["runs"]]
    return result


def cpu_comparison_receipt(path: Path):
    """Retain complete shell diagnostic samples without serial/path fields."""
    report = json.loads(path.read_text(encoding="utf-8"))
    private_fields = {"serial", "profile_path", "command", "stdout", "stderr"}

    def project(value):
        if isinstance(value, dict):
            return {key: project(item) for key, item in value.items()
                    if key not in private_fields and not key.endswith("_path")}
        if isinstance(value, list):
            return [project(item) for item in value]
        return value

    result = project(report)
    result["source_receipt_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["diagnostic_only"] = True
    result["acceptance_eligible"] = False
    return result


def java_chunk_receipt(directory: Path):
    report_path = directory / "report.json"
    execution_path = directory / "execution.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    safe_strings = ("device", "abi", "onnxruntime_version", "model_sha256", "manifest_sha256",
                    "input_sha256", "expected_sha256")
    scope = "Actual Java neural/DSP/lifecycle correctness on x86_64 Android emulator; not physical-phone timing"
    if execution.get("device_is_emulator") is False and report.get("abi") == "arm64-v8a":
        scope = "Actual public Java neural/DSP/lifecycle correctness on physical ARM64 Android; not Even codec pacing or live G2 acceptance"
    result = {
        "scope": scope,
        "source_receipt_sha256": {"report.json": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                                  "execution.json": hashlib.sha256(execution_path.read_bytes()).hexdigest()},
        "report": {key: value for key, value in report.items()
                   if isinstance(value, (bool, int, float)) or key in safe_strings},
        "lifecycle": report["lifecycle"],
        "execution": {key: execution[key] for key in ("scope", "device_is_emulator", "packages", "enhanced_sha256",
                      "benchmark_arm_custom_ops_sha256", "model_sha256", "runtime") if key in execution},
    }
    return result


def package(physical_runs, include_frequency_candidate=False):
    output = ROOT / "dist/reuse-android-sdk"
    # This is a generated staging tree. Resolve its exact location before
    # replacing it so stale files cannot enter the next SDK archive.
    if output.exists():
        if output.resolve().parent != (ROOT / "dist").resolve() or output.name != "reuse-android-sdk":
            raise ValueError("Unexpected SDK staging directory")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("README.md", "LICENSE", "THIRD_PARTY_NOTICES.md"):
        shutil.copy2(ROOT/name, output/name)
    for name in ("build.gradle", "settings.gradle", "gradle.properties", "gradlew", "gradlew.bat", "requirements.txt", ".gitignore"):
        shutil.copy2(ROOT/name, output/name)
    # Include buildable source, not caches, original vendor APKs or link-only
    # CMake outputs. Native/even binaries are experimental APK inputs, not SDK
    # source files and must not accidentally ship the link-only codec stub.
    for directory in ("tools", "tests", "native", "android", "gradle", "scripts"):
        for current, directories, files in os.walk(ROOT/directory):
            directories[:] = [d for d in directories if d not in ("build", "__pycache__", ".cxx", ".externalNativeBuild") and not d.startswith("build-")]
            relative = Path(current).relative_to(ROOT)
            destination = output/relative
            destination.mkdir(parents=True, exist_ok=True)
            for filename in files:
                shutil.copy2(Path(current)/filename, destination/filename)
    shutil.copytree(ROOT/"docs", output/"docs", dirs_exist_ok=True)
    shutil.copytree(ROOT/"licenses", output/"licenses", dirs_exist_ok=True)
    aar = ROOT/"android/reuse/build/outputs/aar/reuse-release.aar"
    shutil.copy2(aar, output/"reuse-android-0.1.0.aar")
    shutil.copytree(ROOT/"build/repository", output/"maven", dirs_exist_ok=True)
    model_names = ("reuse-8k-exit3-native", "reuse-16k-exit3-native",
                 "reuse-8k-exit3-native-step", "reuse-16k-exit3-native-step",
                 "reuse-8k-exit3-native-compact", "reuse-8k-exit3-native-chunk5",
                 "reuse-8k-exit3-native-chunk5-sequence",
                 "reuse-8k-exit3-native-chunk5-sequence-pad",
                 "reuse-8k-exit3-native-chunk5-sequence-pad-softplus",
                 "reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal",
                 "reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal-channelnorm")
    if include_frequency_candidate:
        model_names += ("reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal-channelnorm-frequency",)
    for name in model_names:
        folder = output/"models"/name
        folder.mkdir(parents=True, exist_ok=True)
        for filename in ("model.onnx", "manifest.json", "LICENSE-NVIDIA.txt"):
            shutil.copy2(ROOT/"models"/name/filename, folder/filename)
        for proof_name in ("optimization-parity.json", "optimization.json", "normalization-preservation-proof.json"):
            proof = ROOT/"models"/name/proof_name
            if proof.exists():
                shutil.copy2(proof, folder/proof.name)
    receipts = {
        "upstream-parity-8k-40frames.json": ".research/upstream-parity-8k-40frames.json",
        "upstream-parity-16k-40frames.json": ".research/upstream-parity-16k-40frames.json",
        "android-native-8k.json": ".research/android-native-8k-fixed/report.json",
        "android-native-16k.json": ".research/android-native-16k-fixed/report.json",
        "host-native-8k.json": ".research/speech-8k-native.json",
        "host-native-16k.json": ".research/speech-16k-native.json",
        "even-arm-initialization.json": ".research/even-probe-precise/init-result.json",
        "even-arm-direct-parity.json": ".research/even-probe-precise/direct-parity.json",
        "even-arm-paced-probe.json": ".research/even-probe-precise/probe-result.json",
        "even-arm-execution.json": ".research/even-probe-precise/execution.json",
        "native-step-validation.json": ".research/native-step-validation.json",
        "compact-graph-parity.json": "models/reuse-8k-exit3-native-compact/optimization-parity.json",
        "chunk5-validation.json": ".research/chunk5-validation.json",
        "chunk5-host-speech-parity.json": ".research/chunk5-speech-parity.json",
        "chunk5-host-java-fixture-parity.json": ".research/chunk5-java-fixture-parity.json",
        "sequence-graph-validation.json": ".research/sequence-graph-validation.json",
        "sequence-host-speech-parity.json": ".research/sequence-speech-parity.json",
        "sequence-pad-graph-parity.json": "models/reuse-8k-exit3-native-chunk5-sequence-pad/optimization-parity.json",
        "sequence-pad-softplus-graph-parity.json": "models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus/optimization-parity.json",
        "softplus-validation.json": ".research/softplus-validation.json",
        "sequence-pad-softplus-host-speech-parity.json": ".research/softplus-pad-speech-parity.json",
        "fir-polyphase-host-microbenchmark.json": ".research/fir-polyphase-benchmark.json",
        "temporal-conv-graph-parity.json": "models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal/optimization-parity.json",
        "opcode-summary.json": "dist/validation/opcode-summary.json",
        "opcode-phone-kernel-correctness.json": ".research/opcode-cpu/kernel-combined.json",
        "opcode-phone-fft-diagnostic.json": ".research/opcode-cpu/fft-phone.json",
        "opcode-phone-baseline-profile.json": ".research/opcode-cpu/baseline-profile-summary.json",
        "compute-fusion-summary.json": "dist/validation/compute-fusion-summary.json",
        "channel-norm-graph-parity.json": "models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal-channelnorm/optimization-parity.json",
        "channel-norm-phone-cost-attribution.json": ".research/channelnorm-addressable-phone.json",
        "rejected-channel-norm13-graph-parity.json": "models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal-channelnorm13/optimization-parity.json",
        "rejected-channel-norm13-preservation-proof.json": "models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal-channelnorm13/normalization-preservation-proof.json",
    }
    if include_frequency_candidate:
        receipts["experimental-frequency-conv-host-graph-parity.json"] = "models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal-channelnorm-frequency/optimization-parity.json"
        receipts["experimental-frequency-combined-host-graph-parity.json"] = ".research/frequency-conv-validation/combined_from_ab02.json"
    (output/"validation").mkdir(exist_ok=True)
    for target, source in receipts.items():
        if target == "even-arm-execution.json":
            # This older emulator record predates runtime_receipt projection.
            # Preserve provenance/identity while omitting its ADB identifier.
            source_path = ROOT/source
            execution = json.loads(source_path.read_text(encoding="utf-8-sig"))
            execution.pop("serial", None)
            execution["source_receipt_sha256"] = hashlib.sha256(source_path.read_bytes()).hexdigest()
            (output/"validation"/target).write_text(json.dumps(execution, indent=2) + "\n", encoding="utf-8")
        else:
            shutil.copy2(ROOT/source, output/"validation"/target)
    physical = {"schema_version": 1, "date": "2026-11-01",
                "runs": [physical_receipt(path) for path in physical_runs]}
    (output/"validation/physical-xperia.json").write_text(json.dumps(physical, indent=2) + "\n", encoding="utf-8")
    generated = {
        "chunk5-host-comparison.json": host_comparison_receipt(ROOT/".research/profile-opt-chunk5-abba/report.json"),
        "compact-host-comparison.json": host_comparison_receipt(ROOT/".research/profile-opt-dense-history-abba-repeat/report.json"),
        "sequence-host-comparison.json": host_comparison_receipt(ROOT/".research/profile-opt-sequence-abba/report.json"),
        "sequence-pad-host-comparison.json": host_comparison_receipt(ROOT/".research/profile-otherareas-pad/report.json"),
        "sequence-softplus-host-comparison.json": host_comparison_receipt(ROOT/".research/profile-otherareas-softplus/report.json"),
        "sequence-pad-softplus-host-comparison.json": host_comparison_receipt(ROOT/".research/profile-otherareas-combined/report.json"),
        "chunk5-arm-translation.json": runtime_receipt(ROOT/".research/android-arm-translation-chunk5",
                "Actual ARM64 native codec/model under x86_64 emulator translation; not physical-phone timing"),
        "chunk5-java-android.json": java_chunk_receipt(ROOT/".research/android-java-chunk5"),
        "compute-fusion-java-physical-android.json": java_chunk_receipt(ROOT/".research/android-java-compute-fusion"),
    }
    rejected = host_comparison_receipt(ROOT/".research/profile-otherareas-compact/report.json")
    rejected["scope"] = "Rejected five-hop Gather experiment: slower host ABBA result; graph excluded from SDK model packs"
    generated["rejected-five-hop-gather-host-comparison.json"] = rejected
    for experiment in ("fma", "scan16", "combined", "temporal", "all"):
        generated[f"opcode-{experiment}-phone-abba.json"] = cpu_comparison_receipt(
            ROOT/f".research/opcode-{experiment}-abba/report.json")
    generated["opcode-all-phone-baab.json"] = cpu_comparison_receipt(
        ROOT/".research/opcode-all-baab/report.json")
    generated["channel-norm-host-abba.json"] = host_comparison_receipt(
        ROOT/".research/next-compute-norm-host-abba/report.json")
    generated["compute-fusion-combined-host-abba.json"] = host_comparison_receipt(
        ROOT/".research/next-compute-combined-host-abba/report.json")
    for order in ("abba", "baab"):
        generated[f"channel-norm-phone-{order}.json"] = cpu_comparison_receipt(
            ROOT/f".research/next-compute-norm-{order}/report.json")
    rejected_norm = cpu_comparison_receipt(ROOT/".research/next-compute-norm13-abba/report.json")
    rejected_norm["scope"] = "Rejected selective13 normalization experiment: slower than18-region fusion in physical-phone model screening; not paced acceptance"
    generated["rejected-channel-norm13-phone-abba.json"] = rejected_norm
    grain = cpu_comparison_receipt(ROOT/".research/next-compute-grain32-abba/report.json")
    grain["scope"] = "Not promoted: frequency-Scan grain32 mean result is drift-confounded and median worsens; not paced acceptance"
    generated["unpromoted-scan-grain32-phone-abba.json"] = grain
    frequency = host_comparison_receipt(ROOT/".research/next-compute-frequency-host-abba/report.json")
    frequency["scope"] = "FrequencyConvSilu host ABBA comparison; physical-phone evidence is recorded separately, not pipeline acceptance"
    frequency["candidate_model_included"] = include_frequency_candidate
    generated["experimental-frequency-conv-host-abba.json"] = frequency
    generated["frequency-conv-phone-abba.json"] = cpu_comparison_receipt(
        ROOT/".research/next-compute-frequency-arm-abba/report.json")
    combined_phone = cpu_comparison_receipt(ROOT/".research/next-compute-combined-arm-baab/report.json")
    combined_phone["interpretation"] = "Raw mean is drift-confounded by substantial block variation; not a reliable combined speedup claim"
    generated["compute-fusion-combined-phone-baab.json"] = combined_phone
    generated["compute-fusion-combined-phone-abba.json"] = cpu_comparison_receipt(
        ROOT/".research/next-compute-combined-arm-abba/report.json")
    for filename, value in generated.items():
        (output/"validation"/filename).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    hashes = {str(path.relative_to(output)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sorted(output.rglob("*")) if path.is_file() and path.name != "SHA256.json"}
    (output/"SHA256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    archive = ROOT/"dist/reuse-android-sdk.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as stream:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                stream.write(path, str(path.relative_to(output)).replace("\\", "/"))
    print(f"SDK: {archive} ({archive.stat().st_size:,} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-run", action="append", type=Path,
                        help="Probe receipt directory; repeat to retain multiple sanitized physical runs")
    parser.add_argument("--include-experimental-frequency", action="store_true",
                        help="Include the separate FrequencyConvSilu candidate and its validation; no live G2 acceptance claim")
    args = parser.parse_args()
    package(args.physical_run or [ROOT/".research"/name for name in (
        "xperia-probe-cold-8k-t2", "xperia-audio-priority-20s-8k-t4",
        "xperia-audio-priority-20s-8k-t3", "xperia-hints-cold-8k-t4",
        "xperia-chunk5-cold-8k-t4", "xperia-chunk5-warm-20s-8k-t4",
        "xperia-chunk5-warm-60s-8k-t4", "xperia-chunk5-paced-v2-cold-8k-t4",
        "xperia-chunk5-paced-v2-warm-60s-8k-t4", "xperia-sequence-cold-8k-t4",
        "xperia-sequence-warm-60s-8k-t4", "xperia-sequence-60s-8k-t2",
        "xperia-otherareas-cold-default-8k-t4", "xperia-otherareas-default-20s-a1",
        "xperia-otherareas-bounded_spin-20s-b1", "xperia-otherareas-bounded_spin-20s-b2",
        "xperia-otherareas-default-20s-a2", "xperia-otherareas-dynamic4-20s",
        "xperia-otherareas-dynamic4-60s", "xperia-opcode-setup-2s",
        "xperia-opcode-cold-2s", "xperia-opcode-warm-20s", "xperia-opcode-warm-60s",
        "next-compute-setup-2s", "xperia-compute-fusion-cold-2s",
        "xperia-compute-fusion-warm-60s")], include_frequency_candidate=args.include_experimental_frequency)
