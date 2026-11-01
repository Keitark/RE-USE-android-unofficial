"""Audit local live-capture counts/source reports without inspecting voice content.

This validates internally consistent application evidence, not hardware attestation,
neural waveform parity, or timestamped BLE continuity. Unknown/private fields are
never copied into the public receipt. APK and source artifact identities are separate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Any


RATE = 16000
NAMES = ("before-reuse.wav", "after-reuse.wav")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wave_identity(path: Path) -> dict[str, Any]:
    """Read RIFF structure and hash bytes; never decode/listen to PCM samples."""
    size = path.stat().st_size
    fmt = None
    data_bytes = None
    with path.open("rb") as stream:
        riff = stream.read(12)
        if len(riff) != 12 or riff[:4] != b"RIFF" or riff[8:] != b"WAVE":
            raise ValueError("Invalid RIFF header")
        if struct.unpack_from("<I", riff, 4)[0] + 8 != size:
            raise ValueError("RIFF length differs from file size")
        while stream.tell() < size:
            header = stream.read(8)
            if len(header) != 8:
                raise ValueError("Truncated chunk header")
            kind, length = struct.unpack("<4sI", header)
            if stream.tell() + length + (length & 1) > size:
                raise ValueError("Truncated RIFF chunk")
            if kind == b"fmt ":
                if fmt is not None or length < 16:
                    raise ValueError("Invalid format chunk")
                fmt = struct.unpack("<HHIIHH", stream.read(16))
                stream.seek(length - 16 + (length & 1), 1)
            else:
                if kind == b"data":
                    if data_bytes is not None:
                        raise ValueError("Multiple audio data chunks")
                    data_bytes = length
                stream.seek(length + (length & 1), 1)
    if fmt is None or data_bytes is None:
        raise ValueError("Missing format/data chunk")
    encoding, channels, rate, byte_rate, block_align, bits = fmt
    if encoding != 1 or channels != 1 or rate != RATE or bits != 16 or block_align != 2 or byte_rate != RATE * 2 or data_bytes % 2:
        raise ValueError("Capture must be 16 kHz mono PCM16LE")
    return {"samples": data_bytes // 2, "sample_rate": rate, "channels": channels,
            "sample_width": 2, "payload_bytes": data_bytes, "file_bytes": size,
            "seconds": data_bytes / (RATE * 2), "sha256": sha256(path)}


def integer(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def audit_capture(directory: Path, *, apk: Path | None = None, expected_apk_sha256: str | None = None,
                  source: Path | None = None, expected_source_sha256: str | None = None) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    receipt: dict[str, Any] = {
        "schema": 1,
        "scope": "Metadata/file-integrity audit of the local asynchronous G2-reported pipeline",
        "hardware_attestation": False,
        "waveform_quality_or_neural_parity_verified": False,
        "ble_gap_free_verified": False,
        "source_timestamps_available": False,
        "checks": checks,
    }

    def check(name: str, condition: Any) -> None:
        checks.append({"id": name, "passed": bool(condition)})

    def eq(name: str, value: Any, expected: Any) -> None:
        check(name, integer(value) and value == expected)

    def provenance(kind: str, path: Path | None, expected: str | None) -> None:
        if path is None and expected is None:
            return
        entry: dict[str, Any] = {"supplied": True}
        receipt.setdefault("provenance", {})[kind] = entry
        check(f"{kind}_identity_arguments", path is not None and expected is not None and
              isinstance(expected, str) and len(expected) == 64 and all(c in "0123456789abcdefABCDEF" for c in expected))
        if path is not None:
            try:
                entry["sha256"] = sha256(path)
                check(f"{kind}_expected_sha256", isinstance(expected, str) and entry["sha256"] == expected.lower())
            except OSError:
                check(f"{kind}_artifact_readable", False)

    try:
        stats_path = directory / "capture-stats-private.json"
        identities_path = directory / "file-identities.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8-sig"))
        identities = json.loads(identities_path.read_text(encoding="utf-8-sig"))
        if not isinstance(stats, dict) or not isinstance(identities, dict):
            raise ValueError("Metadata must be objects")
        receipt["metadata_sha256"] = {"capture_stats": sha256(stats_path), "file_identities": sha256(identities_path)}
    except (OSError, ValueError, TypeError):
        check("capture_metadata_readable", False)
        stats, identities = {}, {}

    waves: dict[str, dict[str, Any]] = {}
    for name in NAMES:
        try:
            waves[name] = wave_identity(directory / name)
            check(f"{name}_pcm16_format_and_complete_riff", True)
            declared = identities.get(name)
            check(f"{name}_declared_identity", isinstance(declared, dict))
            if isinstance(declared, dict):
                for field in ("samples", "sample_rate", "channels", "sample_width", "sha256"):
                    check(f"{name}_{field}_identity", declared.get(field) == waves[name][field])
                seconds = declared.get("seconds")
                check(f"{name}_seconds_identity", finite(seconds) and abs(seconds - waves[name]["seconds"]) < 1e-9)
        except (OSError, ValueError, struct.error):
            check(f"{name}_pcm16_format_and_complete_riff", False)
    receipt["recordings"] = waves

    n = waves.get(NAMES[0], {}).get("samples", 0)
    check("nonempty_input_capture", n > 0)
    eq("sample_rate", stats.get("sample_rate"), RATE)
    eq("model_rate", stats.get("model_rate"), 8000)
    check("terminal_stopped_state", stats.get("state") == "stopped" and stats.get("type") == "stopped")
    check("server_does_not_self_attest_hardware", stats.get("live_g2_verified") is False)
    for field in ("input_samples", "processed_input_samples", "raw_recorded_samples"):
        eq(field, stats.get(field), n)
    for field in ("dropped_input_samples", "rejected_after_stop_samples", "queue_overflows", "sequence_gaps",
                  "pending_model_samples", "pending_input_pair_samples", "transport_rejected_input_bytes",
                  "transport_dropped_output_bytes"):
        eq(field, stats.get(field), 0)

    odd_pad = n % 2
    valid_low = (n + odd_pad) // 2 + 47 if n else 0
    calls = (valid_low + 799) // 800
    expected_samples = n + odd_pad + 188 if n else 0
    for field, expected in {
        "output_samples": expected_samples, "valid_model_samples": valid_low,
        "forwarded_model_samples": valid_low, "model_calls": calls,
        "model_input_samples": calls * 800, "model_output_samples": calls * 800,
        "model_padding_samples": calls * 800 - valid_low, "odd_input_pad_samples": odd_pad,
        "fir_drain_input_samples": 94 if n else 0, "fir_tail_output_samples": 188 if n else 0,
        "flush_calls": 1 if n else 0, "transport_input_bytes": n * 2,
    }.items():
        eq(field, stats.get(field), expected)
    eq("final_wave_output_samples", waves.get(NAMES[1], {}).get("samples"), expected_samples)
    final_bytes = waves.get(NAMES[1], {}).get("payload_bytes", 0)
    eq("generated_output_equals_wave_payload", stats.get("transport_generated_output_bytes"), final_bytes)
    eq("written_output_equals_wave_payload", stats.get("transport_written_output_bytes"), final_bytes)
    receipt["finite_drain"] = {"real_input_samples": n, "odd_input_pad_samples": odd_pad,
                               "fir_tail_samples_16k": 188 if n else 0, "expected_output_samples": expected_samples,
                               "model_chunk_padding_discarded_samples_8k": calls * 800 - valid_low}

    client = stats.get("hub_client_stats")
    check("client_stats_present", isinstance(client, dict))
    if not isinstance(client, dict):
        client = {}
    check("requested_and_reported_source_glasses", stats.get("source") == "glasses" and
          client.get("requestedSource") == "glasses" and client.get("observedSource") == "glasses")
    for field in ("deviceConnected", "micStartSucceeded", "micStopSucceeded", "liveInputVerified"):
        check(f"client_{field}", client.get(field) is True)
    check("client_no_reported_error", client.get("error") == "")
    eq("client_protocol", client.get("protocol"), 1)
    frames = stats.get("transport_input_frames")
    check("positive_transport_frame_count", integer(frames) and frames > 0)
    for field in ("callbacks", "sourceValidatedEvents", "sentMessages"):
        eq(f"client_{field}_matches_transport", client.get(field), frames)
    for field in ("callbackBytes", "sentPayloadBytes"):
        eq(f"client_{field}_matches_raw_pcm", client.get(field), n * 2)
    for field in ("emptyCallbacks", "sourceMismatchEvents", "oddByteCallbacks", "trailingOddBytes",
                  "lateAudioEvents", "droppedBytes", "discontinuities"):
        eq(f"client_{field}", client.get(field), 0)
    sizes = client.get("callbackSizeCounts")
    sizes_valid = isinstance(sizes, dict) and bool(sizes) and all(
        isinstance(k, str) and k.isdigit() and int(k) > 0 and int(k) % 2 == 0 and integer(v)
        for k, v in sizes.items())
    check("client_callback_size_histogram_valid", sizes_valid)
    if sizes_valid:
        eq("callback_histogram_event_count", sum(sizes.values()), frames)
        eq("callback_histogram_byte_count", sum(int(k) * v for k, v in sizes.items()), n * 2)
        eq("client_min_callback_bytes", client.get("minCallbackBytes"), min(map(int, sizes)))
        eq("client_max_callback_bytes", client.get("maxCallbackBytes"), max(map(int, sizes)))

    snapshot_bytes = client.get("enhancedPayloadBytes")
    check("client_pre_drain_output_snapshot_bounded", integer(snapshot_bytes) and snapshot_bytes % 2 == 0 and snapshot_bytes <= final_bytes)
    receipt["client_output_snapshot"] = {
        "scope": "Client snapshot before native drain completed; not the final received-byte count",
        "enhanced_payload_bytes_before_drain": snapshot_bytes if integer(snapshot_bytes) else None,
        "final_server_written_payload_bytes": stats.get("transport_written_output_bytes") if integer(stats.get("transport_written_output_bytes")) else None,
        "difference_from_final_bytes": final_bytes - snapshot_bytes if integer(snapshot_bytes) else None,
    }
    receipt["application_pipeline"] = {
        name: stats.get(name) if integer(stats.get(name)) else None for name in (
            "input_samples", "processed_input_samples", "output_samples", "transport_input_frames",
            "dropped_input_samples", "rejected_after_stop_samples", "queue_overflows",
            "transport_rejected_input_bytes", "transport_generated_output_bytes",
            "transport_written_output_bytes", "transport_dropped_output_bytes", "flush_calls",
            "pending_model_samples", "pending_input_pair_samples")
    }
    receipt["client_source_reports"] = {
        "requested_glasses": client.get("requestedSource") == "glasses",
        "observed_glasses": client.get("observedSource") == "glasses",
        **{name: client.get(name) is True for name in (
            "deviceConnected", "micStartSucceeded", "micStopSucceeded", "liveInputVerified")},
        **{name: client.get(name) if integer(client.get(name)) else None for name in (
            "callbacks", "sourceValidatedEvents", "callbackBytes", "sentPayloadBytes")},
        "max_callback_interarrival_ms": client.get("maxInterarrivalMs") if finite(client.get("maxInterarrivalMs")) else None,
        "scope": "Client reports and callback counts; not hardware attestation or source timestamps",
    }

    timing_names = ("processing_rtf", "process_total_ms", "process_mean_ms", "process_p50_ms", "process_p95_ms",
                    "process_max_ms", "queue_age_max_ms", "worker_total_ms", "drain_ms")
    for field in timing_names:
        check(f"{field}_finite_nonnegative", finite(stats.get(field)))
    rtf = stats.get("processing_rtf")
    check("average_async_processing_below_realtime", finite(rtf) and rtf < 1)
    if finite(stats.get("worker_total_ms")) and finite(stats.get("drain_ms")) and n:
        calculated = (stats["worker_total_ms"] + stats["drain_ms"]) / (n * 1000 / RATE)
        check("processing_rtf_consistent", finite(rtf) and abs(rtf - calculated) < 1e-8)
    if finite(stats.get("process_total_ms")) and calls:
        check("process_mean_consistent", finite(stats.get("process_mean_ms")) and
              abs(stats["process_mean_ms"] - stats["process_total_ms"] / calls) < 1e-6)
    misses = stats.get("deadline_misses")
    check("deadline_miss_count_valid", integer(misses) and misses <= calls)
    maximum = stats.get("process_max_ms")
    check("deadline_counter_consistent_with_max", integer(misses) and finite(maximum) and (misses > 0) == (maximum > 100))
    quantiles = [stats.get(name) for name in ("process_p50_ms", "process_p95_ms", "process_max_ms")]
    check("process_quantiles_ordered", all(finite(value) for value in quantiles) and quantiles == sorted(quantiles))
    if all(finite(stats.get(name)) for name in ("worker_total_ms", "drain_ms", "process_total_ms")):
        check("processing_accounting_contains_model_calls", stats["worker_total_ms"] + stats["drain_ms"] + 1e-6 >= stats["process_total_ms"])
    check("queue_bound_respected", integer(stats.get("max_queue_samples")) and stats["max_queue_samples"] <= 8000)
    receipt["timing"] = {field: stats.get(field) if finite(stats.get(field)) else None for field in timing_names}
    receipt["timing"].update({"model_calls": calls, "deadline_misses": misses if integer(misses) else None,
                              "all_100ms_deadlines_met": integer(misses) and misses == 0 and finite(maximum) and maximum <= 100,
                              "max_queue_samples": stats.get("max_queue_samples") if integer(stats.get("max_queue_samples")) else None,
                              "first_output_after_input_ms": stats.get("first_output_after_input_ms") if finite(stats.get("first_output_after_input_ms")) else None,
                              "scope": "Asynchronous native processing; excludes G2/BLE/Hub capture latency"})
    receipt["evidence_limits"] = {
        "client_source_evidence": "Requested/observed glasses, connected state and microphone results are client reports",
        "server_live_g2_verified": stats.get("live_g2_verified") is True,
        "sample_index_validation_enabled": stats.get("sample_index_validation") is True,
        "ble_continuity": "No source timestamps; zero application drops does not establish gap-free BLE",
        "deadlines": "Average asynchronous throughput acceptance does not imply every 100 ms model deadline was met",
    }
    provenance("apk", apk, expected_apk_sha256)
    provenance("source", source, expected_source_sha256)
    receipt["failed_checks"] = [item["id"] for item in checks if not item["passed"]]
    receipt["passed"] = not receipt["failed_checks"]
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apk", type=Path)
    parser.add_argument("--expected-apk-sha256")
    parser.add_argument("--source", type=Path, help="Exact source/receipt artifact, independently hashed from the APK")
    parser.add_argument("--expected-source-sha256")
    args = parser.parse_args()
    receipt = audit_capture(args.directory, apk=args.apk, expected_apk_sha256=args.expected_apk_sha256,
                            source=args.source, expected_source_sha256=args.expected_source_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"passed": receipt["passed"], "failed_checks": receipt["failed_checks"],
                      "all_100ms_deadlines_met": receipt["timing"]["all_100ms_deadlines_met"]}))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
