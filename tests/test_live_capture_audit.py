"""Synthetic metadata/WAV tests; no private G2 voice fixture is loaded."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import wave

import pytest

from tools.audit_live_capture import audit_capture, wave_identity


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


@pytest.fixture
def capture(tmp_path):
    n = 3200
    out = n + 188
    for name, count in (("before-reuse.wav", n), ("after-reuse.wav", out)):
        with wave.open(str(tmp_path / name), "wb") as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
            stream.writeframes(b"\0\0" * count)
    identities = {name: wave_identity(tmp_path / name) for name in ("before-reuse.wav", "after-reuse.wav")}
    stats = {
        "state": "stopped", "type": "stopped", "sample_rate": 16000, "model_rate": 8000,
        "input_samples": n, "processed_input_samples": n, "raw_recorded_samples": n,
        "output_samples": out, "dropped_input_samples": 0, "rejected_after_stop_samples": 0,
        "queue_overflows": 0, "sequence_gaps": 0, "pending_model_samples": 0,
        "pending_input_pair_samples": 0, "sample_index_validation": False,
        "valid_model_samples": 1647, "forwarded_model_samples": 1647, "model_calls": 3,
        "model_input_samples": 2400, "model_output_samples": 2400, "model_padding_samples": 753,
        "odd_input_pad_samples": 0, "fir_drain_input_samples": 94, "fir_tail_output_samples": 188,
        "flush_calls": 1, "transport_input_bytes": n * 2, "transport_input_frames": 4,
        "transport_rejected_input_bytes": 0, "transport_dropped_output_bytes": 0,
        "transport_generated_output_bytes": out * 2, "transport_written_output_bytes": out * 2,
        "source": "glasses", "live_g2_verified": False,
        "processing_rtf": .8, "worker_total_ms": 152., "drain_ms": 8.,
        "process_total_ms": 150., "process_mean_ms": 50., "process_p50_ms": 22.,
        "process_p95_ms": 107.8, "process_max_ms": 107.8, "deadline_misses": 1,
        "queue_age_max_ms": 284., "max_queue_samples": 4800,
        "private_diagnostics": {"serial": "SECRET_SERIAL", "token": "SECRET_TOKEN"},
        "hub_client_stats": {
            "state": "draining", "error": "", "protocol": 1,
            "requestedSource": "glasses", "observedSource": "glasses", "deviceConnected": True,
            "micStartSucceeded": True, "micStopSucceeded": True, "liveInputVerified": True,
            "callbacks": 4, "sourceValidatedEvents": 4, "sentMessages": 4,
            "callbackBytes": n * 2, "sentPayloadBytes": n * 2,
            "emptyCallbacks": 0, "sourceMismatchEvents": 0, "oddByteCallbacks": 0,
            "trailingOddBytes": 0, "lateAudioEvents": 0, "droppedBytes": 0, "discontinuities": 0,
            "callbackSizeCounts": {"1600": 4}, "minCallbackBytes": 1600, "maxCallbackBytes": 1600,
            "enhancedPayloadBytes": out * 2 - 2048,
        },
    }
    write_json(tmp_path / "capture-stats-private.json", stats)
    write_json(tmp_path / "file-identities.json", identities)
    return tmp_path, stats, identities


def test_async_pipeline_pass_keeps_deadline_and_attestation_limits(capture):
    directory, _, _ = capture
    result = audit_capture(directory)
    assert result["passed"] is True
    assert result["timing"]["deadline_misses"] == 1
    assert result["timing"]["all_100ms_deadlines_met"] is False
    assert result["hardware_attestation"] is False
    assert result["ble_gap_free_verified"] is False
    assert result["evidence_limits"]["sample_index_validation_enabled"] is False
    assert result["evidence_limits"]["server_live_g2_verified"] is False
    assert result["client_output_snapshot"]["difference_from_final_bytes"] == 2048
    public = json.dumps(result)
    assert "SECRET_TOKEN" not in public and "SECRET_SERIAL" not in public
    assert "private_diagnostics" not in public


@pytest.mark.parametrize("field,value", [
    ("processed_input_samples", 3199), ("raw_recorded_samples", 3199),
    ("dropped_input_samples", 1), ("queue_overflows", 1), ("rejected_after_stop_samples", 1),
    ("transport_rejected_input_bytes", 2), ("transport_dropped_output_bytes", 2),
    ("transport_written_output_bytes", 100), ("transport_generated_output_bytes", 100),
    ("pending_model_samples", 1), ("flush_calls", 0), ("fir_tail_output_samples", 0),
    ("model_padding_samples", 754), ("live_g2_verified", True), ("processing_rtf", 1.2),
    ("deadline_misses", 0), ("process_p95_ms", 500),
])
def test_inconsistent_or_lossy_capture_fails(capture, field, value):
    directory, stats, _ = capture
    stats[field] = value
    write_json(directory / "capture-stats-private.json", stats)
    assert audit_capture(directory)["passed"] is False


@pytest.mark.parametrize("field,value", [
    ("observedSource", "phone"), ("requestedSource", "phone"), ("deviceConnected", False),
    ("micStartSucceeded", False), ("micStopSucceeded", False), ("liveInputVerified", False),
    ("sourceValidatedEvents", 3), ("callbacks", 3), ("sentMessages", 3),
    ("callbackBytes", 6000), ("sentPayloadBytes", 6000), ("droppedBytes", 2),
    ("sourceMismatchEvents", 1), ("trailingOddBytes", 1),
])
def test_missing_or_inconsistent_source_evidence_fails(capture, field, value):
    directory, stats, _ = capture
    stats["hub_client_stats"][field] = value
    write_json(directory / "capture-stats-private.json", stats)
    assert audit_capture(directory)["passed"] is False
    del stats["hub_client_stats"][field]
    write_json(directory / "capture-stats-private.json", stats)
    assert audit_capture(directory)["passed"] is False


def test_client_snapshot_may_lag_but_not_exceed_final_payload(capture):
    directory, stats, _ = capture
    assert audit_capture(directory)["passed"]
    stats["hub_client_stats"]["enhancedPayloadBytes"] = stats["transport_written_output_bytes"] + 2
    write_json(directory / "capture-stats-private.json", stats)
    assert "client_pre_drain_output_snapshot_bounded" in audit_capture(directory)["failed_checks"]


def test_wav_hash_is_recomputed_and_truncated_riff_rejected(capture):
    directory, _, _ = capture
    path = directory / "after-reuse.wav"
    data = bytearray(path.read_bytes()); data[-1] = 1; path.write_bytes(data)
    assert "after-reuse.wav_sha256_identity" in audit_capture(directory)["failed_checks"]
    path.write_bytes(data[:-2])
    assert "after-reuse.wav_pcm16_format_and_complete_riff" in audit_capture(directory)["failed_checks"]


def test_callback_histogram_must_account_for_every_byte(capture):
    directory, stats, _ = capture
    stats["hub_client_stats"]["callbackSizeCounts"] = {"1598": 4}
    write_json(directory / "capture-stats-private.json", stats)
    assert "callback_histogram_byte_count" in audit_capture(directory)["failed_checks"]


def test_odd_sample_input_accounts_for_pair_pad_separately(capture):
    directory, stats, _ = capture
    n = 3201
    for name, count in (("before-reuse.wav", n), ("after-reuse.wav", n + 1 + 188)):
        with wave.open(str(directory / name), "wb") as stream:
            stream.setnchannels(1); stream.setsampwidth(2); stream.setframerate(16000)
            stream.writeframes(b"\0\0" * count)
    write_json(directory / "file-identities.json", {name: wave_identity(directory / name)
               for name in ("before-reuse.wav", "after-reuse.wav")})
    for name in ("input_samples", "processed_input_samples", "raw_recorded_samples"):
        stats[name] = n
    stats.update(output_samples=n + 1 + 188, valid_model_samples=1648, forwarded_model_samples=1648,
                 model_padding_samples=752, odd_input_pad_samples=1, transport_input_bytes=n * 2,
                 transport_generated_output_bytes=(n + 1 + 188) * 2,
                 transport_written_output_bytes=(n + 1 + 188) * 2, processing_rtf=160 / (n * 1000 / 16000))
    stats["hub_client_stats"].update(callbackBytes=n * 2, sentPayloadBytes=n * 2,
        callbackSizeCounts={"1600": 3, "1602": 1}, maxCallbackBytes=1602)
    write_json(directory / "capture-stats-private.json", stats)
    result = audit_capture(directory)
    assert result["passed"], result["failed_checks"]
    assert result["finite_drain"]["odd_input_pad_samples"] == 1


@pytest.mark.parametrize("channels,width,rate", [(2, 2, 16000), (1, 4, 16000), (1, 2, 8000)])
def test_non_transport_wav_formats_are_rejected(capture, channels, width, rate):
    directory, _, _ = capture
    with wave.open(str(directory / "before-reuse.wav"), "wb") as stream:
        stream.setnchannels(channels); stream.setsampwidth(width); stream.setframerate(rate)
        stream.writeframes(b"\0" * (3200 * channels * width))
    assert "before-reuse.wav_pcm16_format_and_complete_riff" in audit_capture(directory)["failed_checks"]


def test_apk_and_source_artifacts_are_independently_verified(capture):
    directory, _, _ = capture
    apk = directory / "test.apk"; apk.write_bytes(b"APK")
    source = directory / "source-receipt.json"; source.write_bytes(b"SOURCE")
    apk_hash = hashlib.sha256(apk.read_bytes()).hexdigest()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    result = audit_capture(directory, apk=apk, expected_apk_sha256=apk_hash,
                           source=source, expected_source_sha256=source_hash)
    assert result["passed"] and apk_hash != source_hash
    assert result["provenance"]["apk"]["sha256"] == apk_hash
    assert result["provenance"]["source"]["sha256"] == source_hash
    failed = audit_capture(directory, apk=apk, expected_apk_sha256=source_hash)
    assert "apk_expected_sha256" in failed["failed_checks"]


def test_invalid_typed_fields_cannot_leak_private_strings(capture):
    directory, stats, _ = capture
    stats["max_queue_samples"] = "SECRET_SERIAL"
    stats["transport_written_output_bytes"] = "SECRET_TOKEN"
    write_json(directory / "capture-stats-private.json", stats)
    result = audit_capture(directory)
    assert not result["passed"]
    assert "SECRET_SERIAL" not in json.dumps(result) and "SECRET_TOKEN" not in json.dumps(result)


def test_cli_writes_failure_receipt_and_returns_nonzero(capture):
    directory, stats, _ = capture
    stats["flush_calls"] = 0; write_json(directory / "capture-stats-private.json", stats)
    output = directory / "public-audit.json"
    result = subprocess.run([sys.executable, str(ROOT / "tools/audit_live_capture.py"), str(directory),
                             "--output", str(output)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 1
    assert json.loads(output.read_text())["passed"] is False
