"""Acceptance must reject lost work or a requested profile that was not applied."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audit_even_probe import audit


def receipt(directory):
    stream = dict(ok=True, frames_per_call=5, pcm_sample_rate=16000,
                  input_samples=320000, output_samples=320000,
                  inference_hops=1000, inference_calls=200, codec_frames=2000,
                  model_applied_frames=1975, underflow_zero_frames=0,
                  backlog_resets=0, failures=0, fallback_frames=0, rtf=.6, tail_flushed=False)
    report = dict(self_test=stream, diagnostics=dict(idle_resets=0, explicit_resets=0,
                  unsupported_frames=0, model_sha256="a"*64, runtime_profile="dynamic4"),
                  interactive_at_start=True, interactive_at_finish=True,
                  keyguard_locked_at_start=False, keyguard_locked_at_finish=False,
                  effective_runtime_profile="dynamic4", session_reused=False, session_reconfigured=True)
    execution = dict(exact_apk_match=True, device_is_emulator=False, system_abi="arm64-v8a",
                     warm_requested=True, same_process_warm_launch=True,
                     requested_runtime_profile="dynamic4", runtime_profile_matches_requested=True)
    def save():
        (directory/"probe-result.json").write_text(json.dumps(report))
        (directory/"execution.json").write_text(json.dumps(execution))
    save()
    return report, execution, save


def test_complete_warm_process_with_intentional_profile_reconfiguration_passes(tmp_path):
    receipt(tmp_path)
    result = audit(tmp_path)
    assert result["passed"] and result["input_work_complete"]
    assert result["nominal_source_delay_ms_this_run"] == 250
    assert result["paced_tail_flushed"] is False


def test_fast_average_does_not_accept_discarded_input(tmp_path):
    report, _, save = receipt(tmp_path)
    report["self_test"].update(inference_hops=995, inference_calls=199)
    save()
    result = audit(tmp_path)
    assert result["checks"]["average_keeps_up"]
    assert not result["passed"] and not result["input_work_complete"]


def test_native_profile_mismatch_is_rejected_even_if_metadata_claims_match(tmp_path):
    report, _, save = receipt(tmp_path)
    report["diagnostics"]["runtime_profile"] = "default"
    save()
    result = audit(tmp_path)
    assert not result["passed"]
    assert not result["checks"]["requested_runtime_profile_matches"]
