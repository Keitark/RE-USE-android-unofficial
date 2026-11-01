"""The probe profile override is an allowlist, never arbitrary session options."""
import importlib.util
import json
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "android/even-probe-host/prepare_inputs.py"
SPEC = importlib.util.spec_from_file_location("even_probe_prepare", SOURCE)
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)
RUNNER_SOURCE = SOURCE.parents[2] / "tools/run_even_probe.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("even_probe_runner", RUNNER_SOURCE)
RUNNER = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(RUNNER)


def test_omission_preserves_exact_frozen_configuration_bytes():
    original = b'{\r\n "threads": 2, "enabled": true, "sample_rate":8000\r\n}\r\n'
    config, data = PREPARE.prepare_configuration(original)
    assert data is original
    assert "runtime_profile" not in config


@pytest.mark.parametrize("profile", PREPARE.RUNTIME_PROFILES)
def test_explicit_profile_only_adds_the_selected_fixed_name(profile):
    source = {"enabled": True, "sample_rate": 8000, "threads": 4, "other": [1, 2]}
    original = json.dumps(source).encode()
    config, data = PREPARE.prepare_configuration(original, profile)
    assert config == {**source, "runtime_profile": profile}
    assert json.loads(data) == config
    assert json.loads(original) == source
    assert data.endswith(b"\n")


@pytest.mark.parametrize("profile", PREPARE.RUNTIME_PROFILES)
def test_valid_source_profile_is_not_reserialized_without_override(profile):
    original = (' { "runtime_profile" : "' + profile + '" } ').encode()
    config, data = PREPARE.prepare_configuration(original)
    assert data is original
    assert config["runtime_profile"] == profile


@pytest.mark.parametrize("invalid", [None, False, 1, 1.0, [], {}, "unknown", "/sdcard/plugin.so"])
def test_invalid_source_profile_is_rejected_before_native_preparation(invalid):
    original = json.dumps({"runtime_profile": invalid}).encode()
    with pytest.raises(ValueError, match="runtime_profile"):
        PREPARE.prepare_configuration(original)


@pytest.mark.parametrize("invalid", [False, 1, [], {}, "unknown", "bounded_spin "])
def test_invalid_override_is_rejected(invalid):
    with pytest.raises(ValueError, match="runtime_profile"):
        PREPARE.prepare_configuration(b"{}", invalid)


@pytest.mark.parametrize("value", [[], 1, True, None])
def test_configuration_must_be_an_object(value):
    with pytest.raises(ValueError, match="JSON object"):
        PREPARE.prepare_configuration(json.dumps(value).encode())


def test_absent_runtime_intent_retains_the_current_session_profile():
    assert RUNNER.runtime_profile_launch_arguments(None) == []
    evidence = RUNNER.runtime_profile_evidence(None, {
        "effective_runtime_profile": "bounded_spin",
        "diagnostics": {"runtime_profile": "bounded_spin"}})
    assert evidence["requested_runtime_profile"] is None
    assert evidence["effective_runtime_profile"] == "bounded_spin"
    assert evidence["runtime_profile_matches_requested"]


@pytest.mark.parametrize("profile", PREPARE.RUNTIME_PROFILES)
def test_runtime_intent_uses_only_the_allowlisted_profile(profile):
    assert RUNNER.runtime_profile_launch_arguments(profile) == ["--es", "runtimeProfile", profile]
    evidence = RUNNER.runtime_profile_evidence(profile, {
        "effective_runtime_profile": profile, "diagnostics": {"runtime_profile": profile}})
    assert evidence["runtime_profile_matches_requested"]


@pytest.mark.parametrize("invalid", [False, 1, [], {}, "", "unknown", "default;id"])
def test_runtime_intent_rejects_non_profiles(invalid):
    with pytest.raises(ValueError, match="runtime profile"):
        RUNNER.runtime_profile_launch_arguments(invalid)


@pytest.mark.parametrize("java_profile,native_profile", [
    ("default", "default"), ("bounded_spin", "default"),
    ("default", "bounded_spin"), ("bounded_spin", None), (None, None)])
def test_runtime_profile_mismatch_cannot_be_claimed_as_a_success(java_profile, native_profile):
    report = {"effective_runtime_profile": java_profile, "diagnostics": {"runtime_profile": native_profile}}
    evidence = RUNNER.runtime_profile_evidence("bounded_spin", report)
    assert not evidence["runtime_profile_matches_requested"]
