"""Checks for recurrent-state correctness and checkpoint/export portability."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from reuse_port import StreamingReuse, selective_scan


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(77123)


@pytest.mark.parametrize("length", [1, 2, 3, 4, 81, 161])
def test_parallel_scan_matches_independent_recurrence(length):
    decay = torch.rand(2, length, 3, 4) * 0.3 + 0.7
    drive = torch.randn_like(decay)
    h = torch.zeros_like(drive[:, 0])
    expected = []
    for step in range(length):
        h = h * decay[:, step] + drive[:, step]
        expected.append(h)
    torch.testing.assert_close(selective_scan(decay, drive), torch.stack(expected, dim=1), atol=2e-6, rtol=2e-6)


def test_parallel_scan_handles_exact_zero_decay_without_division():
    decay = torch.zeros(1, 161, 2, 3)
    drive = torch.randn_like(decay)
    result = selective_scan(decay, drive)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, drive, atol=0, rtol=0)


def test_streaming_nonzero_states_match_sequential_frequency_scan():
    parallel = StreamingReuse(sample_rate=8000, exit_layer=3)
    sequential = StreamingReuse(parallel.state_dict(), sample_rate=8000, exit_layer=3, scan_mode="sequential")
    state_parallel = parallel.initial_states()
    state_sequential = sequential.initial_states()
    with torch.inference_mode():
        for frame in range(4):
            magnitude = torch.rand(1, 161) * (frame + 1)
            phase = (torch.rand(1, 161) - 0.5) * 6
            actual = parallel(magnitude, phase, *state_parallel)
            reference = sequential(magnitude, phase, *state_sequential)
            for a, b in zip(actual, reference):
                assert torch.isfinite(a).all()
                torch.testing.assert_close(a, b, atol=2e-4, rtol=2e-3)
            state_parallel = actual[2:]
            state_sequential = reference[2:]
    assert any(torch.count_nonzero(state) for state in state_parallel)


def test_native_scan_export_boundary_eager_reference_matches_portable_graph():
    parallel = StreamingReuse(sample_rate=8000, exit_layer=3)
    native = StreamingReuse(parallel.state_dict(), sample_rate=8000, exit_layer=3, scan_mode="native")
    states_parallel = parallel.initial_states()
    states_native = native.initial_states()
    with torch.inference_mode():
        for _ in range(3):
            magnitude = torch.rand(1, 161) * 3
            phase = (torch.rand(1, 161) - 0.5) * 6
            expected = parallel(magnitude, phase, *states_parallel)
            actual = native(magnitude, phase, *states_native)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=0, rtol=0)
            states_parallel = expected[2:]
            states_native = actual[2:]


def test_optional_native_temporal_step_is_exact_eager_and_preserves_nonzero_states():
    baseline = StreamingReuse(sample_rate=8000, scan_mode="native")
    fused = StreamingReuse(baseline.state_dict(), sample_rate=8000, scan_mode="native", fuse_time_step=True)
    states_a = tuple(torch.randn_like(t) * .01 for t in baseline.initial_states())
    states_b = tuple(t.clone() for t in states_a)
    with torch.inference_mode():
        for _ in range(8):
            magnitude, phase = torch.rand(1, 161) * 3, torch.rand(1, 161) * 6 - 3
            expected = baseline(magnitude, phase, *states_a)
            actual = fused(magnitude, phase, *states_b)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=0, rtol=0)
            states_a, states_b = expected[2:], actual[2:]
    assert fused.architecture_manifest()["fuse_time_step"] is True
    with pytest.raises(ValueError, match="requires scan_mode"):
        StreamingReuse(fuse_time_step=True)


def test_export_publishes_exact_static_state_shapes_for_custom_ops():
    import onnx
    from onnx import helper, TensorProto
    from export_model import enforce_fixed_io_shapes

    graph = helper.make_model(helper.make_graph(
        [], "shape_contract",
        [helper.make_tensor_value_info("state", TensorProto.FLOAT, [81, 256, 16])],
        [helper.make_tensor_value_info("next_state", TensorProto.FLOAT, ["batch", "inner", "rank"])]))
    enforce_fixed_io_shapes(graph, {"state": [81, 256, 16]}, {"next_state": [81, 256, 16]})
    assert [d.dim_value for d in graph.graph.output[0].type.tensor_type.shape.dim] == [81, 256, 16]
    assert not any(d.dim_param for d in graph.graph.output[0].type.tensor_type.shape.dim)
    with pytest.raises(RuntimeError, match="names differ"):
        enforce_fixed_io_shapes(graph, {"different_input": [81, 256, 16]}, {"next_state": [81, 256, 16]})


def test_explicit_state_is_not_mutated_and_reset_is_reproducible():
    model = StreamingReuse(sample_rate=8000)
    initial = model.initial_states()
    magnitude = torch.rand(1, 161)
    phase = torch.rand(1, 161)
    with torch.inference_mode():
        first = model(magnitude, phase, *initial)
        model(magnitude, phase, *first[2:])
        reset = model(magnitude, phase, *model.initial_states())
    assert all(torch.count_nonzero(state) == 0 for state in initial)
    for a, b in zip(first, reset):
        torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_phase_origin_matches_atan2_zero_zero_convention():
    model = StreamingReuse(sample_rate=8000)
    with torch.no_grad():
        head = model.mask_decoder_list[2].final_conv
        head.weight.zero_()
        head.bias.fill_(1)
        phase_head = model.phase_decoder_list[2]
        phase_head.phase_conv_r.weight.zero_()
        phase_head.phase_conv_r.bias.zero_()
        phase_head.phase_conv_i.weight.zero_()
        phase_head.phase_conv_i.bias.zero_()
        real, imaginary, *_ = model(torch.zeros(1, 161), torch.zeros(1, 161), *model.initial_states())
    torch.testing.assert_close(real, torch.full_like(real, torch.expm1(torch.tensor(1.0))))
    assert torch.count_nonzero(imaginary) == 0


def test_checkpoint_loading_rejects_missing_or_unknown_active_keys():
    model = StreamingReuse(sample_rate=8000)
    weights = model.state_dict()
    removed = next(iter(weights))
    with pytest.raises(ValueError, match="missing active"):
        model.load_original_checkpoint({k: v for k, v in weights.items() if k != removed})
    with pytest.raises(ValueError, match="Unexpected checkpoint"):
        model.load_original_checkpoint({**weights, "unknown.weight": torch.ones(1)})


def test_fixed_shape_state_contract_and_unsupported_configurations():
    model = StreamingReuse(sample_rate=16000, exit_layer=3)
    assert len(model.state_specs) == 21
    assert model.architecture_manifest()["state_bytes"] == 26408496
    assert model.architecture_manifest()["active_parameters"] == 1823555
    assert model.state_specs[5]["shape"] == [161, 256, 4]
    assert model.state_specs[6]["shape"] == [161, 256, 16]
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(1, 320), torch.zeros(1, 320), *model.initial_states())
    with pytest.raises(ValueError, match="state tensors"):
        model(torch.zeros(1, 321), torch.zeros(1, 321))
    with pytest.raises(ValueError, match="lookahead"):
        StreamingReuse(lookahead=1)
    with pytest.raises(ValueError, match="sample_rate"):
        StreamingReuse(sample_rate=44100)


def test_locked_original_topology_matches_streaming_graph():
    source = ROOT / ".research/upstream"
    if not (source / "model.safetensors").is_file():
        pytest.skip("Run python tools/fetch_model.py --reference for pretrained integration parity")
    if importlib.util.find_spec("einops") is None or importlib.util.find_spec("huggingface_hub") is None:
        pytest.skip("Independent-reference dependencies are not installed")
    from validate_upstream import validate

    receipt = validate(source, sample_rate=8000, exit_layer=3, frames=6, threads=2)
    assert receipt["passed"]
    assert receipt["frames"] >= 4
    assert receipt["raw_log_magnitude_max_abs_error"] < 2e-4
