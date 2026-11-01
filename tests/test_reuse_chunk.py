"""A multi-hop call must preserve the causal single-hop model and state ABI."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from reuse_port import StreamingReuse
from reuse_chunk import ChunkedStreamingReuse, _NativeSelectiveSequence


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.set_num_threads(2)
    torch.manual_seed(77405)


def serial_call(model, magnitude, phase, states):
    real, imag = [], []
    for index in range(magnitude.shape[0]):
        result = model(magnitude[index:index+1], phase[index:index+1], *states)
        real.append(result[0]); imag.append(result[1]); states = result[2:]
    return (torch.cat(real), torch.cat(imag), *states)


@pytest.mark.parametrize("fuse_time_sequence", [False, True])
def test_five_hop_chunk_matches_serial_over_40_recurrent_frames(fuse_time_sequence):
    checkpoint = ROOT / ".research/upstream/model.safetensors"
    baseline = StreamingReuse(checkpoint if checkpoint.is_file() else None,
                             sample_rate=8000, scan_mode="native", fuse_time_step=True)
    chunk = ChunkedStreamingReuse(baseline.state_dict(), sample_rate=8000,
                                 scan_mode="native", fuse_time_step=True,
                                 fuse_time_sequence=fuse_time_sequence)
    assert chunk.state_specs == baseline.state_specs
    assert set(chunk.state_dict()) == set(baseline.state_dict())
    states_a = baseline.initial_states()
    states_b = chunk.initial_states()
    with torch.inference_mode():
        for index in range(8):
            magnitude = torch.rand(5, 161) * (1 + index % 3)
            phase = torch.rand(5, 161) * 6 - 3
            expected = serial_call(baseline, magnitude, phase, states_a)
            actual = chunk(magnitude, phase, *states_b)
            for output_index, (a, e) in enumerate(zip(actual, expected)):
                assert torch.isfinite(a).all()
                torch.testing.assert_close(a, e, atol=3e-4, rtol=2e-3,
                                           msg=f"chunk {index}, output {output_index}")
            states_a, states_b = expected[2:], actual[2:]
    assert any(torch.count_nonzero(value) for value in states_b)


@pytest.mark.parametrize("sample_rate", [8000, 16000])
def test_single_frame_chunk_preserves_baseline_abi(sample_rate):
    baseline = StreamingReuse(sample_rate=sample_rate)
    chunk = ChunkedStreamingReuse(baseline.state_dict(), frames_per_call=1,
                                 sample_rate=sample_rate)
    bins = sample_rate // 50 + 1
    magnitude, phase = torch.rand(1, bins), torch.rand(1, bins) * 6 - 3
    with torch.inference_mode():
        expected = baseline(magnitude, phase, *baseline.initial_states())
        actual = chunk(magnitude, phase, *chunk.initial_states())
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, atol=3e-4, rtol=2e-3)


def test_zero_chunk_reset_is_reproducible_and_input_states_are_immutable():
    model = ChunkedStreamingReuse(sample_rate=8000)
    states = model.initial_states()
    saved = tuple(value.clone() for value in states)
    zero = torch.zeros(5, 161)
    with torch.inference_mode():
        first = model(zero, zero, *states)
        model(zero, zero, *first[2:])
        reset = model(zero, zero, *model.initial_states())
    for a, e in zip(states, saved):
        torch.testing.assert_close(a, e, atol=0, rtol=0)
    for a, e in zip(first, reset):
        torch.testing.assert_close(a, e, atol=0, rtol=0)


def test_chunk_does_not_use_future_frames_to_change_earlier_spectra():
    model = ChunkedStreamingReuse(sample_rate=8000)
    magnitude, phase = torch.rand(5, 161), torch.rand(5, 161) * 6 - 3
    other_magnitude, other_phase = magnitude.clone(), phase.clone()
    other_magnitude[3:] += 7
    other_phase[3:] *= -1
    with torch.inference_mode():
        first = model.forward_raw(magnitude, phase, *model.initial_states())
        changed = model.forward_raw(other_magnitude, other_phase, *model.initial_states())
    for a, e in zip(first[:3], changed[:3]):
        torch.testing.assert_close(a[:3], e[:3], atol=1e-6, rtol=1e-6)


def test_chunk_rejects_partial_features_and_invalid_state_contract():
    with pytest.raises(ValueError, match="frames_per_call"):
        ChunkedStreamingReuse(sample_rate=8000, frames_per_call=3)
    model = ChunkedStreamingReuse(sample_rate=8000)
    with pytest.raises(ValueError, match="shape"):
        model(torch.zeros(4, 161), torch.zeros(4, 161), *model.initial_states())
    with pytest.raises(ValueError, match="state tensors"):
        model(torch.zeros(5, 161), torch.zeros(5, 161))
    states = list(model.initial_states()); states[0] = states[0][..., :-1]
    with pytest.raises(ValueError, match="state_000"):
        model(torch.zeros(5, 161), torch.zeros(5, 161), *states)


@pytest.mark.parametrize("kwargs", [
    {"frames_per_call": 1, "scan_mode": "native", "fuse_time_step": True},
    {"frames_per_call": 5, "scan_mode": "parallel", "fuse_time_step": False},
    {"frames_per_call": 5, "scan_mode": "native", "fuse_time_step": False},
])
def test_sequence_fusion_rejects_incompatible_graph_contract(kwargs):
    with pytest.raises(ValueError, match="fuse_time_sequence requires"):
        ChunkedStreamingReuse(sample_rate=8000, fuse_time_sequence=True, **kwargs)


def test_sequence_reference_preserves_every_step_and_initial_state():
    u = torch.randn(2, 5, 7) * .2
    delta = torch.rand_like(u) * .1
    b, c = torch.randn(2, 5, 16) * .2, torch.randn(2, 5, 16) * .2
    a = -torch.rand(7, 16)
    initial = torch.randn(2, 7, 16) * .1
    saved = initial.clone()
    state, outputs = initial, []
    for frame in range(5):
        state = (torch.exp(delta[:, frame, :, None] * a) * state
                 + (delta[:, frame, :, None] * b[:, frame, None, :]) * u[:, frame, :, None])
        outputs.append((state * c[:, frame, None, :]).sum(-1))
    observed, final = _NativeSelectiveSequence.apply(u, delta, b, c, a, initial)
    torch.testing.assert_close(observed, torch.stack(outputs, dim=1), atol=0, rtol=0)
    torch.testing.assert_close(final, state, atol=0, rtol=0)
    torch.testing.assert_close(initial, saved, atol=0, rtol=0)
    assert final.data_ptr() != initial.data_ptr()


def test_sequence_fusion_keeps_causality_reset_and_state_abi():
    model = ChunkedStreamingReuse(sample_rate=8000, scan_mode="native",
                                 fuse_time_step=True, fuse_time_sequence=True)
    assert model.architecture_manifest()["fuse_time_sequence"] is True
    magnitude, phase = torch.rand(5, 161), torch.rand(5, 161) * 6 - 3
    changed_magnitude, changed_phase = magnitude.clone(), phase.clone()
    changed_magnitude[3:] += 7
    changed_phase[3:] *= -1
    initial = model.initial_states()
    saved = tuple(value.clone() for value in initial)
    with torch.inference_mode():
        first = model(magnitude, phase, *initial)
        changed = model(changed_magnitude, changed_phase, *model.initial_states())
        reset = model(magnitude, phase, *model.initial_states())
    for actual, expected in zip(first[:2], changed[:2]):
        torch.testing.assert_close(actual[:3], expected[:3], atol=1e-6, rtol=1e-6)
    for actual, expected in zip(first, reset):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    for actual, expected in zip(initial, saved):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
