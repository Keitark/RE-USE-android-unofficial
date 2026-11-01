"""Chunk collection must preserve chronological audio and exact output length."""
from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from enhance_audio import OnnxStream, process_hops


class IdentityStream:
    def __init__(self, frames_per_call):
        self.frames_per_call = frames_per_call
        self.shapes = []

    def frame(self, magnitude, phase):
        self.shapes.append(magnitude.shape)
        assert magnitude.shape == phase.shape
        assert magnitude.shape[0] == self.frames_per_call
        amplitude = np.expm1(magnitude)
        return amplitude*np.cos(phase), amplitude*np.sin(phase)


@pytest.mark.parametrize("rate", [8000, 16000])
@pytest.mark.parametrize("input_hops", [1, 5, 7, 10])
def test_grouped_hops_preserve_audio_and_drop_only_call_completion_padding(rate, input_hops):
    rng = np.random.default_rng(rate + input_hops)
    hop = rate // 50
    samples = rng.uniform(-.8, .8, input_hops * hop).astype(np.float32)
    chunk = IdentityStream(5)
    actual, times, model_times, logical_hops, processed_hops = process_hops(chunk, samples, rate)
    baseline, *_ = process_hops(IdentityStream(1), samples, rate)
    np.testing.assert_allclose(actual, samples, atol=2e-6, rtol=2e-6)
    np.testing.assert_array_equal(actual, baseline)
    assert actual.shape == samples.shape
    assert logical_hops == input_hops + 1
    assert processed_hops % 5 == 0 and processed_hops >= logical_hops
    assert len(times) == len(model_times) == len(chunk.shapes) == processed_hops // 5


def test_exact_feature_shape_prevents_broadcasting_one_frame_into_chunk():
    stream = OnnxStream.__new__(OnnxStream)
    stream.magnitude = np.zeros((5, 161), np.float32)
    stream.phase = np.zeros((5, 161), np.float32)
    with pytest.raises(ValueError, match="exact feature shape"):
        stream.frame(np.zeros((1, 161), np.float32), np.zeros((1, 161), np.float32))
    with pytest.raises(ValueError, match="finite"):
        stream.frame(np.full((5, 161), np.nan, np.float32), np.zeros((5, 161), np.float32))


@pytest.mark.parametrize("samples", [np.zeros(0), np.zeros(159), np.full(160, np.inf)])
def test_grouped_hops_reject_invalid_pcm(samples):
    with pytest.raises(ValueError, match="complete hops"):
        process_hops(IdentityStream(5), samples, 8000)
