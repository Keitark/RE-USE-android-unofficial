import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from audio_stream import StreamingAudio


@pytest.mark.parametrize("rate", [8000, 16000])
def test_streaming_identity_preserves_every_sample_and_length(rate):
    frontend = StreamingAudio(rate)
    rng = np.random.default_rng(9001)
    samples = rng.normal(0, .1, 7 * frontend.hop).astype(np.float32)
    samples[0], samples[-1] = .8, -.7
    output = []
    for chunk in list(samples.reshape(-1, frontend.hop)) + [np.zeros(frontend.hop, np.float32)]:
        mag, phase = frontend.analyze(chunk)
        amplitude = np.expm1(mag)
        enhanced = frontend.synthesize(amplitude*np.cos(phase), amplitude*np.sin(phase))
        if enhanced is not None:
            output.append(enhanced)
    np.testing.assert_allclose(np.concatenate(output), samples, atol=2e-7)
    frontend.reset()
    assert frontend.synthesize(np.zeros(frontend.hop+1), np.zeros(frontend.hop+1)) is None


def test_invalid_audio_rejected_without_poisoning_state():
    frontend = StreamingAudio(16000)
    with pytest.raises(ValueError):
        frontend.analyze(np.full(320, np.nan))
    assert frontend.count == 0 and np.count_nonzero(frontend.previous) == 0
