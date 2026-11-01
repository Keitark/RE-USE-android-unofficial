"""Independent NumPy audio frontend matching the Android hop contract."""
from __future__ import annotations

import numpy as np


class StreamingAudio:
    def __init__(self, sample_rate: int):
        if sample_rate not in (8000, 16000):
            raise ValueError("sample_rate must be 8000 or 16000")
        self.hop = sample_rate // 50
        self.nfft = self.hop * 2
        self.window = (.5 - .5 * np.cos(2 * np.pi * np.arange(self.nfft) / self.nfft)).astype(np.float32)
        self.reset()

    def reset(self):
        self.previous = np.zeros(self.hop, dtype=np.float32)
        self.tail = np.zeros(self.hop, dtype=np.float32)
        self.tail_weight = np.zeros(self.hop, dtype=np.float32)
        self.count = 0

    def analyze(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        samples = np.asarray(samples, dtype=np.float32)
        if samples.shape != (self.hop,) or not np.isfinite(samples).all():
            raise ValueError(f"Expected {self.hop} finite mono samples")
        frame = np.concatenate((self.previous, samples)) * self.window
        self.previous = samples.copy()
        spectrum = np.fft.rfft(frame)
        return np.log1p(np.abs(spectrum))[None, :].astype(np.float32), np.angle(spectrum)[None, :].astype(np.float32)

    def synthesize(self, real: np.ndarray, imag: np.ndarray) -> np.ndarray | None:
        spectrum = np.asarray(real).reshape(-1) + 1j * np.asarray(imag).reshape(-1)
        if spectrum.shape != (self.hop + 1,) or not np.isfinite(spectrum).all():
            raise ValueError("Invalid complex spectrum")
        frame = np.fft.irfft(spectrum, n=self.nfft).astype(np.float32) * self.window
        norm = self.window[:self.hop] ** 2 + self.tail_weight
        output = (frame[:self.hop] + self.tail) / np.maximum(norm, 1e-12)
        self.tail = frame[self.hop:].copy()
        self.tail_weight = self.window[self.hop:] ** 2
        self.count += 1
        return output.astype(np.float32) if self.count > 1 else None
