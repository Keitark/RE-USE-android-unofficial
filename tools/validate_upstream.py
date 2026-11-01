"""Check the port against hash-locked NVIDIA source with an independent Mamba.

This validator executes only the three explicitly SHA256-verified upstream
model source files, using an independently implemented sequential CPU
Mamba-1 backend. It does not execute upstream inference entry points and
does not need mamba_ssm, CUDA, torchaudio, or a network connection.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from safetensors.torch import load_file

from fetch_model import ROOT, LOCK, sha256
from reuse_port import StreamingReuse


class ReferenceMamba(nn.Module):
    """Independent sequential Mamba-1 reference with checkpoint-compatible keys.

    Equations follow the official Mamba-1 implementation (Tri Dao and Albert
    Gu): https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba_simple.py
    This implementation deliberately does not call the portable graph's
    Mamba class or selective_scan function.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 4, **_: Any) -> None:
        super().__init__()
        self.inner = d_model * expand
        self.state_size = d_state
        self.rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, 2 * self.inner, bias=False)
        self.conv1d = nn.Conv1d(self.inner, self.inner, d_conv, padding=d_conv - 1, groups=self.inner)
        self.x_proj = nn.Linear(self.inner, self.rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.rank, self.inner)
        self.A_log = nn.Parameter(torch.zeros(self.inner, d_state))
        self.D = nn.Parameter(torch.ones(self.inner))
        self.out_proj = nn.Linear(self.inner, d_model, bias=False)

    def forward(self, hidden: Tensor) -> Tensor:
        projected = self.in_proj(hidden)
        values, gate = projected.split(self.inner, dim=-1)
        values = F.silu(self.conv1d(values.transpose(1, 2))[:, :, : hidden.shape[1]])
        features = self.x_proj(values.transpose(1, 2))
        delta_features, b, c = features.split((self.rank, self.state_size, self.state_size), dim=-1)
        delta = F.softplus(self.dt_proj(delta_features))
        transition = -self.A_log.exp()
        state = values.new_zeros(hidden.shape[0], self.inner, self.state_size)
        history = []
        for step in range(hidden.shape[1]):
            input_t = values[:, :, step]
            dt_t = delta[:, step]
            state = state * torch.exp(dt_t[:, :, None] * transition)
            state = state + input_t[:, :, None] * dt_t[:, :, None] * b[:, step, None, :]
            observed = (state * c[:, step, None, :]).sum(dim=-1)
            observed = (observed + self.D * input_t) * F.silu(gate[:, step])
            history.append(observed)
        return self.out_proj(torch.stack(history, dim=1))


_REFERENCE_FILES = (
    "config.json",
    "model.safetensors",
    "models/streaming_codec_module_time_d1_input_ahead_sep_conv.py",
    "models/streaming_generator_SEMamba_time_d1_random_layer_ahead_sep_conv.py",
    "models/streaming_mamba_block2_SEMamba.py",
)


def verify_reference_files(source: Path) -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8-sig"))
    for name in _REFERENCE_FILES:
        path = source / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}; run python tools/fetch_model.py --reference")
        if sha256(path) != lock["files"][name]:
            raise RuntimeError(f"SHA256 mismatch: refusing to execute/load {path}")
    return lock


def original_model(source: Path) -> nn.Module:
    """Import the locked NVIDIA topology with only its Mamba dependency replaced."""
    verify_reference_files(source)
    package_name = "_reuse_verified_upstream"
    package = ModuleType(package_name)
    package.__path__ = [str(source / "models")]
    sys.modules[package_name] = package
    mamba_shim = ModuleType("mamba_ssm")
    mamba_shim.Mamba = ReferenceMamba
    previous_mamba = sys.modules.get("mamba_ssm")
    sys.modules["mamba_ssm"] = mamba_shim
    # Remove an older run's cached reference modules to prevent stale-source reuse.
    for name in tuple(sys.modules):
        if name.startswith(package_name + "."):
            del sys.modules[name]
    try:
        module_name = package_name + ".streaming_generator_SEMamba_time_d1_random_layer_ahead_sep_conv"
        path = source / "models/streaming_generator_SEMamba_time_d1_random_layer_ahead_sep_conv.py"
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        config = json.loads((source / "config.json").read_text(encoding="utf-8"))
        model = module.SEMamba_decoder_list(config)
        model.load_state_dict(load_file(str(source / "model.safetensors")), strict=True)
        return model.eval()
    finally:
        if previous_mamba is None:
            sys.modules.pop("mamba_ssm", None)
        else:
            sys.modules["mamba_ssm"] = previous_mamba


def validate(source: Path, *, sample_rate: int = 16000, exit_layer: int = 3, frames: int = 12, threads: int = 2) -> dict:
    if frames < 4:
        raise ValueError("At least four frames are required to exercise nonzero recurrent state")
    torch.set_num_threads(threads)
    lock = verify_reference_files(source)
    reference = original_model(source)
    port = StreamingReuse(source / "model.safetensors", sample_rate=sample_rate, exit_layer=exit_layer)
    generator = torch.Generator().manual_seed(824231)
    magnitude = torch.rand(1, port.frequency_bins, frames, generator=generator) * 3
    phase = (torch.rand(1, port.frequency_bins, frames, generator=generator) - 0.5) * (2 * math.pi)
    # Include an abrupt silence transition while retaining temporal state.
    magnitude[:, :, -2:] = 0
    phase[:, :, -2:] = 0
    with torch.inference_mode():
        # Compare the original phase-head Cartesian vectors directly. atan2
        # has an ill-conditioned direction at the origin, so tiny accepted
        # float32 differences in near-zero vectors can become much larger
        # unit-vector differences even when reconstructed audio agrees.
        captured = {}
        decoder = reference.phase_decoder_list[exit_layer - 1]
        handles = [
            decoder.phase_conv_r.register_forward_hook(
                lambda module, inputs, output: captured.__setitem__("r", output.detach())
            ),
            decoder.phase_conv_i.register_forward_hook(
                lambda module, inputs, output: captured.__setitem__("i", output.detach())
            ),
        ]
        try:
            expected_mag, expected_phase, _ = reference(magnitude, phase, exit_layer, 0)
        finally:
            for handle in handles:
                handle.remove()
        expected_phase_r = captured["r"][:, 0, :frames, : port.frequency_bins].transpose(1, 2)
        expected_phase_i = captured["i"][:, 0, :frames, : port.frequency_bins].transpose(1, 2)
        states = port.initial_states()
        raw_mag, raw_phase_r, raw_phase_i = [], [], []
        for frame in range(frames):
            result = port.forward_raw(magnitude[:, :, frame], phase[:, :, frame], *states)
            raw_mag.append(result[0])
            raw_phase_r.append(result[1])
            raw_phase_i.append(result[2])
            states = result[3:]
        actual_mag = torch.stack(raw_mag, dim=-1)
        actual_phase_r = torch.stack(raw_phase_r, dim=-1)
        actual_phase_i = torch.stack(raw_phase_i, dim=-1)
        actual_phase = torch.atan2(actual_phase_i, actual_phase_r)
        torch.testing.assert_close(actual_mag, expected_mag, atol=2e-4, rtol=2e-3)
        torch.testing.assert_close(actual_phase_r, expected_phase_r, atol=2e-4, rtol=2e-3)
        torch.testing.assert_close(actual_phase_i, expected_phase_i, atol=2e-4, rtol=2e-3)
        expected_amplitude = torch.expm1(F.relu(expected_mag))
        zero_fraction = (expected_amplitude == 0).float().mean(dim=1, keepdim=True)
        expected_amplitude = torch.where(zero_fraction > 0.5, torch.zeros_like(expected_amplitude), expected_amplitude)
        states = port.initial_states()
        output_real, output_imag = [], []
        for frame in range(frames):
            result = port(magnitude[:, :, frame], phase[:, :, frame], *states)
            output_real.append(result[0])
            output_imag.append(result[1])
            states = result[2:]
        actual_real = torch.stack(output_real, dim=-1)
        actual_imag = torch.stack(output_imag, dim=-1)
        expected_real = expected_amplitude * expected_phase.cos()
        expected_imag = expected_amplitude * expected_phase.sin()
        torch.testing.assert_close(actual_real, expected_real, atol=2e-4, rtol=2e-3)
        torch.testing.assert_close(actual_imag, expected_imag, atol=2e-4, rtol=2e-3)
        for state in states:
            if not torch.isfinite(state).all():
                raise AssertionError("Nonfinite recurrent state")
    return {
        "reference": "SHA256-verified NVIDIA offline topology + independent sequential CPU Mamba-1",
        "source_revision": lock["revision"],
        "source_checkpoint_sha256": lock["files"]["model.safetensors"],
        "sample_rate": sample_rate,
        "exit_layer": exit_layer,
        "frames": frames,
        "raw_log_magnitude_max_abs_error": float((actual_mag - expected_mag).abs().max()),
        "raw_phase_real_max_abs_error": float((actual_phase_r - expected_phase_r).abs().max()),
        "raw_phase_imag_max_abs_error": float((actual_phase_i - expected_phase_i).abs().max()),
        "phase_cosine_max_abs_error": float((actual_phase.cos() - expected_phase.cos()).abs().max()),
        "phase_sine_max_abs_error": float((actual_phase.sin() - expected_phase.sin()).abs().max()),
        "phase_unit_vectors_diagnostic_only": True,
        "phase_acceptance": "Raw Cartesian phase heads and magnitude-weighted complex spectrum, avoiding atan2 origin singularity",
        "spectrum_real_max_abs_error": float((actual_real - expected_real).abs().max()),
        "spectrum_imag_max_abs_error": float((actual_imag - expected_imag).abs().max()),
        "all_states_finite": True,
        "passed": True,
        "scope": "Neural graph only; audio frontend alignment and target-device latency are separate checks",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / ".research/upstream")
    parser.add_argument("--sample-rate", type=int, choices=(8000, 16000), default=16000)
    parser.add_argument("--exit-layer", type=int, choices=range(3, 13), default=3)
    parser.add_argument("--frames", type=int, default=12)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = validate(args.source, sample_rate=args.sample_rate, exit_layer=args.exit_layer, frames=args.frames, threads=args.threads)
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
