"""Portable, explicit-state PyTorch graph for NVIDIA Real-time RE-USE.

The network topology and checkpoint naming are derived from NVIDIA's
``nvidia/Real-time_RE-USE`` release. Copyright (c) 2025 NVIDIA CORPORATION.
The upstream checkpoint and derived model are subject to NVIDIA's One-Way
Noncommercial License (NSCLv1); see THIRD_PARTY_NOTICES.md. This module does
not make the upstream model commercially licensed.

No CUDA, mamba_ssm, librosa, or torchaudio is required. The graph consumes
one log1p-magnitude/phase STFT frame and returns one complex spectrum plus
all next-state tensors. STFT and overlap-add belong to the audio frontend.
Frequency-axis selective scans use a stable associative prefix operation
expressible as ordinary ONNX ops. The sequential mode is a test reference.
The optional native scan mode exports a com.reuse::SelectiveScan node and
requires the accompanying reuse_ort_ops runtime library.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_DIM = 64
_INNER = 256
_STATE = 16
_CONV = 4
_DT_RANK = 4


def selective_scan(a: Tensor, b: Tensor, mode: str = "parallel") -> Tensor:
    """Solve h[t] = a[t]*h[t-1] + b[t], h[-1]=0, without division.

    Input/output layout is [batch, sequence, inner, state]. The parallel
    implementation composes adjacent affine maps recursively, then fills
    the intervening prefix values. This work-efficient tree moves O(length)
    elements, avoiding the O(length*log(length)) full-tensor copies of a
    recursive-doubling scan. No division is used, even when decay underflows.
    The sequence length is fixed when exporting a deployment graph.
    """
    length = a.shape[1]
    if mode == "sequential":
        h = torch.zeros_like(b[:, 0])
        history = []
        for index in range(length):
            h = a[:, index] * h + b[:, index]
            history.append(h)
        return torch.stack(history, dim=1)
    if mode != "parallel":
        raise ValueError("scan_mode must be 'parallel' or 'sequential'")
    if length == 1:
        return b
    # Pair (0,1), (2,3), ... . The affine map for the later position
    # composes AFTER the earlier one: (ao*ae, bo + ao*be).
    pairs = length // 2
    paired_end = pairs * 2
    even_a = a[:, 0:paired_end:2]
    odd_a = a[:, 1:paired_end:2]
    even_b = b[:, 0:paired_end:2]
    odd_b = b[:, 1:paired_end:2]
    odd_prefix = selective_scan(odd_a * even_a, odd_b + odd_a * even_b, "parallel")

    # h[2k] = b[2k] + a[2k]*h[2k-1], with h[-1]=0.
    # odd_prefix already contains h[1], h[3], ... .
    # With one pair the tail is empty. Avoid materializing zero-length tensors
    # in the fixed-shape ONNX graph; the only even prefix is exactly b[0].
    if pairs == 1:
        even_prefix = even_b
    else:
        even_prefix = torch.cat(
            (even_b[:, :1], even_b[:, 1:] + even_a[:, 1:] * odd_prefix[:, :-1]),
            dim=1,
        )
    interleaved = torch.stack((even_prefix, odd_prefix), dim=2).flatten(1, 2)
    if length % 2:
        last = b[:, -1:] + a[:, -1:] * odd_prefix[:, -1:]
        interleaved = torch.cat((interleaved, last), dim=1)
    return interleaved


class _NativeSelectiveScan(torch.autograd.Function):
    """Export boundary for the optional ORT frequency-scan CPU kernel.

    The eager forward remains the portable tree reference. The runtime
    extension receives post-softplus delta and A already converted to the
    negative transition coefficients; it owns no persistent stream state.
    """

    @staticmethod
    def forward(ctx, u: Tensor, delta: Tensor, b: Tensor, c: Tensor, a: Tensor) -> Tensor:
        decay = torch.exp(delta.unsqueeze(-1) * a)
        drive = delta.unsqueeze(-1) * b.unsqueeze(-2) * u.unsqueeze(-1)
        states = selective_scan(decay, drive, "parallel")
        return (states * c.unsqueeze(-2)).sum(dim=-1)

    @staticmethod
    def symbolic(graph, u, delta, b, c, a):
        # The observed output has the same [batch, frequency, inner] shape
        # and floating-point element type as u.
        return graph.op("com.reuse::SelectiveScan", u, delta, b, c, a).setType(u.type())


class _NativeSelectiveStep(torch.autograd.Function):
    """Optional fused temporal update; eager execution is the exact reference."""

    @staticmethod
    def forward(ctx, u: Tensor, delta: Tensor, b: Tensor, c: Tensor,
                a: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        decay = torch.exp(delta.unsqueeze(-1) * a)
        drive = delta.unsqueeze(-1) * b.unsqueeze(-2) * u.unsqueeze(-1)
        next_state = decay * state + drive
        return (next_state * c.unsqueeze(-2)).sum(dim=-1), next_state

    @staticmethod
    def symbolic(graph, u, delta, b, c, a, state):
        observed, next_state = graph.op("com.reuse::SelectiveStep", u, delta, b, c, a, state, outputs=2)
        return observed.setType(u.type()), next_state.setType(state.type())


class _StateLayout:
    def __init__(self) -> None:
        self.specs: list[dict[str, Any]] = []

    def add(self, label: str, shape: Sequence[int]) -> int:
        index = len(self.specs)
        self.specs.append(
            {
                "input": f"state_{index:03d}",
                "output": f"next_state_{index:03d}",
                "shape": list(shape),
                "dtype": "float32",
                "label": label,
            }
        )
        return index


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels: int = _DIM) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels, eps=1e-5)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DenseBlock(nn.Module):
    def __init__(self, layout: _StateLayout, bins: int, label: str) -> None:
        super().__init__()
        self.dense_block = nn.ModuleList()
        self.state_indices: list[int] = []
        self.cache_lengths: list[int] = []
        for depth in range(4):
            dilation = 2**depth
            channels = _DIM * (depth + 1)
            self.dense_block.append(
                nn.Sequential(
                    nn.Identity(),  # Preserve original state_dict indices.
                    nn.Conv2d(channels, _DIM, (3, 3), dilation=(dilation, 1)),
                    ChannelLayerNorm(),
                    nn.PReLU(_DIM),
                )
            )
            cache_length = 2 * dilation
            self.cache_lengths.append(cache_length)
            self.state_indices.append(
                layout.add(f"{label}.conv{depth}", (1, channels, cache_length, bins))
            )

    def forward(self, x: Tensor, states: Sequence[Tensor], updates: list) -> Tensor:
        skip = x
        for index, block in enumerate(self.dense_block):
            slot = self.state_indices[index]
            cache_length = self.cache_lengths[index]
            history = torch.cat((states[slot], skip), dim=2)
            updates[slot] = history[:, :, -cache_length:, :]
            x = block[1](F.pad(history, (1, 1, 0, 0)))
            x = block[3](block[2](x))
            skip = torch.cat((x, skip), dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, layout: _StateLayout, padded_bins: int) -> None:
        super().__init__()
        self.dense_conv_1_1 = nn.Sequential(
            nn.Conv2d(2, _DIM, (3, 3)), ChannelLayerNorm(), nn.PReLU(_DIM)
        )
        self.input_slot = layout.add("encoder.input", (1, 2, 2, padded_bins))
        self.dense_block = DenseBlock(layout, padded_bins, "encoder.dense")
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(_DIM, _DIM, (1, 3), stride=(1, 2)),
            ChannelLayerNorm(),
            nn.PReLU(_DIM),
        )

    def forward(self, x: Tensor, states: Sequence[Tensor], updates: list) -> Tensor:
        history = torch.cat((states[self.input_slot], x), dim=2)
        updates[self.input_slot] = history[:, :, -2:, :]
        x = self.dense_conv_1_1(F.pad(history, (1, 1, 0, 0)))
        x = self.dense_block(x, states, updates)
        return self.dense_conv_2(x)


class Mamba(nn.Module):
    """Mamba-1 equations with original parameter names, entirely in PyTorch."""

    def __init__(self, scan_mode: str) -> None:
        super().__init__()
        self.scan_mode = scan_mode
        self.fuse_time_step = False
        self.in_proj = nn.Linear(_DIM, 2 * _INNER, bias=False)
        self.conv1d = nn.Conv1d(
            _INNER, _INNER, _CONV, padding=_CONV - 1, groups=_INNER, bias=True
        )
        self.x_proj = nn.Linear(_INNER, _DT_RANK + 2 * _STATE, bias=False)
        self.dt_proj = nn.Linear(_DT_RANK, _INNER, bias=True)
        self.A_log = nn.Parameter(torch.arange(1, _STATE + 1).float().log().repeat(_INNER, 1))
        self.D = nn.Parameter(torch.ones(_INNER))
        self.out_proj = nn.Linear(_INNER, _DIM, bias=False)

    def _coefficients(self, u: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        dt, b, c = torch.split(self.x_proj(u), [_DT_RANK, _STATE, _STATE], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        a = torch.exp(delta.unsqueeze(-1) * (-torch.exp(self.A_log)))
        drive = delta.unsqueeze(-1) * b.unsqueeze(-2) * u.unsqueeze(-1)
        return a, drive, c

    def step(self, x: Tensor, conv_state: Tensor, ssm_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        x, z = self.in_proj(x).chunk(2, dim=-1)
        next_conv = torch.cat((conv_state[:, :, 1:], x.unsqueeze(-1)), dim=-1)
        u = F.silu(
            (next_conv * self.conv1d.weight.squeeze(1)).sum(dim=-1) + self.conv1d.bias
        )
        if self.fuse_time_step:
            dt, b, c = torch.split(self.x_proj(u), [_DT_RANK, _STATE, _STATE], dim=-1)
            delta = F.softplus(self.dt_proj(dt))
            observed, next_ssm = _NativeSelectiveStep.apply(u, delta, b, c, -torch.exp(self.A_log), ssm_state)
        else:
            a, drive, c = self._coefficients(u)
            next_ssm = a * ssm_state + drive
            observed = (next_ssm * c.unsqueeze(-2)).sum(dim=-1)
        y = (observed + self.D * u) * F.silu(z)
        return self.out_proj(y), next_conv, next_ssm

    def forward(self, x: Tensor) -> Tensor:
        x, z = self.in_proj(x).chunk(2, dim=-1)
        length = x.shape[1]
        u = F.silu(self.conv1d(x.transpose(1, 2))[:, :, :length].transpose(1, 2))
        if self.scan_mode == "native":
            dt, b, c = torch.split(self.x_proj(u), [_DT_RANK, _STATE, _STATE], dim=-1)
            delta = F.softplus(self.dt_proj(dt))
            observed = _NativeSelectiveScan.apply(u, delta, b, c, -torch.exp(self.A_log))
        else:
            a, drive, c = self._coefficients(u)
            h = selective_scan(a, drive, self.scan_mode)
            observed = (h * c.unsqueeze(-2)).sum(dim=-1)
        y = (observed + self.D * u) * F.silu(z)
        return self.out_proj(y)


class CausalMambaBlock(nn.Module):
    def __init__(self, layout: _StateLayout, bins: int, label: str, scan_mode: str) -> None:
        super().__init__()
        self.forward_blocks = Mamba(scan_mode)
        self.output_proj = nn.Linear(_DIM, _DIM)
        self.norm = nn.LayerNorm(_DIM, eps=1e-5)
        self.conv_slot = layout.add(f"{label}.conv", (bins, _INNER, _CONV))
        self.ssm_slot = layout.add(f"{label}.ssm", (bins, _INNER, _STATE))

    def forward(self, x: Tensor, states: Sequence[Tensor], updates: list) -> Tensor:
        y, updates[self.conv_slot], updates[self.ssm_slot] = self.forward_blocks.step(
            x, states[self.conv_slot], states[self.ssm_slot]
        )
        return self.norm(self.output_proj(y + x))


class BidirectionalMambaBlock(nn.Module):
    def __init__(self, scan_mode: str) -> None:
        super().__init__()
        self.forward_blocks = Mamba(scan_mode)
        self.backward_blocks = Mamba(scan_mode)
        self.output_proj = nn.Linear(2 * _DIM, _DIM)
        self.norm = nn.LayerNorm(_DIM, eps=1e-5)

    def forward(self, x: Tensor) -> Tensor:
        forward = self.forward_blocks(x) + x
        reversed_x = torch.flip(x, (1,))
        backward = torch.flip(self.backward_blocks(reversed_x) + reversed_x, (1,))
        return self.norm(self.output_proj(torch.cat((forward, backward), dim=-1)))


class TFMambaBlock(nn.Module):
    def __init__(self, layout: _StateLayout, bins: int, layer: int, scan_mode: str) -> None:
        super().__init__()
        self.bins = bins
        self.time_mamba = CausalMambaBlock(layout, bins, f"mamba{layer}.time", scan_mode)
        self.freq_mamba = BidirectionalMambaBlock(scan_mode)

    def forward(self, x: Tensor, states: Sequence[Tensor], updates: list) -> Tensor:
        x = x.squeeze(2).transpose(1, 2).reshape(self.bins, _DIM)
        x = (self.time_mamba(x, states, updates) + x).reshape(1, self.bins, _DIM)
        x = self.freq_mamba(x) + x
        return x.transpose(1, 2).unsqueeze(2)


class SPConvTranspose2d(nn.Module):
    def __init__(self, r: int, padding: tuple[int, int, int, int]) -> None:
        super().__init__()
        self.pad = nn.ConstantPad2d(padding, 0.0)
        self.r = r
        self.conv = nn.Conv2d(_DIM, _DIM * r, (1, 3))

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(self.pad(x))
        batch, channels, height, width = x.shape
        return (
            x.reshape(batch, self.r, channels // self.r, height, width)
            .permute(0, 2, 3, 4, 1)
            .reshape(batch, channels // self.r, height, width * self.r)
        )


class _DecoderBase(nn.Module):
    def __init__(self, layout: _StateLayout, bins: int, label: str) -> None:
        super().__init__()
        self.dense_block = DenseBlock(layout, bins, f"{label}.dense")
        self.up_conv1 = nn.Sequential(
            SPConvTranspose2d(2, (1, 1, 0, 0)), ChannelLayerNorm(), nn.PReLU(_DIM)
        )
        self.up_conv2 = nn.Sequential(
            SPConvTranspose2d(1, (2, 0, 0, 0)), ChannelLayerNorm(), nn.PReLU(_DIM)
        )
        self.up_slot = layout.add(f"{label}.upconv2", (1, _DIM, bins * 2, 2))

    def features(self, x: Tensor, states: Sequence[Tensor], updates: list) -> Tensor:
        x = self.up_conv1(self.dense_block(x, states, updates)).permute(0, 1, 3, 2)
        history = torch.cat((states[self.up_slot], x), dim=3)
        updates[self.up_slot] = history[:, :, :, -2:]
        x = self.up_conv2[0].conv(history)
        x = self.up_conv2[2](self.up_conv2[1](x))
        return x.permute(0, 1, 3, 2)


class MagDecoder(_DecoderBase):
    def __init__(self, layout: _StateLayout, bins: int) -> None:
        super().__init__(layout, bins, "mag")
        self.final_conv = nn.Conv2d(_DIM, 1, (1, 1))

    def forward(self, x: Tensor, states: Sequence[Tensor], updates: list) -> Tensor:
        return self.final_conv(self.features(x, states, updates))


class PhaseDecoder(_DecoderBase):
    def __init__(self, layout: _StateLayout, bins: int) -> None:
        super().__init__(layout, bins, "phase")
        self.phase_conv_r = nn.Conv2d(_DIM, 1, (1, 1))
        self.phase_conv_i = nn.Conv2d(_DIM, 1, (1, 1))

    def forward(self, x: Tensor, states: Sequence[Tensor], updates: list) -> tuple[Tensor, Tensor]:
        features = self.features(x, states, updates)
        return self.phase_conv_r(features), self.phase_conv_i(features)


class StreamingReuse(nn.Module):
    """One 20 ms frame at a time, mono, 8 or 16 kHz, zero look-ahead.

    ``checkpoint`` can be a safetensors path, a directory containing
    model.safetensors, a state dict, or None (random weights for unit tests).
    Loading validates all active weights strictly and rejects unexpected
    keys except the known pruned branches/layers/decoders.
    """

    def __init__(
        self,
        checkpoint: str | Path | Mapping[str, Tensor] | None = None,
        *,
        sample_rate: int = 16000,
        exit_layer: int = 3,
        scan_mode: str = "parallel",
        lookahead: int = 0,
        fuse_time_step: bool = False,
    ) -> None:
        super().__init__()
        if sample_rate not in (8000, 16000):
            raise ValueError("This port supports sample_rate=8000 or 16000")
        if not 3 <= exit_layer <= 12:
            raise ValueError("exit_layer must be between 3 and 12")
        if lookahead != 0:
            raise ValueError("This fixed-state Android port currently supports lookahead=0 only")
        if scan_mode not in ("parallel", "sequential", "native"):
            raise ValueError("scan_mode must be 'parallel', 'sequential', or 'native'")
        if fuse_time_step and scan_mode != "native":
            raise ValueError("fuse_time_step requires scan_mode='native'")
        self.sample_rate = sample_rate
        self.exit_layer = exit_layer
        self.scan_mode = scan_mode
        self.lookahead = lookahead
        self.fuse_time_step = fuse_time_step
        self.n_fft = sample_rate * 40 // 1000
        self.hop_size = sample_rate * 20 // 1000
        self.frequency_bins = self.n_fft // 2 + 1
        self.encoded_bins = (self.frequency_bins + 2 - 3) // 2 + 1
        layout = _StateLayout()
        self.dense_encoder = DenseEncoder(layout, self.frequency_bins + 2)
        self.TSMamba = nn.ModuleList(
            [TFMambaBlock(layout, self.encoded_bins, layer, scan_mode) for layer in range(exit_layer)]
        )
        for block in self.TSMamba:
            block.time_mamba.forward_blocks.fuse_time_step = fuse_time_step
        self.mask_decoder_list = nn.ModuleList(
            [nn.Identity() for _ in range(exit_layer - 1)] + [MagDecoder(layout, self.encoded_bins)]
        )
        self.phase_decoder_list = nn.ModuleList(
            [nn.Identity() for _ in range(exit_layer - 1)] + [PhaseDecoder(layout, self.encoded_bins)]
        )
        self.state_specs = layout.specs
        self.checkpoint_loaded = False
        self.checkpoint_parameter_count: int | None = None
        if checkpoint is not None:
            self.load_original_checkpoint(checkpoint)
        self.eval()

    def load_original_checkpoint(self, checkpoint: str | Path | Mapping[str, Tensor]) -> None:
        if isinstance(checkpoint, (str, Path)):
            from safetensors.torch import load_file

            path = Path(checkpoint)
            if path.is_dir():
                path = path / "model.safetensors"
            original = load_file(str(path), device="cpu")
        else:
            original = dict(checkpoint)
        expected = set(self.state_dict())
        missing = expected - set(original)
        if missing:
            raise ValueError(f"Checkpoint missing active weights: {sorted(missing)}")
        unexpected = []
        for name in set(original) - expected:
            if name.startswith(("mask_decoder.", "phase_decoder.", "dense_encoder.dense_conv_1_2.", "dense_encoder.dense_conv_1_3.")):
                continue
            parts = name.split(".")
            if len(parts) > 1 and parts[0] == "TSMamba" and parts[1].isdigit() and self.exit_layer <= int(parts[1]) < 12:
                continue
            if len(parts) > 1 and parts[0] in ("mask_decoder_list", "phase_decoder_list") and parts[1].isdigit() and 0 <= int(parts[1]) < 12 and int(parts[1]) != self.exit_layer - 1:
                continue
            unexpected.append(name)
        if unexpected:
            raise ValueError(f"Unexpected checkpoint keys: {sorted(unexpected)}")
        self.load_state_dict({key: original[key] for key in expected}, strict=True)
        self.checkpoint_parameter_count = sum(value.numel() for value in original.values())
        self.checkpoint_loaded = True

    def initial_states(self, *, device: str | torch.device | None = None, dtype: torch.dtype = torch.float32) -> tuple[Tensor, ...]:
        return tuple(torch.zeros(spec["shape"], device=device, dtype=dtype) for spec in self.state_specs)

    def architecture_manifest(self) -> dict[str, Any]:
        return {
            "architecture": "Real-time RE-USE / Mamba-1 explicit state",
            "upstream_model": "nvidia/Real-time_RE-USE",
            "license": "NVIDIA One-Way Noncommercial License (NSCLv1)",
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_size": self.hop_size,
            "frequency_bins": self.frequency_bins,
            "encoded_bins": self.encoded_bins,
            "exit_layer": self.exit_layer,
            "lookahead_frames": 0,
            "algorithmic_latency_ms": 40,
            "scan_mode": self.scan_mode,
            "fuse_time_step": self.fuse_time_step,
            "active_parameters": sum(parameter.numel() for parameter in self.parameters()),
            "checkpoint_parameters": self.checkpoint_parameter_count,
            "checkpoint_loaded": self.checkpoint_loaded,
            "state_bytes": sum(math.prod(spec["shape"]) for spec in self.state_specs) * 4,
            "state_specs": self.state_specs,
            "input_domain": "log1p(abs(unnormalized periodic-Hann STFT)), phase radians",
            "output_domain": "unnormalized complex STFT real/imaginary components",
        }

    def forward_raw(self, magnitude: Tensor, phase: Tensor, *states: Tensor) -> tuple[Tensor, ...]:
        if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            expected_shape = (1, self.frequency_bins)
            if tuple(magnitude.shape) != expected_shape or tuple(phase.shape) != expected_shape:
                raise ValueError(f"magnitude and phase must have shape {expected_shape}")
            if len(states) != len(self.state_specs):
                raise ValueError(f"Expected {len(self.state_specs)} state tensors, got {len(states)}")
            for state, spec in zip(states, self.state_specs):
                if tuple(state.shape) != tuple(spec["shape"]):
                    raise ValueError(f"{spec['input']} must have shape {spec['shape']}")
        updates: list[Tensor | None] = [None] * len(self.state_specs)
        x = torch.stack((magnitude, phase), dim=1).unsqueeze(2)
        x = F.pad(x, (0, 2, 0, 0))
        x = self.dense_encoder(x, states, updates)
        for block in self.TSMamba:
            x = block(x, states, updates)
        magnitude = self.mask_decoder_list[self.exit_layer - 1](x, states, updates)
        phase_r, phase_i = self.phase_decoder_list[self.exit_layer - 1](x, states, updates)
        magnitude = magnitude[:, 0, 0, : self.frequency_bins]
        phase_r = phase_r[:, 0, 0, : self.frequency_bins]
        phase_i = phase_i[:, 0, 0, : self.frequency_bins]
        return magnitude, phase_r, phase_i, *updates

    def forward(self, magnitude: Tensor, phase: Tensor, *states: Tensor) -> tuple[Tensor, ...]:
        raw_magnitude, phase_r, phase_i, *updates = self.forward_raw(magnitude, phase, *states)
        amplitude = torch.expm1(F.relu(raw_magnitude))
        # Match the release's sweep-artifact suppression, for every frame.
        zero_fraction = (amplitude == 0).to(amplitude.dtype).mean(dim=1, keepdim=True)
        amplitude = torch.where(zero_fraction > 0.5, torch.zeros_like(amplitude), amplitude)
        norm_squared = phase_r.square() + phase_i.square()
        nonzero = norm_squared > 0
        denominator = torch.sqrt(torch.where(nonzero, norm_squared, torch.ones_like(norm_squared)))
        cosine = torch.where(nonzero, phase_r / denominator, torch.ones_like(phase_r))
        sine = torch.where(nonzero, phase_i / denominator, torch.zeros_like(phase_i))
        return amplitude * cosine, amplitude * sine, *updates


__all__ = ["StreamingReuse", "selective_scan"]
