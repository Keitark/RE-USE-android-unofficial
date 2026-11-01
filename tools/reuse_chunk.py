# Original model work: Copyright (c) 2025 NVIDIA CORPORATION. All rights reserved.
# Model-specific adaptation: NVIDIA NSCLv1; see licenses/NVIDIA-NSCLv1.txt.
"""Fixed-size, causal multi-hop RE-USE graph with unchanged trained weights.

Derived from NVIDIA Real-time RE-USE; subject to NVIDIA NSCLv1 as described
in reuse_port.py and THIRD_PARTY_NOTICES.md. This vectorizes projections,
convolutions and frequency scans across consecutive frames. Temporal state
still advances in chronological order. Collection adds (T-1)*20 ms latency.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F

from reuse_port import StreamingReuse, _NativeSelectiveStep


class _NativeSelectiveSequence(torch.autograd.Function):
    """Export a chronological temporal sequence without intermediate states.

    Eager execution uses the same equations as the single-step reference. The
    native kernel receives/returns the same initial/final state, keeping every
    intermediate update private to one call rather than writing full tensors.
    """

    @staticmethod
    def forward(ctx, u: Tensor, delta: Tensor, b: Tensor, c: Tensor,
                a: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        observations = []
        for frame in range(u.shape[1]):
            decay = torch.exp(delta[:, frame, :, None] * a)
            drive = (delta[:, frame, :, None] * b[:, frame, None, :]) * u[:, frame, :, None]
            state = decay * state + drive
            observations.append((state * c[:, frame, None, :]).sum(dim=-1))
        return torch.stack(observations, dim=1), state

    @staticmethod
    def symbolic(graph, u, delta, b, c, a, state):
        observed, next_state = graph.op(
            "com.reuse::SelectiveSequence", u, delta, b, c, a, state, outputs=2)
        return observed.setType(u.type()), next_state.setType(state.type())


def mamba_sequence(mamba, x: Tensor, conv_state: Tensor,
                   ssm_state: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Process [frequency, time, channel], retaining the one-hop state ABI."""
    value, gate = mamba.in_proj(x).chunk(2, dim=-1)
    history = torch.cat((conv_state, value.transpose(1, 2)), dim=-1)
    next_conv = history[:, :, -conv_state.shape[-1]:]
    u = F.silu(F.conv1d(history[:, :, 1:], mamba.conv1d.weight,
                        mamba.conv1d.bias, groups=mamba.conv1d.groups)).transpose(1, 2)
    dt_rank = mamba.dt_proj.in_features
    state_size = mamba.A_log.shape[-1]
    dt, b, c = torch.split(mamba.x_proj(u), [dt_rank, state_size, state_size], dim=-1)
    delta = F.softplus(mamba.dt_proj(dt))
    transition = -mamba.A_log.exp()
    if getattr(mamba, "fuse_time_sequence", False):
        observed, state = _NativeSelectiveSequence.apply(u, delta, b, c, transition, ssm_state)
    else:
        observations = []
        state = ssm_state
        for frame in range(x.shape[1]):
            if mamba.fuse_time_step:
                observed, state = _NativeSelectiveStep.apply(
                    u[:, frame], delta[:, frame], b[:, frame], c[:, frame], transition, state)
            else:
                decay = torch.exp(delta[:, frame, :, None] * transition)
                drive = (delta[:, frame, :, None] * b[:, frame, None, :]) * u[:, frame, :, None]
                state = decay * state + drive
                observed = (state * c[:, frame, None, :]).sum(dim=-1)
            observations.append(observed)
        observed = torch.stack(observations, dim=1)
    y = (observed + mamba.D * u) * F.silu(gate)
    return mamba.out_proj(y), next_conv, state


def tf_mamba_chunk(block, x: Tensor, states: Sequence[Tensor], updates: list) -> Tensor:
    batch, channels, frames, bins = x.shape
    temporal = x.permute(0, 3, 2, 1).reshape(batch * bins, frames, channels)
    time_block = block.time_mamba
    y, updates[time_block.conv_slot], updates[time_block.ssm_slot] = mamba_sequence(
        time_block.forward_blocks, temporal, states[time_block.conv_slot],
        states[time_block.ssm_slot])
    temporal = time_block.norm(time_block.output_proj(y + temporal)) + temporal
    frequency = temporal.reshape(batch, bins, frames, channels).permute(0, 2, 1, 3)
    frequency = frequency.reshape(batch * frames, bins, channels)
    frequency = block.freq_mamba(frequency) + frequency
    return frequency.reshape(batch, frames, bins, channels).permute(0, 3, 1, 2)


class ChunkedStreamingReuse(StreamingReuse):
    """Fixed [frames_per_call, bins] features; explicit states match StreamingReuse."""

    def __init__(self, checkpoint=None, *, frames_per_call: int = 5,
                 fuse_time_sequence: bool = False, **kwargs):
        if frames_per_call not in (1, 5):
            raise ValueError("frames_per_call must be 1 or 5")
        if fuse_time_sequence and (frames_per_call != 5 or kwargs.get("scan_mode") != "native"
                                   or not kwargs.get("fuse_time_step", False)):
            raise ValueError("fuse_time_sequence requires five frames, native scan and fuse_time_step")
        self.frames_per_call = frames_per_call
        self.fuse_time_sequence = fuse_time_sequence
        super().__init__(checkpoint, **kwargs)
        for block in self.TSMamba:
            block.time_mamba.forward_blocks.fuse_time_sequence = fuse_time_sequence

    def architecture_manifest(self):
        result = super().architecture_manifest()
        result.update(frames_per_call=self.frames_per_call,
                      maximum_collection_delay_ms=(self.frames_per_call - 1) * 20,
                      processing_interval_ms=self.frames_per_call * 20)
        if self.fuse_time_sequence:
            result["fuse_time_sequence"] = True
        return result

    def forward_raw(self, magnitude: Tensor, phase: Tensor, *states: Tensor) -> tuple[Tensor, ...]:
        if not torch.jit.is_tracing() and not torch.onnx.is_in_onnx_export():
            expected_shape = (self.frames_per_call, self.frequency_bins)
            if tuple(magnitude.shape) != expected_shape or tuple(phase.shape) != expected_shape:
                raise ValueError(f"magnitude and phase must have shape {expected_shape}")
            if len(states) != len(self.state_specs):
                raise ValueError(f"Expected {len(self.state_specs)} state tensors, got {len(states)}")
            for state, spec in zip(states, self.state_specs):
                if tuple(state.shape) != tuple(spec["shape"]):
                    raise ValueError(f"{spec['input']} must have shape {spec['shape']}")
        updates: list[Tensor | None] = [None] * len(self.state_specs)
        x = torch.stack((magnitude, phase), dim=0).unsqueeze(0)
        x = self.dense_encoder(F.pad(x, (0, 2, 0, 0)), states, updates)
        for block in self.TSMamba:
            x = tf_mamba_chunk(block, x, states, updates)
        magnitude = self.mask_decoder_list[self.exit_layer - 1](x, states, updates)
        phase_r, phase_i = self.phase_decoder_list[self.exit_layer - 1](x, states, updates)
        return (magnitude[0, 0, :, :self.frequency_bins],
                phase_r[0, 0, :, :self.frequency_bins],
                phase_i[0, 0, :, :self.frequency_bins], *updates)


__all__ = ["ChunkedStreamingReuse"]
