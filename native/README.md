# Native CPU kernels

The optional `reuse_ort_ops` library registers `com.reuse::SelectiveScan` with
ONNX Runtime C API 23 (retained headers; ordinary Android runtime 1.29.0).
It fuses the frequency-axis recurrence
and observation into one CPU kernel without materializing a
`[batch, length, inner, state]` tensor for each scan stage.

Inputs are float32 `u[B,L,I]`, post-softplus `delta[B,L,I]`, `B[B,L,N]`,
`C[B,L,N]`, and already-negative `A[I,N]`. Output is `[B,L,I]`:

```
h = 0
for each frequency position t:
    h = exp(delta[t] * A) * h + delta[t] * B[t] * u[t]
    output[t] = sum(h * C[t], state_axis)
```

The frequency scan starts from zero on every invocation. This is separate
from the temporal recurrent states exposed in the model manifest. `D*u` and
the SiLU gate remain outside this operator.

`com.reuse::SelectiveStep` performs one temporal update with explicit initial
and final state. `com.reuse::SelectiveSequence` performs several chronological
temporal updates in one kernel. Its float32 inputs are `u/delta[B,T,I]`,
`B/C[B,T,N]`, `A[I,N]` and `state[B,I,N]`; outputs are observations `[B,T,I]`
and final state `[B,I,N]`. The state transition and arithmetic order match
`SelectiveStep`. The N=16 implementation retains four SIMD state vectors across
the sequence, avoiding intermediate state tensors. Other bounded state ranks
use reusable stack scratch. Input buffers are never mutated and overlapping
output/input buffers are rejected.

The sequence graph still processes every audio hop and retains the same
trained weights. Five-hop batching adds up to80ms of collection delay; it
does not add neural look-ahead. See `../docs/optimization.md` for measured
performance and the exact hardware validation scope.

`com.reuse::FastSoftplus` is an optional float32 activation kernel. It uses
`max(x,0) + log1p(exp(-abs(x)))` with the existing SIMD exponential and a
bounded polynomial for log1p on its reduced interval. The real-arithmetic
series remainder is below1.1e-9; float error, nonfinite behavior and recurrent
model/PCM parity are checked separately without widening tolerances. This
changes floating-point rounding, not trained weights. Models must explicitly
declare this operation and use the matching AAR/native library.

Work is partitioned across independent groups of 16 inner channels using
ORT's existing intra-op pool; no extra worker pool is created. Current state
requires `B*I*N` floats. AArch64 uses NEON, x86_64 uses SSE2, and other targets
use scalar `std::exp`.

The SIMD exponential uses nearest-integer log2 range reduction, a split
ln(2), and a degree-7 Taylor polynomial on `[-ln(2)/2,ln(2)/2]`. It flushes
results below the minimum normal float to zero. Model exponents are
nonpositive. `tests/test_native_scan.py` measures relative error below 1e-6
over 10,001 exponents from -80 to 0 and tests the complete recurrence against
an independent NumPy reference. Model-level recurrent parity remains an
additional required check after export. No unsafe fast-math compiler flag
is enabled.

Windows build:

```powershell
pwsh -NoProfile -File tools/build-native.ps1
.venv/Scripts/python.exe -m pytest tests/test_native_scan.py -q
```

`-ExactExp` builds an independent scalar-math baseline in
`native/build-windows-exact`. Set `REUSE_CUSTOM_OP_LIBRARY` to that DLL to
run the same tests. Android Gradle builds the same CMake target for
`arm64-v8a` and `x86_64`. All configurations, including Debug, compile the
kernel at `-O3`; Android shared objects use a 16 KiB maximum page size.

`include/` contains vendored MIT-licensed ONNX Runtime headers from
`microsoft/onnxruntime` tag `v1.23.2`; exact hashes are recorded in
`headers-lock.json`, and their license is `include/LICENSE`.
There is no link-time dependency on ORT: the host runtime passes its C API
to `RegisterCustomOps`.

This model-specific kernel is distributed under NVIDIA NSCLv1 as described
in its source header and `../THIRD_PARTY_NOTICES.md`.

Validation scope: Windows x86_64/SSE2 numerical tests have executed. Android
`arm64-v8a`/NEON and `x86_64` shared libraries have compiled with exported
registration symbols and 16 KiB LOAD alignment. Compilation alone does not
verify ARM64 numerical output or physical-phone real-time performance.
