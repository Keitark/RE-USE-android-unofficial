# Computing-time optimization

The latest [normalization and frequency fusion experiments](compute-fusion.md)
measure another **5.51% less model computing time on the Xperia** for both
fusions together against the earlier temporal graph; the direct host comparison
measures 7.98%. Separate normalization phone screens measure 3.69–6.96%, and the
frequency component measures 2.02% on the phone / 6.82% on the host. These
percentages are separate measurements and are not added. The combined native
pipeline also passes one bounded foreground recorded-audio minute: all 600
calls / 3,000 hops, no output gaps or queue resets, RTF 0.550647. The delayed
tail remains unflushed and live G2 use is untested. The selective
13-region graph and drift-confounded Scan-grain screen are not promoted.
These changes preserve the five-hop buffering contract. The recorded-audio
pass does not establish production readiness or live G2 acceptance.

The earlier [CPU opcode/layout pass](opcode-optimization.md) adds optional
NEON FMA, rank-16 frequency-state registers, temporal Conv/SiLU fusion and
bit-exact FFT index simplification. Its physical-phone shell diagnostics are
kept separate from paced pipeline acceptance. The first all-changes 30.25%
comparison is drift-confounded; the reverse-order repeat measures 18.55% less
model computing time. See that record for exact scope and retained receipts.
That opcode build's complete paced minute processes every expected call without resets,
but has two 10 ms output gaps and fails strict continuity. The 20-second trial
also fails with 42 gaps. These pipeline outcomes remain separate from shell
model computing-time reductions.

The five-hop graph reduces measured host computing time by **27.9% at two
threads and 37.6% at four threads**. It keeps the same trained exit-three
network, advances every audio hop and adds up to 80 ms of input collection delay.
Those percentages are Windows measurements. A further padding/Softplus pass
reduces host computing time by another 11.45% against the previous Sequence
graph in its own matched comparison. The earlier five-hop Step graph
also keeps up for a 20-second paced recorded stream on the Xperia at four threads, averaging
63.99 ms per 100 ms call with no post-startup output gaps or queue resets.
Earlier one-minute runs fail with output gaps and resets. The previous
padding/Softplus pipeline completes all one-minute model work without a reset
but still has 32 output gaps. Continuous operation, live G2 and
microphone-to-ASR latency acceptance remain open.

## Further optimization beyond recurrence

The current optional pack is `reuse-8k-exit3-native-chunk5-sequence-pad-softplus`,
graph SHA256 `5c979c328bde1e3c06735588055c7c1289447d6fa23821f52f053bb8dd30bc56`.
It starts from the previous Sequence graph `4716ad93...`, folds 15 constant-zero
spatial pads into Conv attributes, then replaces nine standard Softplus nodes
with `com.reuse::FastSoftplus` v1. The pad-only intermediate (`1bc5f7cb...`) is
also included. Weights, 23 graph outputs, 21 recurrent state shapes, five-hop
collection delay and trained chronology are unchanged.

Fresh profiling ranked Conv at 21.16%, Softplus at 9.69%, Slice at 7.59%,
Transpose at 6.23% and Concat at 5.09%. Layout/history operators emit about
75 MB of tensor outputs per 100 ms call, compared with 13.30 MB per persistent
state bank. This tensor-volume estimate is not measured memory-bus traffic or
peak allocation; aliases and buffer reuse affect the actual cost. Profiled
timings locate work and are not used for speedup claims.

| Four-thread Windows ORT 1.23.2 comparison | Baseline ms / 20 ms audio | Candidate ms / 20 ms audio | Mean reduction |
|---|---:|---:|---:|
| Sequence -> pad folding | 4.91459 | 4.72773 | 3.80% |
| Sequence -> FastSoftplus | 5.11825 | 4.60231 | 10.08% |
| Sequence -> pad folding + FastSoftplus | 4.80387 | 4.25365 | 11.45% |

Each row is a separate unprofiled ABBA experiment, with the same custom-op DLL
for its baseline and candidate. The combined row represents 800 measured audio
hops per graph after warmup and means 24.02 -> 21.27 ms per 100 ms call.
Percentages from separate experiments must not be added. Receipts are
`.research/profile-otherareas-{pad,softplus,combined}/report.json`.
A five-hop temporal Gather candidate is bit-identical but loses 4.42% in its
matched host run, so it is retained only as a reversible research artifact and
is excluded from the recommended SDK model packs.

FastSoftplus evaluates `max(x,0) + log1p(exp(-abs(x)))` with the existing SIMD
exponential and an odd-power log1p expansion through degree 15. Its polynomial
remainder is below 1.1e-9 in real arithmetic; the implemented float error is
tested separately. SIMD exponential underflow below the minimum normal float
is flushed to zero, matching the existing recurrence kernel. It is an
approximate float implementation, not a bit-identical replacement. Focused
normal/subnormal/nonfinite/shape/alias and 2,048-update recurrence checks pass;
model state and complete speech PCM pass the existing tolerances. No acceptance
tolerance was relaxed. See `.research/softplus-validation.json` and the pack's
`optimization-parity.json`.

The Even native pipeline also replaces per-frame queue allocation with fixed
input/output rings (64/128 frames), reuses stream vectors, and wakes its worker
only when an inference batch is ready. The overload gate remains 50 input
frames; ring capacity does not authorize a longer backlog. A polyphase FIR
path skips discarded downsample outputs and known zero interpolation taps,
retaining the 95 original coefficients and double accumulation order. Its
steady down/up cascade drops from 60,800 to 30,400 multiply-accumulates per
20 ms transport hop. The isolated host FIR benchmark is 37.29 -> 21.15 us
(43.28%); that percentage is not a whole-pipeline improvement. Bit-exact legacy
comparison includes reset and mixed resampler modes.

Native probe sessions can optionally select `default`, `no_spin`,
`bounded_spin` or `dynamic4`. The default leaves ORT settings unchanged;
the others change only the new model session and keep CPU execution. Bounded
spinning requires the original Even ORT 1.27 runtime and requests 1,000 us,
backoff 4, with helper spinning stopped when Run returns. The public AAR stays
on ORT 1.23.2, which does not support this bounded-spin setting. Host screens
show no reliable config gain, and Xperia 20-second default/bounded ABBA results
drift substantially; bounded spinning is not promoted. Provider diagnostics
list CPU and NNAPI availability, but no NNAPI provider is appended or activated.
The first bounded-spin trial also has two output gaps and fails its continuity
audit; its second trial passes. `no_spin` has no physical-phone test in this pass.

The new phone correctness replay passes within one PCM16 step against the
previous Sequence reference (32,000 samples; normalized RMSE 6.15e-7). The cold
paced trial fails with 17 gaps and a 114 ms packet lateness outlier; it is
retained separately from the later warm 20-second passes. Offline parity and
average RTF do not override a failed continuity gate.

The `dynamic4` phone profile completes a warm 20-second run at RTF 0.597002
with zero gaps/resets. Its 60-second follow-up processes all 600 calls/3,000
hops with no queue reset, but still fails: 32 post-startup output gaps and 264
100 ms call deadline misses, mean call 83.65 ms / RTF 0.836540. Maximum packet
lateness is 24.898 ms, with one burst over 10 ms late. All input work is
processed, yet continuous output is not maintained. This is not a controlled
phone speedup against an older build or a sustained acceptance result.
Receipts: `.research/xperia-otherareas-dynamic4-{20s,60s}`.
The four new passing 20-second trials have delivered-prefix parity receipts
within one PCM16 step after their counted startup zeros. They cover delivered
audio, not the pending tail; after a gap, no fixed source-delay value is claimed.

## What changed

Profiling the previous fused-step graph found substantial work in history
slicing/concatenation, padding/layout conversion and repeated small operations.
Its 21 recurrent tensors total 13,298,736 bytes at 8 kHz. Parameter count alone
does not describe the amount of state and intermediate traffic per call.

Two alternatives are included:

| Model pack | Audio per call | Change | Additional collection wait |
|---|---:|---|---:|
| `reuse-8k-exit3-native-step` | 20 ms | Previous fused frequency/temporal kernels | None |
| `reuse-8k-exit3-native-compact` | 20 ms | Gather three referenced temporal taps; fold padding into Conv | None |
| `reuse-8k-exit3-native-chunk5` | 100 ms | Vectorize five chronological hops through the graph | Up to 80 ms |
| `reuse-8k-exit3-native-chunk5-sequence` | 100 ms | Also fuse temporal updates within each layer | Up to 80 ms |

The compact graph avoids padding/reordering unused history between dilated
convolution taps. For example, a 17-frame input uses only temporal positions
0, 8 and 16. Nine history compactions and fifteen padding folds remove that
waste. The recurrent state layout and one-hop API stay unchanged. Its matched
host improvement is modest: about 6.2% at two threads and 3.9% in a repeated
four-thread comparison. An earlier 16.5% result had substantial host variation
and is not a reliable speed claim.

The larger improvement comes from a true five-hop graph. It processes
encoder/decoder and frequency-axis work across five frames together, while
temporal Mamba state advances chronologically through all five hops. This
amortizes repeated dispatch and history work. Calling the old graph five times
in a loop would not provide that benefit. Model weights, active parameter count
(1,823,555), sample rate and trained layers are unchanged. The existing native
SelectiveScan/SelectiveStep implementations are reused.

The sequence variant replaces fifteen temporal Step nodes with three
SelectiveSequence nodes. Each layer keeps its 16-element recurrent state in SIMD
vectors across all five updates, instead of materializing intermediate state
tensors. The first state is loaded once and the final state is stored once per
channel. Every chronological update and observation is retained; trained
weights, public feature/state shapes and collection delay are unchanged.

## Matched host timing

ONNX Runtime 1.23.2 CPU on the Windows host; unprofiled sessions with 40 audio
hops of warmup. The order is baseline, chunked, chunked, baseline. Each model
represents 400 measured audio hops at each thread count: 400 one-hop calls or
80 five-hop calls. Results are normalized by represented audio duration.

| Threads | One-hop mean per 20 ms | Chunked mean per 20 ms | Chunked mean per 100 ms call | Computing-time reduction |
|---:|---:|---:|---:|---:|
| 2 | 10.7623 ms | 7.75955 ms | 38.7977 ms | 27.9% |
| 4 | 9.17842 ms | 5.72644 ms | 28.6322 ms | 37.6% |

The 5.73 ms value is amortized computation for 20 ms of audio. It does not mean
that a five-hop call finishes in 5.73 ms, or that output latency is 5.73 ms.
The application collects 100 ms first and incurs frontend, compute, output
buffer and packet timing as well. Host throughput cannot establish Xperia speed.

The profiler freezes graph, manifest and native-library hashes before running.
The receipt is `.research/profile-opt-chunk5-abba/report.json`; its sanitized
summary is included in the SDK. Profiled operator percentages are used only to
locate work, because instrumentation changes measured time.

A second matched ABBA comparison measures the sequence fusion against the
previous five-hop graph, using the new library for both. Its additional mean
reduction is 3.89% at two threads (7.43506 -> 7.14599 ms per hop) and 5.20% at
four threads (4.95474 -> 4.69695 ms per hop). These are separate measurements;
the percentages are not added to the first table. The receipt is
`.research/profile-opt-sequence-abba/report.json`.

## Earlier physical Xperia paced results

The separate ARM64 probe uses the original Even LC3 codec and ORT 1.27.0 /
C API 23 on the Sony SO-51C, Android 14. It receives five 10 ms codec frames
per 50 ms burst and runs the five-hop models at 8 kHz, using the thread counts
shown below.

| Graph / recorded input | Graph calls / audio hops | Mean per 100 ms call | RTF | Output gaps after startup | Queue resets | Call deadline misses |
|---:|---:|---:|---:|---:|---:|---:|
| Step chunk, 20 s warm / 4 threads | 200 / 1,000 | 63.99 ms | 0.639895 | 0 | 0 | 0 |
| Step chunk, first 60 s warm / 4 threads | 595 / 2,975 | 69.11 ms | 0.691146 | 37 | 1 | 2 |
| Step chunk, 60 s paced revision / 4 threads | 590 / 2,950 | 87.85 ms | 0.878537 | 96 | 2 | 79 |
| Sequence, 2 s cold / 4 threads | 20 / 100 | 87.23 ms | 0.872259 | 0 | 0 | 4 |
| Sequence, 60 s warm / 4 threads | 580 / 2,900 | 85.49 ms | 0.854936 | 148 | 4 | 113 |
| Sequence, 60 s reconfigured / 2 threads | 520 / 2,600 | 114.37 ms | 1.1437 | 514 | 16 | 520 |

Each 60-second run requires 600 calls/3,000 audio hops. Only the 20-second Step
run and short cold runs process their complete expected work without post-startup
gaps/resets. The 20-second run inserts 25 initial silent
10 ms frames (250 ms nominal startup buffering). Individual graph calls are
judged against 100 ms of represented audio; 12.80 ms per 20 ms is an amortized
compute number rather than an individual call's latency.

The first 60-second run fails the full pipeline gate: it discards queued audio during
one queue reset and has 37 output gaps despite average model computing time
below the incoming audio duration. That result is retained. Average RTF alone
is insufficient, and the cause of those gaps is not established by this receipt.

The later probe moves recording-file output outside the timed input loop and
records callback/packet timing. Its 60-second stream still fails despite a
60.000120-second paced phase: maximum burst lateness is 7.063 ms, no burst is
more than 10 ms late, and the longest decoder callback is 23.079 ms. Thus this
run does not show a large packet-feeding pause. It does not prove that file I/O
caused the earlier failure. Computing-time variability and complete pipeline
continuity still require more headroom than the average RTF suggests.

The Sequence four-thread one-minute run also fails despite RTF below one. It
processes only 580/600 calls and has 148 gaps/four resets. Its maximum packet
lateness is 12.856 ms, with one burst over 10 ms late. Reconfiguring the same
process to a new two-thread session is slower: only 520/600 calls are processed,
RTF 1.1437, 514 gaps and 16 resets. The latter has maximum burst lateness
2.187 ms and no burst over 10 ms late. There are no codec/model failures,
unsupported frames or passthrough fallbacks, but none of these one-minute runs
passes the complete work/continuity gate. The probe was force-stopped afterward.

The earlier one-hop phone tests used different thermal/session conditions.
Their 22.15 ms/hop result is historical, not a controlled phone baseline for a
percentage speedup. The host ABBA comparison provides the matched percentage.
This recorded-audio result does not establish longer thermal endurance,
background operation, live G2 connectivity or stock ASR behavior.

The paced output length includes startup zeros and the stock native hook does
not flush its delayed tail at session end. Nominal zero-frame buffering and
cumulative source-age counters are not measured microphone-to-ASR wall latency.
The independent offline replay checks the complete waveform separately.
For the three short passing runs, a separate continuity comparison verifies the
counted startup zeros and every delivered enhanced sample against aligned fresh
native replay. Those prefixes match exactly (zero PCM16-step error). This checks
ordering/continuity of delivered audio, not the undelivered final tail.

## Correctness and latency

The five-hop export is checked against five chronological calls of the one-hop
model, with separate recurrent feedback. Across 40 audio hops, maximum ORT
spectrum difference is 2.54e-6 and maximum state difference is 5.61e-5. Static
float input/output metadata and all 21 state shapes are checked. The existing
tolerances were retained.

A 32,000-sample host speech comparison passes within one PCM16 step, with
RMSE 4.18e-7 normalized. The actual ARM64 codec/model implementation under
emulator translation also passes direct PCM parity within one step (RMSE
3.81e-7), using the chunk graph against the one-hop reference. That emulator's
paced stream fails and supplies no physical-phone speed claim.

The public Java chunk API also passes an actual Android neural/DSP/lifecycle
test for the earlier five-hop Step graph on an x86_64 emulator: 16,000 samples
and 20 model calls, maximum PCM error
1.07e-6/RMSE 8.13e-8 against the independent serial fixture. Reset, flush/counts,
invalid input and closed-resource checks pass. The latest full host suite passes
151 Python and 27 JVM tests; release lint/build/local Maven publication pass.
Those checks establish correctness and packaging, not optimized phone speed.

The new sequence kernel has 15 focused cases in addition to the earlier native
tests. Its T=1/T=5 observations and final states match chronological native Step
calls bit-for-bit on Windows. The exported sequence graph agrees with the
one-hop reference over 40 hops within the same spectrum/state tolerances; on
recorded speech, it is bit-identical to the previous five-hop graph for all
23 outputs and recurrent states. Its physical cold two-second replay also
passes direct PCM parity within one PCM16 step (RMSE 4.83e-7 normalized).

`ChunkedReuseEnhancer.process()` accepts exactly 100 ms of mono float PCM at the
model rate. Its first call returns 80 ms; later calls return 100 ms. `flush()`
returns the final 20 ms, preserving total length. There is no additional neural
look-ahead, but collecting a chunk adds up to 80 ms beyond the ordinary one-hop
buffering. The native Even hook additionally has a bounded output prefill and
does not yet have a reliable stock microphone-stop tail flush.

## Reproduce

```powershell
.\.venv\Scripts\python.exe tools\export_model.py --sample-rate 8000 --exit-layer 3 --scan-mode native --fuse-time-step --frames-per-call 5 --validation-frames 40 --custom-op-library native\build-windows\Release\reuse_ort_ops.dll --output models\reuse-8k-exit3-native-chunk5
.\.venv\Scripts\python.exe tools\export_model.py --sample-rate 8000 --exit-layer 3 --scan-mode native --fuse-time-step --fuse-time-sequence --frames-per-call 5 --validation-frames 40 --custom-op-library native\build-windows\Release\reuse_ort_ops.dll --output models\reuse-8k-exit3-native-chunk5-sequence
.\.venv\Scripts\python.exe tools\profile_optimizations.py --model-dir models\reuse-8k-exit3-native-step --compare-model-dir models\reuse-8k-exit3-native-chunk5 --custom-op-library native\build-windows\Release\reuse_ort_ops.dll --output .research\YOUR-NEW-COMPARISON --threads 2 4 --warmup 40 --frames 200
```

Use the new public Java API shown in [README.md](../README.md). Retain the model
and native-kernel NVIDIA noncommercial research license. Remaining acceptance
includes longer phone sessions, live G2 operation, reliable tail/lifecycle
handling and explicit microphone-to-ASR latency/quality evaluation.

To reproduce the additional pad/Softplus pack, first rebuild the host custom-op
library from this source version, then write fresh output directories:

```powershell
pwsh -NoProfile -File tools\build-native.ps1
.\.venv\Scripts\python.exe tools\optimize_dense_history.py --source-dir models\reuse-8k-exit3-native-chunk5-sequence --output-dir models\YOUR-NEW-SEQUENCE-PAD --mode pads --custom-op-library native\build-windows\Release\reuse_ort_ops.dll --speech-pcm .research\even-probe\input.pcm16le
.\.venv\Scripts\python.exe tools\optimize_softplus.py --source-dir models\YOUR-NEW-SEQUENCE-PAD --output-dir models\YOUR-NEW-SEQUENCE-PAD-SOFTPLUS --custom-op-library native\build-windows\Release\reuse_ort_ops.dll --speech-pcm .research\even-probe\input.pcm16le
.\.venv\Scripts\python.exe tools\profile_optimizations.py --model-dir models\reuse-8k-exit3-native-chunk5-sequence --compare-model-dir models\YOUR-NEW-SEQUENCE-PAD-SOFTPLUS --custom-op-library native\build-windows\Release\reuse_ort_ops.dll --output .research\YOUR-NEW-PAD-SOFTPLUS-ABBA --threads 4 --warmup 40 --frames 400
```

This pass used a separately built, frozen DLL (`c4248dd5...`) for both sides of
the final comparison. Do not overwrite a retained comparison's snapshots or
run competing CPU benchmarks during timing.
