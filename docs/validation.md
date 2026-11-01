# Validation record

Validation date: 2026-11-01 (Asia/Tokyo). The exported model uses the locked
NVIDIA Real-time RE-USE revision in `tools/source-lock.json`, exit layer 3,
zero look-ahead. The native 8/16 kHz graphs contain 1,823,555 active parameters.

## Current acceptance

The [live G2 Hub companion](live-g2.md) now passes a bounded asynchronous
capture on the physical Xperia: 872,000 raw PCM samples (54.5 s), all processed,
and 872,188 enhanced samples including the finite FIR tail. It reports zero
app/transport drops or queue overflows, RTF 0.759731 and 67/546 calls over
100 ms. Thus average throughput passes, while all-call deadline acceptance
does not. The microphone was closed and the companion force-stopped after
capture. Source timestamps/sample indexes are unavailable, so this does not
attest BLE-gap-free input or longer/background behavior.

Independent whole-waveform validation includes all 872,188 output samples and
the complete finite FIR tail. Against the original Sequence graph, NumPy
frontend and double-accumulation FIR, maximum error is one PCM16 step and RMSE
6.07e-7, within the unchanged tolerances. This is numerical parity, not an
ASR, noise-reduction or speech-quality score. Both PCM recordings remain local
and are excluded from distribution.

The earlier `f950dcfb...` compute-fusion codec probe passes its first bounded physical
Xperia minute: all 600 calls/3,000 hops complete, with zero output gaps, queue
resets or 100 ms call misses. Mean RTF is 0.550647 (55.0647 ms per 100 ms call).
The auditor confirms the expected graph/APK, ARM64 physical device, complete
input work, awake/unlocked endpoints and matching runtime profile. The result
is a foreground recorded LC3 stream, not live G2 or long/background acceptance.
The nominal startup is 200 ms; it is not a wall-clock microphone-to-ASR
measurement. The native hook still does not flush its delayed stop tail.

The minute's 960,000-sample offline replay agrees with the independent original
Sequence/FIR/NumPy reference within one PCM16 step (RMSE 6.18e-7). After 3,200
startup-zero samples, all 956,800 delivered samples exactly match the aligned
fresh replay. The undelivered final tail is outside that prefix check.

The current public Java API also runs the combined graph on the physical ARM64
Xperia using ORT 1.23.2 in the source-built debug test application. All 16,000
samples, 20 chunk calls, reset/flush/count and lifecycle checks pass, with maximum
PCM error 1.03e-6 and RMSE 8.50e-8 against the retained independent serial
fixture. This correctness run is separate from paced performance acceptance.
The native codec probe contains the exact ARM library bytes from the release
AAR and uses the original Even ORT 1.27.0. Both test applications were stopped.

Current evidence and artifact identities: [compute fusion](compute-fusion.md)
and `validation/compute-fusion-summary.json` in the SDK. The earlier failures
below remain historical evidence and are not replaced by this single pass.

## Previous acceptance stages

The Android library now includes a five-hop throughput graph and public chunked
API. Controlled host measurements show 27.9–37.6% less computing time per audio
hop, with up to 80 ms additional collection delay. The further padding/Softplus
pack reduces host computing time by 11.45% in a separate comparison against
the previous Sequence graph, with no additional delay. Host neural/PCM equivalence
passes, and the chunked native ARM64 path passes direct PCM parity under emulator
translation. The optimized physical Xperia four-thread probe now passes a
20-second paced stream. Earlier one-minute Step/Sequence tests fail with output
gaps and queue resets. The latest padding/Softplus pipeline completes all 600
calls with no reset but still fails with 32 output gaps. Later runs instrument
packet timing and remove timed recording-file
output. Average RTF alone does not establish continuity; those older native
probe builds were not accepted for live use.

The earlier ARM opcode/temporal-fusion pass is documented in
[opcode optimization](opcode-optimization.md). Its reversed-order native-shell
comparison measures 18.55% lower model time, with all measured spectra and final
states passing unchanged tolerances. The AAR-identical ARM library passes full
codec/DSP/model offline PCM parity within one PCM16 step. Its paced 20-second
run fails with 42 gaps; its later minute completes 600 calls/3,000 hops with
zero resets but still fails with two 10 ms gaps. These separate sessions do
not establish a controlled percentage reduction in gaps or sustained readiness.

Earlier one-hop builds ran on a physical Xperia 1 IV (Sony SO-51C, Android
14/API 34). Their offline correctness passed but longer paced runs failed.
Those failures describe the previous graph, not a test of the new chunked
version. Live G2 interception in the modified Even application has not been
accepted. Those interposer trials installed `io.github.reuse.evenprobe` on the
physical phone. The later supported live test installs `io.github.reuse.live`
alongside the official Even app; its installation and data remain preserved.

## Correctness

- **273 Python tests passed** in 36.58 s, covering independent sequential/selective-scan implementations,
  strict checkpoint loading, stream state/reset, original-topology parity,
  native SIMD recurrence/shape checks, streaming audio identity, compact-history
  rewrites, chunked graph/fixture handling, native temporal sequences,
  FastSoftplus, runtime profile/launch handling, frame rings, C++ DSP/FIR parity,
  strict temporal/frequency convolution and channel-normalization fusion,
  selective rewrite proofs, scan-grain and register-scan equivalence.
- **29 JVM tests passed**, covering independent FFT/DFT,
  random/impulse stream identity, boundaries/flush/reset, manifest/config
  validation, chunk collection/output length and canonical phase at negative
  DC/Nyquist endpoints.
- Actual ARM standalone opcode checks pass random and dense exponential/Softplus
  cases, model-shaped scans/sequences and 2,048 recurrent updates. The integer
  FFT indexing change passes 1,218,480 bitwise comparisons on both Windows and
  the physical phone, including speech, random inputs and resets at both rates.
  ARM FFT division instruction sites decrease from seven to three; the small
  frontend microbenchmark gain is separate from whole-model performance.
- Android release AAR, ARM64/x86_64 native libraries and benchmark APKs build;
  release lint and local Maven publication pass.
  The native ELF LOAD alignment is 16 KiB.
- Independent original NVIDIA topology plus sequential CPU Mamba agrees with
  the portable network over 40 recurrent frames at both rates. Maximum complex
  spectrum error is approximately 5.9e-6 at 8 kHz and 3.1e-5 at 16 kHz.
- The fused native-step graphs also agree with PyTorch over 40 recurrent frames:
  maximum spectrum error is 1.88e-6 / 2.62e-6 and maximum state error is
  4.65e-5 / 5.44e-5 at 8/16 kHz. All input/output metadata is static float32
  with shapes checked against the manifest and actual tensors.
- Over 2,048 recurrent updates with near-unit decay, the native temporal update
  has maximum state error 3.40e-6 and observation error 1.22e-6 against the
  independent recurrence.
- The complete Java Android PCM path on 2 seconds of speech matches the
  independent NumPy/ORT frontend: maximum absolute error 1.12e-6 at 8 kHz and
  7.12e-7 at 16 kHz. Counts and reset/flush/failure/lifecycle checks pass.
- On the physical Xperia, the original LC3 codec -> FIR downsampling -> RE-USE
  -> FIR upsampling -> PCM16 offline path matches the independent host reference
  within one PCM16 LSB: maximum 3.0518e-5 normalized, RMSE 7.12e-7, SNR 89.77 dB
  over 320,000 samples with the fused-step 8 kHz model. The earlier scan-only
  32,000-sample comparison also passes (RMSE 7.03e-7, SNR 90.94 dB).
- The compact one-hop graph is bit-identical to its fused-step source for all
  23 outputs over 40 random and 40 speech frames with separate recurrent state.
- The five-hop export agrees with chronological one-hop ORT execution over
  40 audio hops: maximum spectrum error 2.54e-6 and state error 5.61e-5. All
  static feature/state metadata checks pass; existing tolerances are retained.
- Five-hop host speech output matches one-hop processing over 32,000 samples
  within one PCM16 step, normalized RMSE 4.18e-7.
- The actual public Java `ChunkedReuseEnhancer` runs the earlier five-hop Step graph, DSP and
  lifecycle checks on Android API 35/x86_64 emulator using ORT 1.23.2. Its
  16,000-sample/100-hop fixture matches the independent serial reference with
  maximum error 1.0729e-6 and RMSE 8.1314e-8, below unchanged 2e-5/5e-6 limits.
  Reset output is exact; flush/counts, invalid input, failure and closed-resource
  checks pass. The instrumented application and test APK hashes match installed
  files. Emulator timing is not physical-phone performance evidence.
- SelectiveSequence replaces 15 temporal Step nodes with three sequence nodes.
  Native T=1/T=5 tests match chronological Step calls bit-for-bit on Windows,
  and the new graph is bit-identical to the previous chunk graph for all 23
  outputs/states on recorded speech. Actual ARM64 phone cold replay agrees with
  the independent one-hop reference within one PCM16 step (RMSE 4.83e-7). The
  Java Android runtime receipt remains for the earlier Step graph; updated
  manifest/kernel unit tests and host/native-phone checks cover the Sequence
  variant. No new Java-on-Android Sequence execution is claimed.
- The optional Sequence pad-only graph folds 15 zero pads and is bit-identical
  for all 23 outputs over 40 random and 40 speech hops. The combined
  `5c979c32...` graph additionally replaces nine Softplus nodes; maximum spectrum
  difference is 3.34e-6 and state difference 3.17e-5 against original Sequence,
  within unchanged rtol 2e-3 / atol 3e-4. Separate recurrent feedback is checked.
  Its 32,000-sample host speech PCM comparison passes within one PCM16 step,
  normalized RMSE 4.51e-7. Numerical checks include normal and subnormal inputs,
  nonfinite values, shapes, alias rejection and 2,048 recurrent updates.
- The polyphase FIR change is bit-exact against the legacy 95-tap algorithm,
  including reset and changing conversion modes. The isolated host down/up
  benchmark is 37.29 -> 21.15 us per 20 ms transport hop (43.28%); this does not
  measure a 43.28% improvement in the complete pipeline.
- The combined graph and updated native FIR run on the physical ARM64 Xperia
  with the original Even ORT 1.27.0/codec. Cold two-second offline replay against
  the previous Sequence host reference passes within one PCM16 step over
  32,000 samples, normalized RMSE 6.15e-7. That run's paced portion fails with
  17 gaps and a 114 ms packet lateness outlier; offline parity does not turn it
  into a continuity pass. The Java-on-Android receipt still covers the older
  Step graph, not this new FastSoftplus graph.

The original-topology validator uses hash-verified NVIDIA topology code with an
independently implemented sequential CPU Mamba. It does not execute NVIDIA CUDA
kernels. Near-zero phase angles are ill-conditioned, so acceptance compares
Cartesian heads and weighted complex spectra; phase-angle error is diagnostic.
No parity tolerance was relaxed for the physical-phone result. Native FFT/FIR
use double internal accumulation and canonical DC/Nyquist phase endpoints.

## Five-hop host throughput

The additional four-thread padding/Softplus ABBA run uses 800 measured audio
hops per graph and the same new DLL for both. Mean computing time drops from
4.80387 to 4.25365 ms per 20 ms of audio (24.02 -> 21.27 ms per 100 ms call),
an 11.45% reduction. Pad folding alone measures 3.80%; FastSoftplus alone
10.08%, in separate runs. The five-hop Gather candidate is 4.42% slower and
is not recommended. These host percentages are not added together or reported
as phone improvements.

The matched comparison uses the same frozen custom-operation DLL, equal audio
duration and 40 audio hops of warmup. Baseline/chunked/chunked/baseline order
represents 400 measured hops per graph at each thread count: 400 one-hop calls
or 80 five-hop calls. Times below are normalized per 20 ms of audio.

| Host CPU threads | One-hop mean | Five-hop amortized mean | Five-hop call mean | Reduction |
|---:|---:|---:|---:|---:|
| 2 | 10.7623 ms | 7.75955 ms | 38.7977 ms | 27.9% |
| 4 | 9.17842 ms | 5.72644 ms | 28.6322 ms | 37.6% |

These are Windows ORT 1.23.2 results, not Xperia measurements. Chunk calls
collect 100 ms first and add up to 80 ms collection delay. Model weights,
trained depth and temporal causality are unchanged. The compact one-hop
alternative showed a smaller, load-sensitive host improvement (about 4–6%).
See [optimization.md](optimization.md) for method, graph changes and receipts.

## Historical one-hop physical Xperia timing

Each hop represents 20 ms of model audio. RTF below 1 means the average worker
keeps up, but acceptance also requires zero output underflows and queue resets.
The probe delivers five 10 ms LC3 frames per 50 ms burst. Startup silence and
buffering delay are reported separately; a short passing run is not evidence
of sustained operation.

| Run | Duration | Model / threads | Mean hop | RTF | Underflows | Queue resets | Result |
|---|---:|---|---:|---:|---:|---:|---|
| Scan-only, warm | 2 s | 8 kHz / 4 | 19.92 ms | 0.996124 | 0 | 0 | Short run only |
| Scan-only, warm | 20 s | 8 kHz / 4 | 22.55 ms | 1.12769 | 200 | 4 | Fail |
| Fused step, warm | 2 s | 8 kHz / 4 | 17.40 ms | 0.869948 | 1 | 0 | Fail |
| Fused step, prefill/warmup | 20 s | 8 kHz / 4 | 23.47 ms | 1.17348 | 306 | 8 | Fail |
| Fused step, audio priority, precomputed codec inputs | 20 s | 8 kHz / 4 | 22.15 ms | 1.10741 | 221 | 5 | Fail |
| Fused step, audio priority, precomputed codec inputs | 20 s | 8 kHz / 3 | 26.20 ms | 1.30987 | 320 | 13 | Fail |
| Optional performance-hint attempt, cold | 2 s | 8 kHz / 4 | 83.06 ms | 4.1528 | 61 | 2 | Fail |

The representative historical sustained result is the four-thread audio-priority run:
22.1482 ms average per 20 ms hop, 221 output gaps and five queue resets over
20 seconds. It executes 875 inference hops and misses 310 hop deadlines,
with no codec/model exception or passthrough fallback. Resets discard queued
audio, so its RTF describes processed hops and does not imply the entire
input was enhanced. The worker and all three ORT helpers report effective
nice -16 with no priority errors. The three-thread run is slower, as were
six/eight-thread short tests. These results establish physical model execution
but fail sustained real-time acceptance for that one-hop build. The compact
alternative remains unmeasured on the physical phone; newer five-hop results
are separate below. A Dozing run is retained separately
and is not compared as a foreground result.

The audio-priority probe prepares encoding and reference decoding before the
paced phase, as encoding occurs on the glasses. The later performance-hint
revision dynamically resolves Android API 33+ functions and requests a 20 ms
target for only its own worker/helper thread IDs. On this Xperia the symbols
are available, but session creation returns null and no hints are reported.
The 2-second cold result therefore establishes fallback behavior, not a
performance-hint benefit. No affinity, CPU-online state or global setting was
changed. After testing, the probe was force-stopped to release its screen-on
window. The acceptance conditions remain RTF < 1, no underflows, no queue resets,
and no inference/fallback failures.

## Optimized physical Xperia pacing

The five-hop graph runs at 8 kHz/four threads in the separate ARM64 probe on the
physical Sony SO-51C / Android 14. It loads the original Even ORT 1.27.0 through
C API 23 and the LC3 codec, and processes recorded input in five-frame/50 ms
transport bursts. Its model inference calls each represent 100 ms of audio.

| Input / run | Inference calls / audio hops | Mean call time | RTF | Output gaps after startup | Queue resets | Call deadline misses | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| 2 s, cold | 20 / 100 | 86.52 ms | 0.86521 | 0 | 0 | 4 | Short run only |
| 20 s, warm | 200 / 1,000 | 63.99 ms | 0.639895 | 0 | 0 | 0 | Bounded paced pass |
| First 60 s, warm | 595 / 2,975 | 69.11 ms | 0.691146 | 37 | 1 | 2 | Fail |
| 60 s, paced-input revision | 590 / 2,950 | 87.85 ms | 0.878537 | 96 | 2 | 79 | Fail |
| Sequence, 2 s cold / 4 threads | 20 / 100 | 87.23 ms | 0.872259 | 0 | 0 | 4 | Short run only |
| Sequence, 60 s warm / 4 threads | 580 / 2,900 | 85.49 ms | 0.854936 | 148 | 4 | 113 | Fail |
| Sequence, 60 s reconfigured / 2 threads | 520 / 2,600 | 114.37 ms | 1.1437 | 514 | 16 | 520 | Fail |

The 20-second run processes every expected input hop with no codec/model
failure, unsupported input, idle/explicit reset or fallback. It delivers 1,975
enhanced frames in the fixed 2,000-frame paced output; the other 25 frames are
startup silence (250 ms nominal buffering). The short cold run has 350 ms of
nominal startup silence. These startup counts are per run; other diagnostic
counters can be cumulative within the reused process.

The first 60-second run misses the expected 600 calls/3,000 hops because a queue
reset discards work. Its average compute RTF is below one but the complete
pipeline gate fails. The cause of the gaps is not established by its receipt.
It must not be reported as a sustained pass from the average model time alone.

The later timed-input revision preallocates recording output and writes it
after pacing. Its paced phase takes 60,000,120 microseconds; maximum burst
lateness is 7,063 microseconds, no burst exceeds 10 ms lateness, and maximum
decoder callback time is 23,079 microseconds. It still resets twice and processes
only 590/600 calls (2,950/3,000 hops). This rules out a large packet-feeding stall
in that run; it does not establish file I/O as the cause of the earlier failure.
The model/worker and output prefill are unchanged between these two probe
revisions; their APK/native identities are retained with the execution receipts.

The further Sequence variant has the same five-hop contract and trained weights.
Its two-second cold check processes all 20 calls/100 hops without post-startup
gaps/resets and inserts 300 ms of nominal startup silence. Its one-minute
four-thread run still fails: 580/600 calls, 2,900/3,000 hops, 148 gaps/four resets.
Maximum burst lateness is 12.856 ms, with one burst over 10 ms late; paced wall
time is 60,000,113 microseconds.

Changing to two threads creates a new model session within the existing warm
process; it does not reuse the four-thread session. That run is slower and also
fails: 520/600 calls, 2,600/3,000 hops, 514 gaps/16 resets, RTF 1.1437. Its paced
wall time is 60,000,118 microseconds, maximum burst lateness 2.187 ms, no burst
over 10 ms late, and maximum decoder callback time 714 microseconds. All three
later one-minute runs exclude recording-file output from the timed loop. None
has codec/model, unsupported-input or passthrough failures, but none passes
full work/continuity acceptance. The probe was force-stopped after the final run.

`tools/audit_even_probe.py` records the complete count, continuity, failure,
artifact-identity and phone-state checks in `continuous-acceptance.json`.
It correctly rejects the one-minute cases even when their average RTF is below
one. For the three short passing cases, `pipeline-continuity-parity.json`
separately verifies the counted startup zeros and every delivered enhanced
sample against an aligned fresh native replay: zero PCM16-step error. This
checks the delivered prefix and does not cover the pending unflushed tail.

The source-frame age counters are cumulative nominal audio positions, not
wall-clock microphone-to-ASR latency. Fixed output length includes startup
zeros, and `tail_flushed=false` means delayed enhanced output remains pending
at the end. The full waveform is checked separately by offline replay. These
recorded-audio results do not establish live G2 behavior or complete stock
session-boundary handling. The official Even app/data remain unchanged.

The earlier one-hop phone run occurred under different thermal/session
conditions. No controlled phone percentage speedup is inferred from comparing
it with this result; the matched percentage improvement is the host ABBA test.

## Padding/Softplus physical profile trials

The new probe is `dist/even-g2-reuse-probe-otherareas.apk`, SHA256
`e3be9caec5e3784c117ca4e07616aae860bc6ffaff3bbbc37e6fbec75c83e5e2`.
It contains graph `5c979c32...` and the updated FIR/ring pipeline. The four
20-second trials below run in the same physical-phone process, with requested
and effective profile identities checked. Changing a profile configures a new
model session. Each ABBA trial processes all 200 calls/1,000 audio hops with no
queue reset. The first bounded-spin trial has two post-startup output gaps.

| ABBA trial / native runtime profile | Mean call | RTF | Gaps | Queue resets |
|---|---:|---:|---:|---:|
| A1 / `default` | 58.38 ms | 0.583844 | 0 | 0 |
| B1 / `bounded_spin` | 62.95 ms | 0.629498 | 2 | 0 |
| B2 / `bounded_spin` | 82.27 ms | 0.822678 | 0 | 0 |
| A2 / `default` | 85.75 ms | 0.857513 | 0 | 0 |
| Follow-up 20 s / `dynamic4` | 59.70 ms | 0.597002 | 0 | 0 |
| Follow-up 60 s / `dynamic4` | 83.65 ms | 0.836540 | 32 | 0 |

The default results drift from 58.38 to 85.75 ms, so these trials do not show a
clear bounded-spinning benefit. The default session is retained. No NNAPI
provider is activated: diagnostics list CPU/NNAPI availability, while profile
selection changes CPU session settings only. The additional 20-second
`dynamic4` trial also processes every expected call without gaps/resets. Its
one-minute follow-up completes 600/600 calls and 3,000/3,000 audio hops without
a queue reset, but fails continuity: 32 output gaps and 264 call deadline
misses. Paced duration is 60,000,103 us, maximum burst lateness 24,898 us and
one burst is more than 10 ms late. There are no codec/model failures or
fallbacks. Completing the work and average RTF below one do not override the
gaps. These recorded-audio trials do not establish live G2 acceptance.
The first bounded-spin trial also fails continuity because of its two gaps;
the other three ABBA trials and the `dynamic4` 20-second trial pass their
complete-work/continuity audit. Each of those four passing runs has a separate
delivered-prefix comparison within one PCM16 step against aligned fresh native
replay after counted startup zeros. Delayed tail delivery is still unflushed.
The failed minute has 250 ms of initial silence, but later underflows mean its
nominal source delay is not fixed. `no_spin` remains untested on the phone.
Detailed runs are `.research/xperia-otherareas-{default-20s-a1,bounded_spin-20s-b1,bounded_spin-20s-b2,default-20s-a2}`.
The follow-ups are `.research/xperia-otherareas-dynamic4-{20s,60s}`.
After the final run, the separate probe was force-stopped and its process was
confirmed absent. The physical Even application and its data remain intact.

## Host and emulator timing

These earlier results are not physical-phone evidence. Target deadline: 20 ms.

| Environment | Model | Threads | Median hop | p95 hop | RTF | Misses |
|---|---|---:|---:|---:|---:|---:|
| Windows Ryzen 7 5700X, native scan | 8 kHz | 2 | 14.98 ms | 19.13 ms | 0.80 | 0/101 |
| Windows Ryzen 7 5700X, native scan | 16 kHz | 4 | 27.73 ms | 32.80 ms | 1.42 | 101/101 |
| Android 15 x86_64 emulator, native scan | 8 kHz | 4 | 20.15 ms | 26.25 ms | 1.06 | 55/100 |
| Android 15 x86_64 emulator, native scan | 16 kHz | 4 | 35.20 ms | 40.09 ms | 1.77 | 100/100 |

The earlier normal ARM64 harness in emulator translation loads the original Even
ONNX Runtime 1.27.0 through C API 23 and the original LC3 codec. Its offline
parity passes within one PCM16 step. Its paced stream fails (RTF 17.39, three
queue resets); native ARM translation is not a phone performance proxy.

The new chunked ARM64 native harness also loads the original Even ORT 1.27.0 /
C API 23 and LC3 codec. Its direct replay of 32,000 PCM samples passes within
one PCM16 step: RMSE 3.8147e-7 normalized, SNR 96.26 dB. The actual probed graph
is the five-hop `d0549cfc...`; the independent parity receipt names the one-hop
`bc40bc38...` graph as its reference. This is a comparison between those two
graphs, not evidence that the phone ran the reference graph.

The same emulator's paced chunk test fails: RTF 1.40224, 22 underflows and one
queue reset, with no codec/model errors or passthrough. It cannot establish
optimized Xperia speed or be directly compared with older emulator builds.

The separate actual Java chunk test executes 20 graph calls for 100 input hops
and returns all 16,000 samples, with passing neural/DSP parity and lifecycle.
Its measured mean is 45.23 ms per 100 ms call, but it is an x86_64 emulator
correctness run and cannot predict the physical Xperia's performance. The
serial/chunk host fixtures also agree (maximum PCM error 8.94e-8).

## Even application and remaining acceptance

The modified Even APK cannot be launch-validated in the emulator because the
untouched official app also fails in its packed loader before our initializer:
the loader selects unavailable x86/x64 protection assets. No loader checks,
authentication or integrity behavior were bypassed. The original emulator app
was restored. See [even-g2.md](even-g2.md).

Remaining work:

- Sustained physical-phone deadlines for the optimized graphs, measured startup
  and collection/output buffering delay, thermal and battery use.
- Full modified Even app launch and physically connected G2 interception through
  that native hook. The separate official-Hub companion's live microphone run
  and explicit finite-tail drain now pass the bounded test in [live-g2.md](live-g2.md).
- Source-timestamped G2 BLE continuity, broader microphone start/stop coverage
  and stock-app delayed tail handling. The interposer
  has no reliable stock session-end callback and does not flush a stopped tail.
- Stock ASR/cloud behavior and whether combined enhancement improves actual
  speech intelligibility/recognition, including unusual voices and overlap.
- Long sessions and background operation.

Private detailed receipts remain in `.research`; the real before/after PCM and
its private receipts are in `recordings/even-g2-live-20261101`. SDK validation
receipts contain model/artifact hashes and sanitized physical timing metadata;
no device serial or host-specific paths are needed to interpret the result.
