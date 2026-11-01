# Earlier CPU opcode and layout optimization

This record describes the opcode/temporal build. The subsequent
[normalization/frequency fusion](compute-fusion.md) has a bounded one-minute
foreground recorded-audio pass; the earlier failed results below are retained.

This pass targets the existing float32 CPU path: ARM NEON instruction use,
frequency-state register traffic, temporal convolution layouts, and FFT index
arithmetic. It preserves trained weights, exit depth, sample rate and the
five-hop/100 ms call contract. It does not reduce buffering or activate a GPU,
NNAPI or NPU backend.

The measurements below distinguish native-shell model diagnostics, isolated
kernel/frontend diagnostics and the complete paced codec/model pipeline.
Shell model timing alone does not establish continuous output, live G2 use or
microphone-to-ASR latency. The opcode build's minute-long pipeline test completes all 600
calls with two 10 ms output gaps and no queue reset. It fails strict continuity.
The earlier pipeline's minute-long result with 32 gaps is retained separately.

## Changes

| Change | Mechanism | Numerical contract |
|---|---|---|
| Explicit NEON FMA | Fuse multiply/add in recurrence, exponential range reduction/polynomial and FastSoftplus polynomial | FP32 with changed rounding; unchanged model/kernel acceptance tolerances |
| Rank-16 frequency state in registers | Keep four NEON state vectors across the frequency scan instead of repeatedly reading/writing scratch | Same frequency chronology; other state ranks retain the generic implementation |
| `TemporalConvSilu` | Fuse history slicing/concat, four-tap depthwise Conv, SiLU and transposes; vectorize contiguous channels | Exact convolution-history movement; bounded-error FP32 activation/convolution |
| FFT index simplification | Cache recursion stride and remove radix-2 twiddle-index division/remainder | Same double floating operations/order; bit-exact regression |

The native CMake options `REUSE_NEON_FMA` and `REUSE_SCAN_N16_REGISTERS` remain
OFF unless explicitly selected for a build. The benchmark's combined ARM64
library enables both. The AAR for this pass was explicitly built with Gradle
properties `-PreuseNeonFma=true -PreuseScanN16Registers=true`; the properties
default to false for a new source build. FMA is an ARM-only experiment; Windows SSE2 timing does
not measure the ARM FMLA path. The existing code already uses NEON on arm64;
this pass changes its work and instructions. Arm documents the four-lane FP32
FMLA intrinsic in its [Advanced SIMD reference](https://arm-software.github.io/acle/neon_intrinsics/advsimd.html),
and Android enables NEON by default for arm64 in the
[NDK NEON guide](https://developer.android.com/ndk/guides/cpu-arm-neon).

The temporal pack is
`models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal`, graph SHA256
`ab02d6cca7c923751651032b6151c022a8e4b8f560790e8beba3367475b9653a`.
Its manifest additionally requests `com.reuse::TemporalConvSilu` v1 from
`reuse_ort_ops`; older libraries/loaders cannot load this pack. It accepts the
same public `ChunkedReuseEnhancer` PCM/state contract.

The rewrite proves exactly three fixed `[81,5,256]` temporal regions, their
weights, unit strides, four taps, slice ranges, consumers and public output
metadata. The input cache is `[81,256,4]`; the first Conv output uses its last
three values and the first new value. After five updates the final cache
contains the last four new values. All 23 public outputs and state shapes are
preserved. This does not repeat the previously rejected dense-history Gather
experiment.

## Physical-phone model diagnostics

The standalone native runner loads the original Even ORT 1.27 CPU runtime on
the Sony SO-51C / SM8450, four threads, with `dynamic4`. Each block has eight
warmup calls and 40 measured calls; baseline and candidate each represent 80
measured 100 ms calls. The runner advances recurrent state and compares all
measured spectra plus final states with unchanged rtol 0.002 / atol 0.0003.
The fixture is a fixed five-frame feature chunk, not a paced microphone stream.
Calls run consecutively, excluding LC3, resampling, STFT/frontend and transport
queues. This does not reproduce 50 ms packet bursts or idle/active scheduling.
Wall time and summed process CPU time are both retained. A four-thread process
can consume more CPU milliseconds than elapsed wall milliseconds.

| Separate ABBA experiment | Baseline mean call | Candidate mean call | Descriptive reduction |
|---|---:|---:|---:|
| FMA only, same graph | 47.991 ms | 46.749 ms | 2.59% |
| Rank-16 register scan only, same graph | 48.843 ms | 47.375 ms | 3.00% |
| FMA + register scan, same graph | 49.893 ms | 47.132 ms | 5.53% |
| Temporal fusion only | 71.767 ms | 66.238 ms | 7.70% |
| All three, first comparison | 81.171 ms | 56.618 ms | 30.25%; drift-confounded |

These percentages come from separate experiments and must not be added.
The first combined-graph baseline block averages 92.191 ms and the final one
70.151 ms. That large drift prevents a reliable 30.25% speedup claim; the full
record is retained. Temporal-only baseline blocks are much closer at
71.842/71.691 ms, with candidate blocks 66.197/66.280 ms.

A reverse-order repeat uses candidate/baseline/baseline/candidate. Its means
are 49.190 ms baseline and 40.065 ms candidate, an 18.55% model-compute
reduction. Candidate blocks are 38.181/41.949 ms; baseline blocks
48.562/49.818 ms. All endpoints are awake/unlocked, and reported battery
temperature is 38–39 C. This repeat is still a model diagnostic; temperature
and order metadata do not make it a paced pipeline acceptance result.

All six comparisons pass their spectrum/final-state checks. FMA changes
rounding; rank-16 scan alone is bit-identical in its retained phone comparison.
The all-changes comparison has maximum spectrum error 3.60e-6 and final-state
error 9.60e-6. Frozen graph, native library, runner, fixture and runtime hashes
are retained in each receipt.

Receipts: `.research/opcode-{fma,scan16,combined,temporal,all}-abba/report.json`
and `.research/opcode-all-baab/report.json`. Sanitized SDK records retain the
full timing samples, artifact hashes, parity and available per-block phone
state/temperature metadata. Historical comparison records are not replaced.

## Isolated phone correctness and frontend diagnostics

The combined FMA/register kernel runner passes independent scalar-reference
checks for ordinary and rank-16 scans, temporal sequence and long recurrence,
exponential and FastSoftplus. Maximum scan error is 5.96e-8, sequence error
8.94e-8, long-sequence error 1.49e-8 and FastSoftplus error 4.77e-7. The receipt
reports both experimental flags enabled. Its short single-thread kernel times
are diagnostic and do not establish model or pipeline speedup.
Receipt: `.research/opcode-cpu/kernel-combined.json`.

The FFT/frontend phone benchmark retains double arithmetic and performs
1,218,480 bitwise comparisons, including speech/random PCM and FFT cases at
both rates. The same ABBA workload compares 2,000 iterations per block:

| Isolated stage / rate | Baseline per 20 ms hop | Optimized per 20 ms hop | Computing-time reduction |
|---|---:|---:|---:|
| FFT pair / 8 kHz | 29.492 us | 26.106 us | 11.48% |
| Complete frontend / 8 kHz | 39.416 us | 36.323 us | 7.85% |
| FFT pair / 16 kHz | 62.821 us | 55.519 us | 11.62% |
| Complete frontend / 16 kHz | 83.817 us | 76.070 us | 9.24% |

The frontend includes analysis/synthesis but excludes neural inference, codec,
resampling and worker/transport pacing. These microsecond savings cannot be
reported as an 8–12% reduction for the full pipeline. The record explicitly
sets `diagnostic_only=true` and `acceptance_eligible=false`.
Receipt: `.research/opcode-cpu/fft-phone.json`.

The new temporal kernel separately passes 19 focused host tests, including
2,048 exact cache updates/reset, independent double convolution/SiLU reference,
vector tails, serial/parallel equivalence, aliases and graph-proof rejection.
Forty random plus 40 speech hops pass all 23 outputs with separate feedback
and unchanged tolerances. Maximum spectrum error is 4.41e-6, state error 2.34e-5.
That receipt is in the temporal model pack as `optimization-parity.json`.

## Earlier integrated opcode probe

The frozen `dist/even-g2-reuse-probe-opcode.apk` contains the temporal graph,
combined FMA/register-scan library and updated FFT shim. SHA256:
`dfba8558fbd96ed7d72a8e3ecbf470fa57a627cfb2c4b4c8ffec301a52d90eee`.
Its custom-operation ARM64 library is byte-identical to its corresponding AAR's:
`9a2e87cdd4e5abdbd4b6fa0d70261eb8d98278268dec473a07f2d250b5c5e770`.
The AAR is 395,784 bytes, SHA256
`81b0363268b1556338ce71c4cbdbbf8223780041144011ae0b7e5d5272a3a5b4`.

The cold two-second check completes 20 calls without gaps or queue resets,
RTF 0.819882. Its 32,000-sample offline PCM comparison against the earlier
Sequence reference passes within one PCM16 step, normalized RMSE 5.3948e-7.
Its delivered 28,000-sample prefix also matches fresh replay exactly after
4,000 startup zeros (250 ms nominal buffering); this does not cover the tail.
The warm 20-second trial fails continuity despite RTF 0.587715: all 200 calls /
1,000 hops complete, with 42 output gaps, zero queue resets and 11 call deadline
misses. This failure is retained; faster mean model time does not override it.

| Current APK / recorded input | Calls / audio hops | Mean 100 ms call | RTF | Output gaps (10 ms frames) | Queue resets | 100 ms call misses | Strict continuity |
|---|---:|---:|---:|---:|---:|---:|---|
| 2 s cold | 20 / 100 | 81.99 ms | 0.819882 | 0 | 0 | 4 | Short check passes |
| 20 s warm | 200 / 1,000 | 58.77 ms | 0.587715 | 42 | 0 | 11 | Fail |
| 60 s warm | 600 / 3,000 | 74.34 ms | 0.743421 | 2 | 0 | 14 | Fail |

The minute completes every expected model call/hop and has no codec/model
failure or fallback. Pacing takes 60,000,105 us, maximum burst lateness is
7,666 us, no burst is more than 10 ms late, and maximum decoder callback is
10,585 us. Those observations do not establish the cause of its two output
gaps. No controlled pipeline percentage improvement is inferred by comparing
these runs with earlier builds under different conditions. The model timing
comparison above is the controlled comparison; the paced result is a separate
continuity gate.

Receipts are `.research/xperia-opcode-{cold-2s,warm-20s,warm-60s}`. All retain
their strict `continuous-acceptance.json`; the two longer runs are failures.
The source age/startup counters are nominal audio positions, not measured
microphone-to-ASR wall latency. The hook still does not deliver its stopped
tail automatically. After the final run, the probe was force-stopped and its
process confirmed absent.

A read-only CPU policy snapshot around the final run observes CPUs 6/7 offline, CPUs 0–5 online,
and a top-app cpuset configured for 0–7; Battery Saver is off. The snapshot and
frequency limits are retained in the summary. No OS CPU policy or affinity was
changed. The snapshot is not phase-aligned to paced processing or offline replay,
so it does not prove core availability caused the gaps or that changing it
would fix them.

The source/build validation is 185 Python tests in one full run (27.29 s),
28 JVM tests, release lint/build and local Maven publication. The earlier
Java-on-Android neural/DSP receipt continues to cover the older Step graph;
these new graph results are host/native-phone checks, not a new Java Android
runtime receipt. `.research/xperia-opcode-setup-2s` is a retained setup check
with an earlier binary and must not be labeled as the final APK's result.

## Reproduction and remaining boundary

Build the native kernel experiments in separate output directories with
`-DREUSE_NEON_FMA=ON -DREUSE_SCAN_N16_REGISTERS=ON`; use the normal pinned
Android NDK 26.1/CMake toolchain and arm64 ABI. Add
`-DREUSE_BUILD_KERNEL_BENCHMARK=ON` for the isolated numerical runner.
Do not overwrite a frozen baseline library or comparison output directory.
The same source retains a generic fallback for ranks other than 16.

To reproduce the opcode build's selected flags, append
`-PreuseNeonFma=true -PreuseScanN16Registers=true` to the Gradle test, release
build and Maven publication commands. Omitting them uses the source defaults.

`tools/optimize_temporal_conv.py` creates a separate pack from the pad/Softplus
source and validates it with an explicitly supplied current custom-op library.
`tools/benchmark_android_cpu.py` freezes its inputs, performs ABBA model
diagnostics and checks dumped tensors; `--reverse-order` selects BAAB.
`tools/android_cpu_benchmark.cpp` is the shell runner, and
`tools/native_kernel_benchmark.cpp` is the independent kernel runner.
The SDK includes their buildable source, not original vendor APKs.

Before recommending a combined build, retain its actual model/native/AAR/APK
identities, independent PCM parity, complete paced call/hop counts, output
gaps/resets, fallback and phone state. Model diagnostics and short foreground
runs cannot establish live G2 lifecycle, stopped-tail delivery, ASR benefit,
thermal endurance, background behavior or battery use. Final paced results
and current build identities are recorded in [validation.md](validation.md)
and the SDK's `validation/opcode-summary.json`.
