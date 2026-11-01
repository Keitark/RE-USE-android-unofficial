# Further compute fusion

The normalization and frequency-convolution fusions together reduce measured
model computing time on the physical Xperia by **5.51%** against the earlier
temporal-convolution graph in a direct ABBA comparison. The matching direct
host comparison measures 7.98%. Separate component screens measure 3.69–6.96%
for normalization on the phone and another 2.02% for frequency fusion against
that normalization graph. These percentages must not be added. Model timing
and pipeline continuity are checked separately: the combined build also passes
one bounded foreground minute of recorded audio with no output gaps or resets.

The graph call still represents five chronological 20 ms hops. Trained
weights, sample rate, exit depth, 21 recurrent states, public tensor shapes
and up to 80 ms chunk-collection delay are unchanged. No audio hop is skipped,
and neither change selects a GPU/NNAPI/NPU backend or quantizes the model.

## Channel normalization, PReLU and layout

The selected normalization pack is
`reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal-channelnorm`, SHA256
`7d3cc6f1c9e7143eae1c24e0071376533fdf7bee2a2e0a91b2c9e00af7fdb2b5`.
Its `com.reuse::ChannelNormPRelu` v1 operation consumes float32 NHWC data with
64 channels, scale/bias `[64]` and PReLU slopes `[64,1,1]`, then emits NCHW.
The graph explicitly requests the matching current native library; older AARs
cannot load this operation.

The rewrite proves 18 dense/up-convolution sequences of LayerNormalization,
NHWC-to-NCHW transpose and PReLU. It checks fixed shapes, the last-channel axis,
float32 constants, epsilon, consumers and unchanged public graph metadata.
The six three-dimensional Mamba normalizations remain ordinary ONNX nodes.
The native kernel uses stable anchored statistics, retains epsilon, and combines
normalization, affine transform, PReLU and a four-position tiled transpose.
It uses NEON/SSE2 channel vectors and contiguous output stores. Floating
rounding changes are checked against the existing model tolerances.

The actual phone profile attributes 7.67% of instrumented kernel time to the
targeted region: 3.99% for 18 normalizations, 2.95% for 18 PReLUs and 0.73%
for 13 remaining post-normalization transposes. ONNX Runtime already removes
or moves five of the exported transposes. This attribution skips eight warmup
calls per node and uses 20 measured calls; it is not an obtainable-speedup
guarantee. Full proof and attribution records are retained.

| Separate comparison | Baseline mean | Normalization mean | Reduction |
|---|---:|---:|---:|
| Xperia model ABBA | 39.045 ms / 100 ms call | 37.604 ms / call | 3.69% |
| Xperia model BAAB repeat | 40.162 ms / call | 37.367 ms / call | 6.96% |
| Windows host ABBA | 3.76647 ms / 20 ms audio | 3.52302 ms / 20 ms audio | 6.46% |

Both phone comparisons use the same frozen native library for both graphs,
four CPU threads, original Even ORT 1.27 and `dynamic4`. Each block has eight
warmup calls and 40 measured calls, giving 80 measured calls per graph.
The shell runner feeds a fixed five-frame feature chunk and checks all
measured spectra plus final states. It excludes the codec, audio frontend,
transport pacing and output queues. The host comparison uses ORT 1.23.2,
equal audio duration and a shared frozen library, with 800 measured hops per
graph. These model diagnostics do not measure microphone or end-to-end latency.

The phone spectra/final states pass unchanged rtol 0.002 / atol 0.0003.
Maximum spectrum difference is 5.36e-6 and final-state difference 1.17e-5.
Host random and speech validation independently compares all 23 outputs over
40 hops each with separate recurrent feedback; maximum spectrum difference
is 8.58e-6 and state difference 5.29e-5. The focused tests cover low variance,
large constant offsets, exact layout/tails, empty dimensions, nonfinite rows,
input immutability, aliases, serial/parallel identity and graph-proof rejection.

Receipts: `.research/next-compute-norm-{abba,baab}/report.json`,
`.research/next-compute-norm-host-abba/report.json`,
`.research/channelnorm-addressable-phone.json` and the model pack's
`optimization-parity.json`. SDK projections retain every timing sample, graph,
runtime and native-library hashes, plus available phone-state metadata.

## Screens that were not promoted

An explicit 13-region variant preserves the five sites whose exported layout
conversion the runtime already removes. All 18 source patterns are still
proved before choosing the 13 fusions. Unknown or duplicate preservation names
are rejected. This variant passes all-output parity but is **4.52% slower than
the 18-region graph** in its phone screen: 37.257 -> 38.942 ms per call. Its
proof/parity and timing receipts are retained; its model is excluded from the
SDK's selected packs.

A frequency-Scan callback grain experiment changes 16 channels per callback to
32 on the earlier temporal graph. It is not a change to the normalization
kernel's 64-position work grain. Its arithmetic is bit-identical, but the
apparent 7.99% mean reduction is confounded by baseline blocks of 38.461 and
54.620 ms. Median time worsens from 40.151 to 40.991 ms. The default Scan grain
stays 16; this experiment is not promoted.

Receipts: `.research/next-compute-norm13-abba/report.json`, the rejected pack's
`normalization-preservation-proof.json` and `optimization-parity.json`, and
`.research/next-compute-grain32-abba/report.json`. Historical failures are kept
alongside the selected result.

## Frequency-convolution fusion

The separate frequency pack is
`reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal-channelnorm-frequency`,
SHA256 `f950dcfb9321191b2bcb3ab9afa3e100c40e5c64dd1de1d30849be4c1aacce32`.
It adds `com.reuse::FrequencyConvSilu` v1 to the normalization graph. Six
frequency-axis four-tap depthwise Conv/SiLU/layout sequences become contiguous
channel-vector kernels. The rewrite proves the three-position left padding,
retained 81 frequency positions, weights, directions and consumers. Each
frequency scan has independent zero history on entry; this fusion does not
change temporal state or introduce future audio dependencies.

Its four-thread Windows ABBA comparison measures 3.60880 -> 3.36252 ms per
20 ms of audio, a 6.82% reduction. Both sides use the same frozen library and
represent 800 measured audio hops per graph. This is a host CPU result, not
ARM/phone evidence. Host parity passes 40 random plus 40 speech hops for all
23 outputs, against both the normalization source and the earlier temporal
graph, with unchanged tolerances. Against the latter, maximum spectrum error
is 1.00e-5 and state error 5.20e-5.

The separate direct combined host ABBA comparison uses the earlier temporal
graph as baseline: 3.75780 -> 3.45789 ms per 20 ms of audio, a 7.98% reduction.
It also uses equal represented audio and one frozen library for both graphs.
This direct comparison is the combined cost measurement; adding the isolated
6.46% normalization and 6.82% frequency measurements would be incorrect.

Its physical Xperia model ABBA screen measures 37.489 -> 36.731 ms per call,
2.02% less than the normalization graph with the same library. All measured
spectra/final states pass unchanged tolerances (maximum spectrum error 7.29e-6,
state error 9.54e-6). This remains model-level computing time.

A direct combined phone BAAB comparison retains a raw mean of
55.187 -> 37.271 ms per call (32.46%), but has substantial block variation:
baseline blocks average 59.295/51.079 ms and candidate blocks 34.356/40.187 ms.
The raw percentage is drift-confounded and is not a reliable combined-speedup
headline. The full record is retained. The direct host combined comparison and
the separate phone component screens have their own stated scopes.

The direct combined phone ABBA repeat measures 39.322 -> 37.154 ms per call,
**5.51% less computing time**, with passing spectra/final-state parity
(maximum spectrum error 1.93e-6, state error 1.34e-5). This is the conservative
combined phone result used here. It is a shell model test, not paced audio.

The manifest's phone-real-time validation flag remains false. The SDK packaging
option `--include-experimental-frequency` includes this separate pack and its
validation receipts. Source and reversible research artifacts remain available.
The physical public-Java correctness test is recorded separately below.

Receipts: `.research/next-compute-frequency-host-abba/report.json` and
`.research/next-compute-combined-host-abba/report.json`, plus
`.research/next-compute-frequency-arm-abba/report.json`,
`.research/next-compute-combined-arm-{abba,baab}/report.json` and
`.research/frequency-conv-validation/{frequency_delta_from_channelnorm,combined_from_ab02}.json`.

## Bounded recorded-audio pipeline result

The frozen `dist/even-g2-reuse-probe-compute-fusion.apk` uses the combined
frequency graph. SHA256:
`67766cb985fa1e8ca6ec56697720e159ea63f6b17dc2fa0bfc62bd48fe23aad3`.
The corresponding AAR is 408,981 bytes, SHA256
`f3afb7d654543fda94bd8ace5d89cdd61c31ee4c9a09f30551a71a3b2d346703`.
Its ARM64 custom library is byte-identical to the probe's, SHA256
`631965dea904b33139762700d9cd20f14df7367b170bfe7175129f63d75502e5`.

| Combined build / recorded input | Calls / hops | Mean per 100 ms call | RTF | Gaps / resets | Call misses | Strict result |
|---|---:|---:|---:|---:|---:|---|
| 2 s cold | 20 / 100 | 74.94 ms | 0.749430 | 0 / 0 | 2 | Short check passes |
| 60 s warm | 600 / 3,000 | 55.06 ms | 0.550647 | 0 / 0 | 0 | Bounded foreground pass |

The minute processes all expected model work with no codec/model failures,
fallbacks, queue resets or post-startup output gaps. Its paced phase is
60,000,132 us, maximum burst lateness 3,686 us, no burst more than 10 ms late,
and maximum decoder callback time 39,312 us. Nominal startup silence is 200 ms
for the warm minute and 400 ms for the cold check; these are audio-position
counts, not measured microphone-to-ASR latency. The cold offline PCM comparison
also passes 32,000 samples against the earlier Sequence reference within one
PCM16 step, normalized RMSE 5.12e-7.

Independent host replay of the complete 960,000-sample minute also passes
against the earlier Sequence reference within one PCM16 step, normalized RMSE
6.18e-7. After its 3,200 startup-zero samples, all 956,800 delivered samples
match fresh native replay exactly (zero PCM16-step error). The cold check's
25,600 delivered samples likewise match exactly after 6,400 startup zeros.
These prefix checks verify delivered ordering and values, excluding the pending
tail; they do not convert the nominal startup count into wall-clock latency.

This is the first retained one-minute strict paced pass. It covers one
foreground recorded-input run through the native LC3/model/DSP chain, not
longer endurance, background operation or live G2 connectivity. The native
Even hook still leaves a delayed tail pending when the stream stops. The
production-ready and live-G2 acceptance states remain false. Records are
`.research/xperia-compute-fusion-{cold-2s,warm-60s}`.

## Public Java API on the phone

The actual public `ChunkedReuseEnhancer` API also runs the combined graph on
the physical ARM64 Xperia with ORT 1.23.2. Its source-built debug benchmark
and test APKs match the installed hashes. All 16,000 samples / 20 model calls
are checked against the fixture: maximum PCM error 1.03e-6 and RMSE 8.50e-8,
inside unchanged 2e-5 / 5e-6 limits. Reset is exact; flush/counts, invalid input,
failure state and closed-resource lifecycle checks pass.

This is a physical-phone neural/DSP/API correctness run, separate from the
Even codec's paced minute. The debug benchmark's custom library hash is
`b269a659ca6c9a28ae149015de2ab26018ca02d0e6cfb3f2aed95c0710345613`;
it uses the current source and flags, but its bytes differ from the release
AAR. The native Even probe above does use the release AAR's identical ARM64
library. The older Step-graph emulator receipt remains labeled separately.
Records: `.research/android-java-compute-fusion/{report,execution}.json`.
Both test applications were stopped and their processes confirmed absent.

## Historical failures and reproduction

The normalization-only graph has model-level phone execution, not a separate
codec/microphone pipeline acceptance run. A retained cold setup used the
earlier opcode APK and failed: only 10 of 20 expected calls / 50 of 100 hops,
RTF 2.45849, two output gaps and two queue resets. It is labeled as an old-binary
setup failure, not as normalization pipeline timing. The preceding opcode APK's
20-second/60-second runs also remain failures with 42/two output gaps. No live
G2, stopped-tail, ASR, thermal endurance, background or battery acceptance is
inferred from the compute comparisons or the bounded recorded-audio pass.

The final source check passes 273 Python tests in one full run (36.58 s),
29 JVM tests, release lint/build and local Maven publication. The focused
normalization suite has 31 cases; its total is part of the full suite, not an
additional full-suite count.

`tools/optimize_channel_norm.py` writes a fresh pack and performs recurrent
validation with the supplied library. Its repeated `--preserve-normalization`
argument is an explicit research allowlist, defaulting to none.
`tools/optimize_frequency_conv.py` produces the separate frequency candidate.
Both require matching new custom-operation support. Source CMake/Gradle defaults
for optional FMA/register flags and the Scan grain stay documented in
[opcode-optimization.md](opcode-optimization.md); neither rejected screen
changes a default. The summary and final artifact identities are in
`validation/compute-fusion-summary.json` in the SDK and [validation.md](validation.md).
