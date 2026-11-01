# Streaming architecture

## Signal contract

The one-hop public API consumes normalized mono float32 PCM at 8 or 16 kHz in
exactly 20 ms blocks (160 or 320 samples). `ChunkedReuseEnhancer` instead consumes
five hops per call: 100 ms, or 800 samples at 8 kHz. The model's FFT/window duration is 40 ms
(320 or 640 samples). Frequency resolution remains 25 Hz at either sample rate.
The FFT is unnormalized, with a periodic Hann window. Magnitude is `log1p(abs(X))`
and phase is radians. Two high-frequency zero bins are appended inside the
neural graph exactly as in NVIDIA's implementation.

The Java frontend uses a reusable mixed-radix 2/5 FFT. Analysis takes the previous
hop followed by the current hop, with zero padding before the first real input.
Synthesis reconstructs the conjugate-symmetric spectrum and performs inverse
FFT, Hann multiplication and overlap-add normalized by summed Hann squares.
DC/Nyquist imaginary components are discarded as in a real inverse FFT.

The output before input time zero is discarded. `process()` initially returns
null. Subsequent calls each return the preceding input hop. `flush()` analyzes
one zero hop to finish the last real hop. This also makes a one-hop clip work
without losing its start/end. The boundary padding is intentionally causal;
it differs from the upstream offline demo's reflected left padding. NVIDIA's
neural topology is validated using identical input features, independently of
this audio boundary convention.

## Portable neural graph

The graph retains the selected encoder branch (zero look-ahead), first N
temporal/frequency Mamba blocks, and only the decoder pair trained for exit N.
It strictly checks active checkpoint keys and rejects unknown keys; unused
look-ahead branches, deeper blocks and other exit decoders are explicitly pruned.
The original multi-exit checkpoint contains more weights than an active
configuration. At exit 3 the active network has 1,823,555 parameters.

Temporal Mamba uses explicit depthwise-convolution and SSM history tensors.
Frequency Mamba is bidirectional within the current frame. Its recurrence is
represented by standard tensor operators instead of CUDA selective-scan kernels.
The frequency scan is a work-efficient recursive affine prefix scan. For
`h[t] = a[t]*h[t-1] + b[t]`, adjacent transitions compose as
`(a1*a0, b1+a1*b0)`. Recursive pairing and interleaving avoids an ONNX loop per
frequency bin and reduces memory traffic compared with recursive doubling.
There is no division by a potentially underflowed cumulative product. The
optional native graph replaces frequency selective scan with
`com.reuse::SelectiveScan`; it fuses the recurrence and observation, uses ORT's
intra-op worker pool, and retains only current recurrent state. The
`--fuse-time-step` export also uses `com.reuse::SelectiveStep` for each temporal
SSM update and observation. Its input history is read-only and its next history
is written to the opposite state bank. Both operations use the same version-1
domain and custom-operation library. ARM64 NEON and x86_64 SSE2 use a
bounded-error exponential approximation. See [native/README.md](../native/README.md)
for the numerical checks.

The `--fuse-time-sequence` five-hop export additionally uses
`com.reuse::SelectiveSequence`. It advances all five temporal SSM updates inside
one native call per layer, retaining the trained 16-element state in SIMD
vectors and storing only the final state. Three Sequence nodes replace fifteen
Step nodes. The existing Scan/Step operations remain available for older packs;
the Java loader accepts only the fixed supported names, not paths from metadata.

The optional `sequence-pad-softplus` graph folds 15 zero-padding nodes into
Conv attributes and replaces nine Softplus nodes with `com.reuse::FastSoftplus`
v1. The unary float operation keeps tensor shapes and the public state contract
unchanged. It uses stable single-exponential softplus with a tested SIMD
log1p polynomial. Pad folding is bit-identical; FastSoftplus is bounded-error
float math checked against unchanged full-model and PCM tolerances. The latest
AAR contains this operation for both ABIs; earlier AARs cannot load the pack.

All graph input/output types and shapes, including temporal next-state tensors,
are exported as static float32 metadata and checked against the manifest and
actual execution. The Android loader rejects symbolic or mismatched state
metadata rather than weakening its allocation checks.

All original residual connections, channel-only layer normalization, PReLU,
frequency subpixel shuffle, decoder history and magnitude artifact suppression
are retained. Phase output is converted directly from the decoder's real and
imaginary heads to a unit vector, avoiding `atan2` in ONNX. Zero-vector phase
maps to zero radians. ONNX uses Exp/Sub to represent PyTorch's `expm1`; its
numerical difference is covered by multi-frame output/state parity checks.

For 16 kHz/exit 3 the graph has 21 recurrent tensors totaling 26,408,496 bytes
per state bank. The Java runtime double-buffers them and pins output tensors,
so it does not copy the entire state through Java arrays on each hop. Runtime
working memory is additional. Use one enhancer per audio stream.

## Reduced computing time

The compact one-hop graph gathers only the three temporal slices referenced by
each dilated convolution. A depth-three dense convolution needs times 0, 8 and
16, although the original representation padded and reordered all 17 frames.
Compacting those slices and folding spatial padding into convolution attributes
preserves the trained weights, public shapes and recurrent state layout. All 23
outputs are bit-identical to the baseline over 40 random and 40 speech frames in
the retained host comparison. Its host throughput improvement is modest and
varies with load; it is not a phone performance result.

The five-hop graph performs encoder/decoder convolutions and frequency Mamba
work across five chronological frames within one graph call. Temporal Mamba
still advances every 20 ms in order, carrying the convolution/SSM history between
those frames. Frequency scans treat time frames as separate batches. The graph
returns only the final 21 history tensors, with the same shapes as the one-hop
graph, and feature/spectrum tensors shaped `[5, frequencyBins]`.

This amortizes dispatch and cache-history work across 100 ms of audio. It does
not remove a trained layer, skip an input hop, quantize weights or introduce
future-time model dependencies. `frames_per_call`, feature shapes and graph hash
are validated by both the native and Java loaders. The one-hop Java loader
rejects a five-hop manifest; `ChunkedReuseEnhancer` requires it.

The chunked frontend analyzes five consecutive hops before inference and then
synthesizes their spectra in order. Its first call returns four PCM hops, later
calls return five, and flush finishes the final pending hop. Zero padding of a
partial final chunk belongs to the caller, who trims the returned stream. The
extra collection wait is at most 80 ms beyond a one-hop call; first output
requires collecting 100 ms plus computation. Buffering is separate from neural
look-ahead and must be included in application latency measurements.

Controlled Windows comparisons show 27.9%/37.6% lower mean computing time per
hop at two/four threads. They use equal audio duration, warmup and an alternating
graph order. A separate four-thread physical Xperia run of the earlier five-hop
Step graph also keeps up for
20 seconds, averaging 63.99 ms per 100 ms call with no post-startup gaps or
resets. Earlier one-minute runs fail with output gaps/queue resets,
including the additional Sequence fusion and two-thread profile.
These are distinct measurements; the earlier one-hop phone run had
different thermal/session conditions, so no matched phone percentage is claimed.
See [optimization.md](optimization.md).

A separate four-thread comparison of the padding/Softplus pack against the
Sequence graph measures 11.45% less host compute (4.80387 -> 4.25365 ms per
20 ms of represented audio). This is a distinct comparison using the same new
native library for both graphs; it adds no buffering. The five-hop Gather
experiment is excluded because its measured computing time increases.
The earlier padding/Softplus pipeline's 60-second `dynamic4` phone run processes all 600
calls without a queue reset, but still has 32 output gaps. Complete model
work and average RTF below one do not establish continuous output.

The subsequent [normalization/frequency fusion](compute-fusion.md) preserves
this signal/state contract and measures 5.51% less phone model computing time
against the temporal graph. Its native pipeline passes one bounded foreground
recorded-audio minute with 600 calls, no gaps/resets and RTF 0.550647. That
result does not establish longer/background operation, live G2 or automatic
stopped-tail delivery.

## Runtime and integration

The first backend is ONNX Runtime CPU. NNAPI, GPU and vendor NPU execution are
not enabled or validated. Mamba's recurrence and elementwise graph require
operator/partition validation before claiming hardware acceleration. An
advertised phone NPU TOPS figure does not predict this graph's performance.

The Java frontend canonicalizes DC/Nyquist imaginary values to positive zero
before taking phase. Otherwise tiny FFT roundoff can turn positive pi into
negative pi for a negative real endpoint. Those angles represent the same
complex number, but the network takes phase as a scalar, so the distinction
would change its features. An Android speech parity test caught this issue;
the endpoint regression is now covered by unit tests.

Each enhancer uses a sequential ORT session with configurable intra-op thread
count and one inter-op thread. The process-wide ORT environment is shared and
not closed by an enhancer. Constructors close partial native allocations if
loading fails. Manifest validation precedes state allocation. Calls on an
enhancer are synchronized, but audio capture should still use a dedicated
bounded worker pipeline to avoid blocking real-time callbacks.

The AAR contains no model weights, no microphone capture, no resampler, no
network model downloader, and no fallback that substitutes passthrough audio
for failed inference. Applications supply audio at the exported model's exact
sample rate. The benchmark app uses app-specific storage and requires no
microphone permission.

## Even experiment runtime

The separate LC3 interposer owns a bounded worker pipeline, FIR conversion
between 16 kHz transport and the optional 8 kHz model, and double-buffered native
model states. It copies each caller-owned decoded frame before returning. G2
packets deliver five 10 ms codec frames at once. A one-hop graph consumes two
frames per call and uses six ready output frames (60 ms) before playback. A
five-hop graph instead consumes ten frames per 100 ms inference call and uses
a ten-frame (100 ms) output prefill. The first model chunk produces only 80 ms
because synthesis has a pending hop, so native startup can require another
chunk before that prefill is reached. Startup silence and observed pipeline
delay are diagnostic counters, not a promise of 100 ms total latency. Prefill
absorbs bursts but cannot make slow average processing sustainable.

Each newly configured session runs at least four silent hops through a disposable engine
before becoming available. These hops do not seed a real stream's neural/DSP
history. The count is rounded up to a full graph call: four one-hop calls or one
five-hop call. Their startup cost is recorded separately from live processing.

The interposer requests Linux nice -16 only for its own processing worker and
the helper threads created by that ORT session. It records effective niceness
and permission errors; a failed priority request does not bypass Android policy
or abort inference. This follows Android's [audio thread priority definition](https://developer.android.com/reference/android/os/Process#THREAD_PRIORITY_AUDIO),
using per-session [ORT thread callbacks](https://github.com/microsoft/onnxruntime/blob/rel-1.27.0/include/onnxruntime/core/session/onnxruntime_c_api.h).
The reusable Java AAR leaves thread priority to its integrating application.

The native Even worker preallocates 64 input and 128 output frame slots, keeps
the original 50-frame input overload gate, and reuses stream PCM/features.
It signals the worker when enough 10 ms input frames form a complete graph
batch. Neither the larger storage capacity nor fewer wakeups changes reset,
overflow or output-prefill behavior. The resampler retains the same 95 FIR
coefficients and double accumulation order while omitting discarded
downsample outputs and known-zero interpolation products. Resampler regression
tests also preserve shared-history behavior when changing modes without reset.

Optional runtime profiles are confined to a newly configured native model
session: `default` has no overrides, `no_spin` disables helper spinning,
`dynamic4` sets ORT's dynamic block base to four, and `bounded_spin` requests
1,000 us spinning with backoff four and stop-on-return. The bounded profile is
rejected below ORT 1.27. Selecting a different profile creates a new session;
the probe can retain its process and APK for controlled comparisons. The
public Java API and its pinned ORT 1.23.2 session behavior stay unchanged.
Runtime diagnostics report requested/effective profiles, the exact option map
and available providers. Seeing `NnapiExecutionProvider` in availability does
not activate it; these profiles execute with CPU only.

An optional [Android performance-hint session](https://developer.android.com/ndk/reference/group/a-performance-hint)
is resolved dynamically on API 33+ and receives only the interposer worker and
that session's helper thread IDs, with a 20 ms or 100 ms work target matching
the graph call. If creation succeeds, the worker reports measured call durations. Missing functions or a null session
leave the ordinary CPU path active. Diagnostics distinguish symbol availability,
session creation and report counts. The Xperia test found the symbols but
received a null session; it demonstrated no scheduling-hint benefit. This path
does not change CPU affinity, core-online state or global device settings.

The paced probe prepares LC3 packets and an independently decoded baseline
before timing. The phone's measured input loop performs one original LC3 decode
plus enhancement per frame, matching the insertion boundary; actual glasses
perform the encoding. It then replays the baseline through a fresh engine for
offline parity. That replay and the preparatory work are excluded from paced
worker counters and cannot establish real-time acceptance.

Native counters distinguish inference calls from audio hops. RTF divides total
inference time by the represented audio duration. A five-hop deadline miss means
one 100 ms call exceeded 100 ms, not five separate 20 ms misses. The stock Even
microphone stop boundary still lacks a reliable tail callback; its native hook
does not claim automatic flush of the buffered tail. The explicit Java API does
provide flush.

Probe diagnostics are cumulative within the app process; each self-test also
records its own call/hop/error deltas. Acceptance checks the full expected call
and hop counts as well as RTF, gaps, resets, unsupported frames and failures.
For the 20-second Xperia run, 200 graph calls represent all 1,000 input hops.
The returned paced PCM has the original length but includes startup zeros, and
some delayed enhanced tail remains queued when the test stops. Length equality
does not prove every original sample was returned through the live hook.
Nominal startup delay is inferred from zero-frame counts; cumulative source-age
counters are not a wall-clock microphone-to-ASR latency measurement.

## Evidence boundaries

The independent validator imports only hash-verified original NVIDIA topology
files, replacing CUDA Mamba with an independently written sequential CPU
reference. It loads the complete original state dictionary strictly. This
checks the portable graph's topology and state handling against the original,
but it is not a comparison against execution of NVIDIA's CUDA kernels.

ONNX export is checked with the ONNX checker and multiple recurrent frames
against PyTorch. Java audio tests cover independent FFT/DFT agreement,
random/impulse identity, boundaries and lifecycle. Android instrumentation
compares the complete audio stream against an independent NumPy/ORT reference.
These checks establish implementation correctness within recorded tolerances.
Perceptual quality, speech content preservation, physical-phone latency,
sustained thermal behavior, and battery consumption require separate testing.
