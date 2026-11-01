# Experimental Even G2 codec interposer

This isolated prototype is specific to the audited Even Android2.4.0/build246
codec boundary. It does not change authentication, signing checks, integrity
controls, server APIs, or backend behavior.

The replacement `liblc3.so` exports `lc3_setup_decoder` and `lc3_decode`, and
has `DT_NEEDED=liblco.so`. The APK must retain its actual original codec as
`liblco.so`, with that original library's SONAME changed to `liblco.so`.
Every original export other than the two wrappers remains discoverable through
the dependency. Wrappers resolve original functions using an explicit
`dlopen("liblco.so")` handle, preventing self-recursion.

**Never package `LINK_ONLY_NEVER_PACKAGE/liblco.so`.** That is a link-metadata
stub only. CMake can instead link a correctly patched original directly using
`-DREUSE_ORIGINAL_LCO=/absolute/path/liblco.so`.

Only successful decodes with tracked configuration10ms/16kHz,
`LC3_PCM_FORMAT_S16` (format0), and stride1 are enhanced. Other formats,
untracked decoders, failed codec calls, and unsupported rates retain original
behavior and have distinct diagnostic counters. The original codec return
code is preserved. Decoded160-sample buffers are copied before return; no
application-owned PCM pointer is retained.

The model uses the app's existing `libonnxruntime.so`, dynamically resolving
`OrtGetApiBase` and requesting C API23. The app runtime is not overwritten;
the observed runtime version is included in diagnostics. The custom scan is
registered from `libreuse_ort_ops.so`. The model SHA256, manifest, all graph
feature/state names, float32 types, and static shapes are checked before use.

Each tracked compatible decoder has separate recurrent/audio states and a
worker. Five10ms frames arriving in one50ms BLE burst are queued immediately.
The worker processes one20ms hop or five chronological hops per model call,
according to `frames_per_call`. Playback requires60ms of ready output for the
one-hop graph and100ms for a five-hop graph. Five-hop input collection adds
up to80ms. Startup, compute and burst timing add further delay; physical chunk
tests observed250–350ms nominal source-frame age. This is not a wall-clock
microphone-to-ASR latency measurement.

Initial output is explicitly silent. Later output underflow inserts silence
and increments `underflow_zero_frames`, with an overload warning. Neither
initial nor underflow silence counts as model-applied audio. Queues are
bounded to500ms; overflow clears state/queues and is counted as a backlog
reset. Initialization/processing failures log an explicit error and preserve
unfiltered decoded PCM, incrementing fallback counters. No hidden bypass is
reported as processed audio.

Frame storage is preallocated:64 input slots and128 output slots, with the
original50-frame overload policy retained. Extra output storage covers a
worker draining already queued input while a callback is delayed; it does not
increase the accepted buffering limit. Worker notifications occur when a full
model batch becomes ready, and real-stream DSP output vectors are preallocated.

The fast model mode accepts16kHz PCM, applies a95-tap Hamming low-pass FIR
(3.4kHz cutoff), decimates to8kHz for RE-USE, then interpolates back to16kHz
with a matching FIR. This deliberately loses high-frequency bandwidth and
adds about5.9ms combined filter group delay. Full16kHz model mode skips those
resamplers. Samples remain float through the model; only the final S16 stage
clips/rounds. FFT DC/Nyquist imaginary values are canonicalized to positive
zero before phase extraction.

The FIR implementation evaluates only retained decimator outputs and nonzero
polyphase interpolation taps, preserving coefficients and double summation
order. Bitwise regression checks cover silence, tiny signals, resets, changing
block lengths and switching resampling direction without reset.

An optional `runtime_profile` in private config selects per-session CPU options:
`default`, `no_spin`, `bounded_spin` or `dynamic4`. Bounded spinning requires
ORT1.27 or later. Requested/effective settings and available providers are
reported; provider availability does not activate hardware offload. Profiles
can be switched by the debug probe without reinstalling it. They do not modify
system scheduling or CPU affinity. Default behavior remains unchanged.

JNI class: `io.github.reuse.even.ReuseBridge`:

```
boolean nativeConfigure(String modelDir, String nativeLibDir,
                        int modelSampleRate, int threads)
String nativeGetDiagnostics()
void nativeReset()
String nativeSelfTest(String inputPcmPath, String outputPcmPath,
                      int pcmSampleRate /* must be16000 */)
```

`nativeReset()` recreates each compatible pipeline, including failed workers.
Idle gaps of at least2seconds clear state/queues as a heuristic. The stock
app caches decoders across recordings, so setup is not a reliable end-of-stream
signal. **There is currently no automatic EOF flush**: queued tail audio can
be lost when a recording ends. Explicit reset drops those tails too. This is
a known experimental integration limitation, not an accepted live G2 result.

The selftest prepares40-byte packets and an original-codec decoded baseline
before pacing; encoding is performed on the glasses in the actual pipeline.
It calls the wrapped decoder in five-frame bursts every50ms and records output
in preallocated memory, writing files after pacing. Burst lateness, decode-call
time and paced wall time are reported. Equal output length includes startup
zeros and does not imply that the delayed recording tail was delivered.
It waits for its worker to quiesce and joins it before final counters are
recorded. Successful output requires applied model frames, nonzero output,
no processing/codec failures, no fallback, and equal length. A separate
`near_realtime_sustained` field additionally requires no post-startup underflow,
no backlog resets, and average processing faster than incoming audio.
External acceptance also checks complete expected model calls/hops and zero
idle resets, explicit resets or unsupported frames. Timing counters use the
100ms model-call deadline for five-hop graphs, normalized by represented audio.

After the paced test, an independent fresh native pipeline replays the exact
decoded baseline offline and writes `.direct.pcm16le`, with at least one artificial
zero EOF hop padded to a full model call and output trimmed to source length.
This artifact is for model/frontend numerical comparison only;
its computation is excluded from paced-worker counters and real-time claims.

Build ARM64 with Android NDK26.1, CMake3.22.1, `ANDROID_STL=c++_static`, and
target `even_reuse_lc3`. All configurations compile at `-O3`, without unsafe
fast-math, and link with16KiB LOAD alignment. The version script exposes only
the two wrappers and JNI methods, keeping C++ runtime symbols private.

This model-specific processing implementation is subject to NVIDIA NSCLv1.
Physical Xperia execution and a20-second paced pass are recorded, alongside
longer runs that fail sustained pacing. See `../../docs/validation.md` for the
exact model revisions and receipts. Live G2, tail, ASR and long-session
acceptance remain separate from a passing bounded probe.
