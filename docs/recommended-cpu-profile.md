# Recommended CPU demo profile

Start with the causal **8 kHz / exit 3 / FP32 CPU** model. The ordinary phone
demo and the reusable AARs use the same model pipeline. GPU, HTP and partially
quantized graphs remain separate experiments.

## Reproducible configuration

| Setting | Recommended demo build |
| --- | --- |
| Model profile | `reuse-8k-exit3-fp32-cpu` |
| Graph SHA-256 | `f950dcfb9321191b2bcb3ab9afa3e100c40e5c64dd1de1d30849be4c1aacce32` |
| Model | Causal Real-time RE-USE; three Mamba blocks; zero neural look-ahead |
| Selected active parameters | 1,823,555 |
| Numeric representation | Original selected FP32 weights and states; no quantization |
| Model call | Five chronological 20 ms hops, 100 ms / 800 samples at 8 kHz |
| CPU runtime | ONNX Runtime 1.29.0 |
| Execution | Sequential graph, full graph optimization, four intra-op / one inter-op threads, spinning disabled |
| Native ARM settings | NEON FMA and rank-16 register scan enabled in normal Android builds |
| Phone frontend | Mono PCM16 at 16 kHz, exact FIR conversion around the 8 kHz enhancer |
| Backpressure | Bounded 500 ms input queue; explicit failure on overflow or sequence discontinuity |
| Startup / shutdown | Load, three warm-up calls and state reset before capture; ordered stop and complete model/FIR drain |

The default four-thread suggestion comes from Xperia experiments. It is not
a universal optimum. Compare two threads on another phone, and test sustained
operation with the actual capture route. The generic `ReuseConfig(sampleRate)`
constructor retains its existing two-thread default.

The model and pinned export manifest are unchanged. The separate advertised
9.6M noncausal RE-USE model is not a replacement for this streaming graph.
Input/output at 16 kHz does not make this a 16 kHz enhancement model: useful
model bandwidth remains below 4 kHz. Five-hop batching adds collection delay;
zero neural look-ahead does not mean zero end-to-end latency.

## Use from another application

Use `:reuse` for the model/STFT API and `:reuse-audio` for the bounded 16 kHz
PCM session. Neither library opens a microphone or depends on G2 or the demo UI.

```java
import io.github.reuse.RecommendedCpuProfile;
import io.github.reuse.ReuseConfig;

ReuseConfig config = RecommendedCpuProfile.config(); // 8 kHz, exit 3, CPU, 4 threads
ReuseConfig comparison = RecommendedCpuProfile.config(2);
String expectedGraph = RecommendedCpuProfile.MODEL_SHA256;
```

Supply the pinned model separately and verify its identity before inference.
`ChunkedReuseEnhancer` validates the supplied graph against its manifest; the
demo additionally checks the expected graph pin. The profile factory selects
settings; it does not download weights or certify an arbitrary model directory.

The shared `EnhancementSession` retains its four-thread CPU constructor and
offers an explicit thread argument. Its callbacks run on the processing worker.
Use sample-indexed `accept16kS16LE`, keep capture and consumers nonblocking,
call `finish()` for an ordered drain, and wait for a terminal callback before
reading completed WAVs. See the [complete integration example](library.md).

Build the APK and separate library artifacts with the normal defaults:

```powershell
.\gradlew.bat :phone-demo:assembleDebug :phone-demo:lintDebug :reuse:assembleRelease :reuse-audio:assembleRelease
```

The ARM options can be disabled explicitly for a controlled legacy comparison:

```powershell
.\gradlew.bat :phone-demo:assembleDebug -PreuseNeonFma=false -PreuseScanN16Registers=false
```

That is a different native build, so retain its hash and perform fresh numerical
and timing checks. Standalone CMake host defaults remain unchanged. ARM-specific
instructions are not claimed to execute on x86. The legacy experimental
GPU-only NNAPI runtime remains a separately selected ORT 1.23.2 build.

## Observe what actually ran

The ordinary CPU app now offers **Share diagnostics (no audio)**, including
failed preparation attempts. It shares JSON records, not WAV files. Audio
sharing is a separate explicit action. Leaving the screen records a background
stop reason and releases capture through the existing drain lifecycle.

Record the selected model profile, graph hash, thread setting, actual ORT version
and build string, and `RuntimeBuildInfo.kernelBuildInfo()` output. The latter
reports compiled architecture/options, not proof that a specific branch ran
on every call. The native library hash in the build receipt identifies the
packaged bytes. Avoid inferring configuration solely from Gradle command text.

Completed call count, total/maximum time, percentile history and deadline count
are now published together to concurrent observers. This removes a race in
live timing summaries without locking around model inference.

For a run, also retain input/processed/output counts, drops, queue overflows,
queue age, call-time tails, stop reason, flush result and microphone release.
Do not convert an average RTF below one into a speaker-latency or underrun claim.
The demo saves a before/after pair and does not provide live speaker monitoring.

## Why CPU is the recommended route

The instrumented Xperia matrix measured **38.327 ms mean / 45.553 ms maximum**
with four FP32 CPU threads and no 100 ms overrun among 130 calls. This was a
short unpaced public-input experiment using ORT 1.29 and the ARM options above.
It is a configuration-selection input, not a microphone result for a new APK.
See [the telemetry evidence and limits](benchmark-telemetry.md).

The strict full neural GPU experiment passed its numerical limits but averaged
170.449 ms, against its paired 39.629 ms CPU control. The latest hybrid HTP
comparison averaged 52.925 ms versus its paired 50.572 ms two-thread FP32 CPU
control. These are separate runs; they do not isolate a runtime-upgrade benefit.
GPU and NPU profiling may reveal useful future work, but their presence alone
does not justify changing the default.

Partially quantized CPU configurations with two or more threads failed the
fixed cross-thread consistency gate. The experimental microphone adapter now
requires one thread for that choice and rejects incompatible start requests.
One-thread eligibility is not a general perceptual-quality certification.

## Validation boundary

The [new build receipt](validation/recommended-cpu-demo.json) separates build,
unit/JNI, package and device checks. Historical microphone recordings belong
to the APKs identified in their original receipts; they do not automatically
validate this runtime update. A fresh phone capture is required before making
a new microphone-throughput claim, and sustained performance needs longer tests.

ONNX Runtime documents [threading controls](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
and [compatibility policy](https://onnxruntime.ai/docs/reference/compatibility.html).
This project pins a measured version rather than selecting a newer runtime
solely by version number. Preserve the [component license boundaries](license.md).
