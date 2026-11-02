# Reusable Android libraries

The enhancement pipeline is separate from microphone capture and application
UI. Both the phone demo and optional G2 companion consume the same libraries.

| Module | Public role | Application dependencies |
|---|---|---|
| `:reuse` | Model loading, recurrent state, STFT, chunked enhancement, and native custom operations | None |
| `:reuse-audio` | Bounded 16 kHz PCM session, resampling, complete stop/drain, WAV pair, and timing statistics | Depends on `:reuse`, not either demo |
| `:reuse-qnn` | Optional QNN HTP backend, registration lifecycle and provider evidence | Depends on `:reuse`; opts into ORT 1.29 / QNN dependencies |
| `:reuse-audio-qnn` | Optional QNN engine for the ordinary PCM session | Depends on `:reuse-audio` and `:reuse-qnn` |
| `:phone-demo` | `AudioRecord`, permission, UI, playback, and user-initiated sharing | Consumes `:reuse-audio` |
| `:live-companion` | Local HTTP/WebSocket transport for the optional G2 Hub route | Consumes `:reuse-audio` |

The `io.github.reuse.live` Java package name is retained for the shared session
and resampler's JNI/API compatibility. It does not imply a dependency on the
G2 application. Neither library opens the microphone, requests permission,
connects to a server, or requires an Even app installation.

For experimental Qualcomm NPU use, see the [QNN hybrid guide](qnn-hybrid.md),
[backend API](../android/reuse-qnn/README.md), and
[PCM adapter API](../android/reuse-audio-qnn/README.md). Optional adapters leave
the default CPU application's runtime dependency unchanged. Raw copied AARs
require their declared dependencies; the vendor runtime is resolved from Maven,
not bundled inside the adapter AARs. Model weights are supplied separately.

The experimental release's library ZIP includes all four AARs, source JARs,
and POMs. For manually copied QNN AARs, declare the runtime dependencies too:

```kotlin
dependencies {
    implementation(files("libs/reuse-release.aar", "libs/reuse-audio-release.aar",
        "libs/reuse-qnn-release.aar", "libs/reuse-audio-qnn-release.aar"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.29.0")
    implementation("com.qualcomm.qti:onnxruntime-android-qnn:2.7.0")
    implementation("com.qualcomm.qti:qnn-runtime:2.51.0")
}
```

Use the adapter guides' native extraction settings and preserve component
notices. The QNN adapters target compatible ARM64 Qualcomm devices on API 29+.
The default CPU-only dependency example below continues to use ORT 1.23.2.

## Build and consume

Use the [documented Android toolchain](phone-demo.md#build-from-source):

```powershell
.\gradlew.bat :reuse:assembleRelease :reuse-audio:assembleRelease -PreuseNeonFma=true -PreuseScanN16Registers=true
```

The artifacts are:

```text
android/reuse/build/outputs/aar/reuse-release.aar
android/reuse-audio/build/outputs/aar/reuse-audio-release.aar
```

Within this checkout, an application declares:

```groovy
dependencies {
    implementation project(':reuse-audio')
}
```

For manually copied AARs, include both local artifacts and the external runtime
dependency:

```kotlin
dependencies {
    implementation(files("libs/reuse-audio-release.aar", "libs/reuse-release.aar"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.2")
}
```

The `:reuse-audio` AAR contains the FIR JNI library. The core AAR contains the
RE-USE custom operations, and ONNX Runtime is resolved separately. Model
weights are not bundled in either AAR. Prepare the matching model directory
as described in the [phone build guide](phone-demo.md#build-from-source), and
copy its files into app-private storage during application setup.

## Process a 16 kHz PCM stream

`EnhancementSession` accepts signed 16-bit little-endian mono PCM at 16 kHz.
Packet boundaries need not match the model's 100 ms processing interval.
Supply monotonically increasing input sample indices to detect discontinuities
in the application's own stream. Run capture and waiting outside the UI or
audio callback thread.

The following illustrates the lifecycle for a paced iterable of PCM packets.
`modelDirectory` contains the verified `model.onnx`, `manifest.json`, and
`LICENSE-NVIDIA.txt`; `customLibrary` is the installed
`libreuse_ort_ops.so` under the application's `nativeLibraryDir`.

```java
import io.github.reuse.live.EnhancementSession;
import java.nio.file.Path;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

// Invoke this on an application worker, never the main thread.
Map<String, Object> enhance(
        Path modelDirectory, Path outputDirectory, Path customLibrary,
        Iterable<byte[]> pcmPackets) throws Exception {
    CompletableFuture<Void> ready = new CompletableFuture<>();
    CompletableFuture<Map<String, Object>> done = new CompletableFuture<>();
    EnhancementSession.Listener listener = new EnhancementSession.Listener() {
        public void onReady() { ready.complete(null); }
        public void onAudio(byte[] enhancedPcm16le) {
            // Optional live output: hand off to a bounded nonblocking consumer.
            // The session also saves the complete enhanced WAV on termination.
        }
        public void onFinished(Map<String, Object> statistics) {
            done.complete(statistics);
        }
        public void onError(String code, String message,
                            Map<String, Object> statistics) {
            Exception error = new IllegalStateException(code + ": " + message);
            ready.completeExceptionally(error);
            done.completeExceptionally(error);
        }
    };
    try (EnhancementSession session = new EnhancementSession(
            modelDirectory, outputDirectory, customLibrary, listener,
            () -> android.os.Process.setThreadPriority(
                    android.os.Process.THREAD_PRIORITY_AUDIO), 4)) {
        session.start();
        ready.get(30, TimeUnit.SECONDS);
        long firstSample = 0;
        for (byte[] pcm : pcmPackets) {
            if (!session.accept16kS16LE(pcm, firstSample)) {
                throw new IllegalStateException("PCM packet rejected");
            }
            firstSample += pcm.length / 2;
        }
        session.finish();
        return done.get(30, TimeUnit.SECONDS);
    }
}
```

This API is asynchronous. `finish()` orders a normal stop after all accepted
input, then flushes the model and FIR tails. Wait for `onFinished` before
reading finalized WAVs. `close()` requests an abort if the session is still
active; it is not a substitute for a normal `finish()` and drain. Listener
callbacks run on the processing worker and must not block it.

The current session bounds recording to 120 seconds and queued input to
500 ms. Applications can enforce shorter limits; the phone demo stops capture
at 60 seconds. An input queue overflow is an explicit failed session, with
statistics and any partial recording preserved. Rejected input must not be
silently skipped or retried as though the recurrent state were uninterrupted.

## Timing and validity

`statistics()` exposes the session's counts, queue state, processing timing,
deadline misses, and termination status. Active-run snapshots are estimates;
the terminal callback's result is the completed measurement. Processing RTF
below one establishes average processing speed for the tested workload, not
every-call deadline compliance or microphone-to-speaker latency.

A successful nonempty stream has equal accepted and processed input counts,
no application drops or overflow, a complete model flush, and
`N + (N % 2) + 188` output samples. Interpret capture-source and hardware
continuity separately: an application's sample index can detect its own packet
gaps but cannot prove that an upstream driver or wireless source never lost
audio.

For lower-level integration, `ChunkedReuseEnhancer` in `:reuse` accepts mono
float PCM at the model's sample rate with model-specific chunk sizes. Use a
matching graph and manifest; changing a sample-rate setting alone does not
convert the model. See the [core Kotlin example](../README.md#use-from-kotlin)
and [validation scope](validation.md).

## Licensing

The reusable Java integration is MIT. The adapted model implementation,
checkpoint, and model-specific native kernels remain subject to NVIDIA NSCLv1
for noncommercial research and education. Each artifact retains its relevant
notices. See [third-party materials](../THIRD_PARTY_NOTICES.md).
