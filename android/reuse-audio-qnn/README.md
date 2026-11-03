# Optional QNN PCM sessions

`QnnAudioSession` connects the bounded, vendor-independent `:reuse-audio` PCM
pipeline to the optional `:reuse-qnn` backend. Neither the stable phone app nor
the core PCM module gains a Qualcomm dependency. This AAR contains adapter Java
code and notices; model weights and Qualcomm native binaries are not shaded
into it. Its API dependencies resolve the core PCM and optional backend modules.

Target: compatible Qualcomm ARM64 devices running Android 10 or newer. The
backend pins ONNX Runtime 1.29.0, QNN provider 2.7.0, and QNN runtime 2.51.0.
Configure the consuming app to extract and preserve its native libraries:

```groovy
dependencies { implementation project(':reuse-audio-qnn') }
android {
    defaultConfig { ndk { abiFilters 'arm64-v8a' } }
    packaging {
        jniLibs {
            useLegacyPackaging = true
            keepDebugSymbols += ['**/*.so']
        }
    }
}
```

```java
Path output = context.getFilesDir().toPath().resolve("capture-" + UUID.randomUUID());
EnhancementSession session = QnnAudioSession.create(
        context, modelPath, manifestPath, output,
        () -> android.os.Process.setThreadPriority(android.os.Process.THREAD_PRIORITY_AUDIO),
        2, listener);
session.start();
// After listener.onReady(), submit ordinary mono 16 kHz PCM16LE packets:
session.accept16kS16LE(pcmBytes, firstSampleIndex);
// Drain accepted input and save both recordings:
session.finish();
```

Creation returns an unstarted session and defers native work to its processing
worker. The core validates the supplied model hash, manifest and IO. Paths are
generic: this adapter does not hardcode a particular graph hash. The graph must
match the 8 kHz, exit-3, five-hop contract and produce HTP and CPU kernels.
The PCM session performs its existing 16→8→16 kHz FIR conversion, keeps
its 500 ms queue limit, and writes its existing before/after files and statistics.

Startup runs `EnhancementSession.MODEL_WARMUP_CALLS` zero-input calls, currently
three. Before readiness, the adapter ends the warm-up profile and requires
verified QNN execution, a per-kernel event histogram in which every observed QNN
kernel appears exactly once per warm-up call, and CPU events for remaining
computation. Multiple QNN partitions are supported; the histogram total and
unique-kernel count must also agree. It then resets all core state for
actual input. HTP runs in balanced mode with FP16 disabled, graph-IO
quantization on QNN, and explicit CPU fallback for unsupported model operators.
This startup check is provider evidence, not an audio-quality acceptance test.

The application opens its microphone only after `onReady`; this library never
opens one. A failed QNN gate reports a session error rather than starting a new
CPU replacement session. Recording runs without ORT profiling after the startup
check. Diagnostics therefore identify **warm-up** verification explicitly and
leave microphone capture profile verification and audio quality unverified.
The application must assess candidate-model quality, captured input continuity,
queue/drop counts, processing deadlines, and device performance separately.

The engine owns its enhancer and backend. On completion or failure it closes
the enhancer first, then the backend, once, preserving the original failure and
any cleanup errors. Use `finish()` for an ordered drain; `close()` requests an
asynchronous abort through the generic PCM session.

Adapter Java source is MIT. Qualcomm runtime dependencies retain the AI Stack
License. NVIDIA model weights and model-specific native code retain NSCLv1
restrictions. Preserve all dependency notices; this AAR also packages them under
`reuse-audio-qnn-notices/`. The application controls microphone, network, and
Internet permissions.
