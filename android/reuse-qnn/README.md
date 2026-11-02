# Optional QNN HTP backend

This Android library supplies a session backend for `ChunkedReuseEnhancer`.
The core `:reuse` module keeps its existing ONNX Runtime 1.23.2 dependency;
applications opting into `:reuse-qnn` resolve the adapter's pinned ONNX Runtime
1.29.0, QNN provider 2.7.0, and QNN runtime 2.51.0 dependencies. The Qualcomm
runtime is resolved from Maven and is not shaded into this library's AAR.

The target is Android 10 or newer on a compatible Qualcomm ARM64 device.
Model compatibility and actual NPU execution must be verified on the device.
Registering the provider or selecting an NPU device does not establish execution.

```groovy
dependencies { implementation project(':reuse-qnn') }
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

Create a unique profiling prefix beneath an **existing**, writable app-private
files or cache directory. Use a CPU `ReuseConfig`: the external backend owns
provider configuration and cannot be combined with built-in NNAPI configuration.

```java
Path directory = context.getFilesDir().toPath().resolve("qnn-test");
Files.createDirectories(directory);
Path prefix = directory.resolve("stream-" + UUID.randomUUID());
ReuseConfig config = new ReuseConfig(8000, 3, 2);
try (QnnHtpBackend backend = new QnnHtpBackend(context, prefix, true)) {
    try (ChunkedReuseEnhancer enhancer = ChunkedReuseEnhancer.create(
            context, modelPath, manifestPath, config, backend)) {
        // Process actual input, count all profiled model calls, and validate output separately.
        Map<String, Object> evidence = backend.verifyProfile(
                enhancer.endProfiling(), profiledCalls, false);
        // Require Boolean.TRUE.equals(evidence.get("qnn_execution_verified")) before claiming execution.
    }
}
```

The example permits CPU fallback for unsupported model operators. Pass `false`
to require QNN-only kernel execution. HTP mode is `balanced`; graph-I/O
quantization stays on QNN. The session's ORT CPU thread settings remain those
selected in `ReuseConfig`.

End profiling exactly once, after the calls being verified. Include warm-up and
flush calls in `profiledCalls`. Request a single QNN partition only for a graph
expected to produce exactly one QNN kernel event per call; verification then
requires one unique QNN kernel name and exactly `profiledCalls` events. Other
profiles require at least that many QNN events. CPU events are accepted only
when fallback was explicitly enabled. Unknown or unattributed kernel events
fail verification. Counts describe ORT provider events, not hardware utilization
or percentage of computational work. Correct output, streaming continuity,
audio quality, and real-time performance are separate checks.

Each backend instance configures one session. Keep the backend open until its
consuming enhancer has closed, as the nested `try` example does. Adapter
instances share a process-local registration only for the same ORT environment
and canonical provider-library path. The last instance unregisters the plugin
and restores the previous `ADSP_LIBRARY_PATH`. Do not independently register the
same plugin or change that environment variable while adapter instances are
active. A failed native cleanup prevents unsafe registration retries until the
app process restarts.

The adapter does not change application microphone, networking, or Internet
permissions. Its manifest exposes optional `libcdsprpc.so` visibility for
compatible platform FastRPC implementations.

Adapter Java source is MIT. Qualcomm runtime dependencies remain subject to the
Qualcomm AI Stack License; NVIDIA model weights and model-specific native code
retain their existing NSCLv1 restrictions. Original notices are included under
the AAR's `reuse-qnn-notices/` assets and in `docs/licenses/qnn`. Preserve them in
the consuming application. This adapter does not include model weights or
qualify an arbitrary graph for HTP execution.
