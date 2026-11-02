package io.github.reuse.audio.qnn;

import android.content.Context;
import io.github.reuse.ChunkedReuseEnhancer;
import io.github.reuse.ReuseConfig;
import io.github.reuse.live.EnhancementSession;
import io.github.reuse.qnn.QnnHtpBackend;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/** Optional QNN engine adapter for ordinary 16 kHz PCM sessions; it does not open a microphone. */
public final class QnnAudioSession {
    private QnnAudioSession() { }

    /**
     * Returns an unstarted session. Model loading, QNN configuration and verification occur later
     * on its worker. The model must satisfy the core 8 kHz, exit-3, five-hop manifest/IO contract
     * and produce QNN and CPU kernels during warm-up. Output files and profiling
     * belong under an app-private directory. No CPU replacement session is created after failure.
     */
    public static EnhancementSession create(Context context, Path model, Path manifest, Path outputDir,
                                             Runnable workerInitializer, int threads,
                                             EnhancementSession.Listener listener) {
        if (context == null || model == null || manifest == null)
            throw new IllegalArgumentException("Context, model and manifest paths are required");
        Context application = context.getApplicationContext();
        ReuseConfig config = new ReuseConfig(8000, 3, threads);
        return EnhancementSession.withEngine(outputDir, listener, workerInitializer, () -> {
            Files.createDirectories(outputDir);
            QnnHtpBackend backend = new QnnHtpBackend(application, outputDir.resolve("qnn-warmup"), true);
            ChunkedReuseEnhancer enhancer = null;
            try {
                enhancer = ChunkedReuseEnhancer.create(
                        application, model, manifest, config, backend);
                return new Engine(enhancer, backend);
            } catch (Exception | Error failure) {
                if (enhancer != null) OwnedEngineResources.closeAfterCreationFailure(enhancer, failure);
                OwnedEngineResources.closeAfterCreationFailure(backend, failure);
                throw failure;
            }
        });
    }

    private static final class Engine implements EnhancementSession.Engine {
        private final ChunkedReuseEnhancer enhancer;
        private final QnnHtpBackend backend;
        private final OwnedEngineResources resources;
        private boolean warmupVerified;

        Engine(ChunkedReuseEnhancer enhancer, QnnHtpBackend backend) {
            this.enhancer = enhancer; this.backend = backend;
            resources = new OwnedEngineResources(enhancer, backend);
        }

        @Override public float[] process(float[] input) throws Exception { return enhancer.process(input); }
        @Override public float[] flush() throws Exception { return enhancer.flush(); }
        @Override public void reset() { enhancer.reset(); }
        @Override public void close() throws Exception { resources.close(); }

        @Override public void verifyAcceleratorExecution() throws Exception {
            Map<String, Object> evidence = backend.verifyProfile(enhancer.endProfiling(),
                    EnhancementSession.MODEL_WARMUP_CALLS, false);
            QnnWarmupPolicy.requireVerifiedHybrid(evidence, EnhancementSession.MODEL_WARMUP_CALLS);
            warmupVerified = true;
        }

        @Override public Map<String, Object> executionDiagnostics() {
            Map<String, Object> values = new LinkedHashMap<>(enhancer.executionDiagnostics());
            values.putAll(backend.diagnostics());
            values.put("requested_backend", "QNN_HTP");
            values.put("hybrid_execution_requested", true);
            values.put("qnn_hybrid_warmup_verified", warmupVerified);
            values.put("qnn_verification_warmup_calls", EnhancementSession.MODEL_WARMUP_CALLS);
            values.put("qnn_microphone_capture_profile_verified", false);
            values.put("audio_quality_verified", false);
            values.put("execution_verification_scope", "Provider evidence for zero-input warm-up only: every observed QNN kernel executed once per warm-up call, with CPU kernels for remaining computation. Recording is not profiled; numerical correctness, audio quality, continuity and real-time performance require separate validation.");
            return Collections.unmodifiableMap(values);
        }
    }
}
