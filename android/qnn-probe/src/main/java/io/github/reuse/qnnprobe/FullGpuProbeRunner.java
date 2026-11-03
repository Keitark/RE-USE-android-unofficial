package io.github.reuse.qnnprobe;

import android.content.Context;
import android.os.SystemClock;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtLoggingLevel;
import io.github.reuse.ChunkedReuseEnhancer;
import io.github.reuse.ReuseConfig;
import io.github.reuse.SessionOptionsConfigurer;
import io.github.reuse.live.EnhancementSession;
import io.github.reuse.qnn.QnnGpuBackend;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CancellationException;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;
import java.util.stream.Stream;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Strict standard-ONNX neural-graph GPU experiment on a packaged public clip.
 * Host STFT, synthesis, state-buffer management and Java remain CPU work. No microphone is opened.
 */
final class FullGpuProbeRunner {
    static final int INPUT_SAMPLES = 51840, CALLS = 65, CHUNK = 800, WARMUP = 3, THREADS = 4;
    static final String BASELINE_SHA = "f950dcfb9321191b2bcb3ab9afa3e100c40e5c64dd1de1d30849be4c1aacce32";
    static final String CHECKPOINT_SHA = "6eac4e8cc721f99a22cde7eac3c9f8e2826ced5dc8a4234fd1977de1ec0d3ade";
    private static final String REVISION = "4c0c186272ca853cd18d832fc5c28777b62f32d1";
    private static final String LICENSE_SHA = "6d1daa9c89ec421ba80fe051ac9f0dd46010d4fa4e8f4af654eb8e62f64e2ad5";
    private static final String INPUT_SHA = "e6af48deb208432e7eeb624058b98ceb4b4542ab1a206c04fa621f3684ac33b5";
    private static final int MAX_ASSET_BYTES = 64 * 1024 * 1024, MAX_REPORT_BYTES = 8 * 1024 * 1024;
    private static final long BUDGET_NS = TimeUnit.SECONDS.toNanos(180);

    static final class Result {
        final File directory;
        final JSONObject diagnostics;
        Result(File directory, JSONObject diagnostics) { this.directory = directory; this.diagnostics = diagnostics; }
    }

    private static final class ModelFiles {
        final Path model, manifest;
        ModelFiles(Path model, Path manifest) { this.model = model; this.manifest = manifest; }
    }

    private FullGpuProbeRunner() { }

    static Result run(Context context, QnnProbeRunner.Progress progress, BooleanSupplier cancelled) {
        JSONObject report = new JSONObject();
        Path directory = null;
        BenchmarkTelemetry telemetry = null;
        long began = System.nanoTime(), deadline = began + BUDGET_NS;
        String stage = "full_gpu.initialize";
        put(report, "schema_version", 1);
        put(report, "source", "strict_full_neural_qnn_gpu_public_clip");
        put(report, "app_version", BuildConfig.VERSION_NAME);
        put(report, "status", "RUNNING");
        put(report, "microphone_started", false); put(report, "audio_captured", false);
        put(report, "audio_quality_verified", false); put(report, "live_queue_or_playout_verified", false);
        put(report, "full_neural_gpu_verified", false); put(report, "gpu_backend_execution_verified", false);
        put(report, "cpu_ep_fallback_requested_disabled", true); put(report, "cpu_ep_fallback_disabled", false);
        put(report, "gpu_hardware_counter_trace_collected", false);
        put(report, "scope", "Strict standard-ONNX neural graph via libQnnGpu.so with CPU EP fallback disabled. Host STFT, synthesis, state-buffer management and Java remain CPU work. Public offline clip only; no microphone, live playout, perceptual quality, energy or hardware-utilization acceptance.");
        put(report, "soft_budget_seconds", 180);
        put(report, "budget_scope", "Cancellation and wall budget are cooperative checks between synchronous stages/calls. Native session compilation or inference cannot be forcibly interrupted and may exceed the budget.");
        put(report, "whole_max_abs_limit", GpuResourceProbe.WHOLE_MAX_ABS);
        put(report, "whole_normalized_rmse_limit", GpuResourceProbe.WHOLE_NORMALIZED_RMSE);
        put(report, "sample_rate", 8000); put(report, "frames_per_call", 5); put(report, "exit_layers", 3);
        put(report, "requested_cpu_intra_op_threads", THREADS); put(report, "cpu_inter_op_threads", 1);
        put(report, "session_spinning_enabled", false);
        put(report, "timing_scope", "Sequential original FP32 CPU reference then portable FP32 GPU candidate; unprofiled process/flush includes host STFT and synthesis, excludes preparation/warm-up. Clocks and thermal order are uncontrolled; CPU pool size is not proof of active cores.");
        try {
            if (context == null || progress == null || cancelled == null)
                throw new IllegalArgumentException("Context, progress and cancellation supplier are required");
            Context application = context.getApplicationContext();
            directory = captureDirectory(application);
            telemetry = BenchmarkTelemetry.start(application, directory);
            telemetry.mark("strict_full_gpu", "host", "initialize", THREADS, -1);
            put(report, "benchmark_telemetry_instrumented", true);
            put(report, "benchmark_telemetry_scope", "Concurrent clock, temperature and load sampling can affect wall time; process-wide CPU time includes sampler work. Call-event enqueue occurs after both timing snapshots.");
            checkpoint(report, directory, stage);
            checkContinue(cancelled, deadline);
            stage = "full_gpu.runtime.initialize_info";
            telemetry.mark("strict_full_gpu", "host", "runtime_initialize", THREADS, -1);
            checkpoint(report, directory, stage); progress.update("Preparing strict neural GPU test…");
            OrtEnvironment environment = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO);
            environment.setTelemetry(false);
            put(report, "ort_version", environment.getVersion()); put(report, "ort_telemetry_enabled", false);
            put(report, "ort_environment_scope", "Process-wide singleton; an environment initialized earlier may retain its earlier logging level. Missing native identity logs withhold GPU verification.");

            stage = "full_gpu.assets.validate";
            telemetry.mark("strict_full_gpu", "host", "validate_assets", THREADS, -1);
            checkpoint(report, directory, stage); checkContinue(cancelled, deadline);
            JSONObject top = json(asset(application, "manifest.json"));
            JSONObject full = top.getJSONObject("full_stream"), input = full.getJSONObject("input");
            validateInputContract(full);
            JSONObject baselineSpec = full.getJSONObject("baseline"), candidateSpec = top.getJSONObject("full_gpu_standard");
            validateAssetSpec(baselineSpec, "full-baseline");
            validateAssetSpec(candidateSpec, "full-gpu-standard");
            if (!BASELINE_SHA.equals(baselineSpec.getString("model_sha256")))
                throw new IOException("CPU reference is not the pinned original F950 FP32 graph");
            byte[] baselineMetadata = checkedAsset(application, baselineSpec, "manifest");
            byte[] candidateMetadata = checkedAsset(application, candidateSpec, "manifest");
            byte[] validationBytes = checkedAsset(application, candidateSpec, "validation");
            JSONObject baselineManifest = json(baselineMetadata), candidateManifest = json(candidateMetadata);
            JSONObject validation = json(validationBytes);
            validateExportContract(validation, candidateManifest, baselineManifest, candidateSpec.getString("model_sha256"));
            put(report, "host_export_validation", validation);
            put(report, "host_validation_sha256", candidateSpec.getString("validation_sha256"));
            put(report, "baseline_model_sha256", BASELINE_SHA);
            put(report, "candidate_model_sha256", candidateSpec.getString("model_sha256"));
            put(report, "candidate_manifest_sha256", candidateSpec.getString("manifest_sha256"));
            put(report, "public_input_sha256", INPUT_SHA);
            float[] audio = ProbeVerification.decodeFloat32(checkedAsset(application, input.getString("file"), INPUT_SHA), INPUT_SAMPLES);
            requireFinite(audio);
            ModelFiles baseline = copyModel(application, directory, "cpu-reference-model", baselineSpec, baselineMetadata);
            ModelFiles candidate = copyModel(application, directory, "full-gpu-model", candidateSpec, candidateMetadata);
            Path dumps = directory.resolve("qnn-private-graphs"); Files.createDirectory(dumps);

            stage = "full_gpu.cpu_reference";
            JSONObject cpu = new JSONObject(); put(report, "cpu_reference", cpu);
            checkpoint(report, directory, stage);
            float[] reference = completeStream(application, baseline, audio, directory, null, cpu,
                    report, "cpu-reference", deadline, cancelled, progress, telemetry);
            stage = "full_gpu.full_neural_gpu";
            JSONObject gpu = new JSONObject(); put(report, "full_neural_gpu", gpu);
            checkpoint(report, directory, stage); checkContinue(cancelled, deadline);
            float[] actual = completeStream(application, candidate, audio, directory, dumps, gpu,
                    report, "full-gpu", deadline, cancelled, progress, telemetry);
            stage = "full_gpu.whole_wave_compare";
            telemetry.mark("strict_full_gpu", "host", "waveform_comparison", THREADS, -1);
            checkpoint(report, directory, stage); checkContinue(cancelled, deadline);
            JSONObject comparison = FullStreamingBenchmark.audioComparison(actual, reference);
            boolean numerical = GpuResourceProbe.wholeNumericalPass(comparison.optDouble("max_abs", Double.NaN),
                    comparison.optDouble("normalized_rmse", Double.NaN));
            put(comparison, "max_abs_limit", GpuResourceProbe.WHOLE_MAX_ABS);
            put(comparison, "normalized_rmse_limit", GpuResourceProbe.WHOLE_NORMALIZED_RMSE);
            put(comparison, "fixed_numerical_gate_passed", numerical);
            put(comparison, "scope", "Complete chronological public waveform against original F950 FP32 on this device; numerical consistency, not perceptual audio-quality acceptance.");
            put(report, "whole_wave_comparison", comparison);
            boolean verified = acceptsWholeRun(cpu, gpu, comparison);
            put(report, "full_neural_gpu_verified", verified); put(report, "gpu_backend_execution_verified", verified);
            put(gpu, "gpu_backend_execution_verified", verified); put(gpu, "full_neural_gpu_verified", verified);
            put(gpu, "status", verified ? "VERIFIED_FULL_NEURAL_GPU" : "COMPLETE_UNVERIFIED");
            put(report, "status", verified ? "VERIFIED_FULL_NEURAL_GPU_PUBLIC_SEQUENCE"
                    : !numerical ? "WHOLE_WAVE_NUMERICAL_GATE_FAILED" : "GPU_EXECUTION_EVIDENCE_INCOMPLETE");
            if (verified) {
                double cpuMs = cpu.getDouble("mean_call_wall_ms"), gpuMs = gpu.getDouble("mean_call_wall_ms");
                put(report, "cpu_over_gpu_call_speedup", cpuMs / gpuMs);
                put(report, "gpu_faster_than_cpu", gpuMs < cpuMs);
                put(report, "speedup_scope", "Sequential full-pipeline wall-call mean ratio; native-optimized CPU graph and portable GPU graph share model/settings but graph decomposition differs. No controlled thermal, energy or live playout claim.");
            }
            stage = "full_gpu.complete";
        } catch (Exception | LinkageError error) {
            put(report, "status", stopStatus(error));
            put(report, "failure_stage", report.optString("last_stage", stage)); recordError(report, error);
        } finally {
            if (telemetry != null) {
                telemetry.mark("strict_full_gpu", "host", "complete", THREADS, -1);
                telemetry.close();
                put(report, "benchmark_telemetry", telemetry.summary());
            }
            put(report, "total_wall_ms", elapsed(began)); put(report, "final_stage", stage);
            if (directory != null) {
                try { inventoryDumps(report, directory.resolve("qnn-private-graphs")); }
                catch (Exception error) { put(report, "private_graph_inventory_error", error.toString()); }
                try { checkpoint(report, directory, stage); }
                catch (Exception error) { put(report, "diagnostics_write_error", error.toString()); }
            }
        }
        return new Result(directory == null ? null : directory.toFile(), report);
    }

    private static float[] completeStream(Context context, ModelFiles model, float[] audio, Path directory,
                                           Path dumpDirectory, JSONObject result, JSONObject report, String prefix,
                                           long deadline, BooleanSupplier cancelled, QnnProbeRunner.Progress progress,
                                           BenchmarkTelemetry telemetry) throws Exception {
        boolean gpu = dumpDirectory != null;
        String stage = prefix + ".before_load";
        put(result, "status", "RUNNING"); put(result, "cpu_intra_op_threads", THREADS);
        put(result, "cpu_inter_op_threads", 1); put(result, "timed_calls_profiled", false);
        put(result, "benchmark_telemetry_instrumented", telemetry != null);
        put(result, "cpu_ep_fallback_requested_disabled", gpu); put(result, "cpu_ep_fallback_disabled", false);
        put(result, "gpu_backend_execution_verified", false); put(result, "full_neural_gpu_verified", false);
        put(result, "provider_verification_scope", "Exactly three profiled zero-input warm-up calls, then end profiling and reset all streaming state before 65 unprofiled audio calls and one flush.");
        put(result, "host_cpu_work", "STFT, synthesis, state-buffer management, ORT orchestration and Java remain on the host; GPU strictness applies to neural graph execution.");
        ChunkedReuseEnhancer enhancer = null;
        QnnGpuBackend backend = null;
        GpuResourceProbe.BackendLogCapture logs = null;
        boolean profileEnded = false;
        Exception failure = null;
        try {
            checkpoint(report, directory, stage); checkContinue(cancelled, deadline);
            progress.update(gpu ? "Compiling full neural graph for Adreno · CPU fallback disabled…" : "Original FP32 reference · 4 CPU threads…");
            if (telemetry != null) telemetry.mark("strict_full_gpu", prefix, "before_load", THREADS, -1);
            if (gpu) {
                logs = new GpuResourceProbe.BackendLogCapture(directory.resolve("full-gpu-backend.log"), 2 * 1024 * 1024);
                backend = new QnnGpuBackend(context, directory.resolve(prefix + "-ort"), false, dumpDirectory);
                put(result, "backend_configuration", new JSONObject(backend.diagnostics()));
            }
            final QnnGpuBackend configuredBackend = backend;
            SessionOptionsConfigurer configurer = gpu ? configuredBackend : (env, options) ->
                    options.enableProfiling(directory.resolve(prefix + "-ort").toString());
            stage = prefix + ".create_compile";
            if (telemetry != null) telemetry.mark("strict_full_gpu", prefix, "compile", THREADS, -1);
            checkpoint(report, directory, stage); checkContinue(cancelled, deadline);
            long compile = System.nanoTime();
            enhancer = ChunkedReuseEnhancer.create(context, model.model, model.manifest,
                    new ReuseConfig(8000, 3, THREADS), configurer);
            put(result, "session_create_compile_ms", elapsed(compile));
            if (gpu) put(result, "backend_configuration", new JSONObject(backend.diagnostics()));
            stage = prefix + ".profiled_warmup";
            checkpoint(report, directory, stage);
            for (int call = 0; call < WARMUP; call++) {
                if (telemetry != null) telemetry.mark("strict_full_gpu", prefix, "warmup", THREADS, call);
                checkContinue(cancelled, deadline); requireFinite(enhancer.process(new float[CHUNK]));
                put(result, "completed_warmup_calls", call + 1);
            }
            Path profile = enhancer.endProfiling(); profileEnded = true;
            if (telemetry != null) telemetry.mark("strict_full_gpu", prefix, "verify_profile", THREADS, -1);
            put(result, "warmup_profile_file", profile.getFileName().toString());
            JSONObject evidence = FullStreamingBenchmark.profileSummary(new String(boundedFile(profile, MAX_ASSET_BYTES), StandardCharsets.UTF_8));
            FullStreamingBenchmark.requireWarmupCallCount(evidence);
            put(result, "warmup_provider_evidence", evidence);
            boolean providerPass;
            if (gpu) {
                Map<String, Object> verification = backend.verifyProfile(profile, WARMUP, false);
                put(result, "backend_configuration", new JSONObject(verification));
                providerPass = strictProviderPass(evidence, new JSONObject(verification));
            } else {
                providerPass = evidence.optInt("cpu_kernel_events", 0) > 0 && evidence.optInt("qnn_kernel_events", -1) == 0
                        && evidence.optInt("other_provider_kernel_events", -1) == 0 && evidence.optInt("missing_provider_kernel_events", -1) == 0;
            }
            put(result, "warmup_provider_verified", providerPass);
            if (!providerPass) throw new IOException(gpu ? "Strict GPU warm-up contains missing/non-QNN execution or unbalanced calls" : "Original reference did not execute entirely on CPU");
            if (telemetry != null) telemetry.mark("strict_full_gpu", prefix, "reset", THREADS, -1);
            enhancer.reset();
            put(result, "all_streaming_state_reset_after_warmup", true);
            stage = prefix + ".unprofiled_sequence";
            checkpoint(report, directory, stage);
            float[] accumulated = new float[CALLS * CHUNK], input = new float[CHUNK];
            double[] milliseconds = new double[CALLS], cpuMilliseconds = new double[CALLS];
            int written = 0;
            for (int call = 0; call < CALLS; call++) {
                checkContinue(cancelled, deadline);
                if (call % 5 == 0) {
                    progress.update((gpu ? "Full neural GPU" : "Original CPU reference") + " · " + call + "/" + CALLS);
                    checkpoint(report, directory, stage);
                }
                Arrays.fill(input, 0);
                int count = Math.min(CHUNK, audio.length - call * CHUNK);
                if (count > 0) System.arraycopy(audio, call * CHUNK, input, 0, count);
                if (telemetry != null) telemetry.mark("strict_full_gpu", prefix, "audio", THREADS, call);
                long traceStart = telemetry == null ? 0 : SystemClock.elapsedRealtimeNanos();
                long cpu = android.os.Process.getElapsedCpuTime(), start = System.nanoTime();
                float[] output = enhancer.process(input);
                milliseconds[call] = elapsed(start);
                cpuMilliseconds[call] = FullStreamingBenchmark.processCpuDeltaMillis(cpu, android.os.Process.getElapsedCpuTime());
                if (telemetry != null) telemetry.recordCall(traceStart, SystemClock.elapsedRealtimeNanos(),
                        milliseconds[call], (long) cpuMilliseconds[call]);
                requireFinite(output);
                if (output.length != expectedAudioOutput(call) || written + output.length > accumulated.length)
                    throw new IOException("Chronological audio output/sample accounting mismatch at call " + call);
                System.arraycopy(output, 0, accumulated, written, output.length); written += output.length;
                put(result, "completed_audio_calls", call + 1); put(result, "untrimmed_audio_output_samples", written);
            }
            stage = prefix + ".flush";
            checkpoint(report, directory, stage); checkContinue(cancelled, deadline);
            if (telemetry != null) telemetry.mark("strict_full_gpu", prefix, "flush", THREADS, 0);
            long flushTraceStart = telemetry == null ? 0 : SystemClock.elapsedRealtimeNanos();
            long cpu = android.os.Process.getElapsedCpuTime(), start = System.nanoTime();
            float[] tail = enhancer.flush();
            double flushMs = elapsed(start), flushCpu = FullStreamingBenchmark.processCpuDeltaMillis(cpu, android.os.Process.getElapsedCpuTime());
            if (telemetry != null) telemetry.recordCall(flushTraceStart, SystemClock.elapsedRealtimeNanos(), flushMs, (long) flushCpu);
            requireFinite(tail);
            if (tail.length != 160 || written + tail.length != accumulated.length)
                throw new IOException("One-flush output accounting mismatch");
            System.arraycopy(tail, 0, accumulated, written, tail.length); written += tail.length;
            put(result, "flush_calls", 1); put(result, "flush_output_samples", tail.length);
            put(result, "untrimmed_output_samples", written); put(result, "input_samples", INPUT_SAMPLES);
            put(result, "input_zero_padding_samples", CALLS * CHUNK - INPUT_SAMPLES);
            put(result, "trimmed_output_samples", INPUT_SAMPLES); put(result, "all_output_samples_finite", true);
            put(result, "complete_sequence_verified", true);
            merge(result, FullStreamingBenchmark.timingSummary(milliseconds, flushMs, INPUT_SAMPLES));
            merge(result, FullStreamingBenchmark.processCpuSummary(cpuMilliseconds, flushCpu, INPUT_SAMPLES));
            float[] output = Arrays.copyOf(accumulated, INPUT_SAMPLES);
            if (telemetry != null) telemetry.mark("strict_full_gpu", prefix, "save_outputs", THREADS, -1);
            put(result, "saved_public_output", FullStreamingBenchmark.saveAudio(directory, prefix, output));
            put(result, "status", gpu ? "COMPLETE_PENDING_WHOLE_GATE" : "VERIFIED_CPU_REFERENCE");
            return output;
        } catch (Exception error) {
            failure = error;
            put(result, "status", stopStatus(error)); put(result, "failure_stage", stage); recordError(result, error);
            throw error;
        } catch (LinkageError error) {
            failure = new IOException("Native runtime failure", error);
            put(result, "status", "ERROR"); put(result, "failure_stage", stage); recordError(result, error);
            throw failure;
        } finally {
            if (telemetry != null) telemetry.mark("strict_full_gpu", prefix, "cleanup", THREADS, -1);
            Exception cleanupFailure = null;
            if (backend != null) {
                Map<String, Object> configured = backend.diagnostics();
                put(result, "backend_configuration", new JSONObject(configured));
                boolean fallbackDisabled = configured.containsKey("qnn_provider_options")
                        && Boolean.FALSE.equals(configured.get("cpu_fallback_allowed"));
                put(result, "cpu_ep_fallback_disabled", fallbackDisabled);
                put(report, "cpu_ep_fallback_disabled", fallbackDisabled);
            }
            if (enhancer != null) {
                if (!profileEnded) try {
                    Path partial = enhancer.endProfiling();
                    put(result, "failed_warmup_profile_file", partial.getFileName().toString());
                    put(result, "failed_warmup_profile_summary", FullStreamingBenchmark.profileSummary(
                            new String(boundedFile(partial, MAX_ASSET_BYTES), StandardCharsets.UTF_8)));
                } catch (Exception error) { put(result, "failed_warmup_profile_error", error.toString()); }
                try { enhancer.close(); }
                catch (Exception error) { put(result, "enhancer_close_error", error.toString()); cleanupFailure = error; }
            }
            try {
                if (logs != null) {
                    try { put(result, "loaded_backend_evidence", logs.finish()); }
                    catch (Exception error) {
                        put(result, "backend_log_error", error.toString());
                        if (cleanupFailure == null) cleanupFailure = error; else cleanupFailure.addSuppressed(error);
                    }
                    finally { logs.close(); }
                }
            } finally {
                if (backend != null) try { backend.close(); }
                catch (Exception error) {
                    put(result, "backend_close_error", error.toString());
                    if (cleanupFailure == null) cleanupFailure = error; else cleanupFailure.addSuppressed(error);
                }
                try { checkpoint(report, directory, stage); }
                catch (Exception error) {
                    put(result, "checkpoint_error", error.toString());
                    if (cleanupFailure == null) cleanupFailure = error; else cleanupFailure.addSuppressed(error);
                }
            }
            if (cleanupFailure != null) {
                if (failure == null) throw cleanupFailure;
                failure.addSuppressed(cleanupFailure);
            }
        }
    }

    static boolean strictProviderPass(JSONObject summary, JSONObject verification) {
        return summary.optInt("model_run_events", -1) == WARMUP
                && summary.optInt("qnn_kernel_events", 0) > 0 && summary.optInt("cpu_kernel_events", -1) == 0
                && summary.optInt("other_provider_kernel_events", -1) == 0 && summary.optInt("missing_provider_kernel_events", -1) == 0
                && verification.optBoolean("qnn_execution_verified", false)
                && verification.optBoolean("qnn_only_kernel_execution_verified", false)
                && verification.optBoolean("qnn_kernel_counts_match_expected_calls", false)
                && verification.optInt("expected_calls", -1) == WARMUP
                && !verification.optBoolean("cpu_fallback_allowed", true);
    }

    static boolean acceptsWholeRun(JSONObject cpu, JSONObject gpu, JSONObject comparison) {
        JSONObject identity = gpu.optJSONObject("loaded_backend_evidence");
        JSONObject summary = gpu.optJSONObject("warmup_provider_evidence"), verification = gpu.optJSONObject("backend_configuration");
        return "VERIFIED_CPU_REFERENCE".equals(cpu.optString("status"))
                && completeAccounting(cpu) && completeAccounting(gpu)
                && gpu.optBoolean("cpu_ep_fallback_disabled", false) && gpu.optBoolean("warmup_provider_verified", false)
                && summary != null && verification != null && strictProviderPass(summary, verification)
                && identity != null && identity.optBoolean("gpu_backend_identity_verified", false)
                && identity.optBoolean("capture_complete_for_identity", false)
                && comparison.optBoolean("fixed_numerical_gate_passed", false)
                && GpuResourceProbe.wholeNumericalPass(comparison.optDouble("max_abs", Double.NaN),
                        comparison.optDouble("normalized_rmse", Double.NaN));
    }

    static boolean completeAccounting(JSONObject value) {
        return value.optInt("completed_audio_calls", -1) == CALLS && value.optInt("flush_calls", -1) == 1
                && value.optInt("input_samples", -1) == INPUT_SAMPLES
                && value.optInt("trimmed_output_samples", -1) == INPUT_SAMPLES
                && value.optInt("untrimmed_output_samples", -1) == CALLS * CHUNK
                && value.optInt("untrimmed_audio_output_samples", -1) == INPUT_SAMPLES
                && value.optInt("input_zero_padding_samples", -1) == 160
                && value.optInt("flush_output_samples", -1) == 160
                && value.optBoolean("all_streaming_state_reset_after_warmup", false)
                && value.optBoolean("all_output_samples_finite", false) && value.optBoolean("complete_sequence_verified", false);
    }

    static int expectedAudioOutput(int call) {
        if (call < 0 || call >= CALLS) throw new IllegalArgumentException("Audio call index outside fixed public sequence");
        return call == 0 ? 640 : CHUNK;
    }

    static void validateExportContract(JSONObject validation, JSONObject candidate, JSONObject baseline, String modelSha) throws Exception {
        if (!modelSha.matches("[a-f0-9]{64}") || !modelSha.equals(candidate.getString("model_sha256")))
            throw new IOException("Candidate graph/manifest hash contract differs");
        for (JSONObject manifest : new JSONObject[]{candidate, baseline}) {
            if (manifest.getInt("format_version") != 1 || manifest.getInt("sample_rate") != 8000
                    || manifest.getInt("fft_size") != 320 || manifest.getInt("hop_size") != 160
                    || manifest.getInt("frames_per_call") != 5 || manifest.getInt("chunk_size") != CHUNK
                    || manifest.getInt("exit_layers") != 3 || manifest.getInt("look_ahead_frames") != 0
                    || !"nvidia/Real-time_RE-USE".equals(manifest.getString("source_repository"))
                    || !REVISION.equals(manifest.getString("source_revision"))
                    || !CHECKPOINT_SHA.equals(manifest.getString("source_checkpoint_sha256")))
                throw new IOException("Model settings or NVIDIA source provenance differs from pinned FP32 contract");
        }
        if (!BASELINE_SHA.equals(baseline.getString("model_sha256")) || candidate.getJSONArray("custom_ops").length() != 0
                || !"parallel".equals(candidate.getString("scan_mode")))
            throw new IOException("Full GPU candidate must be standard portable FP32; original F950 must remain the CPU reference");
        JSONArray actualStates = candidate.getJSONArray("states"), referenceStates = baseline.getJSONArray("states");
        if (actualStates.length() != 21 || referenceStates.length() != 21) throw new IOException("Exactly 21 recurrent states are required");
        for (int i = 0; i < 21; i++) {
            JSONObject actual = actualStates.getJSONObject(i), reference = referenceStates.getJSONObject(i);
            String expected = String.format(Locale.ROOT, "state_%03d", i);
            if (!expected.equals(actual.getString("input")) || !expected.equals(reference.getString("input"))
                    || !("next_" + expected).equals(actual.getString("output")) || !("next_" + expected).equals(reference.getString("output"))
                    || !"float32".equals(actual.getString("dtype")) || !"float32".equals(reference.getString("dtype"))
                    || !actual.getJSONArray("shape").toString().equals(reference.getJSONArray("shape").toString()))
                throw new IOException("Candidate recurrent-state ABI differs at index " + i);
            JSONArray shape = actual.getJSONArray("shape");
            if (shape.length() < 1 || shape.length() > 6) throw new IOException("Invalid fixed recurrent-state rank");
            for (int axis = 0; axis < shape.length(); axis++) if (shape.getLong(axis) < 1) throw new IOException("Recurrent states must be static");
        }
        if (validation.getInt("schema_version") != 1 || !validation.getBoolean("host_validation_passed")
                || validation.getInt("custom_node_count") != 0 || !validation.getBoolean("standard_domain_only")
                || !validation.getBoolean("static_float32_io") || !validation.getBoolean("state_abi_matches_baseline")
                || validation.getBoolean("quantization_present") || !BASELINE_SHA.equals(validation.getString("source_graph_sha256"))
                || !CHECKPOINT_SHA.equals(validation.getString("source_checkpoint_sha256"))
                || !modelSha.equals(validation.getString("model_sha256"))
                || validation.getInt("input_samples") != INPUT_SAMPLES || validation.getInt("audio_calls") != CALLS
                || validation.getInt("flush_calls") != 1)
            throw new IOException("Candidate host validation is missing, failed or belongs to another graph/protocol");
        JSONObject wave = validation.getJSONObject("waveform_comparison");
        if (wave.getDouble("max_abs_limit") != GpuResourceProbe.WHOLE_MAX_ABS
                || wave.getDouble("normalized_rmse_limit") != GpuResourceProbe.WHOLE_NORMALIZED_RMSE
                || !wave.getBoolean("passed") || !GpuResourceProbe.wholeNumericalPass(wave.getDouble("max_abs"), wave.getDouble("normalized_rmse")))
            throw new IOException("Host waveform validation did not satisfy the unchanged fixed FP32 limits");
    }

    static void validateInputContract(JSONObject full) throws Exception {
        JSONObject input = full.getJSONObject("input");
        if (full.getInt("calls") != CALLS || full.getInt("frames_per_call") != 5
                || !"public-input-8k.f32".equals(input.getString("file")) || !INPUT_SHA.equals(input.getString("sha256"))
                || input.getInt("sample_rate") != 8000 || input.getJSONArray("shape").length() != 1
                || input.getJSONArray("shape").getInt(0) != INPUT_SAMPLES)
            throw new IOException("Public clip protocol differs from pinned chronological 65-call fixture");
    }

    static void checkContinue(BooleanSupplier cancelled, long deadline) {
        if (cancelled.getAsBoolean() || Thread.currentThread().isInterrupted()) throw new CancellationException("Full neural GPU probe cancelled between synchronous calls");
        if (System.nanoTime() > deadline) throw new BudgetExceeded();
    }

    private static final class BudgetExceeded extends RuntimeException {
        BudgetExceeded() { super("Full neural GPU cooperative wall budget reached; native calls cannot be interrupted"); }
    }

    private static String stopStatus(Throwable error) {
        return error instanceof CancellationException ? "CANCELLED" : error instanceof BudgetExceeded ? "INCOMPLETE_WALL_BUDGET" : "ERROR";
    }

    private static Path captureDirectory(Context context) throws IOException {
        Path files = context.getFilesDir().toPath(); Files.createDirectories(files); files = files.toRealPath();
        Path captures = files.resolve("captures"); Files.createDirectories(captures);
        if (Files.isSymbolicLink(captures) || !files.equals(captures.toRealPath().getParent()))
            throw new IOException("Capture root must remain inside app-private files");
        String stamp = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.ROOT).format(new Date());
        Path directory = captures.resolve("full-gpu-" + stamp + "-" + UUID.randomUUID().toString().substring(0, 8));
        Files.createDirectory(directory); return directory.toRealPath();
    }

    private static void validateAssetSpec(JSONObject spec, String expected) throws Exception {
        for (String[] entry : new String[][]{{"model", "model.onnx"}, {"manifest", "manifest.json"}, {"license", "LICENSE-NVIDIA.txt"}})
            if (!(expected + "/" + entry[1]).equals(spec.getString(entry[0]))) throw new IOException("Unexpected packaged full GPU asset path");
        if ("full-gpu-standard".equals(expected) && !"full-gpu-standard/validation.json".equals(spec.getString("validation")))
            throw new IOException("Unexpected packaged GPU validation path");
        if (!LICENSE_SHA.equals(spec.getString("license_sha256"))) throw new IOException("NVIDIA license identity differs");
    }

    private static ModelFiles copyModel(Context context, Path directory, String name, JSONObject spec, byte[] metadata) throws Exception {
        Path target = directory.resolve(name); Files.createDirectory(target);
        atomicWrite(target.resolve("model.onnx"), checkedAsset(context, spec, "model"));
        atomicWrite(target.resolve("manifest.json"), metadata);
        atomicWrite(target.resolve("LICENSE-NVIDIA.txt"), checkedAsset(context, spec, "license"));
        return new ModelFiles(target.resolve("model.onnx"), target.resolve("manifest.json"));
    }

    private static byte[] checkedAsset(Context context, JSONObject spec, String key) throws Exception {
        return checkedAsset(context, spec.getString(key), spec.getString(key + "_sha256"));
    }

    private static byte[] checkedAsset(Context context, String path, String hash) throws Exception {
        byte[] bytes = asset(context, path);
        if (!hash.matches("[a-f0-9]{64}") || !hash.equals(sha256(bytes))) throw new IOException("Packaged GPU asset hash mismatch: " + path);
        return bytes;
    }

    private static byte[] asset(Context context, String path) throws IOException {
        if (!GpuResourceProbe.safeAssetPath(path)) throw new IOException("Unsafe GPU fixture asset path");
        try (InputStream input = context.getAssets().open("qnn/" + path); ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192]; int count;
            while ((count = input.read(buffer)) != -1) {
                if ((long) output.size() + count > MAX_ASSET_BYTES) throw new IOException("Packaged GPU fixture exceeds 64 MiB");
                output.write(buffer, 0, count);
            }
            return output.toByteArray();
        }
    }

    private static byte[] boundedFile(Path path, long limit) throws IOException {
        if (!Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS) || Files.size(path) > limit)
            throw new IOException("Diagnostic file is missing or exceeds its bounded size");
        return Files.readAllBytes(path);
    }

    private static void inventoryDumps(JSONObject report, Path directory) throws Exception {
        JSONArray inventory = new JSONArray(); put(report, "private_graph_diagnostics", inventory);
        put(report, "private_graph_scope", "QNN pre-partition graph, compiled graph and operator trace remain app-private and are not shared by default; may contain model structure or weights.");
        if (!Files.isDirectory(directory, LinkOption.NOFOLLOW_LINKS)) return;
        long hashedBytes = 0;
        try (Stream<Path> files = Files.walk(directory, 2)) {
            Path[] paths = files.filter(path -> Files.isRegularFile(path, LinkOption.NOFOLLOW_LINKS)).limit(129).toArray(Path[]::new);
            for (int i = 0; i < Math.min(paths.length, 128); i++) {
                Path path = paths[i]; JSONObject item = new JSONObject();
                put(item, "file", directory.getFileName() + "/" + directory.relativize(path));
                long size = Files.size(path); put(item, "bytes", size);
                if (size <= MAX_ASSET_BYTES - hashedBytes) {
                    put(item, "sha256", sha256(boundedFile(path, MAX_ASSET_BYTES)));
                    hashedBytes += size;
                } else put(item, "hash_omitted", "File exceeds remaining 64 MiB total diagnostic hashing bound");
                inventory.put(item);
            }
            put(report, "private_graph_inventory_truncated", paths.length > 128);
        }
    }

    private static JSONObject json(byte[] bytes) throws Exception { return new JSONObject(new String(bytes, StandardCharsets.UTF_8)); }
    private static void requireFinite(float[] values) throws IOException {
        if (values == null || values.length == 0) throw new IOException("Streaming output must be nonempty");
        for (float value : values) if (!Float.isFinite(value)) throw new IOException("Nonfinite streaming sample");
    }
    private static String sha256(byte[] bytes) throws Exception {
        StringBuilder result = new StringBuilder();
        for (byte value : MessageDigest.getInstance("SHA-256").digest(bytes)) result.append(String.format(Locale.ROOT, "%02x", value & 255));
        return result.toString();
    }
    private static void recordError(JSONObject target, Throwable error) {
        put(target, "error", error.toString()); put(target, "error_details", EnhancementSession.boundedErrorDetails(error));
    }
    private static void merge(JSONObject target, JSONObject values) throws Exception {
        java.util.Iterator<String> keys = values.keys(); while (keys.hasNext()) { String key = keys.next(); put(target, key, values.get(key)); }
    }
    private static void checkpoint(JSONObject report, Path directory, String stage) throws Exception {
        put(report, "last_stage", stage);
        byte[] bytes = (report.toString(2) + "\n").getBytes(StandardCharsets.UTF_8);
        if (bytes.length > MAX_REPORT_BYTES) throw new IOException("Full GPU report exceeds 8 MiB");
        atomicWrite(directory.resolve("full-gpu-probe.json"), bytes);
        atomicWrite(directory.resolve("resource-runs.json"), bytes);
    }
    private static void atomicWrite(Path path, byte[] bytes) throws IOException {
        Path temporary = path.resolveSibling(path.getFileName() + ".tmp"); Files.write(temporary, bytes);
        try { Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE); }
        catch (AtomicMoveNotSupportedException error) { Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING); }
    }
    private static double elapsed(long began) { return (System.nanoTime() - began) / 1e6; }
    private static void put(JSONObject object, String key, Object value) {
        try { object.put(key, value); }
        catch (org.json.JSONException error) { throw new IllegalArgumentException("Cannot serialize full GPU diagnostic " + key, error); }
    }
}
