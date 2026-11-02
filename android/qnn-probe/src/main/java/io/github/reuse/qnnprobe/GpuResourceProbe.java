package io.github.reuse.qnnprobe;

import android.content.Context;
import android.util.Log;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtLoggingLevel;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import io.github.reuse.ChunkedReuseEnhancer;
import io.github.reuse.ReuseConfig;
import io.github.reuse.SessionOptionsConfigurer;
import io.github.reuse.qnn.QnnGpuBackend;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Collections;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.TimeUnit;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.json.JSONArray;
import org.json.JSONObject;

/** Direct Qualcomm GPU experiment. Uses only packaged public fixtures, never a microphone. */
final class GpuResourceProbe {
    private static final int CHUNK = 800, CALLS = 65, WARMUP = 3, TIMED_ISOLATED = 10;
    private static final int MAX_BYTES = 32 * 1024 * 1024, MAX_PROFILE_BYTES = 64 * 1024 * 1024;
    private static final long BUDGET_NS = TimeUnit.SECONDS.toNanos(45);
    static final double WHOLE_MAX_ABS = .001, WHOLE_NORMALIZED_RMSE = .005;
    private static final String LICENSE_SHA = "6d1daa9c89ec421ba80fe051ac9f0dd46010d4fa4e8f4af654eb8e62f64e2ad5";
    private static final String LOG_TAG = "REUSE_GPU_PROBE";
    // Qualcomm QnnGpuCommon.h defines GPU_QTI_AISW and QNN_BACKEND_ID_GPU = 4.
    // ID 2 is SAVER and must never qualify as a real GPU backend.
    private static final String GPU_PROVIDER_NAME = "GPU_QTI_AISW", GPU_BACKEND_ID = "4";

    private GpuResourceProbe() { }

    /** Runs on the development worker; a native call cannot be interrupted by the cooperative budget. */
    static JSONObject run(Context context, File captureDirectory, QnnProbeRunner.Progress progress) {
        JSONObject report = new JSONObject();
        Path directory = null;
        long began = System.nanoTime(), deadline = began + BUDGET_NS;
        String stage = "gpu.initialize";
        try {
            if (context == null || captureDirectory == null || progress == null)
                throw new IllegalArgumentException("Context, private capture directory and progress callback are required");
            Context application = context.getApplicationContext();
            directory = captureDirectory.toPath().toRealPath();
            Path root = application.getFilesDir().toPath().toRealPath();
            if (!directory.startsWith(root) || !Files.isDirectory(directory))
                throw new IOException("GPU probe directory must be existing app-private storage");
            put(report, "source", "direct_qnn_gpu_public_fixture_probe");
            put(report, "status", "RUNNING");
            put(report, "microphone_started", false);
            put(report, "audio_quality_verified", false);
            put(report, "gpu_backend_execution_verified", false);
            put(report, "gpu_hardware_counter_trace_collected", false);
            put(report, "scope", "Direct libQnnGpu.so compatibility, numerical and provider/backend identity tests; no NNAPI, microphone, live queue, speaker playout or perceptual-quality acceptance.");
            put(report, "budget_seconds", 45);
            put(report, "budget_scope", "Cooperative wall budget checked between synchronous calls; native compilation/inference cannot be interrupted and may exceed this budget.");
            put(report, "whole_fp32_max_abs_limit", WHOLE_MAX_ABS);
            put(report, "whole_fp32_normalized_rmse_limit", WHOLE_NORMALIZED_RMSE);
            put(report, "whole_numerical_policy", "Both fixed waveform limits must pass against a same-device FP32 CPU session, with complete finite sample accounting; this is numerical consistency, not audio-quality acceptance.");
            save(report, directory, stage);
            OrtEnvironment environment = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO);
            environment.setTelemetry(false);
            put(report, "ort_version", environment.getVersion());
            put(report, "ort_telemetry_enabled", false);
            JSONObject manifest = new JSONObject(new String(asset(application, "manifest.json"), StandardCharsets.UTF_8));
            JSONObject fixture = findHeavyFixture(manifest.getJSONArray("cases"));
            stage = "gpu.isolated_fp32_convolution";
            progress.update("Direct Adreno path · strict FP32 convolution prerequisite…");
            save(report, directory, stage);
            JSONObject isolated = isolated(application, environment, fixture, directory, deadline);
            put(report, "isolated_fp32_convolution", isolated);
            save(report, directory, stage);
            if (!isolated.optBoolean("strict_numerical_provider_passed", false)) {
                put(report, "status", "ISOLATED_PREREQUISITE_FAILED");
                put(report, "full_stream_status", "SKIPPED_ISOLATED_NUMERICAL_OR_PROVIDER_FAILURE");
            } else if (!isolated.optBoolean("gpu_backend_execution_verified", false)) {
                put(report, "status", "GPU_BACKEND_IDENTITY_UNVERIFIED");
                put(report, "full_stream_status", "SKIPPED_GPU_BACKEND_IDENTITY_UNVERIFIED");
            } else {
                checkBudget(deadline);
                JSONObject full = manifest.getJSONObject("full_stream");
                Path cache = directory.resolve("gpu-fp32-model");
                Files.createDirectories(cache);
                if (!cache.equals(cache.toRealPath())) throw new IOException("GPU model cache must not be a symbolic link");
                ModelFiles model = extractBaseline(application, manifest, cache);
                JSONObject input = full.getJSONObject("input");
                if (input.getInt("sample_rate") != 8000 || input.getJSONArray("shape").length() != 1
                        || input.getJSONArray("shape").getInt(0) != 51840 || full.getInt("calls") != CALLS
                        || full.getInt("frames_per_call") != 5)
                    throw new IOException("Unsupported public full-stream fixture protocol");
                float[] audio = ProbeVerification.decodeFloat32(checkedAsset(application,
                        input.getString("file"), input.getString("sha256")), 51840);
                put(report, "public_input_sha256", input.getString("sha256"));
                put(report, "baseline_model_sha256", full.getJSONObject("baseline").getString("model_sha256"));
                stage = "gpu.full_cpu_reference";
                progress.update("Direct Adreno path · full FP32 CPU reference…");
                JSONObject cpu = new JSONObject();
                put(report, "full_cpu_reference", cpu);
                save(report, directory, stage);
                float[] reference = completeStream(application, model, audio, directory, false, cpu, deadline, progress);
                save(report, directory, stage);
                if (reference == null) {
                    put(report, "status", "CPU_REFERENCE_INCOMPLETE");
                    put(report, "full_stream_status", "SKIPPED_CPU_REFERENCE_INCOMPLETE");
                } else {
                    stage = "gpu.full_hybrid";
                    progress.update("Direct Adreno path · full GPU + CPU FP32 graph…");
                    JSONObject gpu = new JSONObject();
                    put(report, "full_gpu_hybrid", gpu);
                    save(report, directory, stage);
                    float[] actual = completeStream(application, model, audio, directory, true, gpu, deadline, progress);
                    if (actual == null) {
                        put(report, "status", "FULL_GPU_INCOMPLETE");
                    } else {
                        JSONObject comparison = FullStreamingBenchmark.audioComparison(actual, reference);
                        boolean numeric = wholeNumericalPass(comparison.optDouble("max_abs", Double.NaN),
                                comparison.optDouble("normalized_rmse", Double.NaN));
                        put(comparison, "max_abs_limit", WHOLE_MAX_ABS);
                        put(comparison, "normalized_rmse_limit", WHOLE_NORMALIZED_RMSE);
                        put(comparison, "fixed_numerical_gate_passed", numeric);
                        put(report, "full_gpu_vs_same_device_cpu", comparison);
                        boolean verified = numeric && gpu.optBoolean("gpu_backend_execution_verified", false);
                        put(report, "gpu_backend_execution_verified", verified);
                        put(report, "status", verified ? "VERIFIED_DIRECT_GPU_FULL_PUBLIC_SEQUENCE"
                                : numeric ? "GPU_BACKEND_IDENTITY_UNVERIFIED" : "FULL_GPU_NUMERICAL_GATE_FAILED");
                        put(report, "full_stream_status", report.optString("status"));
                        if (verified) {
                            double cpuMs = cpu.getDouble("mean_call_wall_ms"), gpuMs = gpu.getDouble("mean_call_wall_ms");
                            put(report, "cpu_over_gpu_call_speedup", cpuMs / gpuMs);
                            put(report, "gpu_faster_than_cpu", gpuMs < cpuMs);
                        }
                    }
                }
            }
            stage = "gpu.complete";
        } catch (Exception | LinkageError error) {
            put(report, "status", error instanceof BudgetExceeded ? "INCOMPLETE_WALL_BUDGET" : "ERROR");
            put(report, "failure_stage", stage);
            put(report, "error", error.toString());
            put(report, "error_details", io.github.reuse.live.EnhancementSession.boundedErrorDetails(error));
        } finally {
            put(report, "total_wall_ms", elapsed(began));
            put(report, "final_stage", stage);
            if (directory != null) try { save(report, directory, stage); }
            catch (Exception error) { put(report, "diagnostics_write_error", error.toString()); }
        }
        return report;
    }

    private static JSONObject isolated(Context context, OrtEnvironment environment, JSONObject fixture,
                                        Path directory, long deadline) {
        JSONObject result = new JSONObject();
        String stage = "gpu.isolated.assets";
        try {
            JSONObject fp32 = fixture.getJSONObject("cpu_fp32"), input = fixture.getJSONObject("input"), output = fp32.getJSONObject("output");
            byte[] graph = checkedAsset(context, fp32.getString("model"), fp32.getString("model_sha256"));
            long[] inputShape = shape(input.getJSONArray("shape")), outputShape = shape(output.getJSONArray("shape"));
            float[] values = ProbeVerification.decodeFloat32(checkedAsset(context, input.getString("file"), input.getString("sha256")), elements(inputShape));
            float[] expected = ProbeVerification.decodeFloat32(checkedAsset(context, output.getString("file"), output.getString("sha256")), elements(outputShape));
            double absolute = fp32.getDouble("max_abs_error"), relative = fp32.optDouble("relative_tolerance", 0);
            put(result, "model_sha256", fp32.getString("model_sha256"));
            put(result, "absolute_tolerance", absolute); put(result, "relative_tolerance", relative);
            put(result, "cpu_ep_fallback_disabled", true);
            put(result, "gpu_backend_execution_verified", false);
            Path prefix = directory.resolve("resource-gpu-isolated-ort");
            try (BackendLogCapture logs = new BackendLogCapture(directory, "isolated");
                 QnnGpuBackend backend = new QnnGpuBackend(context, prefix, false);
                 OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.setIntraOpNumThreads(1); options.setInterOpNumThreads(1);
                options.addConfigEntry("session.intra_op.allow_spinning", "0");
                options.addConfigEntry("session.inter_op.allow_spinning", "0");
                backend.configure(environment, options);
                put(result, "backend_configuration", new JSONObject(backend.diagnostics()));
                stage = "gpu.isolated.create_compile";
                checkBudget(deadline);
                long compile = System.nanoTime();
                try (OrtSession session = environment.createSession(graph, options);
                     OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(values), inputShape)) {
                    put(result, "session_create_compile_ms", elapsed(compile));
                    if (session.getNumInputs() != 1 || session.getNumOutputs() != 1
                            || !session.getInputNames().contains(input.getString("name"))
                            || !session.getOutputNames().contains(output.getString("name")))
                        throw new IOException("Isolated FP32 graph IO contract differs");
                    validateShape(session.getInputInfo().get(input.getString("name")).getInfo(), inputShape);
                    validateShape(session.getOutputInfo().get(output.getString("name")).getInfo(), outputShape);
                    Map<String, OnnxTensor> feed = Collections.singletonMap(input.getString("name"), tensor);
                    ProbeVerification.Comparison worst = null;
                    stage = "gpu.isolated.profiled_warmup";
                    for (int i = 0; i < WARMUP; i++) {
                        checkBudget(deadline);
                        ProbeVerification.Comparison c = ProbeVerification.compare(execute(session, feed, output.getString("name")), expected, absolute, relative);
                        if (worst == null || c.maximumAbsoluteError > worst.maximumAbsoluteError) worst = c;
                        if (!c.passed) {
                            recordComparison(result, c);
                            throw new IOException("GPU isolated FP32 warm-up failed its original fixture numerical gate");
                        }
                    }
                    Path profile = new File(session.endProfiling()).toPath();
                    Map<String, Object> evidence = backend.verifyProfile(profile, WARMUP, true);
                    put(result, "warmup_provider_evidence", new JSONObject(evidence));
                    if (!Boolean.TRUE.equals(evidence.get("qnn_only_kernel_execution_verified")))
                        throw new IOException("Strict isolated probe did not establish QNN-only execution");
                    double[] times = new double[TIMED_ISOLATED];
                    stage = "gpu.isolated.unprofiled_timing";
                    for (int i = 0; i < times.length; i++) {
                        checkBudget(deadline);
                        long start = System.nanoTime();
                        float[] actual = execute(session, feed, output.getString("name"));
                        times[i] = elapsed(start);
                        ProbeVerification.Comparison c = ProbeVerification.compare(actual, expected, absolute, relative);
                        if (c.maximumAbsoluteError > worst.maximumAbsoluteError) worst = c;
                        if (!c.passed) {
                            recordComparison(result, c);
                            throw new IOException("GPU isolated FP32 timed output failed its original numerical gate");
                        }
                    }
                    put(result, "mean_call_wall_ms", Arrays.stream(times).average().orElseThrow(() -> new IOException("Missing isolated call times")));
                    put(result, "max_call_wall_ms", Arrays.stream(times).max().orElseThrow(() -> new IOException("Missing isolated call times")));
                    put(result, "timed_calls", times.length);
                    put(result, "timed_calls_profiled", false);
                    recordComparison(result, worst);
                    put(result, "strict_numerical_provider_passed", true);
                }
                JSONObject logEvidence = logs.finish();
                put(result, "loaded_backend_evidence", logEvidence);
                boolean identity = logEvidence.optBoolean("gpu_backend_identity_verified", false);
                put(result, "gpu_backend_execution_verified", identity);
                put(result, "backend_configuration", new JSONObject(backend.diagnostics()));
                put(result, "status", identity ? "VERIFIED_DIRECT_GPU_ISOLATED_FP32"
                        : "QNN_NUMERICAL_PASS_GPU_IDENTITY_UNVERIFIED");
            }
        } catch (Exception | LinkageError error) {
            put(result, "status", error instanceof BudgetExceeded ? "INCOMPLETE_WALL_BUDGET" : "ERROR");
            put(result, "strict_numerical_provider_passed", false);
            put(result, "failure_stage", stage); put(result, "error", error.toString());
            put(result, "error_details", io.github.reuse.live.EnhancementSession.boundedErrorDetails(error));
        }
        return result;
    }

    private static float[] completeStream(Context context, ModelFiles model, float[] audio, Path directory,
                                          boolean gpu, JSONObject result, long deadline,
                                          QnnProbeRunner.Progress progress) {
        String prefix = gpu ? "resource-gpu-full" : "resource-gpu-cpu-reference", stage = prefix + ".initialize";
        put(result, "status", "RUNNING");
        put(result, "gpu_backend_execution_verified", false);
        put(result, "cpu_intra_op_threads", 2);
        put(result, "timed_calls_profiled", false);
        put(result, "provider_verification_scope", "Three zero-input warm-ups only, followed by reset; measured audio and flush are unprofiled");
        try (BackendLogCapture logs = gpu ? new BackendLogCapture(directory, "full") : null;
             QnnGpuBackend backend = gpu ? new QnnGpuBackend(context, directory.resolve(prefix + "-ort"), true) : null) {
            SessionOptionsConfigurer options = backend == null
                    ? (env, config) -> config.enableProfiling(directory.resolve(prefix + "-ort").toString()) : backend;
            checkBudget(deadline);
            long compile = System.nanoTime();
            stage = prefix + ".create_compile";
            try (ChunkedReuseEnhancer enhancer = ChunkedReuseEnhancer.create(context, model.model, model.manifest,
                    new ReuseConfig(8000, 3, 2), options)) {
                put(result, "session_create_compile_ms", elapsed(compile));
                if (gpu) put(result, "backend_configuration", new JSONObject(backend.diagnostics()));
                stage = prefix + ".profiled_warmup";
                for (int i = 0; i < WARMUP; i++) { checkBudget(deadline); finite(enhancer.process(new float[CHUNK])); }
                Path profile = enhancer.endProfiling();
                JSONObject evidence = FullStreamingBenchmark.profileSummary(new String(boundedFile(profile, MAX_PROFILE_BYTES), StandardCharsets.UTF_8));
                FullStreamingBenchmark.requireWarmupCallCount(evidence);
                put(result, "warmup_provider_evidence", evidence);
                if (gpu) {
                    Map<String, Object> gpuEvidence = backend.verifyProfile(profile, WARMUP, false);
                    put(result, "backend_configuration", new JSONObject(gpuEvidence));
                    if (!Boolean.TRUE.equals(gpuEvidence.get("qnn_execution_verified"))
                            || !Boolean.TRUE.equals(gpuEvidence.get("qnn_kernel_counts_match_expected_calls")))
                        throw new IOException("Full GPU warm-up has no balanced QNN provider execution");
                } else if (!cpuOnly(evidence)) throw new IOException("CPU reference warm-up did not execute entirely on CPU");
                enhancer.reset();
                stage = prefix + ".unprofiled_sequence";
                float[] accumulated = new float[CALLS * CHUNK], input = new float[CHUNK];
                double[] times = new double[CALLS];
                long cpuMs = 0;
                int written = 0;
                for (int call = 0; call < CALLS; call++) {
                    checkBudget(deadline);
                    if (call % 5 == 0) progress.update((gpu ? "Direct Adreno" : "GPU CPU reference") + " · " + call + "/" + CALLS);
                    Arrays.fill(input, 0);
                    int count = Math.min(CHUNK, audio.length - call * CHUNK);
                    if (count > 0) System.arraycopy(audio, call * CHUNK, input, 0, count);
                    long cpu = android.os.Process.getElapsedCpuTime(), start = System.nanoTime();
                    float[] output = enhancer.process(input);
                    times[call] = elapsed(start); cpuMs += Math.max(0, android.os.Process.getElapsedCpuTime() - cpu);
                    finite(output);
                    if (output.length != (call == 0 ? 640 : CHUNK) || written + output.length > accumulated.length)
                        throw new IOException("Full streaming output length mismatch");
                    System.arraycopy(output, 0, accumulated, written, output.length); written += output.length;
                    put(result, "completed_audio_calls", call + 1);
                }
                checkBudget(deadline);
                stage = prefix + ".flush";
                long cpu = android.os.Process.getElapsedCpuTime(), start = System.nanoTime();
                float[] tail = enhancer.flush();
                double flushMs = elapsed(start); cpuMs += Math.max(0, android.os.Process.getElapsedCpuTime() - cpu);
                finite(tail);
                if (tail.length != 160 || written + tail.length != accumulated.length)
                    throw new IOException("Full streaming flush count mismatch");
                System.arraycopy(tail, 0, accumulated, written, tail.length);
                float[] trimmed = Arrays.copyOf(accumulated, audio.length);
                JSONObject timing = FullStreamingBenchmark.timingSummary(times, flushMs, audio.length);
                for (java.util.Iterator<String> keys = timing.keys(); keys.hasNext();) {
                    String key = keys.next(); put(result, key, timing.get(key));
                }
                put(result, "process_cpu_time_ms", cpuMs);
                put(result, "process_cpu_time_scope", "Process-wide CPU milliseconds across all app threads during measured calls and flush; not energy or isolated kernel cycles");
                put(result, "input_samples", audio.length); put(result, "trimmed_output_samples", trimmed.length);
                put(result, "all_output_samples_finite", true);
                put(result, "saved_public_output", FullStreamingBenchmark.saveAudio(directory, prefix, trimmed));
                if (gpu) {
                    JSONObject identity = logs.finish();
                    put(result, "loaded_backend_evidence", identity);
                    put(result, "gpu_backend_execution_verified", identity.optBoolean("gpu_backend_identity_verified", false));
                }
                put(result, "status", gpu ? result.optBoolean("gpu_backend_execution_verified", false)
                        ? "VERIFIED_DIRECT_GPU_FINITE_SEQUENCE" : "GPU_BACKEND_IDENTITY_UNVERIFIED" : "VERIFIED_CPU_REFERENCE");
                return trimmed;
            }
        } catch (Exception | LinkageError error) {
            put(result, "status", error instanceof BudgetExceeded ? "INCOMPLETE_WALL_BUDGET" : "ERROR");
            put(result, "failure_stage", stage); put(result, "error", error.toString());
            put(result, "error_details", io.github.reuse.live.EnhancementSession.boundedErrorDetails(error));
            return null;
        }
    }

    static boolean wholeNumericalPass(double maximumAbsolute, double normalizedRmse) {
        return Double.isFinite(maximumAbsolute) && Double.isFinite(normalizedRmse)
                && maximumAbsolute >= 0 && normalizedRmse >= 0
                && maximumAbsolute <= WHOLE_MAX_ABS && normalizedRmse <= WHOLE_NORMALIZED_RMSE;
    }

    private static boolean cpuOnly(JSONObject evidence) {
        return evidence.optInt("cpu_kernel_events") > 0 && evidence.optInt("qnn_kernel_events") == 0
                && evidence.optInt("other_provider_kernel_events") == 0
                && evidence.optInt("missing_provider_kernel_events") == 0;
    }

    private static void recordComparison(JSONObject result, ProbeVerification.Comparison comparison) {
        put(result, "correctness_passed", comparison.passed);
        put(result, "maximum_absolute_error", comparison.maximumAbsoluteError);
        put(result, "rmse", comparison.rmse);
        put(result, "outside_tolerance_elements", comparison.outsideTolerance);
        put(result, "maximum_error_index", comparison.maximumErrorIndex);
        put(result, "maximum_error_actual", comparison.maximumErrorActual);
        put(result, "maximum_error_reference", comparison.maximumErrorReference);
    }

    static JSONObject backendLogEvidence(String logs, String marker) throws Exception {
        JSONObject result = new JSONObject();
        int start = logs.indexOf(marker + " BEGIN"), end = logs.indexOf(marker + " END", Math.max(0, start));
        boolean bounded = start >= 0 && end > start;
        String section = bounded ? logs.substring(start, end) : "";
        Pattern identity = Pattern.compile("Found valid interface,[^\\r\\n]*backend provider name:\\s*(.*?)\\s+backend id:\\s*([0-9]+)", Pattern.CASE_INSENSITIVE);
        Matcher matches = identity.matcher(section);
        String provider = null, id = null;
        int count = 0;
        while (matches.find()) { provider = matches.group(1).trim(); id = matches.group(2); count++; }
        boolean verified = bounded && GPU_PROVIDER_NAME.equals(provider) && GPU_BACKEND_ID.equals(id);
        put(result, "capture_markers_found", bounded);
        put(result, "loaded_interface_records", count);
        put(result, "last_backend_provider_name", provider == null ? JSONObject.NULL : provider);
        put(result, "last_backend_id", id == null ? JSONObject.NULL : id);
        put(result, "required_gpu_provider_name", GPU_PROVIDER_NAME);
        put(result, "required_gpu_backend_id", GPU_BACKEND_ID);
        put(result, "gpu_backend_identity_verified", verified);
        put(result, "scope", "Last native QNN Found valid interface identity between this session's own-PID log markers must match Qualcomm's GPU_QTI_AISW / backend ID 4 definitions; not hardware counters or utilization");
        return result;
    }

    static void applyCaptureCompletion(JSONObject evidence, boolean processExited, boolean readerStopped,
                                        boolean truncated, String captureError) {
        put(evidence, "capture_process_exited", processExited);
        put(evidence, "capture_reader_stopped", readerStopped);
        put(evidence, "capture_truncated", truncated);
        boolean complete = processExited && readerStopped && !truncated && captureError == null;
        put(evidence, "capture_cleanup_complete", processExited && readerStopped);
        put(evidence, "capture_complete_for_identity", complete);
        if (!complete) put(evidence, "gpu_backend_identity_verified", false);
        if (captureError != null) put(evidence, "capture_error", captureError);
    }

    static final class BackendLogCapture implements AutoCloseable {
        private final int limit;
        final String marker = "gpu-" + UUID.randomUUID();
        final Path path;
        final ByteArrayOutputStream data = new ByteArrayOutputStream();
        Process process;
        Thread reader;
        volatile String error;
        volatile boolean truncated, stopping;
        boolean processExited, readerStopped;
        boolean finished;
        BackendLogCapture(Path directory, String phase) {
            this(directory.resolve("resource-gpu-" + phase + "-backend.log"), 256 * 1024);
        }

        BackendLogCapture(Path output, int byteLimit) {
            if (byteLimit < 1 || byteLimit > 2 * 1024 * 1024)
                throw new IllegalArgumentException("GPU log capture must be bounded to at most 2 MiB");
            path = output;
            limit = byteLimit;
            try {
                process = new ProcessBuilder("logcat", "-v", "brief", "--pid", Integer.toString(android.os.Process.myPid()), "-T", "1")
                        .redirectErrorStream(true).start();
                reader = new Thread(() -> {
                    try (InputStream input = process.getInputStream();
                         java.io.OutputStream stored = Files.newOutputStream(path)) {
                        byte[] bytes = new byte[4096]; int count;
                        while ((count = input.read(bytes)) != -1) {
                            synchronized (data) {
                                int available = limit - data.size();
                                if (available <= 0) { truncated = true; process.destroy(); break; }
                                int written = Math.min(available, count);
                                data.write(bytes, 0, written);
                                stored.write(bytes, 0, written);
                                stored.flush();
                                if (count >= available) {
                                    truncated = true;
                                    process.destroy();
                                    break;
                                }
                            }
                        }
                    } catch (IOException failure) {
                        // Stopping our own logcat process can close its pipe. Earlier read failures
                        // remain errors; matching END markers are still required for identity.
                        if (!stopping) error = failure.toString();
                    }
                }, "reuse-gpu-own-pid-log");
                reader.setDaemon(true); reader.start();
                // The only deliberate wait is outside all inference timings.
                Thread.sleep(60);
                Log.i(LOG_TAG, marker + " BEGIN");
            } catch (Exception failure) {
                if (failure instanceof InterruptedException) Thread.currentThread().interrupt();
                error = failure.toString();
            }
        }

        JSONObject finish() throws Exception {
            if (!finished) {
                finished = true;
                Log.i(LOG_TAG, marker + " END");
                try { Thread.sleep(80); }
                catch (InterruptedException failure) { Thread.currentThread().interrupt(); error = failure.toString(); }
                closeProcess();
            }
            byte[] bytes;
            synchronized (data) { bytes = data.toByteArray(); }
            // Never replace a pathname while an unfinished reader still owns its file stream.
            if (readerStopped) atomicWrite(path, bytes);
            JSONObject result = backendLogEvidence(new String(bytes, StandardCharsets.UTF_8), marker);
            put(result, "log_file", path.getFileName().toString());
            put(result, "captured_bytes", bytes.length);
            put(result, "capture_limit_bytes", limit);
            put(result, "own_process_only", true);
            applyCaptureCompletion(result, processExited, readerStopped, truncated, error);
            return result;
        }

        private void closeProcess() {
            stopping = true;
            processExited = process == null;
            if (process != null) {
                process.destroy();
                try {
                    if (!process.waitFor(1, TimeUnit.SECONDS)) {
                        process.destroyForcibly();
                        process.waitFor(1, TimeUnit.SECONDS);
                    }
                    processExited = !process.isAlive();
                } catch (InterruptedException failure) {
                    process.destroyForcibly();
                    Thread.currentThread().interrupt(); error = failure.toString();
                }
            }
            if (reader != null) try { reader.join(1000); }
            catch (InterruptedException failure) { Thread.currentThread().interrupt(); error = failure.toString(); }
            readerStopped = reader == null || !reader.isAlive();
            if ((!processExited || !readerStopped) && error == null)
                error = "Own-PID log capture cleanup incomplete; GPU backend identity withheld";
        }

        @Override public void close() {
            if (!finished) {
                try { finish(); }
                catch (Exception failure) { error = failure.toString(); closeProcess(); }
            }
        }
    }

    private static ModelFiles extractBaseline(Context context, JSONObject manifest, Path target) throws Exception {
        JSONObject spec = manifest.getJSONObject("full_stream").getJSONObject("baseline");
        if (!"full-baseline/model.onnx".equals(spec.getString("model"))
                || !"full-baseline/manifest.json".equals(spec.getString("manifest")))
            throw new IOException("Unexpected baseline asset paths");
        byte[] model = checkedAsset(context, spec.getString("model"), spec.getString("model_sha256"));
        byte[] metadata = checkedAsset(context, spec.getString("manifest"), spec.getString("manifest_sha256"));
        JSONObject modelManifest = new JSONObject(new String(metadata, StandardCharsets.UTF_8));
        if (!spec.getString("model_sha256").equals(modelManifest.getString("model_sha256"))
                || modelManifest.getInt("sample_rate") != 8000 || modelManifest.getInt("exit_layers") != 3
                || modelManifest.getInt("frames_per_call") != 5 || modelManifest.getInt("look_ahead_frames") != 0
                || modelManifest.getJSONArray("states").length() != 21)
            throw new IOException("Baseline model does not match the 8 kHz / exit-3 / 21-state contract");
        for (int i = 0; i < 21; i++) if (!"float32".equals(modelManifest.getJSONArray("states").getJSONObject(i).getString("dtype")))
            throw new IOException("Baseline recurrent states must remain FP32");
        byte[] license = checkedAsset(context, "full-baseline/LICENSE-NVIDIA.txt", LICENSE_SHA);
        atomicWrite(target.resolve("model.onnx"), model);
        atomicWrite(target.resolve("manifest.json"), metadata);
        atomicWrite(target.resolve("LICENSE-NVIDIA.txt"), license);
        return new ModelFiles(target.resolve("model.onnx"), target.resolve("manifest.json"));
    }

    static JSONObject findHeavyFixture(JSONArray cases) throws Exception {
        JSONObject found = null;
        for (int i = 0; i < cases.length(); i++) {
            JSONObject fixture = cases.getJSONObject(i);
            if ("reuse_encoder_dilated_conv_u8".equals(fixture.getString("id"))) {
                if (found != null) throw new IOException("Duplicate heavy-convolution fixture");
                found = fixture;
            }
        }
        if (found == null) throw new IOException("Missing heavy-convolution FP32 prerequisite");
        return found;
    }

    private static float[] execute(OrtSession session, Map<String, OnnxTensor> inputs, String output) throws Exception {
        try (OrtSession.Result values = session.run(inputs)) {
            OnnxValue value = values.get(output).orElseThrow(() -> new IOException("Expected GPU output is missing"));
            if (!(value instanceof OnnxTensor)) throw new IOException("GPU output is not a tensor");
            FloatBuffer data = ((OnnxTensor) value).getFloatBuffer();
            float[] result = new float[data.remaining()]; data.get(result); finite(result); return result;
        }
    }

    private static void validateShape(Object info, long[] expected) throws IOException {
        if (!(info instanceof TensorInfo) || ((TensorInfo) info).type != OnnxJavaType.FLOAT
                || !Arrays.equals(((TensorInfo) info).getShape(), expected))
            throw new IOException("GPU graph tensor type/shape differs from the fixture");
    }

    private static long[] shape(JSONArray dimensions) throws Exception {
        if (dimensions.length() < 1 || dimensions.length() > 8) throw new IOException("Invalid fixture rank");
        long[] result = new long[dimensions.length()];
        for (int i = 0; i < result.length; i++) { result[i] = dimensions.getLong(i); if (result[i] < 1) throw new IOException("Invalid static fixture shape"); }
        elements(result); return result;
    }

    private static int elements(long[] shape) throws IOException {
        long count = 1;
        for (long dimension : shape) { if (dimension < 1 || dimension > MAX_BYTES / 4 || count > MAX_BYTES / 4 / dimension) throw new IOException("Fixture is too large"); count *= dimension; }
        return (int) count;
    }

    private static byte[] checkedAsset(Context context, String name, String sha) throws Exception {
        byte[] data = asset(context, name);
        if (!sha.matches("[a-fA-F0-9]{64}") || !sha.equalsIgnoreCase(sha256(data))) throw new IOException("Packaged GPU fixture hash mismatch: " + name);
        return data;
    }

    private static byte[] asset(Context context, String name) throws IOException {
        if (!safeAssetPath(name)) throw new IOException("Unsafe GPU fixture asset path");
        try (InputStream input = context.getAssets().open("qnn/" + name); ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] bytes = new byte[8192]; int count;
            while ((count = input.read(bytes)) != -1) { if ((long) output.size() + count > MAX_BYTES) throw new IOException("GPU fixture exceeds 32 MiB"); output.write(bytes, 0, count); }
            return output.toByteArray();
        }
    }

    static boolean safeAssetPath(String path) {
        if (path == null || path.isEmpty() || path.startsWith("/") || path.contains("\\") || path.contains(":")) return false;
        for (String part : path.split("/", -1)) if (part.isEmpty() || part.equals(".") || part.equals("..")) return false;
        return true;
    }

    private static byte[] boundedFile(Path file, int limit) throws IOException {
        if (!Files.isRegularFile(file) || Files.size(file) > limit) throw new IOException("GPU profile missing or exceeds its bounded size");
        return Files.readAllBytes(file);
    }

    private static String sha256(byte[] bytes) throws Exception {
        StringBuilder hash = new StringBuilder();
        for (byte value : MessageDigest.getInstance("SHA-256").digest(bytes)) hash.append(String.format(Locale.ROOT, "%02x", value & 255));
        return hash.toString();
    }

    private static void finite(float[] values) throws IOException {
        if (values == null || values.length == 0) throw new IOException("Expected nonempty streaming output");
        for (float value : values) if (!Float.isFinite(value)) throw new IOException("Nonfinite GPU output");
    }

    private static void checkBudget(long deadline) throws BudgetExceeded {
        if (System.nanoTime() > deadline) throw new BudgetExceeded();
    }
    private static final class BudgetExceeded extends IOException {
        BudgetExceeded() { super("Cooperative GPU probe budget exceeded between synchronous calls; result is incomplete"); }
    }
    private static final class ModelFiles {
        final Path model, manifest;
        ModelFiles(Path model, Path manifest) { this.model = model; this.manifest = manifest; }
    }
    private static double elapsed(long began) { return (System.nanoTime() - began) / 1e6; }
    private static void put(JSONObject object, String key, Object value) {
        try { object.put(key, value); }
        catch (org.json.JSONException error) { throw new IllegalArgumentException("Cannot serialize GPU diagnostic " + key, error); }
    }
    private static void atomicWrite(Path file, byte[] bytes) throws IOException {
        Path temp = file.resolveSibling(file.getFileName() + ".tmp"); Files.write(temp, bytes);
        try { Files.move(temp, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE); }
        catch (AtomicMoveNotSupportedException error) { Files.move(temp, file, StandardCopyOption.REPLACE_EXISTING); }
    }
    private static void save(JSONObject report, Path directory, String stage) throws Exception {
        put(report, "last_stage", stage);
        atomicWrite(directory.resolve("gpu-resource-probe.json"), (report.toString(2) + "\n").getBytes(StandardCharsets.UTF_8));
    }
}
