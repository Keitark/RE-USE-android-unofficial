package io.github.reuse.qnnprobe;

import android.content.Context;
import android.os.Build;
import android.system.Os;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtEpDevice;
import ai.onnxruntime.OrtHardwareDevice;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.FloatBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Collections;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import org.json.JSONArray;
import org.json.JSONObject;

/** A bounded, user-triggered QNN HTP test. It never captures audio or runs the full RE-USE model. */
final class QnnProbeRunner {
    private static final String QNN_EP = "QNNExecutionProvider";
    private static final int TIMED_CALLS = 20;
    private static final int MAX_ASSET_BYTES = 32 * 1024 * 1024;
    private final Context context;

    QnnProbeRunner(Context context) { this.context = context.getApplicationContext(); }

    interface Progress { void update(String message); }

    static final class Result {
        final File directory;
        final JSONObject diagnostics;
        Result(File directory, JSONObject diagnostics) {
            this.directory = directory; this.diagnostics = diagnostics;
        }
    }

    private static final int WARMUP_CALLS = 5;
    private static final String METHOD = "sequential CPU_FP32→CPU_QDQ→QNN per case";
    private enum Mode { CPU_FP32, CPU_QDQ, QNN_HTP }

    Result run(Progress progress) {
        return run(progress, null);
    }

    Result run(Progress progress, File existingPrivateCaptureDirectory) {
        JSONObject report = new JSONObject();
        Path directory = null;
        OrtEnvironment environment = null;
        OrtEpDevice selected = null;
        Path backend = null;
        boolean registered = false, environmentChanged = false;
        String originalAdspPath = null;
        String stage = "diagnostics.initialize";
        long started = System.nanoTime();
        try {
            put(report, "schema_version", 2);
            put(report, "source", "qnn_htp_cpu_comparison_probe");
            put(report, "app_version", BuildConfig.VERSION_NAME);
            put(report, "timestamp_utc", utcTimestamp());
            put(report, "microphone_started", false);
            put(report, "audio_captured", false);
            android.os.PowerManager power = context.getSystemService(android.os.PowerManager.class);
            android.app.KeyguardManager keyguard = context.getSystemService(android.app.KeyguardManager.class);
            if (power != null) {
                put(report, "screen_interactive_at_start", power.isInteractive());
                put(report, "device_idle_at_start", power.isDeviceIdleMode());
                put(report, "power_save_at_start", power.isPowerSaveMode());
                put(report, "keyguard_locked_at_start", keyguard == null || keyguard.isKeyguardLocked());
                put(report, "foreground_timing_qualified", power.isInteractive() && !power.isDeviceIdleMode()
                        && keyguard != null && !keyguard.isKeyguardLocked());
            }
            put(report, "full_reuse_acceleration_verified", false);
            put(report, "audio_quality_verified", false);
            put(report, "qnn_htp_test_verified", false);
            put(report, "any_htp_graph_verified", false);
            put(report, "qnn_access_verified", false);
            put(report, "reuse_projection_htp_verified", false);
            put(report, "comparison_ready", false);
            put(report, "comparison_all_verified", false);
            put(report, "scope", "Isolated MatMul and convolution graphs only; no microphone, full-model throughput, or audio-quality claim");
            put(report, "method", METHOD);
            put(report, "comparison_limits", "Sequential measurements; test order and thermal conditions are not controlled. Same ORT version; CPU 2 intra-op/1 inter-op threads; QNN 1/1 with balanced HTP mode. Includes ORT profiling overhead.");
            put(report, "warmup_calls_per_backend", WARMUP_CALLS);
            put(report, "timed_calls_per_backend", TIMED_CALLS);
            put(report, "qnn_plugin_package_version", BuildConfig.QNN_EP_VERSION);
            put(report, "qnn_runtime_package_version", BuildConfig.QNN_RUNTIME_VERSION);
            put(report, "packaged_version_scope", "Pinned Maven dependency versions, not a query of the loaded QNN backend API");
            JSONObject device = new JSONObject();
            put(device, "manufacturer", Build.MANUFACTURER); put(device, "model", Build.MODEL);
            put(device, "device", Build.DEVICE); put(device, "android_release", Build.VERSION.RELEASE);
            put(device, "android_api", Build.VERSION.SDK_INT); put(device, "supported_abis", new JSONArray(Arrays.asList(Build.SUPPORTED_ABIS)));
            if (Build.VERSION.SDK_INT >= 31) {
                put(device, "soc_manufacturer", Build.SOC_MANUFACTURER); put(device, "soc_model", Build.SOC_MODEL);
            }
            put(report, "android_device", device);
            if (existingPrivateCaptureDirectory == null) {
                String stamp = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.ROOT).format(new Date());
                directory = context.getFilesDir().toPath().resolve("captures")
                        .resolve(stamp + "-" + UUID.randomUUID().toString().substring(0, 8));
                Files.createDirectories(directory);
            } else {
                directory = validateCaptureDirectory(context.getFilesDir().toPath().resolve("captures"),
                        existingPrivateCaptureDirectory.toPath());
            }

            stage = "assets.manifest";
            JSONObject manifest = new JSONObject(new String(asset("manifest.json"), StandardCharsets.UTF_8));
            if (manifest.getInt("schema_version") != 2) throw new IOException("Unsupported probe fixture manifest schema");
            JSONArray cases = manifest.getJSONArray("cases");
            if (cases.length() == 0 || cases.length() > 16) throw new IOException("Invalid number of probe fixtures");
            put(report, "fixture_manifest", manifest);
            JSONArray outcomes = new JSONArray();
            put(report, "cases", outcomes);

            stage = "runtime.ort";
            progress.update("Loading ONNX Runtime…");
            checkpoint(report, directory, stage);
            environment = OrtEnvironment.getEnvironment();
            environment.setTelemetry(false);
            put(report, "ort_telemetry_enabled", false);
            put(report, "ort_version", environment.getVersion());

            // QNN failures do not prevent independent CPU baseline sessions.
            JSONObject qnnInitialization = new JSONObject();
            put(report, "qnn_initialization", qnnInitialization);
            try {
                stage = "runtime.libraries";
                Path nativeDir = new File(context.getApplicationInfo().nativeLibraryDir).toPath();
                Path plugin = nativeDir.resolve("libonnxruntime_providers_qnn.so");
                backend = nativeDir.resolve("libQnnHtp.so");
                put(report, "native_library_directory", nativeDir.toString());
                JSONArray libraries = new JSONArray();
                File[] nativeFiles = nativeDir.toFile().listFiles();
                if (nativeFiles != null) {
                    Arrays.sort(nativeFiles);
                    for (File file : nativeFiles) if (file.getName().contains("Qnn") || file.getName().contains("onnxruntime"))
                        libraries.put(file.getName());
                }
                put(report, "packaged_native_libraries", libraries);
                if (!Files.isRegularFile(plugin)) throw new IOException("Packaged QNN execution-provider library is missing");
                if (!Files.isRegularFile(backend)) throw new IOException("Packaged QNN HTP backend library is missing");
                originalAdspPath = Os.getenv("ADSP_LIBRARY_PATH");
                Os.setenv("ADSP_LIBRARY_PATH", nativeDir.toString(), true);
                environmentChanged = true;
                put(report, "adsp_library_path", nativeDir.toString());
                stage = "runtime.register_qnn_plugin";
                progress.update("Checking QNN HTP access…");
                checkpoint(report, directory, stage);
                long registrationStart = System.nanoTime();
                environment.registerExecutionProviderLibrary(QNN_EP, plugin.toString());
                registered = true;
                put(report, "plugin_registration_ms", millisSince(registrationStart));

                stage = "runtime.enumerate_devices";
                checkpoint(report, directory, stage);
                JSONArray devices = new JSONArray();
                List<OrtEpDevice> available = environment.getEpDevices();
                for (OrtEpDevice candidate : available) {
                    devices.put(describeDevice(candidate));
                    if (selected == null && QNN_EP.equals(candidate.getEpName())
                            && candidate.getDevice().getType() == OrtHardwareDevice.OrtHardwareDeviceType.NPU)
                        selected = candidate;
                }
                put(report, "ort_ep_devices", devices);
                if (selected == null) throw new IOException("QNN did not expose an NPU device on this phone");
                put(report, "selected_ep_device", describeDevice(selected));
                put(qnnInitialization, "status", "NPU_DEVICE_SELECTED");
                put(qnnInitialization, "execution_verified", false);
            } catch (Exception | LinkageError error) {
                selected = null;
                put(qnnInitialization, "status", "ERROR");
                put(qnnInitialization, "failure_stage", stage);
                put(qnnInitialization, "error", error.toString());
                put(qnnInitialization, "error_details", stackTrace(error));
                put(qnnInitialization, "execution_verified", false);
                checkpoint(report, directory, "qnn_unavailable_cpu_baselines_continue");
            }

            boolean allHtpVerified = true, allCpuVerified = true, allComparisonsReady = true;
            boolean anyBackendVerified = false, anyHtpVerified = false;
            for (int i = 0; i < cases.length(); i++) {
                stage = "cases.run";
                JSONObject fixture = cases.getJSONObject(i);
                JSONObject outcome = new JSONObject();
                outcomes.put(outcome);
                runCase(environment, selected, backend, directory, fixture, outcome, report, progress);
                allHtpVerified &= outcome.optBoolean("htp_execution_verified", false);
                allCpuVerified &= outcome.optBoolean("cpu_baselines_verified", false);
                allComparisonsReady &= outcome.optBoolean("comparison_ready", false);
                anyBackendVerified |= outcome.optBoolean("any_backend_verified", false);
                anyHtpVerified |= outcome.optBoolean("htp_execution_verified", false);
                if ("synthetic_matmul_u8".equals(fixture.getString("id")))
                    put(report, "synthetic_graph_htp_verified", outcome.optBoolean("htp_execution_verified", false));
                if ("reuse_in_proj_u8".equals(fixture.getString("id")))
                    put(report, "reuse_projection_htp_verified", outcome.optBoolean("htp_execution_verified", false));
                checkpoint(report, directory, fixture.getString("id") + ".complete");
            }
            put(report, "qnn_htp_test_verified", allHtpVerified);
            put(report, "all_htp_graphs_verified", allHtpVerified);
            put(report, "any_htp_graph_verified", anyHtpVerified);
            put(report, "qnn_access_verified", anyHtpVerified);
            put(qnnInitialization, "execution_verified", anyHtpVerified);
            put(report, "cpu_baselines_verified", allCpuVerified);
            put(report, "comparison_ready", allComparisonsReady);
            put(report, "comparison_all_verified", allComparisonsReady);
            put(report, "status", allComparisonsReady ? "COMPARISON_READY"
                    : anyBackendVerified ? "PARTIAL_RESULTS" : "TEST_FAILED");
            put(report, "verification_scope", "Correct fixture outputs plus provider-specific ORT kernel events: CPU-only for baselines; selected QNN NPU with HTP backend and CPU fallback disabled for NPU tests. No hardware-counter trace.");
            stage = "complete";
        } catch (Exception | LinkageError error) {
            put(report, "status", "ERROR");
            put(report, "failure_stage", stage);
            put(report, "error", error.toString());
            put(report, "error_details", stackTrace(error));
        } finally {
            if (registered && environment != null) {
                try { environment.unregisterExecutionProviderLibrary(QNN_EP); }
                catch (Exception | LinkageError error) { put(report, "plugin_cleanup_error", stackTrace(error)); }
            }
            if (environmentChanged) {
                try {
                    if (originalAdspPath == null) Os.unsetenv("ADSP_LIBRARY_PATH");
                    else Os.setenv("ADSP_LIBRARY_PATH", originalAdspPath, true);
                } catch (Exception error) { put(report, "environment_cleanup_error", stackTrace(error)); }
            }
            put(report, "final_stage", stage);
            put(report, "total_test_wall_ms", millisSince(started));
        }
        if (directory != null) {
            try { writeDiagnostics(report, directory); }
            catch (Exception error) { put(report, "diagnostics_write_error", stackTrace(error)); }
        }
        return new Result(directory == null ? null : directory.toFile(), report);
    }

    static Path validateCaptureDirectory(Path capturesRoot, Path requested) throws IOException {
        if (capturesRoot == null || requested == null || !Files.isDirectory(capturesRoot, java.nio.file.LinkOption.NOFOLLOW_LINKS)
                || Files.isSymbolicLink(capturesRoot)
                || !Files.isDirectory(requested, java.nio.file.LinkOption.NOFOLLOW_LINKS)
                || Files.isSymbolicLink(requested))
            throw new IOException("Existing capture directory must be a real app-private directory");
        Path root = capturesRoot.toRealPath(), actual = requested.toRealPath();
        if (!root.equals(actual.getParent()))
            throw new IOException("Existing capture directory must be an immediate child of app files/captures");
        return actual;
    }

    private void runCase(OrtEnvironment environment, OrtEpDevice selected, Path backend,
                         Path directory, JSONObject fixture, JSONObject result, JSONObject report,
                         Progress progress) {
        long began = System.nanoTime();
        try {
            String id = fixture.getString("id"), title = fixture.getString("title");
            if (!id.matches("[a-zA-Z0-9_-]{1,80}")) throw new IOException("Invalid fixture ID");
            put(result, "id", id); put(result, "title", title);
            put(result, "method", METHOD);
            put(result, "comparison_ready", false);
            for (Mode mode : Mode.values()) {
                String key = mode == Mode.CPU_FP32 ? "cpu_fp32" : mode == Mode.CPU_QDQ ? "cpu_qdq" : "qnn_htp";
                JSONObject backendResult = new JSONObject();
                put(result, key, backendResult);
                put(backendResult, "backend", mode.name());
                put(backendResult, "execution_verified", false);
                progress.update(title + " · " + mode.name().replace('_', ' ') + "…");
                if (mode == Mode.QNN_HTP && selected == null) {
                    put(backendResult, "backend", mode.name());
                    put(backendResult, "status", "UNAVAILABLE");
                    put(backendResult, "execution_verified", false);
                    put(backendResult, "htp_execution_verified", false);
                    put(backendResult, "failure_stage", "runtime.qnn_initialization");
                    put(backendResult, "error", "QNN NPU initialization unavailable; see qnn_initialization diagnostics");
                    continue;
                }
                try {
                    JSONObject modelFixture = mode == Mode.CPU_FP32 ? fixture.getJSONObject("cpu_fp32") : fixture;
                    runBackend(environment, selected, backend, directory, id, fixture.getJSONObject("input"),
                            modelFixture, mode, backendResult, report);
                } catch (Exception | LinkageError error) {
                    failBackend(backendResult, "fixture.validate", error);
                }
            }
            JSONObject cpuFp32 = result.getJSONObject("cpu_fp32");
            JSONObject cpuQdq = result.getJSONObject("cpu_qdq");
            JSONObject qnn = result.getJSONObject("qnn_htp");
            // Preserve probe1's flat QNN fields for existing diagnostic consumers.
            java.util.Iterator<String> keys = qnn.keys();
            while (keys.hasNext()) {
                String key = keys.next();
                put(result, key, qnn.get(key));
            }
            boolean fp32Verified = cpuFp32.optBoolean("execution_verified", false);
            boolean qdqVerified = cpuQdq.optBoolean("execution_verified", false);
            boolean htpVerified = qnn.optBoolean("execution_verified", false);
            put(result, "htp_execution_verified", htpVerified);
            put(result, "cpu_baselines_verified", fp32Verified && qdqVerified);
            put(result, "any_backend_verified", fp32Verified || qdqVerified || htpVerified);
            put(result, "comparison_ready", fp32Verified && qdqVerified && htpVerified);
            recordSpeedup(result, "cpu_fp32_over_qnn_speedup", cpuFp32, qnn);
            recordSpeedup(result, "cpu_qdq_over_qnn_speedup", cpuQdq, qnn);
            put(result, "speedup_scope", "Isolated graph wall-call mean ratios only; sequential ordering and thermal conditions not controlled");
            put(result, "status", fp32Verified && qdqVerified && htpVerified ? "VERIFIED_COMPARISON"
                    : fp32Verified || qdqVerified || htpVerified ? "PARTIAL_RESULTS" : "TEST_FAILED");
        } catch (Exception | LinkageError error) {
            put(result, "status", "ERROR");
            put(result, "comparison_ready", false);
            put(result, "failure_stage", "case.fixture_validate");
            put(result, "error", error.toString());
            put(result, "error_details", stackTrace(error));
        }
        put(result, "case_total_wall_ms", millisSince(began));
    }

    private void runBackend(OrtEnvironment environment, OrtEpDevice selected, Path backend,
                            Path directory, String id, JSONObject input, JSONObject modelFixture,
                            Mode mode, JSONObject result, JSONObject report) {
        String stage = "fixture.validate";
        long began = System.nanoTime();
        String prefix = id + "." + mode.name().toLowerCase(Locale.ROOT);
        boolean qnn = mode == Mode.QNN_HTP;
        try {
            put(result, "backend", mode.name());
            put(result, "execution_verified", false);
            if (qnn) put(result, "htp_execution_verified", false);
            put(result, "cpu_ep_fallback_disabled", qnn);
            put(result, "cpu_intra_op_threads", qnn ? 1 : 2);
            put(result, "cpu_inter_op_threads", 1);
            put(result, "provider_selection", qnn ? "Explicit QNN NPU device with HTP backend"
                    : "No EP appended; default ORT CPU execution, verified by CPU-only kernel events");
            byte[] model = checkedAsset(modelFixture.getString("model"), modelFixture.getString("model_sha256"));
            put(result, "model_sha256", modelFixture.getString("model_sha256"));
            JSONObject output = modelFixture.getJSONObject("output");
            long[] inputShape = shape(input.getJSONArray("shape")), outputShape = shape(output.getJSONArray("shape"));
            float[] inputValues = ProbeVerification.decodeFloat32(
                    checkedAsset(input.getString("file"), input.getString("sha256")), elements(inputShape));
            float[] expected = ProbeVerification.decodeFloat32(
                    checkedAsset(output.getString("file"), output.getString("sha256")), elements(outputShape));
            double absoluteTolerance = modelFixture.getDouble("max_abs_error");
            double relativeTolerance = modelFixture.optDouble("relative_tolerance", 0);
            if (!Double.isFinite(absoluteTolerance) || absoluteTolerance < 0
                    || !Double.isFinite(relativeTolerance) || relativeTolerance < 0)
                throw new IOException("Invalid fixture comparison tolerance");
            put(result, "absolute_tolerance", absoluteTolerance);
            put(result, "relative_tolerance", relativeTolerance);
            put(result, "input_shape", new JSONArray(inputShape));
            put(result, "output_shape", new JSONArray(outputShape));
            Map<String, String> providerOptions = new LinkedHashMap<>();
            if (qnn) {
                providerOptions.put("backend_path", backend.toString());
                providerOptions.put("htp_performance_mode", "balanced");
                providerOptions.put("offload_graph_io_quantization", "0");
                providerOptions.put("enable_htp_fp16_precision", "0");
                put(result, "qnn_provider_options", new JSONObject(providerOptions));
            }
            Path profilePrefix = directory.resolve(prefix + "-ort");
            stage = "session.configure";
            checkpoint(report, directory, prefix + "." + stage);
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.setIntraOpNumThreads(qnn ? 1 : 2);
                options.setInterOpNumThreads(1);
                options.enableProfiling(profilePrefix.toString());
                if (qnn) {
                    options.addConfigEntry("session.disable_cpu_ep_fallback", "1");
                    options.addExecutionProvider(Collections.singletonList(selected), providerOptions);
                }
                stage = "session.create_compile";
                checkpoint(report, directory, prefix + "." + stage);
                long compilationBegan = System.nanoTime();
                try (OrtSession session = environment.createSession(model, options)) {
                    put(result, "session_create_compile_ms", millisSince(compilationBegan));
                    stage = "session.io_contract";
                    checkpoint(report, directory, prefix + "." + stage);
                    String inputName = input.getString("name"), outputName = output.getString("name");
                    if (session.getNumInputs() != 1 || session.getNumOutputs() != 1
                            || !session.getInputNames().contains(inputName) || !session.getOutputNames().contains(outputName))
                        throw new IOException("Graph IO does not match the fixture manifest");
                    validateShape(session.getInputInfo().get(inputName).getInfo(), inputShape);
                    validateShape(session.getOutputInfo().get(outputName).getInfo(), outputShape);
                    try (OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(inputValues), inputShape)) {
                        Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, tensor);
                        stage = "session.warmup_execute";
                        checkpoint(report, directory, prefix + "." + stage);
                        double[] warmupTimes = new double[WARMUP_CALLS];
                        ProbeVerification.Comparison worst = null;
                        for (int i = 0; i < WARMUP_CALLS; i++) {
                            long callBegan = System.nanoTime();
                            float[] actual = execute(session, inputs, outputName);
                            warmupTimes[i] = millisSince(callBegan);
                            ProbeVerification.Comparison comparison = ProbeVerification.compare(actual, expected, absoluteTolerance, relativeTolerance);
                            if (worst == null || comparison.maximumAbsoluteError >= worst.maximumAbsoluteError) worst = comparison;
                            if (!comparison.passed) {
                                recordComparison(result, comparison);
                                put(result, "failed_warmup_call", i + 1);
                                throw new IOException(mode + " warm-up output differs from its reference");
                            }
                        }
                        double warmupSum = 0;
                        for (double time : warmupTimes) warmupSum += time;
                        put(result, "warmup_calls", WARMUP_CALLS);
                        put(result, "warmup_call_wall_ms", new JSONArray(warmupTimes));
                        put(result, "warmup_first_call_wall_ms", warmupTimes[0]);
                        put(result, "warmup_total_call_wall_ms", warmupSum);
                        put(result, "warmup_wall_ms", warmupTimes[0]);
                        put(result, "warmup_correct", true);
                        stage = "session.timed_execute";
                        checkpoint(report, directory, prefix + "." + stage);
                        double[] timings = new double[TIMED_CALLS];
                        for (int i = 0; i < TIMED_CALLS; i++) {
                            long callBegan = System.nanoTime();
                            float[] actual = execute(session, inputs, outputName);
                            timings[i] = millisSince(callBegan);
                            ProbeVerification.Comparison comparison = ProbeVerification.compare(actual, expected, absoluteTolerance, relativeTolerance);
                            if (comparison.maximumAbsoluteError >= worst.maximumAbsoluteError) worst = comparison;
                            if (!comparison.passed) {
                                recordComparison(result, comparison);
                                put(result, "failed_timed_call", i + 1);
                                throw new IOException(mode + " output differs from its reference on timed call " + (i + 1));
                            }
                        }
                        put(result, "timed_calls", TIMED_CALLS);
                        put(result, "call_wall_ms", new JSONArray(timings));
                        double sum = 0;
                        for (double time : timings) sum += time;
                        double[] sorted = timings.clone(); Arrays.sort(sorted);
                        put(result, "mean_call_wall_ms", sum / timings.length);
                        put(result, "p50_call_wall_ms", (sorted[9] + sorted[10]) / 2);
                        put(result, "p95_call_wall_ms", sorted[18]);
                        put(result, "max_call_wall_ms", sorted[19]);
                        put(result, "timing_scope", "Synchronous ORT run plus output-buffer readback for every backend; excludes session creation, five warm-up calls, fixture parsing, and numerical comparison; includes ORT profiling overhead; isolated graph only");
                        recordComparison(result, worst);
                        put(result, "reference_elements", expected.length);
                        stage = "session.verify_profile";
                        checkpoint(report, directory, prefix + "." + stage);
                        Path profile = new File(session.endProfiling()).toPath();
                        if (!Files.isRegularFile(profile) || Files.size(profile) > MAX_ASSET_BYTES)
                            throw new IOException("ORT profiling output is missing or too large");
                        ProbeVerification.ExecutionEvidence evidence = ProbeVerification.executionEvidence(
                                new String(Files.readAllBytes(profile), StandardCharsets.UTF_8));
                        put(result, "ort_profile_file", profile.getFileName().toString());
                        put(result, "qnn_kernel_events", evidence.qnnEvents);
                        put(result, "cpu_kernel_events", evidence.cpuEvents);
                        put(result, "other_provider_kernel_events", evidence.otherEvents);
                        put(result, "missing_provider_kernel_events", evidence.missingProviderEvents);
                        boolean verified = qnn
                                ? ProbeVerification.verifiedHtp(selected.getDevice().getType() == OrtHardwareDevice.OrtHardwareDeviceType.NPU, true, evidence)
                                : ProbeVerification.verifiedCpu(true, evidence);
                        put(result, "execution_verified", verified);
                        if (qnn) put(result, "htp_execution_verified", verified);
                        if (!verified) throw new IOException("ORT profile does not establish " + (qnn ? "QNN-only NPU" : "CPU-only baseline") + " kernel execution");
                    }
                }
            }
            put(result, "status", "VERIFIED_ISOLATED_GRAPH");
            stage = "complete";
        } catch (Exception | LinkageError error) {
            failBackend(result, stage, error);
        }
        put(result, "backend_total_wall_ms", millisSince(began));
        put(result, "final_stage", stage);
    }

    private static void failBackend(JSONObject result, String stage, Throwable error) {
        put(result, "status", "ERROR");
        put(result, "execution_verified", false);
        if ("QNN_HTP".equals(result.optString("backend"))) put(result, "htp_execution_verified", false);
        put(result, "failure_stage", stage);
        put(result, "error", error.toString());
        put(result, "error_details", stackTrace(error));
        if ("session.verify_profile".equals(stage)) put(result, "profiling_error", error.toString());
    }

    private static void recordSpeedup(JSONObject result, String key, JSONObject cpu, JSONObject qnn) {
        Double speedup = ProbeVerification.speedup(cpu.optBoolean("execution_verified", false), cpu.optDouble("mean_call_wall_ms", Double.NaN),
                qnn.optBoolean("execution_verified", false), qnn.optDouble("mean_call_wall_ms", Double.NaN));
        if (speedup != null) put(result, key, speedup);
        else put(result, key + "_unavailable", "Both backends must have correct outputs, verified provider execution, and positive finite mean timings");
    }

    private static float[] execute(OrtSession session, Map<String, OnnxTensor> inputs,
                                   String outputName) throws Exception {
        try (OrtSession.Result outputs = session.run(inputs)) {
            OnnxValue value = outputs.get(outputName).orElseThrow(() -> new IOException("Expected graph output is missing"));
            if (!(value instanceof OnnxTensor)) throw new IOException("Graph output is not a tensor");
            FloatBuffer buffer = ((OnnxTensor) value).getFloatBuffer();
            if (buffer == null) throw new IOException("Graph output is not float32");
            float[] actual = new float[buffer.remaining()];
            buffer.get(actual);
            return actual;
        }
    }

    private static void validateShape(Object info, long[] expected) throws IOException {
        if (!(info instanceof TensorInfo) || ((TensorInfo) info).type != OnnxJavaType.FLOAT
                || !Arrays.equals(((TensorInfo) info).getShape(), expected))
            throw new IOException("Graph float32 tensor shape does not match the fixture manifest");
    }

    private byte[] checkedAsset(String filename, String expectedHash) throws Exception {
        if (!expectedHash.matches("[a-f0-9]{64}")) throw new IOException("Invalid fixture SHA-256");
        byte[] bytes = asset(filename);
        byte[] digest = MessageDigest.getInstance("SHA-256").digest(bytes);
        StringBuilder actual = new StringBuilder(64);
        for (byte value : digest) actual.append(String.format(Locale.ROOT, "%02x", value & 255));
        if (!expectedHash.contentEquals(actual)) throw new IOException("SHA-256 mismatch for " + filename);
        return bytes;
    }

    private byte[] asset(String filename) throws IOException {
        if (!filename.matches("[a-zA-Z0-9_.-]{1,160}") || filename.contains(".."))
            throw new IOException("Unsafe fixture asset filename");
        try (InputStream input = context.getAssets().open("qnn/" + filename);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] block = new byte[8192];
            int read;
            while ((read = input.read(block)) != -1) {
                if ((long) output.size() + read > MAX_ASSET_BYTES) throw new IOException("Probe fixture asset is too large");
                output.write(block, 0, read);
            }
            return output.toByteArray();
        }
    }

    private static long[] shape(JSONArray value) throws Exception {
        if (value.length() == 0 || value.length() > 8) throw new IOException("Invalid tensor rank");
        long[] shape = new long[value.length()];
        for (int i = 0; i < shape.length; i++) shape[i] = value.getLong(i);
        elements(shape);
        return shape;
    }

    private static int elements(long[] shape) throws IOException {
        long total = 1;
        for (long dimension : shape) {
            if (dimension <= 0 || dimension > MAX_ASSET_BYTES / Float.BYTES
                    || total > MAX_ASSET_BYTES / Float.BYTES / dimension)
                throw new IOException("Invalid or excessive tensor element count");
            total *= dimension;
        }
        return (int) total;
    }

    private static JSONObject describeDevice(OrtEpDevice value) throws Exception {
        JSONObject json = new JSONObject();
        put(json, "ep_name", value.getEpName()); put(json, "ep_vendor", value.getEpVendor());
        put(json, "ep_metadata", new JSONObject(value.getEpMetadata()));
        put(json, "ep_options", new JSONObject(value.getEpOptions()));
        OrtHardwareDevice device = value.getDevice();
        put(json, "device_type", device.getType().name()); put(json, "device_id", device.getDeviceId());
        put(json, "device_vendor", device.getVendor()); put(json, "device_vendor_id", device.getVendorId());
        put(json, "device_metadata", new JSONObject(device.getMetadata()));
        return json;
    }

    private static void recordComparison(JSONObject result, ProbeVerification.Comparison comparison) {
        put(result, "correctness_passed", comparison.passed);
        put(result, "maximum_absolute_error", Double.isFinite(comparison.maximumAbsoluteError)
                ? comparison.maximumAbsoluteError : "non_finite");
        put(result, "rmse", Double.isFinite(comparison.rmse) ? comparison.rmse : "non_finite");
        put(result, "outside_tolerance_elements", comparison.outsideTolerance);
        put(result, "mean_absolute_error", finiteOrUndefined(comparison.meanAbsoluteError));
        put(result, "maximum_error_index", comparison.maximumErrorIndex);
        put(result, "maximum_error_actual", finiteOrUndefined(comparison.maximumErrorActual));
        put(result, "maximum_error_reference", finiteOrUndefined(comparison.maximumErrorReference));
        put(result, "reference_rms", finiteOrUndefined(comparison.referenceRms));
        put(result, "normalized_rmse", finiteOrUndefined(comparison.normalizedRmse));
        put(result, "numerical_summary_scope", "All summary fields and maximum-error actual/reference pair are from the same call; final successful summary selects the call with greatest maximum absolute error across warm-up and timed calls");
    }

    private static Object finiteOrUndefined(double value) {
        return Double.isFinite(value) ? value : "non_finite_or_undefined";
    }

    private static void atomicWrite(Path destination, byte[] data) throws IOException {
        Path temporary = destination.resolveSibling(destination.getFileName().toString() + ".tmp");
        Files.write(temporary, data);
        Files.move(temporary, destination, StandardCopyOption.REPLACE_EXISTING);
    }

    private static void checkpoint(JSONObject report, Path directory, String stage) throws Exception {
        put(report, "status", "RUNNING");
        put(report, "last_stage", stage);
        put(report, "checkpoint_utc", utcTimestamp());
        writeDiagnostics(report, directory);
    }

    private static void writeDiagnostics(JSONObject report, Path directory) throws Exception {
        byte[] json = (report.toString(2) + "\n").getBytes(StandardCharsets.UTF_8);
        for (String name : new String[]{"stats.json", "phone-capture.json", "qnn-probe.json"})
            atomicWrite(directory.resolve(name), json);
    }

    private static double millisSince(long began) { return (System.nanoTime() - began) / 1e6; }

    private static String utcTimestamp() {
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.ROOT);
        format.setTimeZone(java.util.TimeZone.getTimeZone("UTC"));
        return format.format(new Date());
    }

    private static String stackTrace(Throwable error) {
        StringWriter text = new StringWriter();
        error.printStackTrace(new PrintWriter(text));
        return text.toString();
    }

    private static void put(JSONObject object, String key, Object value) {
        try { object.put(key, value); }
        catch (org.json.JSONException error) { throw new IllegalStateException("Cannot serialize diagnostic field " + key, error); }
    }
}
