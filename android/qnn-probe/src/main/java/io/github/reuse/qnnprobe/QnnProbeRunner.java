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

    Result run(Progress progress) {
        JSONObject report = new JSONObject();
        Path directory = null;
        OrtEnvironment environment = null;
        boolean registered = false, environmentChanged = false;
        String registrationName = QNN_EP;
        String originalAdspPath = null;
        String stage = "diagnostics.initialize";
        long started = System.nanoTime();
        try {
            put(report, "schema_version", 1);
            put(report, "source", "qnn_htp_isolated_graph_probe");
            put(report, "app_version", BuildConfig.VERSION_NAME);
            put(report, "timestamp_utc", utcTimestamp());
            put(report, "microphone_started", false);
            put(report, "audio_captured", false);
            put(report, "full_reuse_acceleration_verified", false);
            put(report, "audio_quality_verified", false);
            put(report, "qnn_htp_test_verified", false);
            put(report, "reuse_projection_htp_verified", false);
            put(report, "scope", "Synthetic graph and one isolated RE-USE projection only; no microphone, full-model throughput, or audio-quality claim");
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
            String stamp = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.ROOT).format(new Date());
            directory = context.getFilesDir().toPath().resolve("captures")
                    .resolve(stamp + "-" + UUID.randomUUID().toString().substring(0, 8));
            Files.createDirectories(directory);

            stage = "assets.manifest";
            JSONObject manifest = new JSONObject(new String(asset("manifest.json"), StandardCharsets.UTF_8));
            if (manifest.getInt("schema_version") != 1) throw new IOException("Unsupported probe fixture manifest schema");
            JSONArray cases = manifest.getJSONArray("cases");
            if (cases.length() == 0 || cases.length() > 8) throw new IOException("Invalid number of probe fixtures");
            put(report, "fixture_manifest", manifest);

            stage = "runtime.libraries";
            Path nativeDir = new File(context.getApplicationInfo().nativeLibraryDir).toPath();
            Path plugin = nativeDir.resolve("libonnxruntime_providers_qnn.so");
            Path backend = nativeDir.resolve("libQnnHtp.so");
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

            stage = "runtime.ort";
            progress.update("Loading ONNX Runtime and the QNN plugin…");
            checkpoint(report, directory, stage);
            environment = OrtEnvironment.getEnvironment();
            environment.setTelemetry(false);
            put(report, "ort_telemetry_enabled", false);
            put(report, "ort_version", environment.getVersion());
            stage = "runtime.register_qnn_plugin";
            checkpoint(report, directory, stage);
            long registrationStart = System.nanoTime();
            environment.registerExecutionProviderLibrary(registrationName, plugin.toString());
            registered = true;
            put(report, "plugin_registration_ms", millisSince(registrationStart));

            stage = "runtime.enumerate_devices";
            checkpoint(report, directory, stage);
            JSONArray devices = new JSONArray();
            OrtEpDevice selected = null;
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

            JSONArray outcomes = new JSONArray();
            put(report, "cases", outcomes);
            boolean allVerified = true;
            for (int i = 0; i < cases.length(); i++) {
                JSONObject fixture = cases.getJSONObject(i);
                progress.update("Testing " + fixture.getString("title") + "…");
                JSONObject outcome = new JSONObject();
                outcomes.put(outcome);
                runCase(environment, selected, backend, directory, fixture, outcome, report);
                boolean verified = outcome.optBoolean("htp_execution_verified", false);
                allVerified &= verified;
                if ("synthetic_matmul_u8".equals(fixture.getString("id")))
                    put(report, "synthetic_graph_htp_verified", verified);
                if ("reuse_in_proj_u8".equals(fixture.getString("id")))
                    put(report, "reuse_projection_htp_verified", verified);
            }
            put(report, "qnn_htp_test_verified", allVerified);
            put(report, "status", allVerified ? "VERIFIED_ISOLATED_GRAPHS" : "TEST_FAILED");
            put(report, "verification_scope", "QNN NPU device selected, HTP backend requested, CPU EP fallback disabled, correct fixture outputs, and ORT QNN kernel events observed; no hardware-counter trace");
            stage = "complete";
        } catch (Exception | LinkageError error) {
            put(report, "status", "ERROR");
            put(report, "failure_stage", stage);
            put(report, "error", error.toString());
            put(report, "error_details", stackTrace(error));
        } finally {
            if (registered && environment != null) {
                try { environment.unregisterExecutionProviderLibrary(registrationName); }
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
            try {
                byte[] json = (report.toString(2) + "\n").getBytes(StandardCharsets.UTF_8);
                for (String name : new String[]{"stats.json", "phone-capture.json", "qnn-probe.json"})
                    atomicWrite(directory.resolve(name), json);
            } catch (Exception error) {
                put(report, "diagnostics_write_error", stackTrace(error));
            }
        }
        return new Result(directory == null ? null : directory.toFile(), report);
    }

    private void runCase(OrtEnvironment environment, OrtEpDevice selected, Path backend,
                         Path directory, JSONObject fixture, JSONObject result, JSONObject report) {
        String stage = "fixture.validate";
        long began = System.nanoTime();
        try {
            String id = fixture.getString("id");
            if (!id.matches("[a-zA-Z0-9_-]{1,80}")) throw new IOException("Invalid fixture ID");
            put(result, "id", id); put(result, "title", fixture.getString("title"));
            put(result, "htp_execution_verified", false);
            put(result, "cpu_ep_fallback_disabled", true);
            byte[] model = checkedAsset(fixture.getString("model"), fixture.getString("model_sha256"));
            put(result, "model_sha256", fixture.getString("model_sha256"));
            JSONObject input = fixture.getJSONObject("input"), output = fixture.getJSONObject("output");
            long[] inputShape = shape(input.getJSONArray("shape")), outputShape = shape(output.getJSONArray("shape"));
            float[] inputValues = ProbeVerification.decodeFloat32(
                    checkedAsset(input.getString("file"), input.getString("sha256")), elements(inputShape));
            float[] expected = ProbeVerification.decodeFloat32(
                    checkedAsset(output.getString("file"), output.getString("sha256")), elements(outputShape));
            double tolerance = fixture.getDouble("max_abs_error");
            if (!Double.isFinite(tolerance) || tolerance < 0) throw new IOException("Invalid maximum absolute error");
            put(result, "absolute_tolerance", tolerance);
            put(result, "relative_tolerance", 0);
            put(result, "input_shape", new JSONArray(inputShape));
            put(result, "output_shape", new JSONArray(outputShape));
            Map<String, String> providerOptions = new LinkedHashMap<>();
            providerOptions.put("backend_path", backend.toString());
            providerOptions.put("htp_performance_mode", "balanced");
            // The probe intentionally requires the graph's boundary Q/DQ operations on QNN too.
            providerOptions.put("offload_graph_io_quantization", "0");
            put(result, "qnn_provider_options", new JSONObject(providerOptions));
            Path profilePrefix = directory.resolve(id + "-ort");
            stage = "session.configure";
            checkpoint(report, directory, id + "." + stage);
            try (OrtSession.SessionOptions options = new OrtSession.SessionOptions()) {
                options.setIntraOpNumThreads(1);
                options.setInterOpNumThreads(1);
                options.addConfigEntry("session.disable_cpu_ep_fallback", "1");
                options.enableProfiling(profilePrefix.toString());
                options.addExecutionProvider(Collections.singletonList(selected), providerOptions);
                stage = "session.create_compile";
                checkpoint(report, directory, id + "." + stage);
                long compilationBegan = System.nanoTime();
                try (OrtSession session = environment.createSession(model, options)) {
                    put(result, "session_create_compile_ms", millisSince(compilationBegan));
                    stage = "session.io_contract";
                    checkpoint(report, directory, id + "." + stage);
                    String inputName = input.getString("name"), outputName = output.getString("name");
                    if (session.getNumInputs() != 1 || session.getNumOutputs() != 1
                            || !session.getInputNames().contains(inputName) || !session.getOutputNames().contains(outputName))
                        throw new IOException("Graph IO does not match the fixture manifest");
                    validateShape(session.getInputInfo().get(inputName).getInfo(), inputShape);
                    validateShape(session.getOutputInfo().get(outputName).getInfo(), outputShape);
                    try (OnnxTensor tensor = OnnxTensor.createTensor(environment, FloatBuffer.wrap(inputValues), inputShape)) {
                        Map<String, OnnxTensor> inputs = Collections.singletonMap(inputName, tensor);
                        stage = "session.warmup_execute";
                        checkpoint(report, directory, id + "." + stage);
                        long warmupBegan = System.nanoTime();
                        ProbeVerification.Comparison warmup = executeAndCompare(session, inputs, outputName, expected, tolerance);
                        put(result, "warmup_wall_ms", millisSince(warmupBegan));
                        put(result, "warmup_correct", warmup.passed);
                        recordComparison(result, warmup);
                        if (!warmup.passed) throw new IOException("NPU warm-up output differs from the CPU QDQ reference");
                        stage = "session.timed_execute";
                        checkpoint(report, directory, id + "." + stage);
                        double[] timings = new double[TIMED_CALLS];
                        ProbeVerification.Comparison worst = warmup;
                        for (int i = 0; i < TIMED_CALLS; i++) {
                            long callBegan = System.nanoTime();
                            float[] actual = execute(session, inputs, outputName);
                            timings[i] = millisSince(callBegan);
                            ProbeVerification.Comparison comparison = ProbeVerification.compare(actual, expected, tolerance, 0);
                            if (comparison.maximumAbsoluteError >= worst.maximumAbsoluteError) worst = comparison;
                            if (!comparison.passed) {
                                recordComparison(result, comparison);
                                put(result, "failed_timed_call", i + 1);
                                throw new IOException("NPU output differs from the CPU QDQ reference on timed call " + (i + 1));
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
                        put(result, "timing_scope", "Synchronous ORT run plus output-buffer readback; excludes session creation, warm-up, fixture parsing, and numerical comparison; includes ORT profiling overhead; isolated layer only");
                        put(result, "correctness_passed", true);
                        recordComparison(result, worst);
                        put(result, "reference_elements", expected.length);
                        stage = "session.verify_profile";
                        checkpoint(report, directory, id + "." + stage);
                        Path profile = new File(session.endProfiling()).toPath();
                        if (!Files.isRegularFile(profile) || Files.size(profile) > MAX_ASSET_BYTES)
                            throw new IOException("ORT profiling output is missing or too large");
                        int events = ProbeVerification.countQnnKernelEvents(
                                new String(Files.readAllBytes(profile), StandardCharsets.UTF_8));
                        put(result, "ort_profile_file", profile.getFileName().toString());
                        put(result, "qnn_kernel_events", events);
                        boolean verified = ProbeVerification.verified(
                                selected.getDevice().getType() == OrtHardwareDevice.OrtHardwareDeviceType.NPU,
                                true, events);
                        put(result, "htp_execution_verified", verified);
                        if (!verified) throw new IOException("No QNN provider kernel execution was found in the ORT profile");
                    }
                }
            }
            put(result, "status", "VERIFIED_ISOLATED_GRAPH");
            stage = "complete";
        } catch (Exception | LinkageError error) {
            put(result, "status", "ERROR");
            put(result, "htp_execution_verified", false);
            put(result, "failure_stage", stage);
            put(result, "error", error.toString());
            put(result, "error_details", stackTrace(error));
            if ("session.verify_profile".equals(stage)) put(result, "profiling_error", error.toString());
        }
        put(result, "case_total_wall_ms", millisSince(began));
        put(result, "final_stage", stage);
    }

    private static ProbeVerification.Comparison executeAndCompare(OrtSession session,
            Map<String, OnnxTensor> inputs, String outputName, float[] expected, double tolerance) throws Exception {
        return ProbeVerification.compare(execute(session, inputs, outputName), expected, tolerance, 0);
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
