package io.github.reuse.qnnprobe;

import android.content.Context;
import android.os.Process;
import io.github.reuse.audio.qnn.QnnAudioSession;
import io.github.reuse.live.EnhancementSession;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import org.json.JSONArray;
import org.json.JSONObject;

/** Paced public PCM through the reusable session/FIR/queue API; never opens a microphone. */
final class PublicPcmBenchmark {
    private static final int RATE = 16000, PACKET_SAMPLES = 800, MAX_BYTES = 32 * 1024 * 1024;
    private static final int MAX_PUBLIC_8K_SAMPLES = 8000 * 15;
    private static final int MAX_PACED_16K_SAMPLES = RATE * 60;
    private static final long WAIT_NS = TimeUnit.SECONDS.toNanos(20);
    private enum Mode { CPU_FP32, CPU_QDQ, QNN_HTP }

    private PublicPcmBenchmark() { }

    /** Call on a development worker. Own checkpoints never overwrite the caller's main report. */
    static JSONObject runAll(Context context, File captureDirectory, QnnProbeRunner.Progress progress) {
        return runAll(context, captureDirectory, progress, false, 1);
    }

    /** The development caller supplies its full hybrid-execution and numerical prerequisite gate. */
    static JSONObject runAll(Context context, File captureDirectory, QnnProbeRunner.Progress progress,
                             boolean qnnNumericalPrerequisitesPassed, int repetitions) {
        JSONObject report = new JSONObject();
        Path directory = null;
        String stage = "pcm.initialize";
        long began = System.nanoTime();
        try {
            if (context == null || captureDirectory == null || progress == null)
                throw new IllegalArgumentException("Context, existing private capture directory and progress callback are required");
            validateRepetitions(repetitions);
            Context application = context.getApplicationContext();
            directory = captureDirectory.toPath().toRealPath();
            Path privateRoot = application.getFilesDir().toPath().toRealPath();
            if (!directory.startsWith(privateRoot) || !Files.isDirectory(directory))
                throw new IOException("PCM benchmark requires an existing app-private capture directory");
            put(report, "source", "paced_packaged_public_pcm");
            put(report, "microphone_started", false);
            put(report, "audio_quality_verified", false);
            put(report, "continuous_audio_track_playout_verified", false);
            put(report, "method", "Sequential CPU_FP32, CPU_QDQ, QNN_HTP through EnhancementSession; no clocks or thermal control");
            put(report, "scope", "Deterministically interpolated public clip through actual 16→8→16 kHz FIR, bounded queue, model and recording lifecycle. No microphone/source-capture or speaker-playout measurement. QNN profile evidence is startup warm-up only.");
            put(report, "packet_samples", PACKET_SAMPLES);
            put(report, "packet_period_ms", 50);
            put(report, "readiness_timeout_seconds", 20);
            put(report, "completion_timeout_seconds", 20);
            put(report, "abort_cleanup_timeout_seconds", 20);
            put(report, "cpu_intra_op_threads", 2);
            put(report, "repetitions", repetitions);
            put(report, "qnn_prerequisites_passed", qnnNumericalPrerequisitesPassed);
            put(report, "prerequisite_scope", "The caller's full-model hybrid execution and graph-specific numerical gate controls whether QNN can run; an isolated operator pass is insufficient. CPU comparisons remain available after a failed gate");
            put(report, "status", "RUNNING");
            stage = "pcm.assets.manifest";
            checkpoint(report, directory, stage);
            byte[] manifestBytes = asset(application, "manifest.json");
            JSONObject manifest = new JSONObject(new String(manifestBytes, StandardCharsets.UTF_8));
            JSONObject full = manifest.getJSONObject("full_stream"), input = full.getJSONObject("input");
            JSONArray shape = input.getJSONArray("shape");
            if (input.getInt("sample_rate") != 8000 || shape.length() != 1
                    || full.getInt("frames_per_call") != 5)
                throw new IOException("Unsupported packaged public PCM protocol");
            int sourceSamples = shape.getInt(0);
            if (sourceSamples <= 0 || sourceSamples > MAX_PUBLIC_8K_SAMPLES)
                throw new IOException("Packaged public input exceeds the bounded 15-second test");
            float[] source = ProbeVerification.decodeFloat32(checkedAsset(application,
                    input.getString("file"), input.getString("sha256")), sourceSamples);
            byte[] basePcm = interpolate16kPcm(source);
            byte[] pcm = repeatPcm(basePcm, repetitions);
            put(report, "fixture_manifest_sha256", sha256(manifestBytes));
            put(report, "public_input_8k_sha256", input.getString("sha256"));
            put(report, "public_input_16k_pcm_sha256", sha256(pcm));
            put(report, "base_public_input_16k_pcm_sha256", sha256(basePcm));
            put(report, "source_8k_samples", sourceSamples);
            put(report, "base_input_16k_samples", basePcm.length / 2);
            put(report, "input_16k_samples", pcm.length / 2);
            put(report, "input_duration_seconds", pcm.length / (2.0 * RATE));
            put(report, "repetition_scope", "Base PCM bytes are concatenated without resetting FIR or model state between repetitions; waveform seams are part of this public test signal");
            put(report, "expected_output_samples", expectedOutputSamples(pcm.length / 2));
            put(report, "interpolation", "Even samples retain the 8 kHz source; odd samples are adjacent-sample means, with the last sample held. PCM16 uses ties-to-even rounding and saturation. This test fixture conversion is not the production FIR resampler.");
            Path cache = application.getFilesDir().toPath().resolve("public-pcm-model-cache").resolve(sha256(manifestBytes));
            Files.createDirectories(cache);
            Map<Mode, ModelFiles> models = new LinkedHashMap<>();
            JSONObject assetErrors = new JSONObject();
            put(report, "model_asset_errors", assetErrors);
            try {
                models.put(Mode.CPU_FP32, extractModel(application, full.getJSONObject("baseline"), "full-baseline", cache, manifest));
            } catch (Exception error) { put(assetErrors, "baseline", stackTrace(error)); }
            try {
                ModelFiles candidate = extractModel(application, full.getJSONObject("candidate"), "full-candidate", cache, manifest);
                models.put(Mode.CPU_QDQ, candidate); models.put(Mode.QNN_HTP, candidate);
            } catch (Exception error) { put(assetErrors, "candidate", stackTrace(error)); }
            JSONObject backends = new JSONObject();
            put(report, "backends", backends);
            boolean allPassed = true, allKeptUp = true, safeToContinue = true;
            for (Mode mode : Mode.values()) {
                String key = mode.name().toLowerCase(Locale.ROOT);
                JSONObject result = new JSONObject();
                put(backends, key, result); put(result, "backend", mode.name());
                String skip = skipReason(mode == Mode.QNN_HTP, qnnNumericalPrerequisitesPassed, safeToContinue);
                if (skip != null) {
                    put(result, "status", skip);
                    put(result, "pipeline_verified", false);
                    if (mode == Mode.QNN_HTP && !qnnNumericalPrerequisitesPassed)
                        put(result, "error", "QNN was not started because the caller's full-model prerequisites did not pass");
                    allPassed = false; allKeptUp = false; continue;
                }
                ModelFiles files = models.get(mode);
                if (files == null) {
                    put(result, "status", "MODEL_ASSET_ERROR");
                    allPassed = false; allKeptUp = false; continue;
                }
                stage = "pcm." + key;
                safeToContinue = runMode(application, mode, files, pcm, directory, result, report, progress);
                allPassed &= result.optBoolean("pipeline_verified", false);
                allKeptUp &= result.optBoolean("average_realtime_criterion_met", false);
                checkpoint(report, directory, stage + ".complete");
            }
            put(report, "all_sessions_pipeline_verified", allPassed);
            put(report, "all_average_realtime_criteria_met", allKeptUp);
            put(report, "all_worker_cleanup_completed", safeToContinue);
            put(report, "status", allPassed ? "PACED_PUBLIC_PCM_COMPLETE" : "PARTIAL_RESULTS");
            stage = "pcm.complete";
        } catch (Exception | LinkageError error) {
            if (error instanceof InterruptedException) Thread.currentThread().interrupt();
            put(report, "status", "ERROR"); put(report, "failure_stage", stage);
            put(report, "error", error.toString()); put(report, "error_details", stackTrace(error));
        } finally {
            put(report, "final_stage", stage); put(report, "total_wall_ms", millisecondsSince(began));
            if (directory != null) try { save(report, directory); }
            catch (Exception error) { put(report, "diagnostics_write_error", error.toString()); }
        }
        return report;
    }

    private static boolean runMode(Context context, Mode mode, ModelFiles model, byte[] pcm,
                                    Path directory, JSONObject result, JSONObject report,
                                    QnnProbeRunner.Progress progress) throws Exception {
        String key = mode.name().toLowerCase(Locale.ROOT);
        Path output = directory.resolve("pcm-" + key.replace('_', '-') + "-" + UUID.randomUUID().toString().substring(0, 8));
        Files.createDirectory(output);
        put(result, "output_directory", output.getFileName().toString());
        put(result, "model_sha256", model.sha256);
        put(result, "pipeline_verified", false);
        put(result, "audio_quality_verified", false);
        put(result, "input_qnn_profile_verified", false);
        put(result, "status", "RUNNING");
        Callbacks callbacks = new Callbacks();
        EnhancementSession session = null;
        boolean cleanupComplete = true;
        String stage = "pcm." + key + ".create_session";
        long began = System.nanoTime();
        try {
            checkpoint(report, directory, stage);
            Runnable initializer = () -> Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);
            if (mode == Mode.QNN_HTP) {
                session = QnnAudioSession.create(context, model.model, model.manifest, output, initializer, 2, callbacks);
            } else {
                Path custom = new File(context.getApplicationInfo().nativeLibraryDir, "libreuse_ort_ops.so").toPath();
                session = new EnhancementSession(model.model.getParent(), output, custom, callbacks, initializer, 2);
            }
            stage = "pcm." + key + ".start_and_warmup";
            checkpoint(report, directory, stage);
            progress.update("Paced public PCM · " + mode.name().replace('_', ' ') + " · preparing…");
            long start = System.nanoTime();
            session.start();
            if (!awaitReadyOrTerminal(callbacks.ready, callbacks.done, WAIT_NS))
                throw new IOException("Session failed before readiness: " + callbacks.errorCode + ": " + callbacks.errorMessage);
            put(result, "start_to_ready_ms", (callbacks.readyAtNs - start) / 1e6);
            stage = "pcm." + key + ".paced_input";
            checkpoint(report, directory, stage);
            int totalSamples = pcm.length / 2, submitted = 0, packets = 0;
            double totalLatenessMs = 0, maximumLatenessMs = 0;
            while (submitted < totalSamples) {
                int count = Math.min(PACKET_SAMPLES, totalSamples - submitted);
                waitUntil(packetDeadlineNs(callbacks.readyAtNs, submitted + count), callbacks.done);
                byte[] packet = Arrays.copyOfRange(pcm, submitted * 2, (submitted + count) * 2);
                double lateness = Math.max(0, (System.nanoTime() - packetDeadlineNs(callbacks.readyAtNs, submitted + count)) / 1e6);
                if (!session.accept16kS16LE(packet, submitted)) break;
                submitted += count; packets++;
                totalLatenessMs += lateness; maximumLatenessMs = Math.max(maximumLatenessMs, lateness);
                if (packets % 20 == 0) progress.update("Paced public PCM · " + mode.name().replace('_', ' ') + " · " + submitted + "/" + totalSamples + " samples");
            }
            put(result, "submitted_input_samples", submitted); put(result, "submitted_packets", packets);
            put(result, "expected_packets", (totalSamples + PACKET_SAMPLES - 1) / PACKET_SAMPLES);
            put(result, "max_submit_lateness_ms", maximumLatenessMs);
            put(result, "mean_submit_lateness_ms", packets == 0 ? 0 : totalLatenessMs / packets);
            put(result, "feed_wall_after_ready_ms", (System.nanoTime() - callbacks.readyAtNs) / 1e6);
            put(result, "pacing_scope", "Absolute monotonic deadlines after readiness; first full packet at 50 ms, final partial packet at its actual duration. Late packets may catch up; scheduling lateness is recorded.");
            stage = "pcm." + key + ".finish_and_drain";
            checkpoint(report, directory, stage);
            session.finish();
            awaitTerminal(callbacks.done, WAIT_NS);
            if (callbacks.errorCode != null) throw new IOException("PCM session error " + callbacks.errorCode + ": " + callbacks.errorMessage);
            Map<String, Object> stats = callbacks.stats;
            if (stats == null || !accountingMatches(stats, totalSamples, callbacks.outputSamples.get()))
                throw new IOException("PCM session sample, queue, callback or drop accounting failed");
            if (Files.size(output.resolve("raw.wav")) != 44L + totalSamples * 2
                    || Files.size(output.resolve("enhanced.wav")) != 44L + expectedOutputSamples(totalSamples) * 2)
                throw new IOException("PCM recording sizes do not match session accounting");
            put(result, "pipeline_verified", true);
            put(result, "pipeline_finite_output_checks_passed", true);
            put(result, "status", "VERIFIED_PACED_PUBLIC_PCM");
        } catch (Exception | LinkageError error) {
            if (error instanceof InterruptedException) Thread.currentThread().interrupt();
            put(result, "status", "ERROR"); put(result, "failure_stage", stage);
            put(result, "error", error.toString()); put(result, "error_details", stackTrace(error));
        } finally {
            if (session != null) {
                if (callbacks.done.getCount() != 0) {
                    session.close();
                    try { awaitTerminal(callbacks.done, WAIT_NS); }
                    catch (Exception error) {
                        cleanupComplete = false;
                        put(result, "abort_cleanup_error", error.toString());
                    }
                }
                Map<String, Object> stats = callbacks.stats == null ? session.statistics() : callbacks.stats;
                put(result, "session_statistics", new JSONObject(stats));
                copyMetric(result, stats, "input_samples", "processed_input_samples", "output_samples",
                        "dropped_input_samples", "queue_overflows", "queued_input_samples", "max_queue_samples",
                        "queue_age_max_ms", "deadline_misses", "model_calls", "process_mean_ms", "process_p95_ms",
                        "process_max_ms", "processing_rtf", "warmup_ms", "load_ms", "qnn_hybrid_warmup_verified");
                if ("CLOSE_ERROR".equals(callbacks.errorCode)) cleanupComplete = false;
            }
            put(result, "listener_output_samples", callbacks.outputSamples.get());
            put(result, "expected_input_samples", pcm.length / 2);
            put(result, "expected_output_samples", expectedOutputSamples(pcm.length / 2));
            put(result, "worker_cleanup_completed", cleanupComplete && (session == null || callbacks.done.getCount() == 0));
            put(result, "session_error_code", callbacks.errorCode == null ? JSONObject.NULL : callbacks.errorCode);
            put(result, "average_realtime_criterion_met", result.optBoolean("pipeline_verified", false)
                    && result.optDouble("processing_rtf", Double.POSITIVE_INFINITY) < 1);
            put(result, "all_100ms_deadlines_met", result.optBoolean("pipeline_verified", false)
                    && result.optLong("deadline_misses", -1) == 0);
            put(result, "mode_wall_ms", millisecondsSince(began)); put(result, "final_stage", stage);
        }
        return cleanupComplete;
    }

    static final class Callbacks implements EnhancementSession.Listener {
        final CountDownLatch ready = new CountDownLatch(1), done = new CountDownLatch(1);
        final AtomicLong outputSamples = new AtomicLong();
        volatile long readyAtNs;
        volatile Map<String, Object> stats;
        volatile String errorCode, errorMessage;
        public void onReady() { readyAtNs = System.nanoTime(); ready.countDown(); }
        public void onAudio(byte[] pcm) {
            if ((pcm.length & 1) != 0) throw new IllegalStateException("Unexpected odd PCM16 callback length");
            outputSamples.addAndGet(pcm.length / 2);
        }
        public void onFinished(Map<String, Object> stats) { this.stats = new LinkedHashMap<>(stats); done.countDown(); }
        public void onError(String code, String message, Map<String, Object> stats) {
            errorCode = code; errorMessage = message; this.stats = new LinkedHashMap<>(stats); done.countDown();
        }
    }

    static boolean awaitReadyOrTerminal(CountDownLatch ready, CountDownLatch done, long timeoutNs) throws Exception {
        if (timeoutNs < 0) throw new IllegalArgumentException("Timeout must not be negative");
        long deadline = System.nanoTime() + timeoutNs;
        while (true) {
            if (ready.getCount() == 0) return true;
            if (done.getCount() == 0) return false;
            long remaining = deadline - System.nanoTime();
            if (remaining <= 0) throw new IOException("PCM session readiness timed out");
            ready.await(Math.min(remaining, TimeUnit.MILLISECONDS.toNanos(20)), TimeUnit.NANOSECONDS);
        }
    }

    static void awaitTerminal(CountDownLatch done, long timeoutNs) throws Exception {
        if (timeoutNs < 0) throw new IllegalArgumentException("Timeout must not be negative");
        if (!done.await(timeoutNs, TimeUnit.NANOSECONDS)) throw new IOException("PCM session completion/cleanup timed out");
    }

    private static void waitUntil(long deadline, CountDownLatch terminal) throws Exception {
        while (true) {
            if (terminal.getCount() == 0) throw new IOException("PCM session terminated while public input was still being paced");
            long remaining = deadline - System.nanoTime();
            if (remaining <= 0) return;
            TimeUnit.NANOSECONDS.sleep(Math.min(remaining, TimeUnit.MILLISECONDS.toNanos(10)));
        }
    }

    static long packetDeadlineNs(long readyNs, long samplesAfterPacket) {
        if (samplesAfterPacket < 0 || samplesAfterPacket > MAX_PACED_16K_SAMPLES)
            throw new IllegalArgumentException("Invalid paced sample count");
        return readyNs + samplesAfterPacket * 1_000_000_000L / RATE;
    }

    static byte[] interpolate16kPcm(float[] source) throws IOException {
        if (source == null || source.length == 0 || source.length > MAX_PUBLIC_8K_SAMPLES)
            throw new IOException("Invalid bounded public input");
        byte[] pcm = new byte[source.length * 4];
        for (int i = 0; i < source.length; i++) {
            float left = source[i], right = source[Math.min(i + 1, source.length - 1)];
            if (!Float.isFinite(left) || !Float.isFinite(right)) throw new IOException("Nonfinite public input");
            putPcm16(pcm, i * 4, left);
            putPcm16(pcm, i * 4 + 2, ((double) left + right) * .5);
        }
        return pcm;
    }

    private static void putPcm16(byte[] output, int offset, double value) {
        int sample = (int) Math.rint(Math.max(-1, Math.min(32767.0 / 32768, value)) * 32768);
        output[offset] = (byte) sample; output[offset + 1] = (byte) (sample >> 8);
    }

    static long expectedOutputSamples(long inputSamples) {
        if (inputSamples < 0 || inputSamples > MAX_PACED_16K_SAMPLES)
            throw new IllegalArgumentException("Invalid bounded PCM input length");
        return inputSamples == 0 ? 0 : inputSamples + (inputSamples & 1) + 188;
    }

    static byte[] repeatPcm(byte[] basePcm, int repetitions) throws IOException {
        validateRepetitions(repetitions);
        if (basePcm == null || basePcm.length == 0 || (basePcm.length & 1) != 0
                || basePcm.length / 2 > MAX_PUBLIC_8K_SAMPLES * 2)
            throw new IOException("Invalid bounded base PCM payload");
        long samples = basePcm.length / 2L * repetitions;
        if (samples > MAX_PACED_16K_SAMPLES) throw new IOException("Repeated PCM input exceeds 60 seconds");
        if (repetitions == 1) return basePcm;
        byte[] repeated = new byte[(int) samples * 2];
        for (int i = 0; i < repetitions; i++) System.arraycopy(basePcm, 0, repeated, i * basePcm.length, basePcm.length);
        return repeated;
    }

    private static void validateRepetitions(int repetitions) {
        if (repetitions < 1 || repetitions > 9) throw new IllegalArgumentException("PCM repetitions must be between 1 and 9");
    }

    static String skipReason(boolean qnnMode, boolean numericalGate, boolean priorCleanupComplete) {
        if (qnnMode && !numericalGate) return "SKIPPED_QNN_PREREQUISITE_FAILED";
        return priorCleanupComplete ? null : "SKIPPED_PREVIOUS_CLEANUP_INCOMPLETE";
    }

    static boolean accountingMatches(Map<String, Object> stats, long inputSamples, long callbackSamples) {
        return "FINISHED".equals(stats.get("state"))
                && number(stats, "input_samples") == inputSamples && number(stats, "processed_input_samples") == inputSamples
                && number(stats, "output_samples") == expectedOutputSamples(inputSamples)
                && callbackSamples == expectedOutputSamples(inputSamples)
                && number(stats, "dropped_input_samples") == 0 && number(stats, "queue_overflows") == 0
                && number(stats, "queued_input_samples") == 0 && number(stats, "sequence_gaps") == 0;
    }

    private static long number(Map<String, Object> stats, String key) {
        Object value = stats.get(key);
        if (!(value instanceof Number)) return -1;
        Number number = (Number) value;
        double numeric = number.doubleValue();
        long integer = number.longValue();
        return Double.isFinite(numeric) && numeric >= 0 && numeric == integer ? integer : -1;
    }

    private static void copyMetric(JSONObject target, Map<String, Object> stats, String... names) {
        for (String name : names) if (stats.containsKey(name)) put(target, name, stats.get(name));
    }

    private static final class ModelFiles {
        final Path model, manifest;
        final String sha256;
        ModelFiles(Path model, Path manifest, String sha256) { this.model = model; this.manifest = manifest; this.sha256 = sha256; }
    }

    private static ModelFiles extractModel(Context context, JSONObject spec, String folder, Path cache,
                                             JSONObject manifest) throws Exception {
        if (!spec.getString("model").equals(folder + "/model.onnx")
                || !spec.getString("manifest").equals(folder + "/manifest.json"))
            throw new IOException("Unexpected public model asset paths");
        byte[] model = checkedAsset(context, spec.getString("model"), spec.getString("model_sha256"));
        byte[] modelManifest = checkedAsset(context, spec.getString("manifest"), spec.getString("manifest_sha256"));
        if (!new JSONObject(new String(modelManifest, StandardCharsets.UTF_8)).getString("model_sha256")
                .equalsIgnoreCase(spec.getString("model_sha256"))) throw new IOException("Model manifest does not bind its graph hash");
        byte[] license = checkedAsset(context, folder + "/LICENSE-NVIDIA.txt", spec.getString("license_sha256"));
        byte[] originalLicense = asset(context, manifest.getString("model_derived_license"));
        if (license.length == 0 || !Arrays.equals(license, originalLicense)) throw new IOException("Public model license differs from the packaged original");
        Path target = cache.resolve(folder); Files.createDirectories(target);
        atomicWrite(target.resolve("model.onnx"), model); atomicWrite(target.resolve("manifest.json"), modelManifest);
        atomicWrite(target.resolve("LICENSE-NVIDIA.txt"), license);
        return new ModelFiles(target.resolve("model.onnx"), target.resolve("manifest.json"), sha256(model));
    }

    private static byte[] checkedAsset(Context context, String name, String expected) throws Exception {
        if (!expected.matches("[a-fA-F0-9]{64}")) throw new IOException("Invalid packaged asset SHA-256");
        byte[] data = asset(context, name);
        if (!sha256(data).equalsIgnoreCase(expected)) throw new IOException("Packaged asset SHA-256 mismatch: " + name);
        return data;
    }

    private static byte[] asset(Context context, String name) throws IOException {
        if (!name.matches("[a-zA-Z0-9][a-zA-Z0-9_.-]{0,159}(/[a-zA-Z0-9][a-zA-Z0-9_.-]{0,159})?") || name.contains(".."))
            throw new IOException("Unsafe public benchmark asset path");
        try (InputStream input = context.getAssets().open("qnn/" + name); ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[8192]; int read;
            while ((read = input.read(buffer)) != -1) {
                if ((long) output.size() + read > MAX_BYTES) throw new IOException("Public asset exceeds 32 MiB");
                output.write(buffer, 0, read);
            }
            return output.toByteArray();
        }
    }

    private static String sha256(byte[] data) throws Exception {
        StringBuilder hash = new StringBuilder(64);
        for (byte value : MessageDigest.getInstance("SHA-256").digest(data)) hash.append(String.format(Locale.ROOT, "%02x", value & 255));
        return hash.toString();
    }

    private static void atomicWrite(Path path, byte[] data) throws IOException {
        Path temporary = path.resolveSibling(path.getFileName() + ".tmp"); Files.write(temporary, data);
        try { Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE); }
        catch (AtomicMoveNotSupportedException ignored) { Files.move(temporary, path, StandardCopyOption.REPLACE_EXISTING); }
    }

    private static void checkpoint(JSONObject report, Path directory, String stage) throws Exception {
        put(report, "status", "RUNNING"); put(report, "last_stage", stage); save(report, directory);
    }
    private static void save(JSONObject report, Path directory) throws Exception {
        atomicWrite(directory.resolve("public-pcm-benchmark.json"), (report.toString(2) + "\n").getBytes(StandardCharsets.UTF_8));
    }
    private static double millisecondsSince(long began) { return (System.nanoTime() - began) / 1e6; }
    private static String stackTrace(Throwable error) {
        StringWriter text = new StringWriter(); error.printStackTrace(new PrintWriter(text)); return text.toString();
    }
    private static void put(JSONObject object, String key, Object value) {
        try { object.put(key, value); }
        catch (org.json.JSONException error) { throw new IllegalStateException("Cannot serialize PCM diagnostic " + key, error); }
    }
}
