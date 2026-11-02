package io.github.reuse.qnnprobe;

import android.content.Context;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtLoggingLevel;
import io.github.reuse.live.EnhancementSession;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CancellationException;
import java.util.concurrent.TimeUnit;
import java.util.function.BooleanSupplier;
import org.json.JSONArray;
import org.json.JSONObject;

/** Independent CPU, HTP and direct GPU comparisons using public data; never opens a microphone. */
final class ResourceComparisonRunner {
    private static final long BUDGET_NS = TimeUnit.MINUTES.toNanos(10);
    private static final int MAX_OUTPUT_SAMPLES = 65 * 800, MAX_REPORT_BYTES = 8 * 1024 * 1024;

    static final class Result {
        final File directory;
        final JSONObject diagnostics;
        Result(File directory, JSONObject diagnostics) { this.directory = directory; this.diagnostics = diagnostics; }
    }

    private ResourceComparisonRunner() { }

    static Result run(Context context, QnnProbeRunner.Progress progress, BooleanSupplier cancelled) {
        JSONObject report = new JSONObject();
        JSONArray errors = new JSONArray(), cpuTrials = new JSONArray(), htpTrials = new JSONArray();
        put(report, "errors", errors); put(report, "cpu_trials", cpuTrials); put(report, "htp_trials", htpTrials);
        Path directory = null;
        long began = System.nanoTime(), deadline = began + BUDGET_NS;
        String stage = "resource.initialize";
        try {
            if (context == null || progress == null || cancelled == null)
                throw new IllegalArgumentException("Context, progress and cancellation supplier are required");
            Context application = context.getApplicationContext();
            Path files = application.getFilesDir().toPath();
            Files.createDirectories(files); files = files.toRealPath();
            Path captures = files.resolve("captures"); Files.createDirectories(captures);
            if (Files.isSymbolicLink(captures) || !files.equals(captures.toRealPath().getParent()))
                throw new IOException("Capture root is outside app-private files");
            String stamp = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.ROOT).format(new Date());
            directory = captures.resolve("resource-" + stamp + "-" + UUID.randomUUID().toString().substring(0, 8));
            Files.createDirectory(directory);
            put(report, "schema_version", 1); put(report, "source", "resource_comparison_public_audio");
            put(report, "app_version", BuildConfig.VERSION_NAME); put(report, "microphone_started", false);
            put(report, "audio_captured", false); put(report, "audio_quality_verified", false);
            put(report, "live_queue_or_playout_verified", false); put(report, "automatic_configuration_applied", false);
            put(report, "status", "RUNNING"); put(report, "soft_budget_seconds", 600);
            put(report, "budget_scope", "Cooperative checks between synchronous calls and stages. Native loading, compilation or execution is not forcibly interrupted and can exceed the wall budget.");
            put(report, "scope", "Available-resource inventory, isolated prerequisites, unpaced full-model CPU-thread/HTP trials and direct GPU experiments on packaged public data. No microphone capture, live queue, speaker playout, energy or perceptual-quality acceptance.");
            put(report, "comparison_limits", "CPU pool-size order is 1,2,3,4,6,8,8,6,4,3,2,1; each model has two repeats. Pool size does not prove active/pinned cores; affinity and spinning policy are unchanged. HTP balanced/burst and GPU follow sequentially. Clocks, thermal conditions and backend order are not controlled. CPU process time includes all app threads and is not energy; provider evidence covers warm-up only for full streaming trials.");
            put(report, "cpu_thread_order", new JSONArray(ResourceComparisonPolicy.cpuThreadOrder()));
            put(report, "cpu_same_model_max_abs_limit", ResourceComparisonPolicy.MAX_ABS_ERROR);
            put(report, "cpu_same_model_normalized_rmse_limit", ResourceComparisonPolicy.MAX_NORMALIZED_RMSE);
            checkpoint(report, directory, stage);
            final long limit = deadline;
            QnnProbeRunner.Progress guarded = message -> {
                ResourceComparisonPolicy.checkContinue(cancelled, System.nanoTime(), limit);
                progress.update(message);
            };

            stage = "resource.runtime.initialize_info";
            checkpoint(report, directory, stage); guarded.update("Preparing runtime and local diagnostics…");
            try {
                OrtEnvironment environment = OrtEnvironment.getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO);
                environment.setTelemetry(false);
                put(report, "ort_version", environment.getVersion()); put(report, "ort_telemetry_enabled", false);
                put(report, "ort_info_logging_requested_before_probes", true);
                put(report, "ort_environment_scope", "Process-wide singleton; an environment previously created by another activity may retain its original environment logging configuration");
            } catch (Exception | LinkageError error) { recordError(report, errors, stage, error); }

            stage = "resource.inventory";
            checkpoint(report, directory, stage); guarded.update("Inspecting CPU and graphics resources…");
            try { put(report, "inventory", ResourceInventory.collect(application)); }
            catch (Exception | LinkageError error) { recordError(report, errors, stage, error); }
            ResourceComparisonPolicy.checkContinue(cancelled, System.nanoTime(), deadline);

            stage = "resource.vulkan_compute_smoke";
            checkpoint(report, directory, stage); guarded.update("Checking independent Vulkan compute availability…");
            try { put(report, "vulkan", VulkanResourceProbe.run()); }
            catch (CancellationException | ResourceComparisonPolicy.BudgetExceededException stop) { throw stop; }
            catch (Exception | LinkageError error) { recordError(report, errors, stage, error); }
            ResourceComparisonPolicy.checkContinue(cancelled, System.nanoTime(), deadline);

            stage = "resource.isolated_prerequisites";
            checkpoint(report, directory, stage);
            QnnProbeRunner.Result prerequisites;
            try {
                prerequisites = new QnnProbeRunner(application).run(guarded, directory.toFile());
                put(report, "prerequisites", prerequisites.diagnostics);
            } catch (CancellationException | ResourceComparisonPolicy.BudgetExceededException stop) { throw stop; }
            catch (Exception | LinkageError error) {
                recordError(report, errors, stage, error);
                prerequisites = new QnnProbeRunner.Result(directory.toFile(), new JSONObject());
            }
            ResourceComparisonPolicy.checkContinue(cancelled, System.nanoTime(), deadline);
            Map<String, Reference> references = new LinkedHashMap<>();
            int[] repeats = new int[9];
            for (int threads : ResourceComparisonPolicy.cpuThreadOrder()) {
                ResourceComparisonPolicy.checkContinue(cancelled, System.nanoTime(), deadline);
                int repeat = ++repeats[threads];
                String key = ResourceComparisonPolicy.trialKey(threads, repeat);
                stage = "resource." + key;
                JSONObject trial = new JSONObject(); cpuTrials.put(trial);
                put(trial, "trial_key", key); put(trial, "threads", threads); put(trial, "repeat", repeat);
                checkpoint(report, directory, stage);
                try {
                    QnnProbeRunner.Result source = cloneResult(prerequisites, directory);
                    QnnProbeRunner.Result completed = new FullStreamingBenchmark(application).run(source,
                            prefixed(guarded, "CPU " + threads + " threads · repeat " + repeat + "/2 · "),
                            threads, false, "balanced", key);
                    JSONObject benchmark = completed.diagnostics.optJSONObject(key);
                    if (benchmark == null) throw new IOException("Full CPU benchmark returned no trial result");
                    put(trial, "benchmark", benchmark);
                    ResourceComparisonPolicy.checkContinue(cancelled, System.nanoTime(), deadline);
                    put(trial, "parity", compareCpuOutputs(benchmark, key, threads, directory, references));
                } catch (CancellationException | ResourceComparisonPolicy.BudgetExceededException stop) { throw stop; }
                catch (Exception | LinkageError error) { recordTrialError(trial, errors, stage, error); }
                put(report, "cpu_summary", ResourceComparisonPolicy.summarizeCpu(cpuTrials));
                checkpoint(report, directory, stage + ".complete");
            }
            boolean registrationCleanupPassed = !prerequisites.diagnostics.has("plugin_cleanup_error")
                    && !prerequisites.diagnostics.has("environment_cleanup_error");
            for (String mode : new String[]{"balanced", "burst"}) {
                ResourceComparisonPolicy.checkContinue(cancelled, System.nanoTime(), deadline);
                String key = "htp_" + mode; stage = "resource." + key;
                JSONObject trial = new JSONObject(); htpTrials.put(trial);
                put(trial, "trial_key", key); put(trial, "threads", 2); put(trial, "htp_performance_mode", mode);
                checkpoint(report, directory, stage);
                if (!registrationCleanupPassed) {
                    put(trial, "status", "SKIPPED_PREREQUISITE_CLEANUP_INCOMPLETE");
                    continue;
                }
                try {
                    QnnProbeRunner.Result source = cloneResult(prerequisites, directory);
                    QnnProbeRunner.Result completed = new FullStreamingBenchmark(application).run(source,
                            prefixed(guarded, "HTP " + mode + " with CPU controls · "), 2, true, mode, key);
                    JSONObject benchmark = completed.diagnostics.optJSONObject(key);
                    if (benchmark == null) throw new IOException("Full HTP benchmark returned no trial result");
                    put(trial, "benchmark", benchmark); put(trial, "status", benchmark.optString("status"));
                } catch (CancellationException | ResourceComparisonPolicy.BudgetExceededException stop) { throw stop; }
                catch (Exception | LinkageError error) { recordTrialError(trial, errors, stage, error); }
                checkpoint(report, directory, stage + ".complete");
            }
            ResourceComparisonPolicy.checkContinue(cancelled, System.nanoTime(), deadline);
            stage = "resource.direct_gpu";
            checkpoint(report, directory, stage); guarded.update("Checking direct Qualcomm GPU execution…");
            if (!registrationCleanupPassed) {
                JSONObject skipped = new JSONObject(); put(skipped, "status", "SKIPPED_PREREQUISITE_CLEANUP_INCOMPLETE");
                put(skipped, "gpu_backend_execution_verified", false); put(report, "gpu", skipped);
            } else try { put(report, "gpu", GpuResourceProbe.run(application, directory.toFile(), guarded)); }
            catch (CancellationException | ResourceComparisonPolicy.BudgetExceededException stop) { throw stop; }
            catch (Exception | LinkageError error) { recordError(report, errors, stage, error); }
            ResourceComparisonPolicy.checkContinue(cancelled, System.nanoTime(), deadline);
            put(report, "status", "COMPARISONS_FINISHED"); put(report, "completed_all_stages", true);
            stage = "resource.complete";
        } catch (CancellationException stop) {
            put(report, "status", "CANCELLED"); put(report, "cancelled", true);
            put(report, "completed_all_stages", false); put(report, "stop_reason", stop.getMessage());
        } catch (ResourceComparisonPolicy.BudgetExceededException stop) {
            put(report, "status", "SOFT_BUDGET_EXCEEDED"); put(report, "completed_all_stages", false);
            put(report, "stop_reason", stop.getMessage());
        } catch (Exception | LinkageError error) {
            put(report, "status", "ERROR"); put(report, "completed_all_stages", false);
            recordError(report, errors, stage, error);
        } finally {
            put(report, "cpu_summary", ResourceComparisonPolicy.summarizeCpu(cpuTrials));
            put(report, "final_stage", stage); put(report, "total_wall_ms", (System.nanoTime() - began) / 1e6);
            put(report, "audio_quality_verified", false); put(report, "microphone_started", false);
            if (directory != null) try { save(report, directory, true); }
            catch (Exception error) { put(report, "diagnostics_write_error", EnhancementSession.boundedErrorDetails(error)); }
        }
        return new Result(directory == null ? null : directory.toFile(), report);
    }

    private static JSONObject compareCpuOutputs(JSONObject benchmark, String trialKey, int threads,
                                                Path directory, Map<String, Reference> references) {
        JSONObject parity = new JSONObject();
        for (String mode : new String[]{"cpu_fp32", "cpu_qdq"}) {
            JSONObject comparison;
            try {
                JSONObject backends = benchmark.optJSONObject("backends");
                JSONObject backend = backends == null ? null : backends.optJSONObject(mode);
                if (backend == null || !backend.optBoolean("execution_verified", false))
                    throw new IOException("CPU sequence was not fully verified");
                JSONObject completeness = new JSONObject();
                put(completeness, "passed", true); put(completeness, "file_sha256_verified", true);
                put(completeness, "max_abs", 0); put(completeness, "normalized_rmse", 0);
                if (!ResourceComparisonPolicy.eligibleCpuResult(benchmark, mode, completeness, threads))
                    throw new IOException("CPU sequence is incomplete or lacks verified unprofiled timing/accounting");
                JSONObject fingerprint = ResourceComparisonPolicy.fingerprint(benchmark, mode);
                float[] actual = loadVerifiedOutput(directory, backend);
                Reference reference = references.get(mode);
                boolean first = reference == null;
                if (first) reference = new Reference(trialKey, fingerprint, actual);
                comparison = ResourceComparisonPolicy.compareSameModel(fingerprint, actual,
                        reference.fingerprint, reference.samples, first);
                put(comparison, "reference_trial_key", reference.trialKey);
                put(comparison, "file_sha256_verified", true);
                boolean eligible = ResourceComparisonPolicy.eligibleCpuResult(benchmark, mode, comparison, threads);
                put(comparison, "eligible_for_cpu_summary", eligible);
                if (first && eligible) references.put(mode, reference);
            } catch (Exception error) {
                comparison = new JSONObject(); put(comparison, "passed", false);
                put(comparison, "error", error.toString());
                put(comparison, "error_details", EnhancementSession.boundedErrorDetails(error));
            }
            put(parity, mode, comparison);
        }
        return parity;
    }

    static float[] loadVerifiedOutput(Path directory, JSONObject backend) throws Exception {
        JSONObject saved = backend.optJSONObject("saved_public_output");
        if (saved == null) throw new IOException("Verified public float output metadata is missing");
        String filename = saved.optString("float32_file"), expectedHash = saved.optString("float32_sha256");
        int samples = saved.optInt("samples", -1);
        if (!filename.matches("[A-Za-z0-9_-]{1,80}\\.f32") || !expectedHash.matches("[a-f0-9]{64}")
                || samples <= 0 || samples > MAX_OUTPUT_SAMPLES || samples != backend.optInt("trimmed_output_samples", -1))
            throw new IOException("Unsafe or inconsistent public float-output metadata");
        Path root = directory.toRealPath(), path = root.resolve(filename);
        if (!Files.isRegularFile(path, java.nio.file.LinkOption.NOFOLLOW_LINKS) || Files.isSymbolicLink(path)
                || !root.equals(path.toRealPath().getParent()) || Files.size(path) != (long) samples * Float.BYTES)
            throw new IOException("Public float output is missing, outside the capture directory or has the wrong size");
        byte[] data = Files.readAllBytes(path);
        if (data.length != (long) samples * Float.BYTES || !sha256(data).equals(expectedHash))
            throw new IOException("Public float output hash differs from its verified backend result");
        return ProbeVerification.decodeFloat32(data, samples);
    }

    private static QnnProbeRunner.Result cloneResult(QnnProbeRunner.Result original, Path directory) throws Exception {
        return new QnnProbeRunner.Result(directory.toFile(), new JSONObject(original.diagnostics.toString()));
    }
    private static QnnProbeRunner.Progress prefixed(QnnProbeRunner.Progress progress, String prefix) {
        return message -> progress.update(prefix + message);
    }
    private static final class Reference {
        final String trialKey; final JSONObject fingerprint; final float[] samples;
        Reference(String trialKey, JSONObject fingerprint, float[] samples) {
            this.trialKey = trialKey; this.fingerprint = fingerprint; this.samples = samples;
        }
    }
    private static void recordTrialError(JSONObject trial, JSONArray errors, String stage, Throwable error) {
        put(trial, "status", "ERROR"); put(trial, "error", error.toString());
        put(trial, "error_details", EnhancementSession.boundedErrorDetails(error));
        recordError(trial, errors, stage, error);
    }
    private static void recordError(JSONObject target, JSONArray errors, String stage, Throwable error) {
        JSONObject detail = new JSONObject(); put(detail, "stage", stage); put(detail, "error", error.toString());
        put(detail, "error_details", EnhancementSession.boundedErrorDetails(error)); errors.put(detail);
        put(target, "last_error_stage", stage);
    }
    private static void checkpoint(JSONObject report, Path directory, String stage) throws IOException {
        put(report, "last_stage", stage); save(report, directory, false);
    }
    private static void save(JSONObject report, Path directory, boolean finalFiles) throws IOException {
        byte[] bytes;
        try { bytes = (report.toString(2) + "\n").getBytes(StandardCharsets.UTF_8); }
        catch (org.json.JSONException error) { throw new IOException("Could not serialize resource diagnostics", error); }
        if (bytes.length > MAX_REPORT_BYTES) throw new IOException("Resource diagnostics exceed the bounded 8 MiB report size");
        String[] names = finalFiles ? new String[]{"resource-runs.json", "stats.json", "phone-capture.json", "qnn-probe.json"}
                : new String[]{"resource-runs.json"};
        for (String name : names) {
            Path file = directory.resolve(name), temporary = directory.resolve(name + ".tmp");
            Files.write(temporary, bytes);
            try { Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE); }
            catch (AtomicMoveNotSupportedException error) { Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING); }
        }
    }
    private static String sha256(byte[] bytes) throws Exception {
        StringBuilder text = new StringBuilder();
        for (byte value : MessageDigest.getInstance("SHA-256").digest(bytes)) text.append(String.format(Locale.ROOT, "%02x", value & 255));
        return text.toString();
    }
    private static void put(JSONObject out, String key, Object value) {
        try { out.put(key, value); }
        catch (org.json.JSONException error) { throw new IllegalArgumentException("Cannot serialize resource comparison field " + key, error); }
    }
}
