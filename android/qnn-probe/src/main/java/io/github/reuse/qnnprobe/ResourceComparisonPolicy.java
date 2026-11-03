package io.github.reuse.qnnprobe;

import java.io.IOException;
import java.util.Arrays;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.function.BooleanSupplier;
import org.json.JSONArray;
import org.json.JSONObject;

/** Pure consistency, completeness and ranking rules for resource comparisons. */
final class ResourceComparisonPolicy {
    static final double MAX_ABS_ERROR = 1e-4, MAX_NORMALIZED_RMSE = 1e-4;
    static final int CALLS = 65, REPEATS = 2;
    private static final String[] CPU_MODES = {"cpu_fp32", "cpu_qdq"};
    private static final int[] THREAD_SETTINGS = {1, 2, 3, 4, 6, 8};
    private static final String[] FINGERPRINT_KEYS = {"model_sha256", "manifest_sha256", "public_input_sha256",
            "fixture_manifest_sha256", "custom_op_library_sha256", "ort_version", "sample_rate", "input_samples",
            "frames_per_call", "exit_layers", "native_neon_fma_build_flag", "native_scan_n16_registers_build_flag"};

    private ResourceComparisonPolicy() { }

    static int[] cpuThreadSettings() { return THREAD_SETTINGS.clone(); }
    static int[] cpuThreadOrder() { return new int[]{1, 2, 3, 4, 6, 8, 8, 6, 4, 3, 2, 1}; }

    static String trialKey(int threads, int repeat) {
        if (Arrays.binarySearch(THREAD_SETTINGS, threads) < 0 || repeat < 1 || repeat > REPEATS)
            throw new IllegalArgumentException("CPU resource trial requires a configured pool size (1,2,3,4,6,8) and repeat 1..2");
        return "cpu_t" + threads + "_pass" + repeat;
    }

    static void checkContinue(BooleanSupplier cancelled, long now, long deadline) throws BudgetExceededException {
        if (cancelled == null) throw new IllegalArgumentException("Cancellation supplier is required");
        if (cancelled.getAsBoolean() || Thread.currentThread().isInterrupted())
            throw new CancellationException("Resource comparison cancelled between native calls/stages");
        if (now >= deadline) throw new BudgetExceededException();
    }

    static final class BudgetExceededException extends RuntimeException {
        BudgetExceededException() { super("Resource comparison cooperative wall budget reached; native calls are not forcibly interrupted"); }
    }

    static JSONObject fingerprint(JSONObject benchmark, String cpuMode) throws IOException {
        if (!Arrays.asList(CPU_MODES).contains(cpuMode)) throw new IOException("Unknown CPU model mode");
        JSONObject backends = benchmark.optJSONObject("backends");
        JSONObject backend = backends == null ? null : backends.optJSONObject(cpuMode);
        if (backend == null) throw new IOException("CPU model result is missing");
        JSONObject result = new JSONObject();
        for (String key : new String[]{"model_sha256", "manifest_sha256"}) put(result, key, checkedSha(backend.optString(key)));
        for (String key : new String[]{"public_input_sha256", "fixture_manifest_sha256", "custom_op_library_sha256"})
            put(result, key, checkedSha(benchmark.optString(key)));
        String ort = benchmark.optString("ort_version");
        if (ort.isEmpty()) throw new IOException("ORT version is missing from model fingerprint");
        put(result, "ort_version", ort);
        for (String key : new String[]{"sample_rate", "input_samples", "frames_per_call", "exit_layers"}) {
            int value = benchmark.optInt(key, -1);
            if (value <= 0) throw new IOException("Missing positive model fingerprint field: " + key);
            put(result, key, value);
        }
        for (String key : new String[]{"native_neon_fma_build_flag", "native_scan_n16_registers_build_flag"}) {
            Object value = benchmark.opt(key);
            if (!(value instanceof Boolean)) throw new IOException("Missing boolean model fingerprint field: " + key);
            put(result, key, value);
        }
        return result;
    }

    static JSONObject compareSameModel(JSONObject actualFingerprint, float[] actual,
                                        JSONObject referenceFingerprint, float[] reference,
                                        boolean referenceOutput) throws IOException {
        for (String key : FINGERPRINT_KEYS) {
            Object actualField = actualFingerprint.opt(key), referenceField = referenceFingerprint.opt(key);
            if (actualField == null || referenceField == null || !actualField.toString().equals(referenceField.toString()))
                throw new IOException("Cross-thread parity fingerprint mismatch: " + key);
        }
        JSONObject result = FullStreamingBenchmark.audioComparison(actual, reference);
        double maximum = result.optDouble("max_abs", Double.NaN);
        double normalized = result.optDouble("normalized_rmse", Double.NaN);
        boolean passed = Double.isFinite(maximum) && Double.isFinite(normalized)
                && maximum <= MAX_ABS_ERROR && normalized <= MAX_NORMALIZED_RMSE;
        put(result, "passed", passed);
        put(result, "reference_output", referenceOutput);
        put(result, "max_abs_limit", MAX_ABS_ERROR);
        put(result, "normalized_rmse_limit", MAX_NORMALIZED_RMSE);
        put(result, "scope", referenceOutput
                ? "First complete same-model output selected as reference; this self-comparison is not independent accuracy or perceptual-quality evidence"
                : "Cross-thread consistency against the first complete output of exactly the same graph, input, native build and ORT version; not perceptual-quality acceptance");
        return result;
    }

    static JSONObject summarizeCpu(JSONArray trials) {
        JSONArray configurations = new JSONArray();
        JSONObject fastestByModel = new JSONObject();
        JSONObject fastest = null;
        int eligibleConfigurations = 0;
        for (int threads : THREAD_SETTINGS) for (String mode : CPU_MODES) {
            JSONObject configuration = summarizeConfiguration(trials, threads, mode);
            configurations.put(configuration);
            if (!configuration.optBoolean("eligible", false)) continue;
            eligibleConfigurations++;
            if (fastest == null || configuration.optDouble("mean_call_wall_ms") < fastest.optDouble("mean_call_wall_ms")) fastest = configuration;
            JSONObject previous = fastestByModel.optJSONObject(mode);
            if (previous == null || configuration.optDouble("mean_call_wall_ms") < previous.optDouble("mean_call_wall_ms"))
                put(fastestByModel, mode, configuration);
        }
        JSONObject summary = new JSONObject();
        put(summary, "configurations", configurations);
        put(summary, "fastest_cpu_configuration", fastest == null ? JSONObject.NULL : fastest);
        put(summary, "fastest_cpu_by_model", fastestByModel);
        put(summary, "eligible_configurations", eligibleConfigurations);
        put(summary, "expected_configurations", THREAD_SETTINGS.length * CPU_MODES.length);
        JSONArray settings = new JSONArray();
        for (int threads : THREAD_SETTINGS) settings.put(threads);
        put(summary, "thread_pool_settings", settings);
        put(summary, "cpu_thread_pool_scope", "Requested ORT intra-op pool sizes, not proof of active/pinned cores. Android scheduling and affinity are not controlled; spinning policy is unchanged.");
        put(summary, "all_cpu_configurations_verified", eligibleConfigurations == THREAD_SETTINGS.length * CPU_MODES.length);
        put(summary, "audio_quality_verified", false);
        put(summary, "automatic_configuration_applied", false);
        put(summary, "ranking_scope", "Lowest mean of two complete same-model, file-verified, warm-up-provider-verified, unprofiled and cross-thread-consistent sequences. Model precision differs between cpu_fp32 and cpu_qdq; this is timing selection, not a model-quality recommendation, microphone test, live-queue or playout guarantee. Order, clocks and thermal conditions are not controlled.");
        return summary;
    }

    private static JSONObject summarizeConfiguration(JSONArray trials, int threads, String mode) {
        JSONObject configuration = new JSONObject();
        put(configuration, "threads", threads); put(configuration, "cpu_mode", mode);
        put(configuration, "expected_repeats", REPEATS);
        int observed = 0, verified = 0;
        double wall = 0, rtf = 0, cpuMean = 0, cpuTotal = 0, cpuRate = 0, maximum = 0, p95 = 0;
        int over = 0;
        Set<Integer> repeats = new LinkedHashSet<>();
        JSONObject firstFingerprint = null;
        boolean consistentFingerprint = true;
        for (int i = 0; i < trials.length(); i++) {
            JSONObject trial = trials.optJSONObject(i);
            if (trial == null || trial.optInt("threads") != threads) continue;
            observed++;
            JSONObject benchmark = trial.optJSONObject("benchmark"), parity = trial.optJSONObject("parity");
            JSONObject comparison = parity == null ? null : parity.optJSONObject(mode);
            if (benchmark == null || !eligibleCpuResult(benchmark, mode, comparison, threads)) continue;
            JSONObject backend = benchmark.optJSONObject("backends").optJSONObject(mode);
            try {
                JSONObject currentFingerprint = fingerprint(benchmark, mode);
                if (firstFingerprint == null) firstFingerprint = currentFingerprint;
                else for (String key : FINGERPRINT_KEYS)
                    consistentFingerprint &= firstFingerprint.opt(key).toString().equals(currentFingerprint.opt(key).toString());
            } catch (IOException error) { continue; }
            int repeat = trial.optInt("repeat", -1);
            if (repeat < 1 || repeat > REPEATS) continue;
            repeats.add(repeat); verified++;
            wall += backend.optDouble("mean_call_wall_ms"); rtf += backend.optDouble("offline_processing_rtf");
            cpuMean += backend.optDouble("mean_call_process_cpu_ms"); cpuTotal += backend.optDouble("processing_process_cpu_ms");
            cpuRate += backend.optDouble("process_cpu_seconds_per_audio_second");
            maximum = Math.max(maximum, backend.optDouble("max_call_wall_ms"));
            p95 = Math.max(p95, backend.optDouble("p95_call_wall_ms"));
            over += backend.optInt("calls_over_100ms");
        }
        boolean eligible = observed == REPEATS && verified == REPEATS && repeats.size() == REPEATS && consistentFingerprint;
        put(configuration, "observed_repeats", observed); put(configuration, "verified_repeats", verified);
        put(configuration, "eligible", eligible); put(configuration, "consistent_model_fingerprint", consistentFingerprint);
        put(configuration, "model_sha256", firstFingerprint == null ? JSONObject.NULL : firstFingerprint.opt("model_sha256"));
        if (firstFingerprint != null) put(configuration, "model_fingerprint", firstFingerprint);
        if (verified > 0) {
            put(configuration, "mean_call_wall_ms", wall / verified); put(configuration, "offline_processing_rtf", rtf / verified);
            put(configuration, "mean_call_process_cpu_ms", cpuMean / verified);
            put(configuration, "processing_process_cpu_ms", cpuTotal / verified);
            put(configuration, "process_cpu_seconds_per_audio_second", cpuRate / verified);
            put(configuration, "max_call_wall_ms", maximum); put(configuration, "maximum_repeat_p95_call_wall_ms", p95);
            put(configuration, "calls_over_100ms", over);
        }
        if (!eligible) put(configuration, "ineligible_reason", "Requires exactly two distinct complete verified repeats of one identical model/input/runtime/native build with passing cross-thread consistency");
        return configuration;
    }

    static boolean eligibleCpuResult(JSONObject benchmark, String mode, JSONObject parity, int threads) {
        JSONObject backends = benchmark.optJSONObject("backends"), backend = backends == null ? null : backends.optJSONObject(mode);
        if (backend == null || parity == null || !parity.optBoolean("passed", false)
                || !parity.optBoolean("file_sha256_verified", false)
                || !backend.optBoolean("execution_verified", false)
                || !"VERIFIED_FINITE_FULL_SEQUENCE".equals(backend.optString("status"))
                || !backend.optBoolean("warmup_provider_execution_verified", false)
                || !backend.optBoolean("all_output_samples_finite", false)
                || backend.optBoolean("timed_calls_profiled", true)
                || backend.optInt("cpu_intra_op_threads", -1) != threads
                || backend.optInt("completed_audio_calls", -1) != CALLS
                || benchmark.optInt("audio_calls", -1) != CALLS
                || backend.optInt("trimmed_output_samples", -1) != benchmark.optInt("input_samples", -2)
                || benchmark.optInt("input_samples", 0) <= 0) return false;
        if (!nonnegative(parity.optDouble("max_abs", Double.NaN)) || parity.optDouble("max_abs") > MAX_ABS_ERROR
                || !nonnegative(parity.optDouble("normalized_rmse", Double.NaN))
                || parity.optDouble("normalized_rmse") > MAX_NORMALIZED_RMSE) return false;
        JSONObject evidence = backend.optJSONObject("warmup_provider_evidence");
        if (evidence == null || evidence.optInt("model_run_events", -1) != 3
                || evidence.optInt("cpu_kernel_events", 0) <= 0 || evidence.optInt("qnn_kernel_events", -1) != 0
                || evidence.optInt("other_provider_kernel_events", -1) != 0 || evidence.optInt("missing_provider_kernel_events", -1) != 0) return false;
        JSONArray times = backend.optJSONArray("audio_call_wall_ms"), cpuTimes = backend.optJSONArray("audio_call_process_cpu_ms");
        if (times == null || cpuTimes == null || times.length() != CALLS || cpuTimes.length() != CALLS) return false;
        for (int i = 0; i < CALLS; i++) if (!nonnegative(times.optDouble(i, Double.NaN)) || !nonnegative(cpuTimes.optDouble(i, Double.NaN))) return false;
        for (String key : new String[]{"mean_call_wall_ms", "offline_processing_rtf", "max_call_wall_ms", "p95_call_wall_ms"})
            if (!positive(backend.optDouble(key, Double.NaN))) return false;
        for (String key : new String[]{"mean_call_process_cpu_ms", "processing_process_cpu_ms", "process_cpu_seconds_per_audio_second"})
            if (!nonnegative(backend.optDouble(key, Double.NaN))) return false;
        if (backend.optInt("calls_over_100ms", -1) < 0) return false;
        try { fingerprint(benchmark, mode); return true; }
        catch (IOException error) { return false; }
    }

    private static boolean positive(double value) { return Double.isFinite(value) && value > 0; }
    private static boolean nonnegative(double value) { return Double.isFinite(value) && value >= 0; }
    private static String checkedSha(String value) throws IOException {
        if (!value.matches("[a-f0-9]{64}")) throw new IOException("Incomplete model/input/native SHA-256 fingerprint");
        return value;
    }
    private static void put(JSONObject out, String key, Object value) {
        try { out.put(key, value); }
        catch (org.json.JSONException error) { throw new IllegalArgumentException("Cannot serialize resource policy field " + key, error); }
    }
}
