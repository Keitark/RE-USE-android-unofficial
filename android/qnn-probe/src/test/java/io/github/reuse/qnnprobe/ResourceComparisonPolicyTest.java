package io.github.reuse.qnnprobe;

import static org.junit.Assert.*;
import java.io.IOException;
import java.util.concurrent.CancellationException;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;

public final class ResourceComparisonPolicyTest {
    @Test public void cpuOrderCounterbalancesEachThreadSettingWithTwoDistinctKeys() {
        int[] order = ResourceComparisonPolicy.cpuThreadOrder();
        assertArrayEquals(new int[]{1, 2, 3, 4, 6, 8, 8, 6, 4, 3, 2, 1}, order);
        order[0] = 99;
        assertEquals(1, ResourceComparisonPolicy.cpuThreadOrder()[0]);
        assertArrayEquals(new int[]{1, 2, 3, 4, 6, 8}, ResourceComparisonPolicy.cpuThreadSettings());
        int[] settings = ResourceComparisonPolicy.cpuThreadSettings(); settings[0] = 99;
        assertEquals(1, ResourceComparisonPolicy.cpuThreadSettings()[0]);
        assertEquals("cpu_t8_pass2", ResourceComparisonPolicy.trialKey(8, 2));
        for (int threads : new int[]{0, 5, 7, 9})
            assertThrows(IllegalArgumentException.class, () -> ResourceComparisonPolicy.trialKey(threads, 1));
        assertThrows(IllegalArgumentException.class, () -> ResourceComparisonPolicy.trialKey(2, 3));
    }

    @Test public void parityRejectsWrongGraphInputOrNativeFingerprintBeforeComparingSamples() throws Exception {
        JSONObject expected = ResourceComparisonPolicy.fingerprint(benchmark(1, 80, 60), "cpu_fp32");
        for (String key : new String[]{"model_sha256", "manifest_sha256", "public_input_sha256", "custom_op_library_sha256", "ort_version"}) {
            JSONObject actual = new JSONObject(expected.toString()).put(key, "different");
            assertThrows(IOException.class, () -> ResourceComparisonPolicy.compareSameModel(actual,
                    new float[]{1}, expected, new float[]{1}, false));
        }
        JSONObject same = ResourceComparisonPolicy.compareSameModel(expected, new float[]{1, -1}, expected, new float[]{1, -1}, true);
        assertTrue(same.getBoolean("passed")); assertTrue(same.getBoolean("reference_output"));
        assertFalse(same.getBoolean("audio_quality_verified"));
        JSONObject wrong = ResourceComparisonPolicy.compareSameModel(expected, new float[]{1.001f, -1}, expected, new float[]{1, -1}, false);
        assertFalse(wrong.getBoolean("passed"));
        assertEquals(1e-4, wrong.getDouble("max_abs_limit"), 0);
        assertEquals(1e-4, wrong.getDouble("normalized_rmse_limit"), 0);
    }

    @Test public void missingUnverifiedAndDuplicateRepeatsCannotWinDespiteFasterReportedTimes() throws Exception {
        JSONArray trials = new JSONArray();
        trials.put(trial(1, 1, 80, 60)); trials.put(trial(1, 2, 82, 62));
        trials.put(trial(2, 1, 2, 1));
        JSONObject unverified = trial(3, 1, 1, .5);
        unverified.getJSONObject("benchmark").getJSONObject("backends").getJSONObject("cpu_qdq").put("execution_verified", false);
        unverified.getJSONObject("benchmark").getJSONObject("backends").getJSONObject("cpu_fp32").put("execution_verified", false);
        trials.put(unverified); trials.put(trial(3, 2, 1, .5));
        JSONObject noFileProof = trial(4, 1, 1, .1);
        noFileProof.getJSONObject("parity").getJSONObject("cpu_qdq").remove("file_sha256_verified");
        noFileProof.getJSONObject("parity").getJSONObject("cpu_fp32").put("passed", false);
        trials.put(noFileProof); trials.put(trial(4, 2, 1, .1));
        JSONObject summary = ResourceComparisonPolicy.summarizeCpu(trials);
        JSONObject fastest = summary.getJSONObject("fastest_cpu_configuration");
        assertEquals(1, fastest.getInt("threads")); assertEquals("cpu_qdq", fastest.getString("cpu_mode"));
        assertEquals(61, fastest.getDouble("mean_call_wall_ms"), 0);
        assertEquals(2, fastest.getInt("verified_repeats"));
        assertFalse(summary.getBoolean("automatic_configuration_applied"));
        assertFalse(summary.getBoolean("audio_quality_verified"));
        JSONArray duplicate = new JSONArray().put(trial(1, 1, 1, .1)).put(trial(1, 1, 1, .1));
        assertTrue(ResourceComparisonPolicy.summarizeCpu(duplicate).isNull("fastest_cpu_configuration"));
    }

    @Test public void inconsistentModelsIncompleteOutputAndBadParityNeverBecomeEligible() throws Exception {
        JSONObject first = trial(1, 1, 50, 40), second = trial(1, 2, 50, 40);
        second.getJSONObject("benchmark").getJSONObject("backends").getJSONObject("cpu_qdq").put("model_sha256", "f".repeat(64));
        JSONObject summary = ResourceComparisonPolicy.summarizeCpu(new JSONArray().put(first).put(second));
        JSONObject qdq = configuration(summary, 1, "cpu_qdq");
        assertFalse(qdq.getBoolean("eligible")); assertFalse(qdq.getBoolean("consistent_model_fingerprint"));
        JSONObject b = benchmark(2, 50, 40), p = parity();
        b.getJSONObject("backends").getJSONObject("cpu_qdq").put("completed_audio_calls", 64);
        assertFalse(ResourceComparisonPolicy.eligibleCpuResult(b, "cpu_qdq", p, 2));
        b = benchmark(2, 50, 40); p.put("max_abs", .001);
        assertFalse(ResourceComparisonPolicy.eligibleCpuResult(b, "cpu_qdq", p, 2));
        assertTrue(ResourceComparisonPolicy.summarizeCpu(new JSONArray()).isNull("fastest_cpu_configuration"));
    }

    @Test public void allConfigurationsRequireTwoVerifiedRunsAndAverageOnlyThoseRepeatTimes() throws Exception {
        JSONArray trials = new JSONArray();
        for (int threads : new int[]{1, 2, 3, 4, 6, 8}) {
            trials.put(trial(threads, 1, 80 - threads, 60 - threads));
            trials.put(trial(threads, 2, 82 - threads, 62 - threads));
        }
        JSONObject summary = ResourceComparisonPolicy.summarizeCpu(trials);
        assertTrue(summary.getBoolean("all_cpu_configurations_verified"));
        assertEquals(12, summary.getInt("eligible_configurations"));
        assertEquals(12, summary.getInt("expected_configurations"));
        JSONObject best = summary.getJSONObject("fastest_cpu_configuration");
        assertEquals(8, best.getInt("threads")); assertEquals(53, best.getDouble("mean_call_wall_ms"), 0);
        assertEquals("b".repeat(64), best.getString("model_sha256"));
        assertEquals(73, summary.getJSONObject("fastest_cpu_by_model").getJSONObject("cpu_fp32").getDouble("mean_call_wall_ms"), 0);
        assertTrue(summary.getString("cpu_thread_pool_scope").contains("not proof"));
    }

    @Test public void cancellationTakesPriorityOverBudgetAndNeitherCanReturnACompletedPolicy() {
        ResourceComparisonPolicy.checkContinue(() -> false, 9, 10);
        assertThrows(CancellationException.class, () -> ResourceComparisonPolicy.checkContinue(() -> true, 9, 10));
        assertThrows(CancellationException.class, () -> ResourceComparisonPolicy.checkContinue(() -> true, 10, 10));
        assertThrows(ResourceComparisonPolicy.BudgetExceededException.class,
                () -> ResourceComparisonPolicy.checkContinue(() -> false, 10, 10));
        assertThrows(IllegalArgumentException.class, () -> ResourceComparisonPolicy.checkContinue(null, 1, 2));
    }

    private static JSONObject configuration(JSONObject summary, int threads, String mode) throws Exception {
        JSONArray values = summary.getJSONArray("configurations");
        for (int i = 0; i < values.length(); i++) {
            JSONObject value = values.getJSONObject(i);
            if (value.getInt("threads") == threads && mode.equals(value.getString("cpu_mode"))) return value;
        }
        throw new AssertionError("Missing configuration");
    }
    private static JSONObject trial(int threads, int repeat, double fp32Ms, double qdqMs) throws Exception {
        return new JSONObject().put("threads", threads).put("repeat", repeat).put("trial_key", ResourceComparisonPolicy.trialKey(threads, repeat))
                .put("benchmark", benchmark(threads, fp32Ms, qdqMs))
                .put("parity", new JSONObject().put("cpu_fp32", parity()).put("cpu_qdq", parity()));
    }
    private static JSONObject parity() throws Exception {
        return new JSONObject().put("passed", true).put("file_sha256_verified", true).put("max_abs", 0).put("normalized_rmse", 0);
    }
    private static JSONObject benchmark(int threads, double fp32Ms, double qdqMs) throws Exception {
        return new JSONObject().put("audio_calls", 65).put("input_samples", 51840).put("sample_rate", 8000)
                .put("frames_per_call", 5).put("exit_layers", 3).put("public_input_sha256", "c".repeat(64))
                .put("fixture_manifest_sha256", "d".repeat(64)).put("custom_op_library_sha256", "e".repeat(64))
                .put("ort_version", "1.29.0").put("native_neon_fma_build_flag", true).put("native_scan_n16_registers_build_flag", true)
                .put("backends", new JSONObject().put("cpu_fp32", backend(threads, fp32Ms, "a".repeat(64)))
                        .put("cpu_qdq", backend(threads, qdqMs, "b".repeat(64))));
    }
    private static JSONObject backend(int threads, double milliseconds, String model) throws Exception {
        JSONArray wall = new JSONArray(), cpu = new JSONArray();
        for (int i = 0; i < 65; i++) { wall.put(milliseconds); cpu.put(milliseconds * 1.5); }
        return new JSONObject().put("model_sha256", model).put("manifest_sha256", "0".repeat(64))
                .put("execution_verified", true).put("status", "VERIFIED_FINITE_FULL_SEQUENCE")
                .put("warmup_provider_execution_verified", true).put("all_output_samples_finite", true)
                .put("timed_calls_profiled", false).put("cpu_intra_op_threads", threads).put("completed_audio_calls", 65)
                .put("trimmed_output_samples", 51840).put("audio_call_wall_ms", wall).put("audio_call_process_cpu_ms", cpu)
                .put("mean_call_wall_ms", milliseconds).put("offline_processing_rtf", milliseconds / 100)
                .put("max_call_wall_ms", milliseconds).put("p95_call_wall_ms", milliseconds).put("calls_over_100ms", 0)
                .put("mean_call_process_cpu_ms", milliseconds * 1.5).put("processing_process_cpu_ms", milliseconds * 65 * 1.5)
                .put("process_cpu_seconds_per_audio_second", milliseconds * 1.5 / 100)
                .put("warmup_provider_evidence", new JSONObject().put("model_run_events", 3).put("cpu_kernel_events", 100)
                        .put("qnn_kernel_events", 0).put("other_provider_kernel_events", 0).put("missing_provider_kernel_events", 0));
    }
}
