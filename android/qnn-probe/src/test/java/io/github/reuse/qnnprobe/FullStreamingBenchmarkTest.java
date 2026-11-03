package io.github.reuse.qnnprobe;

import static org.junit.Assert.*;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;

/** Pure data contracts; these tests never load native inference or start Android audio. */
public class FullStreamingBenchmarkTest {
    @Test public void resourceOptionsBoundThreadsModesAndFilenameSafeReportKeys() {
        for (int threads = 1; threads <= 8; threads++) {
            FullStreamingBenchmark.validateOptions(threads, "balanced", "cpu_t" + threads);
            FullStreamingBenchmark.validateOptions(threads, "burst", "cpu_t" + threads + "_burst");
        }
        FullStreamingBenchmark.validateOptions(2, "balanced", "A".repeat(64));
        for (int threads : new int[]{0, -1, 9, 16})
            assertThrows(IllegalArgumentException.class, () -> FullStreamingBenchmark.validateOptions(threads, "balanced", "cpu_trial"));
        for (String mode : new String[]{null, "", "BALANCED", "balanced\n", "sustained_high_performance"})
            assertThrows(IllegalArgumentException.class, () -> FullStreamingBenchmark.validateOptions(2, mode, "cpu_trial"));
        for (String key : new String[]{null, "", "a".repeat(65), "../trial", "/trial", "trial/path", "trial\\path", "a.b", "trial-name", "trial name", "é", "1trial"})
            assertThrows(IllegalArgumentException.class, () -> FullStreamingBenchmark.validateOptions(2, "balanced", key));
    }

    @Test public void legacyPrefixAndCpuOnlySelectionKeepTrialsSeparateWithoutRequestingHtp() {
        assertEquals("full", FullStreamingBenchmark.outputPrefix("full_stream_benchmark"));
        assertEquals("resource_cpu_t1", FullStreamingBenchmark.outputPrefix("resource_cpu_t1"));
        assertEquals("resource_cpu_t2", FullStreamingBenchmark.outputPrefix("resource_cpu_t2"));
        assertArrayEquals(new FullStreamingBenchmark.Mode[]{FullStreamingBenchmark.Mode.CPU_FP32,
                FullStreamingBenchmark.Mode.CPU_QDQ}, FullStreamingBenchmark.selectedModes(false));
        assertArrayEquals(FullStreamingBenchmark.Mode.values(), FullStreamingBenchmark.selectedModes(true));
    }

    @Test public void resourceTrialsKeepTheirSavedAudioFilesInTheSameDirectory() throws Exception {
        Path directory = Files.createTempDirectory("reuse-resource-audio-test");
        String first = FullStreamingBenchmark.outputPrefix("cpu_t1") + "-cpu-qdq";
        String second = FullStreamingBenchmark.outputPrefix("cpu_t2") + "-cpu-qdq";
        try {
            JSONObject one = FullStreamingBenchmark.saveAudio(directory, first, new float[]{.1f});
            JSONObject two = FullStreamingBenchmark.saveAudio(directory, second, new float[]{.2f});
            assertEquals(first + ".f32", one.getString("float32_file"));
            assertEquals(second + ".f32", two.getString("float32_file"));
            ByteBuffer oneRaw = ByteBuffer.wrap(Files.readAllBytes(directory.resolve(first + ".f32"))).order(ByteOrder.LITTLE_ENDIAN);
            ByteBuffer twoRaw = ByteBuffer.wrap(Files.readAllBytes(directory.resolve(second + ".f32"))).order(ByteOrder.LITTLE_ENDIAN);
            assertEquals(.1f, oneRaw.getFloat(), 0);
            assertEquals(.2f, twoRaw.getFloat(), 0);
        } finally {
            for (String prefix : new String[]{first, second}) {
                Files.deleteIfExists(directory.resolve(prefix + ".f32"));
                Files.deleteIfExists(directory.resolve(prefix + ".wav"));
            }
            Files.deleteIfExists(directory);
        }
    }

    @Test public void processCpuSummaryAllowsParallelCpuTimeAndDoesNotTreatItAsWallDeadlineOrEnergy() throws Exception {
        JSONObject summary = FullStreamingBenchmark.processCpuSummary(new double[]{180, 160}, 20, 1600);
        assertEquals(360, summary.getDouble("processing_process_cpu_ms"), 0);
        assertEquals(170, summary.getDouble("mean_call_process_cpu_ms"), 0);
        assertEquals(180, summary.getDouble("max_call_process_cpu_ms"), 0);
        assertEquals(20, summary.getDouble("flush_process_cpu_ms"), 0);
        assertEquals(1.8, summary.getDouble("process_cpu_seconds_per_audio_second"), 1e-12);
        assertFalse(summary.has("calls_over_100ms"));
        assertFalse(summary.has("offline_processing_rtf"));
        assertTrue(summary.getString("process_cpu_scope").contains("all app threads"));
        assertTrue(summary.getString("process_cpu_scope").contains("energy"));
        assertEquals(0, FullStreamingBenchmark.processCpuDeltaMillis(19, 19), 0);
        assertEquals(12, FullStreamingBenchmark.processCpuDeltaMillis(19, 31), 0);
        assertThrows(IOException.class, () -> FullStreamingBenchmark.processCpuDeltaMillis(-1, 2));
        assertThrows(IOException.class, () -> FullStreamingBenchmark.processCpuDeltaMillis(5, 4));
        for (double invalid : new double[]{-1, Double.NaN, Double.POSITIVE_INFINITY}) {
            assertThrows(IOException.class, () -> FullStreamingBenchmark.processCpuSummary(new double[]{invalid}, 0, 800));
            assertThrows(IOException.class, () -> FullStreamingBenchmark.processCpuSummary(new double[]{0}, invalid, 800));
        }
        assertThrows(IOException.class, () -> FullStreamingBenchmark.processCpuSummary(new double[]{Double.MAX_VALUE, Double.MAX_VALUE}, 0, 800));
        assertThrows(IOException.class, () -> FullStreamingBenchmark.processCpuSummary(new double[0], 0, 800));
    }

    @Test public void rtfIncludesFlushAndUsesRealInputDuration() throws Exception {
        JSONObject summary = FullStreamingBenchmark.timingSummary(new double[]{10, 100, 120}, 20, 800);
        assertEquals(250, summary.getDouble("processing_wall_ms"), 0);
        assertEquals(2.5, summary.getDouble("offline_processing_rtf"), 1e-12);
        assertEquals(120, summary.getDouble("p95_call_wall_ms"), 0);
        assertEquals(1, summary.getInt("calls_over_100ms"));
    }

    @Test public void comparisonUsesUnclippedWaveformsAndDoesNotApproveQuality() throws Exception {
        JSONObject comparison = FullStreamingBenchmark.audioComparison(
                new float[]{.5f, -.5f, .25f, -.25f}, new float[]{1, -1, .5f, -.5f});
        assertEquals(.5, comparison.getDouble("normalized_rmse"), 1e-12);
        assertEquals(6.020599913279624, comparison.getDouble("snr_db"), 1e-10);
        assertEquals(.5, comparison.getDouble("max_abs"), 0);
        assertFalse(comparison.getBoolean("audio_quality_verified"));
        assertEquals("infinite_equal_outputs", FullStreamingBenchmark.audioComparison(new float[]{1}, new float[]{1}).getString("snr_db"));
        try {
            FullStreamingBenchmark.audioComparison(new float[]{Float.NaN}, new float[]{0});
            fail("Non-finite waveform should fail");
        } catch (IOException expected) { }
    }

    @Test public void groupedGateRejectsMultiplePartitionsAndCpuFallback() throws Exception {
        JSONArray profile = new JSONArray();
        for (int i = 0; i < 25; i++) {
            profile.put(new JSONObject().put("cat", "Session").put("name", "model_run"));
            profile.put(kernel("qnn-partition_kernel_time", "QNNExecutionProvider"));
        }
        assertTrue(FullStreamingBenchmark.strictGroupedProfile(FullStreamingBenchmark.profileSummary(profile.toString()), 25));
        profile.put(kernel("cpu-op_kernel_time", "CPUExecutionProvider"));
        assertFalse(FullStreamingBenchmark.strictGroupedProfile(FullStreamingBenchmark.profileSummary(profile.toString()), 25));
        profile.remove(profile.length() - 1);
        profile.put(kernel("other-qnn-partition_kernel_time", "QNNExecutionProvider"));
        assertFalse(FullStreamingBenchmark.strictGroupedProfile(FullStreamingBenchmark.profileSummary(profile.toString()), 25));
    }

    @Test public void timingProtocolAcceptsOnlyTheThreeWarmupProfileCalls() throws Exception {
        FullStreamingBenchmark.requireWarmupCallCount(new JSONObject().put("model_run_events", 3));
        for (int count : new int[]{0, 2, 4, 69}) {
            try {
                FullStreamingBenchmark.requireWarmupCallCount(new JSONObject().put("model_run_events", count));
                fail("Measured calls must not appear in the warm-up-only profile: " + count);
            } catch (IOException expected) { }
        }
    }

    @Test public void multiplePrerequisitesAndOriginalSingleCaseAcceptAnyOutcomeOrder() throws Exception {
        String profile = validGroupedProfile();
        for (int count : new int[]{1, 9, 16}) {
            JSONArray fixtures = fixtureIds(count), outcomes = new JSONArray();
            Map<String, String> profiles = new LinkedHashMap<>();
            for (int i = count - 1; i >= 0; i--) {
                outcomes.put(outcome("conv_" + i));
                profiles.put("conv_" + i, profile);
            }
            JSONObject gate = FullStreamingBenchmark.evaluateGroupedPrerequisites(fixtures, outcomes, profiles);
            assertTrue(gate.getBoolean("verified"));
            assertEquals(count, gate.getInt("case_count"));
            assertEquals(count, gate.getInt("verified_case_count"));
            assertEquals("conv_0", gate.getJSONArray("cases").getJSONObject(0).getString("id"));
        }
        assertThrows(IOException.class, () -> FullStreamingBenchmark.matchGroupedOutcomes(fixtureIds(17), fixtureIds(17)));
        assertThrows(IOException.class, () -> FullStreamingBenchmark.matchGroupedOutcomes(new JSONArray(), new JSONArray()));
    }

    @Test public void prerequisiteIdsMustBeUniqueCompleteAndExactlyMatch() throws Exception {
        JSONArray fixtures = fixtureIds(9);
        assertThrows(IOException.class, () -> FullStreamingBenchmark.matchGroupedOutcomes(fixtures, fixtureIds(8)));
        JSONArray mismatch = fixtureIds(9);
        mismatch.put(8, new JSONObject().put("id", "unexpected_conv"));
        assertThrows(IOException.class, () -> FullStreamingBenchmark.matchGroupedOutcomes(fixtures, mismatch));
        JSONArray duplicateOutcomes = fixtureIds(9);
        duplicateOutcomes.put(8, new JSONObject().put("id", "conv_0"));
        assertThrows(IOException.class, () -> FullStreamingBenchmark.matchGroupedOutcomes(fixtures, duplicateOutcomes));
        JSONArray duplicateFixtures = fixtureIds(9);
        duplicateFixtures.put(8, new JSONObject().put("id", "conv_0"));
        assertThrows(IOException.class, () -> FullStreamingBenchmark.matchGroupedOutcomes(duplicateFixtures, fixtureIds(9)));
    }

    @Test public void oneFailedPrerequisiteRejectsWholeHybridButRecordsEveryCase() throws Exception {
        JSONArray fixtures = fixtureIds(9), outcomes = new JSONArray();
        Map<String, String> profiles = new LinkedHashMap<>();
        String profile = validGroupedProfile();
        for (int i = 0; i < 9; i++) {
            outcomes.put(outcome("conv_" + i));
            profiles.put("conv_" + i, profile);
        }
        outcomes.getJSONObject(4).getJSONObject("qnn_htp").put("correctness_passed", false);
        JSONObject gate = FullStreamingBenchmark.evaluateGroupedPrerequisites(fixtures, outcomes, profiles);
        assertFalse(gate.getBoolean("verified"));
        assertEquals(8, gate.getInt("verified_case_count"));
        assertEquals(9, gate.getJSONArray("cases").length());
        assertFalse(gate.getJSONArray("cases").getJSONObject(4).getBoolean("correctness_passed"));
        assertTrue(gate.getJSONArray("cases").getJSONObject(8).getBoolean("verified"));
        outcomes.getJSONObject(4).getJSONObject("qnn_htp").put("correctness_passed", true);
        profiles.remove("conv_8");
        gate = FullStreamingBenchmark.evaluateGroupedPrerequisites(fixtures, outcomes, profiles);
        assertFalse(gate.getBoolean("verified"));
        assertEquals(8, gate.getInt("verified_case_count"));
        assertEquals(9, gate.getJSONArray("cases").length());
        assertTrue(gate.getJSONArray("cases").getJSONObject(8).getString("error").contains("missing or unreadable"));
    }

    @Test public void savedWavReportsClippingWhileFloatFilePreservesOriginals() throws Exception {
        Path directory = Files.createTempDirectory("reuse-full-wav-test");
        try {
            float[] input = {-2, -1, 0, 1, 2};
            JSONObject saved = FullStreamingBenchmark.saveAudio(directory, "public-test", input);
            assertEquals(2, saved.getInt("float_samples_outside_unit_range"));
            assertEquals(3, saved.getInt("pcm16_saturated_samples"));
            ByteBuffer wav = ByteBuffer.wrap(Files.readAllBytes(directory.resolve("public-test.wav"))).order(ByteOrder.LITTLE_ENDIAN);
            assertEquals(54, wav.capacity());
            assertEquals(8000, wav.getInt(24));
            assertEquals(10, wav.getInt(40));
            short[] expected = {-32768, -32768, 0, 32767, 32767};
            for (int i = 0; i < expected.length; i++) assertEquals(expected[i], wav.getShort(44 + i * 2));
            ByteBuffer raw = ByteBuffer.wrap(Files.readAllBytes(directory.resolve("public-test.f32"))).order(ByteOrder.LITTLE_ENDIAN);
            for (float value : input) assertEquals(value, raw.getFloat(), 0);
        } finally {
            Files.deleteIfExists(directory.resolve("public-test.wav"));
            Files.deleteIfExists(directory.resolve("public-test.f32"));
            Files.deleteIfExists(directory);
        }
    }

    @Test public void assetPathsPermitOnlyFlatFilesOrOneNamedSubdirectory() {
        assertTrue(FullStreamingBenchmark.safeAssetPath("full-baseline/model.onnx"));
        assertTrue(FullStreamingBenchmark.safeAssetPath("public-input-8k.f32"));
        for (String unsafe : new String[]{"../model.onnx", "/model.onnx", "full/model/extra.onnx", "full\\model.onnx", "./model.onnx"})
            assertFalse(unsafe, FullStreamingBenchmark.safeAssetPath(unsafe));
    }

    private static JSONObject kernel(String name, String provider) throws Exception {
        return new JSONObject().put("cat", "Node").put("name", name)
                .put("args", new JSONObject().put("provider", provider));
    }

    private static JSONArray fixtureIds(int count) throws Exception {
        JSONArray values = new JSONArray();
        for (int i = 0; i < count; i++) values.put(new JSONObject().put("id", "conv_" + i));
        return values;
    }

    private static JSONObject outcome(String id) throws Exception {
        return new JSONObject().put("id", id).put("qnn_htp", new JSONObject()
                .put("execution_verified", true).put("correctness_passed", true).put("cpu_ep_fallback_disabled", true));
    }

    private static String validGroupedProfile() throws Exception {
        JSONArray events = new JSONArray();
        for (int i = 0; i < 25; i++) {
            events.put(new JSONObject().put("cat", "Session").put("name", "model_run"));
            events.put(kernel("one_qnn_partition_kernel_time", "QNNExecutionProvider"));
        }
        return events.toString();
    }
}
