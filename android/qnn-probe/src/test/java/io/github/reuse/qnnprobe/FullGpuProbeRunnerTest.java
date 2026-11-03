package io.github.reuse.qnnprobe;

import static org.junit.Assert.*;
import java.io.IOException;
import java.util.concurrent.CancellationException;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;

/** Pure admission gates; these tests make no GPU or Android-device execution claim. */
public final class FullGpuProbeRunnerTest {
    private static final String CANDIDATE = "1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef";

    @Test public void validatedPortableGraphKeepsOriginalSettingsAndAllStateBindings() throws Exception {
        JSONObject baseline = manifest(false), portable = manifest(true);
        FullGpuProbeRunner.validateExportContract(validation(), portable, baseline, CANDIDATE);
        portable.getJSONArray("states").getJSONObject(5).getJSONArray("shape").put(1, 128);
        assertThrows(IOException.class, () -> FullGpuProbeRunner.validateExportContract(validation(), portable, baseline, CANDIDATE));
    }

    @Test public void customOpsQuantizationAndMismatchedHostValidationCannotEnterStrictProbe() throws Exception {
        JSONObject baseline = manifest(false);
        for (int failure = 0; failure < 7; failure++) {
            JSONObject portable = manifest(true), validation = validation();
            switch (failure) {
                case 0: portable.getJSONArray("custom_ops").put(new JSONObject().put("name", "TemporalConvSilu")); break;
                case 1: validation.put("quantization_present", true); break;
                case 2: validation.put("source_graph_sha256", CANDIDATE); break;
                case 3: validation.put("source_checkpoint_sha256", CANDIDATE); break;
                case 4: validation.put("model_sha256", FullGpuProbeRunner.BASELINE_SHA); break;
                case 5: validation.put("host_validation_passed", false); break;
                default: validation.put("static_float32_io", false);
            }
            assertThrows("Failure " + failure, IOException.class,
                    () -> FullGpuProbeRunner.validateExportContract(validation, portable, baseline, CANDIDATE));
        }
    }

    @Test public void hostGateCannotLoosenThePredeterminedWaveformLimitsOrHideBadErrors() throws Exception {
        for (int failure = 0; failure < 4; failure++) {
            JSONObject validation = validation(), portable = manifest(true), baseline = manifest(false);
            JSONObject wave = validation.getJSONObject("waveform_comparison");
            switch (failure) {
                case 0: wave.put("max_abs_limit", .1); break;
                case 1: wave.put("normalized_rmse_limit", .05); break;
                case 2: wave.put("max_abs", .00101); break;
                default: wave.put("normalized_rmse", .00501);
            }
            assertThrows(IOException.class,
                    () -> FullGpuProbeRunner.validateExportContract(validation, portable, baseline, CANDIDATE));
        }
    }

    @Test public void changedLookaheadSampleRateAndRecurrentPrecisionAreRejected() throws Exception {
        for (int failure = 0; failure < 4; failure++) {
            JSONObject portable = manifest(true), baseline = manifest(false), validation = validation();
            switch (failure) {
                case 0: portable.put("sample_rate", 16000); break;
                case 1: portable.put("look_ahead_frames", 1); break;
                case 2: portable.getJSONArray("states").getJSONObject(2).put("dtype", "float16"); break;
                default: portable.getJSONArray("states").getJSONObject(2).put("output", "next_state_003");
            }
            assertThrows(IOException.class,
                    () -> FullGpuProbeRunner.validateExportContract(validation, portable, baseline, CANDIDATE));
        }
    }

    @Test public void strictProviderGateRejectsAnyCpuUnknownProviderOrUnbalancedWarmup() throws Exception {
        assertTrue(FullGpuProbeRunner.strictProviderPass(summary(), verification()));
        for (String key : new String[]{"cpu_kernel_events", "other_provider_kernel_events", "missing_provider_kernel_events"}) {
            JSONObject summary = summary().put(key, 1);
            assertFalse(FullGpuProbeRunner.strictProviderPass(summary, verification()));
        }
        assertFalse(FullGpuProbeRunner.strictProviderPass(summary().put("model_run_events", 2), verification()));
        assertFalse(FullGpuProbeRunner.strictProviderPass(summary(), verification().put("cpu_fallback_allowed", true)));
        assertFalse(FullGpuProbeRunner.strictProviderPass(summary(), verification().put("qnn_kernel_counts_match_expected_calls", false)));
        assertFalse(FullGpuProbeRunner.strictProviderPass(summary(), verification().put("expected_calls", 2)));
    }

    @Test public void wholeGateNeedsGpuIdentityProviderEvidenceCompleteChronologyAndNumericsTogether() throws Exception {
        JSONObject cpu = complete().put("status", "VERIFIED_CPU_REFERENCE"), gpu = gpu();
        JSONObject comparison = comparison();
        assertTrue(FullGpuProbeRunner.acceptsWholeRun(cpu, gpu, comparison));
        for (int failure = 0; failure < 8; failure++) {
            JSONObject invalid = gpu();
            switch (failure) {
                case 0: invalid.put("completed_audio_calls", 64); break;
                case 1: invalid.put("flush_calls", 0); break;
                case 2: invalid.put("all_output_samples_finite", false); break;
                case 3: invalid.put("cpu_ep_fallback_disabled", false); break;
                case 4: invalid.getJSONObject("loaded_backend_evidence").put("gpu_backend_identity_verified", false); break;
                case 5: invalid.getJSONObject("loaded_backend_evidence").put("capture_complete_for_identity", false); break;
                case 6: invalid.getJSONObject("warmup_provider_evidence").put("cpu_kernel_events", 1); break;
                default: invalid.put("all_streaming_state_reset_after_warmup", false);
            }
            assertFalse("Failure " + failure, FullGpuProbeRunner.acceptsWholeRun(cpu, invalid, comparison));
        }
        assertFalse(FullGpuProbeRunner.acceptsWholeRun(cpu, gpu, comparison().put("max_abs", .002)));
        assertFalse(FullGpuProbeRunner.acceptsWholeRun(cpu, gpu, comparison().put("normalized_rmse", .006)));
        assertFalse(FullGpuProbeRunner.acceptsWholeRun(cpu, gpu, comparison().put("fixed_numerical_gate_passed", false)));
    }

    @Test public void completeSampleAccountingIncludesStartupAndOneFlushWithoutDroppingPadding() throws Exception {
        int samples = 0;
        for (int call = 0; call < FullGpuProbeRunner.CALLS; call++) samples += FullGpuProbeRunner.expectedAudioOutput(call);
        assertEquals(51840, samples);
        assertEquals(52000, samples + 160);
        assertTrue(FullGpuProbeRunner.completeAccounting(complete()));
        assertFalse(FullGpuProbeRunner.completeAccounting(complete().put("untrimmed_output_samples", 51840)));
        assertFalse(FullGpuProbeRunner.completeAccounting(complete().put("input_zero_padding_samples", 0)));
        assertThrows(IllegalArgumentException.class, () -> FullGpuProbeRunner.expectedAudioOutput(-1));
        assertThrows(IllegalArgumentException.class, () -> FullGpuProbeRunner.expectedAudioOutput(65));
    }

    @Test public void cancellationIsCheckedBeforeNativeWorkAndBudgetIsCooperative() {
        FullGpuProbeRunner.checkContinue(() -> false, System.nanoTime() + 1_000_000_000L);
        assertThrows(CancellationException.class, () -> FullGpuProbeRunner.checkContinue(() -> true, Long.MAX_VALUE));
        assertThrows(RuntimeException.class, () -> FullGpuProbeRunner.checkContinue(() -> false, System.nanoTime() - 1));
    }

    private static JSONObject manifest(boolean portable) throws Exception {
        JSONObject result = new JSONObject().put("format_version", 1).put("sample_rate", 8000)
                .put("fft_size", 320).put("hop_size", 160).put("frames_per_call", 5).put("chunk_size", 800)
                .put("exit_layers", 3).put("look_ahead_frames", 0).put("scan_mode", portable ? "parallel" : "native")
                .put("source_repository", "nvidia/Real-time_RE-USE")
                .put("source_revision", "4c0c186272ca853cd18d832fc5c28777b62f32d1")
                .put("source_checkpoint_sha256", FullGpuProbeRunner.CHECKPOINT_SHA)
                .put("model_sha256", portable ? CANDIDATE : FullGpuProbeRunner.BASELINE_SHA).put("custom_ops", new JSONArray());
        JSONArray states = new JSONArray();
        for (int i = 0; i < 21; i++) {
            String input = String.format(java.util.Locale.ROOT, "state_%03d", i);
            states.put(new JSONObject().put("input", input).put("output", "next_" + input).put("dtype", "float32")
                    .put("shape", new JSONArray().put(81).put(256).put(4)));
        }
        return result.put("states", states);
    }

    private static JSONObject validation() throws Exception {
        return new JSONObject().put("schema_version", 1).put("host_validation_passed", true)
                .put("custom_node_count", 0).put("standard_domain_only", true).put("static_float32_io", true)
                .put("state_abi_matches_baseline", true).put("quantization_present", false)
                .put("source_graph_sha256", FullGpuProbeRunner.BASELINE_SHA)
                .put("source_checkpoint_sha256", FullGpuProbeRunner.CHECKPOINT_SHA).put("model_sha256", CANDIDATE)
                .put("input_samples", 51840).put("audio_calls", 65).put("flush_calls", 1)
                .put("waveform_comparison", comparison().put("max_abs_limit", .001).put("normalized_rmse_limit", .005).put("passed", true));
    }

    private static JSONObject complete() throws Exception {
        return new JSONObject().put("completed_audio_calls", 65).put("flush_calls", 1).put("input_samples", 51840)
                .put("trimmed_output_samples", 51840).put("untrimmed_audio_output_samples", 51840)
                .put("untrimmed_output_samples", 52000).put("flush_output_samples", 160).put("input_zero_padding_samples", 160)
                .put("all_output_samples_finite", true).put("complete_sequence_verified", true)
                .put("all_streaming_state_reset_after_warmup", true);
    }

    private static JSONObject summary() throws Exception {
        return new JSONObject().put("model_run_events", 3).put("qnn_kernel_events", 3)
                .put("cpu_kernel_events", 0).put("other_provider_kernel_events", 0).put("missing_provider_kernel_events", 0);
    }

    private static JSONObject verification() throws Exception {
        return new JSONObject().put("qnn_execution_verified", true).put("qnn_only_kernel_execution_verified", true)
                .put("qnn_kernel_counts_match_expected_calls", true).put("expected_calls", 3).put("cpu_fallback_allowed", false);
    }

    private static JSONObject gpu() throws Exception {
        return complete().put("cpu_ep_fallback_disabled", true).put("warmup_provider_verified", true)
                .put("warmup_provider_evidence", summary()).put("backend_configuration", verification())
                .put("loaded_backend_evidence", new JSONObject().put("gpu_backend_identity_verified", true).put("capture_complete_for_identity", true));
    }

    private static JSONObject comparison() throws Exception {
        return new JSONObject().put("max_abs", .00001).put("normalized_rmse", .0001).put("fixed_numerical_gate_passed", true);
    }
}
