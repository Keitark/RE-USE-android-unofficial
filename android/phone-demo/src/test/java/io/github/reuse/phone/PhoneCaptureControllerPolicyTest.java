package io.github.reuse.phone;

import static org.junit.Assert.*;
import io.github.reuse.live.EnhancementSession;
import io.github.reuse.RecommendedCpuProfile;
import java.nio.file.Path;
import java.util.Map;
import org.junit.Test;
import org.json.JSONObject;

public final class PhoneCaptureControllerPolicyTest {
    @Test public void cpuAndExistingGpuRequestsKeepTheirOriginalModes() {
        PhoneCaptureController.validateRequestedModes(2, false, false, false, false, false, false);
        PhoneCaptureController.validateRequestedModes(4, false, false, false, true, false, false);
        PhoneCaptureController.validateRequestedModes(4, true, true, false, true, false, false);
        assertThrows(IllegalArgumentException.class, () ->
                PhoneCaptureController.validateRequestedModes(4, true, false, false, false, false, false));
    }

    @Test public void qnnRequiresBothTheOptInBuildAndItsSessionFactory() {
        PhoneCaptureController.validateRequestedModes(2, false, false, true, false, true, true);
        assertThrows(IllegalArgumentException.class, () ->
                PhoneCaptureController.validateRequestedModes(2, false, false, true, false, false, true));
        assertThrows(IllegalArgumentException.class, () ->
                PhoneCaptureController.validateRequestedModes(2, false, false, true, false, true, false));
    }

    @Test public void conflictingAcceleratorsFp16AndUnsupportedThreadCountsFailBeforePreparation() {
        assertThrows(IllegalArgumentException.class, () ->
                PhoneCaptureController.validateRequestedModes(4, true, false, true, true, true, true));
        assertThrows(IllegalArgumentException.class, () ->
                PhoneCaptureController.validateRequestedModes(4, false, true, true, false, true, true));
        assertThrows(IllegalArgumentException.class, () ->
                PhoneCaptureController.validateRequestedModes(3, false, false, false, false, false, false));
    }

    @Test public void quantizedCpuRequiresItsOwnFactoryAndTheCandidateBuild() {
        PhoneCaptureController.validateRequestedModes(1, false, false, false, true, false, true, false, true);
        PhoneCaptureController.validateRequestedModes(4, false, false, false, false, false, true, false, false);
        assertThrows(IllegalArgumentException.class, () ->
                PhoneCaptureController.validateRequestedModes(1, false, false, false, true, false, false, false, true));
        assertThrows(IllegalArgumentException.class, () ->
                PhoneCaptureController.validateRequestedModes(1, false, false, false, true, false, true, true, false));
    }

    @Test public void rejectedQuantizedThreadCountsCannotReachModelPreparation() {
        for (int threads : new int[]{2, 3, 4, 8}) {
            IllegalArgumentException error = assertThrows(IllegalArgumentException.class, () ->
                    PhoneCaptureController.validateRequestedModes(threads, false, false, false, true,
                            false, true, false, true));
            assertTrue(error.getMessage().contains("requires 1 thread"));
        }
        assertThrows(IllegalArgumentException.class, () ->
                PhoneCaptureController.validateRequestedModes(1, false, false, false, false, false, false));
    }

    @Test public void quantizedCpuCannotBeCombinedWithGpuQnnOrFp16() {
        assertThrows(IllegalArgumentException.class, () ->
                PhoneCaptureController.validateRequestedModes(1, true, false, false, true, true, true, true, true));
        assertThrows(IllegalArgumentException.class, () ->
                PhoneCaptureController.validateRequestedModes(1, false, false, true, true, true, true, true, true));
        assertThrows(IllegalArgumentException.class, () ->
                PhoneCaptureController.validateRequestedModes(1, false, true, false, true, false, true, false, true));
    }

    @Test public void cpuProfileIdentityPreservesActualRuntimeAndNativeBuildWithoutVerificationClaim() {
        Map<String, Object> identity = PhoneCaptureController.processingIdentity(false, false, false, 4,
                "actual-runtime", "actual-build", "{\"architecture\":\"x86_64\",\"neon_fma_compiled\":false}", "");
        assertEquals(RecommendedCpuProfile.ID, identity.get("recommended_cpu_profile_id"));
        assertEquals(4, identity.get("recommended_cpu_default_threads"));
        assertEquals(Boolean.TRUE, identity.get("recommended_cpu_thread_setting_selected"));
        assertEquals("actual-runtime", identity.get("ort_version"));
        assertEquals("actual-build", identity.get("ort_runtime_build_info"));
        JSONObject kernel = (JSONObject) identity.get("native_kernel_build_info");
        assertEquals("x86_64", kernel.optString("architecture"));
        assertFalse(kernel.optBoolean("neon_fma_compiled", true));
        assertFalse(identity.containsKey("gpu_execution_verified"));
        assertFalse(identity.containsKey("realtime_verified"));
    }

    @Test public void partialStartupIdentityAndExperimentalModesCannotClaimRecommendedDefaults() {
        Map<String, Object> partial = PhoneCaptureController.processingIdentity(false, false, false, 2,
                "not_read", "not_read", "not_read", "native library unavailable");
        assertEquals("not_read", partial.get("ort_version"));
        assertEquals("not_read", partial.get("native_kernel_build_info"));
        assertEquals("native library unavailable", partial.get("runtime_provenance_error"));
        assertEquals(Boolean.FALSE, partial.get("recommended_cpu_thread_setting_selected"));
        for (int backend = 0; backend < 3; backend++) {
            Map<String, Object> experiment = PhoneCaptureController.processingIdentity(
                    backend == 0, backend == 1, backend == 2, backend == 2 ? 1 : 4,
                    "version", "build", "not_read", "");
            assertFalse(experiment.containsKey("recommended_cpu_profile_id"));
        }
    }

    @Test public void candidatePrecisionDescriptionDoesNotClaimNpuExecution() {
        String candidate = PhoneCaptureController.modelPrecisionLabel(true, false);
        assertTrue(candidate.contains("QDQ"));
        assertTrue(candidate.contains("FP32"));
        assertFalse(candidate.contains("NPU"));
        assertEquals("FP32", PhoneCaptureController.modelPrecisionLabel(false, false));
        assertTrue(PhoneCaptureController.modelPrecisionLabel(false, true).contains("NNAPI FP16 relaxation requested"));
    }

    @Test public void preparedSessionRetainsActualCandidateIdentityWithoutStartingTheEngine() {
        try (EnhancementSession session = unstartedSession()) {
            String uppercase = "A1".repeat(32);
            PhoneCaptureController.PreparedSession prepared = new PhoneCaptureController.PreparedSession(session, uppercase);
            assertSame(session, prepared.session);
            assertEquals(uppercase.toLowerCase(java.util.Locale.ROOT), prepared.modelSha256);
            assertEquals("NEW", session.statistics().get("state"));
        }
    }

    @Test public void malformedPreparedIdentityAndNullSessionAreRejected() {
        try (EnhancementSession session = unstartedSession()) {
            assertThrows(IllegalArgumentException.class, () -> new PhoneCaptureController.PreparedSession(null, "a".repeat(64)));
            assertThrows(IllegalArgumentException.class, () -> new PhoneCaptureController.PreparedSession(session, null));
            assertThrows(IllegalArgumentException.class, () -> new PhoneCaptureController.PreparedSession(session, "a".repeat(63)));
            assertThrows(IllegalArgumentException.class, () -> new PhoneCaptureController.PreparedSession(session, "g".repeat(64)));
        }
    }

    private static EnhancementSession unstartedSession() {
        return EnhancementSession.withEngine(Path.of("unused-test-output"), new EnhancementSession.Listener() {
            public void onReady() { throw new AssertionError("Test session must remain unstarted"); }
            public void onAudio(byte[] pcm) { throw new AssertionError("Test session must remain unstarted"); }
            public void onFinished(Map<String, Object> stats) { throw new AssertionError("Test session must remain unstarted"); }
            public void onError(String code, String message, Map<String, Object> stats) { throw new AssertionError("Test session must remain unstarted"); }
        }, () -> { }, () -> { throw new AssertionError("Prepared-session validation started a model"); });
    }
}
