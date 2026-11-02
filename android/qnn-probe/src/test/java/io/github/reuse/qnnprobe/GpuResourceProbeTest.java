package io.github.reuse.qnnprobe;

import static org.junit.Assert.*;
import java.io.IOException;
import org.json.JSONArray;
import org.json.JSONObject;
import org.junit.Test;

public final class GpuResourceProbeTest {
    @Test public void backendEvidenceRequiresOwnBoundedSessionAndActualLoadedGpuIdentity() throws Exception {
        String good = "I/test: marker BEGIN\nI/onnxruntime: Found valid interface, version: 2.40.0 backend provider name: GPU_QTI_AISW backend id: 4\nI/test: marker END\n";
        JSONObject evidence = GpuResourceProbe.backendLogEvidence(good, "marker");
        assertTrue(evidence.getBoolean("gpu_backend_identity_verified"));
        assertEquals("GPU_QTI_AISW", evidence.getString("last_backend_provider_name"));
        assertFalse(GpuResourceProbe.backendLogEvidence(good, "other-session").getBoolean("gpu_backend_identity_verified"));
        assertFalse(GpuResourceProbe.backendLogEvidence(good.replace("marker END", "missing END"), "marker").getBoolean("gpu_backend_identity_verified"));
    }

    @Test public void staleGpuBeforeMarkerOrHtpLoadedLastCannotVerifyGpu() throws Exception {
        String gpu = "Found valid interface, version: 2.40.0 backend provider name: GPU_QTI_AISW backend id: 4\n";
        String htp = "Found valid interface, version: 2.40.0 backend provider name: HTP_QTI_AISW backend id: 6\n";
        assertFalse(GpuResourceProbe.backendLogEvidence(gpu + "s BEGIN\n" + htp + "s END", "s").getBoolean("gpu_backend_identity_verified"));
        assertFalse(GpuResourceProbe.backendLogEvidence("s BEGIN\n" + gpu + htp + "s END", "s").getBoolean("gpu_backend_identity_verified"));
        assertTrue(GpuResourceProbe.backendLogEvidence("s BEGIN\n" + htp + gpu + "s END", "s").getBoolean("gpu_backend_identity_verified"));
    }

    @Test public void intendedSaverIdentityOrUnrelatedGpuWordsAreInsufficient() throws Exception {
        for (String text : new String[]{
                "Intended backend provider name: GPU_QTI_AISW backend id: 4",
                "Found valid interface, version: 1 backend provider name: SaverGpu backend id: 2",
                "QNNExecutionProvider executed GPU model",
                "Found valid interface, version: 1 backend provider name: GPU_QTI_AISW backend id: 0",
                "Found valid interface, version: 1 backend provider name: GPU_QTI_AISW backend id: 2",
                "Found valid interface, version: 1 backend provider name: GPU_QTI_AISW backend id: 6",
                "Found valid interface, version: 1 backend provider name: OtherGpu backend id: 4"})
            assertFalse(GpuResourceProbe.backendLogEvidence("s BEGIN\n" + text + "\ns END", "s").getBoolean("gpu_backend_identity_verified"));
    }

    @Test public void wholeFp32GateRequiresBothPredeterminedLimitsAndFiniteErrors() {
        assertTrue(GpuResourceProbe.wholeNumericalPass(.001, .005));
        assertFalse(GpuResourceProbe.wholeNumericalPass(.0010001, .00001));
        assertFalse(GpuResourceProbe.wholeNumericalPass(.00001, .0050001));
        assertFalse(GpuResourceProbe.wholeNumericalPass(Double.NaN, 0));
        assertFalse(GpuResourceProbe.wholeNumericalPass(0, Double.POSITIVE_INFINITY));
        assertFalse(GpuResourceProbe.wholeNumericalPass(-1, 0));
    }

    @Test public void validBackendTextCannotOverrideIncompleteOrTruncatedCapture() throws Exception {
        String text = "s BEGIN\nFound valid interface, version: 2.40.0 backend provider name: GPU_QTI_AISW backend id: 4\ns END";
        for (int failure = 0; failure < 4; failure++) {
            JSONObject evidence = GpuResourceProbe.backendLogEvidence(text, "s");
            assertTrue(evidence.getBoolean("gpu_backend_identity_verified"));
            GpuResourceProbe.applyCaptureCompletion(evidence, failure != 0, failure != 1,
                    failure == 2, failure == 3 ? "read failed" : null);
            assertFalse(evidence.getBoolean("gpu_backend_identity_verified"));
            assertFalse(evidence.getBoolean("capture_complete_for_identity"));
        }
        JSONObject complete = GpuResourceProbe.backendLogEvidence(text, "s");
        GpuResourceProbe.applyCaptureCompletion(complete, true, true, false, null);
        assertTrue(complete.getBoolean("gpu_backend_identity_verified"));
    }

    @Test public void fixturePathsCannotEscapePackagedRoot() {
        assertTrue(GpuResourceProbe.safeAssetPath("full-baseline/model.onnx"));
        for (String value : new String[]{"", "/absolute", "../model", "model/../secret", "model//name", "C:/model", "model\\name", "model/"})
            assertFalse(value, GpuResourceProbe.safeAssetPath(value));
    }

    @Test public void gpuPrerequisiteMustBeUniqueAndUseTheKnownHeavyConvolution() throws Exception {
        JSONObject fixture = new JSONObject().put("id", "reuse_encoder_dilated_conv_u8");
        assertSame(fixture, GpuResourceProbe.findHeavyFixture(new JSONArray().put(new JSONObject().put("id", "other")).put(fixture)));
        assertThrows(IOException.class, () -> GpuResourceProbe.findHeavyFixture(new JSONArray()));
        assertThrows(IOException.class, () -> GpuResourceProbe.findHeavyFixture(new JSONArray().put(fixture).put(fixture)));
    }
}
