package io.github.reuse.phone;

import static org.junit.Assert.*;
import org.junit.Test;

public class GpuUiPolicyTest {
    @Test public void gpuCannotBeSelectedWhilePendingUnavailableOrRecording() {
        assertFalse(GpuUiPolicy.canChooseGpu(false, false, false));
        assertFalse(GpuUiPolicy.canChooseGpu(false, true, false));
        assertFalse(GpuUiPolicy.canChooseGpu(true, true, true));
        assertTrue(GpuUiPolicy.canChooseGpu(false, true, true));
    }

    @Test public void unavailableDriverErrorExplainsCpuRecoveryWithoutNativeStack() {
        String raw = "PROCESSING_ERROR: ai.onnxruntime.OrtException: F:/dev/source.cpp:74 "
                + "RE-USE GPU-only NNAPI found no GPU driver; CPU/DSP/NPU fallback is disabled";
        assertTrue(GpuUiPolicy.isUnsupportedGpu(raw));
        assertTrue(GpuUiPolicy.isUnsupportedGpu("NNAPI delegated no kernels during warm-up"));
        String friendly = GpuUiPolicy.readableError(raw);
        assertTrue(friendly.contains("Recording did not start"));
        assertTrue(friendly.contains("CPU"));
        assertFalse(friendly.contains("source.cpp"));
        assertFalse(GpuUiPolicy.isUnsupportedGpu("Microphone permission is required"));
        assertEquals("Microphone permission is required", GpuUiPolicy.readableError("Microphone permission is required"));
    }

    @Test public void cpuRunsPreserveUnavailableGpuStatusAndLongErrorsAreBounded() {
        assertEquals(GpuUiPolicy.UNAVAILABLE,
                GpuUiPolicy.backendStatus(true, false, GpuUiPolicy.UNAVAILABLE, false, false));
        assertTrue(GpuUiPolicy.backendStatus(false, false, "", false, false).contains("checking GPU"));
        assertTrue(GpuUiPolicy.readableError(new String(new char[500]).replace('\0', 'x')).length() < 240);
    }
}
