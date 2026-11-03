package io.github.reuse.phone;

import org.junit.Test;
import static org.junit.Assert.*;

public final class GpuAvailabilityTest {
    private static final String RUNTIME = "ORT 1.23.2, REUSE_NNAPI_GPU_ONLY_V1";

    @Test public void unsupportedRoutesAreNotOfferedAsGpu() {
        assertFalse(GpuAvailability.fromDevices(RUNTIME, new String[0]).available);
        assertFalse(GpuAvailability.fromDevices(RUNTIME, new String[]{"2\tCPU", "4\tGPU-named accelerator"}).available);
        assertFalse(GpuAvailability.fromDevices("stock ORT", new String[]{"3\tGPU"}).available);
    }

    @Test public void gpuTypeIsRequiredAndResultOwnsItsDescriptors() {
        String[] input = {"2\tCPU", "3\tGPU-driver"};
        GpuAvailability result = GpuAvailability.fromDevices(RUNTIME, input);
        assertTrue(result.available);
        input[1] = "4\tchanged";
        result.deviceDescriptions()[1] = "4\tchanged-again";
        assertEquals("3\tGPU-driver", result.deviceDescriptions()[1]);
    }
}
