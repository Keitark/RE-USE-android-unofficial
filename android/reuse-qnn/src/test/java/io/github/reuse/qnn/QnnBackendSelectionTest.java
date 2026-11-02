package io.github.reuse.qnn;

import static org.junit.Assert.*;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.Test;

public final class QnnBackendSelectionTest {
    @Test public void htpStillRequiresAnActualNpuDiscoveryDevice() {
        assertEquals(1, QnnHtpBackend.preferredDeviceIndex(List.of("GPU", "NPU"), false));
        assertEquals(-1, QnnHtpBackend.preferredDeviceIndex(List.of("GPU"), false));
    }

    @Test public void gpuPrefersGpuAndRetainsExplicitTokenFallback() {
        assertEquals(1, QnnHtpBackend.preferredDeviceIndex(List.of("NPU", "GPU"), true));
        assertEquals(0, QnnHtpBackend.preferredDeviceIndex(List.of("NPU"), true));
        assertEquals(-1, QnnHtpBackend.preferredDeviceIndex(List.of(), true));
    }

    @Test public void gpuOptionsCannotAccidentallyConfigureHtpOrFp16() {
        Path library = Path.of("/native/libQnnGpu.so");
        Map<String, String> options = QnnHtpBackend.providerOptions(library, true, "balanced");
        assertEquals(Map.of("backend_path", library.toString()), options);
        assertThrows(UnsupportedOperationException.class, () -> options.put("htp_performance_mode", "burst"));
    }

    @Test public void htpPerformanceModeIsExplicitAndBounded() {
        Map<String, String> options = QnnHtpBackend.providerOptions(Path.of("/native/libQnnHtp.so"), false, "burst");
        assertEquals("burst", options.get("htp_performance_mode"));
        assertEquals("0", options.get("enable_htp_fp16_precision"));
        assertThrows(IllegalArgumentException.class, () -> QnnHtpBackend.validatePerformanceMode(null));
        assertThrows(IllegalArgumentException.class, () -> QnnHtpBackend.validatePerformanceMode("fastest"));
    }
}
