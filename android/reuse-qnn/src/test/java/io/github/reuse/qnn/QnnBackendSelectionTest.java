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

    @Test public void gpuDiagnosticsAreExplicitAndCannotAddHtpOrPrecisionOptions() {
        Path directory = Path.of("/private/diagnostics");
        Map<String, String> options = QnnHtpBackend.providerOptions(Path.of("/native/libQnnGpu.so"), true, "balanced", directory);
        assertEquals(7, options.size());
        for (String key : List.of("dump_qnn_ep_input_graph", "dump_json_qnn_graph", "enable_framework_op_trace"))
            assertEquals("1", options.get(key));
        for (String key : List.of("dump_qnn_ep_input_graph_dir", "json_qnn_graph_dir", "framework_op_trace_dir"))
            assertEquals(directory.toString(), options.get(key));
        assertFalse(options.containsKey("enable_htp_fp16_precision"));
        assertThrows(IllegalArgumentException.class, () ->
                QnnHtpBackend.providerOptions(Path.of("/native/libQnnHtp.so"), false, "balanced", directory));
        assertThrows(UnsupportedOperationException.class, () -> options.put("backend_path", "other"));
    }
}
