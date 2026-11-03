package io.github.reuse.audio.qnn;

import static org.junit.Assert.*;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;
import org.junit.Test;

public final class QnnWarmupPolicyTest {
    @Test public void verifiedSinglePartitionHybridWarmupAllowsReadiness() throws Exception {
        QnnWarmupPolicy.requireVerifiedHybrid(evidence(), 3);
    }

    @Test public void loadingOrDeviceSelectionWithoutExecutedProofDoesNotAllowReadiness() {
        Map<String, Object> data = evidence();
        data.put("qnn_execution_verified", false);
        data.put("npu_device_selected", true);
        assertThrows(IOException.class, () -> QnnWarmupPolicy.requireVerifiedHybrid(data, 3));
    }

    @Test public void gateRejectsMissingCpuWorkAndFallbackPolicy() {
        for (String key : new String[]{"cpu_fallback_allowed"}) {
            Map<String, Object> data = evidence(); data.put(key, false);
            assertThrows(IOException.class, () -> QnnWarmupPolicy.requireVerifiedHybrid(data, 3));
        }
        Map<String, Object> data = evidence(); data.put("cpu_kernel_events", 0);
        assertThrows(IOException.class, () -> QnnWarmupPolicy.requireVerifiedHybrid(data, 3));
    }

    @Test public void gateRejectsWrongCallCountOrUnknownProviders() {
        for (String key : new String[]{"expected_calls", "qnn_kernel_events", "unique_qnn_kernel_count",
                "other_provider_kernel_events", "missing_provider_kernel_events"}) {
            Map<String, Object> data = evidence(); data.put(key, 2);
            assertThrows(IOException.class, () -> QnnWarmupPolicy.requireVerifiedHybrid(data, 3));
        }
    }

    @Test public void balancedMultiplePartitionsAllowReadinessWithoutSinglePartitionProof() throws Exception {
        Map<String, Object> data = evidence();
        Map<String, Integer> histogram = new LinkedHashMap<>();
        for (int i = 0; i < 12; i++) histogram.put("partition_" + i + "_kernel_time", 3);
        data.put("single_qnn_partition_verified", false);
        data.put("qnn_kernel_events", 36);
        data.put("unique_qnn_kernel_count", 12);
        data.put("qnn_kernel_event_counts", histogram);
        QnnWarmupPolicy.requireVerifiedHybrid(data, 3);
    }

    @Test public void correctTotalDoesNotHideSkippedOrDuplicatedPartitionExecutions() {
        Map<String, Object> data = evidence();
        Map<String, Integer> histogram = new LinkedHashMap<>();
        histogram.put("part_a_kernel_time", 2); histogram.put("part_b_kernel_time", 4);
        data.put("qnn_kernel_events", 6); data.put("unique_qnn_kernel_count", 2);
        data.put("qnn_kernel_event_counts", histogram);
        assertThrows(IOException.class, () -> QnnWarmupPolicy.requireVerifiedHybrid(data, 3));
        data.remove("qnn_kernel_event_counts");
        assertThrows(IOException.class, () -> QnnWarmupPolicy.requireVerifiedHybrid(data, 3));
    }

    @Test public void missingAndNonIntegerEventCountsFailClosed() {
        Map<String, Object> missing = evidence(); missing.remove("missing_provider_kernel_events");
        assertThrows(IOException.class, () -> QnnWarmupPolicy.requireVerifiedHybrid(missing, 3));
        Map<String, Object> nonInteger = evidence(); nonInteger.put("other_provider_kernel_events", .5);
        assertThrows(IOException.class, () -> QnnWarmupPolicy.requireVerifiedHybrid(nonInteger, 3));
        assertThrows(IllegalArgumentException.class, () -> QnnWarmupPolicy.requireVerifiedHybrid(evidence(), 0));
    }

    private static Map<String, Object> evidence() {
        Map<String, Object> data = new LinkedHashMap<>();
        data.put("qnn_execution_verified", true);
        data.put("single_qnn_partition_verified", true);
        data.put("cpu_fallback_allowed", true);
        data.put("expected_calls", 3);
        data.put("qnn_kernel_events", 3);
        data.put("unique_qnn_kernel_count", 1);
        Map<String, Integer> histogram = new LinkedHashMap<>();
        histogram.put("partition_kernel_time", 3);
        data.put("qnn_kernel_event_counts", histogram);
        data.put("cpu_kernel_events", 150);
        data.put("other_provider_kernel_events", 0);
        data.put("missing_provider_kernel_events", 0);
        return data;
    }
}
