package io.github.reuse.audio.qnn;

import java.io.IOException;
import java.util.Map;

/** The readiness gate is deliberately narrower than arbitrary QNN graph compatibility. */
final class QnnWarmupPolicy {
    private QnnWarmupPolicy() { }

    static void requireVerifiedHybrid(Map<String, Object> evidence, int expectedCalls) throws IOException {
        if (expectedCalls < 1) throw new IllegalArgumentException("Warm-up count must be positive");
        if (!Boolean.TRUE.equals(evidence.get("qnn_execution_verified"))
                || !Boolean.TRUE.equals(evidence.get("cpu_fallback_allowed"))
                || count(evidence, "expected_calls") != expectedCalls
                || !balancedQnnKernels(evidence, expectedCalls)
                || count(evidence, "cpu_kernel_events") <= 0
                || count(evidence, "other_provider_kernel_events") != 0
                || count(evidence, "missing_provider_kernel_events") != 0)
            throw new IOException("QNN startup requires every observed HTP kernel once per warm-up call and CPU execution for remaining computation; microphone readiness withheld");
    }

    private static boolean balancedQnnKernels(Map<String, Object> evidence, int expectedCalls) {
        Object raw = evidence.get("qnn_kernel_event_counts");
        if (!(raw instanceof Map)) return false;
        Map<?, ?> histogram = (Map<?, ?>) raw;
        if (histogram.isEmpty() || count(evidence, "unique_qnn_kernel_count") != histogram.size()
                || count(evidence, "qnn_kernel_events") != (long) expectedCalls * histogram.size()) return false;
        for (Map.Entry<?, ?> entry : histogram.entrySet()) {
            if (!(entry.getKey() instanceof String) || ((String) entry.getKey()).isEmpty()
                    || integralCount(entry.getValue()) != expectedCalls) return false;
        }
        return true;
    }

    private static long count(Map<String, Object> evidence, String key) {
        return integralCount(evidence.get(key));
    }

    private static long integralCount(Object value) {
        if (!(value instanceof Number)) return -1;
        Number number = (Number) value;
        double numeric = number.doubleValue();
        long integer = number.longValue();
        return Double.isFinite(numeric) && numeric >= 0 && numeric == integer ? integer : -1;
    }
}
