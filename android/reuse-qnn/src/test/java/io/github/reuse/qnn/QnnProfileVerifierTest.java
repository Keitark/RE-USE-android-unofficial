package io.github.reuse.qnn;

import static org.junit.Assert.*;
import java.io.IOException;
import java.util.List;
import java.util.Map;
import org.junit.Test;

public final class QnnProfileVerifierTest {
    @Test public void strictSinglePartitionRequiresOneKernelPerExpectedCall() throws Exception {
        Map<String, Object> result = QnnProfileVerifier.verify(profile(qnn("part"), qnn("part")), 2, false, true);
        assertEquals(true, result.get("qnn_execution_verified"));
        assertEquals(true, result.get("single_qnn_partition_verified"));
        assertEquals(2, result.get("qnn_kernel_events"));
        assertEquals(1, result.get("unique_qnn_kernel_count"));
        assertEquals(false, QnnProfileVerifier.verify(profile(qnn("part")), 2, false, true).get("qnn_execution_verified"));
        assertEquals(false, QnnProfileVerifier.verify(profile(qnn("part"), qnn("part")), 1, false, true).get("qnn_execution_verified"));
        assertEquals(false, QnnProfileVerifier.verify(profile(qnn("partA"), qnn("partB")), 2, false, true).get("qnn_execution_verified"));
    }

    @Test public void cpuFallbackRequiresExplicitPermissionAndStillRequiresQnnExecution() throws Exception {
        String hybrid = profile(qnn("part"), cpu("scan"));
        assertEquals(false, QnnProfileVerifier.verify(hybrid, 1, false, false).get("qnn_execution_verified"));
        Map<String, Object> allowed = QnnProfileVerifier.verify(hybrid, 1, true, false);
        assertEquals(true, allowed.get("qnn_execution_verified"));
        assertEquals(true, allowed.get("cpu_fallback_observed"));
        assertEquals(false, allowed.get("qnn_only_kernel_execution_verified"));
        assertEquals(false, QnnProfileVerifier.verify(profile(cpu("scan")), 1, true, false).get("qnn_execution_verified"));
    }

    @Test public void multipleQnnPartitionsAreAcceptedOnlyWhenSinglePartitionIsNotRequired() throws Exception {
        String twoPartitions = profile(qnn("partA"), qnn("partB"));
        assertEquals(true, QnnProfileVerifier.verify(twoPartitions, 1, false, false).get("qnn_execution_verified"));
        assertEquals(false, QnnProfileVerifier.verify(twoPartitions, 1, false, true).get("qnn_execution_verified"));
    }

    @Test public void unknownOrMissingProviderAttributionFailsEvenWithCpuFallbackAllowed() throws Exception {
        assertEquals(false, QnnProfileVerifier.verify(profile(qnn("part"), node("gpu", "CUDAExecutionProvider")), 1, true, false)
                .get("qnn_execution_verified"));
        assertEquals(false, QnnProfileVerifier.verify(profile(qnn("part"), node("missing", "")), 1, true, false)
                .get("qnn_execution_verified"));
    }

    @Test public void registrationAndFenceEventsAreNotExecutionEvidence() throws Exception {
        String events = "[{\"cat\":\"Session\",\"name\":\"register_kernel_time\",\"args\":{\"provider\":\"QNNExecutionProvider\"}},"
                + "{\"cat\":\"Node\",\"name\":\"part_fence_before\",\"args\":{\"provider\":\"QNNExecutionProvider\"}}]";
        Map<String, Object> result = QnnProfileVerifier.verify(events, 1, true, false);
        assertEquals(0, result.get("qnn_kernel_events"));
        assertEquals(false, result.get("qnn_execution_verified"));
    }

    @Test public void malformedProfileAndInvalidExpectedCountAreRejected() {
        assertThrows(IOException.class, () -> QnnProfileVerifier.verify("not-json", 1, false, false));
        assertThrows(IOException.class, () -> QnnProfileVerifier.verify(null, 1, false, false));
        assertThrows(IllegalArgumentException.class, () -> QnnProfileVerifier.verify("[]", 0, false, false));
    }

    @SuppressWarnings("unchecked")
    @Test public void returnedProfileEvidenceAndCollectionsAreImmutable() throws Exception {
        Map<String, Object> result = QnnProfileVerifier.verify(profile(qnn("part")), 1, false, true);
        assertThrows(UnsupportedOperationException.class, () -> result.put("qnn_execution_verified", false));
        assertThrows(UnsupportedOperationException.class, () -> ((List<String>) result.get("unique_qnn_kernel_names")).add("fake"));
        assertThrows(UnsupportedOperationException.class, () -> ((List<String>) result.get("verification_errors")).add("fake"));
        assertThrows(UnsupportedOperationException.class, () -> ((Map<String, Integer>) result.get("qnn_kernel_event_counts")).put("fake", 10));
    }

    @SuppressWarnings("unchecked")
    @Test public void histogramDistinguishesBalancedPartitionCallsFromTheSameTotalWithSkips() throws Exception {
        Map<String, Object> balanced = QnnProfileVerifier.verify(profile(qnn("a"), qnn("b"), qnn("a"), qnn("b")), 2, true, false);
        assertEquals(true, balanced.get("qnn_kernel_counts_match_expected_calls"));
        assertEquals(Map.of("a_kernel_time", 2, "b_kernel_time", 2), balanced.get("qnn_kernel_event_counts"));
        Map<String, Object> skewed = QnnProfileVerifier.verify(profile(qnn("a"), qnn("a"), qnn("a"), qnn("b")), 2, true, false);
        assertEquals(false, skewed.get("qnn_kernel_counts_match_expected_calls"));
        Map<String, Integer> counts = (Map<String, Integer>) skewed.get("qnn_kernel_event_counts");
        assertEquals(Integer.valueOf(3), counts.get("a_kernel_time"));
        assertEquals(Integer.valueOf(1), counts.get("b_kernel_time"));
    }

    private static String qnn(String name) { return node(name, "QNNExecutionProvider"); }
    private static String cpu(String name) { return node(name, "CPUExecutionProvider"); }
    private static String node(String name, String provider) {
        return "{\"cat\":\"Node\",\"name\":\"" + name + "_kernel_time\",\"args\":{\"provider\":\"" + provider + "\"}}";
    }
    private static String profile(String... nodes) { return "[" + String.join(",", nodes) + "]"; }
}
