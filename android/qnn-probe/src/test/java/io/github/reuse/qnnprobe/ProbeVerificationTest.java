package io.github.reuse.qnnprobe;

import static org.junit.Assert.*;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.Test;

public final class ProbeVerificationTest {
    @Test public void countsExecutionEventsAndRejectsProviderRegistrationAsEvidence() throws Exception {
        String profile = "["
                + "{\"cat\":\"Node\",\"name\":\"QNN_partition_kernel_time\",\"args\":{\"provider\":\"QNNExecutionProvider\"}},"
                + "{\"cat\":\"Node\",\"name\":\"conv_kernel_time\",\"args\":{\"provider\":\"CPUExecutionProvider\"}},"
                + "{\"cat\":\"Session\",\"name\":\"register_kernel_time\",\"args\":{\"provider\":\"QNNExecutionProvider\"}},"
                + "{\"cat\":\"Node\",\"name\":\"QNN_partition_fence_before\",\"args\":{\"provider\":\"QNNExecutionProvider\"}},"
                + "{\"cat\":\"Node\",\"name\":\"other_kernel_time\"},null]";
        assertEquals(1, ProbeVerification.countQnnKernelEvents(profile));
        assertEquals(0, ProbeVerification.countQnnKernelEvents("[]"));
    }

    @Test public void malformedProfileFailsClosed() {
        assertThrows(IOException.class, () -> ProbeVerification.countQnnKernelEvents("{not-an-array}"));
    }

    @Test public void verificationRequiresDeviceNumericsAndKernelExecutionTogether() {
        assertTrue(ProbeVerification.verified(true, true, 21));
        assertFalse(ProbeVerification.verified(false, true, 21));
        assertFalse(ProbeVerification.verified(true, false, 21));
        assertFalse(ProbeVerification.verified(true, true, 0));
    }

    @Test public void decodesLittleEndianAndRequiresExactFixtureLength() throws Exception {
        byte[] bytes = ByteBuffer.allocate(12).order(ByteOrder.LITTLE_ENDIAN)
                .putFloat(1.25f).putFloat(-2.5f).putFloat(0).array();
        assertArrayEquals(new float[]{1.25f, -2.5f, 0}, ProbeVerification.decodeFloat32(bytes, 3), 0);
        assertThrows(IOException.class, () -> ProbeVerification.decodeFloat32(bytes, 2));
        assertThrows(IOException.class, () -> ProbeVerification.decodeFloat32(bytes, -1));
    }

    @Test public void rejectsNonFiniteFixtureAndOutput() throws Exception {
        byte[] bytes = ByteBuffer.allocate(4).order(ByteOrder.LITTLE_ENDIAN).putFloat(Float.NaN).array();
        assertThrows(IOException.class, () -> ProbeVerification.decodeFloat32(bytes, 1));
        ProbeVerification.Comparison comparison = ProbeVerification.compare(
                new float[]{Float.POSITIVE_INFINITY}, new float[]{1}, .1, 0);
        assertFalse(comparison.passed);
    }

    @Test public void comparesSignedOutputsAgainstAbsoluteTolerance() throws Exception {
        ProbeVerification.Comparison pass = ProbeVerification.compare(
                new float[]{-1.125f, .125f, 3}, new float[]{-1, 0, 3}, .125, 0);
        assertTrue(pass.passed);
        assertEquals(.125, pass.maximumAbsoluteError, 0);
        assertEquals(0, pass.outsideTolerance);
        ProbeVerification.Comparison fail = ProbeVerification.compare(
                new float[]{-1.25f, 0, 3}, new float[]{-1, 0, 3}, .125, 0);
        assertFalse(fail.passed);
        assertEquals(1, fail.outsideTolerance);
    }

    @Test public void invalidOutputContractCannotPass() {
        assertThrows(IOException.class, () -> ProbeVerification.compare(new float[1], new float[2], 0, 0));
        assertThrows(IOException.class, () -> ProbeVerification.compare(new float[0], new float[0], 0, 0));
        assertThrows(IOException.class, () -> ProbeVerification.compare(new float[1], new float[1], Double.NaN, 0));
        assertThrows(IOException.class, () -> ProbeVerification.compare(new float[1], new float[1], -1, 0));
    }

    @Test public void cpuBaselineRequiresOnlyCpuKernelEvidence() throws Exception {
        assertTrue(ProbeVerification.verifiedCpu(true, ProbeVerification.executionEvidence(profile("CPUExecutionProvider"))));
        assertFalse(ProbeVerification.verifiedCpu(false, ProbeVerification.executionEvidence(profile("CPUExecutionProvider"))));
        assertFalse(ProbeVerification.verifiedCpu(true, ProbeVerification.executionEvidence(profile("QNNExecutionProvider"))));
        assertFalse(ProbeVerification.verifiedCpu(true, ProbeVerification.executionEvidence(profile("CPUExecutionProvider", "QNNExecutionProvider"))));
        assertFalse(ProbeVerification.verifiedCpu(true, ProbeVerification.executionEvidence(profile("CPUExecutionProvider", "XnnpackExecutionProvider"))));
        assertFalse(ProbeVerification.verifiedCpu(true, ProbeVerification.executionEvidence(profile("CPUExecutionProvider", ""))));
        assertFalse(ProbeVerification.verifiedCpu(true, ProbeVerification.executionEvidence("[]")));
    }

    @Test public void strictHtpTestRejectsMixedProviderKernelExecution() throws Exception {
        ProbeVerification.ExecutionEvidence qnn = ProbeVerification.executionEvidence(profile("QNNExecutionProvider"));
        assertTrue(ProbeVerification.verifiedHtp(true, true, qnn));
        assertFalse(ProbeVerification.verifiedHtp(false, true, qnn));
        assertFalse(ProbeVerification.verifiedHtp(true, false, qnn));
        assertFalse(ProbeVerification.verifiedHtp(true, true,
                ProbeVerification.executionEvidence(profile("QNNExecutionProvider", "CPUExecutionProvider"))));
    }

    @Test public void speedupRequiresVerifiedBackendsAndPositiveFiniteTiming() {
        assertEquals(2.5, ProbeVerification.speedup(true, 5, true, 2), 0);
        assertEquals(.5, ProbeVerification.speedup(true, 1, true, 2), 0);
        assertNull(ProbeVerification.speedup(false, 5, true, 2));
        assertNull(ProbeVerification.speedup(true, 5, false, 2));
        assertNull(ProbeVerification.speedup(true, 0, true, 2));
        assertNull(ProbeVerification.speedup(true, 5, true, 0));
        assertNull(ProbeVerification.speedup(true, -1, true, 2));
        assertNull(ProbeVerification.speedup(true, Double.NaN, true, 2));
        assertNull(ProbeVerification.speedup(true, 5, true, Double.POSITIVE_INFINITY));
        assertNull(ProbeVerification.speedup(true, Double.MAX_VALUE, true, Double.MIN_VALUE));
    }

    @Test public void numericalSummaryPreservesWorstIndexActualAndReferencePair() throws Exception {
        ProbeVerification.Comparison comparison = ProbeVerification.compare(
                new float[]{1, 3, -5}, new float[]{1, 2, -1}, 5, 0);
        assertEquals(5.0 / 3, comparison.meanAbsoluteError, 1e-12);
        assertEquals(2, comparison.maximumErrorIndex);
        assertEquals(-5, comparison.maximumErrorActual, 0);
        assertEquals(-1, comparison.maximumErrorReference, 0);
        assertEquals(Math.sqrt(17.0 / 3), comparison.rmse, 1e-12);
        assertEquals(Math.sqrt(2), comparison.referenceRms, 1e-12);
        assertEquals(Math.sqrt(17.0 / 6), comparison.normalizedRmse, 1e-12);
    }

    @Test public void fp32RelativeToleranceStillRequiresAbsoluteAccuracyNearZero() throws Exception {
        assertTrue(ProbeVerification.compare(new float[]{1000.0625f}, new float[]{1000}, 0, 1e-4).passed);
        assertFalse(ProbeVerification.compare(new float[]{.01f}, new float[]{0}, 1e-4, 1e-4).passed);
        assertEquals(0, ProbeVerification.compare(new float[]{0}, new float[]{0}, 0, 0).normalizedRmse, 0);
    }

    private static String profile(String... providers) {
        StringBuilder json = new StringBuilder("[");
        for (int i = 0; i < providers.length; i++) {
            if (i > 0) json.append(',');
            json.append("{\"cat\":\"Node\",\"name\":\"node").append(i)
                    .append("_kernel_time\",\"args\":{\"provider\":\"").append(providers[i]).append("\"}}");
        }
        return json.append(']').toString();
    }
}
