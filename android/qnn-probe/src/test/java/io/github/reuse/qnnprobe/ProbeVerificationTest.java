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
}
