package io.github.reuse.qnnprobe;

import static org.junit.Assert.*;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import org.junit.Test;

public final class PublicPcmBenchmarkTest {
    @Test public void interpolationPreservesCountAndUsesAdjacentMeansWithLastSampleHeld() throws Exception {
        byte[] pcm = PublicPcmBenchmark.interpolate16kPcm(new float[]{0, .5f, -.5f});
        assertEquals(3 * 2 * 2, pcm.length);
        assertArrayEquals(new short[]{0, 8192, 16384, 0, -16384, -16384}, samples(pcm));
        assertArrayEquals(new short[]{4096, 4096}, samples(PublicPcmBenchmark.interpolate16kPcm(new float[]{.125f})));
    }

    @Test public void conversionSaturatesAndRejectsNonfiniteOrUnboundedSource() throws Exception {
        assertArrayEquals(new short[]{32767, 0, -32768, -32768}, samples(PublicPcmBenchmark.interpolate16kPcm(new float[]{2, -2})));
        assertThrows(IOException.class, () -> PublicPcmBenchmark.interpolate16kPcm(new float[]{Float.NaN}));
        assertThrows(IOException.class, () -> PublicPcmBenchmark.interpolate16kPcm(new float[]{Float.POSITIVE_INFINITY}));
        assertThrows(IOException.class, () -> PublicPcmBenchmark.interpolate16kPcm(new float[0]));
        assertThrows(IOException.class, () -> PublicPcmBenchmark.interpolate16kPcm(new float[120001]));
    }

    @Test public void deadlinesRepresentCollectedSamplesIncludingFirstAndPartialFinalPacket() {
        long start = 1000;
        assertEquals(start + 50_000_000, PublicPcmBenchmark.packetDeadlineNs(start, 800));
        assertEquals(start + 100_000_000, PublicPcmBenchmark.packetDeadlineNs(start, 1600));
        assertEquals(start + 200_125_000, PublicPcmBenchmark.packetDeadlineNs(start, 3202));
        assertThrows(IllegalArgumentException.class, () -> PublicPcmBenchmark.packetDeadlineNs(start, -1));
    }

    @Test public void expectedOutputIncludesOnlyOddInputPaddingAndTheExistingFirTail() {
        assertEquals(0, PublicPcmBenchmark.expectedOutputSamples(0));
        assertEquals(3388, PublicPcmBenchmark.expectedOutputSamples(3200));
        assertEquals(3390, PublicPcmBenchmark.expectedOutputSamples(3201));
        assertThrows(IllegalArgumentException.class, () -> PublicPcmBenchmark.expectedOutputSamples(-1));
    }

    @Test public void queueAndCallbackAccountingMustMatchWithoutDrops() {
        Map<String, Object> stats = finishedStats();
        assertTrue(PublicPcmBenchmark.accountingMatches(stats, 3200, 3388));
        assertFalse(PublicPcmBenchmark.accountingMatches(stats, 3200, 3387));
        for (String key : new String[]{"input_samples", "processed_input_samples", "output_samples", "dropped_input_samples",
                "queue_overflows", "queued_input_samples", "sequence_gaps"}) {
            Map<String, Object> modified = finishedStats();
            modified.put(key, ((Number) modified.get(key)).longValue() + 1);
            assertFalse(key, PublicPcmBenchmark.accountingMatches(modified, 3200, 3388));
        }
        stats.put("state", "ERROR");
        assertFalse(PublicPcmBenchmark.accountingMatches(stats, 3200, 3388));
    }

    @Test public void missingAndMalformedCountsCannotProduceAPass() {
        Map<String, Object> missing = finishedStats(); missing.remove("queued_input_samples");
        assertFalse(PublicPcmBenchmark.accountingMatches(missing, 3200, 3388));
        Map<String, Object> malformed = finishedStats(); malformed.put("dropped_input_samples", Double.NaN);
        assertFalse(PublicPcmBenchmark.accountingMatches(malformed, 3200, 3388));
    }

    @Test public void failedPreparationNeverBecomesReadinessAndWaitsHaveExplicitBounds() throws Exception {
        CountDownLatch ready = new CountDownLatch(1), done = new CountDownLatch(1);
        assertThrows(IOException.class, () -> PublicPcmBenchmark.awaitReadyOrTerminal(ready, done, 0));
        assertThrows(IOException.class, () -> PublicPcmBenchmark.awaitTerminal(done, 0));
        done.countDown();
        assertFalse(PublicPcmBenchmark.awaitReadyOrTerminal(ready, done, 0));
        PublicPcmBenchmark.awaitTerminal(done, 0);
        assertThrows(IllegalArgumentException.class, () -> PublicPcmBenchmark.awaitTerminal(done, -1));
    }

    @Test public void callbacksSignalTerminalAfterPublishingOwnedStatistics() throws Exception {
        PublicPcmBenchmark.Callbacks callbacks = new PublicPcmBenchmark.Callbacks();
        callbacks.onReady();
        assertTrue(callbacks.readyAtNs != 0);
        assertTrue(PublicPcmBenchmark.awaitReadyOrTerminal(callbacks.ready, callbacks.done, 0));
        callbacks.onAudio(new byte[4]);
        assertEquals(2, callbacks.outputSamples.get());
        assertThrows(IllegalStateException.class, () -> callbacks.onAudio(new byte[1]));
        Map<String, Object> stats = finishedStats();
        callbacks.onFinished(stats); stats.put("input_samples", -1);
        PublicPcmBenchmark.awaitTerminal(callbacks.done, 0);
        assertEquals(3200L, callbacks.stats.get("input_samples"));
        assertNull(callbacks.errorCode);
    }

    @Test public void failedQnnPrerequisiteSkipsNpuWhileCpuComparisonsRemainEligible() {
        assertNull(PublicPcmBenchmark.skipReason(false, false, true));
        assertEquals("SKIPPED_QNN_PREREQUISITE_FAILED", PublicPcmBenchmark.skipReason(true, false, true));
        assertNull(PublicPcmBenchmark.skipReason(true, true, true));
        assertEquals("SKIPPED_PREVIOUS_CLEANUP_INCOMPLETE", PublicPcmBenchmark.skipReason(false, true, false));
        assertEquals("SKIPPED_PREVIOUS_CLEANUP_INCOMPLETE", PublicPcmBenchmark.skipReason(true, true, false));
    }

    @Test public void repetitionsPreserveBaseBytesAndAllowTheExactNineClipDuration() throws Exception {
        byte[] base = {1, 2, 3, 4};
        assertSame(base, PublicPcmBenchmark.repeatPcm(base, 1));
        assertArrayEquals(new byte[]{1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4}, PublicPcmBenchmark.repeatPcm(base, 3));
        byte[] publicClip = new byte[51840 * 4];
        byte[] repeated = PublicPcmBenchmark.repeatPcm(publicClip, 9);
        assertEquals(933120, repeated.length / 2);
        assertEquals(58_320_000_000L, PublicPcmBenchmark.packetDeadlineNs(0, repeated.length / 2));
        assertEquals(933308, PublicPcmBenchmark.expectedOutputSamples(repeated.length / 2));
    }

    @Test public void repetitionsAreBoundedSeparatelyFromBaseFixtureAndSixtySecondPacingLimit() throws Exception {
        assertThrows(IllegalArgumentException.class, () -> PublicPcmBenchmark.repeatPcm(new byte[4], 0));
        assertThrows(IllegalArgumentException.class, () -> PublicPcmBenchmark.repeatPcm(new byte[4], 10));
        assertThrows(IOException.class, () -> PublicPcmBenchmark.repeatPcm(new byte[3], 1));
        byte[] fifteenSeconds = new byte[8000 * 15 * 4];
        assertEquals(16000 * 60, PublicPcmBenchmark.repeatPcm(fifteenSeconds, 4).length / 2);
        assertThrows(IOException.class, () -> PublicPcmBenchmark.repeatPcm(fifteenSeconds, 5));
        assertThrows(IllegalArgumentException.class, () -> PublicPcmBenchmark.packetDeadlineNs(0, 16000 * 60 + 1));
    }

    private static Map<String, Object> finishedStats() {
        Map<String, Object> stats = new LinkedHashMap<>();
        stats.put("state", "FINISHED"); stats.put("input_samples", 3200L); stats.put("processed_input_samples", 3200L);
        stats.put("output_samples", 3388L); stats.put("dropped_input_samples", 0L); stats.put("queue_overflows", 0L);
        stats.put("queued_input_samples", 0L); stats.put("sequence_gaps", 0L);
        return stats;
    }
    private static short[] samples(byte[] pcm) {
        short[] values = new short[pcm.length / 2];
        ByteBuffer.wrap(pcm).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(values);
        return values;
    }
}
