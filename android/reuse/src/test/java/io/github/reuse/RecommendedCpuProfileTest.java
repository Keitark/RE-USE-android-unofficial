package io.github.reuse;

import static org.junit.Assert.*;
import org.junit.Test;

public final class RecommendedCpuProfileTest {
    @Test public void namedProfileSelectsTheCausalFp32CpuFormatExplicitly() {
        ReuseConfig config = RecommendedCpuProfile.config();
        assertEquals(8000, config.sampleRate);
        assertEquals(3, config.exitLayers);
        assertEquals(4, config.intraOpThreads);
        assertEquals(160, config.hopSize);
        assertEquals(100, RecommendedCpuProfile.FRAMES_PER_CALL * config.hopSize * 1000 / config.sampleRate);
        assertEquals(ReuseConfig.Backend.CPU, config.backend);
        assertFalse(config.nnapiUseFp16);
        assertNull(config.nnapiProfilingPrefix);
        assertTrue(RecommendedCpuProfile.MODEL_SHA256.matches("[0-9a-f]{64}"));
    }

    @Test public void comparisonsChangeThePoolWithoutChangingTheModelFormat() {
        ReuseConfig twoThreads = RecommendedCpuProfile.config(2);
        assertEquals(2, twoThreads.intraOpThreads);
        assertEquals(RecommendedCpuProfile.SAMPLE_RATE, twoThreads.sampleRate);
        assertEquals(RecommendedCpuProfile.EXIT_LAYERS, twoThreads.exitLayers);
        assertEquals(ReuseConfig.Backend.CPU, twoThreads.backend);
        assertThrows(IllegalArgumentException.class, () -> RecommendedCpuProfile.config(0));
        assertThrows(IllegalArgumentException.class, () -> RecommendedCpuProfile.config(17));
    }

    @Test public void genericConfigurationsRetainTheirPreviousDefaults() {
        assertEquals(2, new ReuseConfig(8000).intraOpThreads);
        assertEquals(2, new ReuseConfig(16000).intraOpThreads);
        assertEquals(16000, new ReuseConfig(16000).sampleRate);
    }
}
