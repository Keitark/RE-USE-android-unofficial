package io.github.reuse;

import static org.junit.Assert.*;
import org.junit.Test;

public class ReuseConfigTest {
    @Test public void exactStreamingFormatAndLatencyArePublic() {
        ReuseConfig config = new ReuseConfig(16000);
        assertEquals(640, config.fftSize);
        assertEquals(320, config.hopSize);
        assertEquals(321, config.frequencyBins);
        assertEquals(640, config.startupInputSamples());
        assertEquals(320, config.bufferedDelaySamples());
        assertEquals(3, config.exitLayers);
        assertEquals(ReuseConfig.Backend.CPU, config.backend);
        assertFalse(config.nnapiUseFp16);
        assertNull(config.nnapiProfilingPrefix);
    }

    @Test public void unsupportedFormatsFailEarly() {
        assertThrows(IllegalArgumentException.class, () -> new ReuseConfig(48000));
        assertThrows(IllegalArgumentException.class, () -> new ReuseConfig(8000, 2, 1));
        assertThrows(IllegalArgumentException.class, () -> new ReuseConfig(8000, 13, 1));
        assertThrows(IllegalArgumentException.class, () -> new ReuseConfig(8000, 3, 0));
    }

    @Test public void acceleratorPrecisionAndVerificationAreExplicit() {
        java.nio.file.Path profile = java.nio.file.Paths.get("private-capture", "nnapi-warmup");
        ReuseConfig config = new ReuseConfig(8000, 3, 4,
                ReuseConfig.Backend.NNAPI_ACCELERATOR, true, profile);
        assertEquals(ReuseConfig.Backend.NNAPI_ACCELERATOR, config.backend);
        assertTrue(config.nnapiUseFp16);
        assertEquals(profile, config.nnapiProfilingPrefix);
        assertEquals(160, config.hopSize);
        assertThrows(IllegalArgumentException.class,
                () -> new ReuseConfig(8000, 3, 4, null, false, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ReuseConfig(8000, 3, 4, ReuseConfig.Backend.CPU, true, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ReuseConfig(8000, 3, 4, ReuseConfig.Backend.NNAPI_ACCELERATOR, false, null));
        assertThrows(IllegalArgumentException.class,
                () -> new ReuseConfig(8000, 3, 4, ReuseConfig.Backend.CPU, false, profile));
    }
}
