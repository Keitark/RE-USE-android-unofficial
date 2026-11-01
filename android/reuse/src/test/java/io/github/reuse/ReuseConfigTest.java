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
    }

    @Test public void unsupportedFormatsFailEarly() {
        assertThrows(IllegalArgumentException.class, () -> new ReuseConfig(48000));
        assertThrows(IllegalArgumentException.class, () -> new ReuseConfig(8000, 2, 1));
        assertThrows(IllegalArgumentException.class, () -> new ReuseConfig(8000, 13, 1));
        assertThrows(IllegalArgumentException.class, () -> new ReuseConfig(8000, 3, 0));
    }
}
