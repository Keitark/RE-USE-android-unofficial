package io.github.reuse;

import static org.junit.Assert.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.Test;

public class StreamingStftTest {
    private static final StreamingStft.SpectrumTransform IDENTITY = (mag, phase, real, imag) -> {
        for (int i = 0; i < mag.length; i++) {
            double amplitude = Math.expm1(mag[i]);
            real[i] = (float) (amplitude * Math.cos(phase[i]));
            imag[i] = (float) (amplitude * Math.sin(phase[i]));
        }
    };

    @Test public void randomAudioReconstructsAtBothRatesIncludingBothEdges() throws Exception {
        for (int rate : new int[]{8000, 16000}) {
            ReuseConfig config = new ReuseConfig(rate);
            float[] input = new float[config.hopSize * 19];
            Random random = new Random(1234);
            for (int i = 0; i < input.length; i++) input[i] = random.nextFloat() * 1.8f - 0.9f;
            assertArrayEquals(input, processAll(new StreamingStft(config), input, config.hopSize), 2e-6f);
        }
    }

    @Test public void impulsesAtBoundaryAndLastSampleAreNotDroppedOrShifted() throws Exception {
        ReuseConfig config = new ReuseConfig(16000);
        for (int position : new int[]{0, 1, config.hopSize - 1, config.hopSize,
                config.hopSize + 1, 3 * config.hopSize - 1}) {
            float[] input = new float[config.hopSize * 3];
            input[position] = 1.0f;
            assertArrayEquals(input, processAll(new StreamingStft(config), input, config.hopSize), 2e-6f);
        }
    }

    @Test public void firstOutputNeedsTwoHopsAndFlushReturnsExactlyOneHop() throws Exception {
        ReuseConfig config = new ReuseConfig(8000);
        StreamingStft dsp = new StreamingStft(config);
        assertNull(dsp.process(new float[config.hopSize], IDENTITY));
        assertEquals(config.hopSize, dsp.flush(IDENTITY).length);
        assertThrows(IllegalStateException.class, () -> dsp.flush(IDENTITY));
        assertThrows(IllegalStateException.class, () -> dsp.process(new float[config.hopSize], IDENTITY));
        dsp.reset();
        assertNull(dsp.flush(IDENTITY));
    }

    @Test public void resetClearsOverlapAndTerminalState() throws Exception {
        ReuseConfig config = new ReuseConfig(8000);
        StreamingStft dsp = new StreamingStft(config);
        float[] loud = new float[config.hopSize];
        java.util.Arrays.fill(loud, 0.75f);
        dsp.process(loud, IDENTITY);
        dsp.flush(IDENTITY);
        dsp.reset();
        float[] zeros = new float[config.hopSize * 3];
        assertArrayEquals(zeros, processAll(dsp, zeros, config.hopSize), 0.0f);
    }

    @Test public void rejectsInvalidInputBeforeChangingHistory() throws Exception {
        ReuseConfig config = new ReuseConfig(8000);
        StreamingStft dsp = new StreamingStft(config);
        assertThrows(IllegalArgumentException.class, () -> dsp.process(null, IDENTITY));
        assertThrows(IllegalArgumentException.class, () -> dsp.process(new float[2], IDENTITY));
        for (float invalid : new float[]{Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY}) {
            float[] input = new float[config.hopSize]; input[17] = invalid;
            assertThrows(IllegalArgumentException.class, () -> dsp.process(input, IDENTITY));
        }
        assertNull(dsp.process(new float[config.hopSize], IDENTITY));
    }

    @Test public void rejectsNonFiniteModelOutput() {
        ReuseConfig config = new ReuseConfig(8000);
        StreamingStft dsp = new StreamingStft(config);
        assertThrows(IllegalStateException.class, () -> dsp.process(new float[config.hopSize],
                (mag, phase, real, imag) -> real[3] = Float.NaN));
    }

    @Test public void negativeDcAndNyquistUseCanonicalPositivePiPhase() throws Exception {
        for (int rate : new int[]{8000, 16000}) {
            ReuseConfig config = new ReuseConfig(rate);
            StreamingStft dsp = new StreamingStft(config);
            Random random = new Random(20261101);
            StreamingStft.SpectrumTransform inspect = (mag, phase, real, imag) -> {
                assertEquals((float) Math.PI, phase[0], 0.0f);
                assertEquals((float) Math.PI, phase[phase.length - 1], 0.0f);
                // Synthesis is irrelevant to this analysis-phase contract.
                java.util.Arrays.fill(real, 0.0f);
                java.util.Arrays.fill(imag, 0.0f);
            };
            for (int block = 0; block < 12; block++) {
                float[] input = new float[config.hopSize];
                for (int i = 0; i < input.length; i++) {
                    input[i] = -0.25f + ((i & 1) == 0 ? -0.25f : 0.25f)
                            + 0.05f * (random.nextFloat() - 0.5f);
                }
                dsp.process(input, inspect);
            }
        }
    }

    private float[] processAll(StreamingStft dsp, float[] input, int hop) throws Exception {
        List<float[]> outputs = new ArrayList<>();
        for (int offset = 0; offset < input.length; offset += hop) {
            float[] block = java.util.Arrays.copyOfRange(input, offset, offset + hop);
            float[] result = dsp.process(block, IDENTITY);
            if (result != null) outputs.add(result);
        }
        float[] tail = dsp.flush(IDENTITY);
        if (tail != null) outputs.add(tail);
        float[] result = new float[outputs.size() * hop];
        for (int i = 0; i < outputs.size(); i++) System.arraycopy(outputs.get(i), 0, result, i * hop, hop);
        return result;
    }
}
