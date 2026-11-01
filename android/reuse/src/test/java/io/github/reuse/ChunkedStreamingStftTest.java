package io.github.reuse;

import static org.junit.Assert.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import org.junit.Test;

public class ChunkedStreamingStftTest {
    private static final int FRAMES = 5;
    private static final ChunkedStreamingStft.SpectrumTransform IDENTITY = (mag, phase, real, imag) -> {
        for (int i = 0; i < mag.length; i++) {
            double amplitude = Math.expm1(mag[i]);
            real[i] = (float) (amplitude * Math.cos(phase[i]));
            imag[i] = (float) (amplitude * Math.sin(phase[i]));
        }
    };

    @Test public void randomContinuousAudioReconstructsAtBothRatesIncludingFlush() throws Exception {
        for (int rate : new int[]{8000, 16000}) {
            ReuseConfig config = new ReuseConfig(rate);
            float[] input = new float[config.hopSize * FRAMES * 7];
            Random random = new Random(5678);
            for (int i = 0; i < input.length; i++) input[i] = random.nextFloat() * 1.8f - 0.9f;
            assertArrayEquals(input, processAll(new ChunkedStreamingStft(config), input,
                    config.hopSize * FRAMES), 2e-6f);
        }
    }

    @Test public void impulsesAtHopChunkAndFinalBoundariesAreNotDroppedOrShifted() throws Exception {
        ReuseConfig config = new ReuseConfig(16000);
        int chunk = config.hopSize * FRAMES;
        for (int position : new int[]{0, 1, config.hopSize - 1, config.hopSize,
                config.hopSize + 1, chunk - 1, chunk, chunk + 1, 3 * chunk - 1}) {
            float[] input = new float[chunk * 3];
            input[position] = 1.0f;
            assertArrayEquals(input, processAll(new ChunkedStreamingStft(config), input, chunk), 2e-6f);
        }
    }

    @Test public void outputLengthsRetainOnlyRealInputWhenFlushing() throws Exception {
        ReuseConfig config = new ReuseConfig(8000);
        ChunkedStreamingStft dsp = new ChunkedStreamingStft(config);
        int chunk = config.hopSize * FRAMES;
        assertEquals(config.hopSize * (FRAMES - 1), dsp.process(new float[chunk], IDENTITY).length);
        assertEquals(chunk, dsp.process(new float[chunk], IDENTITY).length);
        assertEquals(config.hopSize, dsp.flush(IDENTITY).length);
        assertThrows(IllegalStateException.class, () -> dsp.flush(IDENTITY));
        assertThrows(IllegalStateException.class, () -> dsp.process(new float[chunk], IDENTITY));
        dsp.reset();
        assertNull(dsp.flush(IDENTITY));
    }

    @Test public void resetClearsAnalysisOverlapAndTerminalState() throws Exception {
        ReuseConfig config = new ReuseConfig(8000);
        ChunkedStreamingStft dsp = new ChunkedStreamingStft(config);
        int chunk = config.hopSize * FRAMES;
        float[] loud = new float[chunk];
        Arrays.fill(loud, 0.75f);
        dsp.process(loud, IDENTITY);
        dsp.flush(IDENTITY);
        dsp.reset();
        float[] zeros = new float[chunk * 3];
        assertArrayEquals(zeros, processAll(dsp, zeros, chunk), 0.0f);
    }

    @Test public void rejectsInvalidInputBeforeChangingAnyHistory() throws Exception {
        ReuseConfig config = new ReuseConfig(8000);
        ChunkedStreamingStft dsp = new ChunkedStreamingStft(config);
        int chunk = config.hopSize * FRAMES;
        assertThrows(IllegalArgumentException.class, () -> dsp.process(null, IDENTITY));
        assertThrows(IllegalArgumentException.class, () -> dsp.process(new float[config.hopSize], IDENTITY));
        assertThrows(IllegalArgumentException.class, () -> dsp.process(new float[chunk - 1], IDENTITY));
        for (float invalid : new float[]{Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY}) {
            // Invalid sample in the final frame must not partially advance the
            // earlier four frames' histories before input validation fails.
            float[] input = new float[chunk];
            Arrays.fill(input, 0.5f);
            input[chunk - 1] = invalid;
            assertThrows(IllegalArgumentException.class, () -> dsp.process(input, IDENTITY));
        }
        assertArrayEquals(new float[config.hopSize * (FRAMES - 1)],
                dsp.process(new float[chunk], IDENTITY), 0.0f);
        assertArrayEquals(new float[config.hopSize], dsp.flush(IDENTITY), 0.0f);
    }

    @Test public void rejectsNonFiniteModelOutputInAnyFrame() {
        ReuseConfig config = new ReuseConfig(8000);
        int chunk = config.hopSize * FRAMES;
        for (int frame : new int[]{0, FRAMES - 1}) {
            ChunkedStreamingStft dsp = new ChunkedStreamingStft(config);
            final int outputIndex = frame * config.frequencyBins + 3;
            assertThrows(IllegalStateException.class, () -> dsp.process(new float[chunk],
                    (mag, phase, real, imag) -> real[outputIndex] = Float.NaN));
        }
    }

    @Test public void eachChunkUsesOneTransformWithFiveRowMajorFrames() throws Exception {
        ReuseConfig config = new ReuseConfig(8000);
        int[] calls = {0};
        ChunkedStreamingStft dsp = new ChunkedStreamingStft(config);
        float[] input = new float[config.hopSize * FRAMES];
        for (int frame = 0; frame < FRAMES; frame++) {
            Arrays.fill(input, frame * config.hopSize, (frame + 1) * config.hopSize,
                    (frame + 1) * 0.1f);
        }
        dsp.process(input, (mag, phase, real, imag) -> {
            calls[0]++;
            assertEquals(FRAMES * config.frequencyBins, mag.length);
            assertEquals(mag.length, phase.length);
            assertEquals(mag.length, real.length);
            assertEquals(mag.length, imag.length);
            // Increasing constant levels create increasing DC magnitudes in
            // consecutive causal windows. Their DC bins begin each row.
            for (int frame = 1; frame < FRAMES; frame++) {
                assertTrue(mag[frame * config.frequencyBins] > mag[(frame - 1) * config.frequencyBins]);
            }
            IDENTITY.transform(mag, phase, real, imag);
        });
        assertEquals(1, calls[0]);
    }

    @Test public void negativeDcAndNyquistUseCanonicalPositivePiInEveryFrame() throws Exception {
        for (int rate : new int[]{8000, 16000}) {
            ReuseConfig config = new ReuseConfig(rate);
            ChunkedStreamingStft dsp = new ChunkedStreamingStft(config);
            Random random = new Random(20261101);
            ChunkedStreamingStft.SpectrumTransform inspect = (mag, phase, real, imag) -> {
                for (int frame = 0; frame < FRAMES; frame++) {
                    assertEquals((float) Math.PI, phase[frame * config.frequencyBins], 0.0f);
                    assertEquals((float) Math.PI, phase[(frame + 1) * config.frequencyBins - 1], 0.0f);
                }
                Arrays.fill(real, 0.0f);
                Arrays.fill(imag, 0.0f);
            };
            for (int block = 0; block < 3; block++) {
                float[] input = new float[config.hopSize * FRAMES];
                for (int i = 0; i < input.length; i++) {
                    input[i] = -0.25f + ((i & 1) == 0 ? -0.25f : 0.25f)
                            + 0.05f * (random.nextFloat() - 0.5f);
                }
                dsp.process(input, inspect);
            }
        }
    }

    private float[] processAll(ChunkedStreamingStft dsp, float[] input, int chunk) throws Exception {
        List<float[]> outputs = new ArrayList<>();
        int samples = 0;
        for (int offset = 0; offset < input.length; offset += chunk) {
            float[] result = dsp.process(Arrays.copyOfRange(input, offset, offset + chunk), IDENTITY);
            if (result != null) {
                outputs.add(result);
                samples += result.length;
            }
        }
        float[] tail = dsp.flush(IDENTITY);
        if (tail != null) {
            outputs.add(tail);
            samples += tail.length;
        }
        float[] result = new float[samples];
        int offset = 0;
        for (float[] output : outputs) {
            System.arraycopy(output, 0, result, offset, output.length);
            offset += output.length;
        }
        return result;
    }
}
