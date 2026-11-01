package io.github.reuse;

import ai.onnxruntime.OrtException;
import java.util.Arrays;

/** The same causal STFT/OLA as StreamingStft, with five spectra per model call. */
final class ChunkedStreamingStft {
    static final int FRAMES_PER_CALL = 5;

    interface SpectrumTransform {
        /** Feature arrays contain five consecutive rows of frequency bins. */
        void transform(float[] magnitude, float[] phase, float[] outputReal, float[] outputImag)
                throws OrtException;
    }

    private final int size;
    private final int hop;
    private final int bins;
    private final MixedRadixFft fft;
    private final double[] window;
    private final double[] timeReal;
    private final double[] timeImag;
    private final double[] overlap;
    private final double[] weights;
    private final float[] previous;
    private final float[] magnitude;
    private final float[] phase;
    private final float[] outputReal;
    private final float[] outputImag;
    private boolean started;
    private boolean finished;

    ChunkedStreamingStft(ReuseConfig config) {
        size = config.fftSize;
        hop = config.hopSize;
        bins = config.frequencyBins;
        fft = new MixedRadixFft(size);
        window = new double[size];
        timeReal = new double[size];
        timeImag = new double[size];
        overlap = new double[size];
        weights = new double[size];
        previous = new float[hop];
        magnitude = new float[FRAMES_PER_CALL * bins];
        phase = new float[magnitude.length];
        outputReal = new float[magnitude.length];
        outputImag = new float[magnitude.length];
        for (int i = 0; i < size; i++) {
            window[i] = 0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / size);
        }
    }

    float[] process(float[] input, SpectrumTransform transform) throws OrtException {
        if (finished) throw new IllegalStateException("Stream flushed; call reset before process");
        if (input == null || input.length != FRAMES_PER_CALL * hop) {
            throw new IllegalArgumentException("Expected exactly " + FRAMES_PER_CALL * hop + " mono samples");
        }
        for (float value : input) {
            if (!Float.isFinite(value)) throw new IllegalArgumentException("PCM must be finite");
        }
        return processFrames(input, transform);
    }

    private float[] processFrames(float[] input, SpectrumTransform transform) throws OrtException {
        // Analysis history advances one hop at a time, independent of inference
        // batching. No analysis frame contains future samples from the chunk.
        for (int frame = 0; frame < FRAMES_PER_CALL; frame++) {
            int inputOffset = frame * hop;
            for (int i = 0; i < hop; i++) {
                timeReal[i] = previous[i] * window[i];
                timeReal[hop + i] = input[inputOffset + i] * window[hop + i];
            }
            Arrays.fill(timeImag, 0.0);
            fft.transform(timeReal, timeImag, false);
            timeImag[0] = +0.0;
            timeImag[bins - 1] = +0.0;
            int featureOffset = frame * bins;
            for (int k = 0; k < bins; k++) {
                magnitude[featureOffset + k] = (float) Math.log1p(Math.hypot(timeReal[k], timeImag[k]));
                phase[featureOffset + k] = (float) Math.atan2(timeImag[k], timeReal[k]);
            }
            System.arraycopy(input, inputOffset, previous, 0, hop);
        }
        transform.transform(magnitude, phase, outputReal, outputImag);
        for (int i = 0; i < outputReal.length; i++) {
            if (!Float.isFinite(outputReal[i]) || !Float.isFinite(outputImag[i])) {
                throw new IllegalStateException("Model produced a non-finite spectrum");
            }
        }
        float[] result = new float[(started ? FRAMES_PER_CALL : FRAMES_PER_CALL - 1) * hop];
        int outputOffset = 0;
        for (int frame = 0; frame < FRAMES_PER_CALL; frame++) {
            int featureOffset = frame * bins;
            for (int k = 0; k < bins; k++) {
                timeReal[k] = outputReal[featureOffset + k];
                timeImag[k] = (k == 0 || k == bins - 1) ? 0.0 : outputImag[featureOffset + k];
            }
            for (int k = bins; k < size; k++) {
                timeReal[k] = timeReal[size - k];
                timeImag[k] = -timeImag[size - k];
            }
            fft.transform(timeReal, timeImag, true);
            for (int i = 0; i < size; i++) {
                overlap[i] += timeReal[i] * window[i];
                weights[i] += window[i] * window[i];
            }
            if (started) {
                for (int i = 0; i < hop; i++) {
                    result[outputOffset + i] = weights[i] > 1e-12 ? (float) (overlap[i] / weights[i]) : 0.0f;
                    if (!Float.isFinite(result[outputOffset + i])) {
                        throw new IllegalStateException("Synthesis produced non-finite PCM");
                    }
                }
                outputOffset += hop;
            }
            System.arraycopy(overlap, hop, overlap, 0, hop);
            System.arraycopy(weights, hop, weights, 0, hop);
            Arrays.fill(overlap, hop, size, 0.0);
            Arrays.fill(weights, hop, size, 0.0);
            started = true;
        }
        return result;
    }

    /** Padding completes a graph call; only the one pending real hop is kept. */
    float[] flush(SpectrumTransform transform) throws OrtException {
        if (finished) throw new IllegalStateException("Stream already flushed");
        float[] result = started ? Arrays.copyOf(processFrames(new float[FRAMES_PER_CALL * hop], transform), hop) : null;
        finished = true;
        return result;
    }

    void reset() {
        Arrays.fill(previous, 0.0f);
        Arrays.fill(overlap, 0.0);
        Arrays.fill(weights, 0.0);
        started = false;
        finished = false;
    }
}
