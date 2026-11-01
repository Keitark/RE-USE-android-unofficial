package io.github.reuse;

import ai.onnxruntime.OrtException;
import java.util.Arrays;

/** Causal 40 ms periodic-Hann analysis, 20 ms hop and squared-window OLA. */
final class StreamingStft {
    interface SpectrumTransform {
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

    StreamingStft(ReuseConfig config) {
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
        magnitude = new float[bins];
        phase = new float[bins];
        outputReal = new float[bins];
        outputImag = new float[bins];
        for (int i = 0; i < size; i++) {
            window[i] = 0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / size);
        }
    }

    float[] process(float[] input, SpectrumTransform transform) throws OrtException {
        if (finished) throw new IllegalStateException("Stream flushed; call reset before process");
        if (input == null || input.length != hop) {
            throw new IllegalArgumentException("Expected exactly " + hop + " mono samples");
        }
        for (float value : input) {
            if (!Float.isFinite(value)) throw new IllegalArgumentException("PCM must be finite");
        }
        return processFrame(input, transform);
    }

    private float[] processFrame(float[] input, SpectrumTransform transform) throws OrtException {
        for (int i = 0; i < hop; i++) {
            timeReal[i] = previous[i] * window[i];
            timeReal[hop + i] = input[i] * window[hop + i];
        }
        Arrays.fill(timeImag, 0.0);
        fft.transform(timeReal, timeImag, false);
        // Real FFT endpoints are real-valued. Canonical +0 avoids an erroneous -pi branch
        // from atan2 when round-off leaves a tiny negative imaginary part at Nyquist.
        timeImag[0] = +0.0;
        timeImag[bins - 1] = +0.0;
        for (int k = 0; k < bins; k++) {
            magnitude[k] = (float) Math.log1p(Math.hypot(timeReal[k], timeImag[k]));
            phase[k] = (float) Math.atan2(timeImag[k], timeReal[k]);
        }
        transform.transform(magnitude, phase, outputReal, outputImag);
        for (int k = 0; k < bins; k++) {
            if (!Float.isFinite(outputReal[k]) || !Float.isFinite(outputImag[k])) {
                throw new IllegalStateException("Model produced a non-finite spectrum");
            }
            timeReal[k] = outputReal[k];
            timeImag[k] = (k == 0 || k == bins - 1) ? 0.0 : outputImag[k];
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
        float[] result = null;
        if (started) {
            result = new float[hop];
            for (int i = 0; i < hop; i++) {
                result[i] = weights[i] > 1e-12 ? (float) (overlap[i] / weights[i]) : 0.0f;
                if (!Float.isFinite(result[i])) {
                    throw new IllegalStateException("Synthesis produced non-finite PCM");
                }
            }
        }
        System.arraycopy(overlap, hop, overlap, 0, hop);
        System.arraycopy(weights, hop, weights, 0, hop);
        Arrays.fill(overlap, hop, size, 0.0);
        Arrays.fill(weights, hop, size, 0.0);
        System.arraycopy(input, 0, previous, 0, hop);
        started = true;
        return result;
    }

    float[] flush(SpectrumTransform transform) throws OrtException {
        if (finished) throw new IllegalStateException("Stream already flushed");
        float[] result = started ? processFrame(new float[hop], transform) : null;
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
