package io.github.reuse;

/** Fixed streaming format. The ONNX export must match this configuration exactly. */
public final class ReuseConfig {
    public final int sampleRate;
    public final int fftSize;
    public final int hopSize;
    public final int frequencyBins;
    public final int exitLayers;
    public final int intraOpThreads;

    public ReuseConfig(int sampleRate) { this(sampleRate, 3, 2); }

    public ReuseConfig(int sampleRate, int exitLayers, int intraOpThreads) {
        if (sampleRate != 8000 && sampleRate != 16000) {
            throw new IllegalArgumentException("Supported sample rates are 8000 and 16000 Hz");
        }
        if (exitLayers < 3 || exitLayers > 12) {
            throw new IllegalArgumentException("exitLayers must be between 3 and 12");
        }
        if (intraOpThreads < 1 || intraOpThreads > 16) {
            throw new IllegalArgumentException("intraOpThreads must be between 1 and 16");
        }
        this.sampleRate = sampleRate;
        this.fftSize = sampleRate * 40 / 1000;
        this.hopSize = fftSize / 2;
        this.frequencyBins = fftSize / 2 + 1;
        this.exitLayers = exitLayers;
        this.intraOpThreads = intraOpThreads;
    }

    /** Input needed before the first output block is returned, excluding computation. */
    public int startupInputSamples() { return fftSize; }

    /** The returned block describes the input block one hop before the newest input. */
    public int bufferedDelaySamples() { return hopSize; }
}
