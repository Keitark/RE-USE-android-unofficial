package io.github.reuse;

import java.nio.file.Path;

/** Fixed streaming format. The ONNX export must match this configuration exactly. */
public final class ReuseConfig {
    /** NNAPI may use GPU or another accelerator unless the application supplies a stricter runtime. */
    public enum Backend { CPU, NNAPI_ACCELERATOR }

    public final int sampleRate;
    public final int fftSize;
    public final int hopSize;
    public final int frequencyBins;
    public final int exitLayers;
    public final int intraOpThreads;
    public final Backend backend;
    /** Requests NNAPI's relaxed FP16 arithmetic; model IO and recurrent states remain float32. */
    public final boolean nnapiUseFp16;
    /** Writable app-private prefix for the short accelerator warm-up profile; null for CPU. */
    public final Path nnapiProfilingPrefix;

    public ReuseConfig(int sampleRate) { this(sampleRate, 3, 2); }

    public ReuseConfig(int sampleRate, int exitLayers, int intraOpThreads) {
        this(sampleRate, exitLayers, intraOpThreads, Backend.CPU, false, null);
    }

    /**
     * Opt-in hybrid NNAPI execution, with an explicit writable profile prefix to verify
     * delegated warm-up work. Provider registration alone does not prove GPU execution.
     */
    public ReuseConfig(int sampleRate, int exitLayers, int intraOpThreads,
                       Backend backend, boolean nnapiUseFp16, Path nnapiProfilingPrefix) {
        if (sampleRate != 8000 && sampleRate != 16000) {
            throw new IllegalArgumentException("Supported sample rates are 8000 and 16000 Hz");
        }
        if (exitLayers < 3 || exitLayers > 12) {
            throw new IllegalArgumentException("exitLayers must be between 3 and 12");
        }
        if (intraOpThreads < 1 || intraOpThreads > 16) {
            throw new IllegalArgumentException("intraOpThreads must be between 1 and 16");
        }
        if (backend == null) throw new IllegalArgumentException("backend is required");
        if (backend == Backend.CPU && nnapiUseFp16)
            throw new IllegalArgumentException("NNAPI FP16 relaxation requires the NNAPI_ACCELERATOR backend");
        if (backend == Backend.NNAPI_ACCELERATOR && nnapiProfilingPrefix == null)
            throw new IllegalArgumentException("NNAPI acceleration requires a writable warm-up profiling prefix");
        if (backend == Backend.CPU && nnapiProfilingPrefix != null)
            throw new IllegalArgumentException("NNAPI profiling prefix requires the NNAPI_ACCELERATOR backend");
        this.sampleRate = sampleRate;
        this.fftSize = sampleRate * 40 / 1000;
        this.hopSize = fftSize / 2;
        this.frequencyBins = fftSize / 2 + 1;
        this.exitLayers = exitLayers;
        this.intraOpThreads = intraOpThreads;
        this.backend = backend;
        this.nnapiUseFp16 = nnapiUseFp16;
        this.nnapiProfilingPrefix = nnapiProfilingPrefix;
    }

    /** Input needed before the first output block is returned, excluding computation. */
    public int startupInputSamples() { return fftSize; }

    /** The returned block describes the input block one hop before the newest input. */
    public int bufferedDelaySamples() { return hopSize; }
}
