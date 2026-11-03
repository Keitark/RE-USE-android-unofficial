package io.github.reuse;

/**
 * Named baseline for the bundled causal FP32, 8 kHz, exit-three, five-hop model.
 * Four threads are a measured Xperia starting point, not a speed guarantee for every device.
 * Applications still verify model bytes and report their actual runtime/native build identity.
 */
public final class RecommendedCpuProfile {
    public static final String ID = "reuse-8k-exit3-fp32-cpu";
    public static final String MODEL_SHA256 =
            "f950dcfb9321191b2bcb3ab9afa3e100c40e5c64dd1de1d30849be4c1aacce32";
    public static final int SAMPLE_RATE = 8000;
    public static final int EXIT_LAYERS = 3;
    public static final int FRAMES_PER_CALL = 5;
    public static final int DEFAULT_THREADS = 4;
    public static final String ORT_VERSION = "1.29.0";

    private RecommendedCpuProfile() { }

    /** Explicit baseline; generic {@link ReuseConfig} constructors keep their existing defaults. */
    public static ReuseConfig config() { return config(DEFAULT_THREADS); }

    /** Same model/precision with a caller-selected CPU pool for device comparisons. */
    public static ReuseConfig config(int intraOpThreads) {
        return new ReuseConfig(SAMPLE_RATE, EXIT_LAYERS, intraOpThreads,
                ReuseConfig.Backend.CPU, false, null);
    }
}
