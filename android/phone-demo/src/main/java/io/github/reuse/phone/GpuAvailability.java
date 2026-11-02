package io.github.reuse.phone;

import java.util.Arrays;

/** Driver availability only; execution coverage and performance require separate checks. */
public final class GpuAvailability {
    private static final String MARKER = "REUSE_NNAPI_GPU_ONLY_V1";
    private static final String GPU_TYPE_PREFIX = "3\t"; // Android NNAPI DEVICE_GPU.
    public final boolean available;
    public final String message;
    public final String runtimeBuildInfo;
    public final String error;
    private final String[] descriptions;

    private GpuAvailability(boolean available, String message, String runtime, String error, String[] descriptions) {
        this.available = available; this.message = message; this.runtimeBuildInfo = runtime;
        this.error = error; this.descriptions = descriptions.clone();
    }

    static GpuAvailability fromDevices(String runtime, String[] descriptions) {
        if (runtime == null || !runtime.contains(MARKER))
            return unavailable("GPU runtime could not be verified. CPU mode is available.",
                    "Missing GPU-only runtime marker", runtime == null ? "" : runtime);
        if (descriptions == null) throw new IllegalArgumentException("NNAPI device descriptions are required");
        boolean gpu = Arrays.stream(descriptions).anyMatch(row -> row != null
                && row.startsWith(GPU_TYPE_PREFIX) && row.length() > GPU_TYPE_PREFIX.length());
        return new GpuAvailability(gpu, gpu
                ? "NNAPI GPU driver detected. GPU execution and performance still require a test."
                : "GPU unavailable through NNAPI on this phone. CPU mode is available.",
                runtime, "", descriptions);
    }

    static GpuAvailability unavailable(String message, String error, String runtime) {
        return new GpuAvailability(false, message, runtime, error, new String[0]);
    }

    public String[] deviceDescriptions() { return descriptions.clone(); }
}
