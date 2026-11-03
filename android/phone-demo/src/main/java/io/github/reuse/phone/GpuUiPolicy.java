package io.github.reuse.phone;

import java.util.Locale;

/** Presentation rules for optional acceleration; these do not select or start a backend. */
final class GpuUiPolicy {
    static final String UNAVAILABLE = "GPU unavailable through NNAPI. CPU mode is available.";

    private GpuUiPolicy() { }

    static boolean canChooseGpu(boolean active, boolean probeComplete, boolean available) {
        return !active && probeComplete && available;
    }

    static boolean isUnsupportedGpu(String message) {
        String value = message == null ? "" : message.toLowerCase(Locale.ROOT);
        return value.contains("gpu-only nnapi found no gpu driver")
                || value.contains("nnapi delegated no kernels")
                || value.contains("runtime does not have the gpu-only device filter");
    }

    static String readableError(String message) {
        if (isUnsupportedGpu(message))
            return UNAVAILABLE + " Recording did not start. CPU mode is selected; press Start recording to retry.";
        if (message == null || message.trim().isEmpty())
            return "Unable to record. Please try again. Details are available in Share diagnostics.";
        if (message.length() > 240 || message.contains("OrtException") || message.contains("onnxruntime/"))
            return "Processing could not continue. Try CPU mode. Full details are saved in Share diagnostics (no audio).";
        return message;
    }

    static String backendStatus(boolean probeComplete, boolean available, String availabilityMessage,
                                boolean gpuRequested, boolean acceleratorExecuted) {
        if (!probeComplete) return "CPU mode is available · checking GPU support…";
        if (!available) return availabilityMessage;
        if (gpuRequested)
            return acceleratorExecuted ? "GPU driver delegation observed · hybrid GPU + CPU"
                    : "GPU requested · checking delegated execution before recording";
        return "CPU mode · " + availabilityMessage;
    }
}
