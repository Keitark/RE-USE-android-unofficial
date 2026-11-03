package io.github.reuse;

import ai.onnxruntime.OrtEnvironment;

/** Reads the loaded ORT runtime's C API build identity, without choosing a backend. */
public final class RuntimeBuildInfo {
    private RuntimeBuildInfo() { }

    public static String read() {
        OrtEnvironment.getEnvironment();
        System.loadLibrary("reuse_ort_ops");
        return nativeRuntimeBuildInfo();
    }

    /**
     * JSON identity of the loaded custom-kernel library's compiled capabilities.
     * Build flags do not prove a particular kernel executed or establish numerical parity.
     */
    public static String kernelBuildInfo() {
        System.loadLibrary("reuse_ort_ops");
        return nativeKernelBuildInfo();
    }

    /** Enumerates NNAPI device descriptors as "numeric device type\tname", without inference. */
    public static String[] nnapiDevices() {
        System.loadLibrary("reuse_ort_ops");
        return nativeNnapiDevices();
    }

    private static native String nativeRuntimeBuildInfo();
    private static native String nativeKernelBuildInfo();
    private static native String[] nativeNnapiDevices();
}
