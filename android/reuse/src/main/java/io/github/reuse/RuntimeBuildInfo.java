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

    /** Enumerates NNAPI device descriptors as "numeric device type\tname", without inference. */
    public static String[] nnapiDevices() {
        System.loadLibrary("reuse_ort_ops");
        return nativeNnapiDevices();
    }

    private static native String nativeRuntimeBuildInfo();
    private static native String[] nativeNnapiDevices();
}
