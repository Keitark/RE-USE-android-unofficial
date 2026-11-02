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

    private static native String nativeRuntimeBuildInfo();
}
