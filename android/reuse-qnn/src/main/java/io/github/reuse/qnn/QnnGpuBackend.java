package io.github.reuse.qnn;

import android.content.Context;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.github.reuse.SessionOptionsConfigurer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;

/**
 * Experimental direct QNN GPU configuration, bypassing NNAPI. A QNN registration token
 * reported as NPU is retained as such in diagnostics. Provider events alone never prove GPU
 * hardware use; callers must also check the loaded backend and numerical results.
 */
public final class QnnGpuBackend implements SessionOptionsConfigurer, AutoCloseable {
    private final QnnHtpBackend delegate;

    public QnnGpuBackend(Context context, Path privateProfilePrefix, boolean allowCpuFallback) throws IOException {
        delegate = QnnHtpBackend.gpu(context, privateProfilePrefix, allowCpuFallback);
    }

    /**
     * Adds bounded-probe graph diagnostics in an existing writable app-private directory.
     * The dumps may contain model structure or weights and are not automatically shared.
     */
    public QnnGpuBackend(Context context, Path privateProfilePrefix, boolean allowCpuFallback,
                         Path privateDiagnosticDirectory) throws IOException {
        if (privateDiagnosticDirectory == null)
            throw new IllegalArgumentException("Private diagnostic directory is required");
        delegate = QnnHtpBackend.gpu(context, privateProfilePrefix, allowCpuFallback, privateDiagnosticDirectory);
    }

    @Override public void configure(OrtEnvironment environment, OrtSession.SessionOptions options)
            throws IOException, OrtException {
        delegate.configure(environment, options);
    }

    public Map<String, Object> diagnostics() { return delegate.diagnostics(); }

    public Map<String, Object> verifyProfile(Path profile, int expectedCalls, boolean requireSinglePartition)
            throws IOException {
        return delegate.verifyProfile(profile, expectedCalls, requireSinglePartition);
    }

    @Override public void close() throws IOException { delegate.close(); }
}
