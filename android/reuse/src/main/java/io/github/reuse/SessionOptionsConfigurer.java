package io.github.reuse;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.io.IOException;

/**
 * Optional session configuration supplied by a separately packaged execution backend.
 * The caller owns the configurer and must keep its runtime resources alive until the enhancer closes.
 */
@FunctionalInterface
public interface SessionOptionsConfigurer {
    /** Called after the enhancer's standard options and custom operations are registered. */
    void configure(OrtEnvironment environment, OrtSession.SessionOptions options) throws IOException, OrtException;
}
