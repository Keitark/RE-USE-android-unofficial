package io.github.reuse;

import android.content.Context;
import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Stateful mono PCM speech enhancement. Each instance owns one stream and one ORT session.
 * Run on a dedicated worker thread, never in an AudioRecord/AAudio callback or on the UI thread.
 * PCM is normally in [-1,1]; no clipping or automatic gain control is applied.
 *
 * <p>Each process call consumes exactly 20 ms. The first returns null; the second returns the
 * first input block. With zero graph look-ahead the first output is available after 40 ms of
 * input plus computation. Call flush once to retrieve the final input block, then reset before
 * processing a new stream. Exactly N input blocks yield N output blocks including flush.
 */
public final class ReuseEnhancer implements AutoCloseable {
    private static final String MAGNITUDE = "magnitude";
    private static final String PHASE = "phase";
    private static final String REAL = "spectrum_real";
    private static final String IMAG = "spectrum_imag";

    private static final class TensorBuffer implements AutoCloseable {
        final FloatBuffer buffer;
        final OnnxTensor tensor;

        TensorBuffer(OrtEnvironment environment, long[] shape) throws OrtException {
            long count = 1;
            for (long dimension : shape) count = Math.multiplyExact(count, dimension);
            if (count > Integer.MAX_VALUE / 4) throw new IllegalArgumentException("Tensor is too large");
            buffer = ByteBuffer.allocateDirect((int) count * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
            tensor = OnnxTensor.createTensor(environment, buffer, shape);
        }

        void clear() {
            for (int i = 0; i < buffer.capacity(); i++) buffer.put(i, 0.0f);
        }

        @Override public void close() { tensor.close(); }
    }

    private static final class StateBuffers {
        final ModelManifest.State spec;
        TensorBuffer current;
        TensorBuffer next;

        StateBuffers(ModelManifest.State spec, TensorBuffer current, TensorBuffer next) {
            this.spec = spec;
            this.current = current;
            this.next = next;
        }
    }

    private final ReuseConfig config;
    private final StreamingStft dsp;
    private final List<TensorBuffer> ownedBuffers = new ArrayList<>();
    private final List<StateBuffers> states = new ArrayList<>();
    private final Map<String, OnnxTensor> inputTensors = new LinkedHashMap<>();
    private final Map<String, OnnxTensor> outputTensors = new LinkedHashMap<>();
    private final StreamingStft.SpectrumTransform transform = this::infer;
    private OrtSession.SessionOptions options;
    private OrtSession session;
    private Map<String, Object> executionDiagnostics = Collections.emptyMap();
    private TensorBuffer magnitude;
    private TensorBuffer phase;
    private TensorBuffer real;
    private TensorBuffer imag;
    private boolean closed;
    private boolean failed;

    public ReuseEnhancer(Path modelPath, Path manifestPath, ReuseConfig config)
            throws IOException, OrtException {
        this(modelPath, manifestPath, config, null);
    }

    /**
     * Loads either a portable graph or the bundled SelectiveScan custom operation.
     * The latter requires native libraries to be extracted by the consuming application:
     * {@code android { packaging { jniLibs { useLegacyPackaging = true } } }}.
     * No library path supplied by model metadata is executed.
     */
    public static ReuseEnhancer create(Context context, Path modelPath, Path manifestPath, ReuseConfig config)
            throws IOException, OrtException {
        if (context == null) throw new IllegalArgumentException("context is required");
        Path library = new java.io.File(context.getApplicationInfo().nativeLibraryDir,
                "libreuse_ort_ops.so").toPath();
        return new ReuseEnhancer(modelPath, manifestPath, config, library);
    }

    /**
     * Explicit custom-operation location for applications controlling native library deployment.
     * A portable graph ignores this argument. Custom graphs require the library bundled with
     * this release; the manifest can request only the supported operation, never a file path.
     */
    public ReuseEnhancer(Path modelPath, Path manifestPath, ReuseConfig config, Path customOpLibrary)
            throws IOException, OrtException {
        if (config == null) throw new IllegalArgumentException("config is required");
        this.config = config;
        this.dsp = new StreamingStft(config);
        ModelManifest manifest = ModelManifest.read(manifestPath, modelPath, config);
        if (manifest.framesPerCall != 1) {
            throw new IOException("This model requires ChunkedReuseEnhancer; ReuseEnhancer accepts one frame per call");
        }
        if (manifest.requiresSelectiveScan && (customOpLibrary == null || !Files.isRegularFile(customOpLibrary))) {
            throw new IOException("This model requires libreuse_ort_ops.so; use ReuseEnhancer.create(context, ...) " +
                    "with packaging.jniLibs.useLegacyPackaging=true, or provide its absolute path");
        }
        // OrtEnvironment is a process-wide singleton; this instance must not close other users' environment.
        OrtEnvironment environment = OrtEnvironment.getEnvironment();
        try {
            options = new OrtSession.SessionOptions();
            options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
            options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL);
            options.setIntraOpNumThreads(config.intraOpThreads);
            options.setInterOpNumThreads(1);
            options.addConfigEntry("session.intra_op.allow_spinning", "0");
            options.addConfigEntry("session.inter_op.allow_spinning", "0");
            BackendSessionOptions.apply(config, options);
            executionDiagnostics = BackendSessionOptions.diagnostics(config, environment);
            if (manifest.requiresSelectiveScan) {
                options.registerCustomOpLibrary(customOpLibrary.toAbsolutePath().toString());
            }
            session = environment.createSession(modelPath.toAbsolutePath().toString(), options);
            validateGraph(manifest);
            long[] spectrumShape = {1, config.frequencyBins};
            magnitude = allocate(environment, spectrumShape);
            phase = allocate(environment, spectrumShape);
            real = allocate(environment, spectrumShape);
            imag = allocate(environment, spectrumShape);
            inputTensors.put(MAGNITUDE, magnitude.tensor);
            inputTensors.put(PHASE, phase.tensor);
            outputTensors.put(REAL, real.tensor);
            outputTensors.put(IMAG, imag.tensor);
            for (ModelManifest.State state : manifest.states) {
                StateBuffers buffers = new StateBuffers(state,
                        allocate(environment, state.shape), allocate(environment, state.shape));
                states.add(buffers);
                inputTensors.put(state.input, buffers.current.tensor);
                outputTensors.put(state.output, buffers.next.tensor);
            }
        } catch (IOException | OrtException | RuntimeException exception) {
            try { close(); } catch (OrtException closeError) { exception.addSuppressed(closeError); }
            throw exception;
        }
    }

    public ReuseConfig config() { return config; }

    /** Requested backend/runtime metadata. Provider availability is not measured GPU coverage. */
    public Map<String, Object> executionDiagnostics() { return executionDiagnostics; }

    /**
     * Call after zero-audio warm-up and before recording. Stops startup profiling and rejects
     * an NNAPI request that actually delegated no kernels. CPU mode is a no-op.
     */
    public synchronized void verifyAcceleratorExecution() throws IOException, OrtException {
        checkReady();
        if (config.backend == ReuseConfig.Backend.CPU
                || Boolean.TRUE.equals(executionDiagnostics.get("nnapi_execution_verified"))) return;
        executionDiagnostics = BackendSessionOptions.finishVerification(session, executionDiagnostics);
        if (!Boolean.TRUE.equals(executionDiagnostics.get("nnapi_execution_verified"))) {
            failed = true;
            throw new IOException("NNAPI delegated no kernels during warm-up; select CPU mode instead");
        }
    }

    /** Returns the preceding input block, or null on the first call. */
    public synchronized float[] process(float[] monoPcm) throws OrtException {
        checkReady();
        try {
            return dsp.process(monoPcm, transform);
        } catch (OrtException exception) {
            failed = true;
            throw exception;
        } catch (IllegalStateException exception) {
            failed = true;
            throw exception;
        }
    }

    /** Returns the last input block (null for an empty stream) and terminates the stream. */
    public synchronized float[] flush() throws OrtException {
        checkReady();
        try {
            return dsp.flush(transform);
        } catch (OrtException | IllegalStateException exception) {
            failed = true;
            throw exception;
        }
    }

    /** Clears all model, analysis and synthesis history. The loaded model is retained. */
    public synchronized void reset() {
        if (closed) throw new IllegalStateException("Enhancer is closed");
        for (TensorBuffer buffer : ownedBuffers) buffer.clear();
        dsp.reset();
        failed = false;
    }

    private void checkReady() {
        if (closed) throw new IllegalStateException("Enhancer is closed");
        if (failed) throw new IllegalStateException("Stream failed; call reset before reusing it");
    }

    private TensorBuffer allocate(OrtEnvironment environment, long[] shape) throws OrtException {
        TensorBuffer result = new TensorBuffer(environment, shape);
        ownedBuffers.add(result);
        return result;
    }

    private void infer(float[] inputMagnitude, float[] inputPhase, float[] outputReal, float[] outputImag)
            throws OrtException {
        for (int i = 0; i < config.frequencyBins; i++) {
            magnitude.buffer.put(i, inputMagnitude[i]);
            phase.buffer.put(i, inputPhase[i]);
        }
        // Pinned outputs and double-buffered recurrent state avoid copying native state per hop.
        try (OrtSession.Result ignored = session.run(inputTensors, outputTensors)) {
            for (int i = 0; i < config.frequencyBins; i++) {
                outputReal[i] = real.buffer.get(i);
                outputImag[i] = imag.buffer.get(i);
            }
        }
        for (StateBuffers state : states) {
            TensorBuffer old = state.current;
            state.current = state.next;
            state.next = old;
            inputTensors.put(state.spec.input, state.current.tensor);
            outputTensors.put(state.spec.output, state.next.tensor);
        }
    }

    private void validateGraph(ModelManifest manifest) throws OrtException, IOException {
        Map<String, NodeInfo> inputs = session.getInputInfo();
        Map<String, NodeInfo> outputs = session.getOutputInfo();
        Set<String> expectedInputs = new HashSet<>(Arrays.asList(MAGNITUDE, PHASE));
        Set<String> expectedOutputs = new HashSet<>(Arrays.asList(REAL, IMAG));
        long[] spectrumShape = {1, config.frequencyBins};
        requireTensor(inputs, MAGNITUDE, spectrumShape);
        requireTensor(inputs, PHASE, spectrumShape);
        requireTensor(outputs, REAL, spectrumShape);
        requireTensor(outputs, IMAG, spectrumShape);
        for (ModelManifest.State state : manifest.states) {
            expectedInputs.add(state.input);
            expectedOutputs.add(state.output);
            requireTensor(inputs, state.input, state.shape);
            requireTensor(outputs, state.output, state.shape);
        }
        if (!inputs.keySet().equals(expectedInputs) || !outputs.keySet().equals(expectedOutputs)) {
            throw new IOException("Model graph input/output names do not match the manifest");
        }
    }

    private static void requireTensor(Map<String, NodeInfo> nodes, String name, long[] shape)
            throws IOException {
        NodeInfo node = nodes.get(name);
        if (node == null || !(node.getInfo() instanceof TensorInfo)) {
            throw new IOException("Missing tensor " + name);
        }
        TensorInfo info = (TensorInfo) node.getInfo();
        if (info.type != OnnxJavaType.FLOAT || !Arrays.equals(info.getShape(), shape)) {
            throw new IOException("Expected float32 tensor " + name + " " + Arrays.toString(shape));
        }
    }

    @Override public synchronized void close() throws OrtException {
        if (closed) return;
        closed = true;
        for (TensorBuffer buffer : ownedBuffers) buffer.close();
        ownedBuffers.clear();
        states.clear();
        inputTensors.clear();
        outputTensors.clear();
        OrtException failure = null;
        if (session != null) {
            try { session.close(); } catch (OrtException exception) { failure = exception; }
            session = null;
        }
        if (options != null) { options.close(); options = null; }
        if (failure != null) throw failure;
    }
}
