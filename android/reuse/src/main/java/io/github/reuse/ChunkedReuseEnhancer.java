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
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Stateful enhancement using a fixed five-hop (100 ms) causal model graph.
 * Run on a dedicated worker thread, never the UI or an audio callback.
 *
 * <p>Each process call accepts exactly {@code inputChunkSamples()} mono floats.
 * The first call returns 80 ms and subsequent calls return 100 ms. A final
 * {@link #flush()} returns the pending 20 ms, so total output length equals
 * total input length. Batching adds up to 80 ms collection delay to the
 * ordinary one-hop synthesis delay; it does not add model look-ahead.
 *
 * <p>The supplied model must declare {@code frames_per_call=5}, with feature
 * tensors shaped {@code [5, frequencyBins]}. Use {@link ReuseEnhancer} for a
 * one-hop model. PCM is normally in [-1,1]; no clipping or gain is applied.
 */
public final class ChunkedReuseEnhancer implements AutoCloseable {
    public static final int FRAMES_PER_CALL = 5;
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
    private final ChunkedStreamingStft dsp;
    private final List<TensorBuffer> ownedBuffers = new ArrayList<>();
    private final List<StateBuffers> states = new ArrayList<>();
    private final Map<String, OnnxTensor> inputTensors = new LinkedHashMap<>();
    private final Map<String, OnnxTensor> outputTensors = new LinkedHashMap<>();
    private final ChunkedStreamingStft.SpectrumTransform transform = this::infer;
    private OrtSession.SessionOptions options;
    private OrtSession session;
    private TensorBuffer magnitude;
    private TensorBuffer phase;
    private TensorBuffer real;
    private TensorBuffer imag;
    private boolean closed;
    private boolean failed;

    public ChunkedReuseEnhancer(Path modelPath, Path manifestPath, ReuseConfig config)
            throws IOException, OrtException {
        this(modelPath, manifestPath, config, null);
    }

    /**
     * Loads either a portable graph or the bundled SelectiveScan custom operation.
     * The latter requires native libraries to be extracted by the consuming application:
     * {@code android { packaging { jniLibs { useLegacyPackaging = true } } }}.
     * No library path supplied by model metadata is executed.
     */
    public static ChunkedReuseEnhancer create(Context context, Path modelPath, Path manifestPath, ReuseConfig config)
            throws IOException, OrtException {
        if (context == null) throw new IllegalArgumentException("context is required");
        Path library = new java.io.File(context.getApplicationInfo().nativeLibraryDir,
                "libreuse_ort_ops.so").toPath();
        return new ChunkedReuseEnhancer(modelPath, manifestPath, config, library);
    }

    /**
     * Explicit custom-operation location for applications controlling native library deployment.
     * A portable graph ignores this argument. Custom graphs require the library bundled with
     * this release; the manifest can request only the supported operation, never a file path.
     */
    public ChunkedReuseEnhancer(Path modelPath, Path manifestPath, ReuseConfig config, Path customOpLibrary)
            throws IOException, OrtException {
        if (config == null) throw new IllegalArgumentException("config is required");
        this.config = config;
        this.dsp = new ChunkedStreamingStft(config);
        ModelManifest manifest = ModelManifest.read(manifestPath, modelPath, config);
        if (manifest.framesPerCall != FRAMES_PER_CALL) {
            throw new IOException("ChunkedReuseEnhancer requires a five-frame model (frames_per_call=5); use ReuseEnhancer for a one-frame model");
        }
        if (manifest.requiresSelectiveScan && (customOpLibrary == null || !Files.isRegularFile(customOpLibrary))) {
            throw new IOException("This model requires libreuse_ort_ops.so; use ChunkedReuseEnhancer.create(context, ...) " +
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
            if (manifest.requiresSelectiveScan) {
                options.registerCustomOpLibrary(customOpLibrary.toAbsolutePath().toString());
            }
            session = environment.createSession(modelPath.toAbsolutePath().toString(), options);
            validateGraph(manifest);
            long[] spectrumShape = {FRAMES_PER_CALL, config.frequencyBins};
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

    /** Exactly 100 ms of mono PCM at the configured sample rate. */
    public int inputChunkSamples() { return FRAMES_PER_CALL * config.hopSize; }

    /** Returns four hops on the first call and five on subsequent calls. */
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

    /** Returns the final pending hop (null for an empty stream) and terminates the stream. */
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
        for (int i = 0; i < FRAMES_PER_CALL * config.frequencyBins; i++) {
            magnitude.buffer.put(i, inputMagnitude[i]);
            phase.buffer.put(i, inputPhase[i]);
        }
        // Pinned outputs and double-buffered recurrent state avoid copying native state per chunk.
        try (OrtSession.Result ignored = session.run(inputTensors, outputTensors)) {
            for (int i = 0; i < FRAMES_PER_CALL * config.frequencyBins; i++) {
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
        long[] spectrumShape = {FRAMES_PER_CALL, config.frequencyBins};
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
