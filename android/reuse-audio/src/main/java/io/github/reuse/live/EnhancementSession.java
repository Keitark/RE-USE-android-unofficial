package io.github.reuse.live;

import io.github.reuse.ChunkedReuseEnhancer;
import io.github.reuse.ReuseConfig;
import java.io.IOException;
import java.io.RandomAccessFile;
import java.io.PrintWriter;
import java.io.Writer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.charset.StandardCharsets;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * One explicit 16 kHz PCM session, owning its bounded queue and processing worker.
 * Callbacks execute on that worker; onAudio must hand off to a nonblocking writer.
 * No microphone capture, Android dependency, processing-path file IO, or codec loading.
 */
public final class EnhancementSession implements AutoCloseable {
    /** Zero-input engine calls before backend verification, reset and onReady. */
    public static final int MODEL_WARMUP_CALLS = 3;
    public interface Listener {
        void onReady();
        void onAudio(byte[] pcm16le);
        void onFinished(Map<String, Object> statistics);
        void onError(String code, String message, Map<String, Object> statistics);
    }
    /**
     * Owned 8 kHz mono engine used by the PCM adapter. Each process call accepts 800 samples,
     * returns 640 samples after reset on its first call and 800 thereafter; flush returns the
     * remaining 160 samples, or null for an empty stream. Methods run on the session worker.
     */
    public interface Engine extends AutoCloseable {
        float[] process(float[] input) throws Exception;
        float[] flush() throws Exception;
        void reset();
        void close() throws Exception;
        /** Called after three zero-input warm-up calls and before reset and onReady. */
        default void verifyAcceleratorExecution() throws Exception { }
        default Map<String, Object> executionDiagnostics() { return Collections.emptyMap(); }
    }
    /** Creates an exclusively owned engine on the session worker, after its initializer runs. */
    @FunctionalInterface
    public interface EngineFactory { Engine create() throws Exception; }
    private static final int RATE = 16000, CHUNK = 800, QUEUE_SAMPLES = 8000;
    private static final int MAX_INPUT_SAMPLES = 120 * RATE, FIR_TAIL_SAMPLES = 188;
    private final Object lock = new Object();
    private final ArrayDeque<byte[]> queue = new ArrayDeque<>();
    private final Path outputDir;
    private final Listener listener;
    private final Runnable workerInitializer;
    private final EngineFactory factory;
    private final int queueLimit;
    // Bounded 120-second captures. Write only after processing has ended.
    private final byte[] raw = new byte[MAX_INPUT_SAMPLES * 2];
    private final byte[] enhanced = new byte[(MAX_INPUT_SAMPLES + FIR_TAIL_SAMPLES + 2) * 2];
    private final float[] modelInput = new float[CHUNK], upOutput = new float[CHUNK * 2];
    private final float[] pair = new float[2], lowPair = new float[1];
    private final float[] downInput = new float[QUEUE_SAMPLES + 2], downOutput = new float[QUEUE_SAMPLES / 2 + 1];
    private final float[] downZeros = new float[94], lowTail = new float[47], upZeros = new float[47];
    private Thread worker;
    private String state = "NEW", failureCode, failureMessage;
    private volatile boolean finishRequested, abortRequested;
    private int queuedSamples, maxQueuedSamples, rawBytes, enhancedBytes, fill;
    private boolean haveOddSample;
    private float oddSample;
    private long acceptedSamples, processedSamples, droppedSamples, rejectedAfterStopSamples;
    private long validLowSamples, forwardedLowSamples, modelInputSamples, modelOutputSamples;
    private long oddPadSamples, firDrainInputSamples, firTailOutputSamples, modelPaddingSamples;
    private long calls, processNs, maxProcessNs, deadlineMisses, warmupNs, loadNs, flushNs, firNs;
    private long queueAgeMaxNs, sessionStartNs, firstOutputNs, queueOverflows, sequenceGaps;
    private long firstInputNs;
    private long workerNs, drainNs;
    private final long[] callTimes = new long[1300];
    private int flushCalls;
    private boolean sampleIndexValidation;
    private ReuseConfig.Backend requestedBackend = ReuseConfig.Backend.CPU;
    private boolean nnapiUseFp16;
    private Map<String, Object> backendDiagnostics = Collections.emptyMap();

    public EnhancementSession(Path modelDir, Path outputDir, Path customOpLibrary, Listener listener) {
        this(modelDir, outputDir, customOpLibrary, listener, () -> { });
    }

    /** Initializer runs before ORT creates helper threads (e.g. Android audio priority). */
    public EnhancementSession(Path modelDir, Path outputDir, Path customOpLibrary, Listener listener,
                              Runnable workerInitializer) {
        this(modelDir, outputDir, customOpLibrary, listener, workerInitializer, 4);
    }

    /** Configurable CPU threads; existing constructors retain their four-thread configuration. */
    public EnhancementSession(Path modelDir, Path outputDir, Path customOpLibrary, Listener listener,
                              Runnable workerInitializer, int intraOpThreads) {
        this(modelDir, outputDir, customOpLibrary, listener, workerInitializer, intraOpThreads,
                ReuseConfig.Backend.CPU, false);
    }

    /**
     * Opt-in hybrid NNAPI processing. Verifies delegated work during zero-audio warm-up before
     * onReady; it does not prove GPU selection on an ordinary NNAPI runtime.
     */
    public EnhancementSession(Path modelDir, Path outputDir, Path customOpLibrary, Listener listener,
                              Runnable workerInitializer, int intraOpThreads,
                              ReuseConfig.Backend backend, boolean nnapiUseFp16) {
        this(outputDir, listener, workerInitializer,
                modelFactory(modelDir, outputDir, customOpLibrary, intraOpThreads, backend, nnapiUseFp16), QUEUE_SAMPLES);
        this.requestedBackend = backend;
        this.nnapiUseFp16 = nnapiUseFp16;
    }

    /**
     * Uses a separately supplied engine while retaining the bounded 500 ms PCM queue, FIR
     * conversion, padding, recording and terminal lifecycle. The session creates and closes
     * the engine on its worker; the factory must not return an engine shared with another session.
     * Accelerator implementations belong in optional adapters and report their evidence through
     * {@link Engine#executionDiagnostics()}; the PCM adapter does not infer execution from loading.
     */
    public static EnhancementSession withEngine(Path outputDir, Listener listener,
                                                 Runnable workerInitializer, EngineFactory factory) {
        return new EnhancementSession(outputDir, listener, workerInitializer, factory, QUEUE_SAMPLES);
    }

    private static EngineFactory modelFactory(Path modelDir, Path outputDir, Path customOpLibrary,
                                             int intraOpThreads, ReuseConfig.Backend backend, boolean nnapiUseFp16) {
        ReuseConfig config = new ReuseConfig(8000, 3, intraOpThreads, backend, nnapiUseFp16,
                backend == ReuseConfig.Backend.NNAPI_ACCELERATOR && outputDir != null
                        ? outputDir.resolve("nnapi-warmup") : null);
        return () -> {
            if (backend == ReuseConfig.Backend.NNAPI_ACCELERATOR) Files.createDirectories(outputDir);
            final ChunkedReuseEnhancer e = new ChunkedReuseEnhancer(modelDir.resolve("model.onnx"),
                    modelDir.resolve("manifest.json"), config, customOpLibrary);
            return new Engine() {
                public float[] process(float[] input) throws Exception { return e.process(input); }
                public float[] flush() throws Exception { return e.flush(); }
                public void reset() { e.reset(); }
                public void close() throws Exception { e.close(); }
                public void verifyAcceleratorExecution() throws Exception { e.verifyAcceleratorExecution(); }
                public Map<String, Object> executionDiagnostics() { return e.executionDiagnostics(); }
            };
        };
    }

    // Dependency injection exercises the actual queue, FIR, padding, recording and lifecycle.
    EnhancementSession(Path outputDir, Listener listener, Runnable initializer, EngineFactory factory, int queueLimit) {
        if (outputDir == null || listener == null || initializer == null || factory == null || queueLimit <= 0)
            throw new IllegalArgumentException("Session arguments are required");
        this.outputDir = outputDir; this.listener = listener; this.workerInitializer = initializer;
        this.factory = factory; this.queueLimit = queueLimit;
    }

    public void start() {
        synchronized (lock) {
            if (!state.equals("NEW")) throw new IllegalStateException("Session already started");
            state = "PREPARING";
            worker = new Thread(this::run, "reuse-live-enhancer");
            worker.start();
        }
    }

    public boolean accept16kS16LE(byte[] data) { return accept(data, -1); }
    public boolean accept16kS16LE(byte[] data, long firstSample) {
        if (firstSample < 0) throw new IllegalArgumentException("Negative source sample index");
        return accept(data, firstSample);
    }
    private boolean accept(byte[] data, long firstSample) {
        if (data == null || (data.length & 1) != 0) throw new IllegalArgumentException("PCM16LE requires even byte count");
        int samples = data.length / 2;
        synchronized (lock) {
            if (!state.equals("RUNNING") || finishRequested || abortRequested || failureCode != null) {
                rejectedAfterStopSamples += samples; return false;
            }
            if (firstSample >= 0) {
                sampleIndexValidation = true;
                if (firstSample != acceptedSamples) {
                    sequenceGaps++; droppedSamples += samples;
                    failLocked("SEQUENCE_GAP", "Source sample index is discontinuous"); return false;
                }
            }
            if (acceptedSamples + samples > MAX_INPUT_SAMPLES) {
                droppedSamples += samples; failLocked("DURATION_LIMIT", "120-second recording/session limit reached"); return false;
            }
            if (samples > queueLimit - queuedSamples) {
                queueOverflows++; droppedSamples += samples;
                failLocked("INPUT_OVERFLOW", "Bounded 500 ms input queue overflow"); return false;
            }
            if (samples == 0) return true;
            long arrival = System.nanoTime();
            if (firstInputNs == 0) firstInputNs = arrival;
            queue.addLast(data.clone()); arrivalTimes.addLast(arrival);
            acceptedSamples += samples; queuedSamples += samples;
            maxQueuedSamples = Math.max(maxQueuedSamples, queuedSamples);
            lock.notifyAll(); return true;
        }
    }
    private final ArrayDeque<Long> arrivalTimes = new ArrayDeque<>();
    private void failLocked(String code, String message) {
        if (failureCode == null) { failureCode = code; failureMessage = message; }
        lock.notifyAll();
    }

    /** Ordered asynchronous stop: all successfully accepted audio is drained first. */
    public void finish() {
        synchronized (lock) {
            if (state.equals("NEW")) throw new IllegalStateException("Start the session first");
            finishRequested = true; lock.notifyAll();
        }
    }

    /** Asynchronous abort. Resources/captures are finalized by the worker. */
    @Override public void close() {
        synchronized (lock) {
            if (state.equals("NEW")) { state = "CLOSED"; return; }
            if (state.equals("FINISHED") || state.equals("ERROR") || state.equals("CLOSED")) return;
            abortRequested = true; lock.notifyAll();
        }
    }

    private void run() {
        Engine engine = null; FirResampler down = null, up = null;
        String terminalCode = null, terminalMessage = null;
        Throwable terminalFailure = null;
        try {
            workerInitializer.run();
            long begin = System.nanoTime(); engine = factory.create(); loadNs = System.nanoTime() - begin;
            synchronized (lock) { backendDiagnostics = new LinkedHashMap<>(engine.executionDiagnostics()); }
            begin = System.nanoTime();
            for (int i = 0; i < MODEL_WARMUP_CALLS; i++) engine.process(new float[CHUNK]);
            try { engine.verifyAcceleratorExecution(); }
            finally { synchronized (lock) { backendDiagnostics = new LinkedHashMap<>(engine.executionDiagnostics()); } }
            engine.reset(); warmupNs = System.nanoTime() - begin;
            down = new FirResampler(); up = new FirResampler(); down.reset(); up.reset();
            synchronized (lock) { state = finishRequested ? "DRAINING" : "RUNNING"; sessionStartNs = System.nanoTime(); }
            if (!finishRequested && !abortRequested) listener.onReady();
            while (true) {
                byte[] data;
                synchronized (lock) {
                    while (queue.isEmpty() && !finishRequested && !abortRequested && failureCode == null) lock.wait();
                    if (abortRequested || failureCode != null) {
                        droppedSamples += queuedSamples; queue.clear(); arrivalTimes.clear(); queuedSamples = 0;
                        throw new SessionFailure(failureCode == null ? "ABORTED" : failureCode,
                                failureMessage == null ? "Session closed before drain" : failureMessage);
                    }
                    if (queue.isEmpty()) { state = "DRAINING"; break; }
                    data = queue.removeFirst(); queuedSamples -= data.length / 2;
                    queueAgeMaxNs = Math.max(queueAgeMaxNs, System.nanoTime() - arrivalTimes.removeFirst());
                }
                long packetBegin = System.nanoTime();
                System.arraycopy(data, 0, raw, rawBytes, data.length); rawBytes += data.length;
                int count = 0;
                if (haveOddSample) { downInput[count++] = oddSample; haveOddSample = false; }
                for (int i = 0; i < data.length; i += 2) {
                    short sample = (short) ((data[i] & 255) | (data[i + 1] << 8));
                    downInput[count++] = sample / 32768f;
                }
                processedSamples += data.length / 2;
                if ((count & 1) != 0) { oddSample = downInput[--count]; haveOddSample = true; }
                long firBegin = System.nanoTime(); int lowCount = down.downsample(downInput, count, downOutput); firNs += System.nanoTime() - firBegin;
                for (int i = 0; i < lowCount; i++) pushLow(downOutput[i], up, engine);
                workerNs += System.nanoTime() - packetBegin;
            }
            long drainBegin = System.nanoTime();
            if (processedSamples != 0) {
                if (haveOddSample) { push16(0, down, up, engine); oddPadSamples = 1; }
                long beginFir = System.nanoTime(); down.downsample(downZeros, downZeros.length, lowTail); firNs += System.nanoTime() - beginFir;
                firDrainInputSamples = 94;
                for (float value : lowTail) pushLow(value, up, engine);
                if (fill != 0) {
                    modelPaddingSamples = CHUNK - fill; Arrays.fill(modelInput, fill, CHUNK, 0);
                    processChunk(up, engine); fill = 0;
                }
                long beginFlush = System.nanoTime(); float[] tail = engine.flush(); flushCalls++;
                flushNs = System.nanoTime() - beginFlush;
                if (tail != null) emitLow(tail, up);
                if (forwardedLowSamples != validLowSamples) throw new IOException("Model flush did not cover the valid input prefix");
                beginFir = System.nanoTime(); int count = up.upsample(upZeros, upZeros.length, upOutput); firNs += System.nanoTime() - beginFir;
                emit16(upOutput, count); firTailOutputSamples = 188;
                long expected = processedSamples + oddPadSamples + FIR_TAIL_SAMPLES;
                if (enhancedBytes / 2L != expected) throw new IOException("Terminal output sample count mismatch");
            }
            drainNs = System.nanoTime() - drainBegin;
        } catch (SessionFailure error) {
            terminalCode = error.code; terminalMessage = error.getMessage();
            terminalFailure = error;
        } catch (Throwable error) {
            terminalFailure = error;
            synchronized (lock) {
                terminalCode = failureCode != null ? failureCode : abortRequested ? "ABORTED" : "PROCESSING_ERROR";
                terminalMessage = failureMessage != null ? failureMessage : error.toString();
            }
        } finally {
            try { if (engine != null) engine.close(); } catch (Exception error) {
                if (terminalCode == null) { terminalCode = "CLOSE_ERROR"; terminalMessage = error.toString(); }
                if (terminalFailure == null) terminalFailure = error;
                else if (terminalFailure != error) terminalFailure.addSuppressed(error);
            }
            if (down != null) down.close(); if (up != null) up.close();
            synchronized (lock) {
                droppedSamples += queuedSamples; queuedSamples = 0; queue.clear(); arrivalTimes.clear();
                state = terminalCode == null ? "FINISHED" : "ERROR";
            }
            try {
                Files.createDirectories(outputDir);
                writeWave(outputDir.resolve("raw.wav"), raw, rawBytes);
                writeWave(outputDir.resolve("enhanced.wav"), enhanced, enhancedBytes);
            } catch (Exception error) {
                terminalCode = "RECORDING_ERROR"; terminalMessage = error.toString();
                if (terminalFailure != null && terminalFailure != error) error.addSuppressed(terminalFailure);
                terminalFailure = error;
                synchronized (lock) { state = "ERROR"; }
            }
            Map<String, Object> stats = statistics();
            if (terminalCode != null) {
                stats.put("error_code", terminalCode); stats.put("error", terminalMessage);
                stats.put("error_details", boundedErrorDetails(terminalFailure));
            }
            try { Files.write(outputDir.resolve("stats.json"), (json(stats) + "\n").getBytes(StandardCharsets.UTF_8)); }
            catch (Exception ignored) { stats.put("stats_file_error", true); }
            try {
                if (terminalCode == null) listener.onFinished(stats);
                else listener.onError(terminalCode, terminalMessage, stats);
            } catch (Throwable ignored) { /* Terminal listener cannot leak native resources. */ }
        }
    }

    private void push16(float sample, FirResampler down, FirResampler up, Engine engine) throws Exception {
        if (!haveOddSample) { oddSample = sample; haveOddSample = true; return; }
        pair[0] = oddSample; pair[1] = sample; haveOddSample = false;
        long begin = System.nanoTime(); down.downsample(pair, 2, lowPair); firNs += System.nanoTime() - begin;
        pushLow(lowPair[0], up, engine);
    }
    private void pushLow(float sample, FirResampler up, Engine engine) throws Exception {
        modelInput[fill++] = sample; validLowSamples++;
        if (fill == CHUNK) { processChunk(up, engine); fill = 0; }
    }
    private void processChunk(FirResampler up, Engine engine) throws Exception {
        long begin = System.nanoTime(); float[] output = engine.process(modelInput);
        long elapsed = System.nanoTime() - begin;
        processNs += elapsed; maxProcessNs = Math.max(maxProcessNs, elapsed); calls++;
        callTimes[(int) calls - 1] = elapsed;
        if (elapsed > 100_000_000L) deadlineMisses++;
        modelInputSamples += CHUNK;
        if (output == null || output.length != (calls == 1 ? 640 : 800)) throw new IOException("Unexpected enhancer output length");
        emitLow(output, up);
    }
    private void emitLow(float[] values, FirResampler up) throws Exception {
        modelOutputSamples += values.length;
        int valid = (int) Math.min(values.length, validLowSamples - forwardedLowSamples);
        if (valid <= 0) return;
        long begin = System.nanoTime(); int count = up.upsample(values, valid, upOutput); firNs += System.nanoTime() - begin;
        emit16(upOutput, count); forwardedLowSamples += valid;
    }
    private void emit16(float[] values, int count) throws Exception {
        byte[] pcm = new byte[count * 2];
        if (enhancedBytes + pcm.length > enhanced.length) throw new IOException("Enhanced capture capacity exceeded");
        for (int i = 0; i < count; i++) {
            float value = values[i]; if (!Float.isFinite(value)) throw new IOException("Nonfinite enhanced audio");
            int sample = (int) Math.rint(Math.max(-1f, Math.min(32767f / 32768f, value)) * 32768f);
            pcm[2 * i] = (byte) sample; pcm[2 * i + 1] = (byte) (sample >> 8);
        }
        System.arraycopy(pcm, 0, enhanced, enhancedBytes, pcm.length); enhancedBytes += pcm.length;
        if (firstOutputNs == 0) firstOutputNs = System.nanoTime();
        listener.onAudio(pcm);
    }

    public Map<String, Object> statistics() {
        synchronized (lock) {
            // A live observer can sample while the worker finishes another call.
            // Use one length for the copied array and its percentile indexes.
            int completedCalls = (int) calls;
            Map<String, Object> s = new LinkedHashMap<>();
            s.put("state", state); s.put("sample_rate", RATE); s.put("model_rate", 8000);
            s.put("requested_backend", requestedBackend.name());
            s.put("nnapi_fp16_relaxation_requested", nnapiUseFp16);
            s.put("hybrid_execution_requested", requestedBackend == ReuseConfig.Backend.NNAPI_ACCELERATOR);
            s.put("nnapi_execution_verified", false);
            s.put("gpu_execution_verified", false);
            s.putAll(backendDiagnostics);
            s.put("input_samples", acceptedSamples); s.put("processed_input_samples", processedSamples);
            s.put("output_samples", enhancedBytes / 2); s.put("raw_recorded_samples", rawBytes / 2);
            s.put("dropped_input_samples", droppedSamples); s.put("rejected_after_stop_samples", rejectedAfterStopSamples);
            s.put("queue_overflows", queueOverflows); s.put("max_queue_samples", maxQueuedSamples);
            s.put("queued_input_samples", queuedSamples);
            s.put("queue_age_max_ms", queueAgeMaxNs / 1e6); s.put("sequence_gaps", sequenceGaps);
            s.put("sample_index_validation", sampleIndexValidation); s.put("model_calls", calls);
            s.put("model_input_samples", modelInputSamples); s.put("model_output_samples", modelOutputSamples);
            s.put("valid_model_samples", validLowSamples); s.put("forwarded_model_samples", forwardedLowSamples);
            s.put("pending_model_samples", validLowSamples - forwardedLowSamples);
            s.put("pending_input_pair_samples", haveOddSample ? 1 : 0);
            s.put("model_padding_samples", modelPaddingSamples); s.put("odd_input_pad_samples", oddPadSamples);
            s.put("fir_drain_input_samples", firDrainInputSamples); s.put("fir_tail_output_samples", firTailOutputSamples);
            s.put("flush_calls", flushCalls); s.put("process_total_ms", processNs / 1e6);
            s.put("process_mean_ms", completedCalls == 0 ? 0 : processNs / 1e6 / completedCalls); s.put("process_max_ms", maxProcessNs / 1e6);
            long[] sorted = Arrays.copyOf(callTimes, completedCalls); Arrays.sort(sorted);
            s.put("process_p50_ms", completedCalls == 0 ? 0 : sorted[(completedCalls - 1) / 2] / 1e6);
            s.put("process_p95_ms", completedCalls == 0 ? 0 : sorted[(int) Math.ceil(completedCalls * .95) - 1] / 1e6);
            s.put("deadline_misses", deadlineMisses); s.put("flush_ms", flushNs / 1e6); s.put("fir_ms", firNs / 1e6);
            s.put("worker_total_ms", workerNs / 1e6); s.put("drain_ms", drainNs / 1e6);
            s.put("processing_rtf", processedSamples == 0 ? 0 : (workerNs + drainNs) / (processedSamples * 1e9 / RATE));
            s.put("load_ms", loadNs / 1e6); s.put("warmup_ms", warmupNs / 1e6); s.put("warmup_calls", MODEL_WARMUP_CALLS);
            s.put("first_output_after_ready_ms", firstOutputNs == 0 ? 0 : (firstOutputNs - sessionStartNs) / 1e6);
            s.put("first_output_after_input_ms", firstOutputNs == 0 || firstInputNs == 0 ? 0 : (firstOutputNs - firstInputNs) / 1e6);
            s.put("latency_scope", "native queue/processing; excludes source capture/transport latency");
            s.put("raw_file", "raw.wav"); s.put("enhanced_file", "enhanced.wav"); s.put("recording_limit_seconds", 120);
            return s;
        }
    }
    private static final class SessionFailure extends Exception {
        final String code;
        SessionFailure(String code, String text) { super(text); this.code = code; }
    }

    /** Stack, causes and suppressed failures, bounded to 64 KiB of UTF-8 text; attaches no audio. */
    public static String boundedErrorDetails(Throwable error) {
        if (error == null) return "";
        final String marker = "\n[error_details truncated at 64 KiB]\n";
        final int limit = 64 * 1024 - marker.length();
        StringBuilder text = new StringBuilder();
        final int[] bytes = {0};
        final boolean[] truncated = {false};
        try (PrintWriter output = new PrintWriter(new Writer() {
            @Override public void write(char[] chars, int offset, int length) {
                for (int i = offset; i < offset + length; i++) {
                    if (truncated[0]) return;
                    char value = chars[i];
                    boolean pair = Character.isHighSurrogate(value) && i + 1 < offset + length
                            && Character.isLowSurrogate(chars[i + 1]);
                    int count = pair ? 4 : value <= 0x7f ? 1 : value <= 0x7ff ? 2 : 3;
                    if (bytes[0] + count > limit) { truncated[0] = true; return; }
                    text.append(value); bytes[0] += count;
                    if (pair) text.append(chars[++i]);
                }
            }
            @Override public void flush() { }
            @Override public void close() { }
        })) { error.printStackTrace(output); }
        if (truncated[0]) {
            if (text.length() > 0 && Character.isHighSurrogate(text.charAt(text.length() - 1)))
                text.setLength(text.length() - 1);
            text.append(marker);
        }
        return text.toString();
    }
    private static void writeWave(Path path, byte[] data, int length) throws IOException {
        try (RandomAccessFile out = new RandomAccessFile(path.toFile(), "rw")) {
            out.setLength(0); out.writeBytes("RIFF"); le32(out, 36 + length); out.writeBytes("WAVEfmt ");
            le32(out, 16); le16(out, 1); le16(out, 1); le32(out, RATE); le32(out, RATE * 2);
            le16(out, 2); le16(out, 16); out.writeBytes("data"); le32(out, length); out.write(data, 0, length);
        }
    }
    private static void le16(RandomAccessFile out, int value) throws IOException { out.write(value); out.write(value >>> 8); }
    private static void le32(RandomAccessFile out, int value) throws IOException { le16(out, value); le16(out, value >>> 16); }
    private static String json(Map<String, Object> values) {
        StringBuilder text = new StringBuilder("{"); boolean first = true;
        for (Map.Entry<String, Object> e : values.entrySet()) {
            if (!first) text.append(','); first = false; text.append('"').append(escape(e.getKey())).append("\":");
            Object value = e.getValue();
            if (value instanceof Number || value instanceof Boolean) text.append(value);
            else text.append('"').append(escape(String.valueOf(value))).append('"');
        }
        return text.append('}').toString();
    }
    private static String escape(String value) {
        StringBuilder text = new StringBuilder();
        for (int i = 0; i < value.length(); i++) { char c = value.charAt(i);
            if (c == '"' || c == '\\') text.append('\\').append(c);
            else if (c < 32) text.append(String.format("\\u%04x", (int) c)); else text.append(c);
        }
        return text.toString();
    }
}
