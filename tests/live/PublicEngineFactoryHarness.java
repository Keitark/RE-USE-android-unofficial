package io.github.reuse.externaltest;

import io.github.reuse.live.EnhancementSession;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/** Exercises the public engine API from outside its package, with the actual queue and JNI FIR. */
public final class PublicEngineFactoryHarness {
    private static void require(boolean value, String message) {
        if (!value) throw new AssertionError(message);
    }
    private static void await(CountDownLatch latch) throws Exception {
        require(latch.await(10, TimeUnit.SECONDS), "Timed out waiting for session callback");
    }
    private static long number(Map<String, Object> stats, String key) { return ((Number) stats.get(key)).longValue(); }

    private static final class Result implements EnhancementSession.Listener {
        final CountDownLatch ready = new CountDownLatch(1), done = new CountDownLatch(1);
        Map<String, Object> stats;
        String error;
        int audioBytes;
        public void onReady() { ready.countDown(); }
        public void onAudio(byte[] data) { audioBytes += data.length; }
        public void onFinished(Map<String, Object> stats) { this.stats = stats; done.countDown(); }
        public void onError(String code, String message, Map<String, Object> stats) {
            this.error = code; this.stats = stats; done.countDown();
        }
    }

    private static final class Engine implements EnhancementSession.Engine {
        final List<String> lifecycle = Collections.synchronizedList(new ArrayList<>());
        final AtomicInteger closes = new AtomicInteger();
        final CountDownLatch entered = new CountDownLatch(1), release = new CountDownLatch(1);
        Thread owner;
        boolean first = true, verified;
        volatile boolean blockNext;
        float[] pending;

        private void worker(String event) {
            require(Thread.currentThread() == owner, "Engine lifecycle escaped the initialized worker");
            lifecycle.add(event);
        }
        public float[] process(float[] input) throws Exception {
            worker("process");
            require(input.length == 800, "Wrong engine input size");
            if (blockNext) { blockNext = false; entered.countDown(); await(release); }
            float[] output = new float[first ? 640 : 800];
            if (!first) System.arraycopy(pending, 0, output, 0, 160);
            System.arraycopy(input, 0, output, first ? 0 : 160, 640);
            pending = Arrays.copyOfRange(input, 640, 800); first = false;
            return output;
        }
        public float[] flush() { worker("flush"); return first ? null : pending; }
        public void reset() { worker("reset"); first = true; pending = null; }
        public void verifyAcceleratorExecution() { worker("verify"); verified = true; }
        public void close() { worker("close"); closes.incrementAndGet(); }
        public Map<String, Object> executionDiagnostics() {
            Map<String, Object> diagnostics = new LinkedHashMap<>();
            diagnostics.put("requested_backend", "TEST_ENGINE");
            diagnostics.put("test_engine_verified", verified);
            return diagnostics;
        }
    }

    private static EnhancementSession session(Path root, Result result, Engine engine, AtomicInteger creations) {
        Thread caller = Thread.currentThread();
        EnhancementSession.EngineFactory factory = () -> {
            require(Thread.currentThread() != caller, "Factory executed on the caller thread");
            require(engine.owner == Thread.currentThread(), "Initializer did not precede factory creation");
            engine.lifecycle.add("create"); creations.incrementAndGet();
            return engine;
        };
        return EnhancementSession.withEngine(root, result, () -> {
            engine.owner = Thread.currentThread(); engine.lifecycle.add("initialize");
        }, factory);
    }

    private static void lifecycle(Path root) throws Exception {
        Result result = new Result(); Engine engine = new Engine(); AtomicInteger creations = new AtomicInteger();
        EnhancementSession session = session(root, result, engine, creations);
        require(creations.get() == 0, "Factory was not deferred until start");
        session.start(); await(result.ready);
        require(engine.lifecycle.equals(Arrays.asList("initialize", "create", "process", "process", "process", "verify", "reset")),
                "Warm-up, verification and reset order changed: " + engine.lifecycle);
        require(session.accept16kS16LE(new byte[1601 * 2], 0), "Public PCM input rejected");
        session.finish(); await(result.done); session.close();
        require(result.error == null, "Injected engine failed: " + result.error);
        require(creations.get() == 1 && engine.closes.get() == 1, "Engine factory/close ownership changed");
        require(number(result.stats, "input_samples") == 1601 && number(result.stats, "processed_input_samples") == 1601,
                "Injected session lost PCM samples");
        require(number(result.stats, "output_samples") == 1790 && result.audioBytes == 1790 * 2,
                "Odd-sample/FIR-tail accounting changed");
        require("TEST_ENGINE".equals(result.stats.get("requested_backend")), "Injected diagnostics were lost");
        require(Boolean.TRUE.equals(result.stats.get("test_engine_verified")), "Post-warm-up diagnostics were lost");
        require(Files.size(root.resolve("raw.wav")) == 44 + 1601 * 2, "Raw recording length changed");
        require(Files.size(root.resolve("enhanced.wav")) == 44 + result.audioBytes, "Enhanced recording length changed");
    }

    private static void creationFailure(Path root) throws Exception {
        Result result = new Result(); AtomicInteger creations = new AtomicInteger();
        EnhancementSession session = EnhancementSession.withEngine(root, result, () -> { }, () -> {
            creations.incrementAndGet();
            java.io.IOException failure = new java.io.IOException("Optional backend unavailable",
                    new UnsatisfiedLinkError("libQnnHtp.so: test missing driver dependency"));
            failure.addSuppressed(new java.io.IOException("test backend registration cleanup failed"));
            throw failure;
        });
        session.start(); await(result.done); session.close();
        require(creations.get() == 1, "Factory retried without authorization");
        require(result.ready.getCount() == 1, "Failed factory exposed microphone readiness");
        require("PROCESSING_ERROR".equals(result.error), "Factory failure was hidden");
        require(number(result.stats, "input_samples") == 0 && number(result.stats, "output_samples") == 0,
                "Factory failure became audio processing");
        String details = (String) result.stats.get("error_details");
        require(details.contains("Caused by: java.lang.UnsatisfiedLinkError")
                && details.contains("test missing driver dependency"), "Backend root cause was lost");
        require(details.contains("Suppressed: java.io.IOException: test backend registration cleanup failed"),
                "Backend suppressed cleanup cause was lost");
        require("java.io.IOException: Optional backend unavailable".equals(result.stats.get("error")),
                "The UI error summary was replaced with a stack trace");
        require(Files.readString(root.resolve("stats.json")).contains("test missing driver dependency"),
                "Saved terminal JSON lost the nested cause");
    }

    private static void processingFailureDetails(Path root) throws Exception {
        Result result = new Result();
        EnhancementSession session = EnhancementSession.withEngine(root, result, () -> { }, () -> new EnhancementSession.Engine() {
            public float[] process(float[] input) throws Exception {
                throw new java.io.IOException("QNN warm-up refused", new IllegalArgumentException("test invalid backend configuration"));
            }
            public float[] flush() { throw new AssertionError("Failed warm-up must not flush audio"); }
            public void reset() { throw new AssertionError("Failed warm-up must not expose readiness"); }
            public void close() throws Exception { throw new java.io.IOException("test native cleanup failed"); }
        });
        session.start(); await(result.done); session.close();
        require(result.ready.getCount() == 1 && "PROCESSING_ERROR".equals(result.error),
                "Warm-up failure became readiness or changed its terminal code");
        String details = (String) result.stats.get("error_details");
        require(details.contains("Caused by: java.lang.IllegalArgumentException: test invalid backend configuration"),
                "Warm-up nested cause was lost");
        require(details.contains("Suppressed: java.io.IOException: test native cleanup failed"),
                "Actual engine-close error did not remain attached to the processing failure");
        require("java.io.IOException: QNN warm-up refused".equals(result.stats.get("error")),
                "Cleanup changed the existing terse processing error");
    }

    private static void boundedDetails(Path root) throws Exception {
        String details = EnhancementSession.boundedErrorDetails(new java.io.IOException("\u65e5\u672c\u8a9e\uD83D\uDE42".repeat(20000)));
        require(details.getBytes(java.nio.charset.StandardCharsets.UTF_8).length <= 64 * 1024,
                "Diagnostic error details exceeded the UTF-8 byte bound");
        require(details.contains("error_details truncated at 64 KiB"), "Truncation was not disclosed");
        require(!Character.isHighSurrogate(details.charAt(details.length() - 1)), "Trace ended within a surrogate pair");
        require(EnhancementSession.boundedErrorDetails(null).isEmpty(), "Null diagnostic should be empty");
    }

    private static void defaultQueue(Path root) throws Exception {
        Result result = new Result(); Engine engine = new Engine();
        EnhancementSession session = session(root, result, engine, new AtomicInteger());
        session.start(); await(result.ready); engine.blockNext = true;
        require(session.accept16kS16LE(new byte[1600 * 2]), "Initial PCM block rejected");
        await(engine.entered);
        require(session.accept16kS16LE(new byte[8000 * 2]), "500 ms queue capacity rejected");
        require(!session.accept16kS16LE(new byte[2]), "Public factory enlarged the bounded queue");
        engine.release.countDown(); await(result.done);
        require("INPUT_OVERFLOW".equals(result.error), "Queue overflow was hidden");
        require(number(result.stats, "max_queue_samples") == 8000, "Public queue bound changed");
        require(engine.closes.get() == 1, "Overflow leaked injected engine");
    }

    private static void expectInvalid(Runnable operation) {
        try { operation.run(); throw new AssertionError("Invalid public factory argument accepted"); }
        catch (IllegalArgumentException expected) { }
    }
    private static void nullArguments(Path root) {
        Result result = new Result();
        expectInvalid(() -> EnhancementSession.withEngine(root, null, () -> { }, Engine::new));
        expectInvalid(() -> EnhancementSession.withEngine(root, result, () -> { }, null));
        expectInvalid(() -> EnhancementSession.withEngine(root, result, null, Engine::new));
        expectInvalid(() -> EnhancementSession.withEngine(null, result, () -> { }, Engine::new));
    }

    public static void main(String[] args) throws Exception {
        String kind = args[0]; Path root = Path.of(args[1]); Files.createDirectories(root);
        if (kind.equals("factory_lifecycle")) lifecycle(root);
        else if (kind.equals("factory_failure")) creationFailure(root);
        else if (kind.equals("factory_details_processing")) processingFailureDetails(root);
        else if (kind.equals("factory_details_bound")) boundedDetails(root);
        else if (kind.equals("factory_queue")) defaultQueue(root);
        else if (kind.equals("factory_nulls")) nullArguments(root);
        else throw new IllegalArgumentException("Unknown public factory test");
        System.out.println("{\"case\":\"" + kind + "\",\"passed\":true}");
    }
}
