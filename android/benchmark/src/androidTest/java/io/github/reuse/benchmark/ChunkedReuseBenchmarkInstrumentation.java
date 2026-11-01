package io.github.reuse.benchmark;

import android.app.Activity;
import android.app.Instrumentation;
import android.os.Build;
import android.os.Bundle;
import android.os.Debug;
import android.util.Log;
import io.github.reuse.ReuseConfig;
import io.github.reuse.ChunkedReuseEnhancer;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.util.Arrays;
import org.json.JSONObject;

/** Standalone five-hop Android inference/parity runner; no microphone or DSP bypass. */
public final class ChunkedReuseBenchmarkInstrumentation extends Instrumentation {
    private Bundle arguments;
    private File directory;

    @Override public void onCreate(Bundle arguments) {
        this.arguments = arguments == null ? new Bundle() : arguments;
        start();
    }

    @Override public void onStart() {
        Bundle result = new Bundle();
        try {
            JSONObject report = benchmark();
            result.putString("stream", report.toString(2) + "\n");
            // Always retain the report and enhanced audio before surfacing an acceptance failure.
            boolean passed = report.optBoolean("lifecycle_pass", false) &&
                    (!report.has("parity_pass") || report.getBoolean("parity_pass"));
            if (!passed) result.putString("Error", "RE-USE lifecycle or reference parity failed; see report.json");
            finish(passed ? Activity.RESULT_OK : Activity.RESULT_CANCELED, result);
        } catch (Throwable error) {
            Log.e("ReuseChunkBenchmark", "Benchmark failed", error);
            result.putString("stream", "ERROR: " + error + "\n");
            if (directory != null && directory.isDirectory()) {
                try {
                    JSONObject failed = new JSONObject().put("lifecycle_pass", false).put("parity_pass", false)
                            .put("error", error.toString()).put("runner", getClass().getName());
                    Files.write(new File(directory, "report.json").toPath(),
                            failed.toString(2).getBytes(java.nio.charset.StandardCharsets.UTF_8));
                } catch (Exception recordingError) { Log.e("ReuseChunkBenchmark", "Could not retain failure", recordingError); }
            }
            finish(Activity.RESULT_CANCELED, result);
        }
    }

    private JSONObject benchmark() throws Exception {
        int rate = integer("sampleRate", 8000);
        ReuseConfig config = new ReuseConfig(rate, integer("exitLayers", 3), integer("threads", 2));
        File defaultDirectory = new File(getTargetContext().getExternalFilesDir(null), "model");
        directory = new File(arguments.getString("modelDir", defaultDirectory.getAbsolutePath()));
        File inputFile = new File(directory, "input.f32");
        if (!inputFile.isFile()) throw new IOException("Provide input.f32 for independent serial-reference parity");
        File expectedFile = new File(directory, "expected.f32");
        if (!expectedFile.isFile()) throw new IOException("Provide expected.f32 from the independent serial reference");
        float[] input = readFloats(inputFile);
        int chunkSamples = ChunkedReuseEnhancer.FRAMES_PER_CALL * config.hopSize;
        if (input.length == 0 || input.length % chunkSamples != 0) {
            throw new IOException("input.f32 must contain an integral, nonzero number of 100 ms float32 chunks");
        }
        int frames = input.length / config.hopSize;
        int calls = input.length / chunkSamples;
        int warmup = integer("warmup", 2);
        if (warmup < 0 || warmup > 5000) throw new IllegalArgumentException("warmup must be 0..5000");
        float[] enhanced = new float[input.length];
        long[] times = new long[calls];
        long loadStart = System.nanoTime();
        long loadNs, loopNs, flushNs;
        JSONObject lifecycle;
        float[] chunk = new float[chunkSamples];
        try (ChunkedReuseEnhancer enhancer = ChunkedReuseEnhancer.create(getTargetContext(), new File(directory, "model.onnx").toPath(),
                new File(directory, "manifest.json").toPath(), config)) {
            loadNs = System.nanoTime() - loadStart;
            for (int i = 0; i < warmup; i++) {
                System.arraycopy(input, (i % calls) * chunk.length, chunk, 0, chunk.length);
                enhancer.process(chunk);
            }
            enhancer.reset();
            int written = 0;
            long loopStart = System.nanoTime();
            for (int call = 0; call < calls; call++) {
                System.arraycopy(input, call * chunk.length, chunk, 0, chunk.length);
                long begin = System.nanoTime();
                float[] output = enhancer.process(chunk);
                times[call] = System.nanoTime() - begin;
                int expectedLength = (call == 0 ? 4 : 5) * config.hopSize;
                if (output == null || output.length != expectedLength) throw new AssertionError("Unexpected chunk output length at call " + call);
                if (output != null) {
                    System.arraycopy(output, 0, enhanced, written, output.length);
                    written += output.length;
                }
            }
            long flushStart = System.nanoTime();
            float[] tail = enhancer.flush();
            flushNs = System.nanoTime() - flushStart;
            if (tail == null || tail.length != config.hopSize) throw new AssertionError("Flush must return exactly one pending real hop");
            if (tail != null) {
                System.arraycopy(tail, 0, enhanced, written, tail.length);
                written += tail.length;
            }
            loopNs = System.nanoTime() - loopStart;
            if (written != input.length) throw new AssertionError("Output length differs from input");
            for (float value : enhanced) if (!Float.isFinite(value)) throw new AssertionError("Non-finite PCM");
            lifecycle = verifyLifecycle(enhancer, config, input, enhanced);
        }
        writeFloats(new File(directory, "enhanced.f32"), enhanced);
        long sum = 0, misses = 0;
        for (long time : times) { sum += time; if (time > 100_000_000L) misses++; }
        Arrays.sort(times);
        double inputSeconds = (double) input.length / rate;
        double rtf = sum / (inputSeconds * 1e9);
        JSONObject report = new JSONObject();
        report.put("device", Build.MANUFACTURER + " " + Build.MODEL);
        report.put("abi", Build.SUPPORTED_ABIS[0]);
        report.put("api_level", Build.VERSION.SDK_INT);
        report.put("onnxruntime_version", ai.onnxruntime.OrtEnvironment.getEnvironment().getVersion());
        report.put("sample_rate", rate);
        report.put("exit_layers", config.exitLayers);
        report.put("intra_op_threads", config.intraOpThreads);
        report.put("input_source", "input.f32 with required independent serial expected.f32");
        report.put("model_sha256", sha256(new File(directory, "model.onnx")));
        report.put("manifest_sha256", sha256(new File(directory, "manifest.json")));
        report.put("input_sha256", sha256(inputFile));
        report.put("expected_sha256", sha256(expectedFile));
        report.put("frames_per_call", ChunkedReuseEnhancer.FRAMES_PER_CALL);
        report.put("model_calls", calls);
        report.put("input_samples", input.length);
        report.put("output_samples", enhanced.length);
        report.put("frames", frames);
        report.put("warmup_calls", warmup);
        report.put("warmup_frames", warmup * ChunkedReuseEnhancer.FRAMES_PER_CALL);
        report.put("audio_seconds", inputSeconds);
        report.put("load_ms", loadNs / 1e6);
        report.put("hop_ms_amortized_mean", sum / (frames * 1e6));
        report.put("call_ms_mean", sum / (calls * 1e6));
        report.put("call_ms_p50", percentile(times, 0.5) / 1e6);
        report.put("call_ms_p95", percentile(times, 0.95) / 1e6);
        report.put("call_ms_p99", percentile(times, 0.99) / 1e6);
        report.put("call_ms_max", times[times.length - 1] / 1e6);
        report.put("call_deadline_ms", 100);
        report.put("deadline_misses", misses);
        report.put("real_time_factor", rtf);
        report.put("average_keeps_up", rtf < 1.0);
        report.put("p95_meets_100ms", percentile(times, 0.95) <= 100_000_000L);
        report.put("loop_wall_ms_including_flush", loopNs / 1e6);
        report.put("flush_ms", flushNs / 1e6);
        report.put("startup_input_ms", 100);
        report.put("maximum_collection_delay_ms", 80);
        report.put("buffered_audio_delay_ms", 20);
        report.put("pss_kib_after_run", Debug.getPss());
        report.put("java_heap_bytes_after_run", Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory());
        report.put("native_heap_bytes_after_run", Debug.getNativeHeapAllocatedSize());
        report.put("device_fingerprint", Build.FINGERPRINT);
        report.put("emulator_indicators_present", Build.FINGERPRINT.contains("generic") ||
                Build.MODEL.toLowerCase(java.util.Locale.ROOT).contains("sdk"));
        report.put("lifecycle", lifecycle);
        report.put("lifecycle_pass", lifecycle.getBoolean("pass"));
        report.put("device_is_emulator", Build.HARDWARE.contains("ranchu") || Build.HARDWARE.contains("goldfish") ||
                report.getBoolean("emulator_indicators_present"));
        report.put("scope", "Actual Java chunked neural/DSP correctness; emulator timing is not physical-phone evidence");
        addParity(report, enhanced, readFloats(expectedFile));
        Files.write(new File(directory, "report.json").toPath(), report.toString(2).getBytes(java.nio.charset.StandardCharsets.UTF_8));
        Log.i("ReuseChunkBenchmark", report.toString());
        return report;
    }

    private static JSONObject verifyLifecycle(ChunkedReuseEnhancer enhancer, ReuseConfig config,
            float[] input, float[] baseline) throws Exception {
        float[] chunk = new float[ChunkedReuseEnhancer.FRAMES_PER_CALL * config.hopSize];
        boolean terminalRejected = rejectsState(() -> enhancer.process(chunk));
        boolean failedRejected = rejectsState(enhancer::flush);
        enhancer.reset();
        boolean wrongSizeRejected = rejectsArgument(() -> enhancer.process(new float[1]));
        chunk[0] = Float.NaN;
        boolean nonFiniteRejected = rejectsArgument(() -> enhancer.process(chunk));
        chunk[0] = Float.POSITIVE_INFINITY;
        nonFiniteRejected &= rejectsArgument(() -> enhancer.process(chunk));
        System.arraycopy(input, 0, chunk, 0, chunk.length);
        // Invalid inputs must not advance any analysis or neural history.
        float[] first = enhancer.process(chunk);
        boolean resetShape = first != null && first.length == 4 * config.hopSize;
        double resetMaximum = 0;
        int compared = resetShape ? first.length : 0;
        if (resetShape) for (int i = 0; i < first.length; i++) {
            resetMaximum = Math.max(resetMaximum, Math.abs(first[i] - baseline[i]));
        }
        enhancer.reset();
        boolean emptyFlushNull = enhancer.flush() == null;
        boolean repeatedFlushRejected = rejectsState(enhancer::flush);
        enhancer.reset();
        enhancer.close();
        boolean closedProcessRejected = rejectsState(() -> enhancer.process(chunk));
        boolean closedResetRejected = rejectsState(enhancer::reset);
        boolean closedFlushRejected = rejectsState(enhancer::flush);
        enhancer.close();
        boolean passed = terminalRejected && failedRejected && wrongSizeRejected && nonFiniteRejected &&
                resetShape && emptyFlushNull && repeatedFlushRejected && closedProcessRejected &&
                closedResetRejected && closedFlushRejected && resetMaximum <= 1e-6;
        return new JSONObject().put("pass", passed).put("reset_max_abs", resetMaximum)
                .put("reset_compared_samples", compared).put("reset_first_output_shape", resetShape)
                .put("terminal_process_rejected", terminalRejected)
                .put("failed_stream_rejected", failedRejected).put("invalid_length_rejected", wrongSizeRejected)
                .put("non_finite_input_rejected", nonFiniteRejected).put("empty_flush_null", emptyFlushNull)
                .put("repeated_flush_rejected", repeatedFlushRejected)
                .put("closed_process_rejected", closedProcessRejected).put("closed_reset_rejected", closedResetRejected)
                .put("closed_flush_rejected", closedFlushRejected);
    }

    private interface Action { void run() throws Exception; }

    private static boolean rejectsState(Action action) throws Exception {
        try { action.run(); return false; } catch (IllegalStateException expected) { return true; }
    }

    private static boolean rejectsArgument(Action action) throws Exception {
        try { action.run(); return false; } catch (IllegalArgumentException expected) { return true; }
    }

    private void addParity(JSONObject report, float[] actual, float[] expected) throws Exception {
        if (actual.length != expected.length) throw new AssertionError("Reference output length mismatch");
        double sumSquared = 0, maximum = 0, referencePower = 0;
        for (int i = 0; i < actual.length; i++) {
            if (!Float.isFinite(expected[i])) throw new IOException("Reference contains non-finite values");
            double error = actual[i] - expected[i];
            sumSquared += error * error;
            maximum = Math.max(maximum, Math.abs(error));
            referencePower += (double) expected[i] * expected[i];
        }
        double rmse = Math.sqrt(sumSquared / actual.length);
        double maxTolerance = Double.parseDouble(arguments.getString("maxAbsTolerance", "0.00002"));
        double rmseTolerance = Double.parseDouble(arguments.getString("rmseTolerance", "0.000005"));
        if (!Double.isFinite(maxTolerance) || !Double.isFinite(rmseTolerance) || maxTolerance <= 0 || rmseTolerance <= 0 ||
                maxTolerance > 2e-5 || rmseTolerance > 5e-6) {
            throw new IllegalArgumentException("Chunk parity tolerances may only tighten the defaults 2e-5 / 5e-6");
        }
        report.put("parity_max_abs", maximum);
        report.put("parity_rmse", rmse);
        report.put("parity_snr_db", 10 * Math.log10(Math.max(referencePower, 1e-30) / Math.max(sumSquared, 1e-30)));
        report.put("parity_max_abs_tolerance", maxTolerance);
        report.put("parity_rmse_tolerance", rmseTolerance);
        report.put("parity_pass", maximum <= maxTolerance && rmse <= rmseTolerance);
    }

    private int integer(String name, int fallback) {
        return Integer.parseInt(arguments.getString(name, Integer.toString(fallback)));
    }

    private static long percentile(long[] sorted, double probability) {
        return sorted[Math.max(0, (int) Math.ceil(sorted.length * probability) - 1)];
    }

    private static String sha256(File file) throws Exception {
        byte[] digest = java.security.MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file.toPath()));
        StringBuilder result = new StringBuilder(64);
        for (byte value : digest) result.append(String.format(java.util.Locale.ROOT, "%02x", value & 255));
        return result.toString();
    }

    private static float[] readFloats(File file) throws IOException {
        if (Files.size(file.toPath()) > 120L * 16000 * 4) throw new IOException("Float fixture exceeds 120 seconds at 16 kHz");
        byte[] bytes = Files.readAllBytes(file.toPath());
        if (bytes.length % 4 != 0) throw new IOException("Float32 file length must be divisible by four");
        float[] result = new float[bytes.length / 4];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(result);
        return result;
    }

    private static void writeFloats(File file, float[] data) throws IOException {
        ByteBuffer bytes = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        bytes.asFloatBuffer().put(data);
        Files.write(file.toPath(), bytes.array());
    }
}
