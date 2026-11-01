package io.github.reuse.benchmark;

import android.app.Activity;
import android.app.Instrumentation;
import android.os.Build;
import android.os.Bundle;
import android.os.Debug;
import android.util.Log;
import io.github.reuse.ReuseConfig;
import io.github.reuse.ReuseEnhancer;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.util.Arrays;
import org.json.JSONObject;

/** Standalone adb instrumentation runner; performs actual model inference, never a DSP bypass. */
public final class ReuseBenchmarkInstrumentation extends Instrumentation {
    private Bundle arguments;

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
            Log.e("ReuseBenchmark", "Benchmark failed", error);
            result.putString("stream", "ERROR: " + error + "\n");
            finish(Activity.RESULT_CANCELED, result);
        }
    }

    private JSONObject benchmark() throws Exception {
        int rate = integer("sampleRate", 8000);
        ReuseConfig config = new ReuseConfig(rate, integer("exitLayers", 3), integer("threads", 2));
        File defaultDirectory = new File(getTargetContext().getExternalFilesDir(null), "model");
        File directory = new File(arguments.getString("modelDir", defaultDirectory.getAbsolutePath()));
        File inputFile = new File(directory, "input.f32");
        float[] input = inputFile.isFile() ? readFloats(inputFile) : synthetic(config, integer("seconds", 4));
        if (input.length == 0 || input.length % config.hopSize != 0) {
            throw new IOException("input.f32 must contain an integral, nonzero number of 20 ms float32 hops");
        }
        int frames = input.length / config.hopSize;
        int warmup = integer("warmup", 10);
        if (warmup < 0 || warmup > 5000) throw new IllegalArgumentException("warmup must be 0..5000");
        float[] enhanced = new float[input.length];
        long[] times = new long[frames];
        long loadStart = System.nanoTime();
        long loadNs, loopNs, flushNs;
        JSONObject lifecycle;
        float[] hop = new float[config.hopSize];
        try (ReuseEnhancer enhancer = ReuseEnhancer.create(getTargetContext(), new File(directory, "model.onnx").toPath(),
                new File(directory, "manifest.json").toPath(), config)) {
            loadNs = System.nanoTime() - loadStart;
            for (int i = 0; i < warmup; i++) {
                System.arraycopy(input, (i % frames) * hop.length, hop, 0, hop.length);
                enhancer.process(hop);
            }
            enhancer.reset();
            int written = 0;
            long loopStart = System.nanoTime();
            for (int frame = 0; frame < frames; frame++) {
                System.arraycopy(input, frame * hop.length, hop, 0, hop.length);
                long begin = System.nanoTime();
                float[] output = enhancer.process(hop);
                times[frame] = System.nanoTime() - begin;
                if (output != null) {
                    System.arraycopy(output, 0, enhanced, written, output.length);
                    written += output.length;
                }
            }
            long flushStart = System.nanoTime();
            float[] tail = enhancer.flush();
            flushNs = System.nanoTime() - flushStart;
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
        for (long time : times) { sum += time; if (time > 20_000_000L) misses++; }
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
        report.put("input_source", inputFile.isFile() ? "input.f32" : "synthetic (speed test only)");
        report.put("input_samples", input.length);
        report.put("output_samples", enhanced.length);
        report.put("frames", frames);
        report.put("warmup_frames", warmup);
        report.put("audio_seconds", inputSeconds);
        report.put("load_ms", loadNs / 1e6);
        report.put("hop_ms_mean", sum / (frames * 1e6));
        report.put("hop_ms_p50", percentile(times, 0.5) / 1e6);
        report.put("hop_ms_p95", percentile(times, 0.95) / 1e6);
        report.put("hop_ms_p99", percentile(times, 0.99) / 1e6);
        report.put("hop_ms_max", times[times.length - 1] / 1e6);
        report.put("hop_deadline_ms", 20);
        report.put("deadline_misses", misses);
        report.put("real_time_factor", rtf);
        report.put("average_keeps_up", rtf < 1.0);
        report.put("p95_meets_20ms", percentile(times, 0.95) <= 20_000_000L);
        report.put("loop_wall_ms_including_flush", loopNs / 1e6);
        report.put("flush_ms", flushNs / 1e6);
        report.put("startup_input_ms", 40);
        report.put("buffered_audio_delay_ms", 20);
        report.put("pss_kib_after_run", Debug.getPss());
        report.put("java_heap_bytes_after_run", Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory());
        report.put("native_heap_bytes_after_run", Debug.getNativeHeapAllocatedSize());
        report.put("device_fingerprint", Build.FINGERPRINT);
        report.put("emulator_indicators_present", Build.FINGERPRINT.contains("generic") ||
                Build.MODEL.toLowerCase(java.util.Locale.ROOT).contains("sdk"));
        report.put("lifecycle", lifecycle);
        report.put("lifecycle_pass", lifecycle.getBoolean("pass"));
        File expectedFile = new File(directory, "expected.f32");
        if (expectedFile.isFile()) addParity(report, enhanced, readFloats(expectedFile));
        Files.write(new File(directory, "report.json").toPath(), report.toString(2).getBytes(java.nio.charset.StandardCharsets.UTF_8));
        Log.i("ReuseBenchmark", report.toString());
        return report;
    }

    private static JSONObject verifyLifecycle(ReuseEnhancer enhancer, ReuseConfig config,
            float[] input, float[] baseline) throws Exception {
        float[] hop = new float[config.hopSize];
        boolean terminalRejected = rejectsState(() -> enhancer.process(hop));
        boolean failedRejected = rejectsState(enhancer::flush);
        enhancer.reset();
        boolean wrongSizeRejected = rejectsArgument(() -> enhancer.process(new float[1]));
        hop[0] = Float.NaN;
        boolean nonFiniteRejected = rejectsArgument(() -> enhancer.process(hop));
        double resetMaximum = 0;
        int prefixFrames = Math.min(input.length / hop.length, 6);
        int compared = 0;
        // No inference has occurred since reset; invalid PCM must not have advanced history.
        for (int frame = 0; frame < prefixFrames; frame++) {
            System.arraycopy(input, frame * hop.length, hop, 0, hop.length);
            float[] output = enhancer.process(hop);
            if (frame == 0 && output != null) throw new AssertionError("Reset did not clear DSP history");
            if (frame > 0 && output == null) throw new AssertionError("Missing output after reset");
            if (output != null) {
                for (int i = 0; i < output.length; i++) {
                    resetMaximum = Math.max(resetMaximum, Math.abs(output[i] - baseline[compared++]));
                }
            }
        }
        enhancer.reset();
        boolean emptyFlushNull = enhancer.flush() == null;
        boolean repeatedFlushRejected = rejectsState(enhancer::flush);
        enhancer.reset();
        enhancer.close();
        boolean closedProcessRejected = rejectsState(() -> enhancer.process(hop));
        boolean closedResetRejected = rejectsState(enhancer::reset);
        enhancer.close(); // idempotent
        boolean passed = terminalRejected && failedRejected && wrongSizeRejected && nonFiniteRejected &&
                emptyFlushNull && repeatedFlushRejected && closedProcessRejected && closedResetRejected &&
                resetMaximum <= 1e-6;
        return new JSONObject().put("pass", passed).put("reset_max_abs", resetMaximum)
                .put("reset_compared_samples", compared).put("terminal_process_rejected", terminalRejected)
                .put("failed_stream_rejected", failedRejected).put("invalid_length_rejected", wrongSizeRejected)
                .put("non_finite_input_rejected", nonFiniteRejected).put("empty_flush_null", emptyFlushNull)
                .put("repeated_flush_rejected", repeatedFlushRejected)
                .put("closed_process_rejected", closedProcessRejected).put("closed_reset_rejected", closedResetRejected);
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
        double maxTolerance = Double.parseDouble(arguments.getString("maxAbsTolerance", "0.002"));
        double rmseTolerance = Double.parseDouble(arguments.getString("rmseTolerance", "0.0002"));
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

    private static float[] synthetic(ReuseConfig config, int seconds) {
        if (seconds < 1 || seconds > 600) throw new IllegalArgumentException("seconds must be 1..600");
        float[] input = new float[config.sampleRate * seconds];
        java.util.Random noise = new java.util.Random(20261101);
        for (int i = 0; i < input.length; i++) {
            double time = (double) i / config.sampleRate;
            input[i] = (float) (0.15 * Math.sin(2 * Math.PI * 173 * time) +
                    0.08 * Math.sin(2 * Math.PI * 346 * time) + 0.02 * noise.nextGaussian());
        }
        return input;
    }

    private static float[] readFloats(File file) throws IOException {
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
