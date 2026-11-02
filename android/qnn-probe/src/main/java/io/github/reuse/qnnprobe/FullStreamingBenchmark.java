package io.github.reuse.qnnprobe;

import android.content.Context;
import ai.onnxruntime.OrtEnvironment;
import io.github.reuse.ChunkedReuseEnhancer;
import io.github.reuse.ReuseConfig;
import io.github.reuse.RuntimeBuildInfo;
import io.github.reuse.SessionOptionsConfigurer;
import io.github.reuse.qnn.QnnHtpBackend;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.PrintWriter;
import java.io.StringWriter;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.text.SimpleDateFormat;
import java.util.Arrays;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import org.json.JSONArray;
import org.json.JSONObject;

/** Complete causal inference on one packaged public clip; never opens a microphone. */
final class FullStreamingBenchmark {
    private static final int SAMPLE_RATE = 8000;
    private static final int CHUNK = 800;
    private static final int WARMUP_CALLS = 3;
    private static final int EXPECTED_AUDIO_CALLS = 65;
    private static final int GROUPED_CALLS = 25;
    private static final int MAX_BYTES = 32 * 1024 * 1024;
    private static final int MAX_PROFILE_BYTES = 64 * 1024 * 1024;
    private final Context context;

    FullStreamingBenchmark(Context context) { this.context = context.getApplicationContext(); }

    private enum Mode {
        CPU_FP32("cpu_fp32"), CPU_QDQ("cpu_qdq"), QNN_HTP("qnn_htp");
        final String key;
        Mode(String key) { this.key = key; }
    }

    QnnProbeRunner.Result run(QnnProbeRunner.Result groupedResult, QnnProbeRunner.Progress progress) {
        JSONObject report = groupedResult.diagnostics;
        JSONObject benchmark = new JSONObject();
        put(report, "full_stream_benchmark", benchmark);
        put(report, "full_reuse_acceleration_verified", false);
        put(report, "full_model_hybrid_execution_verified", false);
        put(report, "audio_quality_verified", false);
        put(report, "scope", "Grouped encoder verification and complete causal offline public-clip processing; no microphone, live playout, or general audio-quality claim");
        put(benchmark, "complete", false);
        put(benchmark, "audio_quality_verified", false);
        put(benchmark, "microphone_started", false);
        put(benchmark, "timing_scope", "Sum of complete unprofiled enhancer.process calls plus flush, including Java STFT/synthesis and synchronous inference; excludes compilation, profiled zero warm-up, profile verification, checkpoints, comparisons and file writes. Offline unpaced processing, not live playout.");
        put(benchmark, "comparison_limits", "Sequential CPU_FP32, CPU_QDQ, QNN_HTP runs on one public calibration clip. Same ORT and two CPU intra-op threads; QNN HTP balanced mode with FP16 disabled. Provider evidence covers three zero-input warm-up calls only; profiling ends before reset and the measured sequence. Test order, clocks and thermal conditions are not controlled.");
        put(benchmark, "profiling_protocol", "three_zero_warmup_calls_then_unprofiled_sequence");
        put(benchmark, "timed_calls_profiled", false);
        put(benchmark, "warmup_calls_per_backend", WARMUP_CALLS);
        JSONObject backends = new JSONObject();
        put(benchmark, "backends", backends);
        String stage = "full.assets.manifest";
        long began = System.nanoTime();
        Path directory = groupedResult.directory == null ? null : groupedResult.directory.toPath();
        try {
            if (directory == null || !Files.isDirectory(directory)) throw new IOException("Grouped probe capture directory is unavailable");
            checkpoint(report, directory, stage);
            byte[] manifestBytes = asset("manifest.json");
            String manifestHash = sha256(manifestBytes);
            JSONObject manifest = new JSONObject(new String(manifestBytes, StandardCharsets.UTF_8));
            JSONObject full = manifest.getJSONObject("full_stream");
            JSONObject input = full.getJSONObject("input");
            JSONArray dimensions = input.getJSONArray("shape");
            if (input.getInt("sample_rate") != SAMPLE_RATE || dimensions.length() != 1
                    || full.getInt("frames_per_call") != 5 || full.getInt("calls") != EXPECTED_AUDIO_CALLS)
                throw new IOException("Unsupported full streaming fixture protocol");
            int sampleCount = dimensions.getInt(0);
            if (sampleCount <= 0 || sampleCount > MAX_BYTES / 4
                    || (sampleCount + CHUNK - 1) / CHUNK != EXPECTED_AUDIO_CALLS)
                throw new IOException("Public clip length does not match the bounded call count");
            float[] audio = ProbeVerification.decodeFloat32(checkedAsset(input.getString("file"), input.getString("sha256")), sampleCount);
            put(benchmark, "fixture_manifest_sha256", manifestHash);
            put(benchmark, "public_input_sha256", input.getString("sha256"));
            put(benchmark, "public_sample_source_sha256", full.optString("public_sample_sha256"));
            put(benchmark, "input_samples", sampleCount);
            put(benchmark, "sample_rate", SAMPLE_RATE);
            put(benchmark, "real_input_duration_seconds", sampleCount / (double) SAMPLE_RATE);
            put(benchmark, "audio_calls", EXPECTED_AUDIO_CALLS);
            put(benchmark, "expected_profile_model_calls", WARMUP_CALLS);
            put(benchmark, "expected_total_model_calls", WARMUP_CALLS + EXPECTED_AUDIO_CALLS + 1);
            put(benchmark, "unprofiled_sequence_model_calls", EXPECTED_AUDIO_CALLS + 1);
            put(benchmark, "frames_per_call", 5);
            put(benchmark, "input_chunk_samples", CHUNK);
            put(benchmark, "input_chunk_ms", 100);
            put(benchmark, "lookahead_hops", 0);
            put(benchmark, "exit_layers", 3);
            put(benchmark, "recurrent_states", "All 21 FP32 states advance chronologically in independent buffers per backend; reset after warm-up");
            put(benchmark, "no_live_queue_or_underrun_measurement", true);
            Path cache = context.getFilesDir().toPath().resolve("qnn-full-model-cache").resolve(manifestHash);
            Files.createDirectories(cache);
            put(benchmark, "model_cache_manifest_sha256", manifestHash);
            stage = "full.assets.models";
            checkpoint(report, directory, stage);
            JSONObject modelErrors = new JSONObject();
            put(benchmark, "model_asset_errors", modelErrors);
            ModelFiles baseline = prepareModel(full, "baseline", "full-baseline", cache, manifest, modelErrors);
            ModelFiles candidate = prepareModel(full, "candidate", "full-candidate", cache, manifest, modelErrors);
            if (baseline != null) {
                put(benchmark, "baseline_model_sha256", baseline.modelHash);
                put(benchmark, "baseline_manifest_sha256", baseline.manifestHash);
            }
            if (candidate != null) {
                put(benchmark, "candidate_model_sha256", candidate.modelHash);
                put(benchmark, "candidate_manifest_sha256", candidate.manifestHash);
            }
            put(benchmark, "native_neon_fma_build_flag", buildFlag("REUSE_NEON_FMA"));
            put(benchmark, "native_scan_n16_registers_build_flag", buildFlag("REUSE_SCAN_N16_REGISTERS"));
            Path nativeLibrary = new File(context.getApplicationInfo().nativeLibraryDir, "libreuse_ort_ops.so").toPath();
            if (Files.isRegularFile(nativeLibrary)) put(benchmark, "custom_op_library_sha256", sha256(boundedFile(nativeLibrary)));
            stage = "full.runtime.identity";
            checkpoint(report, directory, stage);
            try {
                put(benchmark, "ort_version", OrtEnvironment.getEnvironment().getVersion());
                put(benchmark, "ort_build_info", RuntimeBuildInfo.read());
            } catch (Exception | LinkageError error) {
                put(benchmark, "runtime_identity_error", error.toString());
            }
            JSONObject groupGate = groupedGate(report, manifest, directory);
            put(benchmark, "grouped_qnn_prerequisite", groupGate);
            boolean qnnAllowed = groupGate.optBoolean("verified", false);
            Map<Mode, float[]> outputs = new LinkedHashMap<>();
            boolean allComplete = true;
            for (Mode mode : Mode.values()) {
                JSONObject result = new JSONObject();
                put(backends, mode.key, result);
                put(result, "backend", mode.name());
                put(result, "execution_verified", false);
                ModelFiles files = mode == Mode.CPU_FP32 ? baseline : candidate;
                if (files == null) {
                    put(result, "status", "MODEL_ASSET_ERROR");
                    put(result, "error", modelErrors.optString(mode == Mode.CPU_FP32 ? "baseline" : "candidate"));
                    allComplete = false;
                    checkpoint(report, directory, "full." + mode.key + ".model_asset_error");
                    continue;
                }
                if (mode == Mode.QNN_HTP && !qnnAllowed) {
                    put(result, "status", "SKIPPED_GROUPED_PREREQUISITE_FAILED");
                    put(result, "error", "Full hybrid QNN run requires a numerically verified single-partition grouped HTP test and successful plugin cleanup");
                    allComplete = false;
                    checkpoint(report, directory, "full.qnn_htp.skipped");
                    continue;
                }
                float[] output = runBackend(mode, files, audio, result, report, directory, progress);
                if (output != null) outputs.put(mode, output);
                allComplete &= result.optBoolean("execution_verified", false);
            }
            JSONObject comparisons = new JSONObject();
            put(benchmark, "output_comparisons", comparisons);
            if (outputs.containsKey(Mode.CPU_FP32) && outputs.containsKey(Mode.CPU_QDQ))
                put(comparisons, "cpu_qdq_vs_cpu_fp32", audioComparison(outputs.get(Mode.CPU_QDQ), outputs.get(Mode.CPU_FP32)));
            if (outputs.containsKey(Mode.CPU_QDQ) && outputs.containsKey(Mode.QNN_HTP))
                put(comparisons, "qnn_htp_vs_cpu_qdq", audioComparison(outputs.get(Mode.QNN_HTP), outputs.get(Mode.CPU_QDQ)));
            put(comparisons, "scope", "Reference waveform errors for one public clip; no perceptual or speech-quality acceptance gate");
            boolean qnnVerified = backends.optJSONObject("qnn_htp") != null
                    && backends.getJSONObject("qnn_htp").optBoolean("execution_verified", false);
            JSONObject qnnResult = backends.optJSONObject("qnn_htp");
            JSONObject qnnEvidence = qnnResult == null ? null : qnnResult.optJSONObject("warmup_provider_evidence");
            boolean hybridVerified = qnnVerified && qnnEvidence != null && qnnEvidence.optInt("cpu_kernel_events") > 0;
            put(report, "full_model_hybrid_execution_verified", hybridVerified);
            put(benchmark, "hybrid_execution_verified", hybridVerified);
            put(report, "full_reuse_acceleration_verification_scope", "Full-product acceleration remains unverified. Complete hybrid execution and measured speed ratios are separate fields; no perceptual audio-quality approval.");
            put(benchmark, "hybrid_execution_verification_scope", "Finite complete causal sequence in the same static session whose three zero-input warm-up calls show QNN HTP and CPU provider events. Timed calls and flush are unprofiled; no per-timed-call provider trace, compute-share or speed-gain claim follows from warm-up evidence.");
            put(report, "full_model_hybrid_timed_execution_profile_verified", false);
            recordMeasuredSpeed(benchmark, report, backends, "cpu_fp32", "cpu_fp32");
            recordMeasuredSpeed(benchmark, report, backends, "cpu_qdq", "cpu_qdq");
            put(benchmark, "complete", allComplete);
            put(benchmark, "status", allComplete ? "FULL_SEQUENCE_COMPARISON_READY" : "PARTIAL_RESULTS");
            put(report, "status", allComplete ? "FULL_SEQUENCE_COMPARISON_READY" : "FULL_SEQUENCE_PARTIAL_RESULTS");
            stage = "full.complete";
        } catch (Exception | LinkageError error) {
            put(benchmark, "status", "ERROR");
            put(benchmark, "failure_stage", stage);
            put(benchmark, "error", error.toString());
            put(benchmark, "error_details", stackTrace(error));
            put(report, "status", "FULL_SEQUENCE_ERROR");
        } finally {
            put(benchmark, "total_test_wall_ms", elapsed(began));
            put(benchmark, "final_stage", stage);
            put(report, "last_stage", stage);
            if (directory != null) {
                try { writeDiagnostics(report, directory); }
                catch (Exception error) { put(report, "full_diagnostics_write_error", error.toString()); }
            }
        }
        return new QnnProbeRunner.Result(groupedResult.directory, report);
    }

    private float[] runBackend(Mode mode, ModelFiles files, float[] audio, JSONObject result,
                               JSONObject report, Path directory, QnnProbeRunner.Progress progress) {
        String stage = "full." + mode.key + ".initialize";
        long began = System.nanoTime();
        float[] trimmed = null;
        try {
            put(result, "model_sha256", files.modelHash);
            put(result, "manifest_sha256", files.manifestHash);
            put(result, "cpu_intra_op_threads", 2);
            put(result, "cpu_inter_op_threads", 1);
            put(result, "cpu_ep_fallback_allowed", mode == Mode.QNN_HTP);
            put(result, "fp16_relaxation_enabled", false);
            put(result, "expected_profile_model_calls", WARMUP_CALLS);
            put(result, "expected_total_model_calls", WARMUP_CALLS + EXPECTED_AUDIO_CALLS + 1);
            put(result, "timed_calls_profiled", false);
            put(result, "provider_verification_scope", "Three zero-input warm-up calls only; measured sequence and flush are unprofiled");
            Path profilePrefix = directory.resolve("full-" + mode.key + "-ort");
            stage = "full." + mode.key + ".session_create_compile";
            checkpoint(report, directory, stage);
            progress.update("Complete public clip · " + mode.name().replace('_', ' ') + " · preparing…");
            long compilationBegan = System.nanoTime();
            try (QnnHtpBackend qnn = mode == Mode.QNN_HTP ? new QnnHtpBackend(context, profilePrefix, true) : null) {
                SessionOptionsConfigurer configuration = qnn == null
                        ? (environment, options) -> options.enableProfiling(profilePrefix.toString()) : qnn;
                try (ChunkedReuseEnhancer enhancer = ChunkedReuseEnhancer.create(context, files.model,
                        files.manifest, new ReuseConfig(SAMPLE_RATE, 3, 2), configuration)) {
                    put(result, "session_create_compile_ms", elapsed(compilationBegan));
                    put(result, "enhancer_execution_diagnostics", new JSONObject(enhancer.executionDiagnostics()));
                    double[] warmupMs = new double[WARMUP_CALLS];
                    for (int i = 0; i < WARMUP_CALLS; i++) {
                        stage = "full." + mode.key + ".warmup." + (i + 1);
                        checkpoint(report, directory, stage);
                        long started = System.nanoTime();
                        float[] warm = enhancer.process(new float[CHUNK]);
                        warmupMs[i] = elapsed(started);
                        requireFinite(warm);
                    }
                    put(result, "warmup_call_wall_ms", numericArray(warmupMs));
                    stage = "full." + mode.key + ".verify_warmup_profile";
                    checkpoint(report, directory, stage);
                    Path profile = enhancer.endProfiling();
                    JSONObject evidence = profileSummary(new String(boundedProfileFile(profile), StandardCharsets.UTF_8));
                    put(result, "ort_profile_file", profile.getFileName().toString());
                    put(result, "warmup_provider_evidence", evidence);
                    requireWarmupCallCount(evidence);
                    if (qnn != null) {
                        JSONObject verified = new JSONObject(qnn.verifyProfile(profile, WARMUP_CALLS, false));
                        put(result, "qnn_backend_verification", verified);
                        put(result, "qnn_backend_diagnostics", new JSONObject(qnn.diagnostics()));
                        if (!verified.optBoolean("qnn_execution_verified", false))
                            throw new IOException("Full candidate has no verified QNN HTP execution during zero-input warm-up");
                    } else if (!cpuOnlyProfile(evidence)) {
                        throw new IOException("CPU warm-up profile contains missing or non-CPU provider attribution");
                    }
                    put(result, "warmup_provider_execution_verified", true);
                    stage = "full." + mode.key + ".reset";
                    checkpoint(report, directory, stage);
                    enhancer.reset();
                    float[] accumulated = new float[EXPECTED_AUDIO_CALLS * CHUNK];
                    float[] input = new float[CHUNK];
                    double[] milliseconds = new double[EXPECTED_AUDIO_CALLS];
                    int written = 0;
                    for (int call = 0; call < EXPECTED_AUDIO_CALLS; call++) {
                        if (call % 5 == 0) {
                            stage = "full." + mode.key + ".process_calls." + call + "-" + Math.min(call + 4, EXPECTED_AUDIO_CALLS - 1);
                            put(result, "completed_audio_calls", call);
                            checkpoint(report, directory, stage);
                            progress.update("Complete public clip · " + mode.name().replace('_', ' ') + " · " + call + "/" + EXPECTED_AUDIO_CALLS);
                        }
                        Arrays.fill(input, 0);
                        int available = Math.min(CHUNK, audio.length - call * CHUNK);
                        if (available > 0) System.arraycopy(audio, call * CHUNK, input, 0, available);
                        long started = System.nanoTime();
                        float[] output = enhancer.process(input);
                        milliseconds[call] = elapsed(started);
                        requireFinite(output);
                        int expected = call == 0 ? CHUNK - 160 : CHUNK;
                        if (output.length != expected || written + output.length > accumulated.length)
                            throw new IOException("Unexpected streaming output length at call " + call);
                        System.arraycopy(output, 0, accumulated, written, output.length);
                        written += output.length;
                    }
                    put(result, "completed_audio_calls", EXPECTED_AUDIO_CALLS);
                    stage = "full." + mode.key + ".flush";
                    checkpoint(report, directory, stage);
                    long flushBegan = System.nanoTime();
                    float[] tail = enhancer.flush();
                    double flushMs = elapsed(flushBegan);
                    requireFinite(tail);
                    if (tail.length != 160 || written + tail.length != accumulated.length)
                        throw new IOException("Unexpected final streaming synthesis tail");
                    System.arraycopy(tail, 0, accumulated, written, tail.length);
                    written += tail.length;
                    trimmed = Arrays.copyOf(accumulated, audio.length);
                    JSONObject timings = timingSummary(milliseconds, flushMs, audio.length);
                    for (java.util.Iterator<String> keys = timings.keys(); keys.hasNext();) {
                        String key = keys.next(); put(result, key, timings.get(key));
                    }
                    put(result, "source_input_samples", audio.length);
                    put(result, "padded_input_samples", accumulated.length);
                    put(result, "raw_output_samples_after_flush", written);
                    put(result, "trimmed_output_samples", trimmed.length);
                    put(result, "all_output_samples_finite", true);
                    stage = "full." + mode.key + ".save_public_outputs";
                    checkpoint(report, directory, stage);
                    put(result, "saved_public_output", saveAudio(directory, "full-" + mode.key.replace('_', '-'), trimmed));
                    stage = "full." + mode.key + ".close_enhancer";
                    checkpoint(report, directory, stage);
                }
                stage = "full." + mode.key + ".close_backend";
                checkpoint(report, directory, stage);
            }
            put(result, "execution_verified", true);
            put(result, "status", "VERIFIED_FINITE_FULL_SEQUENCE");
            put(result, "verification_scope", "Complete unprofiled public-clip causal sequence with exact sample accounting and finite output; provider attribution verified during three preceding zero-input warm-ups only. No per-timed-call profile, perceptual quality pass or live playback claim.");
        } catch (Exception | LinkageError error) {
            put(result, "execution_verified", false);
            put(result, "status", "ERROR");
            put(result, "failure_stage", stage);
            put(result, "error", error.toString());
            put(result, "error_details", stackTrace(error));
        } finally {
            put(result, "backend_total_wall_ms", elapsed(began));
            put(result, "final_stage", stage);
            try { writeDiagnostics(report, directory); }
            catch (Exception error) { put(result, "diagnostics_write_error", error.toString()); }
        }
        return trimmed;
    }

    private JSONObject groupedGate(JSONObject report, JSONObject manifest, Path directory) {
        JSONObject gate = new JSONObject();
        put(gate, "verified", false);
        try {
            JSONArray cases = report.getJSONArray("cases");
            JSONArray fixtures = manifest.getJSONArray("cases");
            Map<String, JSONObject> outcomes = matchGroupedOutcomes(fixtures, cases);
            Map<String, String> profiles = new LinkedHashMap<>(), profileErrors = new LinkedHashMap<>();
            for (Map.Entry<String, JSONObject> entry : outcomes.entrySet()) {
                try {
                    JSONObject qnn = entry.getValue().getJSONObject("qnn_htp");
                    String profileName = qnn.getString("ort_profile_file");
                    if (!safeAssetPath(profileName) || profileName.contains("/"))
                        throw new IOException("Unsafe prerequisite profile filename");
                    profiles.put(entry.getKey(), new String(boundedProfileFile(directory.resolve(profileName)), StandardCharsets.UTF_8));
                } catch (Exception error) { profileErrors.put(entry.getKey(), error.toString()); }
            }
            gate = evaluateGroupedPrerequisites(fixtures, cases, profiles);
            put(gate, "profile_load_errors", new JSONObject(profileErrors));
            JSONObject device = report.optJSONObject("selected_ep_device");
            boolean npu = device != null && "NPU".equals(device.optString("device_type"));
            boolean cleanup = !report.has("plugin_cleanup_error");
            put(gate, "npu_device_selected", npu);
            put(gate, "plugin_cleanup_succeeded", cleanup);
            if (!npu || !cleanup) {
                put(gate, "verified", false);
                put(gate, "global_error", !npu ? "Prerequisite QNN device was not an NPU" : "Prerequisite plugin cleanup failed");
            }
        } catch (Exception error) { put(gate, "verified", false); put(gate, "error", error.toString()); }
        return gate;
    }

    /** Fail closed on missing, duplicate, extra or nonmatching prerequisite IDs. */
    static Map<String, JSONObject> matchGroupedOutcomes(JSONArray fixtures, JSONArray outcomes) throws IOException {
        Map<String, JSONObject> expected = prerequisiteIds(fixtures, "fixture");
        Map<String, JSONObject> actual = prerequisiteIds(outcomes, "outcome");
        if (!expected.keySet().equals(actual.keySet()))
            throw new IOException("Prerequisite fixture and outcome ID sets do not match exactly");
        Map<String, JSONObject> ordered = new LinkedHashMap<>();
        for (String id : expected.keySet()) ordered.put(id, actual.get(id));
        return ordered;
    }

    private static Map<String, JSONObject> prerequisiteIds(JSONArray values, String label) throws IOException {
        if (values == null || values.length() < 1 || values.length() > 16)
            throw new IOException("Prerequisite " + label + " count must be between 1 and 16");
        Map<String, JSONObject> result = new LinkedHashMap<>();
        for (int i = 0; i < values.length(); i++) {
            JSONObject value = values.optJSONObject(i);
            Object rawId = value == null ? null : value.opt("id");
            if (!(rawId instanceof String) || !((String) rawId).matches("[a-zA-Z0-9_-]{1,80}"))
                throw new IOException("Invalid prerequisite " + label + " ID");
            String id = (String) rawId;
            if (result.put(id, value) != null) throw new IOException("Duplicate prerequisite " + label + " ID: " + id);
        }
        return result;
    }

    /** Pure prerequisite policy; every case gets separate numerical and raw-profile evidence. */
    static JSONObject evaluateGroupedPrerequisites(JSONArray fixtures, JSONArray outcomes,
                                                   Map<String, String> profiles) throws IOException {
        Map<String, JSONObject> matched = matchGroupedOutcomes(fixtures, outcomes);
        JSONObject gate = new JSONObject();
        JSONArray cases = new JSONArray();
        int verified = 0;
        for (Map.Entry<String, JSONObject> entry : matched.entrySet()) {
            JSONObject result = new JSONObject();
            put(result, "id", entry.getKey());
            put(result, "verified", false);
            cases.put(result);
            try {
                JSONObject qnn = entry.getValue().getJSONObject("qnn_htp");
                boolean executed = qnn.optBoolean("execution_verified", false);
                boolean correct = qnn.optBoolean("correctness_passed", false);
                boolean noFallback = qnn.optBoolean("cpu_ep_fallback_disabled", false);
                put(result, "execution_reported_verified", executed);
                put(result, "correctness_passed", correct);
                put(result, "cpu_ep_fallback_disabled", noFallback);
                String profile = profiles == null ? null : profiles.get(entry.getKey());
                if (profile == null) throw new IOException("Prerequisite profile is missing or unreadable");
                JSONObject evidence = profileSummary(profile);
                put(result, "profile_evidence", evidence);
                boolean strict = strictGroupedProfile(evidence, GROUPED_CALLS);
                put(result, "single_partition_profile_verified", strict);
                boolean passed = executed && correct && noFallback && strict;
                put(result, "verified", passed);
                if (passed) verified++;
                else put(result, "error", "Prerequisite needs correct QNN output, fallback disabled, and exactly one QNN partition for all 25 profiled calls with no CPU events");
            } catch (Exception error) { put(result, "error", error.toString()); }
        }
        put(gate, "case_count", matched.size());
        put(gate, "verified_case_count", verified);
        put(gate, "cases", cases);
        put(gate, "verified", verified == matched.size());
        if (verified != matched.size()) put(gate, "error", "At least one required QNN prerequisite did not pass");
        put(gate, "scope", "Every packaged prerequisite ID must have one matching verified numerical outcome and a single-partition, no-CPU-fallback profile; all prerequisites must pass before the full hybrid run");
        return gate;
    }

    static boolean strictGroupedProfile(JSONObject evidence, int calls) {
        return evidence.optInt("qnn_kernel_events") == calls
                && evidence.optInt("unique_qnn_kernel_count") == 1
                && evidence.optInt("cpu_kernel_events") == 0
                && evidence.optInt("other_provider_kernel_events") == 0
                && evidence.optInt("missing_provider_kernel_events") == 0
                && evidence.optInt("model_run_events") == calls;
    }

    static JSONObject profileSummary(String text) throws Exception {
        JSONArray events = new JSONArray(text);
        Map<String, Integer> providers = new LinkedHashMap<>();
        Set<String> qnnNames = new TreeSet<>();
        int qnn = 0, cpu = 0, other = 0, missing = 0, runs = 0;
        for (int i = 0; i < events.length(); i++) {
            JSONObject event = events.optJSONObject(i);
            if (event == null) continue;
            if ("Session".equals(event.optString("cat")) && "model_run".equals(event.optString("name"))) runs++;
            if (!"Node".equals(event.optString("cat")) || !event.optString("name").endsWith("_kernel_time")) continue;
            JSONObject args = event.optJSONObject("args");
            String provider = args == null ? "" : args.optString("provider");
            providers.put(provider.isEmpty() ? "missing" : provider, providers.getOrDefault(provider.isEmpty() ? "missing" : provider, 0) + 1);
            if ("QNNExecutionProvider".equals(provider)) { qnn++; qnnNames.add(event.optString("name")); }
            else if ("CPUExecutionProvider".equals(provider)) cpu++;
            else if (provider.isEmpty()) missing++;
            else other++;
        }
        JSONObject result = new JSONObject();
        put(result, "qnn_kernel_events", qnn); put(result, "cpu_kernel_events", cpu);
        put(result, "other_provider_kernel_events", other); put(result, "missing_provider_kernel_events", missing);
        put(result, "provider_kernel_counts", new JSONObject(providers)); put(result, "model_run_events", runs);
        put(result, "unique_qnn_kernel_count", qnnNames.size()); put(result, "unique_qnn_kernel_names", new JSONArray(qnnNames));
        put(result, "scope", "ORT provider attribution; event counts are not accelerator compute share or hardware utilization");
        return result;
    }

    private static boolean cpuOnlyProfile(JSONObject evidence) {
        return evidence.optInt("cpu_kernel_events") > 0 && evidence.optInt("qnn_kernel_events") == 0
                && evidence.optInt("other_provider_kernel_events") == 0 && evidence.optInt("missing_provider_kernel_events") == 0;
    }

    static void requireWarmupCallCount(JSONObject evidence) throws IOException {
        if (evidence.optInt("model_run_events", -1) != WARMUP_CALLS)
            throw new IOException("Profile must contain exactly three zero-input warm-up calls; measured sequence must be unprofiled");
    }

    static JSONObject timingSummary(double[] calls, double flushMs, int realSamples) throws IOException {
        if (calls.length == 0 || realSamples <= 0 || !Double.isFinite(flushMs) || flushMs < 0)
            throw new IOException("Invalid complete-stream timing data");
        double sum = 0, maximum = 0;
        int over = 0;
        for (double call : calls) {
            if (!Double.isFinite(call) || call < 0) throw new IOException("Invalid inference call time");
            sum += call; maximum = Math.max(maximum, call); if (call > 100) over++;
        }
        double[] sorted = calls.clone(); Arrays.sort(sorted);
        JSONObject result = new JSONObject();
        put(result, "audio_call_wall_ms", numericArray(calls));
        put(result, "mean_call_wall_ms", sum / calls.length);
        put(result, "p95_call_wall_ms", sorted[Math.max(0, (int) Math.ceil(sorted.length * .95) - 1)]);
        put(result, "max_call_wall_ms", maximum); put(result, "calls_over_100ms", over);
        put(result, "flush_wall_ms", flushMs); put(result, "processing_wall_ms", sum + flushMs);
        put(result, "offline_processing_rtf", (sum + flushMs) / (realSamples * 1000.0 / SAMPLE_RATE));
        put(result, "rtf_scope", "Unprofiled measured process plus flush time divided by real, unpadded input duration; no profiled warm-up/compilation or live playout");
        return result;
    }

    static JSONObject audioComparison(float[] actual, float[] reference) throws IOException {
        if (actual.length == 0 || actual.length != reference.length) throw new IOException("Waveform comparison lengths differ");
        requireFinite(actual); requireFinite(reference);
        double errorSquares = 0, referenceSquares = 0, absoluteSum = 0, maximum = 0;
        for (int i = 0; i < actual.length; i++) {
            double error = (double) actual[i] - reference[i];
            errorSquares += error * error; absoluteSum += Math.abs(error); maximum = Math.max(maximum, Math.abs(error));
            referenceSquares += (double) reference[i] * reference[i];
        }
        double rmse = Math.sqrt(errorSquares / actual.length), referenceRms = Math.sqrt(referenceSquares / actual.length);
        JSONObject result = new JSONObject();
        put(result, "samples", actual.length); put(result, "rmse", rmse); put(result, "max_abs", maximum);
        put(result, "mean_abs", absoluteSum / actual.length); put(result, "reference_rms", referenceRms);
        put(result, "normalized_rmse", referenceRms > 0 ? rmse / referenceRms : rmse == 0 ? 0 : "undefined_zero_reference");
        put(result, "snr_db", errorSquares == 0 ? "infinite_equal_outputs"
                : referenceSquares == 0 ? "undefined_zero_reference" : 10 * Math.log10(referenceSquares / errorSquares));
        put(result, "audio_quality_verified", false);
        return result;
    }

    static JSONObject saveAudio(Path directory, String prefix, float[] values) throws Exception {
        requireFinite(values);
        if (!prefix.matches("[a-z0-9-]{1,80}") || values.length > (MAX_BYTES - 44) / 4)
            throw new IOException("Invalid bounded output audio request");
        ByteBuffer raw = ByteBuffer.allocate(values.length * 4).order(ByteOrder.LITTLE_ENDIAN);
        raw.asFloatBuffer().put(values);
        Path floatFile = directory.resolve(prefix + ".f32");
        atomicWrite(floatFile, raw.array());
        ByteBuffer wav = ByteBuffer.allocate(44 + values.length * 2).order(ByteOrder.LITTLE_ENDIAN);
        wav.put("RIFF".getBytes(StandardCharsets.US_ASCII)).putInt(36 + values.length * 2)
                .put("WAVEfmt ".getBytes(StandardCharsets.US_ASCII)).putInt(16).putShort((short) 1).putShort((short) 1)
                .putInt(SAMPLE_RATE).putInt(SAMPLE_RATE * 2).putShort((short) 2).putShort((short) 16)
                .put("data".getBytes(StandardCharsets.US_ASCII)).putInt(values.length * 2);
        int outsideUnit = 0, saturated = 0;
        double peak = 0;
        for (float value : values) {
            if (value < -1 || value > 1) outsideUnit++;
            peak = Math.max(peak, Math.abs((double) value));
            long quantized = Math.round((double) value * 32768);
            if (quantized < -32768 || quantized > 32767) saturated++;
            wav.putShort((short) Math.max(-32768, Math.min(32767, quantized)));
        }
        Path waveFile = directory.resolve(prefix + ".wav");
        atomicWrite(waveFile, wav.array());
        JSONObject result = new JSONObject();
        put(result, "float32_file", floatFile.getFileName().toString()); put(result, "float32_sha256", sha256(raw.array()));
        put(result, "wav_file", waveFile.getFileName().toString()); put(result, "wav_sha256", sha256(wav.array()));
        put(result, "sample_rate", SAMPLE_RATE); put(result, "samples", values.length); put(result, "peak_abs_float", peak);
        put(result, "float_samples_outside_unit_range", outsideUnit); put(result, "pcm16_saturated_samples", saturated);
        put(result, "scope", "Packaged public test clip, not microphone recording; float32 file preserves unclipped samples and WAV is clipped PCM16");
        return result;
    }

    private ModelFiles copyModel(JSONObject spec, String expectedDirectory, Path cache, JSONObject topManifest) throws Exception {
        String modelName = spec.getString("model"), manifestName = spec.getString("manifest");
        if (!modelName.equals(expectedDirectory + "/model.onnx") || !manifestName.equals(expectedDirectory + "/manifest.json"))
            throw new IOException("Unexpected complete-model asset paths");
        byte[] model = checkedAsset(modelName, spec.getString("model_sha256"));
        byte[] manifestBytes = asset(manifestName);
        String manifestHash = sha256(manifestBytes);
        if (spec.has("manifest_sha256") && !manifestHash.equalsIgnoreCase(spec.getString("manifest_sha256")))
            throw new IOException("Full-model manifest hash mismatch");
        JSONObject modelManifest = new JSONObject(new String(manifestBytes, StandardCharsets.UTF_8));
        if (!spec.getString("model_sha256").equalsIgnoreCase(modelManifest.getString("model_sha256")))
            throw new IOException("Full-model manifest does not bind the packaged graph hash");
        String licenseName = expectedDirectory + "/LICENSE-NVIDIA.txt";
        byte[] license = asset(licenseName), originalLicense = asset(topManifest.getString("model_derived_license"));
        if (license.length == 0 || !Arrays.equals(license, originalLicense)) throw new IOException("Full-model NVIDIA license is missing or differs");
        if (spec.has("license_sha256") && !sha256(license).equalsIgnoreCase(spec.getString("license_sha256")))
            throw new IOException("Full-model license hash mismatch");
        Path target = cache.resolve(expectedDirectory);
        Files.createDirectories(target);
        atomicWrite(target.resolve("model.onnx"), model);
        atomicWrite(target.resolve("manifest.json"), manifestBytes);
        atomicWrite(target.resolve("LICENSE-NVIDIA.txt"), license);
        return new ModelFiles(target.resolve("model.onnx"), target.resolve("manifest.json"), sha256(model), manifestHash);
    }

    private ModelFiles prepareModel(JSONObject full, String key, String directory, Path cache,
                                    JSONObject manifest, JSONObject errors) {
        try { return copyModel(full.getJSONObject(key), directory, cache, manifest); }
        catch (Exception error) { put(errors, key, stackTrace(error)); return null; }
    }

    private static void recordMeasuredSpeed(JSONObject benchmark, JSONObject report, JSONObject backends,
                                            String cpuKey, String label) {
        JSONObject cpu = backends.optJSONObject(cpuKey), qnn = backends.optJSONObject("qnn_htp");
        double cpuMean = cpu == null ? Double.NaN : cpu.optDouble("mean_call_wall_ms", Double.NaN);
        double qnnMean = qnn == null ? Double.NaN : qnn.optDouble("mean_call_wall_ms", Double.NaN);
        boolean valid = cpu != null && qnn != null && cpu.optBoolean("execution_verified", false)
                && qnn.optBoolean("execution_verified", false) && Double.isFinite(cpuMean) && Double.isFinite(qnnMean)
                && cpuMean > 0 && qnnMean > 0;
        put(benchmark, label + "_over_hybrid_call_speedup", valid ? cpuMean / qnnMean : JSONObject.NULL);
        put(benchmark, "hybrid_faster_than_" + label, valid ? qnnMean < cpuMean : JSONObject.NULL);
        put(report, "full_model_hybrid_faster_than_" + label, valid ? qnnMean < cpuMean : JSONObject.NULL);
    }

    private static final class ModelFiles {
        final Path model, manifest;
        final String modelHash, manifestHash;
        ModelFiles(Path model, Path manifest, String modelHash, String manifestHash) {
            this.model = model; this.manifest = manifest; this.modelHash = modelHash; this.manifestHash = manifestHash;
        }
    }

    private byte[] checkedAsset(String filename, String expectedHash) throws Exception {
        if (!expectedHash.matches("[a-fA-F0-9]{64}")) throw new IOException("Invalid asset SHA-256");
        byte[] data = asset(filename);
        if (!sha256(data).equalsIgnoreCase(expectedHash)) throw new IOException("Packaged asset hash mismatch: " + filename);
        return data;
    }

    static boolean safeAssetPath(String filename) {
        return filename != null && filename.matches("[a-zA-Z0-9][a-zA-Z0-9_.-]{0,159}(/[a-zA-Z0-9][a-zA-Z0-9_.-]{0,159})?")
                && !filename.contains("..");
    }

    private byte[] asset(String filename) throws IOException {
        if (!safeAssetPath(filename)) throw new IOException("Unsafe full-stream asset filename");
        try (InputStream input = context.getAssets().open("qnn/" + filename);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] block = new byte[8192]; int read;
            while ((read = input.read(block)) != -1) {
                if ((long) output.size() + read > MAX_BYTES) throw new IOException("Asset exceeds 32 MiB");
                output.write(block, 0, read);
            }
            return output.toByteArray();
        }
    }

    private static byte[] boundedFile(Path file) throws IOException {
        if (!Files.isRegularFile(file) || Files.size(file) > MAX_BYTES) throw new IOException("Required file is missing or exceeds 32 MiB: " + file.getFileName());
        return Files.readAllBytes(file);
    }

    private static byte[] boundedProfileFile(Path file) throws IOException {
        if (!Files.isRegularFile(file) || Files.size(file) > MAX_PROFILE_BYTES)
            throw new IOException("ORT profile is missing or exceeds 64 MiB: " + file.getFileName());
        return Files.readAllBytes(file);
    }

    private static void requireFinite(float[] values) throws IOException {
        if (values == null || values.length == 0) throw new IOException("Expected nonempty finite PCM output");
        for (float value : values) if (!Float.isFinite(value)) throw new IOException("PCM contains a non-finite value");
    }

    private static Object buildFlag(String name) {
        try { return BuildConfig.class.getField(name).get(null); }
        catch (ReflectiveOperationException error) { return "reported_build_flag_unavailable"; }
    }

    private static String sha256(byte[] data) throws Exception {
        StringBuilder result = new StringBuilder();
        for (byte value : MessageDigest.getInstance("SHA-256").digest(data)) result.append(String.format(Locale.ROOT, "%02x", value & 255));
        return result.toString();
    }

    private static void atomicWrite(Path file, byte[] data) throws IOException {
        Path temporary = file.resolveSibling(file.getFileName().toString() + ".tmp");
        Files.write(temporary, data);
        try { Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE); }
        catch (AtomicMoveNotSupportedException error) { Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING); }
    }

    private static void checkpoint(JSONObject report, Path directory, String stage) throws Exception {
        put(report, "status", "RUNNING"); put(report, "last_stage", stage); put(report, "checkpoint_utc", utc());
        writeDiagnostics(report, directory);
    }

    private static void writeDiagnostics(JSONObject report, Path directory) throws Exception {
        byte[] data = (report.toString(2) + "\n").getBytes(StandardCharsets.UTF_8);
        for (String name : new String[]{"stats.json", "phone-capture.json", "qnn-probe.json"}) atomicWrite(directory.resolve(name), data);
    }

    private static String utc() {
        SimpleDateFormat format = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss.SSS'Z'", Locale.ROOT);
        format.setTimeZone(java.util.TimeZone.getTimeZone("UTC")); return format.format(new Date());
    }

    private static double elapsed(long started) { return (System.nanoTime() - started) / 1e6; }
    private static JSONArray numericArray(double[] values) {
        JSONArray result = new JSONArray();
        for (double value : values) {
            try { result.put(value); }
            catch (org.json.JSONException error) { throw new IllegalArgumentException("Non-finite numeric array", error); }
        }
        return result;
    }
    private static String stackTrace(Throwable error) {
        StringWriter text = new StringWriter(); error.printStackTrace(new PrintWriter(text)); return text.toString();
    }
    private static void put(JSONObject object, String key, Object value) {
        try { object.put(key, value); }
        catch (org.json.JSONException error) { throw new IllegalStateException("Cannot serialize " + key, error); }
    }
}
