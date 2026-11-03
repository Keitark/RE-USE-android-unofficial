package io.github.reuse;

import android.os.Build;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.providers.NNAPIFlags;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collections;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/** Shared provider policy for both streaming entry points. */
final class BackendSessionOptions {
    private BackendSessionOptions() { }

    static void validateAccelerator(ReuseConfig config, int androidApi, Set<OrtProvider> providers)
            throws IOException {
        if (config.backend == ReuseConfig.Backend.CPU) return;
        if (androidApi < 29)
            throw new IOException("Experimental NNAPI acceleration requires Android 10 / API 29 or newer");
        if (!providers.contains(OrtProvider.NNAPI))
            throw new IOException("This ONNX Runtime build does not include the NNAPI execution provider");
    }

    static void apply(ReuseConfig config, OrtSession.SessionOptions options) throws IOException, OrtException {
        if (config.backend == ReuseConfig.Backend.CPU) return;
        validateAccelerator(config, Build.VERSION.SDK_INT, OrtEnvironment.getAvailableProviders());
        EnumSet<NNAPIFlags> flags = EnumSet.of(NNAPIFlags.CPU_DISABLED);
        if (config.nnapiUseFp16) flags.add(NNAPIFlags.USE_FP16);
        Path parent = config.nnapiProfilingPrefix.toAbsolutePath().getParent();
        if (parent == null || !Files.isDirectory(parent) || !Files.isWritable(parent))
            throw new IOException("NNAPI warm-up profiling directory must already exist and be writable");
        options.enableProfiling(config.nnapiProfilingPrefix.toAbsolutePath().toString());
        // Unsupported operators may still execute through ORT CPU kernels. Do not substitute a
        // second CPU session if provider registration or session creation fails.
        options.addNnapi(flags);
    }

    static Map<String, Object> diagnostics(ReuseConfig config, OrtEnvironment environment) {
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("requested_backend", config.backend.name());
        values.put("nnapi_fp16_relaxation_requested", config.nnapiUseFp16);
        values.put("hybrid_execution_requested", config.backend == ReuseConfig.Backend.NNAPI_ACCELERATOR);
        values.put("accelerator_coverage_verified", false);
        values.put("gpu_execution_verified", false);
        values.put("nnapi_execution_verified", false);
        values.put("execution_scope", config.backend == ReuseConfig.Backend.CPU ? "ORT CPU"
                : "NNAPI with ORT CPU for unsupported operators; delegated operators and actual device unverified");
        values.put("ort_version", environment.getVersion());
        StringBuilder providers = new StringBuilder();
        for (OrtProvider provider : OrtEnvironment.getAvailableProviders()) {
            if (providers.length() != 0) providers.append(',');
            providers.append(provider.getName());
        }
        values.put("available_execution_providers", providers.toString());
        return Collections.unmodifiableMap(values);
    }

    static Map<String, Object> finishVerification(OrtSession session, Map<String, Object> previous)
            throws OrtException, IOException {
        String filename = session.endProfiling();
        Path file = java.nio.file.Paths.get(filename);
        // Startup captures contain no microphone samples. Bound parsing despite app-private input.
        if (!Files.isRegularFile(file) || Files.size(file) > 32L * 1024 * 1024)
            throw new IOException("NNAPI warm-up profiling output is missing or too large");
        int count = countNnapiKernelEvents(new String(Files.readAllBytes(file), StandardCharsets.UTF_8));
        Map<String, Object> values = new LinkedHashMap<>(previous);
        values.put("nnapi_warmup_kernel_events", count);
        values.put("nnapi_execution_verified", count > 0);
        values.put("nnapi_warmup_profile", filename);
        values.put("accelerator_verification_scope", "NNAPI provider kernel execution during zero-audio warm-up only; device and operator coverage unverified");
        return Collections.unmodifiableMap(values);
    }

    static int countNnapiKernelEvents(String text) throws IOException {
        try {
            JSONArray events = new JSONArray(text);
            int count = 0;
            for (int i = 0; i < events.length(); i++) {
                JSONObject event = events.optJSONObject(i);
                if (event == null || !"Node".equals(event.optString("cat"))
                        || !event.optString("name").endsWith("_kernel_time")) continue;
                JSONObject args = event.optJSONObject("args");
                if (args != null && "NnapiExecutionProvider".equals(args.optString("provider"))) count++;
            }
            return count;
        } catch (JSONException error) {
            throw new IOException("Invalid NNAPI warm-up profiling output", error);
        }
    }
}
