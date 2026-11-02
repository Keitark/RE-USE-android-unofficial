package io.github.reuse.qnn;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.TreeMap;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/** Provider-attribution evidence, separated from Android and native runtime loading for testing. */
final class QnnProfileVerifier {
    private QnnProfileVerifier() { }

    static Map<String, Object> verify(String text, int expectedCalls, boolean allowCpuFallback,
                                      boolean requireSinglePartition) throws IOException {
        if (expectedCalls < 1) throw new IllegalArgumentException("expectedCalls must be positive");
        if (text == null) throw new IOException("ORT profile JSON is required");
        int qnn = 0, cpu = 0, other = 0, missing = 0;
        Set<String> names = new TreeSet<>();
        Map<String, Integer> kernelCounts = new TreeMap<>();
        try {
            JSONArray events = new JSONArray(text);
            for (int i = 0; i < events.length(); i++) {
                JSONObject event = events.optJSONObject(i);
                if (event == null || !"Node".equals(event.optString("cat"))
                        || !event.optString("name").endsWith("_kernel_time")) continue;
                JSONObject args = event.optJSONObject("args");
                String provider = args == null ? "" : args.optString("provider");
                if ("QNNExecutionProvider".equals(provider)) {
                    qnn++;
                    String name = event.optString("name");
                    names.add(name);
                    kernelCounts.put(name, kernelCounts.getOrDefault(name, 0) + 1);
                } else if ("CPUExecutionProvider".equals(provider)) cpu++;
                else if (provider.isEmpty()) missing++;
                else other++;
            }
        } catch (JSONException error) {
            throw new IOException("Invalid ORT profile JSON", error);
        }
        List<String> errors = new ArrayList<>();
        if (qnn == 0) errors.add("No QNN provider kernel execution was observed");
        if (qnn < expectedCalls) errors.add("QNN kernel events are fewer than the expected inference calls");
        if (!allowCpuFallback && cpu > 0) errors.add("CPU kernel events violate the no-fallback policy");
        if (other > 0) errors.add("Another execution provider appeared in the profile");
        if (missing > 0) errors.add("Some kernel events have no provider attribution");
        boolean singlePartition = qnn == expectedCalls && names.size() == 1;
        if (requireSinglePartition && !singlePartition)
            errors.add("Single-partition verification requires exactly one QNN kernel name and one event per expected call");
        boolean verified = errors.isEmpty();
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("qnn_kernel_events", qnn);
        result.put("cpu_kernel_events", cpu);
        result.put("other_provider_kernel_events", other);
        result.put("missing_provider_kernel_events", missing);
        result.put("unique_qnn_kernel_names", Collections.unmodifiableList(new ArrayList<>(names)));
        result.put("unique_qnn_kernel_count", names.size());
        result.put("qnn_kernel_event_counts", Collections.unmodifiableMap(new LinkedHashMap<>(kernelCounts)));
        result.put("qnn_kernel_counts_match_expected_calls", !kernelCounts.isEmpty()
                && kernelCounts.values().stream().allMatch(count -> count == expectedCalls));
        result.put("expected_calls", expectedCalls);
        result.put("cpu_fallback_allowed", allowCpuFallback);
        result.put("cpu_fallback_observed", cpu > 0);
        result.put("require_single_partition", requireSinglePartition);
        result.put("single_partition_condition_met", singlePartition);
        result.put("single_qnn_partition_verified", verified && singlePartition);
        result.put("qnn_execution_observed", qnn > 0);
        result.put("qnn_execution_verified", verified);
        result.put("qnn_only_kernel_execution_verified", verified && cpu == 0);
        result.put("verification_errors", Collections.unmodifiableList(errors));
        result.put("execution_verification_scope", "ORT provider kernel events only; event counts do not measure compute share or hardware utilization. Numerical correctness and audio quality require separate caller validation.");
        return Collections.unmodifiableMap(result);
    }
}
