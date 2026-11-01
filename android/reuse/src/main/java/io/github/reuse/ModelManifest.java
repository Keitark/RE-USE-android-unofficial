package io.github.reuse;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/** Export metadata is checked before allocating native buffers or executing the graph. */
final class ModelManifest {
    static final class State {
        final String input;
        final String output;
        final long[] shape;
        final int elements;

        State(String input, String output, long[] shape, int elements) {
            this.input = input;
            this.output = output;
            this.shape = shape;
            this.elements = elements;
        }
    }

    final List<State> states;
    final boolean requiresSelectiveScan;
    final int framesPerCall;

    private ModelManifest(List<State> states, boolean requiresSelectiveScan, int framesPerCall) {
        this.states = states;
        this.requiresSelectiveScan = requiresSelectiveScan;
        this.framesPerCall = framesPerCall;
    }

    static ModelManifest read(Path path, Path model, ReuseConfig config) throws IOException {
        if (Files.size(path) > 1024 * 1024) throw new IOException("Manifest exceeds 1 MiB");
        try {
            JSONObject json = new JSONObject(new String(Files.readAllBytes(path), StandardCharsets.UTF_8));
            requireInt(json, "format_version", 1);
            requireInt(json, "sample_rate", config.sampleRate);
            requireInt(json, "fft_size", config.fftSize);
            requireInt(json, "hop_size", config.hopSize);
            requireInt(json, "exit_layers", config.exitLayers);
            requireInt(json, "look_ahead_frames", 0);
            int framesPerCall = 1;
            if (json.has("frames_per_call")) {
                Object value = json.get("frames_per_call");
                if (!(value instanceof Number)) throw new IOException("frames_per_call must be 1 or 5");
                double count = ((Number) value).doubleValue();
                if (count != 1.0 && count != 5.0) throw new IOException("frames_per_call must be 1 or 5");
                framesPerCall = (int) count;
            }
            if (json.has("chunk_size")) {
                Object value = json.get("chunk_size");
                if (!(value instanceof Number) || ((Number) value).doubleValue() != framesPerCall * config.hopSize) {
                    throw new IOException("Manifest chunk_size must equal frames_per_call * hop_size");
                }
            }
            String expectedHash = json.getString("model_sha256");
            if (!expectedHash.matches("[0-9a-fA-F]{64}") ||
                    !expectedHash.equalsIgnoreCase(sha256(model))) {
                throw new IOException("Model SHA-256 does not match manifest");
            }
            boolean requiresSelectiveScan = false;
            if (json.has("custom_ops")) {
                JSONArray customOps = json.getJSONArray("custom_ops");
                if (customOps.length() > 7) throw new IOException("Unsupported custom operation list");
                Set<String> operations = new HashSet<>();
                for (int i = 0; i < customOps.length(); i++) {
                    JSONObject operation = customOps.getJSONObject(i);
                    String name = operation.getString("name");
                    if (!operation.getString("domain").equals("com.reuse") ||
                            !(name.equals("SelectiveScan") || name.equals("SelectiveStep") || name.equals("SelectiveSequence") || name.equals("FastSoftplus") || name.equals("TemporalConvSilu") || name.equals("ChannelNormPRelu") || name.equals("FrequencyConvSilu")) ||
                            !operations.add(name) ||
                            operation.getInt("version") != 1 ||
                            !operation.getString("library").equals("reuse_ort_ops")) {
                        throw new IOException("Unsupported or duplicate custom operation; expected a bundled com.reuse v1 operation");
                    }
                    requiresSelectiveScan = true;
                }
            }
            JSONArray entries = json.getJSONArray("states");
            if (entries.length() == 0 || entries.length() > 256) {
                throw new IOException("Manifest must contain 1..256 explicit recurrent states");
            }
            List<State> states = new ArrayList<>();
            Set<String> inputs = new HashSet<>(), outputs = new HashSet<>();
            long totalElements = 0;
            for (int i = 0; i < entries.length(); i++) {
                JSONObject entry = entries.getJSONObject(i);
                String input = entry.getString("input"), output = entry.getString("output");
                if (!input.matches("state_[0-9]{3}") || !output.equals("next_" + input) ||
                        !inputs.add(input) || !outputs.add(output)) {
                    throw new IOException("Invalid or duplicate state names");
                }
                JSONArray dimensions = entry.getJSONArray("shape");
                if (dimensions.length() < 1 || dimensions.length() > 6) {
                    throw new IOException("Invalid recurrent state rank");
                }
                long[] shape = new long[dimensions.length()];
                long count = 1;
                for (int d = 0; d < shape.length; d++) {
                    shape[d] = dimensions.getLong(d);
                    if (shape[d] < 1 || shape[d] > 16 * 1024 * 1024 ||
                            count > (16L * 1024 * 1024) / shape[d]) {
                        throw new IOException("Recurrent state too large or not static");
                    }
                    count *= shape[d];
                }
                totalElements += count;
                if (totalElements > 32L * 1024 * 1024) {
                    throw new IOException("Recurrent state allocation exceeds 128 MiB per buffer set");
                }
                states.add(new State(input, output, shape, (int) count));
            }
            return new ModelManifest(states, requiresSelectiveScan, framesPerCall);
        } catch (JSONException exception) {
            throw new IOException("Invalid RE-USE model manifest", exception);
        }
    }

    private static void requireInt(JSONObject json, String key, int expected) throws IOException, JSONException {
        if (json.getInt(key) != expected) {
            throw new IOException("Manifest " + key + " must equal " + expected);
        }
    }

    private static String sha256(Path path) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] bytes = new byte[65536];
            try (InputStream stream = Files.newInputStream(path)) {
                int read;
                while ((read = stream.read(bytes)) != -1) digest.update(bytes, 0, read);
            }
            StringBuilder result = new StringBuilder(64);
            for (byte value : digest.digest()) result.append(String.format("%02x", value & 255));
            return result.toString();
        } catch (NoSuchAlgorithmException impossible) {
            throw new AssertionError(impossible);
        }
    }
}
