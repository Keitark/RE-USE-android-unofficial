package io.github.reuse.phone;

import android.content.Context;
import io.github.reuse.audio.qnn.QnnAudioSession;
import io.github.reuse.live.EnhancementSession;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.LinkOption;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.Locale;
import org.json.JSONArray;
import org.json.JSONObject;

/** Optional QNN session preparation. Never opens a microphone or replaces a failed backend with CPU. */
final class PhoneSessionProviders {
    private static final int MAX_MODEL_BYTES = 32 * 1024 * 1024;
    private static final int MAX_MANIFEST_BYTES = 1024 * 1024;
    private static final int MAX_LICENSE_BYTES = 128 * 1024;
    private static final String NVIDIA_LICENSE_SHA = "6d1daa9c89ec421ba80fe051ac9f0dd46010d4fa4e8f4af654eb8e62f64e2ad5";
    private static final Object CACHE_LOCK = new Object();

    private PhoneSessionProviders() { }

    static PhoneCaptureController.SessionFactory optionalQnn(Context context) {
        if (context == null) throw new IllegalArgumentException("Android context is required");
        Context application = context.getApplicationContext();
        return (outputDirectory, threads, listener, initializer) -> {
            ModelFiles files = prepareModel(application);
            EnhancementSession session = QnnAudioSession.create(application, files.model, files.manifest,
                    outputDirectory, initializer, threads, listener);
            return new PhoneCaptureController.PreparedSession(session, files.modelSha);
        };
    }

    /** Explicit CPU comparison using the same candidate graph; this is never a fallback. */
    static PhoneCaptureController.SessionFactory optionalQuantizedCpu(Context context) {
        if (context == null) throw new IllegalArgumentException("Android context is required");
        Context application = context.getApplicationContext();
        return (outputDirectory, threads, listener, initializer) -> {
            ModelFiles files = prepareModel(application);
            Path customOps = application.getApplicationInfo().nativeLibraryDir == null ? null
                    : new java.io.File(application.getApplicationInfo().nativeLibraryDir, "libreuse_ort_ops.so").toPath();
            if (customOps == null || !Files.isRegularFile(customOps))
                throw new IOException("Packaged CPU custom-operation library is missing");
            EnhancementSession session = new EnhancementSession(files.model.getParent(), outputDirectory,
                    customOps, listener, initializer, threads);
            return new PhoneCaptureController.PreparedSession(session, files.modelSha);
        };
    }

    private static ModelFiles prepareModel(Context context) throws Exception {
        synchronized (CACHE_LOCK) {
            String expected = checkedSha(BuildConfig.QNN_MODEL_SHA, "Configured QNN model SHA-256");
            byte[] model = asset(context, "model.onnx", MAX_MODEL_BYTES);
            String actual = sha256(model);
            if (!actual.equals(expected)) throw new IOException("Packaged QNN model SHA-256 differs from the configured candidate");
            byte[] manifestBytes = asset(context, "manifest.json", MAX_MANIFEST_BYTES);
            String manifestSha = sha256(manifestBytes);
            String configuredManifestSha = optionalManifestSha();
            if (configuredManifestSha != null && !manifestSha.equals(configuredManifestSha))
                throw new IOException("Packaged QNN model manifest SHA-256 differs from its build configuration");
            JSONObject manifest = new JSONObject(new String(manifestBytes, StandardCharsets.UTF_8));
            validateManifest(manifest, actual, model.length);
            byte[] license = asset(context, "LICENSE-NVIDIA.txt", MAX_LICENSE_BYTES);
            if (license.length == 0 || !NVIDIA_LICENSE_SHA.equals(sha256(license)))
                throw new IOException("Packaged QNN candidate does not contain the complete pinned NVIDIA NSCLv1 license");

            Path filesRoot = context.getFilesDir().toPath();
            Files.createDirectories(filesRoot);
            filesRoot = filesRoot.toRealPath();
            Path cache = filesRoot.resolve("qnn-model-" + actual);
            Files.createDirectories(cache);
            if (!cache.equals(cache.toRealPath()) || !cache.getParent().equals(filesRoot))
                throw new IOException("QNN model cache is outside the expected private directory");
            Path modelFile = ensureCached(cache, "model.onnx", model, actual, MAX_MODEL_BYTES);
            Path manifestFile = ensureCached(cache, "manifest.json", manifestBytes, manifestSha, MAX_MANIFEST_BYTES);
            ensureCached(cache, "LICENSE-NVIDIA.txt", license, NVIDIA_LICENSE_SHA, MAX_LICENSE_BYTES);
            // Re-read both bound files after any repair. The returned identity is
            // measured from the cached graph that the deferred session will open.
            String cachedSha = fileSha(modelFile, MAX_MODEL_BYTES);
            if (!cachedSha.equals(actual) || !fileSha(manifestFile, MAX_MANIFEST_BYTES).equals(manifestSha))
                throw new IOException("QNN model cache verification failed after preparation");
            return new ModelFiles(modelFile, manifestFile, cachedSha);
        }
    }

    private static void validateManifest(JSONObject manifest, String modelSha, int modelBytes) throws Exception {
        if (manifest.getInt("format_version") != 1 || manifest.getInt("sample_rate") != 8000
                || manifest.getInt("exit_layers") != 3 || manifest.getInt("frames_per_call") != 5
                || manifest.getInt("look_ahead_frames") != 0 || manifest.getInt("fft_size") != 320
                || manifest.getInt("hop_size") != 160 || manifest.getInt("chunk_size") != 800)
            throw new IOException("QNN candidate must use 8 kHz, exit 3, five hops and zero look-ahead");
        if (!checkedSha(manifest.getString("model_sha256"), "QNN manifest model SHA-256").equals(modelSha)
                || manifest.getLong("model_bytes") != modelBytes)
            throw new IOException("QNN manifest does not bind the packaged model bytes");
        if (manifest.optBoolean("commercial_use", true) || !manifest.getString("license").contains("NSCLv1"))
            throw new IOException("QNN candidate license metadata differs from NVIDIA NSCLv1");
        JSONArray states = manifest.getJSONArray("states");
        if (states.length() != 21) throw new IOException("QNN candidate must retain all 21 FP32 recurrent states");
        for (int i = 0; i < states.length(); i++) {
            if (!"float32".equals(states.getJSONObject(i).getString("dtype")))
                throw new IOException("QNN candidate recurrent states must remain FP32");
        }
    }

    private static Path ensureCached(Path directory, String filename, byte[] bytes, String expected, int limit) throws Exception {
        Path file = directory.resolve(filename);
        boolean valid = Files.isRegularFile(file, LinkOption.NOFOLLOW_LINKS)
                && Files.size(file) == bytes.length && expected.equals(fileSha(file, limit));
        if (!valid) {
            // A unique newly created temporary file avoids following a stale
            // .tmp symlink. Only these three files in this checked cache change.
            Path temporary = Files.createTempFile(directory, filename + ".", ".tmp");
            try {
                Files.write(temporary, bytes);
                if (!expected.equals(fileSha(temporary, limit))) throw new IOException("QNN cache temporary-file hash mismatch");
                try { Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING, StandardCopyOption.ATOMIC_MOVE); }
                catch (AtomicMoveNotSupportedException error) { Files.move(temporary, file, StandardCopyOption.REPLACE_EXISTING); }
            } finally { Files.deleteIfExists(temporary); }
        }
        if (!Files.isRegularFile(file, LinkOption.NOFOLLOW_LINKS) || !expected.equals(fileSha(file, limit)))
            throw new IOException("QNN candidate cache hash mismatch: " + filename);
        return file;
    }

    private static byte[] asset(Context context, String filename, int limit) throws IOException {
        // All asset names are constants; model metadata supplies no file paths.
        try (InputStream input = context.getAssets().open("qnn-model/" + filename);
             ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] block = new byte[8192]; int read;
            while ((read = input.read(block)) != -1) {
                if ((long) output.size() + read > limit) throw new IOException("QNN candidate asset exceeds its bounded size: " + filename);
                output.write(block, 0, read);
            }
            return output.toByteArray();
        }
    }

    private static String fileSha(Path file, int limit) throws Exception {
        if (!Files.isRegularFile(file, LinkOption.NOFOLLOW_LINKS) || Files.size(file) > limit)
            throw new IOException("QNN cached file is missing, symbolic, or too large: " + file.getFileName());
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream input = Files.newInputStream(file)) {
            byte[] block = new byte[8192]; int read, total = 0;
            while ((read = input.read(block)) != -1) {
                total += read;
                if (total > limit) throw new IOException("QNN cached file changed beyond its size limit");
                digest.update(block, 0, read);
            }
        }
        return hex(digest.digest());
    }

    private static String checkedSha(String value, String label) throws IOException {
        if (value == null || !value.matches("[0-9a-fA-F]{64}")) throw new IOException(label + " must contain 64 hexadecimal characters");
        return value.toLowerCase(Locale.ROOT);
    }

    private static String optionalManifestSha() throws Exception {
        try {
            Object value = BuildConfig.class.getField("QNN_MANIFEST_SHA").get(null);
            if (!(value instanceof String)) throw new IOException("Configured QNN manifest SHA-256 must be a string");
            return checkedSha((String) value, "Configured QNN manifest SHA-256");
        } catch (NoSuchFieldException absent) { return null; }
    }

    private static String sha256(byte[] bytes) throws Exception { return hex(MessageDigest.getInstance("SHA-256").digest(bytes)); }
    private static String hex(byte[] bytes) {
        StringBuilder value = new StringBuilder();
        for (byte item : bytes) value.append(String.format(Locale.ROOT, "%02x", item & 255));
        return value.toString();
    }

    private static final class ModelFiles {
        final Path model, manifest;
        final String modelSha;
        ModelFiles(Path model, Path manifest, String modelSha) { this.model = model; this.manifest = manifest; this.modelSha = modelSha; }
    }
}
