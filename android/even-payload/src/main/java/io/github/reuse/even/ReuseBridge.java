package io.github.reuse.even;

import android.content.Context;
import android.content.res.AssetManager;
import android.util.Log;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Arrays;
import org.json.JSONException;
import org.json.JSONObject;

/** Framework-only bootstrap and JNI boundary for the LC3 interposer. */
public final class ReuseBridge {
    private static final String TAG = "ReuseEven";
    private static boolean attempted;
    private static boolean configured;
    private static boolean nativeLoaded;
    private static int sampleRate = 8000;
    private static int threads = 2;
    private static File activeModelDirectory;
    private static byte[] activeConfigBytes;
    private static String activeRuntimeProfile = "default";
    private static boolean probeReconfigured;
    private static volatile String status = "{\"configured\":false,\"stage\":\"not_initialized\"}";

    private ReuseBridge() { }

    public static native boolean nativeConfigure(String modelDir, String nativeLibDir,
            int sampleRate, int threads);
    public static native String nativeGetDiagnostics();
    public static native void nativeReset();
    public static native String nativeSelfTest(String inputPcmPath, String outputPcmPath,
            int sampleRate);

    /** Runs once per application process, before the normal activity starts. */
    public static synchronized boolean initialize(Context context) {
        if (attempted) return configured;
        attempted = true;
        File directory = modelDirectory(context);
        JSONObject report = new JSONObject();
        try {
            if (!directory.isDirectory() && !directory.mkdirs()) {
                throw new IOException("Cannot create private RE-USE directory");
            }
            byte[] configBytes = readAsset(context.getAssets(), "reuse/config.json", 16384);
            JSONObject config = new JSONObject(new String(configBytes, StandardCharsets.UTF_8));
            String runtimeProfile = runtimeProfile(config);
            boolean enabled = config.getBoolean("enabled");
            sampleRate = config.getInt("sample_rate");
            threads = config.getInt("threads");
            if ((sampleRate != 8000 && sampleRate != 16000) || threads < 1 || threads > 16) {
                throw new IOException("Invalid RE-USE sample_rate or threads");
            }
            writeAtomically(new File(directory, "config.json"), configBytes);
            report.put("enabled", enabled).put("sample_rate", sampleRate).put("threads", threads)
                    .put("runtime_profile", runtimeProfile);
            if (!enabled) {
                report.put("configured", false).put("stage", "disabled");
                record(directory, report);
                return false;
            }

            byte[] manifestBytes = readAsset(context.getAssets(), "reuse/manifest.json", 1024 * 1024);
            JSONObject manifest = new JSONObject(new String(manifestBytes, StandardCharsets.UTF_8));
            if (manifest.getInt("sample_rate") != sampleRate || manifest.getInt("look_ahead_frames") != 0) {
                throw new IOException("Model manifest and enabled stream configuration disagree");
            }
            String expectedHash = manifest.getString("model_sha256");
            if (!expectedHash.matches("[a-fA-F0-9]{64}")) throw new IOException("Invalid model SHA-256");
            File model = new File(directory, "model.onnx");
            if (!model.isFile() || !expectedHash.equalsIgnoreCase(sha256(model))) {
                copyVerifiedAsset(context.getAssets(), "reuse/model.onnx", model, expectedHash);
            }
            // Only publish metadata after the matching model is fully written and verified.
            writeAtomically(new File(directory, "manifest.json"), manifestBytes);
            report.put("model_sha256", expectedHash.toLowerCase(java.util.Locale.ROOT));
            report.put("model_dir", directory.getAbsolutePath());
            String nativeDirectory = context.getApplicationInfo().nativeLibraryDir;
            report.put("native_lib_dir", nativeDirectory);
            System.loadLibrary("lc3");
            nativeLoaded = true;
            configured = nativeConfigure(directory.getAbsolutePath(), nativeDirectory, sampleRate, threads);
            if (configured) {
                activeModelDirectory = directory;
                activeConfigBytes = configBytes;
                activeRuntimeProfile = runtimeProfile;
            }
            report.put("configured", configured).put("stage", configured ? "configured" : "native_rejected");
            report.put("diagnostics", diagnostics());
            record(directory, report);
            return configured;
        } catch (Exception | LinkageError error) {
            configured = false;
            try {
                report.put("configured", false).put("stage", "failed")
                        .put("error", error.getClass().getSimpleName() + ": " + error.getMessage());
                record(directory, report);
            } catch (Exception recordingError) {
                status = "{\"configured\":false,\"stage\":\"failed_to_record_initialization\"}";
                Log.e(TAG, "Cannot record initialization result", recordingError);
            }
            Log.e(TAG, "RE-USE initialization failed", error);
            return false;
        }
    }

    public static synchronized int sampleRate() { return sampleRate; }
    public static synchronized int threads() { return threads; }
    public static synchronized String runtimeProfile() { return activeRuntimeProfile; }
    public static synchronized boolean probeReconfigured() { return probeReconfigured; }

    /** Debug probe only: bounded configuration selection, never an intent-supplied path. */
    public static synchronized boolean configureForProbe(Context context, int rate, int requestedThreads)
            throws IOException, JSONException {
        return configureForProbe(context, rate, requestedThreads, null);
    }

    /** Null retains the current profile; an explicit allowlisted name may recreate the session. */
    public static synchronized boolean configureForProbe(Context context, int rate, int requestedThreads,
            String requestedRuntimeProfile) throws IOException, JSONException {
        if ((rate != 8000 && rate != 16000) || requestedThreads < 1 || requestedThreads > 8) {
            throw new IOException("Probe supports modelRate 8000/16000 and threads 1..8");
        }
        String selectedProfile = requestedRuntimeProfile == null ? activeRuntimeProfile :
                validateRuntimeProfile(requestedRuntimeProfile);
        probeReconfigured = false;
        if (configured && sampleRate == rate && threads == requestedThreads &&
                activeRuntimeProfile.equals(selectedProfile)) return true;
        File directory = activeModelDirectory;
        if (directory == null || sampleRate != rate) {
            directory = new File(new File(context.getFilesDir(), "reuse-models"), Integer.toString(rate));
            if (!directory.isDirectory() && !directory.mkdirs()) throw new IOException("Cannot create probe model directory");
            String asset = "reuse-models/" + rate + "/";
            byte[] metadata = readAsset(context.getAssets(), asset + "manifest.json", 1024 * 1024);
            JSONObject manifest = new JSONObject(new String(metadata, StandardCharsets.UTF_8));
            if (manifest.getInt("sample_rate") != rate || manifest.getInt("look_ahead_frames") != 0) {
                throw new IOException("Probe model manifest does not match selected rate");
            }
            String expected = manifest.getString("model_sha256");
            if (!expected.matches("[a-fA-F0-9]{64}")) throw new IOException("Invalid probe model SHA-256");
            File graph = new File(directory, "model.onnx");
            if (!graph.isFile() || !expected.equalsIgnoreCase(sha256(graph))) {
                copyVerifiedAsset(context.getAssets(), asset + "model.onnx", graph, expected);
            }
            writeAtomically(new File(directory, "manifest.json"), metadata);
        }
        // Alternate-rate models live in a different private directory. Carry
        // only the same validated configuration bytes there; sample rate and
        // thread selection remain the explicit nativeConfigure arguments.
        byte[] configBytes = activeConfigBytes != null ? activeConfigBytes :
                readAsset(context.getAssets(), "reuse/config.json", 16384);
        JSONObject config = new JSONObject(new String(configBytes, StandardCharsets.UTF_8));
        String storedProfile = runtimeProfile(config);
        if (!storedProfile.equals(selectedProfile)) {
            config.put("runtime_profile", selectedProfile);
            configBytes = config.toString().getBytes(StandardCharsets.UTF_8);
        }
        writeAtomically(new File(directory, "config.json"), configBytes);
        if (!nativeLoaded) {
            System.loadLibrary("lc3");
            nativeLoaded = true;
        }
        configured = nativeConfigure(directory.getAbsolutePath(), context.getApplicationInfo().nativeLibraryDir,
                rate, requestedThreads);
        probeReconfigured = true;
        if (configured) {
            sampleRate = rate;
            threads = requestedThreads;
            activeModelDirectory = directory;
            activeConfigBytes = configBytes;
            activeRuntimeProfile = selectedProfile;
        }
        return configured;
    }
    public static String status() { return status; }

    private static String runtimeProfile(JSONObject config) throws IOException, JSONException {
        if (!config.has("runtime_profile")) return "default";
        Object value = config.get("runtime_profile");
        if (!(value instanceof String)) throw new IOException("runtime_profile must be a string");
        return validateRuntimeProfile((String) value);
    }

    private static String validateRuntimeProfile(String profile) throws IOException {
        if (!profile.equals("default") && !profile.equals("no_spin") &&
                !profile.equals("bounded_spin") && !profile.equals("dynamic4")) {
            throw new IOException("Unsupported runtime_profile");
        }
        return profile;
    }

    static File modelDirectory(Context context) { return new File(context.getFilesDir(), "reuse"); }

    static synchronized JSONObject diagnostics() throws JSONException {
        if (!nativeLoaded) return new JSONObject().put("native_loaded", false);
        try {
            String raw = nativeGetDiagnostics();
            try { return new JSONObject(raw); }
            catch (JSONException invalid) { return new JSONObject().put("invalid_native_diagnostics", raw); }
        } catch (LinkageError error) {
            return new JSONObject().put("native_diagnostics_error", error.toString());
        }
    }

    private static void record(File directory, JSONObject report) throws IOException, JSONException {
        report.put("schema_version", 1).put("recorded_at_epoch_ms", System.currentTimeMillis());
        status = report.toString();
        writeAtomically(new File(directory, "init-result.json"), status.getBytes(StandardCharsets.UTF_8));
        Log.i(TAG, status);
    }

    private static byte[] readAsset(AssetManager assets, String name, int maximum) throws IOException {
        try (InputStream input = assets.open(name);
                ByteArrayOutputStream output = new ByteArrayOutputStream()) {
            byte[] buffer = new byte[4096];
            int read;
            while ((read = input.read(buffer)) != -1) {
                if (output.size() > maximum - read) throw new IOException("Asset exceeds size limit: " + name);
                output.write(buffer, 0, read);
            }
            return output.toByteArray();
        }
    }

    private static void copyVerifiedAsset(AssetManager assets, String name, File target, String expectedHash)
            throws IOException {
        File temporary = File.createTempFile("reuse-model-", ".tmp", target.getParentFile());
        try {
            try (InputStream input = assets.open(name);
                    FileOutputStream output = new FileOutputStream(temporary)) {
                byte[] buffer = new byte[65536];
                int read;
                while ((read = input.read(buffer)) != -1) output.write(buffer, 0, read);
                output.getFD().sync();
            }
            if (!expectedHash.equalsIgnoreCase(sha256(temporary))) throw new IOException("Packaged model SHA-256 mismatch");
            moveAtomically(temporary, target);
        } finally { if (temporary.exists() && !temporary.delete()) temporary.deleteOnExit(); }
    }

    static void writeAtomically(File target, byte[] bytes) throws IOException {
        if (target.isFile() && Arrays.equals(Files.readAllBytes(target.toPath()), bytes)) return;
        File temporary = File.createTempFile("reuse-meta-", ".tmp", target.getParentFile());
        try {
            try (FileOutputStream output = new FileOutputStream(temporary)) {
                output.write(bytes);
                output.getFD().sync();
            }
            moveAtomically(temporary, target);
        } finally { if (temporary.exists() && !temporary.delete()) temporary.deleteOnExit(); }
    }

    private static void moveAtomically(File source, File target) throws IOException {
        try {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.ATOMIC_MOVE,
                    StandardCopyOption.REPLACE_EXISTING);
        } catch (AtomicMoveNotSupportedException unsupported) {
            Files.move(source.toPath(), target.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
    }

    private static String sha256(File file) throws IOException {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            try (InputStream input = new FileInputStream(file)) {
                byte[] buffer = new byte[65536];
                int read;
                while ((read = input.read(buffer)) != -1) digest.update(buffer, 0, read);
            }
            StringBuilder result = new StringBuilder(64);
            for (byte value : digest.digest()) result.append(String.format("%02x", value & 255));
            return result.toString();
        } catch (NoSuchAlgorithmException impossible) { throw new AssertionError(impossible); }
    }
}
