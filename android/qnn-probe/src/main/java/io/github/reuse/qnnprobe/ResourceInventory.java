package io.github.reuse.qnnprobe;

import android.app.ActivityManager;
import android.content.Context;
import android.content.pm.FeatureInfo;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.PowerManager;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.TreeSet;
import org.json.JSONArray;
import org.json.JSONObject;

/** Read-only device capabilities; availability is separate from actual model execution. */
final class ResourceInventory {
    private ResourceInventory() { }
    private static native String openClInventory();

    static JSONObject collect(Context context) {
        JSONObject out = new JSONObject();
        put(out, "scope", "Read-only public compute capabilities; no microphone, kernels, model execution or hardware utilization measurement");
        put(out, "manufacturer", Build.MANUFACTURER);
        put(out, "model", Build.MODEL);
        put(out, "android_api", Build.VERSION.SDK_INT);
        put(out, "supported_abis", new JSONArray(java.util.Arrays.asList(Build.SUPPORTED_ABIS)));
        put(out, "available_processors", Runtime.getRuntime().availableProcessors());
        put(out, "gpu_execution_verified", false);
        if (Build.VERSION.SDK_INT >= 31) { put(out, "soc_manufacturer", Build.SOC_MANUFACTURER); put(out, "soc_model", Build.SOC_MODEL); }
        JSONArray cores = new JSONArray();
        // File names and keys are fixed. Never collect /proc/cpuinfo Serial or identifiers.
        for (int i = 0; i < 32; i++) {
            Path cpu = java.nio.file.Paths.get("/sys/devices/system/cpu/cpu" + i);
            if (!Files.isDirectory(cpu)) continue;
            JSONObject core = new JSONObject(); put(core, "cpu", i);
            readInteger(core, "max_frequency_khz", cpu.resolve("cpufreq/cpuinfo_max_freq"));
            readInteger(core, "current_frequency_khz", cpu.resolve("cpufreq/scaling_cur_freq"));
            readInteger(core, "online", cpu.resolve("online")); cores.put(core);
        }
        put(out, "cpu_cores", cores);
        try {
            JSONArray libraries = new JSONArray();
            FileNames.addPackagedLibraries(context, libraries);
            put(out, "packaged_native_libraries", libraries);
            put(out, "packaging_scope", "Packaged backends are candidates only, not execution or compatibility evidence");
        } catch (Exception error) { put(out, "packaged_libraries_error", error.toString()); }
        try {
            Path info = java.nio.file.Paths.get("/proc/cpuinfo");
            if (Files.size(info) <= 1024 * 1024) {
                TreeSet<String> featureLines = new TreeSet<>();
                for (String line : Files.readAllLines(info, StandardCharsets.US_ASCII))
                    if (line.startsWith("Features") && line.indexOf(':') >= 0)
                        featureLines.add(line.substring(line.indexOf(':') + 1).trim());
                put(out, "cpu_feature_sets", new JSONArray(featureLines));
            }
        } catch (Exception error) { put(out, "cpu_features_error", error.toString()); }
        try {
            ActivityManager manager = context.getSystemService(ActivityManager.class);
            if (manager != null) put(out, "declared_gles_version", manager.getDeviceConfigurationInfo().getGlEsVersion());
            JSONArray vulkan = new JSONArray();
            for (FeatureInfo feature : context.getPackageManager().getSystemAvailableFeatures())
                if (PackageManager.FEATURE_VULKAN_HARDWARE_VERSION.equals(feature.name)
                        || PackageManager.FEATURE_VULKAN_HARDWARE_LEVEL.equals(feature.name)
                        || "android.hardware.vulkan.compute".equals(feature.name)) {
                    JSONObject f = new JSONObject(); put(f, "feature", feature.name); put(f, "version", feature.version); vulkan.put(f);
                }
            put(out, "declared_vulkan_features", vulkan);
        } catch (Exception error) { put(out, "graphics_features_error", error.toString()); }
        try {
            System.loadLibrary("reuse_resource_inventory");
            put(out, "opencl", new JSONObject(openClInventory()));
        } catch (Exception | LinkageError error) { put(out, "opencl_error", error.toString()); }
        put(out, "power", power(context));
        return out;
    }

    static JSONObject power(Context context) {
        JSONObject out = new JSONObject();
        PowerManager power = context.getSystemService(PowerManager.class);
        if (power != null) {
            put(out, "interactive", power.isInteractive()); put(out, "idle", power.isDeviceIdleMode());
            put(out, "power_save", power.isPowerSaveMode()); put(out, "thermal_status", power.getCurrentThermalStatus());
        }
        return out;
    }

    private static void readInteger(JSONObject object, String key, Path file) {
        try {
            if (Files.isRegularFile(file) && Files.size(file) <= 4096) {
                String text = new String(Files.readAllBytes(file), StandardCharsets.US_ASCII).trim();
                put(object, key, Long.parseLong(text));
            }
        } catch (Exception ignored) { /* Optional sysfs capability may be hidden by Android. */ }
    }
    private static final class FileNames {
        static void addPackagedLibraries(Context context, JSONArray out) {
            String[] names = new java.io.File(context.getApplicationInfo().nativeLibraryDir).list();
            if (names == null) return;
            java.util.Arrays.sort(names);
            for (String name : names) if (name.endsWith(".so") && name.matches("[A-Za-z0-9_.-]+")) out.put(name);
        }
    }
    private static void put(JSONObject object, String key, Object value) {
        try { object.put(key, value); } catch (org.json.JSONException error) { throw new IllegalArgumentException(error); }
    }
}
