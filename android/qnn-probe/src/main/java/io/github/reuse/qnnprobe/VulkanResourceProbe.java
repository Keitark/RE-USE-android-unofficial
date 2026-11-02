package io.github.reuse.qnnprobe;

import org.json.JSONObject;

/** Independent Vulkan arithmetic smoke test; never substitutes for a RE-USE GPU benchmark. */
final class VulkanResourceProbe {
    private VulkanResourceProbe() { }

    static JSONObject run() {
        try {
            System.loadLibrary("reuse_resource_inventory");
            return new JSONObject(runNative());
        } catch (Exception | LinkageError error) {
            JSONObject result = new JSONObject();
            try {
                result.put("source", "vulkan_compute_smoke");
                result.put("status", "ERROR");
                result.put("failure_stage", "java.load_or_decode");
                result.put("error", error.toString());
                result.put("gpu_backend_execution_verified", false);
                result.put("model_acceleration_verified", false);
                result.put("microphone_started", false);
            } catch (org.json.JSONException impossible) { throw new IllegalStateException(impossible); }
            return result;
        }
    }

    private static native String runNative();
}
