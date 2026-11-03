package io.github.reuse.qnn;

import android.content.Context;
import android.os.Build;
import android.system.Os;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtEpDevice;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtHardwareDevice;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.OrtLoggingLevel;
import java.security.MessageDigest;
import io.github.reuse.SessionOptionsConfigurer;
import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Optional Qualcomm HTP session backend. Loading or selecting a device is not proof of execution.
 * One instance configures one session. Close the consuming session/enhancer before this backend:
 * <pre>{@code
 * try (QnnHtpBackend backend = new QnnHtpBackend(context, privateProfilePrefix, true)) {
 *     try (ChunkedReuseEnhancer enhancer = ChunkedReuseEnhancer.create(
 *             context, model, manifest, cpuConfig, backend)) {
 *         // Process input and independently validate its output.
 *         Map<String, Object> evidence = backend.verifyProfile(enhancer.endProfiling(), calls, false);
 *     }
 * }
 * }</pre>
 * The application must extract its native libraries and preserve vendor binaries and notices.
 */
public final class QnnHtpBackend implements SessionOptionsConfigurer, AutoCloseable {
    private static final String QNN_EP = "QNNExecutionProvider";
    private static final long MAX_PROFILE_BYTES = 64L * 1024 * 1024;
    private static final Object REGISTRATION_LOCK = new Object();
    private static final RegistrationPolicy REGISTRATIONS = new RegistrationPolicy();
    private static SharedRegistration shared;

    private final Path nativeDirectory, pluginLibrary, backendLibrary, profilingPrefix, diagnosticDirectory;
    private final boolean gpu;
    private final String performanceMode;
    private final boolean allowCpuFallback;
    private boolean attempted, acquired, configured, closed;
    private OrtEnvironment environment;
    private volatile Map<String, Object> diagnostics;

    private static final class SharedRegistration {
        final OrtEnvironment environment;
        final String pluginPath, originalAdspPath;
        final List<OrtEpDevice> qnnDevices;
        final List<Map<String, Object>> availableDevices;
        SharedRegistration(OrtEnvironment environment, String pluginPath, String originalAdspPath,
                           List<OrtEpDevice> qnnDevices, List<Map<String, Object>> availableDevices) {
            this.environment = environment; this.pluginPath = pluginPath;
            this.originalAdspPath = originalAdspPath;
            this.qnnDevices = Collections.unmodifiableList(new ArrayList<>(qnnDevices));
            this.availableDevices = Collections.unmodifiableList(new ArrayList<>(availableDevices));
        }
    }

    /**
     * Validates packaged files and an existing writable app-private profiling parent directory.
     * Native loading and device selection occur later in {@link #configure}; neither proves execution.
     */
    public QnnHtpBackend(Context context, Path profilingPrefix, boolean allowCpuFallback) throws IOException {
        this(context, profilingPrefix, allowCpuFallback, "balanced");
    }

    /** Explicit performance setting for bounded comparisons; existing callers retain balanced mode. */
    public QnnHtpBackend(Context context, Path profilingPrefix, boolean allowCpuFallback,
                         String performanceMode) throws IOException {
        this(context, profilingPrefix, allowCpuFallback, performanceMode, false);
    }

    static QnnHtpBackend gpu(Context context, Path profilingPrefix, boolean allowCpuFallback) throws IOException {
        return new QnnHtpBackend(context, profilingPrefix, allowCpuFallback, "balanced", true);
    }

    static QnnHtpBackend gpu(Context context, Path profilingPrefix, boolean allowCpuFallback,
                             Path diagnosticDirectory) throws IOException {
        return new QnnHtpBackend(context, profilingPrefix, allowCpuFallback, "balanced", true, diagnosticDirectory);
    }

    private QnnHtpBackend(Context context, Path profilingPrefix, boolean allowCpuFallback,
                          String performanceMode, boolean gpu) throws IOException {
        this(context, profilingPrefix, allowCpuFallback, performanceMode, gpu, null);
    }

    private QnnHtpBackend(Context context, Path profilingPrefix, boolean allowCpuFallback,
                          String performanceMode, boolean gpu, Path diagnosticDirectory) throws IOException {
        validatePerformanceMode(performanceMode);
        this.gpu = gpu;
        this.performanceMode = performanceMode;
        if (context == null || profilingPrefix == null)
            throw new IllegalArgumentException("Context and private profiling prefix are required");
        if (Build.VERSION.SDK_INT < 29) throw new IOException("QNN HTP requires Android 10 / API 29 or newer");
        Context application = context.getApplicationContext();
        nativeDirectory = new File(application.getApplicationInfo().nativeLibraryDir).toPath().toRealPath();
        pluginLibrary = regularLibrary(nativeDirectory, "libonnxruntime_providers_qnn.so");
        backendLibrary = regularLibrary(nativeDirectory, gpu ? "libQnnGpu.so" : "libQnnHtp.so");
        this.profilingPrefix = privateProfilingPrefix(application, profilingPrefix);
        this.diagnosticDirectory = diagnosticDirectory == null ? null
                : privateProfilingPrefix(application, diagnosticDirectory.resolve("qnn-diagnostic")).getParent();
        if (this.diagnosticDirectory != null && !gpu)
            throw new IllegalArgumentException("Graph diagnostic options are restricted to the explicit GPU probe");
        this.allowCpuFallback = allowCpuFallback;
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("requested_backend", gpu ? "QNN_GPU" : "QNN_HTP");
        values.put("backend_library", backendLibrary.toString());
        values.put("backend_library_sha256", librarySha256(backendLibrary));
        values.put("gpu_backend_requested", gpu);
        values.put("gpu_device_selected", false);
        values.put("gpu_hardware_execution_verified", false);
        values.put("qnn_plugin_package_version", BuildConfig.QNN_EP_VERSION);
        values.put("qnn_runtime_package_version", BuildConfig.QNN_RUNTIME_VERSION);
        values.put("packaged_version_scope", "Pinned Maven versions; not a query of the loaded QNN backend API");
        values.put("native_library_directory", nativeDirectory.toString());
        values.put("profiling_prefix", this.profilingPrefix.toString());
        if (this.diagnosticDirectory != null) values.put("private_diagnostic_directory", this.diagnosticDirectory.toString());
        values.put("cpu_fallback_allowed", allowCpuFallback);
        values.put("npu_device_selected", false);
        values.put("qnn_execution_verified", false);
        values.put("execution_verification_scope", "Packaged libraries validated only; device selection and execution are unverified");
        diagnostics = Collections.unmodifiableMap(values);
    }

    /** Configures one session. Keep this backend open until that session has closed. */
    @Override public synchronized void configure(OrtEnvironment environment, OrtSession.SessionOptions options)
            throws IOException, OrtException {
        if (closed) throw new IllegalStateException("QNN backend is closed");
        if (attempted) throw new IllegalStateException("A QNN backend instance configures only one session");
        if (environment == null || options == null) throw new IllegalArgumentException("Environment and options are required");
        attempted = true;
        this.environment = environment;
        try {
            SharedRegistration registration = acquireShared(environment, nativeDirectory, pluginLibrary);
            acquired = true;
            List<String> deviceTypes = new ArrayList<>();
            for (OrtEpDevice candidate : registration.qnnDevices) deviceTypes.add(candidate.getDevice().getType().name());
            int selectedIndex = preferredDeviceIndex(deviceTypes, gpu);
            if (selectedIndex < 0) throw new IOException(gpu ? "QNN exposed no registration device for the direct GPU probe" : "QNN exposed no NPU device");
            OrtEpDevice selected = registration.qnnDevices.get(selectedIndex);
            boolean gpuDevice = selected.getDevice().getType() == OrtHardwareDevice.OrtHardwareDeviceType.GPU;
            Map<String, String> providerOptions = providerOptions(backendLibrary, gpu, performanceMode, diagnosticDirectory);
            options.addConfigEntry("session.disable_cpu_ep_fallback", allowCpuFallback ? "0" : "1");
            options.enableProfiling(profilingPrefix.toString());
            if (gpu) options.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_INFO);
            options.addExecutionProvider(Collections.singletonList(selected), providerOptions);
            Map<String, Object> values = new LinkedHashMap<>(diagnostics);
            values.put("ort_version", environment.getVersion());
            values.put("selected_ep_device", describeDevice(selected));
            values.put("registration_token_device_type", selected.getDevice().getType().name());
            values.put("gpu_device_selected", gpu && gpuDevice);
            values.put("gpu_registration_token_workaround", gpu && !gpuDevice);
            values.put("ort_ep_devices", registration.availableDevices);
            values.put("qnn_provider_options", Collections.unmodifiableMap(new LinkedHashMap<>(providerOptions)));
            values.put("npu_device_selected", !gpu);
            values.put("execution_verification_scope", gpu
                    ? "Direct QNN GPU backend requested. The discovery device is a registration token when it is not GPU; QNN events alone do not establish the active hardware backend. Numerical output and loaded-backend identity require separate verification."
                    : "QNN NPU device selected and HTP configured; execution and numerical correctness remain unverified");
            diagnostics = Collections.unmodifiableMap(values);
            configured = true;
        } catch (IOException | OrtException | RuntimeException error) {
            releaseAfterFailedConfiguration(error);
            throw error;
        } catch (LinkageError error) {
            IOException failure = new IOException("Could not load the configured QNN runtime", error);
            releaseAfterFailedConfiguration(failure);
            throw failure;
        }
    }

    /** Immutable configuration/evidence snapshot. Device selection alone leaves execution unverified. */
    public Map<String, Object> diagnostics() { return diagnostics; }

    /**
     * Verifies an ended ORT profile belonging to this backend's private prefix.
     * Expected calls include every execution captured by profiling, including warm-up and flush.
     * A single-partition request requires exactly one QNN kernel name and one event per call.
     * CPU events are allowed only when the constructor explicitly enabled CPU fallback;
     * unknown or missing provider attribution always fails verification.
     * Output correctness remains the caller's responsibility.
     */
    public synchronized Map<String, Object> verifyProfile(Path profile, int expectedCalls,
                                                         boolean requireSinglePartition) throws IOException {
        if (!configured) throw new IllegalStateException("QNN backend has not configured a session");
        if (profile == null) throw new IllegalArgumentException("Ended profile path is required");
        Path actual = profile.toRealPath();
        if (!profilingPrefix.getParent().equals(actual.getParent())
                || !actual.getFileName().toString().startsWith(profilingPrefix.getFileName().toString() + "_"))
            throw new IOException("Profile does not belong to this backend's private profiling prefix");
        if (!Files.isRegularFile(actual) || Files.size(actual) > MAX_PROFILE_BYTES)
            throw new IOException("ORT profile is missing or exceeds the verification size limit");
        Map<String, Object> evidence = QnnProfileVerifier.verify(
                new String(Files.readAllBytes(actual), StandardCharsets.UTF_8),
                expectedCalls, allowCpuFallback, requireSinglePartition);
        Map<String, Object> values = new LinkedHashMap<>(diagnostics);
        values.putAll(evidence);
        if (gpu) {
            values.put("gpu_hardware_execution_verified", false);
            values.put("execution_verification_scope", "QNN provider kernel events observed for a session configured with the direct GPU backend. Events alone do not identify GPU versus HTP; require independent loaded-backend identity and numerical checks before claiming GPU execution.");
        }
        values.put("ort_profile_file", actual.toString());
        diagnostics = Collections.unmodifiableMap(values);
        return diagnostics;
    }

    /** Releases this instance's registration lease. Close every consuming enhancer/session first. */
    @Override public synchronized void close() throws IOException {
        if (closed) return;
        closed = true;
        if (acquired) {
            acquired = false;
            releaseShared(environment, pluginLibrary.toString());
        }
    }

    private void releaseAfterFailedConfiguration(Throwable failure) {
        if (!acquired) return;
        acquired = false;
        try { releaseShared(environment, pluginLibrary.toString()); }
        catch (IOException cleanup) { failure.addSuppressed(cleanup); }
    }

    private static SharedRegistration acquireShared(OrtEnvironment environment, Path nativeDirectory,
                                                     Path pluginLibrary) throws IOException {
        synchronized (REGISTRATION_LOCK) {
            String pluginPath = pluginLibrary.toString();
            boolean first = REGISTRATIONS.acquire(environment, pluginPath);
            if (!first) return shared;
            boolean changedAdsp = false, registered = false;
            String originalAdsp = null;
            String stage = "setting ADSP_LIBRARY_PATH";
            try {
                originalAdsp = Os.getenv("ADSP_LIBRARY_PATH");
                Os.setenv("ADSP_LIBRARY_PATH", nativeDirectory.toString(), true);
                changedAdsp = true;
                stage = "registering QNNExecutionProvider";
                environment.registerExecutionProviderLibrary(QNN_EP, pluginPath);
                registered = true;
                stage = "enumerating QNN registration devices";
                List<OrtEpDevice> qnnDevices = new ArrayList<>();
                List<Map<String, Object>> descriptions = new ArrayList<>();
                for (OrtEpDevice device : environment.getEpDevices()) {
                    descriptions.add(describeDevice(device));
                    if (QNN_EP.equals(device.getEpName())) qnnDevices.add(device);
                }
                if (qnnDevices.isEmpty()) throw new IOException("QNN exposed no registration devices");
                shared = new SharedRegistration(environment, pluginPath, originalAdsp, qnnDevices, descriptions);
                return shared;
            } catch (Exception | LinkageError error) {
                IOException failure = new IOException("QNN initialization failed while " + stage, error);
                REGISTRATIONS.release(environment, pluginPath);
                Throwable cleanup = cleanupRegistration(environment, registered, changedAdsp, originalAdsp);
                if (cleanup != null) {
                    failure.addSuppressed(cleanup);
                    REGISTRATIONS.poison(cleanup);
                }
                shared = null;
                throw failure;
            }
        }
    }

    private static void releaseShared(OrtEnvironment environment, String pluginPath) throws IOException {
        synchronized (REGISTRATION_LOCK) {
            if (!REGISTRATIONS.release(environment, pluginPath)) return;
            SharedRegistration previous = shared;
            shared = null;
            Throwable cleanup = cleanupRegistration(previous.environment, true, true, previous.originalAdspPath);
            if (cleanup != null) {
                REGISTRATIONS.poison(cleanup);
                throw new IOException("Could not release the final QNN registration; close consuming sessions first", cleanup);
            }
        }
    }

    private static Throwable cleanupRegistration(OrtEnvironment environment, boolean registered,
                                                  boolean changedAdsp, String originalAdsp) {
        Throwable failure = null;
        if (registered) {
            try { environment.unregisterExecutionProviderLibrary(QNN_EP); }
            catch (Exception | LinkageError error) { failure = error; }
        }
        if (changedAdsp) {
            try {
                if (originalAdsp == null) Os.unsetenv("ADSP_LIBRARY_PATH");
                else Os.setenv("ADSP_LIBRARY_PATH", originalAdsp, true);
            } catch (Exception error) {
                if (failure == null) failure = error;
                else failure.addSuppressed(error);
            }
        }
        return failure;
    }


    static void validatePerformanceMode(String mode) {
        if (!"balanced".equals(mode) && !"burst".equals(mode))
            throw new IllegalArgumentException("HTP performance mode must be balanced or burst");
    }

    static Map<String, String> providerOptions(Path library, boolean gpu, String mode) {
        return providerOptions(library, gpu, mode, null);
    }

    static Map<String, String> providerOptions(Path library, boolean gpu, String mode, Path diagnosticDirectory) {
        validatePerformanceMode(mode);
        if (diagnosticDirectory != null && !gpu)
            throw new IllegalArgumentException("Graph diagnostic options require the GPU backend");
        Map<String, String> values = new LinkedHashMap<>();
        values.put("backend_path", library.toString());
        if (!gpu) {
            values.put("htp_performance_mode", mode);
            values.put("offload_graph_io_quantization", "0");
            values.put("enable_htp_fp16_precision", "0");
        }
        if (diagnosticDirectory != null) {
            values.put("dump_qnn_ep_input_graph", "1");
            values.put("dump_qnn_ep_input_graph_dir", diagnosticDirectory.toString());
            values.put("dump_json_qnn_graph", "1");
            values.put("json_qnn_graph_dir", diagnosticDirectory.toString());
            values.put("enable_framework_op_trace", "1");
            values.put("framework_op_trace_dir", diagnosticDirectory.toString());
        }
        return Collections.unmodifiableMap(values);
    }

    static int preferredDeviceIndex(List<String> types, boolean gpu) {
        String preferred = gpu ? "GPU" : "NPU";
        for (int i = 0; i < types.size(); i++) if (preferred.equals(types.get(i))) return i;
        // Pinned QNN 2.7 permits an explicit GPU backend override when Android discovery
        // supplies only its synthesized NPU token. The token is never relabeled as GPU.
        return gpu && !types.isEmpty() ? 0 : -1;
    }

    private static String librarySha256(Path file) throws IOException {
        try {
            StringBuilder value = new StringBuilder();
            for (byte b : MessageDigest.getInstance("SHA-256").digest(Files.readAllBytes(file)))
                value.append(String.format(java.util.Locale.ROOT, "%02x", b & 255));
            return value.toString();
        } catch (java.security.NoSuchAlgorithmException error) {
            throw new IOException("SHA-256 is unavailable", error);
        }
    }

    private static Path regularLibrary(Path directory, String filename) throws IOException {
        Path library = directory.resolve(filename).toRealPath();
        if (!Files.isRegularFile(library) || !directory.equals(library.getParent()))
            throw new IOException("Expected packaged native library " + filename);
        return library;
    }

    private static Path privateProfilingPrefix(Context context, Path prefix) throws IOException {
        Path absolute = prefix.toAbsolutePath().normalize();
        if (absolute.getParent() == null || absolute.getFileName() == null)
            throw new IOException("Profiling prefix requires a filename and an existing private parent directory");
        Path parent = absolute.getParent().toRealPath();
        Path files = context.getFilesDir().toPath().toRealPath();
        Path cache = context.getCacheDir().toPath().toRealPath();
        if ((!parent.startsWith(files) && !parent.startsWith(cache))
                || !Files.isDirectory(parent) || !Files.isWritable(parent))
            throw new IOException("Profiling parent must be an existing writable app-private files/cache directory");
        return parent.resolve(absolute.getFileName().toString());
    }

    private static Map<String, Object> describeDevice(OrtEpDevice device) {
        Map<String, Object> values = new LinkedHashMap<>();
        values.put("ep_name", device.getEpName()); values.put("ep_vendor", device.getEpVendor());
        values.put("ep_metadata", immutableCopy(device.getEpMetadata()));
        values.put("ep_options", immutableCopy(device.getEpOptions()));
        values.put("device_type", device.getDevice().getType().name());
        values.put("device_id", device.getDevice().getDeviceId());
        values.put("device_vendor", device.getDevice().getVendor());
        values.put("device_vendor_id", device.getDevice().getVendorId());
        values.put("device_metadata", immutableCopy(device.getDevice().getMetadata()));
        return Collections.unmodifiableMap(values);
    }

    private static Map<String, String> immutableCopy(Map<String, String> original) {
        return Collections.unmodifiableMap(new LinkedHashMap<>(original));
    }
}
