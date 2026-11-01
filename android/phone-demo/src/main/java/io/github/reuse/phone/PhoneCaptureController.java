package io.github.reuse.phone;

import android.Manifest;
import android.content.Context;
import android.content.pm.PackageManager;
import android.media.AudioDeviceInfo;
import android.media.AudioFormat;
import android.media.AudioManager;
import android.media.AudioRecord;
import android.media.AudioRecordingConfiguration;
import android.media.MediaRecorder;
import android.os.Build;
import android.os.Handler;
import android.os.Looper;
import android.os.Process;
import android.os.SystemClock;
import io.github.reuse.live.EnhancementSession;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.json.JSONObject;

/** Visible-activity capture; no network, Bluetooth, global routing changes or speaker monitoring. */
public final class PhoneCaptureController implements AutoCloseable {
    public interface Listener {
        void onState(StateSnapshot state);
        void onFinished(File directory, Map<String, Object> statistics);
        void onError(String message);
    }

    public static final class StateSnapshot {
        public final String phase, message, route;
        public final boolean active, processingBehind;
        public final long capturedSamples, modelCalls, deadlineMisses;
        public final int threads;
        public final double elapsedSeconds, level, estimatedProcessingRtf, currentQueueMs;
        private StateSnapshot(String phase, String message, boolean active, Run run) {
            this.phase = phase; this.message = message; this.active = active;
            capturedSamples = run.samples; threads = run.threads; route = run.route;
            elapsedSeconds = capturedSamples / 16000.0; level = run.peak;
            Map<String, Object> stats = run.session == null ? new LinkedHashMap<>() : run.session.statistics();
            modelCalls = number(stats, "model_calls").longValue();
            deadlineMisses = number(stats, "deadline_misses").longValue();
            // Approximate live model compute, excludes FIR and the in-flight call.
            estimatedProcessingRtf = modelCalls == 0 ? 0 : number(stats, "process_total_ms").doubleValue() / (modelCalls * 100.0);
            currentQueueMs = number(stats, "queued_input_samples").doubleValue() / 16.0;
            processingBehind = (modelCalls >= 10 && estimatedProcessingRtf >= 1) || currentQueueMs >= 200;
        }
        public double seconds() { return elapsedSeconds; }
    }

    private static final int RATE = 16000, READ_SAMPLES = 800, MAX_SAMPLES = RATE * 60;
    private static final int MAX_OUTPUT_TRACE_EVENTS = 1300;
    private static final String MODEL_SHA = "f950dcfb9321191b2bcb3ab9afa3e100c40e5c64dd1de1d30849be4c1aacce32";
    private final Context context;
    private final Listener listener;
    private final Handler main = new Handler(Looper.getMainLooper());
    private final Object lock = new Object();
    private final ExecutorService preparation = Executors.newSingleThreadExecutor(r -> new Thread(r, "reuse-phone-prepare"));
    private volatile boolean closed;
    private Run active;
    private volatile File latestDirectory;

    private static final class Run {
        final int threads;
        final Path directory;
        final Object recorderLock = new Object();
        volatile EnhancementSession session;
        volatile AudioRecord recorder;
        volatile Thread captureThread;
        volatile boolean stopRequested;
        boolean terminalScheduled;
        volatile String stopReason = "not_stopped", captureError;
        volatile String route = "Waiting for microphone";
        volatile int routeType = -1, actualSource = -1, actualRate;
        volatile boolean preferredBuiltInAccepted, microphoneStarted, microphoneReleased, platformSilenced;
        volatile long samples, readCalls, zeroReads, beginNs, endNs;
        volatile long captureStartHostNs, firstInputHostNs;
        volatile double peak;
        // Written only by the enhancement worker; serialized only after its terminal callback.
        final long[] outputHostNs = new long[MAX_OUTPUT_TRACE_EVENTS];
        final long[] outputFirstSamples = new long[MAX_OUTPUT_TRACE_EVENTS];
        final int[] outputSampleCounts = new int[MAX_OUTPUT_TRACE_EVENTS];
        int outputTraceEvents, omittedOutputTraceEvents;
        long callbackOutputSamples;
        Runnable durationStop;
        Run(int threads, Path directory) { this.threads = threads; this.directory = directory; }
    }

    public PhoneCaptureController(Context context, Listener listener) {
        if (context == null || listener == null) throw new IllegalArgumentException("Context and listener are required");
        this.context = context.getApplicationContext(); this.listener = listener;
    }

    public boolean isActive() { synchronized (lock) { return active != null; } }
    public File latestDir() { return latestDirectory; }

    public void start(int threads) {
        if (threads != 2 && threads != 4) throw new IllegalArgumentException("Choose 2 or 4 CPU threads");
        final Run run;
        synchronized (lock) {
            if (closed) throw new IllegalStateException("Controller is closed");
            if (active != null) return;
            if (context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED) {
                postError("Microphone permission is required. Press Start and allow microphone access."); return;
            }
            String stamp = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.ROOT).format(new Date());
            run = new Run(threads, context.getFilesDir().toPath().resolve("captures")
                    .resolve(stamp + "-" + UUID.randomUUID().toString().substring(0, 8)));
            active = run;
        }
        publish(run, "PREPARING", "Preparing RE-USE. The microphone opens when the model is ready.", true);
        preparation.execute(() -> prepare(run));
    }

    public void stop() { stop("user_stop"); }

    /** The activity can supply "background" to explain lifecycle stops in the saved receipt. */
    public void stop(String reason) {
        Run run;
        synchronized (lock) { run = active; }
        if (run != null) requestStop(run, reason == null ? "user_stop" : reason);
    }

    private void requestStop(Run run, String reason) {
        EnhancementSession finish = null;
        synchronized (lock) {
            if (active != run) return;
            if (!run.stopRequested) { run.stopRequested = true; run.stopReason = reason; }
            if (run.captureThread == null) finish = run.session;
        }
        publish(run, "STOPPING", "Stopping microphone and saving both recordings…", true);
        stopRecorderAsync(run);
        if (finish != null) finish.finish();
    }

    private void prepare(Run run) {
        try {
            if (cancelBeforeEngine(run)) return;
            Path model = extractModel();
            if (cancelBeforeEngine(run)) return;
            Path custom = new File(context.getApplicationInfo().nativeLibraryDir, "libreuse_ort_ops.so").toPath();
            if (!Files.isRegularFile(custom)) throw new IOException("The RE-USE native library is unavailable for this device");
            Files.createDirectories(run.directory);
            EnhancementSession session = new EnhancementSession(model, run.directory, custom,
                    new EnhancementSession.Listener() {
                        @Override public void onReady() { startMicrophone(run); }
                        @Override public void onAudio(byte[] pcm) { traceOutputAvailability(run, pcm); }
                        @Override public void onFinished(Map<String, Object> stats) { scheduleTerminal(run, stats, null); }
                        @Override public void onError(String code, String message, Map<String, Object> stats) {
                            scheduleTerminal(run, stats, code + ": " + message);
                        }
                    }, () -> Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO), run.threads);
            synchronized (lock) {
                if (active != run || run.stopRequested || closed) session.close();
                else { run.session = session; session.start(); return; }
            }
            cancelBeforeEngine(run);
        } catch (Exception error) { failPreparation(run, error.toString()); }
    }

    private boolean cancelBeforeEngine(Run run) {
        synchronized (lock) {
            if (active != run) return true;
            if (!run.stopRequested && !closed) return false;
            active = null;
        }
        publish(run, "IDLE", "Stopped before the microphone opened.", false);
        shutdownIfClosed(); return true;
    }

    private void startMicrophone(Run run) {
        synchronized (lock) {
            if (active != run || closed || run.stopRequested) { run.session.finish(); return; }
            run.captureThread = new Thread(() -> capture(run), "reuse-phone-microphone");
            run.captureThread.start();
        }
    }

    private void capture(Run run) {
        AudioRecord recorder = null;
        try {
            Process.setThreadPriority(Process.THREAD_PRIORITY_AUDIO);
            if (run.stopRequested) return;
            recorder = createRecorder();
            AudioManager manager = context.getSystemService(AudioManager.class);
            if (manager != null) for (AudioDeviceInfo device : manager.getDevices(AudioManager.GET_DEVICES_INPUTS)) {
                if (device.getType() == AudioDeviceInfo.TYPE_BUILTIN_MIC) {
                    run.preferredBuiltInAccepted = recorder.setPreferredDevice(device); break;
                }
            }
            synchronized (run.recorderLock) {
                run.recorder = recorder;
                synchronized (lock) {
                    // Start and cancellation are ordered by the same lock: no delayed open after cancellation.
                    if (run.stopRequested || closed || active != run) return;
                    recorder.startRecording();
                    if (recorder.getRecordingState() != AudioRecord.RECORDSTATE_RECORDING)
                        throw new IOException("Android did not start microphone recording");
                    run.microphoneStarted = true; run.beginNs = SystemClock.elapsedRealtimeNanos();
                    run.captureStartHostNs = System.nanoTime();
                    run.actualSource = recorder.getAudioSource(); run.actualRate = recorder.getSampleRate();
                    if (run.actualRate != RATE) throw new IOException("Microphone format is not 16 kHz");
                }
            }
            updateRoute(run, recorder);
            run.durationStop = () -> requestStop(run, "duration_limit");
            main.postDelayed(run.durationStop, 60_000);
            publish(run, "RECORDING", "Recording phone microphone. Speak normally; stops automatically after 60 seconds.", true);
            short[] buffer = new short[READ_SAMPLES];
            long lastUpdate = 0;
            while (!run.stopRequested && run.samples < MAX_SAMPLES) {
                int read = recorder.read(buffer, 0, (int) Math.min(buffer.length, MAX_SAMPLES - run.samples), AudioRecord.READ_BLOCKING);
                run.readCalls++;
                if (read <= 0) {
                    if (run.stopRequested) break;
                    if (read < 0) throw new IOException("Microphone read failed (Android code " + read + ")");
                    if (++run.zeroReads > 32) throw new IOException("Microphone returned no audio");
                    continue;
                }
                // A read already in flight contributes its complete packet before the ordered stop barrier.
                byte[] bytes = new byte[read * 2]; int peak = 0;
                for (int i = 0; i < read; i++) {
                    int value = buffer[i]; peak = Math.max(peak, Math.abs(value));
                    bytes[i * 2] = (byte) value; bytes[i * 2 + 1] = (byte) (value >> 8);
                }
                if (run.firstInputHostNs == 0) run.firstInputHostNs = System.nanoTime();
                if (!run.session.accept16kS16LE(bytes, run.samples))
                    throw new IOException("RE-USE could not accept microphone audio; the recording was stopped");
                run.samples += read; run.peak = peak / 32768.0;
                long now = SystemClock.elapsedRealtimeNanos();
                if (now - lastUpdate >= 250_000_000L) {
                    updateRoute(run, recorder);
                    if (Build.VERSION.SDK_INT >= 29) {
                        AudioRecordingConfiguration config = recorder.getActiveRecordingConfiguration();
                        if (config != null && config.isClientSilenced()) {
                            run.platformSilenced = true;
                            throw new IOException("Android silenced this microphone capture. Check microphone access or another recording app.");
                        }
                    }
                    publish(run, run.stopRequested ? "STOPPING" : "RECORDING",
                            run.stopRequested ? "Saving both recordings…" : "Recording phone microphone…", true);
                    lastUpdate = now;
                }
            }
            if (!run.stopRequested) { run.stopRequested = true; run.stopReason = "duration_limit"; }
        } catch (Exception error) {
            if (!run.stopRequested) { run.captureError = error.toString(); run.stopReason = "capture_error"; }
        } finally {
            if (run.durationStop != null) main.removeCallbacks(run.durationStop);
            synchronized (run.recorderLock) {
                if (recorder != null) {
                    try { if (recorder.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) recorder.stop(); }
                    catch (RuntimeException ignored) { /* Release still owns the platform resource. */ }
                    try { recorder.release(); }
                    catch (RuntimeException error) {
                        if (run.captureError == null) { run.captureError = error.toString(); run.stopReason = "capture_error"; }
                    }
                }
                run.recorder = null; run.microphoneReleased = true; run.endNs = SystemClock.elapsedRealtimeNanos();
            }
            publish(run, "STOPPING", "Microphone stopped. Finishing RE-USE and saving both recordings…", true);
            if (run.captureError != null) run.session.close(); else run.session.finish();
        }
    }

    private AudioRecord createRecorder() throws IOException {
        if (context.checkSelfPermission(Manifest.permission.RECORD_AUDIO) != PackageManager.PERMISSION_GRANTED)
            throw new SecurityException("Microphone permission is no longer available");
        int minimum = AudioRecord.getMinBufferSize(RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_16BIT);
        if (minimum <= 0) throw new IOException("This device does not support 16 kHz mono PCM microphone capture");
        AudioFormat format = new AudioFormat.Builder().setSampleRate(RATE).setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                .setEncoding(AudioFormat.ENCODING_PCM_16BIT).build();
        RuntimeException last = null;
        for (int source : new int[]{MediaRecorder.AudioSource.VOICE_RECOGNITION, MediaRecorder.AudioSource.MIC}) {
            AudioRecord recorder = null;
            try {
                recorder = new AudioRecord.Builder().setAudioSource(source).setAudioFormat(format)
                        .setBufferSizeInBytes(Math.max(minimum * 4, RATE * 2)).build();
                if (recorder.getState() == AudioRecord.STATE_INITIALIZED) return recorder;
            } catch (RuntimeException error) { last = error; }
            if (recorder != null) recorder.release();
        }
        throw new IOException("Android could not initialize the microphone", last);
    }

    private void stopRecorderAsync(Run run) {
        if (run.recorder == null) return;
        new Thread(() -> {
            synchronized (run.recorderLock) {
                AudioRecord recorder = run.recorder;
                if (recorder != null) try {
                    if (recorder.getRecordingState() == AudioRecord.RECORDSTATE_RECORDING) recorder.stop();
                } catch (RuntimeException ignored) { /* Capture worker performs the single release. */ }
            }
        }, "reuse-phone-stop").start();
    }

    private static void updateRoute(Run run, AudioRecord recorder) {
        AudioDeviceInfo route = recorder.getRoutedDevice();
        if (route == null) { run.route = "Android microphone"; return; }
        run.routeType = route.getType();
        switch (route.getType()) {
            case AudioDeviceInfo.TYPE_BUILTIN_MIC: run.route = "Built-in microphone"; break;
            case AudioDeviceInfo.TYPE_WIRED_HEADSET: run.route = "Wired headset microphone"; break;
            case AudioDeviceInfo.TYPE_USB_DEVICE:
            case AudioDeviceInfo.TYPE_USB_HEADSET: run.route = "USB microphone"; break;
            case AudioDeviceInfo.TYPE_BLUETOOTH_SCO: run.route = "Bluetooth microphone"; break;
            default: run.route = "Android microphone (route " + route.getType() + ")";
        }
    }

    /** Output availability only: no AudioTrack is opened and no playout timing is implied. */
    private static void traceOutputAvailability(Run run, byte[] pcm) {
        int samples = pcm.length / 2;
        int event = run.outputTraceEvents;
        if (event < MAX_OUTPUT_TRACE_EVENTS) {
            run.outputFirstSamples[event] = run.callbackOutputSamples;
            run.outputSampleCounts[event] = samples;
            // Timestamp at callback completion after recording its offsets; memory operations only.
            run.outputHostNs[event] = System.nanoTime();
            run.outputTraceEvents = event + 1;
        } else run.omittedOutputTraceEvents++;
        run.callbackOutputSamples += samples;
    }

    private void scheduleTerminal(Run run, Map<String, Object> engineStats, String engineError) {
        synchronized (lock) {
            if (run.terminalScheduled) return;
            run.terminalScheduled = true;
            if (!run.stopRequested) { run.stopRequested = true; run.stopReason = engineError == null ? "completed" : "processing_error"; }
        }
        stopRecorderAsync(run);
        preparation.execute(() -> finishTerminal(run, engineStats, engineError));
    }

    private void finishTerminal(Run run, Map<String, Object> engineStats, String engineError) {
        String error = engineError != null ? engineError : run.captureError;
        try {
            Thread capture = run.captureThread;
            if (capture != null) {
                capture.join(3000);
                if (capture.isAlive()) throw new IOException("Microphone worker did not stop; capture cannot be marked complete");
            } else run.microphoneReleased = true;
            if (error == null) error = run.captureError;
            Map<String, Object> phone = captureMetadata(run, engineStats, error == null);
            Map<String, Object> stats = new LinkedHashMap<>(engineStats);
            stats.put("source", "android_audio_record"); stats.put("phone_capture", phone);
            if (error != null) { stats.put("phone_capture_error", error); stats.put("state", "ERROR"); }
            Files.write(run.directory.resolve("phone-capture.json"),
                    (new JSONObject(phone).toString(2) + "\n").getBytes(StandardCharsets.UTF_8));
            Files.write(run.directory.resolve("stats.json"),
                    (new JSONObject(stats).toString(2) + "\n").getBytes(StandardCharsets.UTF_8));
            synchronized (lock) { if (active == run) active = null; }
            if (error != null) { publish(run, "ERROR", "Recording stopped: " + error, false); postError(error); }
            else if (run.samples == 0) publish(run, "IDLE", "Stopped before the microphone opened.", false);
            else {
                latestDirectory = run.directory.toFile();
                publish(run, "IDLE", "Saved. Play Before or After to compare the same recording.", false);
                main.post(() -> { if (!closed && !isActive()) listener.onFinished(run.directory.toFile(), stats); });
            }
        } catch (Exception failure) {
            synchronized (lock) { if (active == run) active = null; }
            publish(run, "ERROR", "Could not finalize recording: " + failure, false); postError(failure.toString());
        } finally { shutdownIfClosed(); }
    }

    private static Map<String, Object> captureMetadata(Run run, Map<String, Object> engine, boolean success) {
        Map<String, Object> s = new LinkedHashMap<>();
        s.put("source", "android_audio_record"); s.put("session_success", success);
        s.put("requested_sample_rate", RATE); s.put("actual_sample_rate", run.actualRate);
        s.put("channels", 1); s.put("encoding", "s16le"); s.put("model_threads", run.threads);
        s.put("requested_source_preference", "VOICE_RECOGNITION then MIC"); s.put("actual_android_source", run.actualSource);
        s.put("preferred_builtin_accepted", run.preferredBuiltInAccepted); s.put("route", run.route); s.put("route_type", run.routeType);
        s.put("microphone_started", run.microphoneStarted); s.put("microphone_released", run.microphoneReleased);
        s.put("platform_silencing_detected", run.platformSilenced);
        s.put("read_calls", run.readCalls); s.put("zero_reads", run.zeroReads); s.put("accepted_input_samples", run.samples);
        s.put("capture_seconds", run.samples / (double) RATE); s.put("stop_reason", run.stopReason);
        s.put("capture_wall_ms", run.beginNs == 0 ? 0 : (run.endNs - run.beginNs) / 1e6);
        s.put("hardware_gap_free_verified", false);
        s.put("scope", "Android-delivered microphone PCM before RE-USE; platform processing may already apply");
        boolean noDrops = number(engine, "dropped_input_samples").longValue() == 0 && number(engine, "queue_overflows").longValue() == 0;
        boolean allInputProcessed = number(engine, "input_samples").longValue() == run.samples
                && number(engine, "processed_input_samples").longValue() == run.samples;
        boolean averageKeptUp = success && run.samples > 0 && allInputProcessed
                && number(engine, "processing_rtf").doubleValue() < 1 && noDrops;
        boolean everyDeadline = success && run.samples > 0 && number(engine, "deadline_misses").longValue() == 0;
        s.put("average_realtime_criterion_met", averageKeptUp);
        s.put("all_100ms_deadlines_met", everyDeadline);
        Map<String, Object> realTime = new LinkedHashMap<>();
        realTime.put("average_processing_and_no_application_drops", averageKeptUp);
        realTime.put("all_100ms_deadlines_met", everyDeadline);
        realTime.put("continuous_audio_track_playout_verified", false);
        realTime.put("audio_track_underrun_count", "not_measured");
        s.put("real_time_condition", realTime);
        s.put("output_timing_clock", "System.nanoTime monotonic host clock");
        s.put("capture_start_host_monotonic_ns", run.captureStartHostNs);
        s.put("first_input_host_monotonic_ns", run.firstInputHostNs);
        s.put("first_input_relative_capture_start_ms", run.firstInputHostNs == 0 ? 0
                : (run.firstInputHostNs - run.captureStartHostNs) / 1e6);
        s.put("output_timing_trace_reference", "AudioRecord.startRecording completion; output timestamps are availability at enhancement callback completion");
        s.put("output_timing_trace_scope", "Model output callback availability, not AudioTrack speaker playout; offline startup-buffer simulation only");
        s.put("audio_track_playout_verified", false);
        s.put("audio_track_underruns_verified", false);
        s.put("output_timing_trace_event_capacity", MAX_OUTPUT_TRACE_EVENTS);
        s.put("output_timing_trace_omitted_events", run.omittedOutputTraceEvents);
        s.put("output_timing_callback_samples", run.callbackOutputSamples);
        s.put("output_timing_trace_complete", run.omittedOutputTraceEvents == 0
                && run.callbackOutputSamples == number(engine, "output_samples").longValue());
        ArrayList<Map<String, Object>> trace = new ArrayList<>(run.outputTraceEvents);
        for (int i = 0; i < run.outputTraceEvents; i++) {
            Map<String, Object> event = new LinkedHashMap<>();
            event.put("relativeCaptureStartMs", (run.outputHostNs[i] - run.captureStartHostNs) / 1e6);
            event.put("outputFirstSample", run.outputFirstSamples[i]);
            event.put("outputSamples", run.outputSampleCounts[i]);
            trace.add(event);
        }
        s.put("output_timing_trace", trace);
        s.put("error", run.captureError == null ? "" : run.captureError);
        return s;
    }

    private static Number number(Map<String, Object> values, String key) {
        Object value = values.get(key); return value instanceof Number ? (Number) value : 0;
    }
    private void failPreparation(Run run, String message) {
        synchronized (lock) { if (active == run) active = null; }
        publish(run, "ERROR", "Could not prepare recording: " + message, false); postError(message); shutdownIfClosed();
    }
    private void publish(Run run, String phase, String message, boolean isActive) {
        StateSnapshot snapshot = new StateSnapshot(phase, message, isActive, run);
        main.post(() -> {
            synchronized (lock) {
                if (closed || (active != run && (active != null || isActive))) return;
            }
            listener.onState(snapshot);
        });
    }
    private void postError(String error) { main.post(() -> { if (!closed && !isActive()) listener.onError(error); }); }

    private Path extractModel() throws Exception {
        Path model = context.getFilesDir().toPath().resolve("phone-model-" + MODEL_SHA);
        Files.createDirectories(model);
        if (Files.isRegularFile(model.resolve("model.onnx")) && Files.isRegularFile(model.resolve("manifest.json"))) {
            String declared = new JSONObject(new String(Files.readAllBytes(model.resolve("manifest.json")), StandardCharsets.UTF_8)).getString("model_sha256");
            if (MODEL_SHA.equals(declared) && MODEL_SHA.equals(sha256(model.resolve("model.onnx")))) return model;
        }
        for (String name : new String[]{"model.onnx", "manifest.json", "LICENSE-NVIDIA.txt"}) {
            Path part = model.resolve(name + "." + UUID.randomUUID() + ".part");
            try (InputStream source = context.getAssets().open("model/" + name)) { Files.copy(source, part, StandardCopyOption.REPLACE_EXISTING); }
            Files.move(part, model.resolve(name), StandardCopyOption.REPLACE_EXISTING);
        }
        String declared = new JSONObject(new String(Files.readAllBytes(model.resolve("manifest.json")), StandardCharsets.UTF_8)).getString("model_sha256");
        if (!MODEL_SHA.equals(declared) || !MODEL_SHA.equals(sha256(model.resolve("model.onnx")))) throw new IOException("Bundled model hash mismatch");
        return model;
    }
    private static String sha256(Path path) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        try (InputStream in = Files.newInputStream(path)) {
            byte[] buffer = new byte[65536]; int read;
            while ((read = in.read(buffer)) != -1) digest.update(buffer, 0, read);
        }
        StringBuilder text = new StringBuilder(64);
        for (byte b : digest.digest()) text.append(String.format(Locale.ROOT, "%02x", b & 255));
        return text.toString();
    }

    /** Destruction stops capture; work is drained rather than silently continuing in background. */
    @Override public void close() {
        synchronized (lock) { closed = true; }
        stop("activity_destroyed"); shutdownIfClosed();
    }
    private void shutdownIfClosed() { synchronized (lock) { if (closed && active == null) preparation.shutdown(); } }
}
