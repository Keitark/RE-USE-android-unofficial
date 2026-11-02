package io.github.reuse.qnnprobe;

import android.content.Context;
import android.content.Intent;
import android.content.IntentFilter;
import android.os.BatteryManager;
import android.os.Build;
import android.os.Debug;
import android.os.PowerManager;
import android.os.Process;
import android.os.SystemClock;
import android.system.Os;
import android.system.OsConstants;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.AccessDeniedException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.NoSuchFileException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.LockSupport;
import org.json.JSONArray;
import org.json.JSONObject;

/**
 * Bounded best-effort benchmark observations. A separate sampler still competes with inference:
 * these are instrumented runs, and collection cost is not claimed to be excluded from timings.
 */
final class BenchmarkTelemetry implements AutoCloseable {
    static final String EVENTS_FILE = "benchmark-telemetry.jsonl";
    static final String SUMMARY_FILE = "benchmark-telemetry-summary.json";
    static final long SAMPLE_INTERVAL_MS = 500;
    private static final long INTERVAL_NS = TimeUnit.MILLISECONDS.toNanos(SAMPLE_INTERVAL_MS);
    private static final long MAX_BYTES = 8L * 1024 * 1024;
    private static final int MAX_SAMPLES = 5000, MAX_CALLS = 5000, MAX_QUEUED_CALLS = 512;
    private final Context context;
    private final Path directory;
    private final ArrayBlockingQueue<CallEvent> calls = new ArrayBlockingQueue<>(MAX_QUEUED_CALLS);
    private final AtomicInteger callsOffered = new AtomicInteger();
    private final AtomicLong droppedCalls = new AtomicLong();
    private final List<CpuSource> cpus = new ArrayList<>();
    private final List<ThermalSource> thermals = new ArrayList<>();
    private final List<TextSource> allSources = new ArrayList<>();
    private volatile Stage stage = new Stage("", "", "initializing", 0, -1);
    private volatile Thread worker;
    private volatile boolean running, closed, limitReached, started, workerExited;
    private volatile String status = "DISABLED", errorClass;
    private volatile long sampleCount, callCount, bytesWritten, lagNs, skippedIntervals;
    private volatile long observerWallNs, observerCpuNs, startElapsedNs, lastElapsedNs;
    private volatile long lastSampleElapsedNs, previousProcessCpuMs = -1, previousObserverCpuNs = -1;
    private volatile long sourceReadErrors, invalidSourceValues;
    private long ticksPerSecond = -1, lastCheckpointNs, batteryReadNs = -1, powerReadNs = -1;
    private long headroomReadNs = -1;
    private JSONObject batteryCache = new JSONObject(), powerCache = new JSONObject(), headroomCache = new JSONObject();
    private BenchmarkTelemetryMath.CpuTimes previousSystemCpu;
    private TextSource procStat, loadAverage, cpuPressure;
    private BufferedWriter writer;

    private BenchmarkTelemetry(Context context, Path directory) {
        this.context = context;
        this.directory = directory;
    }

    /** All methods are deliberately nonthrowing so inaccessible telemetry cannot invalidate audio. */
    static BenchmarkTelemetry start(Context context, Path captureDirectory) {
        BenchmarkTelemetry result = new BenchmarkTelemetry(context, captureDirectory);
        try {
            if (result.context == null || captureDirectory == null) throw new IllegalArgumentException("Context and capture directory required");
            Path root = result.context.getFilesDir().toPath().resolve("captures").toRealPath();
            Path actual = captureDirectory.toRealPath();
            if (!Files.isDirectory(actual) || !root.equals(actual.getParent()))
                throw new IOException("Telemetry requires an existing immediate app-private capture directory");
            // Existing files are never truncated if a caller accidentally starts a second collector.
            result.writer = Files.newBufferedWriter(actual.resolve(EVENTS_FILE), StandardCharsets.UTF_8,
                    StandardOpenOption.CREATE_NEW, StandardOpenOption.WRITE);
            result.startElapsedNs = SystemClock.elapsedRealtimeNanos();
            result.lastElapsedNs = result.startElapsedNs;
            result.running = true; result.started = true; result.status = "RUNNING";
            result.worker = new Thread(result::sampleLoop, "reuse-benchmark-telemetry");
            result.worker.setDaemon(true);
            result.worker.start();
        } catch (Exception | LinkageError error) {
            result.running = false; result.status = "DISABLED"; result.errorClass = error.getClass().getSimpleName();
            if (result.writer != null) try { result.writer.close(); } catch (IOException ignored) { }
        }
        return result;
    }

    /** No sensors, clock reads, JSON construction or file writes occur on this foreground path. */
    void mark(String trial, String backend, String phase, int threads, int call) {
        if (!running) return;
        stage = new Stage(BenchmarkTelemetryMath.label(trial), BenchmarkTelemetryMath.label(backend),
                BenchmarkTelemetryMath.label(phase), threads, call);
    }

    /** Called after both original wall-time and process-CPU snapshots; only queues primitive data. */
    void recordCall(long startElapsedRealtimeNs, long endElapsedRealtimeNs, double wallMs, long processCpuMs) {
        if (!running || closed) return;
        if (startElapsedRealtimeNs < 0 || endElapsedRealtimeNs < startElapsedRealtimeNs
                || !Double.isFinite(wallMs) || wallMs < 0 || processCpuMs < 0) {
            droppedCalls.incrementAndGet(); return;
        }
        int number = callsOffered.getAndIncrement();
        if (number >= MAX_CALLS || !calls.offer(new CallEvent(stage, startElapsedRealtimeNs,
                endElapsedRealtimeNs, wallMs, processCpuMs))) droppedCalls.incrementAndGet();
    }

    JSONObject summary() {
        JSONObject out = new JSONObject();
        put(out, "schema_version", 1); put(out, "source", "app_benchmark_telemetry");
        put(out, "status", status); put(out, "enabled", started); put(out, "instrumented_run", started);
        put(out, "events_file", started ? EVENTS_FILE : JSONObject.NULL);
        put(out, "summary_file", SUMMARY_FILE); put(out, "sample_interval_ms", SAMPLE_INTERVAL_MS);
        put(out, "sample_count", sampleCount); put(out, "call_events_written", callCount);
        put(out, "call_events_offered", callsOffered.get()); put(out, "dropped_call_events", droppedCalls.get());
        put(out, "pending_call_events", calls.size()); put(out, "event_bytes_written", bytesWritten);
        put(out, "max_event_bytes", MAX_BYTES); put(out, "max_samples", MAX_SAMPLES); put(out, "max_call_events", MAX_CALLS);
        put(out, "limit_reached", limitReached); put(out, "sampler_thread_stopped", workerExited || !started);
        put(out, "closed", closed); put(out, "start_elapsed_realtime_ns", startElapsedNs);
        put(out, "last_elapsed_realtime_ns", lastElapsedNs); put(out, "skipped_sample_intervals", skippedIntervals);
        put(out, "maximum_sample_scheduling_lag_ms", lagNs / 1_000_000.0);
        put(out, "observer_collection_wall_ms", observerWallNs / 1_000_000.0);
        put(out, "observer_thread_cpu_ms", observerCpuNs / 1_000_000.0);
        put(out, "observer_cost_scope", "Sampler discovery, sensor reads, event serialization, writes and periodic summary work; excludes sleep, foreground queue/context allocation, final summary write, and driver/OS side effects. Observer competes with inference and contributes to process CPU time.");
        put(out, "clock_scope", "scaling_cur_freq is reported/requested driver frequency, not guaranteed actual hardware clock; cpuinfo_cur_freq is separately labelled when readable. Sparse samples are not continuous frequency traces.");
        put(out, "temperature_scope", "Thermal-zone labels are vendor-reported. Battery temperature is separate and is not CPU temperature. Implausible/sentinel readings remain invalid.");
        put(out, "load_scope", "App CPU percentage is one-core equivalent and can exceed 100%. System/background load is unavailable when Android denies aggregate proc files. Other-system estimates, when available, include kernel/drivers and other foreground apps; no per-app attribution or causal slowdown claim.");
        put(out, "source_read_errors", sourceReadErrors); put(out, "invalid_source_values", invalidSourceValues);
        put(out, "cpu_count_discovered", cpus.size()); put(out, "thermal_sensors_selected", thermals.size());
        put(out, "proc_stat_status", procStat == null ? "not_probed" : procStat.status);
        put(out, "load_average_status", loadAverage == null ? "not_probed" : loadAverage.status);
        put(out, "cpu_pressure_status", cpuPressure == null ? "not_probed" : cpuPressure.status);
        put(out, "system_load_available", procStat != null && "available".equals(procStat.status));
        put(out, "other_app_identities_collected", false); put(out, "microphone_started", false);
        put(out, "error_class", errorClass == null ? JSONObject.NULL : errorClass);
        return out;
    }

    @Override public void close() {
        if (closed) return;
        running = false;
        Thread thread = worker;
        if (thread != null && thread != Thread.currentThread()) {
            // Wake a parked sampler without interrupting a sysfs/JSON writer mid-read or mid-write.
            LockSupport.unpark(thread);
            try { thread.join(2500); } catch (InterruptedException error) { Thread.currentThread().interrupt(); }
        }
        closed = true;
        if (thread != null && thread.isAlive()) {
            status = "STOP_REQUESTED_SAMPLER_ACTIVE";
            // A potentially blocked writer remains owned by its thread; do not race it here.
        } else if (started) {
            status = errorClass != null ? "STOPPED_WITH_ERROR" : limitReached ? "LIMIT_REACHED" : "COMPLETE";
            writeSummary();
        }
    }

    private void sampleLoop() {
        try {
            Process.setThreadPriority(Process.THREAD_PRIORITY_BACKGROUND);
            long began = SystemClock.elapsedRealtimeNanos(), cpu = Debug.threadCpuTimeNanos();
            discover();
            writeLine(metadata());
            observerWallNs += Math.max(0, SystemClock.elapsedRealtimeNanos() - began);
            observerCpuNs += Math.max(0, Debug.threadCpuTimeNanos() - cpu);
            long next = SystemClock.elapsedRealtimeNanos();
            while (running && !limitReached) {
                began = SystemClock.elapsedRealtimeNanos(); cpu = Debug.threadCpuTimeNanos();
                drainCalls();
                long now = SystemClock.elapsedRealtimeNanos();
                if (now >= next) {
                    lagNs = Math.max(lagNs, now - next);
                    if (sampleCount >= MAX_SAMPLES) limitReached = true;
                    else sample(now);
                    next += INTERVAL_NS;
                    if (now >= next) {
                        long missed = (now - next) / INTERVAL_NS + 1;
                        skippedIntervals += missed; next += missed * INTERVAL_NS;
                    }
                }
                now = SystemClock.elapsedRealtimeNanos();
                if (now - lastCheckpointNs >= TimeUnit.SECONDS.toNanos(1)) {
                    writer.flush(); writeSummary(); lastCheckpointNs = now;
                }
                observerWallNs += Math.max(0, SystemClock.elapsedRealtimeNanos() - began);
                observerCpuNs += Math.max(0, Debug.threadCpuTimeNanos() - cpu);
                if (!running || limitReached) break;
                long waitNs = Math.min(TimeUnit.MILLISECONDS.toNanos(100), Math.max(1, next - SystemClock.elapsedRealtimeNanos()));
                LockSupport.parkNanos(waitNs);
            }
            began = SystemClock.elapsedRealtimeNanos(); cpu = Debug.threadCpuTimeNanos();
            drainCalls(); writer.flush();
            observerWallNs += Math.max(0, SystemClock.elapsedRealtimeNanos() - began);
            observerCpuNs += Math.max(0, Debug.threadCpuTimeNanos() - cpu);
        } catch (Exception | LinkageError error) {
            errorClass = error.getClass().getSimpleName(); status = "STOPPED_WITH_ERROR";
        } finally {
            running = false; lastElapsedNs = SystemClock.elapsedRealtimeNanos();
            if (writer != null) try { writer.close(); } catch (IOException error) { errorClass = error.getClass().getSimpleName(); }
            workerExited = true;
        }
    }

    private void discover() {
        procStat = source(Paths.get("/proc/stat"), true); loadAverage = source(Paths.get("/proc/loadavg"));
        cpuPressure = source(Paths.get("/proc/pressure/cpu"));
        procStat.read(); loadAverage.read(); cpuPressure.read();
        try { ticksPerSecond = Os.sysconf(OsConstants._SC_CLK_TCK); } catch (Exception ignored) { }
        for (int index = 0; index < 32; index++) {
            Path path = Paths.get("/sys/devices/system/cpu/cpu" + index);
            if (!Files.isDirectory(path)) continue;
            CpuSource cpu = new CpuSource(index, path); cpus.add(cpu);
            cpu.reported.read(); cpu.hardware.read(); cpu.online.read();
            cpu.policyMinimum.read(); cpu.policyMaximum.read();
            cpu.minimumValue = cpu.minimum.read(); cpu.maximumValue = cpu.maximum.read();
        }
        // Discover only generic vendor sensor labels, never paths beyond this fixed sysfs subtree.
        int[] selectedByGroup = new int[5];
        final int[] groupLimits = {8, 2, 2, 2, 2}; // CPU, GPU, SoC, skin, battery.
        for (int index = 0; index < 128 && thermals.size() < 16; index++) {
            Path path = Paths.get("/sys/class/thermal/thermal_zone" + index);
            if (!Files.isDirectory(path)) continue;
            TextSource type = source(path.resolve("type")); String text = type.read();
            if (text == null) continue;
            String normalized = text.toLowerCase(Locale.ROOT);
            int group = normalized.contains("battery") ? 4 : normalized.contains("skin") ? 3
                    : normalized.contains("gpu") ? 1 : normalized.contains("soc") ? 2 : normalized.contains("cpu") ? 0 : -1;
            if (group < 0 || selectedByGroup[group] >= groupLimits[group]) continue;
            ThermalSource sensor = new ThermalSource(index, BenchmarkTelemetryMath.label(text), source(path.resolve("temp")));
            sensor.temperature.read(); thermals.add(sensor); selectedByGroup[group]++;
        }
    }

    private JSONObject metadata() {
        JSONObject out = new JSONObject(); put(out, "type", "metadata"); put(out, "schema_version", 1);
        put(out, "sample_interval_ms", SAMPLE_INTERVAL_MS); put(out, "elapsed_realtime_clock", "Android elapsedRealtimeNanos; nanoseconds since boot including sleep");
        put(out, "clock_ticks_per_second", ticksPerSecond > 0 ? ticksPerSecond : JSONObject.NULL);
        JSONArray cpu = new JSONArray();
        for (CpuSource value : cpus) {
            JSONObject item = new JSONObject(); put(item, "cpu", value.index);
            put(item, "reported_frequency_source", value.reported.describe()); put(item, "hardware_frequency_source", value.hardware.describe());
            put(item, "online_source", value.online.describe()); put(item, "minimum_frequency_source", value.minimum.describe());
            put(item, "maximum_frequency_source", value.maximum.describe());
            put(item, "policy_minimum_frequency_source", value.policyMinimum.describe());
            put(item, "policy_maximum_frequency_source", value.policyMaximum.describe());
            put(item, "minimum_frequency_khz", integer(value.minimum, value.minimumValue, false));
            put(item, "maximum_frequency_khz", integer(value.maximum, value.maximumValue, false)); cpu.put(item);
        }
        put(out, "cpus", cpu);
        JSONArray temperatures = new JSONArray();
        for (ThermalSource value : thermals) {
            JSONObject item = new JSONObject(); put(item, "zone", value.index); put(item, "vendor_type", value.type);
            put(item, "source", value.temperature.describe());
            put(item, "raw_unit", BenchmarkTelemetryMath.temperatureUnitVerified(value.type)
                    ? "millidegree_celsius" : "vendor_unit_unverified"); temperatures.put(item);
        }
        put(out, "thermal_sensors", temperatures);
        put(out, "thermal_selection", "Up to 8 CPU, 2 GPU, 2 SoC, 2 skin and 2 battery vendor-labelled sensors among at most 128 zones");
        put(out, "system_cpu_source", procStat.describe());
        put(out, "load_average_source", loadAverage.describe()); put(out, "cpu_pressure_source", cpuPressure.describe());
        put(out, "privacy_scope", "No application names, other-process identities, device serials, microphone or network data.");
        return out;
    }

    private void sample(long now) throws IOException {
        long processCpu = Process.getElapsedCpuTime(), observerCpu = Debug.threadCpuTimeNanos();
        JSONObject out = new JSONObject(); put(out, "type", "sample"); put(out, "elapsed_realtime_ns", now);
        put(out, "sample_index", sampleCount); stage.add(out);
        JSONObject process = new JSONObject(); put(process, "total_process_cpu_ms", processCpu);
        put(process, "interval_status", lastSampleElapsedNs == 0 ? "baseline_only" : "delta");
        Double processPercent = lastSampleElapsedNs == 0 ? null : BenchmarkTelemetryMath.processOneCorePercent(
                previousProcessCpuMs, processCpu, lastSampleElapsedNs, now);
        put(process, "one_core_percent", processPercent);
        long processDelta = previousProcessCpuMs < 0 ? -1 : processCpu - previousProcessCpuMs;
        long observerDelta = previousObserverCpuNs < 0 ? -1 : observerCpu - previousObserverCpuNs;
        put(process, "observer_thread_cpu_delta_ms", observerDelta < 0 ? null : observerDelta / 1_000_000.0);
        if (processPercent != null && observerDelta >= 0 && now > lastSampleElapsedNs)
            put(process, "excluding_sampler_approximate_one_core_percent", Math.max(0, processPercent - 100.0 * observerDelta / (now - lastSampleElapsedNs)));
        put(out, "app_process_cpu", process);
        JSONObject system = new JSONObject(); String stat = procStat.read();
        BenchmarkTelemetryMath.CpuTimes current = null;
        if (stat != null) try { current = BenchmarkTelemetryMath.parseProcStat(stat); }
        catch (RuntimeException error) { procStat.status = "invalid"; invalidSourceValues++; }
        put(system, "status", procStat.status); put(system, "capacity_percent", BenchmarkTelemetryMath.systemBusyPercent(previousSystemCpu, current));
        put(system, "other_system_estimated_one_core_percent", BenchmarkTelemetryMath.otherSystemOneCorePercent(
                previousSystemCpu, current, processDelta, now - lastSampleElapsedNs, ticksPerSecond));
        put(system, "other_system_estimate_scope", "Common-window aggregate busy CPU minus app CPU; includes kernel/drivers and any other foreground/background work, not per-app attribution.");
        put(out, "system_cpu", system);
        JSONObject load = new JSONObject(); String loadText = loadAverage.read();
        if (loadText != null) try {
            double[] values = BenchmarkTelemetryMath.parseLoadAverage(loadText);
            put(load, "one_minute", values[0]); put(load, "five_minutes", values[1]); put(load, "fifteen_minutes", values[2]);
        } catch (RuntimeException error) { loadAverage.status = "invalid"; invalidSourceValues++; }
        put(load, "status", loadAverage.status); put(load, "scope", "Runnable/uninterruptible task averages, not CPU utilization percentages"); put(out, "load_average", load);
        JSONObject pressure = new JSONObject(); String pressureText = cpuPressure.read();
        if (pressureText != null) try {
            for (var item : BenchmarkTelemetryMath.parseCpuPressure(pressureText).entrySet()) put(pressure, item.getKey(), item.getValue());
        } catch (RuntimeException error) { cpuPressure.status = "invalid"; invalidSourceValues++; }
        put(pressure, "status", cpuPressure.status); put(out, "cpu_pressure", pressure);
        JSONArray frequencies = new JSONArray();
        for (CpuSource value : cpus) {
            JSONObject item = new JSONObject(); put(item, "cpu", value.index);
            put(item, "reported_driver_frequency_khz", integer(value.reported, value.reported.read(), false));
            put(item, "hardware_derived_frequency_khz", integer(value.hardware, value.hardware.read(), false));
            put(item, "policy_min_frequency_khz", integer(value.policyMinimum, value.policyMinimum.read(), false));
            put(item, "policy_max_frequency_khz", integer(value.policyMaximum, value.policyMaximum.read(), false));
            put(item, "online", integer(value.online, value.online.read(), true)); frequencies.put(item);
        }
        put(out, "cpu_clocks", frequencies);
        JSONArray temperatures = new JSONArray();
        for (ThermalSource value : thermals) {
            JSONObject item = new JSONObject(); put(item, "zone", value.index); put(item, "vendor_type", value.type);
            String raw = value.temperature.read();
            put(item, "status", value.temperature.status);
            if (raw != null) try {
                long milli = Long.parseLong(raw);
                boolean verifiedUnit = BenchmarkTelemetryMath.temperatureUnitVerified(value.type);
                Double celsius = verifiedUnit ? BenchmarkTelemetryMath.milliCelsius(milli) : null;
                put(item, "raw_value", milli); put(item, "raw_unit", verifiedUnit ? "millidegree_celsius" : "vendor_unit_unverified");
                if (verifiedUnit) put(item, "raw_millicelsius", milli);
                put(item, "celsius", celsius);
                if (!verifiedUnit) put(item, "status", "vendor_unit_unverified");
                else if (celsius == null) { put(item, "status", "invalid_or_sentinel"); invalidSourceValues++; }
            } catch (RuntimeException error) { put(item, "status", "invalid"); invalidSourceValues++; }
            temperatures.put(item);
        }
        put(out, "thermal_sensors", temperatures);
        if (batteryReadNs < 0 || now - batteryReadNs >= TimeUnit.SECONDS.toNanos(1)) readBattery(now);
        if (powerReadNs < 0 || now - powerReadNs >= TimeUnit.SECONDS.toNanos(1)) readPower(now);
        if (Build.VERSION.SDK_INT >= 30 && (headroomReadNs < 0 || now - headroomReadNs >= TimeUnit.SECONDS.toNanos(10))) readHeadroom(now);
        put(out, "battery", batteryCache); put(out, "battery_age_ms", (now - batteryReadNs) / 1_000_000.0);
        put(out, "power", powerCache); put(out, "power_age_ms", (now - powerReadNs) / 1_000_000.0);
        put(out, "thermal_headroom", headroomCache);
        put(out, "thermal_headroom_age_ms", headroomReadNs < 0 ? null : (now - headroomReadNs) / 1_000_000.0);
        put(out, "collection_end_elapsed_realtime_ns", SystemClock.elapsedRealtimeNanos());
        if (writeLine(out)) sampleCount++;
        lastSampleElapsedNs = now; previousProcessCpuMs = processCpu; previousObserverCpuNs = observerCpu;
        previousSystemCpu = current; lastElapsedNs = SystemClock.elapsedRealtimeNanos();
    }

    private void readBattery(long now) {
        JSONObject out = new JSONObject(); batteryReadNs = now;
        put(out, "source", "ACTION_BATTERY_CHANGED"); put(out, "temperature_scope", "Battery temperature; not CPU temperature");
        try {
            Intent intent = context.registerReceiver(null, new IntentFilter(Intent.ACTION_BATTERY_CHANGED));
            if (intent == null || !intent.hasExtra(BatteryManager.EXTRA_TEMPERATURE)) put(out, "status", "unavailable");
            else {
                int raw = intent.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Integer.MIN_VALUE);
                Double value = BenchmarkTelemetryMath.batteryTenthsCelsius(raw);
                put(out, "raw_tenths_celsius", raw); put(out, "celsius", value);
                put(out, "status", value == null ? "invalid_or_sentinel" : "available");
                put(out, "plugged", intent.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0));
                put(out, "level", intent.getIntExtra(BatteryManager.EXTRA_LEVEL, -1));
                put(out, "scale", intent.getIntExtra(BatteryManager.EXTRA_SCALE, -1));
            }
        } catch (RuntimeException error) { put(out, "status", "unavailable"); put(out, "error_class", error.getClass().getSimpleName()); }
        batteryCache = out;
    }

    private void readPower(long now) {
        JSONObject out = new JSONObject(); powerReadNs = now;
        try {
            PowerManager manager = context.getSystemService(PowerManager.class);
            if (manager == null) put(out, "status", "unavailable");
            else {
                put(out, "status", "available"); put(out, "thermal_status", manager.getCurrentThermalStatus());
                put(out, "interactive", manager.isInteractive()); put(out, "idle", manager.isDeviceIdleMode());
                put(out, "power_save", manager.isPowerSaveMode());
            }
        } catch (RuntimeException error) { put(out, "status", "unavailable"); put(out, "error_class", error.getClass().getSimpleName()); }
        powerCache = out;
    }

    @android.annotation.TargetApi(30)
    private void readHeadroom(long now) {
        JSONObject out = new JSONObject(); headroomReadNs = now;
        put(out, "forecast_seconds", 0); put(out, "poll_interval_minimum_ms", 10000);
        try {
            PowerManager manager = context.getSystemService(PowerManager.class);
            float value = manager == null ? Float.NaN : manager.getThermalHeadroom(0);
            put(out, "value", Float.isFinite(value) ? value : null);
            put(out, "status", Float.isFinite(value) ? "available" : "unsupported_or_unavailable");
        } catch (RuntimeException error) { put(out, "status", "unavailable"); put(out, "error_class", error.getClass().getSimpleName()); }
        headroomCache = out;
    }

    private JSONObject integer(TextSource source, String text, boolean online) {
        JSONObject out = new JSONObject(); put(out, "status", source.status);
        if (text != null) try {
            long value = Long.parseLong(text);
            if ((online && value != 0 && value != 1) || (!online && value <= 0)) throw new IllegalArgumentException("Invalid clock/online value");
            put(out, "value", value);
        } catch (RuntimeException error) { put(out, "status", "invalid"); invalidSourceValues++; }
        return out;
    }

    private void drainCalls() throws IOException {
        CallEvent event;
        while ((event = calls.poll()) != null) {
            if (limitReached) { droppedCalls.incrementAndGet(); continue; }
            JSONObject out = new JSONObject(); put(out, "type", "call"); event.stage.add(out);
            put(out, "start_elapsed_realtime_ns", event.startNs); put(out, "end_elapsed_realtime_ns", event.endNs);
            put(out, "wall_ms", event.wallMs); put(out, "process_cpu_ms", event.processCpuMs);
            if (writeLine(out)) callCount++; else droppedCalls.incrementAndGet();
        }
    }

    private boolean writeLine(JSONObject value) throws IOException {
        String text = value.toString() + "\n";
        long size = text.getBytes(StandardCharsets.UTF_8).length;
        if (bytesWritten + size > MAX_BYTES) { limitReached = true; return false; }
        writer.write(text); bytesWritten += size; return true;
    }

    private void writeSummary() {
        if (directory == null || !started) return;
        try {
            Path destination = directory.resolve(SUMMARY_FILE), temporary = directory.resolve(SUMMARY_FILE + ".tmp");
            Files.write(temporary, summary().toString(2).getBytes(StandardCharsets.UTF_8));
            try { Files.move(temporary, destination, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING); }
            catch (AtomicMoveNotSupportedException error) { Files.move(temporary, destination, StandardCopyOption.REPLACE_EXISTING); }
        } catch (Exception error) { errorClass = error.getClass().getSimpleName(); }
    }

    private TextSource source(Path path) { return source(path, false); }
    private TextSource source(Path path, boolean firstLineOnly) {
        TextSource result = new TextSource(path, firstLineOnly); allSources.add(result); return result;
    }

    private final class TextSource {
        final Path path;
        final boolean firstLineOnly;
        volatile String status = "not_probed", error;
        boolean disabled;
        TextSource(Path path, boolean firstLineOnly) { this.path = path; this.firstLineOnly = firstLineOnly; }
        String read() {
            if (disabled) return null;
            try (InputStream stream = Files.newInputStream(path)) {
                ByteArrayOutputStream data = new ByteArrayOutputStream(); byte[] buffer = new byte[512]; int count;
                while ((count = stream.read(buffer)) != -1) {
                    int keep = count;
                    if (firstLineOnly) for (int i = 0; i < count; i++) if (buffer[i] == '\n') { keep = i; break; }
                    if (data.size() + keep > 4096) throw new IOException("Source exceeds fixed byte cap");
                    data.write(buffer, 0, keep);
                    if (keep < count) break;
                }
                status = "available"; error = null;
                return new String(data.toByteArray(), StandardCharsets.US_ASCII).trim();
            } catch (Exception failure) {
                String message = String.valueOf(failure.getMessage()).toLowerCase(Locale.ROOT);
                status = failure instanceof AccessDeniedException || failure instanceof SecurityException
                        || message.contains("permission denied") || message.contains("eacces") ? "permission_denied"
                        : failure instanceof NoSuchFileException || message.contains("no such file") ? "missing" : "unavailable";
                error = failure.getClass().getSimpleName(); disabled = true; sourceReadErrors++; return null;
            }
        }
        JSONObject describe() {
            JSONObject out = new JSONObject(); put(out, "file", path.toString()); put(out, "status", status);
            put(out, "error_class", error == null ? JSONObject.NULL : error); put(out, "denied_or_missing_retried", false); return out;
        }
    }

    private final class CpuSource {
        final int index;
        final TextSource reported, hardware, online, minimum, maximum, policyMinimum, policyMaximum;
        String minimumValue, maximumValue;
        CpuSource(int index, Path directory) {
            this.index = index;
            reported = source(directory.resolve("cpufreq/scaling_cur_freq")); hardware = source(directory.resolve("cpufreq/cpuinfo_cur_freq"));
            online = source(directory.resolve("online")); minimum = source(directory.resolve("cpufreq/cpuinfo_min_freq"));
            maximum = source(directory.resolve("cpufreq/cpuinfo_max_freq"));
            policyMinimum = source(directory.resolve("cpufreq/scaling_min_freq"));
            policyMaximum = source(directory.resolve("cpufreq/scaling_max_freq"));
        }
    }

    private static final class ThermalSource {
        final int index; final String type; final TextSource temperature;
        ThermalSource(int index, String type, TextSource temperature) { this.index = index; this.type = type; this.temperature = temperature; }
    }

    private static final class Stage {
        final String trial, backend, phase; final int threads, call;
        Stage(String trial, String backend, String phase, int threads, int call) {
            this.trial = trial; this.backend = backend; this.phase = phase; this.threads = threads; this.call = call;
        }
        void add(JSONObject out) {
            put(out, "trial", trial); put(out, "backend", backend); put(out, "phase", phase);
            put(out, "requested_intra_op_threads", threads); put(out, "call_index", call);
        }
    }

    private static final class CallEvent {
        final Stage stage; final long startNs, endNs, processCpuMs; final double wallMs;
        CallEvent(Stage stage, long startNs, long endNs, double wallMs, long processCpuMs) {
            this.stage = stage; this.startNs = startNs; this.endNs = endNs; this.wallMs = wallMs; this.processCpuMs = processCpuMs;
        }
    }

    private static void put(JSONObject object, String key, Object value) {
        try {
            if (value instanceof Double && !Double.isFinite((Double) value)) value = null;
            if (value instanceof Float && !Float.isFinite((Float) value)) value = null;
            object.put(key, value == null ? JSONObject.NULL : value);
        } catch (org.json.JSONException ignored) { /* Telemetry cannot invalidate an otherwise valid benchmark. */ }
    }
}
