# Benchmark clock, temperature and load diagnostics

The ordinary microphone demo now uses the
[recommended FP32 CPU configuration](recommended-cpu-profile.md) and exposes
no-audio diagnostic sharing. The clock/temperature timeline described here
belongs to the separate compute-test APK; microphone-demo JSON does not imply
that those sensor timelines or ADB-only system counters were collected.

The `0.7.0-telemetry-probe1` compute-test APK records a timestamped timeline
alongside the resource comparison and strict neural GPU experiments. Use
**Share diagnostics (no audio)** to export:

- `benchmark-telemetry.jsonl`: periodic sensor/load samples and completed model
  call events, tagged with trial, backend, phase, thread count and call index.
- `benchmark-telemetry-summary.json`: source availability, collection counts,
  dropped events, sampling lag and observer overhead.
- `adb-system-load.jsonl` and `adb-system-load-summary.json`, when a connected
  developer has explicitly collected and attached system load through ADB.

[Download the telemetry APK and checksums](https://github.com/Keitark/RE-USE-android-unofficial/releases/tag/v0.7.0-telemetry-probe1).
The [build and device-check receipt](validation/telemetry-probe1-build.json)
records 148 passing Java tests, 13 Python tests with 20 subtests, 230 package
checks, and the bounded Xperia logging test.

The APK keeps the same model, numerical gates, worker thread settings and
microphone/network permission policy. The resource button is labelled
**Compare CPU / NPU / legacy GPU** to distinguish the older hybrid test from
**Test full neural GPU (experimental)**.

## What the APK can measure

| Field | Meaning |
| --- | --- |
| `scaling_cur_freq` | Driver-reported/requested CPU frequency in kHz; not guaranteed measured hardware frequency |
| `cpuinfo_cur_freq` | Separate hardware-derived frequency reading when exposed by the driver |
| CPU online / frequency limits | Best-effort sysfs readings; missing and denied files are explicit |
| Named thermal zones | Vendor-labelled CPU, GPU, SoC, skin or battery sensors; raw readings, interpreted units and validity are retained |
| Battery temperature | Android battery temperature in degrees Celsius; it is not CPU temperature |
| Thermal status / headroom | Android thermal pressure signals; headroom is cached with age and polled no faster than once per ten seconds |
| App process CPU load | CPU-time delta divided by elapsed wall time, as one-core-equivalent percent; more than 100% is valid |
| Aggregate system CPU / load / pressure | Available only when Android permits the relevant proc files |

On the tested Xperia Android 14 app UID, CPU clock files and named thermal
zones are readable, while `/proc/stat`, `/proc/loadavg` and `/proc/pressure/cpu`
are denied. The APK records that denial and does not substitute zero or infer
other applications' activity from its own CPU time. No other app names, device
serial or microphone audio are collected.
Vendor control readings whose temperature units are unverified, such as the
Xperia's `socd` zone, retain their raw value with Celsius set to null.

The background sampler targets a 500 ms interval. Completed calls record
monotonic start/end timestamps and their existing wall/process-CPU measurements.
Sensor reads and file writes occur on the sampler thread, outside the measured
synchronous call region. The sampler still competes with inference and its CPU
time contributes to whole-process CPU time. These are instrumented runs;
collection cost, lag and event/file limits are reported, not assumed negligible.

## Xperia instrumented check

A foreground public-input resource comparison completed in 139.7 seconds with
280 app samples, 2,056 call events and zero dropped logging events. The matching
ADB recording supplied 240 samples and covered the entire app run; both files
were attached to that capture for sharing. Four-thread FP32 averaged 38.3 ms per
100 ms call, with a 45.6 ms maximum and no process call over 100 ms among 130.

The prime core's reported policy frequency cap reached 2.9952 GHz in early
trials and fell to 2.5152 GHz later; CPU 4's cap reached 2.496 GHz early and
2.2272 GHz later.
Vendor CPU temperature sensors peaked at 94.8 degrees Celsius. The slower
repeat did not coincide with an obvious aggregate CPU-load spike in matching
windows. These observations support changing frequency/power conditions as a
contributor. They do not establish the exact thermal/power-policy cause, actual
hardware frequency, per-app background activity or sustained microphone behavior.

The sampler reported 1,212 ms of its own thread CPU time across the run. This
discloses observer work; it is not an estimate of energy or proof that the
observer had no effect on the benchmark.

## Connected system-load collection

Android's ADB shell can expose aggregate system counters which the application
UID cannot read. The optional host tool collects those counters without other
process names and aligns them using device uptime:

```powershell
.\.venv\Scripts\python.exe tools\record_benchmark_device_load.py --adb <adb.exe> --serial <connected-device> --duration 240 --interval 1 --output .research/adb-system-load.jsonl
```

It writes a new host JSONL file and a summary. These files are explicitly marked
as ADB-shell measurements. A developer may attach them to the matching
app-private capture as `adb-system-load.jsonl` and
`adb-system-load-summary.json` before using the app's Share action. The tool
does not upload anything or identify which background app caused load.

System CPU busy percent, load-average queue values and CPU-pressure stalls are
different measurements. Compare overlapping time intervals with the app's
call and sensor timeline. Correlation between a slowdown and lower clocks,
temperature or system load is evidence for a follow-up controlled comparison,
not proof of a single cause. Charging, consecutive tests, core scheduling and
other activity can change conditions between runs.

## Build and test

Use the [full GPU asset preparation](full-gpu-reconstruction.md) first. Then:

```powershell
.\gradlew.bat :qnn-probe:testDebugUnitTest :qnn-probe:assembleDebug :qnn-probe:lintDebug :reuse-qnn:testDebugUnitTest -PreuseNeonFma=true -PreuseScanN16Registers=true
```

The fixed telemetry filenames are included in diagnostic sharing; model dumps
and generated public audio remain private. Sampling is bounded and stopped
when the benchmark completes, fails or is cancelled.

Relevant primary references: [Linux CPUFreq fields](https://cdn.kernel.org/doc/html/latest/admin-guide/pm/cpufreq.html),
[Android thermal guidance](https://developer.android.com/games/optimize/adpf/thermal),
and [Android 14 application SELinux restrictions](https://android.googlesource.com/platform/system/sepolicy/+/android-14.0.0_r1/private/app_neverallows.te).
