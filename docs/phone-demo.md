# Phone microphone demo

The standalone `:phone-demo` application records from the ordinary Android
microphone, processes the input through RE-USE while it arrives, and saves a
before/after WAV pair. Its package is `io.github.reuse.phone`. No Even app,
G2 glasses, Bluetooth connection, login, or cloud audio service is required.

## Install and run

Download `reuse-phone-demo.apk` from the
[v0.1.0-preview1 research release](https://github.com/Keitark/reuse-android/releases/tag/v0.1.0-preview1),
or use the [direct APK link](https://github.com/Keitark/reuse-android/releases/download/v0.1.0-preview1/reuse-phone-demo.apk).
Compare its SHA-256 with `SHA256SUMS` from that same release before installing.
The build is debug-signed
for research evaluation, not a Play Store release. Install it through Android's
package installer, or with an authorized ADB connection:

```powershell
adb install -r reuse-phone-demo.apk
```

1. Open **RE-USE Library Demo** and choose two or four processing threads. Start
   with the default four; two can perform better on some processors.
2. Press **Start recording** and grant microphone access. Model loading and
   warm-up happen before the microphone opens.
3. Speak near the phone. Keep this screen visible. Recording stops after
   60 seconds, when you press **Stop & save**, or when you leave the screen.
4. Wait for processing to finish draining. Use **Play before RE-USE** and
   **Play after RE-USE** to listen to the completed recordings. Playback and recording are mutually
   exclusive; there is no live speaker monitoring.
5. Use **Share both WAV files** if you want to export the pair. Android asks you
   to choose the destination. The app does not upload recordings automatically.

Microphone access can be denied and retried. If another app or Android privacy
settings prevent capture, the demo reports an error. “Before” means the PCM
delivered by Android before RE-USE; it does not imply access to untouched
microphone hardware channels or bypass the phone's audio processing.

## Requirements and scope

| Item | Current demo |
|---|---|
| Android version | Android 8.0 / API 26 or newer |
| Native architectures | `arm64-v8a` and `x86_64`; no 32-bit ARM build |
| Input | Android `AudioRecord`, 16 kHz, mono, PCM16 |
| Enhancement | CPU, 8 kHz streaming model, five-hop / 100 ms calls |
| Output | Mono 16 kHz PCM16 WAV |
| Permission | `RECORD_AUDIO`, requested when starting a recording |
| Runtime network | No Internet permission or audio server |
| Operation | Visible-screen demo; no background microphone service |
| Capture limit | 60 seconds per recording |

The controller prefers the built-in microphone and records the actual route
when Android makes it available. The requested recording source is
`VOICE_RECOGNITION`, with a `MIC` initialization fallback. Phone and Android
audio behavior can still differ by device. An emulator can check installation
and lifecycle behavior; it cannot establish performance on a physical phone.

The 16 kHz input/output uses the shared exact 95-tap FIR resampler around the
8 kHz model. This demo does not replace Android's system microphone or filter
audio in other apps.

## When processing cannot keep up

The status display reports estimated processing speed and queue delay while a
capture is active. Final statistics are authoritative for the completed run.
RTF is processing time divided by the corresponding audio duration: a value
below `1.0` means average processing was faster than incoming audio. It is not
an end-to-end latency measurement.

An output timing trace records when enhanced PCM callbacks became available
to the application. Saved WAVs contain the sample sequence, not wall-clock
waiting time. Listening to a completed WAV therefore cannot establish whether
live speaker playback would have stalled. This demo does not measure
`AudioTrack` underruns or provide a speaker playout latency result.

- A short call over 100 ms can use the bounded input buffer. Recording may
  continue while the worker catches up.
- Sustained overload makes the queue grow. The display reports catching up
  or falling behind; it does not treat an open microphone as proof of real-time
  performance.
- A full 500 ms input queue stops the session with an explicit overflow error.
  The microphone is closed. A failed or partial recording is not a successful
  real-time result.

After overload, let the device cool, close competing workloads, and compare
the other thread setting in a new capture. There is no automatic bypass,
hidden frame dropping, or state reset to conceal missed work. Passing a short
capture does not guarantee indefinite operation under thermal throttling.

The earlier live G2 companion tests on the Xperia are separate measurements.
For example, one G2 capture measured RTF `0.7905` with `106/544` model calls
over 100 ms. Those numbers must not be presented as a result from this
phone-microphone app. See [the G2 route and its limits](live-g2.md).

## Saved recordings and diagnostics

Each session has its own directory under app-private
`files/captures/<session>/`:

- `raw.wav`: received Android microphone PCM before RE-USE.
- `enhanced.wav`: processed audio including the complete finite FIR tail.
- `stats.json`: enhancement counts, timing, queue, flush, and error information.
- `phone-capture.json`: requested/actual capture configuration, route, read
  counts, and stop reason.

A successful nonempty input of `N` samples produces
`N + (N % 2) + 188` output samples. The extra samples cover odd-sample padding
and the finite FIR drain. Model warm-up is not included in the recording.
Ordinary before/after playback uses the saved WAVs without separate gain
normalization.

For device results, record model and APK hashes, thread setting, capture
duration, input/processed/output counts, drops, queue overflows, average RTF,
calls over 100 ms, maximum queue delay, and whether the microphone stopped.
Audio content does not need to be published. Do not use the G2-specific
`audit_live_capture.py` to validate this route; it checks Even Hub metadata.

## Build from source

Use JDK 17, Android SDK platform 35, NDK `26.1.10909125`, and SDK CMake `3.22.1`.
Set `ANDROID_HOME` or create an untracked `local.properties` containing your
SDK's `sdk.dir`. The checked-in wrapper downloads pinned Gradle 8.10.

The phone app bundles this verified model pack at build time:

```text
models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal-channelnorm-frequency/
  model.onnx
  manifest.json
  LICENSE-NVIDIA.txt
```

The required `model.onnx` SHA-256 is:

```text
f950dcfb9321191b2bcb3ab9afa3e100c40e5c64dd1de1d30849be4c1aacce32
```

Model weights are not committed to Git. A released demo APK contains the
verified graph, manifest, and license. Download the APK and its published
SHA-256 from the same research release, then prepare the build assets:

```powershell
python tools\prepare_phone_model.py --apk .\reuse-phone-demo.apk --expected-apk-sha256 'PASTE_APK_SHA256_FROM_RELEASE_HERE'
```

The preparation tool verifies the APK identity and required model identity
before extracting the three model files to the expected directory. To rebuild
the model instead, use the model export and optimization tools in the
[library guide](../README.md#export-the-real-model) and retain the matching
manifest and NVIDIA license. An arbitrary RE-USE
checkpoint, the newer noncausal 9.6M checkpoint, or a graph from an older native
kernel is not a compatible replacement. Application setup verifies the graph
identity before inference.

After preparing those model files:

```powershell
.\gradlew.bat :phone-demo:assembleDebug :phone-demo:lintDebug -PreuseNeonFma=true -PreuseScanN16Registers=true
```

The APK is written to
`android/phone-demo/build/outputs/apk/debug/phone-demo-debug.apk`.
The optional properties select the ARM arithmetic paths used in the project's
performance build. Omitting them retains the earlier arithmetic paths and is
a different build configuration that needs its own timing result.

The module depends on the reusable `:reuse-audio` library, which exposes the
shared `EnhancementSession` and `FirResampler` and depends on the core `:reuse`
library. It does not import application source from the G2 companion or include
its WebSocket server, Hub client, or Bluetooth route. See
[library integration](library.md) to use the pipeline in another application.

Run relevant shared-engine checks after changing the processing path:

```powershell
python -m pytest tests/test_live_session.py tests/test_prepare_phone_model.py
.\gradlew.bat :live-companion:testDebugUnitTest -PreuseNeonFma=true -PreuseScanN16Registers=true
```

These tests check bounded queues, DSP, finite tails, and existing integration
behavior. A new physical phone still needs permission, microphone, stop/drain,
playback, lifecycle, and sustained-performance checks.

## Physical phone validation

Two user-started recordings exercised the installed phone demo on a physical
Xperia 1 IV. Both used the actual built-in microphone, `VOICE_RECOGNITION`,
16 kHz mono PCM16, and four processing threads. All received input samples were
processed. Each recording completed one model flush and released the
microphone.

| Measurement | First recording | Second recording |
|---|---:|---:|
| Input duration | 9.62 s | 5.70 s |
| Input / processed samples | 153,920 / 153,920 | 91,200 / 91,200 |
| Output samples, including finite tail | 154,108 | 91,388 |
| Model calls | 97 | 58 |
| Mean model call | 55.3217 ms | 56.7985 ms |
| p95 model call | 65.2809 ms | 67.2082 ms |
| Complete processing RTF | 0.576880 | 0.596971 |
| Calls over 100 ms | 0 | 0 |
| Application input drops | 0 | 0 |
| Queue overflows | 0 | 0 |
| Maximum queued input | 800 samples / 50 ms | 800 samples / 50 ms |
| Maximum observed queue age | 13.593 ms | 11.322 ms |

The [phone-specific validation receipt](validation/phone-demo-xperia-initial.json)
records artifact identity and validation scope. These results establish two
short foreground phone-microphone captures, separately from the earlier G2
route. They do not establish indefinite thermal stability, background capture,
other phones' performance, or speaker playout timing.

The shared-path checks passed 25 host cases: 10 JNI/session cases, including a
concurrent statistics snapshot regression, and 15 model-preparation cases.
Twelve G2 companion JVM tests also passed after the shared-library change.
The phone APK assembled and Android lint completed successfully. Host tests do
not substitute for the microphone checks above.

## License

Java integration and UI code are MIT. The streaming checkpoint, adapted
model inference implementation, and model-specific native kernels are covered
by NVIDIA NSCLv1 for noncommercial research and education. This is an
unofficial implementation, not an NVIDIA product. See
[third-party notices](../THIRD_PARTY_NOTICES.md) and the bundled license.
