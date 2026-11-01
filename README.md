# RE-USE Android

[![Android 8.0+](https://img.shields.io/badge/Android-8.0%2B-3DDC84?logo=android&logoColor=white)](docs/phone-demo.md#requirements-and-scope)
[![64-bit ARM and x86](https://img.shields.io/badge/ABI-arm64--v8a%20%7C%20x86__64-blue)](docs/phone-demo.md#requirements-and-scope)
[![Preview release](https://img.shields.io/badge/release-v0.1.0--preview1-blue)](https://github.com/Keitark/reuse-android/releases/tag/v0.1.0-preview1)
[![Noncommercial research](https://img.shields.io/badge/use-noncommercial%20research-orange)](docs/license.md)

A phone-microphone demo and Android library for on-device speech enhancement
using NVIDIA's **Real-time RE-USE** streaming checkpoint. The phone demo uses
the ordinary Android microphone: it needs no G2 glasses, Even app, account,
Bluetooth connection, or audio server.

## Phone microphone demo

Open **RE-USE Library Demo**, allow microphone access, and press **Start recording**.
Speak, press **Stop & save**, then compare **Play before RE-USE** and
**Play after RE-USE**. Captures stop automatically after 60 seconds or when you
leave the recording screen.
The app can share the completed WAV pair through Android's share sheet.

The demo targets Android 8.0+ and 64-bit ARM phones. An x86_64 build is also
included for development. It processes audio on the CPU and offers two- and
four-thread settings. Audio stays in app-private storage unless you choose
to share it; the phone demo has no Internet permission.

[Download the v0.1.0-preview1 phone demo APK](https://github.com/Keitark/reuse-android/releases/download/v0.1.0-preview1/reuse-phone-demo.apk)
and verify it against the release's `SHA256SUMS`. This is a debug-signed research
preview. See the [phone demo guide](docs/phone-demo.md) for installation,
building, microphone behavior, and timing diagnostics.

Two user-started recordings on a physical Xperia 1 IV used its built-in
microphone: 9.62 seconds at RTF `0.576880` and 5.70 seconds at RTF `0.596971`.
Both processed all received samples, with no application drops, queue
overflows, or calls over 100 ms. These are short foreground tests; sustained
thermal behavior and speaker playout timing have not been established.
See the [phone-specific validation receipt](docs/validation/phone-demo-xperia-initial.json).

### What “real-time” means here

The app processes microphone input while recording. Whether a phone keeps up
depends on its CPU, thermal state, and thread setting. The UI reports estimated
processing speed and buffered audio; the saved statistics record the final
result. A short processing overrun can be buffered. If the bounded 500 ms input
queue fills, the app stops with an error instead of silently dropping audio or
reporting a successful capture. No universal low-latency guarantee is claimed.

### Documentation

- [Phone demo: install, run, build, and diagnose overload](docs/phone-demo.md)
- [Reusable libraries and PCM-stream integration](docs/library.md)
- [Optional Even G2 companion and measured live test](docs/live-g2.md)
- [Validation scope](docs/validation.md)
- [Licensing and third-party notices](THIRD_PARTY_NOTICES.md)
- [License boundaries and redistribution](docs/license.md)
- [Contributing and reporting device results](CONTRIBUTING.md)

The Java integration is MIT. The model, adapted inference code, and
model-specific native kernels use NVIDIA NSCLv1 for noncommercial research and
education. This is an unofficial research implementation.

| Module | Responsibility |
|---|---|
| `:reuse` | Core model, stateful STFT, and native ONNX operations; reusable AAR |
| `:reuse-audio` | Bounded 16 kHz PCM session, FIR resampling, recording, and statistics; reusable AAR |
| `:phone-demo` | Android microphone, permission, buttons, playback, and sharing |
| `:live-companion` | Optional Even G2 / Hub transport application |

Both applications consume the shared libraries. The libraries have no
dependency on the phone demo, Even app, or G2 transport.

## Library and earlier G2 research

An Android AAR for stateful, on-device speech enhancement using NVIDIA's
**Real-time RE-USE** checkpoint. This is a functional research prototype.
A separate companion now processes a real 54.5-second G2 microphone capture on
the Xperia: all 872,000 input samples, no reported app/transport drops, RTF
0.759731, with 67 of 546 calls over 100 ms. Both raw and enhanced PCM recordings
were saved and the microphone was closed. See [live G2 results](docs/live-g2.md).
The earlier recorded 60-second codec-probe stream passes with zero output gaps,
queue resets or call misses; it is a separate benchmark. Longer/background
operation and stock-app stop-tail handling remain open. The 20 ms input hops
alone do not imply that inference meets a 20 ms deadline.

The library accepts mono float PCM, keeps neural and DSP history across calls,
and performs actual streaming STFT, enhancement and overlap-add reconstruction.
It does not require CUDA, a server, microphone permission, or Internet access at
runtime. Recording/playback belongs to the integrating application.

## Included

- Java API usable from Kotlin; Android API 26+.
- CPU ONNX Runtime 1.23.2, fixed 8 kHz or 16 kHz graphs.
- Three Mamba blocks by default (1,823,555 active parameters), zero look-ahead.
  Export can select exit layers 3–12 for a different quality/cost trade-off.
- Fixed 40 ms periodic-Hann window, 20 ms hop, exact stream flush/reset behavior.
- Explicit recurrent states in persistent direct buffers with pinned outputs.
- Optional fused native frequency scan and temporal update (ARM64 NEON /
  x86_64 SSE2) to reduce intermediate memory traffic. The AAR contains both
  native ABIs. It supports SelectiveScan, SelectiveStep, SelectiveSequence,
  FastSoftplus, TemporalConvSilu, ChannelNormPRelu and FrequencyConvSilu;
  each model manifest selects the required operations.
- A five-hop graph and `ChunkedReuseEnhancer` for 100 ms processing calls.
  It preserves the trained weights and causal state transitions while reducing
  repeated graph work. Controlled host measurements show 27.9–37.6% lower
  computing time per audio hop. The earlier five-hop Step graph's four-thread Xperia probe
  keeps up with a 20-second paced recorded stream without output gaps or resets;
  60-second probe runs still have gaps and queue resets.
- A separate five-hop `sequence-pad-softplus` pack folds 15 zero-padding nodes
  into convolutions and replaces nine Softplus nodes with a tested SIMD kernel.
  A matched four-thread host comparison shows **11.45% further computing-time
  reduction** over the previous Sequence graph, with unchanged weights, state
  shapes and buffering. The Even interposer additionally reuses frame storage
  and halves steady resampler multiply-accumulates. See [optimization details](docs/optimization.md).
- The optional `sequence-pad-softplus-temporal` graph fuses three temporal
  convolution/history/SiLU paths. Together with explicit ARM NEON FMA and
  register-held frequency state, a reversed-order phone comparison measures
  49.19 -> 40.07 ms per 100 ms call (**18.55% lower model time**).
  This is a native-shell diagnostic, separate from the paced pipeline test:
  that build's minute completes all 3,000 hops with no reset but two output gaps.
  See [opcode measurements and build flags](docs/opcode-optimization.md).
- The latest `sequence-pad-softplus-temporal-channelnorm-frequency` pack
  additionally fuses 18 normalization/activation/layout regions and six
  frequency-convolution regions. A matched Xperia model-only comparison shows
  **5.51% further mean compute reduction** against the previous temporal pack.
  The separate paced minute processes all 3,000 hops at RTF 0.550647 with
  zero gaps, resets or call misses. This is bounded recorded-audio acceptance,
  with a nominal 200 ms startup and an undelivered native stop tail.
  See [compute fusion and validation scope](docs/compute-fusion.md).
- Hash-locked model export, independent NVIDIA-topology parity validation,
  DSP tests, and an Android correctness/performance runner.

**Terms:** the streaming checkpoint and model-specific native kernel are NVIDIA
NSCLv1, for noncommercial research and education; Java integration code is MIT.
Model weights are separate from the AAR. The commercial
9.6M NemotronLabs model discussed in the announcement is noncausal and is not
interchangeable with this streaming graph. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Build

Use JDK 17, Android SDK platform 35, NDK 26.1.10909125 and SDK CMake 3.22.1.
Set `ANDROID_HOME` or create
`local.properties` with your SDK's `sdk.dir`. The checked-in Gradle wrapper
downloads the pinned Gradle 8.10 distribution.

```powershell
.\gradlew.bat :reuse:testDebugUnitTest :reuse:lintRelease :reuse:assembleRelease
.\gradlew.bat :reuse:publishReleasePublicationToMavenRepository
```

The packaged opcode AAR was built with both optional properties enabled:

```powershell
.\gradlew.bat :reuse:assembleRelease :reuse:publishReleasePublicationToMavenRepository -PreuseNeonFma=true -PreuseScanN16Registers=true
```

Omitting these properties retains the earlier arithmetic paths. The fusion
graph requires this version's native library; it is independent of those flags.

The AAR is `android/reuse/build/outputs/aar/reuse-release.aar`. The local Maven
repository is `build/repository`. Its POM includes the required ONNX Runtime
dependency. To consume a copied AAR directly, also declare:

```kotlin
implementation(files("libs/reuse-release.aar"))
implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.2")
```

## Export the real model

Python 3.11 was used for validation. GPU packages are unnecessary.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe tools\fetch_model.py --reference
.\.venv\Scripts\python.exe tools\validate_upstream.py --output .research\upstream-parity.json
.\.venv\Scripts\python.exe tools\export_model.py --sample-rate 16000 --exit-layer 3 --output models\reuse-16k-exit3
```

For the faster native graph, build the host validation plugin, then export:

```powershell
pwsh -NoProfile -File tools\build-native.ps1
.\.venv\Scripts\python.exe tools\export_model.py --sample-rate 8000 --exit-layer 3 --scan-mode native --fuse-time-step --custom-op-library native\build-windows\Release\reuse_ort_ops.dll --output models\reuse-8k-exit3-native-step
```

Change `--sample-rate` to 16000 for a wider-band model at higher processing cost.
The host DLL validates the export; Android uses the corresponding `.so` already
included in the AAR. Do not copy a Windows DLL to Android.

For the five-hop throughput variant, export a separate graph:

```powershell
.\.venv\Scripts\python.exe tools\export_model.py --sample-rate 8000 --exit-layer 3 --scan-mode native --fuse-time-step --frames-per-call 5 --validation-frames 40 --custom-op-library native\build-windows\Release\reuse_ort_ops.dll --output models\reuse-8k-exit3-native-chunk5
```

To fuse the five temporal updates within each layer as well, add
`--fuse-time-sequence` and use the separate output directory
`models\reuse-8k-exit3-native-chunk5-sequence`. That graph requires the latest
AAR's SelectiveSequence operation and uses the same `ChunkedReuseEnhancer` API.

The previous padding/activation pack is
`models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus` (graph SHA256
`5c979c328bde1e3c06735588055c7c1289447d6fa23821f52f053bb8dd30bc56`).
It requires this release's FastSoftplus operation. The pad-only pack remains
available as `models/reuse-8k-exit3-native-chunk5-sequence-pad`.
Both use the same chunked API; older AARs cannot load FastSoftplus packs.

The temporal pack adds `-temporal` to that directory name (graph SHA256
`ab02d6cca7c923751651032b6151c022a8e4b8f560790e8beba3367475b9653a`).
It requires the bundled `TemporalConvSilu` operation and retains the same
weights, 21 state shapes, five-hop API and buffering.

The newest pack is
`models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal-channelnorm-frequency`
(graph SHA256 `f950dcfb9321191b2bcb3ab9afa3e100c40e5c64dd1de1d30849be4c1aacce32`).
It requires this release's ChannelNormPRelu and FrequencyConvSilu operations.
The normalization-only pack is also included. Both preserve the same weights,
state shapes and chunked API. The actual Java API passes neural/DSP,
reset/flush/count and lifecycle checks on the physical Xperia with ORT 1.23.2;
the minute-long codec probe separately uses the original Even ORT 1.27.0.

This graph processes five consecutive 20 ms feature frames together. It is not
five independent streams or a loop around the one-hop graph. Its manifest
declares `frames_per_call=5`; use the matching chunked Java API below. The
optional `reuse-8k-exit3-native-compact` pack retains the one-hop API and removes
unused convolution-history work. Details: [computing-time changes](docs/optimization.md).

The model directory contains `model.onnx`, `manifest.json`, and the NVIDIA
license. Move those files into app-private storage during app setup. Model
loading validates the graph SHA256, configuration, tensor names, types and shapes.
Use a separately exported graph and matching manifest for 8 kHz; do not merely
change the Android sample rate.

## Use from Kotlin

```kotlin
val config = ReuseConfig(8000, 3, 2) // sample rate, exit layer, CPU threads
ReuseEnhancer.create(context, modelFile.toPath(), manifestFile.toPath(), config).use { enhancer ->
    // Run on a dedicated worker thread. Every 8 kHz input has exactly 160 samples.
    for (pcmHop: FloatArray in inputHops) {
        enhancer.process(pcmHop)?.let { enhancedHop -> sink.write(enhancedHop) }
    }
    enhancer.flush()?.let { enhancedHop -> sink.write(enhancedHop) }
}
```

Native graphs require an extracted native library path. Add this to the
consuming application's Gradle Kotlin configuration:

```kotlin
android { packaging { jniLibs { useLegacyPackaging = true } } }
```

`create(context,...)` resolves the bundled custom-operation library from
`applicationInfo.nativeLibraryDir`. The three-argument constructor remains
available for portable graphs; the four-argument constructor accepts an explicit
custom-operation library path. There is no automatic fallback to another model.

PCM normally lies in `[-1, 1]`; the library does not clip or apply automatic
gain control. It returns `null` on the first call. The next call returns the
first input hop's enhanced samples. First output requires **40 ms of input plus
computation**; steady output has **20 ms of buffered audio delay plus
computation**. Calling `flush()` retrieves the last hop. N input hops yield
exactly N output hops. Zero-pad a partial final input hop, then trim the returned
audio to the original sample count.

After flushing, call `reset()` before beginning another stream. One enhancer
owns one stream. Close it to release native tensors and its session. Invalid
PCM is rejected; inference failures require a reset. Do not call inference from
an audio callback or UI thread. Use a bounded worker queue; if processing misses
its deadline, report the overload or restart the stream rather than allowing an
ever-growing audio backlog. Do not silently skip a hop without resetting state.

## Use 100 ms processing calls

```kotlin
val config = ReuseConfig(8000, 3, 4)
ChunkedReuseEnhancer.create(context, chunkModel.toPath(), chunkManifest.toPath(), config).use { enhancer ->
    // Exactly 800 samples per call at 8 kHz; collect them on a bounded worker queue.
    for (pcmChunk: FloatArray in inputChunks) {
        val enhanced = enhancer.process(pcmChunk)
        sink.write(enhanced) // first call: 640 samples; later calls: 800
    }
    enhancer.flush()?.let { sink.write(it) } // final pending 160 samples
}
```

`inputChunkSamples()` returns the required input length. The first call collects
100 ms and returns 80 ms after inference; later calls return 100 ms, and `flush()`
returns the final 20 ms. Total output length equals total input length. Pad a
partial final 100 ms chunk with zeros, then trim all returned audio to the
original sample count. Reset before another stream and close resources after use.

Chunking adds up to **80 ms of collection delay** to the ordinary overlap-add
delay. It preserves zero temporal model look-ahead, but increases end-to-end
buffering. A 100 ms call must average less than 100 ms to keep up continuously;
individual internal hops no longer have a separate 20 ms dispatch deadline.
The one-hop and five-hop manifests are deliberately rejected by the wrong API.

## Validate on Android

```powershell
.\gradlew.bat :benchmark:assembleDebug :benchmark:assembleDebugAndroidTest
.\.venv\Scripts\python.exe tools\enhance_audio.py --model-dir models\reuse-16k-exit3 --input .research\upstream\noisy_audio\mic_test.wav --output .research\enhanced.wav --fixture .research\fixture-16k --frames 100
.\scripts\benchmark-android.ps1 -Serial emulator-5554 -ModelDir models\reuse-16k-exit3 -FixtureDir .research\fixture-16k
```

Replace the serial with a connected phone's serial for meaningful ARM64 speed
measurements. The runner reports PCM parity, output length, lifecycle behavior,
hop p50/p95/p99/max, deadline misses, RTF and device metadata. It exercises the
real graph, not a bypass. Emulator timing is not physical-phone evidence.

Python tests: `.\.venv\Scripts\python.exe -m pytest tests -q`.
Detailed design and validation boundaries: [docs/architecture.md](docs/architecture.md).
Measured results and remaining work: [docs/validation.md](docs/validation.md).
Optimization measurements and their latency trade-off: [docs/optimization.md](docs/optimization.md).

## Even G2 / Xperia 1 IV experiment

The current supported route uses the official Even Hub glasses microphone and
the separate `io.github.reuse.live` Android companion. Its real 54.5-second
capture saved 872,000 raw PCM samples and 872,188 enhanced samples, including
the explicit FIR tail. All input was processed without reported app drops;
average RTF was 0.759731, while 67/546 model calls exceeded 100 ms. Independent
whole-waveform parity passes within one PCM16 step. The microphone was closed
and the companion stopped after capture. Source timestamps are unavailable,
so this is not a BLE-loss or all-deadline guarantee. Build/run instructions and
the exact scope are in [docs/live-g2.md](docs/live-g2.md). The official Even app
and its data remain unchanged.

The earlier native APK interposer experiment targets
the Even app's LC3 decoder boundary, before Dart receives the PCM. It
preserves the app's existing ONNX Runtime, Dart code, and backend/authentication
logic. The fast mode downsamples to 8 kHz for RE-USE, then returns 16 kHz PCM;
this limits useful audio bandwidth to below 4 kHz. A full 16 kHz mode is also
available for measurement. See [docs/even-g2.md](docs/even-g2.md) for the exact
version, experimental scope and validation record.

The earlier one-hop ARM64 probe ran on a physical Xperia 1 IV (Sony SO-51C,
Android 14). Its representative 20-second run averaged 22.15 ms per 20 ms hop,
with 221 output gaps/five queue resets. Offline correctness passed. That failure
predates the new compact and chunked graphs; it is not their phone benchmark.

The earlier frozen `dist/even-g2-reuse-probe-chunk5.apk` ran on the Xperia with
the original Even codec/ORT runtime: its 20-second four-thread test processes
all 1,000 audio hops in 200 graph calls, averaging **63.99 ms per 100 ms call**,
with no post-startup output gaps, queue resets, call deadline misses or fallback.
It inserts 250 ms of nominal startup silence in that run. This is a bounded
recorded-audio pipeline test, not live G2 or microphone-to-ASR latency acceptance.
The first 60-second run fails with 37 output gaps and one reset. A second probe
revision removes recording-file writes from the timed input loop and measures
packet lateness; its 60-second run also fails (96 gaps/two resets), while all
packet bursts arrive within 10 ms of their scheduled time. Average computing
time alone is insufficient; longer operation remains open.

The further SelectiveSequence variant passes a cold two-second check, but its
one-minute tests also fail: at four threads it has 148 gaps/four resets; at two
threads it has 514 gaps/16 resets. Those builds were **not ready for live G2
use**; none passed a one-minute test. Later recorded-audio and Hub-companion
results above are separate acceptance stages.

The earlier frozen `dist/even-g2-reuse-probe-otherareas.apk` tests the combined
padding/Softplus graph, preallocated worker buffers and polyphase FIR path.
Its four-thread 20-second `default` and `dynamic4` trials process all 200 calls
with zero post-startup gaps or resets. One `bounded_spin` trial has two output
gaps; its other trial passes. The ABBA timings drift and show no clear benefit.
A 60-second `dynamic4` trial completes
all 600 calls/3,000 audio hops with no queue reset, but has **32 output gaps**
and 264 call deadline misses at RTF 0.836540. It fails continuity. These are
recorded-audio tests; they do not establish live G2 latency, quality, battery
life or stock microphone-stop tail handling.

`dist/even-g2-reuse-probe-compact.apk` remains an optional one-hop alternative
whose physical-phone speed is unmeasured. All included variants pass their host
correctness checks; actual ARM64 phone replay also passes within one PCM16 step.
The public Java Android runtime receipt covers the earlier five-hop Step graph;
the new Sequence graph has host, native-phone and updated manifest unit checks.
The native hook still needs reliable stock session-end tail handling and actual
G2 interception testing. Those interposer benchmarks installed a separate
probe; the current live Hub route installs a different companion and explicitly
drains its finite tail. Neither replaces the official Even app or its data.
