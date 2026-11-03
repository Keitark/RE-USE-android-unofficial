# Experimental GPU microphone demo

This preview adds GPU offload to the ordinary phone-microphone demo. It uses
a narrowly patched ONNX Runtime 1.23.2 NNAPI backend that selects only devices
reported by Android as GPUs. Supported graph partitions can run through that
GPU driver; the model's 45 custom CPU operators, including its Mamba selective
scans, remain on ONNX Runtime CPU. This is **hybrid GPU/CPU execution**, not a
complete GPU port.

The experimental app is separate from the existing CPU demo:

| Item | GPU preview |
|---|---|
| Launcher | RE-USE GPU Preview (Unofficial) |
| Package | `io.github.reuse.phone.gpu` |
| Version | `0.2.0-gpu-preview2` |
| Android | Android 10 / API 29 or newer |
| Architecture | `arm64-v8a` only |
| Input / saved output | 16 kHz mono PCM16 |
| Enhancement model | Existing 8 kHz, exit-3, zero-look-ahead FP32 graph |
| Processing calls | Five chronological 20 ms hops / 100 ms |
| Modes | CPU, GPU with FP32, GPU with optional FP16 relaxation |
| Runtime network | None; audio stays on the phone unless shared |

The stable CPU APK and its recordings are preserved. Installing the GPU
preview does not update or replace `io.github.reuse.phone`.
Preview 2 updates the existing GPU-preview package in place and keeps its
saved files. It selects CPU by default and enables the GPU choices only after
a background device check finds an NNAPI GPU-type driver. A slow driver check
uses a separate worker and does not hold up CPU model preparation.

A user-provided Xperia screenshot from preview 1 reported no NNAPI GPU driver,
before microphone capture began. Preview 2 makes that unsupported route clear
and leaves CPU recording available. This change does not add a different GPU
backend or establish successful GPU execution on that phone.

## Validation scope

The preview was built without access to a physical phone. Android
compilation, host tests, or successful installation do not establish actual
GPU compatibility, audio quality, speedup, or sustained real-time performance.
There are no measured GPU performance results yet. The earlier Xperia CPU
measurements in the [phone demo guide](phone-demo.md#physical-phone-validation)
are CPU results and must not be reused as GPU results.

The [preview 2 build receipt](validation/gpu-preview2-build.json) records APK
assembly, signature and payload checks, 32 core JVM tests and 5 phone-demo JVM
regressions for driver classification and unsupported-mode presentation. The
[original build receipt](validation/gpu-preview-build.json) retains the shared
JNI/session and GPU runtime selection checks. These are build/host checks,
not successful GPU device acceptance.

NNAPI was deprecated in Android 15. This is a bounded research experiment to
evaluate GPU offload on existing devices, not a commitment to NNAPI as the
library's long-term GPU architecture. See the [Android NNAPI documentation](https://developer.android.com/ndk/guides/neuralnetworks).
A dedicated GPU runtime, such as the [QNN GPU backend](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html#running-a-model-with-qnn-eps-gpu-backend),
would require separate compatibility and model-port work.

## Install and compare

Use the GPU preview APK and `SHA256SUMS` from the same
[GPU research release](https://github.com/Keitark/RE-USE-android-unofficial/releases/tag/v0.2.0-gpu-preview2).
The ordinary `reuse-phone-demo.apk` from the existing CPU release does not
include this experimental runtime. Verify the published checksum before
installing. The preview is debug-signed for research evaluation.

1. Open **RE-USE GPU Preview (Unofficial)**. Start with CPU mode and record a
   short comparison to check microphone permission, capture, stop, and playback.
2. If the GPU choices become enabled, select GPU FP32 and press **Start recording**. GPU compilation,
   warm-up, and accelerator verification happen before the microphone opens.
   Initial preparation may take longer than CPU model loading.
   If the app reports GPU unavailable through NNAPI, keep CPU selected. FP16
   uses the same driver route and cannot fix a missing GPU-type device.
3. Keep the screen visible and speak near the phone. Press **Stop & save** and
   wait for accepted audio to finish processing. Recording also stops after
   60 seconds or when the app leaves the screen.
4. Listen to **both WAV files**. Check that enhancement preserves speech and
   does not introduce clicks, missing speech, unstable levels, or other artifacts.
5. Compare CPU and GPU runs using the same thread setting and recording
   conditions. Repeat in alternating order; phone temperature and competing
   workloads can change the result. Extend to a complete 60-second capture
   only after the short test succeeds.
6. Try FP16 relaxation separately. It permits lower-precision computation in
   supported NNAPI partitions; it does not quantize the full model to INT8.
   FP16 audio quality and performance remain unverified and may differ by driver.

Two or four CPU threads still affect the CPU portions of the hybrid graph.
GPU mode is not automatically faster, and a successful short capture does not
establish indefinite operation under thermal throttling.

The app opens no live speaker-monitoring path. Playback uses the saved WAVs
after recording. Saved WAVs concatenate produced PCM; processing waits are not
inserted as silence. A clean saved recording therefore does not prove that
continuous `AudioTrack` playback would have avoided underruns.

## How GPU selection is checked

The generic library backend is named `NNAPI_ACCELERATOR`. In the standard
ONNX Runtime runtime, `CPU_DISABLED` excludes NNAPI CPU devices but can still
select a GPU or another accelerator. That setting alone is not evidence of
GPU execution. See the [upstream NNAPI provider options](https://onnxruntime.ai/docs/execution-providers/NNAPI-ExecutionProvider.html#configuration-options).

The separate GPU preview supplies a pinned, patched runtime:

- Its device filter accepts only Android NNAPI devices with GPU type when
  `CPU_DISABLED` is requested. It rejects older device-selection APIs and an
  empty GPU list instead of returning to automatic device selection.
- Its build-information string includes `REUSE_NNAPI_GPU_ONLY_V1`. Runtime
  source, patch, and artifact provenance accompany the build; the APK bundles
  `gpu-runtime/runtime-provenance.json`.
- Startup runs three silent model calls and ends an ONNX Runtime profile.
  At least one `NnapiExecutionProvider` kernel event must be present before the
  microphone opens. A failed check stops preparation with an error.
- Unsupported graph operators still run on ORT CPU. The startup check proves
  that some work was delegated to the selected NNAPI GPU driver; it does not
  establish whole-model GPU execution, hardware utilization, a speedup, audio
  quality, or real-time performance.

If GPU initialization, compilation, or verification fails, keep the error
and use CPU mode explicitly for another capture. The app does not silently
relabel an all-CPU run as GPU acceleration.
Use **Share diagnostics (no audio)** to export `stats.json`,
`phone-capture.json`, and the latest `gpu-probe.json` when available. Driver
enumeration is saved without opening the microphone, so the check can be
shared even without a recording. Failed-startup receipts are retained too.
This works without USB debugging and shares only JSON through Android's chooser;
the separate WAV-sharing action remains available after a successful recording.

For a paired, connected wireless ADB session, the PC can keep app-scoped logs:

```powershell
pwsh -NoProfile -File tools/capture-phone-logs.ps1 -Serial DEVICE_IP:ADB_PORT
```

This uses the already-authorized connection; it does not pair the phone or
open its microphone. It captures only this app's UID-filtered logcat and
whitelisted JSON diagnostics, including the silent warm-up profile. Logs stay
under ignored `.research/phone-live-logs/` by default. The recorder stops after
60 minutes or a 50 MiB capture budget; use `-MaxMinutes` or `-MaxLogMiB` to
change those limits. Ctrl+C, or creating `stop.requested` in the reported output
directory, stops its own logcat client. It never clears device logs or copies
WAV/PCM recordings. Tailscale connectivity alone does not enable ADB.

## Overload and diagnostics

The bounded input queue remains 500 ms. A temporary call over 100 ms can use
that buffer while processing catches up. Sustained overload grows the queue;
a full queue stops capture with an explicit overflow error. There is no hidden
frame dropping, passthrough substitution, or state reset to conceal overload.

Each recording retains `raw.wav`, `enhanced.wav`, `stats.json`, and
`phone-capture.json` under app-private `files/captures/<session>/`. Retain the
requested backend, FP16 setting, runtime build information, startup NNAPI
profile evidence, APK/model hashes, thread count, duration, input/processed
counts, drops, overflow count, model-call distribution, processing RTF, queue
delay, and microphone release status when reporting a result.

RTF below `1.0` means average processing was faster than the represented
audio duration. It is not an end-to-end latency measurement and does not
establish that every 100 ms deadline was met. GPU startup and compilation
costs should be reported separately from steady processing. No voice content
needs to be published; the [contribution guide](../CONTRIBUTING.md) describes
the ordinary phone-result reporting boundaries.

## Build from source

Prepare the same hash-verified F950 model assets as the
[ordinary phone demo](phone-demo.md#build-from-source). Use its JDK 17,
Android SDK 35, NDK `26.1.10909125`, and SDK CMake `3.22.1` setup for the app.
The separate ONNX Runtime source build needs a newer isolated CMake toolchain
as described by its build script.

Build the patched runtime first:

```powershell
.\tools\build-ort-gpu.ps1
```

The script pins the original ORT source and applies the documented GPU-device
filter patch. Its ignored local Maven repository contains:

```text
com.microsoft.onnxruntime:onnxruntime-android-gpuonly:1.23.2-reuse-gpuonly1
```

The experimental AAR retains the same-version Java/JNI interface and required
notices, packages the patched ARM64 runtime, and excludes other ABI runtimes
that would not have the GPU-only patch. Preserve the runtime provenance
receipt when redistributing the APK.

Then build the separate preview:

```powershell
.\gradlew.bat :phone-demo:assembleDebug :phone-demo:lintDebug -PreuseGpuExperimental=true -PreuseNeonFma=true -PreuseScanN16Registers=true
```

The APK is written to
`android/phone-demo/build/outputs/apk/debug/phone-demo-debug.apk`. Copy it to a
distinct GPU-preview filename before distributing it. Omitting
`-PreuseGpuExperimental=true` selects the standard runtime and ordinary CPU
application identity. Build the normal and experimental configurations
separately; do not replace a published stable asset with an experimental APK.

The reusable core and PCM-session APIs remain separate from the phone UI.
See the [library guide](library.md) for ownership and lifecycle contracts.
Use the generic NNAPI backend name in an application using an unmodified ORT
runtime; the GPU-specific claim belongs only to the pinned GPU-preview runtime.

## License

This is unofficial and is not affiliated with or endorsed by NVIDIA.
The NVIDIA streaming checkpoint, adapted model code, and model-specific native
kernels retain NSCLv1's noncommercial research and education restriction.
Independent integration/UI code and ONNX Runtime are MIT with their stated
exceptions and third-party notices. Changing the execution backend does not
change the checkpoint's use restrictions. Include the full licenses, notices,
runtime patch, and provenance with a research distribution; see
[license boundaries](license.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
