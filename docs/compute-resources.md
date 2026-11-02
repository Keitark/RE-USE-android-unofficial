# Compute-resource comparison APK

This test app compares the resources actually accessible to an Android app:
CPU threading, direct Adreno GPU access, and Qualcomm HTP/NPU modes. It does not
assume that using an accelerator, more threads, or every processor simultaneously
will reduce processing time.

**This APK is built for later device testing. No new Xperia, Adreno, or Vulkan
execution or speed result is claimed by the build.** The earlier
[eight-convolution measurements](qnn-convolution-experiments.md) remain separate.

[Download the test APK and checksums](https://github.com/Keitark/RE-USE-android-unofficial/releases/tag/v0.5.0-resource-probe1).
The launcher is **RE-USE Compute Test (Unofficial)**, version
`0.5.0-resource-probe1`, package `io.github.reuse.qnnprobe`. This updates the
earlier benchmark app. It is a test application, not a microphone recorder.

## Run on the phone

1. Install `reuse-resource-probe.apk` after comparing its SHA-256 with the
   release's `SHA256SUMS`. Android 10+ / ARM64 is required. Qualcomm GPU/HTP
   compatibility depends on the device and driver; CPU trials are independent.
2. Open the app. **Inspect available resources** records CPU capabilities,
   optional CPU clock information, declared graphics features, and OpenCL GPU
   discovery. It does not execute model inference or GPU kernels.
3. Press **Compare CPU / Adreno / NPU**. Keep the phone unlocked and the app
   visible for several minutes. It uses only its bundled public fixtures.
4. **Stop after current call**, or leaving the app, requests cancellation.
   Native compilation, driver calls and inference cannot be forcibly interrupted;
   a stop takes effect when control returns. Partial diagnostics remain available.
5. Press **Share diagnostics (no audio)** to export the JSON reports and bounded
   GPU backend logs. Sharing is user initiated. Nothing is uploaded automatically.

The APK has no microphone or Internet permission. Its saved WAV/float files are
generated from public NVIDIA test audio, not from the microphone. The diagnostic
sharing action does not include those audio files.

## What is compared

| Test | Evidence / purpose |
| --- | --- |
| CPU inventory | Available processor count, CPU feature flags and optional sysfs frequency information; no serial identifiers |
| OpenCL inventory | Driver-reported GPU name, vendor, compute units and memory; discovery is not execution proof |
| Vulkan compute smoke | A small deterministic FP32 shader, checked against exact CPU values; independent of QNN and OpenCL |
| Full-model CPU | Original FP32 and eight-Conv QDQ graphs with 1, 2, 3, 4, 6 and 8 ORT threads, each repeated in mirrored order |
| QNN HTP | Balanced and burst modes with two CPU threads, after all eight original isolated numerical prerequisites pass |
| Direct QNN GPU | FP32 convolution prerequisite with CPU fallback disabled, then the original full FP32 graph with CPU fallback |

The CPU pool order is `1,2,3,4,6,8,8,6,4,3,2,1`. Pool size is not a claim
that those cores are pinned, occupied, or equally fast. Graph, input, runtime and
native-library fingerprints must match before cross-thread results can rank.
Both repetitions must complete and their outputs must meet the fixed numerical
consistency gates. Precision/model choices are reported separately.

Full-model trials preserve 8 kHz, exit 3, five chronological 20 ms hops per
100 ms call, zero neural look-ahead, and all 21 FP32 recurrent states. Each
session profiles three zero-input warm-ups, ends profiling, resets its states,
then processes the chronological public clip and flushes without profiling.
Compilation and warm-up are excluded from steady timing.

The reports include mean/p95/max wall time, 100 ms overruns, process CPU time,
and thermal/power snapshots. Process CPU time covers all threads in the app
process; it is neither exclusive model-kernel time nor a battery-energy
measurement. Backend order, clocks and thermal conditions are not controlled.
The ten-minute total wall budget is cooperative, not a hard driver timeout.

## Adreno paths and verification

The earlier Xperia error concerned **NNAPI GPU discovery**. This test bypasses
NNAPI and requests the pinned QNN runtime's `libQnnGpu.so` directly.
Optional `libOpenCL.so` visibility is declared for Android's native-library
namespace. Runtime dependency versions remain ORT 1.29.0, QNN provider 2.7.0 and
QNN runtime 2.51.0.

QNN 2.7 may expose only an NPU discovery token on Android. When that token is
used to request the explicit GPU backend, the report retains its actual NPU
metadata. It does not relabel the token as a GPU. GPU backend evidence requires:

- Correct outputs under the original FP32 convolution fixture tolerances.
- Strict QNN-only provider execution for that isolated graph, with CPU fallback
  disabled.
- This session's native backend identity log: provider `GPU_QTI_AISW`, backend
  ID `4`. QNN provider event names alone cannot distinguish GPU from HTP.
- Complete, bounded, own-process log capture. Missing, truncated, ambiguous or
  incomplete evidence withholds GPU verification.

Only after that prerequisite passes does the full original FP32 model run.
Its output is compared with a fresh CPU session on the same device. The fixed
whole-wave consistency limits are maximum absolute error `0.001` and normalized
RMSE `0.005`; they are not perceptual-quality acceptance limits. Unsupported
custom Mamba and normalization operators can still run on CPU. The graph is
not presumed to be a complete GPU port.

The separate Vulkan smoke verifies only a tiny compute primitive. It can reveal
an accessible GPU route even when the QNN/OpenCL route is unavailable. Passing
it does not establish a RE-USE backend, model compatibility, or a model RTF.
Driver-reported compute-unit counts also cannot be compared directly across
different GPU architectures.

Relevant primary documentation:

- [Pinned QNN GPU backend and provider configuration](https://github.com/onnxruntime/onnxruntime-qnn/blob/rel-2.7.0/docs/execution_providers/QNN-ExecutionProvider.md#running-a-model-with-qnn-eps-gpu-backend)
- [Qualcomm GPU backend identifier](https://docs.qualcomm.com/nav/home/api-rst_file_include_QNN_GPU_QnnGpuCommon_h.html?product=924033590759186372)
- [Khronos Vulkan compute tutorial](https://docs.vulkan.org/tutorial/latest/11_Compute_Shader.html)

## Interpreting the result

`COMPARISONS_FINISHED` means the test sequence finished; it does not mean every
backend passed. Inspect individual stage outcomes. Unsupported GPU/HTP routes
remain explicit failures while valid CPU results are retained. No configuration
is applied automatically to the microphone demo.

These are unpaced public-input tests, not microphone capture, a live queue test,
speaker playout or general audio-quality validation. Even a fast configuration
needs a paced/sustained test before being selected for the microphone app.
Using CPU, GPU and NPU concurrently would require independent graph regions and
explicit scheduling; simply enabling multiple processors does not provide that.
The independent mask/phase decoder branches are possible later scheduling
targets after compatibility and costs have been measured.

## Build

Prepare the same eight-Conv public asset pack as in the
[convolution experiment build instructions](qnn-convolution-experiments.md).
Use JDK 17, SDK 35, NDK `26.1.10909125` and CMake `3.22.1`:

```powershell
.\gradlew.bat :qnn-probe:testDebugUnitTest :reuse-qnn:testDebugUnitTest :qnn-probe:assembleDebug :qnn-probe:lintDebug -PreuseNeonFma=true -PreuseScanN16Registers=true
```

The NDK's bundled `glslc` compiles the included Vulkan smoke shader during the
build. Vendor runtimes are resolved from Maven; no GPU driver or SDK is installed
on the phone by this test. The old isolated/full test activity remains available
to developers at `io.github.reuse.qnnprobe/.QnnProbeActivity`.

Independent resource-inventory/probe code is MIT. NVIDIA models, model-derived
fixtures and model-specific native code retain NSCLv1 noncommercial restrictions;
Qualcomm dependencies retain their AI Stack terms. Preserve the
[component notices](licenses/qnn/README.md).
