# Qualcomm QNN HTP probe

This separate app tests direct Qualcomm Hexagon NPU access, without NNAPI.
It does not record audio or run the full RE-USE enhancement pipeline. The
existing CPU microphone demo remains the app to use for speech enhancement.

The probe is unofficial and is not affiliated with or endorsed by NVIDIA or
Qualcomm. Its NVIDIA-derived model fixtures are for noncommercial research
and education under NSCLv1.

## Try the APK

[Download the CPU/NPU comparison APK](https://github.com/Keitark/RE-USE-android-unofficial/releases/download/v0.3.0-qnn-probe2/reuse-qnn-probe.apk)
from the [release and checksums](https://github.com/Keitark/RE-USE-android-unofficial/releases/tag/v0.3.0-qnn-probe2).
It is debug-signed and updates probe1 in place, retaining previous diagnostics.
The original [probe1 release](https://github.com/Keitark/RE-USE-android-unofficial/releases/tag/v0.3.0-qnn-probe1)
is preserved.

Install `reuse-qnn-probe.apk` and open **RE-USE QNN Probe (Unofficial)**. Press
**Run NPU test**, keep the screen open, then use **Share diagnostics** to export
the result. The probe has no microphone or Internet permission. It installs
alongside the CPU and GPU preview apps.

Android 10 or newer and ARM64 are required. The initial target is Xperia IV's
Snapdragon 8 Gen 1 / SM8450, whose Hexagon generation is V69. The packaged
runtime includes V69 support. The user-supplied [Xperia probe1 report](validation/qnn-xperia-probe1-user-report.json)
records successful HTP execution of both original graphs on Sony SO-51C,
SM8450, Android 14: the actual RE-USE projection averaged **1.876 ms**, with
**2.093 ms p95**, across 20 timed calls. CPU fallback was disabled and each graph
reported 21 QNN kernel events. This is application-level evidence from the
reported JSON; raw profiles and hardware-counter traces were not supplied.
Other devices, new convolution cases and full-model performance need their own
tests. An emulator cannot establish NPU compatibility or speed.

The original [probe1 build receipt](validation/qnn-probe1-build.json) records 13 passing host
tests, package checks, and the separate emulator UI check. In the API 35 x86
emulator, the UI launches, but the ARM-translated ORT 1.29 native library crashes
during initialization before QNN registration. The trace points to CPU feature
initialization; this is not a physical Xperia result. Reopening the app restores
the interrupted-stage diagnostics, and the share chooser exposes the three JSON
files. No NPU execution or performance result has been obtained from that test.

## What is tested

| Test | Workload | Purpose |
| --- | --- | --- |
| Synthetic matrix multiplication | Fixed quantized matrix operation with nonconstant input | Check HTP graph execution and numerical output |
| RE-USE input projection | A trained `[64,512]` weight matrix, with input flattened to `[405,64]` | Check one actual model operation, about 13.27 million multiply-accumulates |
| RE-USE encoder stride-2 Conv | `[1,64,5,163]` input, `[64,64,1,3]` weights, stride `[1,2]` | Check a conventional convolution and its transfer/layout costs |
| RE-USE encoder dilated Conv | `[1,256,21,163]` input, `[64,256,3,3]` weights, dilation `[8,1]` | Test a heavier operation, about 120.18 million multiply-accumulates |

Each case includes an original FP32 graph and an explicit UINT8
quantize/dequantize (QDQ) graph. Conv bias is quantized to INT32 with the product
of activation and weight scales. Actual model cases use intermediate activations
from NVIDIA's pinned public example audio, not a user recording. The generator
records quantization error relative to FP32 and writes both reference outputs.
One example clip is a smoke-test calibration source, not a general audio-quality
validation set.

The app registers `QNNExecutionProvider`, selects its execution-provider device,
and requests the absolute `libQnnHtp.so` backend path. CPU fallback is disabled.
A passing case requires correct output and observed QNN kernel execution in the
ORT profile. Finding a device or loading a library alone is not a pass.

Each case runs **CPU FP32, CPU QDQ, then QNN HTP QDQ** on identical input
fixtures. CPU uses two intra-op threads and one inter-op thread. QNN uses ORT
intra/inter-op thread settings of one, the `balanced` performance setting, and
HTP hardware; this is not a core-for-core comparison. CPU baselines require CPU-only kernel evidence, and a
failed QNN case does not prevent CPU results or the other cases from running.

Every backend receives **five warm-up calls and 20 timed calls**. Compilation,
first warm-up, total warm-up and steady-call distributions are reported
separately. Timing includes synchronous ORT dispatch, output readback and ORT
profiling overhead. Fixture loading and numerical comparisons are excluded.
These short sequential tests do not control thermal state or DVFS, do not
simulate audio arrival, and do not establish sustained performance.

Ratios are emitted only for verified, numerically passing results. A ratio of
`CPU mean / NPU mean` above one means the NPU was faster for that isolated case;
below one means CPU was faster. FP32-to-NPU comparisons include a precision
change. CPU-QDQ-to-NPU compares the same quantized graph on different backends.
Backend-to-reference error is distinct from FP32 quantization error: the probe1
projection's roughly two-step maximum difference was already relative to a CPU
QDQ reference, so ordinary weight quantization loss alone does not explain it.
No RTF is assigned to isolated layers, and passing their numerical tolerances
does not validate speech quality or full-model streaming latency.

Progress is saved before native stages so a vendor-library process crash can
leave its last stage available for diagnostics. Reopen the app after a crash
and share the previous result before retrying.

## Runtime dependencies

This module uses the separately distributed QNN plugin, avoiding a custom ORT
build or an SDK download:

| Maven dependency | Version |
| --- | --- |
| `com.microsoft.onnxruntime:onnxruntime-android` | `1.29.0` |
| `com.qualcomm.qti:onnxruntime-android-qnn` | `2.7.0` |
| `com.qualcomm.qti:qnn-runtime` | `2.51.0` |

These are the Android versions specified by the
[QNN plugin release](https://github.com/onnxruntime/onnxruntime-qnn/releases/tag/v2.7.0)
and its [pinned documentation](https://github.com/onnxruntime/onnxruntime-qnn/blob/rel-2.7.0/docs/execution-providers/QNN-ExecutionProvider.md).
The app extracts native libraries so FastRPC can load the Hexagon skeleton by
path, sets `ADSP_LIBRARY_PATH` to its native-library directory, and declares
optional visibility of the vendor `libcdsprpc.so` library. It does not supply or
replace the phone's vendor driver.

The QNN module's runtime is independent of `:reuse` and `:phone-demo`, which keep
their existing dependency versions. Complete original licenses and notices are
packaged in `qnn-notices/` and listed [here](licenses/qnn/README.md). Qualcomm's
runtime is proprietary; this repository does not redistribute a standalone QNN
runtime AAR or SDK.

## Build from source

Use JDK 17, Android SDK 35 and the repository's Python environment. Prepare the
pinned F950 model as described in the [phone build guide](phone-demo.md#build-from-source).
The fixture generator also needs the pinned public NVIDIA sample and a host
build of this repository's custom ONNX operations; see its `--help` options.

```powershell
.venv\Scripts\python.exe tools\prepare_qnn_probe.py --help
.venv\Scripts\python.exe tools\fetch_model.py --reference
cmake -S native -B native/build-qnn-reference
cmake --build native/build-qnn-reference --config Release
.venv\Scripts\python.exe tools\prepare_qnn_probe.py --custom-op-library native/build-qnn-reference/Release/reuse_ort_ops.dll
.venv\Scripts\python.exe tools\test_prepare_qnn_probe.py
.\gradlew.bat :qnn-probe:testDebugUnitTest :qnn-probe:assembleDebug :qnn-probe:lintDebug
```

Generated model/fixture assets stay under `.research/qnn-probe2-assets/` and are
copied into the APK at build time. They are not checked into Git. The APK is
written to `android/qnn-probe/build/outputs/apk/debug/`.

## Collect diagnostics over ADB

JSON files are saved under the app's private `files/captures/` directory. They
include device/runtime identity, the failing stage, output error, timing scope
and provider verification. The existing app-scoped collector can read them:

```powershell
pwsh -File tools/capture-phone-logs.ps1 -Serial 'PHONE_IP:ADB_PORT' `
  -PackageId io.github.reuse.qnnprobe
```

The phone must already be paired and connected to ADB. Tailscale connectivity
alone does not enable Android wireless debugging. The collector does not start
the test, install apps or read audio files.

## Next model work

Use the same-device comparison to choose useful larger regions, then measure
the complete CPU/NPU pipeline. The current model has custom CPU
Mamba, normalization and fused convolution operations. Full-model acceleration
requires compatible lowering or deliberate hybrid partitions, representative
chronological state calibration, and audio-quality checks. Fast execution of a
single projection does not establish a useful full-model speedup.
