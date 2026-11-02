# Qualcomm QNN HTP probe

This separate app tests direct Qualcomm Hexagon NPU access, without NNAPI.
It does not record audio or run the full RE-USE enhancement pipeline. The
existing CPU microphone demo remains the app to use for speech enhancement.

The probe is unofficial and is not affiliated with or endorsed by NVIDIA or
Qualcomm. Its NVIDIA-derived projection fixture is for noncommercial research
and education under NSCLv1.

## Try the APK

[Download the research probe APK](https://github.com/Keitark/RE-USE-android-unofficial/releases/download/v0.3.0-qnn-probe1/reuse-qnn-probe.apk)
from the [release and checksums](https://github.com/Keitark/RE-USE-android-unofficial/releases/tag/v0.3.0-qnn-probe1).
It is about 102 MB and debug-signed.

Install `reuse-qnn-probe.apk` and open **RE-USE QNN Probe (Unofficial)**. Press
**Run NPU test**, keep the screen open, then use **Share diagnostics** to export
the result. The probe has no microphone or Internet permission. It installs
alongside the CPU and GPU preview apps.

Android 10 or newer and ARM64 are required. The initial target is Xperia IV's
Snapdragon 8 Gen 1 / SM8450, whose Hexagon generation is V69. The packaged
runtime includes V69 support; actual device firmware and FastRPC access still
need a physical test. An emulator cannot establish NPU compatibility or speed.

The [build receipt](validation/qnn-probe1-build.json) records 13 passing host
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

Both graphs use explicit UINT8 quantize/dequantize boundaries. The projection
uses intermediate activations from NVIDIA's pinned public example audio, not a
user recording. The generator records quantization error relative to the FP32
projection and writes CPU quantized-reference outputs for device verification.
One example clip is a smoke-test calibration source, not a general audio-quality
validation set.

The app registers `QNNExecutionProvider`, selects its execution-provider device,
and requests the absolute `libQnnHtp.so` backend path. CPU fallback is disabled.
A passing case requires correct output and observed QNN kernel execution in the
ORT profile. Finding a device or loading a library alone is not a pass.

Initialization/compilation, warm-up and repeated session-call timings are
reported separately. Session-call times include host dispatch and output
handling; they are not hardware-only kernel times. No RTF is assigned to these
isolated layers. They do not establish streaming latency or full-model speed.

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

Generated model/fixture assets stay under `.research/qnn-probe-assets/` and are
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

After physical HTP execution is established, test larger convolution regions
and measure the complete CPU/NPU pipeline. The current model has custom CPU
Mamba, normalization and fused convolution operations. Full-model acceleration
requires compatible lowering or deliberate hybrid partitions, representative
chronological state calibration, and audio-quality checks. Fast execution of a
single projection does not establish a useful full-model speedup.
