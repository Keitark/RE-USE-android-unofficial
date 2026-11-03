# Experimental GPU-only NNAPI runtime

This directory contains a narrow MIT-licensed patch to ONNX Runtime **1.23.2**,
commit `a83fc4d58cb48eb68890dd689f94f28288cf2278`, and its packaging tool.
It is an unofficial research experiment, not an upstream Microsoft runtime.

In this modified build, NNAPI's `CPU_DISABLED` selection accepts only devices
whose Android NNAPI device type is `ANEURALNETWORKS_DEVICE_GPU`. DSP/NPU devices
are excluded as well as CPU devices. API levels before 29 and systems without
an enumerated GPU driver fail explicitly. Other selection modes retain their
upstream behavior. Supported NNAPI partitions are compiled for the enumerated
devices using upstream `ANeuralNetworksCompilation_createForDevices`.

`OrtGetBuildInfoString` exposes `REUSE_NNAPI_GPU_ONLY_V1` so an application can
reject an unpatched runtime. This marker identifies the build; it does not
prove that any graph operation executed on a GPU. The application must verify
actual NNAPI kernel events during warm-up before advertising GPU participation.
Unsupported graph regions still execute through ORT CPU kernels: this is a
**hybrid GPU/CPU path**, including the existing CPU-only RE-USE custom kernels.

NNAPI is deprecated starting Android 15. This experiment provides a bounded
way to assess the current phone's existing driver; it is not a commitment to
NNAPI for a future complete GPU port. GPU coverage, numerical audio quality,
speed, battery consumption and thermal stability require physical-device tests.

## Build

Prepare the project's Python environment and pinned Android toolchain. The
script uses CMake 3.30.9 in an isolated ignored directory because ORT 1.23.2
requires CMake 3.28 or newer. It downloads the pinned ORT source, its ONNX
submodule and the upstream dependency archives needed by CMake.
Set `JAVA_HOME` to JDK 17 and `ANDROID_HOME` to your SDK directory, or pass
`-JavaHome` and `-Sdk` explicitly. The pinned NDK is `26.1.10909125`; SDK CMake
`3.22.1` supplies Ninja while the isolated CMake configures the ORT build.

```powershell
pwsh -NoProfile -File tools/build-ort-gpu.ps1
```

Optional arguments provide different SDK/JDK paths and the stock ORT AAR.
The output is an ARM64/API29-only local Maven dependency:

```text
.research/ort-gpu-build/repository/
  com/microsoft/onnxruntime/onnxruntime-android-gpuonly/1.23.2-reuse-gpuonly1/
```

The default native configuration is **Release** (`-O3`) so CPU fallback regions
retain speed optimizations. `-Configuration MinSizeRel` is available only as a
separately identified size experiment; its timing is not interchangeable.
Standard and contrib operations remain enabled; ML operations are disabled.
Disabling contrib operations in this upstream version causes an ARM64 FP16
convolution link error, so it is not a supported setting for this artifact.

The build retains the stock ORT 1.23.2 `classes.jar`, ARM64 JNI library and
headers. It replaces only ARM64 `libonnxruntime.so`, removes unpatched ABIs,
and adds the complete upstream MIT license, source-version ThirdPartyNotices,
patch and runtime provenance as AAR assets. The stock AAR SHA-256 is pinned by
`package_runtime.py`. Packaging checks the ARM64 ELF machine, 16 KiB LOAD
alignment, marker, ZIP CRC, unchanged Java/JNI and ABI exclusion.

`runtime-provenance.json` beside the local repository records source, patch,
native-library and AAR hashes. No binary is committed to Git. Re-run with
`-PackageOnly` to package an already-built native library.

Packaging verifies that the complete tracked ORT source diff equals the
reviewed patch and that the ONNX submodule is pinned and unmodified. The host
contract checker compiles the actual patched device-selection function with
fake NNAPI calls to exercise GPU filtering, missing/old-driver failures and
unchanged CPU/all-device modes:

```powershell
.venv/Scripts/python.exe native/ort-gpu/verify_device_filter.py --source .research/ort-gpu-build-source --work-dir .research/ort-gpu-build-filter-test
```

These host checks verify selection logic; they do not execute an Android GPU.

References: [official Android build instructions](https://onnxruntime.ai/docs/build/android.html),
[pinned device-selection implementation](https://github.com/microsoft/onnxruntime/blob/v1.23.2/onnxruntime/core/providers/nnapi/nnapi_builtin/nnapi_api_helper.cc),
[Android NNAPI status](https://developer.android.com/ndk/guides/neuralnetworks).
