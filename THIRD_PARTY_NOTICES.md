# Third-party materials

This is an unofficial Android integration; it is not an NVIDIA product.

## NVIDIA Real-time RE-USE

- Source/checkpoint: https://huggingface.co/nvidia/Real-time_RE-USE
- Pinned revision: `4c0c186272ca853cd18d832fc5c28777b62f32d1`
- Original copyright: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
- Model card license: NVIDIA One-Way Noncommercial License (NSCLv1), linked by
  NVIDIA to https://github.com/NVlabs/HMAR/blob/main/LICENSE
- Local copy: `licenses/NVIDIA-NSCLv1.txt`

`tools/reuse_port.py` adapts that architecture/inference implementation to
explicit recurrent state and portable tensor operators. It and exported graphs
are subject to those terms. `native/selective_scan.cpp` and its compiled
libraries carry those same terms. `native/even/` contains the model-specific
LC3 interposer/DSP for the experimental Even APK integration, under those same
noncommercial terms. It dynamically delegates to the original codec and runtime;
no ownership or licensing of the original Even app is asserted.
The AAR includes that optional native
component and a complete license copy. The exported checkpoint is for research and
noncommercial evaluation. No model weights are bundled in the Android AAR.
The newer commercial 9.6M NemotronLabs RE-USE checkpoint is a different,
noncausal model; it cannot replace the streaming checkpoint in this library.

## PCM session library and demo applications

`android/reuse-audio` exposes the Java PCM session and owns the JNI FIR bridge
in `native/live`. The FIR implementation includes model-specific DSP from
`native/even/dsp.hpp`, under the NVIDIA noncommercial terms above. The Java
session/controller/UI code is MIT. The `phone-demo` and `live-companion`
applications are separate consumers of `reuse-audio`; neither demo is a
requirement for using the reusable libraries. Demo APKs bundle the pinned
model with its complete NVIDIA license. AARs do not bundle model weights.

## ONNX Runtime

Android dependency: `com.microsoft.onnxruntime:onnxruntime-android:1.23.2`.
MIT license. Source: https://github.com/microsoft/onnxruntime
The runtime dependency is resolved separately by Gradle, not shaded into the
library AAR. Its distribution includes its own third-party notices.
The native plugin builds against vendored ONNX Runtime 1.23.2 API headers;
those headers retain their separate Microsoft MIT license in `native/include`.

The optional GPU preview uses an unofficial modified ORT 1.23.2 runtime from
upstream commit `a83fc4d58cb48eb68890dd689f94f28288cf2278`. Its MIT-licensed
device-filter patch and packaging source are in `native/ort-gpu/`. This fork
changes NNAPI `CPU_DISABLED` to select GPU devices exclusively; it is not a
Microsoft-maintained release. The experimental AAR/APK retains the same-version
Java/JNI library, upstream MIT license, full source-version third-party notices,
patch and provenance under `assets/reuse-gpu-runtime/`. The independent
runtime-identity JNI bridge in `native/runtime_build_info_jni.cpp` is MIT.
These runtime licenses do not remove the NVIDIA model restrictions above.

## Android demo dependencies

The separate `qnn-probe` app and opt-in QNN microphone build use ONNX Runtime Android 1.29.0, the MIT-licensed
QNN execution-provider plugin 2.7.0, and the proprietary Qualcomm QNN runtime
2.51.0. Original licenses and third-party notices are retained under
`docs/licenses/qnn/` and packaged in the optional QNN APKs and adapter notices. The Qualcomm runtime is
resolved from Maven Central and distributed only as incorporated in the app;
this repository does not publish a standalone Qualcomm runtime AAR or SDK.
The trained projection/Conv weights, full model candidates, sample-derived fixtures and model-specific
generators remain under the NVIDIA noncommercial terms above. The
independent Java adapters, probe UI and synthetic test are MIT. Adapter AARs
declare Maven dependencies without shading Qualcomm native binaries. See the
[QNN component notices](docs/licenses/qnn/README.md).

The phone demo uses AndroidX Core 1.13.1 and its transitive AndroidX/Kotlin
dependencies, under Apache License 2.0 and their retained notices. A complete
Apache 2.0 license is included in `licenses/Apache-2.0.txt` and in the phone APK.
ONNX Runtime's bundled third-party notice file is also retained in the APK.

The optional G2 companion uses NanoHTTPD 2.3.1 (BSD 3-Clause) and ZXing Core
3.5.3 (Apache 2.0). Its web client imports the MIT-licensed Even Hub SDK 0.0.16.
These components' separate terms do not remove the NVIDIA model restrictions.

## Model papers

Szu-Wei Fu et al., *One Model, Many Latencies: Universal Speech Enhancement
for Diverse Real-Time Applications*, IEEE SLT 2026,
https://arxiv.org/abs/2606.25621

Szu-Wei Fu et al., *Rethinking Training Targets, Architectures and Data Quality
for Universal Speech Enhancement*, NeurIPS 2026,
https://arxiv.org/abs/2603.02641
