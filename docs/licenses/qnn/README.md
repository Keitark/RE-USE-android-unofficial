# QNN probe third-party licenses

This directory is copied unmodified to the APK's `qnn-notices/` assets. The
standalone probe has its own dependencies; the existing microphone demos keep
their original runtimes.

| Component | Pinned version | License and notices |
| --- | --- | --- |
| Independent probe application | This source revision | `RE-USE-Project-LICENSE.txt` |
| ONNX Runtime Android | 1.29.0 | `ONNXRuntime-MIT.txt`, `ONNXRuntime-ThirdPartyNotices.txt` |
| ONNX Runtime QNN execution-provider plugin | 2.7.0 | `QNN-EP-MIT.txt`, `QNN-EP-ThirdPartyNotices.txt` |
| Qualcomm QNN runtime | 2.51.0 | `Qualcomm-AI-Stack-License.pdf`, its extracted text, and `Qualcomm-NOTICE.txt` |
| NVIDIA Real-time RE-USE derived projection and Conv fixtures | Existing checkpoint weights | `NVIDIA-NSCLv1.txt` |
| AndroidX Core | 1.13.1 | `Apache-2.0.txt` and the dependency's original notices |

The Qualcomm runtime is proprietary software distributed under its AI Stack
License. Section 1 permits development, demonstration, benchmarking, and
distribution of object code incorporated in an application. It does not grant
standalone redistribution rights. This project resolves its runtime from Maven
Central and does not publish a repackaged Qualcomm runtime AAR or SDK. Preserve
the original license and notices when distributing this probe APK. The MIT
licenses on the application or execution-provider plugin do not relicense the
Qualcomm runtime or NVIDIA-derived weights.

The NVIDIA-derived model fixture remains subject to the NVIDIA One-Way
Noncommercial License (NSCLv1), including its noncommercial research and
education restrictions. The tiny synthetic hardware test has no NVIDIA weights.

Original source files and artifact metadata:

- ONNX Runtime license: https://github.com/microsoft/onnxruntime/blob/v1.29.0/LICENSE
- ONNX Runtime notices: https://github.com/microsoft/onnxruntime/blob/v1.29.0/ThirdPartyNotices.txt
- QNN plugin license: https://github.com/onnxruntime/onnxruntime-qnn/blob/rel-2.7.0/LICENSE
- QNN plugin notices: https://github.com/onnxruntime/onnxruntime-qnn/blob/rel-2.7.0/ThirdPartyNotices.txt
- QNN plugin release and compatible runtime versions: https://github.com/onnxruntime/onnxruntime-qnn/releases/tag/v2.7.0
- QNN runtime artifact containing the original license and notices: https://repo.maven.apache.org/maven2/com/qualcomm/qti/qnn-runtime/2.51.0/qnn-runtime-2.51.0.aar
- NVIDIA model license: https://github.com/NVlabs/HMAR/blob/main/LICENSE

`Qualcomm-AI-Stack-License.txt` is a text extraction of the original PDF for
reading within the app; the original PDF is retained alongside it.
