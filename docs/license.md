# Licensing and public distribution

Reviewed on **2026-11-01**. This project has mixed licenses. The pinned
[NVIDIA model card](https://huggingface.co/nvidia/Real-time_RE-USE/blob/4c0c186272ca853cd18d832fc5c28777b62f32d1/README.md)
identifies NSCLv1; the revision and source hashes are recorded in
[`tools/source-lock.json`](../tools/source-lock.json).

| Component | License |
|---|---|
| Independently written Java integration/DSP APIs and phone UI | MIT, with the exceptions in the root [`LICENSE`](../LICENSE) |
| Checkpoint, exported graphs, adapted model/export/optimization Python, and model-specific native code | NVIDIA NSCLv1; noncommercial research and education only |
| ONNX Runtime 1.23.2 and its API headers | [MIT](https://github.com/microsoft/onnxruntime/blob/v1.23.2/LICENSE), plus its bundled third-party notices |
| Optional GPU-only ORT device-filter patch, packaging tools and independent runtime-identity JNI bridge | MIT; upstream runtime license and full third-party notices are retained in the experimental AAR/APK |
| Separate QNN probe application and synthetic test | MIT; its ORT 1.29.0 and QNN EP 2.7.0 dependencies carry their own MIT and third-party notices |
| Qualcomm QNN runtime 2.51.0 in the QNN probe | Proprietary Qualcomm AI Stack License; original license and notices are [included](licenses/qnn/README.md) |
| RE-USE-derived QNN projection graph, weights and sample-derived fixtures | NVIDIA NSCLv1; noncommercial research and education |
| AndroidX Core 1.13.1 | Apache 2.0, as declared in the [official Maven metadata](https://dl.google.com/dl/android/maven2/androidx/core/core/1.13.1/core-1.13.1.pom) |

The native NSCLv1 scope includes `native/selective_scan.cpp`, `native/even/`,
and `native/live/`, including their compiled libraries. MIT licensing of the
independent application code does not make the complete model-bearing APK or
pipeline commercially permissive. See [third-party notices](../THIRD_PARTY_NOTICES.md)
for component attribution.

## Public research releases

The [official NVIDIA license](https://github.com/NVlabs/HMAR/blob/main/LICENSE)
grants redistribution in section 2.1, subject to its conditions. Section 3.1
requires distribution under that license, a complete license copy, and
unchanged copyright, patent, trademark, and attribution notices. Section 3.2
preserves the use limitation for derivatives; section 3.3 restricts their use
to noncommercial research and educational purposes.

Accordingly, public source and compiled research APK/AAR releases can be
distributed with those conditions. Publishing on GitHub does not remove them
or authorize commercial deployment of the checkpoint or its derivatives.

For each source or binary release:

- Include the complete [`NVIDIA-NSCLv1.txt`](../licenses/NVIDIA-NSCLv1.txt),
  applicable MIT/Apache notices, and dependency notices with the distribution.
- Retain the original NVIDIA 2025 copyright and accompanying upstream notices
  in adapted material; keep contributor notices separately.
- Preserve the model's license, provenance, and noncommercial designation.
  A license URL alone does not replace the required full copy.

## Attribution and verification

This is an unofficial integration with no NVIDIA affiliation or endorsement.
Section 3.5 grants no trademark rights beyond reproducing required notices.
Upstream names identify source material, not sponsorship.

The local NVIDIA license is byte-identical to the official license checked on
the review date (SHA-256
`6d1daa9c89ec421ba80fe051ac9f0dd46010d4fa4e8f4af654eb8e62f64e2ad5`).
The fetched pinned model card also matches its source-lock hash. This records
the reviewed terms and release requirements; it is not a guarantee of rights
for a different model, dependency, or intended use.
