# Runtime notice selection

The ordinary CPU and QNN builds use the exact upstream ONNX Runtime 1.29.0
third-party notice text from
[`v1.29.0/ThirdPartyNotices.txt`](https://github.com/microsoft/onnxruntime/blob/v1.29.0/ThirdPartyNotices.txt).
The Maven Android AAR does not contain that consolidated notice document.
The separately patched GPU-only runtime remains based on 1.23.2 and retains
the previously bundled 1.23.2 notices. `copyPhoneAssets` selects the matching
file and packages it as `reuse-phone-notices/ONNXRuntime-ThirdPartyNotices.txt`.

Do not translate or edit the upstream notice texts. These documents cover
the upstream dependency catalogue; their inclusion does not mean every
listed component is compiled into every Android ABI.
