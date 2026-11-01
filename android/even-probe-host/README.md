# Isolated ARM64 codec and RE-USE probe

This is a separate research application, `io.github.reuse.evenprobe`. It uses
the same framework-only Java bootstrap/probe sources as `even-payload`, with
an ordinary `android.app.Application`. It contains no Even DEX, account flow,
Flutter application or protected application loader. It does not modify or
replace the installed Even application.

`gradlew.bat :even-probe-host:assembleDebug` prepares frozen inputs in
`.research/even-harness` and packages ARM64 only:

- `liblc3.so`, delegated original `liblco.so` and `libreuse_ort_ops.so` from
  `dist/even-reuse-fast8k/split_config.arm64_v8a.apk`.
- The unmodified app's `libonnxruntime.so` from
  `.research/even-play-2.4.0/native-original/libonnxruntime.so`.
- Original `libc++_shared.so` only if required by the ELF dependency closure.
- The selected frozen model, manifest, configuration and license assets from
  `dist/even-reuse-fast8k/base.apk`.

Preparation verifies ARM64 ELF format, dependency closure, the delegated codec's
SONAME and model checksum, then writes per-file hashes/provenance to
`.research/even-harness/inputs.json`. No native source is rebuilt by this module.
Generated binaries and assets stay outside the source tree. For another frozen
model package, provide `-PevenProbeBase=... -PevenProbeSplit=...`.
To test an explicitly rebuilt interposer, add
`-PevenProbeShim=native/even/build-arm64/liblc3.so`. This overrides only the
shim; the delegated original codec, custom operation library, model and original
ONNX Runtime remain sourced from the frozen inputs above. The selected shim's
path and SHA-256 are recorded in `inputs.json`.

APK output:
`android/even-probe-host/build/outputs/apk/debug/even-probe-host-debug.apk`.

The debuggable probe exposes
`io.github.reuse.even.ReuseProbeActivity` as its launcher. Provide 16 kHz mono
PCM16LE in the app-private `files/reuse/input.pcm16le`, then launch that activity.
It writes `init-result.json`, `probe-result.json`, `enhanced.pcm16le` and the
native baseline output into `files/reuse`. The configured model defaults to
8 kHz; native rate conversion handles the 16 kHz G2 codec boundary.

This validates only the isolated codec/inference path. ARM translation on an
x86 emulator cannot establish physical-phone latency or a connected G2 mic
path. A successful probe also cannot establish that a modified Even APK starts;
the original and modified app have an independent baseline launch failure on
the present emulator. No protection bypass is attempted here.
