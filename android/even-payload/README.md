# Even APK Java payload

This module produces framework-only classes for insertion into a separately
obtained Even APK. It adds no AndroidX, Java ONNX Runtime or native dependencies.
The isolated APK packager supplies the native LC3 interposer, original LC3
library under its delegated name, custom operation library, model and manifest.

Build with `gradlew.bat :even-payload:assembleDebug`. The output is
`android/even-payload/build/outputs/apk/debug/even-payload-debug.apk`; extract
its `classes.dex` as a new, collision-checked DEX in the target APK. No generated
`R` resource identifier is referenced by the Java classes.

Injected classes:

- `io.github.reuse.even.ReuseInitProvider`: non-exported startup provider,
  initOrder 100. The packager chooses an authority belonging to the target app.
- `io.github.reuse.even.ReuseBridge`: extracts hash-verified model assets,
  loads `lc3` and invokes the native interposer's JNI methods.
- `io.github.reuse.even.ReuseProbeActivity`: exported only in the debug payload.
  It also refuses execution when the target application is not debuggable.

Assets must contain `reuse/model.onnx`, `reuse/manifest.json` and
`reuse/config.json`. The configuration is
`{"sample_rate":8000,"threads":2,"enabled":true}` by default; a 16 kHz export
must be paired with `sample_rate:16000`. The provider verifies the model SHA-256
from its manifest before atomically publishing the file into private
`files/reuse`, then records `init-result.json`. A configuration or native error
is recorded without throwing from application startup. Native passthrough and
failure handling are the interposer's responsibility.

JNI contract:

```java
static native boolean nativeConfigure(String modelDir, String nativeLibDir,
                                     int sampleRate, int threads);
static native String nativeGetDiagnostics();
static native void nativeReset();
static native String nativeSelfTest(String inputPcmPath, String outputPcmPath,
                                   int sampleRate);
```

The native self-test returns JSON with `ok` (or `success`) and numeric evidence.
For an explicitly debuggable emulator copy, provide **16 kHz** PCM16LE mono input in
`files/reuse/input.pcm16le`, then launch the probe activity. It writes
`enhanced.pcm16le` and `probe-result.json` in the same directory. These are local
file tests; they do not demonstrate a connected G2 microphone path or physical
Xperia performance. The payload requests no microphone or network permission.

The module's standalone APK contains no model/native libraries and is a DEX
producer, not a functioning Even replacement application.

The probe tests the actual 16 kHz G2 LC3 boundary. Its PCM sample rate is distinct
from the configured model rate: the 8 kHz model uses native rate conversion.
`nativeReset()` exposes an explicit stream-boundary reset for controlled callers.
