# Live G2 microphone companion

The G2 microphone was exercised through the official Even Hub audio API on the
Xperia 1 IV on 2026-11-01. One 54.5-second capture processed all 872,000 input
samples and saved before/after WAVs with no reported input or transport drops.
Average processing kept up with capture; 67 model calls exceeded 100 ms. This
is a bounded live asynchronous pipeline result, not a guarantee that every
100 ms processing deadline, every source-side BLE frame, or background use is
covered.

The microphone was closed and the companion force-stopped after capture. The
sanitized [live capture receipt](../dist/validation/live-g2-capture-summary.json)
records the installed APK identity, counts, timings, numerical parity and limits.

## Route and scope

```text
G2 microphone → official Even app / Hub PCM callback
  → authenticated same-phone WebSocket
  → 95-tap FIR downsample 16k→8k
  → ChunkedReuseEnhancer / f950 five-hop model
  → 95-tap FIR upsample 8k→16k
  → enhanced PCM back to Hub + private raw/enhanced WAV recordings
```

The Android app is a separate package, `io.github.reuse.live`. It uses the
official `com.even.sg` installation for G2 connection and microphone control.
No vendor APK, DEX, native codec, protected loader, signing certificate,
account, pairing record, or official app data is replaced. This route supplies
enhanced audio to this Hub application; it does not install a global filter in
Even's built-in transcription or Android's system microphone.

The web application explicitly requests `audioControl(true, 'glasses')`, checks
the connected G2 and the source field on every nonempty callback, and provides
no phone-microphone fallback. The companion needs no `RECORD_AUDIO` or Bluetooth
permission. Its foreground service uses `dataSync` for the user-started local
transfer/processing task. Internet, foreground-service and notification
permissions are declared. The Android companion's network policy permits
cleartext only for `127.0.0.1`; the actual official Even WebView was separately
tested with this loopback URL.

Both ends bind/use `127.0.0.1:8765`. A random token changes with each bridge
launch. Static files are served from bundled assets; the PCM WebSocket and
status/recording endpoints require that token, and WebSocket handshakes require
the exact loopback Host and Origin. The server limits active clients,
connections, frame allocation, input/output queues and session length. It does
not contact an external audio service.

## Run on the phone

1. Install `dist/reuse-even-g2-live.apk` as the separate companion. The tested
   artifact is a debug-signed research build, not a Play release. It can be
   installed alongside the official Even app because their package IDs differ.
2. Connect the G2 in the official Even app. Open **RE-USE G2 Live** and press
   **Start local bridge**. The foreground notification and local URL appear.
   Android 13+ may ask for notification permission; the app does not ask for
   microphone permission.
3. Copy the current URL into Even Hub's **Developer Hub → Prototype mode →
   Can't scan QR → Type a URL** flow. The companion also shows the URL as a QR.
   Use the exact URL including its `#token=...` fragment. Keep this temporary
   local credential private.
4. Press **Start** on the Hub page. The companion loads the model, processes
   three zero warm-up chunks, resets state/FIR history, and reports readiness
   before the page opens the glasses microphone. Speak near the glasses.
5. Press **Stop**, or let the page stop automatically at 55 seconds. The page
   closes the G2 microphone, waits for callback quiet, sends the ordered stop
   barrier, and waits for enhanced audio and final statistics. Leave the page
   visible until it reports that capture was saved.
6. Stop the local bridge after capture. Stopping the bridge during a capture is
   an explicit abort and cannot be counted as a successful tail drain.

The web cap is 55 seconds, the server accepts at most 60 seconds of input, and
the worker's independent storage limit is 120 seconds. A new capture uses a new
socket and fresh model/DSP state. Hiding/exiting the web page requests a stop;
longer sessions, screen-off use and unattended/background operation have not
been accepted.

## Observed live result

| Measurement | This capture |
|---|---:|
| Source / format | Glasses, mono 16 kHz PCM16LE |
| SDK callbacks | 1,090; every callback 1,600 bytes |
| Input samples / duration | 872,000 / 54.5 s |
| Processed input samples | 872,000 |
| Enhanced samples / duration | 872,188 / 54.51175 s |
| Dropped input / queue overflows | 0 / 0 |
| Rejected transport input / dropped transport output | 0 / 0 bytes |
| Generated / written enhanced PCM | 1,744,376 / 1,744,376 bytes |
| Model calls, excluding warm-up | 546 |
| Mean / p95 / maximum model processing | 74.866 / 107.829 / 117.558 ms |
| Calls over the 100 ms processing interval | 67 |
| Processing RTF, including drain | 0.759731 |
| Maximum queue age | 284.103 ms |
| Maximum queued input | 4,800 samples (300 ms at 16 kHz) |
| First enhanced output after first native input | 121.105 ms |
| Final drain / neural flush | 198.933 / 92.416 ms |
| Microphone start / stop confirmations | Both true |
| Source-mismatch / late callbacks / client discontinuities | 0 / 0 / 0 |

RTF here is processing-worker busy time plus drain divided by input duration.
It excludes model loading, warm-up, idle time, G2/BLE/Hub acquisition latency and
socket delivery time. The first-output and queue measurements use native local
clocks; neither is microphone-to-ASR latency. This capture needed buffering to
absorb slower calls, so mean RTF below 1 does not imply that every call met
100 ms.

The client observed a maximum callback interarrival of 187.3 ms. The SDK did
not provide source sample indexes or capture timestamps in this test, and the
worker's `sample_index_validation` is false. Sequence checks cover ordered
transport after callbacks; they cannot independently attest that no samples
were lost before the public SDK callback. Callback counts, source checks and
successful start/stop establish the supported live Glasses path and nonempty
PCM, not an independent firmware or microphone-array identity attestation.

The raw server field `live_g2_verified` deliberately remains false: a socket
server cannot independently authenticate physical microphone origin. The
bounded client receipt reports `hub_client_stats.liveInputVerified=true` from
the actual connected-device, microphone-control and callback-source checks.
Its enhanced-byte counter is a snapshot taken when stop was sent, before final
drain; use the final server counters and completed WAV for terminal length.

The local comparison files are
`recordings/even-g2-live-20261101/before-reuse.wav` and `after-reuse.wav`.
Headerless `before-reuse.pcm` and `after-reuse.pcm` contain the exact respective
WAV data payloads: signed 16-bit little-endian mono at 16 kHz, without delay
adjustment. Here “before” means PCM delivered by the official Hub SDK before
RE-USE, not separate unprocessed hardware microphone channels.
They are mono PCM16 at 16 kHz and preserve the captured/processed levels; they
are not loudness-normalized. They contain the user's voice and are excluded
from the SDK, APK/source archive and public validation distribution. Listening
comparison and signal level changes alone do not establish a speech-quality or
SNR improvement without a clean reference.

Independent whole-waveform comparison against the original Sequence graph,
NumPy frontend and double-accumulation FIR passes across all 872,188 enhanced
samples, including the complete finite tail: maximum error is one PCM16 step,
RMSE 6.07e-7. The existing normalized tolerances remain 5e-4 maximum and 1e-4
RMSE. This numerical parity is separate from perceived noise/speech quality.

## Finite tail and signal contract

The model runs at 8 kHz in five-hop, 800-sample chunks (100 ms). The 16 kHz
callback format is restored after enhancement, but the 8 kHz model discards the
original upper speech band. This is the existing causal real-time checkpoint,
with its original trained weights and exit-3 graph; the newer noncausal 9.6M
checkpoint is a different model.

Each session has independent down/up FIRs, neural state, STFT history, framing
and output counts. The worker pads the last model chunk, retains only the
known valid output prefix, calls the neural/STFT flush once, and drains both
finite FIRs. For an even, nonempty input the output includes **188 extra
16 kHz samples (11.75 ms)**. Odd input adds one explicitly counted pair pad.
This is a causal FIR drain, not a claim that all recurrent model state has
decayed to zero. The captured output is not delay-compensated or trimmed to
match input length.

Raw/enhanced audio is held in bounded memory during processing and written to
the companion's private `files/sessions/<session-id>/` directory after terminal
drain/abort. There is no processing-path disk writer. Completed `raw.wav`,
`enhanced.wav` and combined `stats.json` can be retrieved locally from the
authenticated `/sessions/latest/` endpoints while the bridge is running.
They are not uploaded or copied into a release archive automatically.

## Build and integration

The module is `android/live-companion`; the web application is
`web/even-g2-live`; its plain Java session adapter and `native/live` JNI wrapper
reuse `android/reuse` and `native/even/dsp.hpp`. The LC3 interposer is not loaded.
The source-built Android route uses ONNX Runtime **1.23.2**, CPU execution and
four intra-op threads. This is separate from the earlier native-codec probe's
Even ONNX Runtime 1.27 execution path.

Prerequisites are JDK 17, Android SDK 35, NDK 26.1.10909125, CMake 3.22.1,
Node >=22.12.0, and the checked-in Gradle/npm lockfiles. Build web assets first:

```powershell
Push-Location web/even-g2-live
npm ci
npm test
npm run build
Pop-Location
.\gradlew.bat :live-companion:testDebugUnitTest :live-companion:assembleDebug :live-companion:lintDebug -PreuseNeonFma=true -PreuseScanN16Registers=true
```

Gradle copies the selected graph, manifest and model license from
`models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal-channelnorm-frequency`
and the frozen web `dist/` into APK assets. The graph SHA256 is
`f950dcfb9321191b2bcb3ab9afa3e100c40e5c64dd1de1d30849be4c1aacce32`.
Custom-op FMA and rank-16 register paths are explicitly enabled; scan grain
remains 16. The app's worker sets Android audio thread priority before ORT
creates its helper threads. No CPU-affinity, governor or global OS setting is
changed.

The tested current APK is `dist/reuse-even-g2-live.apk`, 24,815,030 bytes,
SHA256 `15cfc215835d59732b79df2bfe2fb9420c415dad074b0801c29bd675d07c91bc`.
The initial pre-refresh build is retained separately. This debug APK's custom
library is source-built; its bytes are not the release AAR's bytes. The older
`reuse-android-sdk.zip` remains unchanged and records its earlier codec-probe
validation separately.

The companion's 12 JVM tests cover protocol/authorization, bounded frame
allocation, real HTTP/WebSocket exchange, single-client enforcement, ordered
drain, stop totals, sequence rejection and slow output consumers. The native
JNI/session host harness covers nine contract cases including fragmentation,
short/odd clips, exact finite tails, overflow, abort and resource cleanup. Fake
engines/bridges in these tests are explicitly test-only. They do not substitute
for the live G2 result above. Full build/lint passed before the UI-only refresh;
that bounded label-refresh change was then compiled with `assembleDebug` and
installed before this capture.

The NVIDIA model and model-specific native components use the NVIDIA
noncommercial NSCLv1 terms in `licenses/NVIDIA-NSCLv1.txt`. Java integration
code, ONNX Runtime API headers/runtime, NanoHTTPD and ZXing retain their own
licenses. See `THIRD_PARTY_NOTICES.md`; this is an unofficial research
integration.

Primary interface/platform references:
[Even audio API](https://hub.evenrealities.com/docs/build/device-apis#audio),
[Even local testing](https://hub.evenrealities.com/docs/test/local-testing),
[Android foreground service types](https://developer.android.com/develop/background-work/services/fgs/service-types),
[Android profile app behavior](https://developer.android.com/work/guide).
