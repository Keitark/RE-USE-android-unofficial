# Even G2 microphone integration experiment

Target: the user's Xperia 1 IV, using ARM64. Sony lists its platform as
[Snapdragon 8 Gen 1](https://www.sony.jp/xperia/xperia/xperia1m4/spec_sony.html).
The local Xperia-sized Android emulator has an x86_64 host with ARM64 translation;
its timing does not represent that phone.

The current [live G2 companion](live-g2.md) uses the official Even Hub audio API
and a separate Android package. A real 54.5-second capture processes all
872,000 input samples with no reported app/transport drops, RTF 0.759731 and
67/546 calls over 100 ms. Both raw and enhanced PCM were saved; the microphone
was closed and the companion stopped. Source-side BLE continuity is not
attested because capture timestamps/sample indexes are unavailable.

The earlier [compute-fusion codec probe](compute-fusion.md) passes one bounded
foreground 60-second recorded-audio test: all 600 calls, no gaps/resets, RTF
0.550647. That benchmark does not test live G2; its native-interposer stopped
tail remains unflushed. The Hub companion has explicit finite-tail draining.
The earlier builds and their failed gates below remain historical evidence.

## Verified source

On 2026-11-01 the official [Even app download page](https://evenapp.evenrealities.com/)
was followed to [Google Play com.even.sg](https://play.google.com/store/apps/details?id=com.even.sg).
The app was installed into the isolated emulator, and its base/split APKs were
pulled through `pm path`/ADB. The unmodified app was later launch-tested as a
baseline for the emulator loader failure; no account sign-in was performed.
The original package set is retained under `.research/even-play-2.4.0`.

- App: Even Realities **2.4.0**, version code **246**, minSdk 28, targetSdk 36.
- Base SHA256: `b7a12016278dddfd511ec88a31a05f05e2cf7812225818e447d75794b120283d`.
- ARM64 split SHA256: `1be56195f98a616bd23e829050af6604f7be082c7d7d491e8dbc7e5cc954ef6e`.
- Observed Play signer certificate SHA256:
  `f1ff544cfa3112ab594a1591283d239b4b202496bf7ba03295a77a1f00bba779`.
- Original `libapp.so` SHA256:
  `931fc100cd061402c2d91c6c4d9d1f1acb2fa07d1fe85d4d0e7a43633cf3f3a8`.
- Original `liblc3.so` SHA256:
  `8176f80ed7077cb4c04b7853539bc0bad2f8ae10be34a7467e875b9e40b7d67d`.
- Existing `libonnxruntime.so` SHA256:
  `f9aefe58b331128fd33366f8c81218fe6dd5da5028a54438e5b301b23cb2f6ad`.

## Insertion boundary

Static analysis identifies the exported LC3 C decoder as a narrow insertion
point. It produces 160 signed 16-bit mono samples at 16 kHz for each 40-byte,
10 ms LC3 frame. Dart immediately copies that PCM before the next decode.
G2 BLE payloads commonly contain five such frames (50 ms of audio), so decoder
calls can arrive in bursts. The interposer must modify the output before
returning and must never retain a caller-owned PCM pointer.

The public [Even Hub audio API](https://hub.evenrealities.com/docs/build/device-apis#audio)
also exposes 16 kHz PCM to a JavaScript WebView. The current [live route](live-g2.md)
sends that PCM to a separate local Android companion running the AAR. It does
not load the AAR into JavaScript or provide a global stock-app audio filter.
The APK-specific native boundary below is the earlier interposer experiment.

The test shim delegates original LC3 decoding, filters the resulting PCM, and
returns the original format/length. The original codec is retained as a
separately named dependency. The existing app ONNX Runtime is preserved, with
its C API checked before model execution. Model initialization/processing
failures are counted and logged; a pass-through fallback is never counted as
successful RE-USE processing.

The fast profile uses 8 kHz RE-USE between anti-alias downsampling and
interpolation back to the app's 16 kHz transport. It loses upper speech
bandwidth above 4 kHz. The full profile uses 16 kHz RE-USE. Both have buffering
and processing delay in addition to the glasses' own BLE delivery latency.
The one-hop interposer waits for six ready 10 ms output frames before playback
because each incoming G2 burst consumes five frames immediately. The chunked
variant waits for ten ready frames and consumes ten inputs per model call.
Each new session is primed with at least four silent hops, rounded up to a full
call, in a disposable engine; startup time is reported separately. Buffering
and warmup do not relax the sustained deadline/underflow acceptance conditions.

The earlier padding/Softplus native model is the five-hop Sequence graph with 15 zero
pads folded into Conv and nine `FastSoftplus` operations, SHA256
`5c979c328bde1e3c06735588055c7c1289447d6fa23821f52f053bb8dd30bc56`.
It uses the same PCM/state contract and collection delay. That pass's AAR and
probe contain the required custom operation; older libraries cannot load it.
The interposer reuses fixed frame rings and stream buffers, and its 95-tap FIR
skips discarded/zero products while preserving coefficient and double-sum order.
The 50-frame overload gate and output prefill are unchanged.

Only the separate probe has optional `default`, `no_spin`, `bounded_spin` and
`dynamic4` CPU session profiles. `bounded_spin` requires Even's ORT 1.27,
requests 1,000 us/backoff four, and stops helper spinning after Run returns.
The default adds no overrides. Provider availability reports CPU and NNAPI,
but no NNAPI provider is activated. Four warm 20-second ABBA trials complete
all expected work without queue resets, with RTFs 0.583844/default,
0.629498/bounded, 0.822678/bounded and 0.857513/default. The drift prevents a
clear bounded-spin benefit; its first trial has two output gaps and fails,
while the other three pass. Default remains the selected setting; `no_spin`
is untested on the phone. The isolated
FIR host saving and host model saving are documented separately in
[optimization.md](optimization.md), rather than inferred from these phone runs.
The optional `dynamic4` follow-up passes 20 seconds at RTF 0.597002, then fails
one-minute continuity at RTF 0.836540: all 600 calls/3,000 audio hops complete
with no queue reset, but 32 output gaps and 264 call deadline misses remain.
Those profile trials do not accept the LC3 interposer for live G2 use. The
separate supported Hub-companion capture is documented in [live-g2.md](live-g2.md).

## Experimental boundaries

The LC3 decoder is cached across microphone sessions. Decoder construction is
not a reliable start/stop event. Explicit reset and an idle-gap heuristic are
available; an idle gap is not proof of an exact session boundary. The LC3 shim
alone cannot flush a final delayed tail when a stock feature stops immediately.
This requires physical stream/lifecycle testing before normal use.
The interposer precedes the stock app's own audio enhancement stage; that
existing stage is retained. Combined filtering needs actual G2 speech/ASR
evaluation rather than assuming improved recognition from model execution alone.

The test APK set is locally re-signed and is not an official Even release.
Its certificate differs from Google Play's certificate, so it cannot update an
existing official installation in place. The modified Even application was
installed only in the isolated emulator. A separate normal probe package has
now been installed and tested on the physical Xperia alongside its official
Even app. The physical Even package/data were not replaced or uninstalled.
No account sign-in, G2 pairing, authentication change or integrity bypass was
performed by this test.

## APK runtime result

The experimental five-part APK set installs successfully in the isolated
emulator. Launch fails **before RE-USE initialization** in
`com.stub.StubApp.attachBaseContext`, reporting:

```
java.lang.UnsatisfiedLinkError:
dlopen failed: library "/data/data/com.even.sg/.jiagu/libjiagu.so" not found
```

The original, untouched Google Play APK set was restored and launched in the
same emulator. It fails with the **same exception and stack location**. Static
inspection explains the mismatch: this loader selects x86/x64 assets when the
host advertises x86, while the APK contains the ARM64 Jiagu asset. This occurs
before the native protection library, our provider, or the RE-USE shim runs.
No loader checks or protected startup behavior were changed to get past it.

Consequently the modified Even app is **not runtime accepted**. A successful
split install and preservation hashes do not establish app launch, account
compatibility, or live G2 interception. The original official test installation
has been restored. A separate normal Android probe is used to test the original
codec/runtime and processing stage without executing Even's protected launcher.

A codec/PCM/model self-test can establish that the inserted native stage runs
on known audio. It does not establish live G2 behavior, cloud feature behavior,
speech recognition improvement, sustained phone deadlines, thermals or battery.
The final receipt must report those separately.

## Retained artifacts and physical Xperia result

- `dist/even-reuse-fast8k-precise`: retained five-APK experimental candidate,
  signed for local research. Its receipt verifies packaging only; it predates
  the newer fused temporal-update and scheduling experiments.
- `dist/even-reuse-xperia-step8k.apks`: newer locally signed Even package set
  using the fused-step 8 kHz graph and current interposer. Packaging validation
  does not establish launch or live G2 acceptance; it was not installed on the phone.
- `dist/even-reuse-xperia-chunk5.apks`: locally signed five-hop Step package
  set, statically checked and not installed. It is an older experimental
  integration artifact, not a new Sequence build or live-accepted application.
- `dist/even-g2-reuse-probe-hints.apk`: retained previous one-hop probe;
  its Xperia pacing fails.
- `dist/even-g2-reuse-probe-compact.apk`: compiled separate probe using the
  compact one-hop graph; physical-phone timing remains unmeasured.
- `dist/even-g2-reuse-probe-chunk5.apk`: compiled separate probe using the
  vectorized five-hop graph; 20-second Xperia pacing passes, while its first
  60-second run has output gaps and a queue reset.
- `dist/even-g2-reuse-probe-sequence.apk`: earlier frozen separate probe using
  three fused temporal Sequence nodes. Cold two-second pacing/correctness
  passes; one-minute four-thread and two-thread runs fail continuity.
- `android/even-probe-host/build/outputs/apk/debug/even-probe-host-debug.apk`:
  separate ARM64 codec/model harness, installable alongside Even.
- `.research/even-probe-precise`: earlier emulator APK identity,
  initialization, paced failure, decoded baseline, direct replay and parity.
- `.research/xperia-probe-cold-8k-t2`: physical ARM64 direct parity within one
  PCM16 step (RMSE 7.03e-7 normalized, 90.94 dB SNR).
- `.research/xperia-burst-ready-20s-8k-t4`: a retained physical 20-second
  fused-step run, RTF 1.17348, 306 underflows and eight bounded queue resets.
- `.research/xperia-audio-priority-20s-8k-t4`: representative sustained run,
  RTF 1.10741 (22.15 ms/hop), 221 underflows and five queue resets. Its
  320,000-sample direct parity passes within one PCM16 step.
- `.research/xperia-audio-priority-20s-8k-t3`: slower three-thread comparison,
  RTF 1.30987, 320 underflows and 13 queue resets.
- `.research/xperia-hints-cold-8k-t4`: optional scheduling-hint fallback test;
  session creation returned null, no hints were reported, and pacing failed.
- `.research/xperia-chunk5-warm-20s-8k-t4`: optimized physical run, all 200
  calls/1,000 hops processed, RTF 0.639895 and no post-startup output gaps,
  resets, call deadline misses or failures. Nominal startup silence is 250 ms.
- `.research/xperia-chunk5-warm-60s-8k-t4`: first longer optimized run fails,
  with 37 output gaps/one reset and only 595 calls/2,975 hops rather than the
  expected 600/3,000. Its mean model RTF 0.691146 alone is not a pacing pass.
- `.research/xperia-chunk5-paced-v2-warm-60s-8k-t4`: a later revision moves
  recording-file output outside the timed input loop and adds pacing telemetry.
  It still fails: 590/600 calls, 2,950/3,000 hops, 96 gaps/two resets, RTF
  0.878537. Maximum burst lateness is 7.063 ms, with no burst over 10 ms late;
  the packet feeder does not have a large pause in this run. This does not
  establish the cause of the earlier failure.
- `.research/xperia-sequence-cold-8k-t4`: Sequence graph cold two-second pass,
  20 calls/100 hops and no post-startup gaps/resets. Direct PCM agrees with the
  one-hop reference within one PCM16 step; delivered paced prefix matches
  fresh native replay exactly after the counted startup zeros.
- `.research/xperia-sequence-warm-60s-8k-t4`: Sequence four-thread one-minute
  failure, RTF 0.854936, 580/600 calls, 148 gaps and four resets.
- `.research/xperia-sequence-60s-8k-t2`: new two-thread model session in the
  same process; one-minute failure, RTF 1.1437, 520/600 calls, 514 gaps and
  16 resets. This is a reconfigured session, not reuse of the four-thread model.
- `.research/android-arm-translation-chunk5`: chunked native ARM64 probe under
  emulator translation. Its 32,000-sample direct PCM comparison passes within
  one PCM16 step (RMSE 3.8147e-7) against the one-hop host reference. Paced RTF
  1.40224/22 underflows/one reset fails. This is not physical-phone evidence.
- `.research/profile-opt-chunk5-abba`: controlled Windows throughput comparison,
  27.9%/37.6% less computing time per hop at two/four threads. See
  [optimization.md](optimization.md) for equal-duration measurements and latency.

The physical device is a Sony SO-51C (Xperia 1 IV), Android 14/API 34,
arm64-v8a. The separate harness loads the original app runtime 1.27.0 through
C API 23 and the actual codec/model path. Offline output matches the
independent reference within one PCM16 step. The historical four-thread
20-second one-hop run averages 22.15 ms per 20 ms hop and has 221 output gaps/five queue
resets. It has no codec/model exception or fallback, but fails sustained
real-time acceptance. Full timing details and scheduling experiments are in
[validation.md](validation.md).

The new five-hop graph keeps the same trained weights and 20 ms internal state
updates while running encoder/decoder and frequency work over 100 ms together.
The native worker collects ten 10 ms LC3 frames per inference call. It uses
100 ms of ready output prefill, compared with 60 ms in the one-hop pipeline.
The first call yields only 80 ms of synthesized audio, so startup can require
another chunk before prefill is reached. The probe records startup silence,
pipeline delay, inference calls and represented hops; 100 ms processing is not
a promise of 100 ms total end-to-end latency.

Chunk collection adds up to 80 ms beyond ordinary one-hop buffering. It creates
no extra temporal model look-ahead and skips no audio hop, but the delayed tail
still needs a reliable stock microphone-stop callback before interposer live acceptance.
The public Java `ChunkedReuseEnhancer` has an explicit flush API. The earlier
chunked probe passes a bounded 20-second Xperia test at 63.99 ms per 100 ms call;
the earlier one-minute Step/Sequence tests still have output gaps/resets, and
the previous padding/Softplus run has 32 gaps with no reset. The compact one-hop
variant remains unmeasured on the phone. The official Even installation is
unchanged. Startup silence and nominal source-frame age are not a measurement
of live microphone-to-ASR wall latency; output length also includes the startup
zeros and the hook does not flush its buffered tail.

The probe prepares LC3 encoding and baseline decoding before
the paced run, matching the phone-side workload because G2 encodes on the
glasses. Its own worker and ORT session helper threads request audio priority,
with effective priority and errors included in diagnostics. It still requires
longer pacing/lifecycle acceptance before interposer live use. The optional Android performance-hint
session falls back when unavailable or creation fails. The Xperia exposes its
functions but returns no session, so there is no measured hint benefit. No CPU
affinity or global setting was changed.

For reproducing the earlier padding/Softplus benchmark without replacing Even:

```powershell
.\.venv\Scripts\python.exe tools\run_even_probe.py --adb "$env:ANDROID_HOME\platform-tools\adb.exe" --serial YOUR_XPERIA_SERIAL --apk dist\even-g2-reuse-probe-otherareas.apk --input .research\even-probe\input.pcm16le --output .research\xperia-otherareas-probe --model-rate 8000 --threads 4 --runtime-profile default --hold-screen
.\.venv\Scripts\python.exe tools\audit_even_probe.py .research\xperia-otherareas-probe --expected-model-sha256 5c979c328bde1e3c06735588055c7c1289447d6fa23821f52f053bb8dd30bc56
```

The input must be 16 kHz mono PCM16LE. The script installs only
`io.github.reuse.evenprobe`, transfers the fixture and retains reports. It
returns a failure exit status if the sustained paced conditions fail while
retaining offline artifacts. Use `--warm` with the same frozen APK to retain
the existing process/session for a subsequent test when rate, threads and
runtime profile match. Changing one reconfigures the model session while
retaining the process. `--runtime-profile` selects an allowlisted profile
through a launch intent and does not require another APK. Check the
requested/effective/native profile fields and `runtime_profile_matches_requested`
in the execution receipt. `--hold-screen` applies
only to the visible probe window; finish or force-stop the probe after testing
to release it. This is recorded-audio testing through the actual native chain;
the real G2 Hub-companion test is separate, as described in [live-g2.md](live-g2.md).

The example input is a short fixture. Use a longer fixture to measure longer
continuity. The postcheck requires complete expected model call/hop counts,
zero post-startup gaps/resets/fallbacks and matching artifact/phone identity;
average RTF or equal output length alone is insufficient. Those earlier
one-minute failures are retained; the subsequent combined build passes one
bounded foreground minute as detailed in [compute-fusion.md](compute-fusion.md).
That codec-probe result does not establish live G2 or tail delivery. The original Even app and
its data were preserved.
