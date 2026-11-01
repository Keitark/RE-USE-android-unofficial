# Official Even Hub → local RE-USE companion

This web plugin runs inside the installed official Even app. It requests only the **G2 glasses** microphone and sends signed16LE/mono/16k PCM to the separate Android companion on the same phone. No phone microphone, CDN, external API or cloud audio route is used. It does not replace Even's built-in transcription input or act as a system microphone.

SDK0.0.16 / CLI0.2.0 / Vite8.3.5 are pinned in package-lock.json. SDK0.0.16 requires Even app2.2.10 or newer. Host2.4.0 was independently observed by the root agent. A normal browser remains at the native-bridge waiting screen and cannot manufacture a live-input result.

```powershell
npm ci
npm test
npm run build
```

The Android server copies `dist/` into its packaged web assets. Serve that directory at `http://127.0.0.1:8765/`, with the authenticated same-origin WebSocket at `/pcm`. In the official Even app, Developer Hub → Prototype mode → Can't scan QR → Type a URL accepts the companion's exact launch URL:

```text
http://127.0.0.1:8765/#token=<the current Android companion token>
```

Root physically verified this loopback/Flutter bridge path with a static preflight page; that did not open a microphone or prove live enhancement. A fresh token comes from the Android companion; this plugin never creates its own server credential. No custom deep-link scheme or JSON-wrapped URL is required. The official CLI0.2.0 QR implementation encodes the original HTTP URL directly. Optional deterministic QR helper:

```powershell
npm run qr -- --url "http://127.0.0.1:8765/#token=<companion token>" --out launch-qr.svg
```

The helper uses local code and does not open a viewer or contact a server. The QR and URL grant this temporary local session; do not publish them. Prefer the manual URL field when available.

The SDK creates one glasses startup page before enabling Start. Each Start verifies a connected G2 (`getDeviceInfo`, with an older-host `getGlassesInfo` alias fallback), obtains companion readiness, then explicitly requests `audioControl(true, 'glasses')`. Every audio event must report `source === 'glasses'`. Source mismatch, socket backlog, enhanced sequence gap or invalid PCM stops the session visibly; there is no phone-mic fallback. SDK callbacks are not assumed to have fixed duration. Typed-array offsets and odd-byte boundaries are preserved.

Keep the plugin visible. Hide/page exit requests a stop; returning does not auto-resume old neural state. Stop awaits microphone closure, then at least120ms of callback quiet (bounded2s), sends ordered end-of-input, and waits for native drain/stopped. Capture auto-stops at55s, below the server's60s byte limit. A fresh Start uses a new socket and model session. Zero-input sessions are explicitly labeled as having no verified G2 audio.

Protocol1:

* Client JSON start: `{type:'start', protocol:1, source:'glasses', rate:16000, format:'s16le'}`.
* Server JSON ready has those same protocol/source/rate/format fields.
* Client binary frames: uint32LE sequence starting0, then even-length raw PCM payload. Callback fragments can be combined/split; sequence counts transport frames.
* Enhanced binary frames: independent uint32LE output sequence, then16k PCM. Enhanced output includes an explicitly accounted native FIR tail; input/output lengths need not match.
* Client JSON stop: type, lastSequence (-1 ifnone), inputBytes, reason, bounded clientStats.
* Server JSON stats/stopped/error. Socket stays open through all enhanced output and stopped.

The socket buffer cap is16KiB; outbound frames are at most8196 bytes including header. Client stats include connection/mic/source assertions, actual callback lengths and interarrival, incomplete-sample status, transport counters and `liveInputVerified`. That flag establishes this client's supported source/control path and nonempty PCM, not an independent firmware/microphone-array identity attestation. Android owns raw/enhanced recordings; no PCM is persisted in browser storage. `window.__reuseLiveDiagnostics()` exposes counters only, never synthetic audio injection.

Tests use a fake bridge/socket explicitly for protocol/state coverage. They cannot pass the physical G2 route gate. Physical live validation and audio quality/near-real-time results require the actual official host, glasses and selected native model/backend.

Primary references: [official audio API](https://hub.evenrealities.com/docs/build/device-apis#audio), [local development](https://hub.evenrealities.com/docs/test/local-testing), [foreground/background behavior](https://hub.evenrealities.com/docs/build/background-lifecycle), [version policy](https://hub.evenrealities.com/docs/reference/versioning).
