# Contributing

This project combines a standalone phone-microphone demo, an Android library,
model-export tools, and optional G2 research integrations. Keep changes scoped
to the route being tested. Report model-only timing, recorded-stream timing,
and actual microphone capture as separate results.

## Development

Follow the [phone demo build guide](docs/phone-demo.md#build-from-source) or the
[library build instructions](README.md#build). Keep local SDK paths, downloaded
models, caches, signing keys, APKs, voice recordings, account information, and
private diagnostic data out of Git. Preserve the original license notices.

Open an issue describing the concrete behavior and acceptance criteria before
a substantial change. Include relevant tests in a pull request. Changes to
model math or native arithmetic also need numerical parity evidence; timing
alone does not validate an optimization.

## Report a phone result

Include the following without uploading voice content or personal device IDs:

- Phone model, Android version, ABI, and whether the device was charging.
- APK and model SHA-256, thread setting, and relevant build flags.
- Capture source/route, duration, input/processed/output sample counts.
- Average RTF, model-call timing, calls over 100 ms, and maximum queue delay.
- Application drops, overflows, errors, stop/drain result, and whether the
  microphone was released.
- Whether capture remained visible and whether the device became hot.

Successful installation is not proof of real-time performance. Average RTF
below one does not guarantee every processing deadline or measure source-to-
speaker latency. A failed or partial capture should retain its failure status.

## Sharing audio

The demo can share WAVs only through an explicit user action. Audio is optional
for bug reports. If a reproducible signal is useful, prefer a synthetic or
properly licensed fixture. Never commit someone else's voice or private test
recordings simply to reproduce a performance result.
