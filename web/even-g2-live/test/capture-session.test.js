import test from 'node:test';
import assert from 'node:assert/strict';
import { CaptureSession, launchConfiguration, requireConnectedG2, hasEvenNativeBridge, waitForNativeBridge } from '../src/capture-session.js';

class Bridge {
  constructor() { this.controls = []; this.connected = true; this.acceptMic = true; this.acceptStop = true; }
  onEvenHubEvent(listener) { this.listener = listener; return () => { this.listener = null; }; }
  async getDeviceInfo() { return { model: 'g2', status: { isConnected: () => this.connected } }; }
  async audioControl(on, source) { this.controls.push([on, source]); return on ? this.acceptMic : this.acceptStop; }
  audio(bytes, source = 'glasses') { this.listener?.({ audioEvent: { source, audioPcm: bytes } }); }
}

class Socket {
  constructor(options = {}) {
    this.options = options; this.readyState = 0; this.bufferedAmount = 0; this.sent = [];
    queueMicrotask(() => { if (this.readyState === 0) { this.readyState = 1; this.onopen?.(); } });
  }
  send(value) {
    this.sent.push(value);
    if (typeof value !== 'string') return;
    const message = JSON.parse(value);
    if (message.type === 'start' && !this.options.noReady) queueMicrotask(() => this.control({ type: 'ready', protocol: 1, source: this.options.source ?? 'glasses', rate: 16000, format: 's16le' }));
    if (message.type === 'stop') queueMicrotask(() => this.control({ type: 'stopped', inputBytes: message.inputBytes }));
  }
  control(value) { this.onmessage?.({ data: JSON.stringify(value) }); }
  output(sequence, payload) {
    const frame = new Uint8Array(4 + payload.length); new DataView(frame.buffer).setUint32(0, sequence, true); frame.set(payload, 4);
    this.onmessage?.({ data: frame.buffer });
  }
  close() { this.readyState = 3; queueMicrotask(() => this.onclose?.()); }
  pcm() { return this.sent.filter(x => x instanceof ArrayBuffer); }
  controls() { return this.sent.filter(x => typeof x === 'string').map(x => JSON.parse(x)); }
}

function setup(t, options = {}) {
  const bridge = new Bridge(); const sockets = [];
  const session = new CaptureSession({ bridge, socketUrl: 'ws://127.0.0.1:8765/pcm?token=test',
    socketFactory: () => { const s = new Socket(options.socket); sockets.push(s); return s; },
    quietMs: 2, sessionMs: 60000, timeoutMs: 200, ...options.session });
  t.after(() => session.dispose());
  return { bridge, sockets, session };
}

test('launch requires exact loopback origin and fragment token; no external destination', () => {
  const good = 'http://127.0.0.1:8765/#token=abcdefghijklmnop0123456789';
  const config = launchConfiguration(good);
  assert.equal(config.socketUrl, 'ws://127.0.0.1:8765/pcm?token=abcdefghijklmnop0123456789');
  assert.equal(new URL(config.socketUrl).hostname, new URL(good).hostname);
  for (const bad of ['https://127.0.0.1:8765/#token=abcdefghijklmnop', 'http://evil.example:8765/#token=abcdefghijklmnop', 'http://127.0.0.1:5173/#token=abcdefghijklmnop', 'http://127.0.0.1:8765/', 'http://127.0.0.1:8765/#token=short']) {
    assert.throws(() => launchConfiguration(bad));
  }
});

test('normal browser cannot pass the native-host gate', async () => {
  const host = {}; let waits = 0;
  assert.equal(hasEvenNativeBridge(host), false);
  await waitForNativeBridge(host, async () => { waits++; host.flutter_inappwebview = { callHandler() {} }; });
  assert.equal(waits, 1); assert.equal(hasEvenNativeBridge(host), true);
});

test('only confirmed connected G2 passes device gate', () => {
  assert.equal(requireConnectedG2({ model: 'g2', status: { connectType: 7 } }, 7), true);
  for (const info of [null, { model: 'g1', status: { isConnected: () => true } }, { model: 'g2', status: { isConnected: () => false } }, { model: 'g2', status: {} }]) assert.throws(() => requireConnectedG2(info));
});

test('no microphone opens before explicit start, connected-device check and server ready', async t => {
  const { session, bridge, sockets } = setup(t);
  assert.deepEqual(bridge.controls, []);
  await session.start();
  assert.equal(session.state, 'recording'); assert.deepEqual(bridge.controls, [[true, 'glasses']]);
  assert.deepEqual(sockets[0].controls()[0], { type: 'start', protocol: 1, source: 'glasses', rate: 16000, format: 's16le' });
  await session.stop(); assert.equal(session.state, 'stopped');
});

test('disconnected glasses and refused mic never substitute the phone microphone', async t => {
  const { session, bridge, sockets } = setup(t); bridge.connected = false;
  await assert.rejects(session.start(), /Connect the G2/); assert.equal(sockets.length, 0); assert.deepEqual(bridge.controls, []);
  bridge.connected = true; bridge.acceptMic = false;
  await assert.rejects(session.start(), /refused/); await session.cleanupPromise;
  assert.deepEqual(bridge.controls, [[true, 'glasses'], [false, undefined]]);
});

test('typed-array slices and arbitrary odd callback boundaries preserve every PCM byte', async t => {
  const { session, bridge, sockets } = setup(t); await session.start();
  bridge.audio(new Uint8Array([90, 1, 2, 3, 91]).subarray(1, 4));
  bridge.audio(new Uint8Array([4, 5, 6]));
  const frames = sockets[0].pcm(); assert.equal(frames.length, 2);
  assert.equal(new DataView(frames[0]).getUint32(0, true), 0); assert.equal(new DataView(frames[1]).getUint32(0, true), 1);
  assert.deepEqual([...new Uint8Array(frames[0]).subarray(4), ...new Uint8Array(frames[1]).subarray(4)], [1, 2, 3, 4, 5, 6]);
  await session.stop(); assert.equal(session.stats.sentPayloadBytes, 6); assert.equal(session.stats.trailingOddBytes, 0);
  const stop = sockets[0].controls().at(-1);
  assert.equal(stop.type, 'stop'); assert.equal(stop.lastSequence, 1); assert.equal(stop.inputBytes, 6); assert.equal(stop.reason, 'user');
  assert.equal(stop.clientStats.liveInputVerified, true); assert.equal(stop.clientStats.callbackBytes, 6);
  assert.equal(Object.hasOwn(stop.clientStats, 'server'), false);
});

test('records actual callback sizes and interarrival without a fixed packet assumption', async t => {
  let now = 0; const { session, bridge } = setup(t, { session: { now: () => now, quietMs: 0 } }); await session.start();
  bridge.audio(new Uint8Array(640)); now = 23; bridge.audio(new Uint8Array(3200)); now = 104; bridge.audio(new Uint8Array(1280));
  assert.deepEqual(session.stats.callbackSizeCounts, { 640: 1, 1280: 1, 3200: 1 });
  assert.equal(session.stats.maxInterarrivalMs, 81); assert.equal(session.stats.sentPayloadBytes, 5120); await session.stop();
});

test('wrong source stops capture and never sends those bytes', async t => {
  const { session, bridge, sockets } = setup(t); await session.start(); bridge.audio(new Uint8Array(640), 'phone');
  assert.equal(session.state, 'error'); assert.equal(session.stats.sourceMismatchEvents, 1); assert.equal(sockets[0].pcm().length, 0);
  await session.cleanupPromise; assert.equal(bridge.controls.at(-1)[0], false);
});

test('transport backpressure fails visibly before hiding dropped PCM', async t => {
  const { session, bridge, sockets } = setup(t); await session.start(); sockets[0].bufferedAmount = 16384;
  bridge.audio(new Uint8Array(640)); assert.equal(session.state, 'error'); assert.match(session.error, /backlog/);
  assert.equal(session.stats.discontinuities, 1); assert.equal(sockets[0].pcm().length, 0);
});

test('enhanced output has its own sequence and may include the terminal FIR tail', async t => {
  const { session, bridge, sockets } = setup(t); await session.start(); bridge.audio(new Uint8Array(640));
  sockets[0].output(0, new Uint8Array(640)); sockets[0].output(1, new Uint8Array(376));
  assert.equal(session.stats.enhancedPayloadBytes, 1016); assert.equal(session.state, 'recording'); await session.stop();
});

test('enhanced sequence discontinuity aborts rather than reporting continuity', async t => {
  const { session, sockets } = setup(t); await session.start(); sockets[0].output(1, new Uint8Array(20));
  assert.equal(session.state, 'error'); assert.match(session.error, /sequence/);
});

test('mic stop quiet barrier includes final callbacks after close acknowledgment', async t => {
  const { session, bridge } = setup(t, { session: { quietMs: 12 } }); await session.start(); bridge.audio(new Uint8Array(640));
  const stop = session.stop(); setTimeout(() => bridge.audio(new Uint8Array(320)), 3);
  await stop; assert.equal(session.stats.sentPayloadBytes, 960); assert.equal(session.stats.micStopSucceeded, true);
});

test('incomplete final sample is an explicit failure', async t => {
  const { session, bridge } = setup(t); await session.start(); bridge.audio(new Uint8Array([1]));
  await assert.rejects(session.stop(), /incomplete sample/); assert.equal(session.state, 'error');
});

test('visibility loss stops once; returning never reconnects or reuses neural state', async t => {
  const { session, bridge, sockets } = setup(t); await session.start(); await session.visibilityChanged(true);
  assert.equal(session.state, 'stopped'); assert.equal(session.stats.stopReason, 'hidden');
  await session.visibilityChanged(false); assert.equal(sockets.length, 1); assert.deepEqual(bridge.controls, [[true, 'glasses'], [false, undefined]]);
  await session.start(); assert.equal(sockets.length, 2); assert.equal(session.inputSequence, 0); await session.stop();
});

test('server format mismatch and timeout prevent microphone start', async t => {
  const wrong = setup(t, { socket: { source: 'phone' } });
  await assert.rejects(wrong.session.start(), /format\/source/); assert.deepEqual(wrong.bridge.controls, []);
  const timeout = setup(t, { socket: { noReady: true }, session: { timeoutMs: 5 } });
  await assert.rejects(timeout.session.start(), /did not become ready/); assert.deepEqual(timeout.bridge.controls, []);
});

test('automatic session limit ends capture explicitly', async t => {
  const { session } = setup(t, { session: { sessionMs: 5 } }); await session.start();
  await new Promise(resolve => setTimeout(resolve, 25)); assert.equal(session.state, 'stopped'); assert.equal(session.stats.stopReason, 'time-limit');
});
