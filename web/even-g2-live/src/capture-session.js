export const PROTOCOL = 1;
export const INPUT_RATE = 16000;
export const MAX_SESSION_MS = 55000;
const MAX_PENDING_BYTES = 16 * 1024;
const MAX_PAYLOAD_BYTES = 8192;
const MAX_INPUT_BYTES = 60 * INPUT_RATE * 2;

export const hasEvenNativeBridge = host => typeof host?.flutter_inappwebview?.callHandler === 'function';

export async function waitForNativeBridge(host, wait = ms => new Promise(resolve => setTimeout(resolve, ms))) {
  while (!hasEvenNativeBridge(host)) await wait(100);
}

export function launchConfiguration(value) {
  const page = new URL(value);
  if (page.protocol !== 'http:' || page.hostname !== '127.0.0.1' || page.port !== '8765') {
    throw new Error('Open the companion launch URL at http://127.0.0.1:8765/.');
  }
  const token = new URLSearchParams(page.hash.slice(1)).get('token');
  if (!token || !/^[A-Za-z0-9_-]{16,256}$/.test(token)) throw new Error('A valid companion session token is required.');
  const socketUrl = new URL('/pcm', page);
  socketUrl.protocol = 'ws:';
  socketUrl.searchParams.set('token', token);
  return { socketUrl: socketUrl.href, token, pageOrigin: page.origin };
}

export function requireConnectedG2(info, connectedValue) {
  const status = info?.status;
  const connected = typeof status?.isConnected === 'function'
    ? status.isConnected() === true
    : connectedValue !== undefined && status?.connectType === connectedValue;
  if (info?.model !== 'g2' || !connected) throw new Error('Connect the G2 glasses in the official Even app first.');
  return true;
}

export class CaptureSession {
  constructor({ bridge, socketUrl, socketFactory = value => new WebSocket(value), connectedValue,
    onChange = () => {}, now = () => performance.now(), visible = () => true,
    quietMs = 120, sessionMs = MAX_SESSION_MS, timeoutMs = 15000 }) {
    this.bridge = bridge; this.socketUrl = socketUrl; this.socketFactory = socketFactory;
    this.connectedValue = connectedValue; this.onChange = onChange; this.now = now; this.visible = visible;
    this.quietMs = quietMs; this.sessionMs = sessionMs; this.timeoutMs = timeoutMs;
    this.state = 'idle'; this.error = ''; this.socket = null; this.stopRequested = false;
    this.cleanupPromise = Promise.resolve();
    this.unsubscribe = bridge.onEvenHubEvent(event => this.audioEvent(event?.audioEvent));
    this.resetCounters();
  }

  resetCounters() {
    this.stats = { protocol: PROTOCOL, requestedSource: 'glasses', observedSource: null,
      deviceConnected: false, micStartSucceeded: false, micStopSucceeded: false,
      callbacks: 0, emptyCallbacks: 0, sourceMismatchEvents: 0, sourceValidatedEvents: 0,
      callbackBytes: 0, minCallbackBytes: null, maxCallbackBytes: 0, callbackSizeCounts: {},
      maxInterarrivalMs: 0, lastInterarrivalMs: null, oddByteCallbacks: 0,
      sentPayloadBytes: 0, sentMessages: 0, enhancedPayloadBytes: 0, enhancedMessages: 0,
      maxSocketPendingBytes: 0, trailingOddBytes: 0, lateAudioEvents: 0, stopReason: null,
      droppedBytes: 0, discontinuities: 0, server: null };
    this.inputSequence = 0; this.outputSequence = 0; this.oddByte = null;
    this.lastCallbackAt = null; this.stopRequested = false; this.stopPromise = null;
    this.micMayBeOn = false; this.micStartPromise = null;
    this.readyResolve = null; this.readyReject = null; this.stoppedResolve = null; this.stoppedReject = null;
  }

  snapshot() { return { state: this.state, error: this.error, ...this.stats,
    liveInputVerified: this.stats.deviceConnected && this.stats.micStartSucceeded && this.stats.sourceValidatedEvents > 0 && this.stats.sentPayloadBytes > 0 && this.stats.sourceMismatchEvents === 0 }; }
  emit() { this.onChange(this.snapshot()); }
  change(state) { this.state = state; this.emit(); }

  async start() {
    if (!['idle', 'stopped', 'error'].includes(this.state)) throw new Error('A capture session is already active.');
    await this.cleanupPromise;
    const previous = this.socket; this.socket = null; previous?.close();
    this.resetCounters(); this.error = ''; this.change('connecting');
    try {
      const getInfo = this.bridge.getDeviceInfo ?? this.bridge.getGlassesInfo;
      if (typeof getInfo !== 'function') throw new Error('The Even host does not expose device information.');
      const info = await this.deadline(getInfo.call(this.bridge), 'G2 device check timed out');
      requireConnectedG2(info, this.connectedValue); this.stats.deviceConnected = true;
      if (this.stopRequested || !this.visible()) throw new Error('Keep the Even plugin visible to start capture.');
      await this.connect();
      if (this.stopRequested || !this.visible()) throw new Error('Capture was cancelled before microphone start.');
      this.change('starting_mic'); this.micMayBeOn = true;
      this.micStartPromise = this.deadline(this.bridge.audioControl(true, 'glasses'), 'Microphone start timed out');
      if (await this.micStartPromise !== true) throw new Error('The Even app refused the G2 microphone request.');
      this.stats.micStartSucceeded = true;
      if (this.stopRequested || !this.visible()) { await this.stop('hidden'); return; }
      this.change('recording');
      this.durationTimer = setTimeout(() => { void this.stop('time-limit').catch(() => {}); }, this.sessionMs);
    } catch (error) {
      if (!this.stopRequested || this.state === 'error') this.fail(error);
      throw error;
    }
  }

  connect() {
    return this.deadline(new Promise((resolve, reject) => {
      this.readyResolve = resolve; this.readyReject = reject;
      const socket = this.socketFactory(this.socketUrl); this.socket = socket; socket.binaryType = 'arraybuffer';
      socket.onopen = () => { if (socket === this.socket && !this.stopRequested) socket.send(JSON.stringify({ type: 'start', protocol: PROTOCOL, source: 'glasses', rate: INPUT_RATE, format: 's16le' })); };
      socket.onmessage = event => { if (socket === this.socket) this.message(event.data); };
      socket.onerror = () => { if (socket === this.socket) { reject(new Error('Local companion WebSocket failed.')); this.fail(new Error('Local companion WebSocket failed.')); } };
      socket.onclose = () => {
        if (socket !== this.socket) return;
        if (!['stopped', 'error'].includes(this.state)) {
          const error = new Error('Local companion disconnected before session completion.'); reject(error); this.fail(error);
        }
      };
    }), 'Local companion did not become ready');
  }

  message(value) {
    try {
      if (typeof value === 'string') {
        const data = JSON.parse(value);
        if (data.type === 'ready') {
          if (data.protocol !== PROTOCOL || data.source !== 'glasses' || data.rate !== INPUT_RATE || data.format !== 's16le') {
            throw new Error('Companion audio format/source differs from the requested G2 stream.');
          }
          this.readyResolve?.(); this.readyResolve = null;
        } else if (data.type === 'stats') { this.stats.server = data; this.emit(); }
        else if (data.type === 'stopped') {
          if (this.state !== 'draining') throw new Error('Companion stopped before the client requested drain.');
          this.stats.server = data; this.change('stopped'); this.stoppedResolve?.(data); this.socket?.close(1000, 'complete');
        } else if (data.type === 'error') throw new Error(`${data.code ?? 'companion'}: ${data.message ?? 'Processing failed'}`);
        else throw new Error('Unknown companion control message.');
      } else {
        if (!['starting_mic', 'recording', 'stopping', 'draining'].includes(this.state)) throw new Error('Enhanced output arrived outside an active session.');
        if (!(value instanceof ArrayBuffer) || value.byteLength < 6 || (value.byteLength - 4) % 2) throw new Error('Malformed enhanced PCM frame.');
        const seq = new DataView(value).getUint32(0, true);
        if (seq !== this.outputSequence++) throw new Error('Enhanced output sequence is discontinuous.');
        this.stats.enhancedMessages++; this.stats.enhancedPayloadBytes += value.byteLength - 4;
        // Enhanced audio is recorded by Android; never play feedback into the G2 mic.
        this.emit();
      }
    } catch (error) { this.fail(error); }
  }

  audioEvent(audio) {
    if (!audio) return;
    if (this.state === 'draining') { this.stats.lateAudioEvents++; this.fail(new Error('Audio arrived after microphone stop/drain barrier.')); return; }
    if (!['starting_mic', 'recording', 'stopping'].includes(this.state)) return;
    try {
      if (audio.source !== 'glasses') { this.stats.sourceMismatchEvents++; throw new Error('Rejected microphone event whose source is not G2 glasses.'); }
      if (!(audio.audioPcm instanceof Uint8Array)) throw new Error('Even SDK returned an invalid PCM byte buffer.');
      const pcm = audio.audioPcm; const timestamp = this.now();
      this.stats.callbacks++; this.stats.sourceValidatedEvents++; this.stats.observedSource = audio.source;
      this.stats.callbackBytes += pcm.byteLength;
      this.stats.minCallbackBytes = Math.min(this.stats.minCallbackBytes ?? pcm.byteLength, pcm.byteLength);
      this.stats.maxCallbackBytes = Math.max(this.stats.maxCallbackBytes, pcm.byteLength);
      const key = String(pcm.byteLength);
      if (Object.hasOwn(this.stats.callbackSizeCounts, key) || Object.keys(this.stats.callbackSizeCounts).length < 32) {
        this.stats.callbackSizeCounts[key] = (this.stats.callbackSizeCounts[key] ?? 0) + 1;
      }
      if (this.lastCallbackAt !== null) {
        this.stats.lastInterarrivalMs = timestamp - this.lastCallbackAt;
        this.stats.maxInterarrivalMs = Math.max(this.stats.maxInterarrivalMs, this.stats.lastInterarrivalMs);
      }
      this.lastCallbackAt = timestamp;
      if (!pcm.byteLength) { this.stats.emptyCallbacks++; this.emit(); return; }
      if (pcm.byteLength % 2) this.stats.oddByteCallbacks++;
      if (this.stats.callbackBytes > MAX_INPUT_BYTES) throw new Error('The bounded live session exceeded 60 seconds of input.');
      let bytes = pcm;
      if (this.oddByte !== null) { bytes = new Uint8Array(pcm.byteLength + 1); bytes[0] = this.oddByte; bytes.set(pcm, 1); this.oddByte = null; }
      if (bytes.byteLength % 2) { this.oddByte = bytes[bytes.byteLength - 1]; bytes = bytes.subarray(0, -1); }
      this.stats.trailingOddBytes = this.oddByte === null ? 0 : 1;
      for (let start = 0; start < bytes.byteLength; start += MAX_PAYLOAD_BYTES) {
        const payload = bytes.subarray(start, start + MAX_PAYLOAD_BYTES);
        if (this.socket?.readyState !== 1) throw new Error('Local companion is not accepting PCM.');
        if (this.socket.bufferedAmount + payload.byteLength + 4 > MAX_PENDING_BYTES) throw new Error('Local transport backlog exceeded 512 ms; capture stopped.');
        if (this.inputSequence > 0xffffffff) throw new Error('PCM sequence exhausted.');
        const frame = new Uint8Array(payload.byteLength + 4);
        new DataView(frame.buffer).setUint32(0, this.inputSequence++, true); frame.set(payload, 4); this.socket.send(frame.buffer);
        this.stats.sentMessages++; this.stats.sentPayloadBytes += payload.byteLength;
        this.stats.maxSocketPendingBytes = Math.max(this.stats.maxSocketPendingBytes, this.socket.bufferedAmount);
      }
      this.emit();
    } catch (error) { this.fail(error); }
  }

  stop(reason = 'user') {
    if (this.stopPromise) return this.stopPromise;
    if (!['connecting', 'starting_mic', 'recording', 'stopping'].includes(this.state)) return Promise.resolve();
    this.stopRequested = true; clearTimeout(this.durationTimer); this.stats.stopReason = reason;
    if (this.state === 'connecting') {
      this.change('stopped'); this.readyReject?.(new Error('Capture cancelled')); this.socket?.close(); return Promise.resolve();
    }
    this.change('stopping');
    this.stopPromise = this.finishStop(reason);
    return this.stopPromise;
  }

  async finishStop(reason) {
    try {
      await this.micStartPromise;
      if (await this.deadline(this.bridge.audioControl(false), 'Microphone stop timed out') !== true) throw new Error('The Even app did not confirm microphone stop.');
      this.micMayBeOn = false; this.stats.micStopSucceeded = true;
      const stopAt = this.now();
      while (this.now() - Math.max(this.lastCallbackAt ?? stopAt, stopAt) < this.quietMs) {
        if (this.now() - stopAt > 2000) throw new Error('G2 audio did not become quiet after microphone stop.');
        await new Promise(resolve => setTimeout(resolve, Math.max(1, this.quietMs / 2)));
      }
      if (this.state === 'error') return;
      if (this.oddByte !== null) throw new Error('Final G2 PCM event ended with an incomplete sample.');
      this.change('draining');
      const finished = new Promise((resolve, reject) => { this.stoppedResolve = resolve; this.stoppedReject = reject; });
      const { server: _server, ...clientStats } = this.snapshot();
      this.socket.send(JSON.stringify({ type: 'stop', lastSequence: this.inputSequence - 1, inputBytes: this.stats.sentPayloadBytes, reason, clientStats }));
      await this.deadline(finished, 'Companion did not finish draining');
    } catch (error) { this.fail(error); throw error; }
  }

  async visibilityChanged(hidden) {
    if (hidden) await this.stop('hidden');
    // Resuming never reconnects or silently shares the previous recurrent state.
  }

  fail(error) {
    if (this.state === 'error') return;
    this.error = error instanceof Error ? error.message : String(error);
    this.stats.discontinuities++; clearTimeout(this.durationTimer); this.change('error');
    this.readyReject?.(new Error(this.error)); this.stoppedReject?.(new Error(this.error));
    this.socket?.close();
    if (this.micMayBeOn) { this.micMayBeOn = false; this.cleanupPromise = Promise.resolve(this.micStartPromise).catch(() => {}).then(() => this.bridge.audioControl(false)).catch(() => {}); }
  }

  deadline(promise, message) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => reject(new Error(message)), this.timeoutMs);
      Promise.resolve(promise).then(value => { clearTimeout(timer); resolve(value); }, error => { clearTimeout(timer); reject(error); });
    });
  }

  dispose() {
    clearTimeout(this.durationTimer); this.unsubscribe?.(); this.socket?.close();
    if (this.micMayBeOn) { this.micMayBeOn = false; void this.bridge.audioControl(false).catch(() => {}); }
  }
}
