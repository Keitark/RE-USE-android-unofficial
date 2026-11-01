package io.github.reuse.live;

import fi.iki.elonen.NanoHTTPD;
import fi.iki.elonen.NanoWSD;
import org.json.JSONObject;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

/** Same-phone HTTP/WebSocket bridge. It never binds a LAN interface or accepts a phone microphone source. */
final class LoopbackServer extends NanoWSD {
    static final int PORT = 8765;
    static final long MAX_INPUT_BYTES = 16000L * 2 * 60;
    interface Assets { InputStream open(String path) throws IOException; }
    interface Observer { void onStatus(Map<String, Object> status); }
    interface Engine {
        void start(); boolean accept(byte[] pcm); void finish(); void close();
    }
    interface EngineFactory { Engine create(Path outputDir, EnhancementSession.Listener listener) throws Exception; }

    private final String token, host;
    private final Assets assets;
    private final Path sessionRoot;
    private final EngineFactory factory;
    private final Observer observer;
    private final AtomicReference<PcmSocket> active = new AtomicReference<>();
    private final ScheduledExecutorService watchdog = Executors.newSingleThreadScheduledExecutor();
    private volatile Map<String, Object> status = initialStatus();
    private volatile Path latestOutput;

    LoopbackServer(String token, Assets assets, Path sessionRoot, EngineFactory factory, Observer observer) {
        this(PORT, token, assets, sessionRoot, factory, observer);
    }
    LoopbackServer(int port, String token, Assets assets, Path sessionRoot, EngineFactory factory, Observer observer) {
        super("127.0.0.1", port);
        this.host = "127.0.0.1:" + port;
        this.token = token;
        this.assets = assets;
        this.sessionRoot = sessionRoot;
        this.factory = factory;
        this.observer = observer;
        setAsyncRunner(new BoundedRequests());
        watchdog.scheduleAtFixedRate(() -> {
            PcmSocket socket = active.get();
            if (socket != null) socket.checkIdle();
        }, 1, 1, TimeUnit.SECONDS);
    }
    String launchUrl() { return "http://" + host + "/#token=" + token; }
    Map<String, Object> snapshot() { return new LinkedHashMap<>(status); }
    private static Map<String, Object> initialStatus() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("state", "listening"); result.put("source", "glasses");
        result.put("live_g2_verified", false);
        return result;
    }
    private void publish(Map<String, Object> next) {
        status = new LinkedHashMap<>(next);
        observer.onStatus(new LinkedHashMap<>(next));
    }
    @Override public Response serve(IHTTPSession request) {
        if (!host.equals(request.getHeaders().get("host"))) return text(Response.Status.FORBIDDEN, "Host rejected");
        if (request.getMethod() != Method.GET) return text(Response.Status.METHOD_NOT_ALLOWED, "GET only");
        if (isWebsocketRequested(request)) {
            if (!"/pcm".equals(request.getUri()) || !authorized(request, true))
                return text(Response.Status.FORBIDDEN, "WebSocket origin or authorization rejected");
            synchronized (active) {
                if (active.get() != null) return text(Response.Status.CONFLICT, "One client is already connected");
                return super.serve(request);
            }
        }
        return serveHttp(request);
    }
    private boolean authorized(IHTTPSession request, boolean websocket) {
        return LiveProtocol.authorized(request.getHeaders(), request.getParameters(), token, host, websocket);
    }
    @Override protected Response serveHttp(IHTTPSession request) {
        String uri = request.getUri();
        if (uri.equals("/health") || uri.equals("/status")) {
            if (!authorized(request, false)) return text(Response.Status.FORBIDDEN, "Authorization required");
            Map<String, Object> data = snapshot();
            data.put("protocol", 1); data.put("server", "reuse-g2-live");
            return json(data);
        }
        if (uri.startsWith("/sessions/latest/")) {
            if (!authorized(request, false)) return text(Response.Status.FORBIDDEN, "Authorization required");
            String name = uri.substring("/sessions/latest/".length());
            if (!(name.equals("raw.wav") || name.equals("enhanced.wav") || name.equals("stats.json")))
                return text(Response.Status.NOT_FOUND, "No such recording");
            Path dir = latestOutput;
            // Files are exposed only after the engine's terminal callback, never while being written.
            if (dir == null) return text(Response.Status.NOT_FOUND, "No completed recording");
            try {
                Path file = dir.resolve(name);
                Response response = newFixedLengthResponse(Response.Status.OK,
                        name.endsWith(".wav") ? "audio/wav" : "application/json", Files.newInputStream(file), Files.size(file));
                response.addHeader("Content-Disposition", "attachment; filename=\"" + name + "\"");
                return headers(response);
            } catch (IOException exception) { return text(Response.Status.NOT_FOUND, "Recording unavailable"); }
        }
        if (!uri.startsWith("/") || uri.contains("..") || uri.contains("\\") || uri.contains("\0"))
            return text(Response.Status.NOT_FOUND, "Not found");
        String file = uri.equals("/") ? "index.html" : uri.substring(1);
        try {
            String mime = file.endsWith(".html") ? "text/html; charset=utf-8" : file.endsWith(".js") ? "application/javascript"
                    : file.endsWith(".css") ? "text/css" : file.endsWith(".json") ? "application/json" : "application/octet-stream";
            return headers(newChunkedResponse(Response.Status.OK, mime, assets.open(file)));
        } catch (IOException exception) { return text(Response.Status.NOT_FOUND, "Not found"); }
    }
    private static Response headers(Response response) {
        response.addHeader("Cache-Control", "no-store");
        response.addHeader("X-Content-Type-Options", "nosniff");
        response.addHeader("Referrer-Policy", "no-referrer");
        return response;
    }
    private static Response text(Response.Status code, String text) {
        return headers(newFixedLengthResponse(code, "text/plain; charset=utf-8", text));
    }
    private static Response json(Map<String, Object> value) {
        return headers(newFixedLengthResponse(Response.Status.OK, "application/json", new JSONObject(value).toString()));
    }
    @Override protected WebSocket openWebSocket(IHTTPSession request) {
        PcmSocket socket = new PcmSocket(new GuardedSession(request));
        active.set(socket); // serve() holds active lock during the handshake construction.
        return socket;
    }
    @Override public void stop() {
        watchdog.shutdownNow();
        PcmSocket socket = active.getAndSet(null);
        if (socket != null) socket.abort("service_stopped", "Companion stopped");
        super.stop();
    }

    private final class PcmSocket extends WebSocket implements EnhancementSession.Listener {
        private final ArrayBlockingQueue<Object> outgoing = new ArrayBlockingQueue<>(32);
        private volatile boolean closed, terminal, terminalDelivered;
        private volatile String state = "connected";
        private volatile long lastInputNs = System.nanoTime();
        private long expectedSequence, inputBytes, rejectedInputBytes, outputSequence, outputBytes, writtenOutputBytes;
        private long droppedOutputBytes, queuedOutputBytes;
        private Engine engine;
        private JSONObject clientStats;
        private Path outputDir;
        private Thread writer;

        PcmSocket(IHTTPSession session) { super(session); }
        @Override protected void onOpen() {
            writer = new Thread(this::writeLoop, "reuse-live-socket-writer");
            writer.setDaemon(true); writer.start();
            update();
        }
        @Override protected synchronized void onMessage(WebSocketFrame frame) {
            if (closed) return;
            if (terminal) {
                if (frame.getOpCode() == WebSocketFrame.OpCode.Binary) {
                    rejectedInputBytes += Math.max(0, frame.getBinaryPayload().length - 4);
                    abort("late_audio", "PCM arrived after the stop barrier");
                }
                return;
            }
            lastInputNs = System.nanoTime();
            try {
                if (frame.getOpCode() == WebSocketFrame.OpCode.Text) {
                    if (frame.getBinaryPayload().length > 4096) throw new IllegalArgumentException("Control message too large");
                    JSONObject message = new JSONObject(frame.getTextPayload());
                    String type = message.optString("type");
                    if (type.equals("start")) {
                        if (!state.equals("connected")) throw new IllegalArgumentException("Session already started");
                        LiveProtocol.validateStart(message);
                        outputDir = sessionRoot.resolve(UUID.randomUUID().toString());
                        Files.createDirectories(outputDir);
                        engine = factory.create(outputDir, this);
                        state = "preparing"; update(); engine.start();
                    } else if (type.equals("stop")) {
                        if (!state.equals("ready") && !state.equals("running")) throw new IllegalArgumentException("Session not accepting stop");
                        if (message.optLong("lastSequence", Long.MIN_VALUE) != expectedSequence - 1
                                || message.optLong("inputBytes", -1) != inputBytes)
                            throw new IllegalArgumentException("Stop totals disagree with accepted audio");
                        if (message.has("clientStats")) clientStats = message.getJSONObject("clientStats");
                        state = "draining"; update(); engine.finish();
                    } else throw new IllegalArgumentException("Unsupported control type");
                } else if (frame.getOpCode() == WebSocketFrame.OpCode.Binary) {
                    if (!state.equals("ready") && !state.equals("running")) throw new IllegalArgumentException("PCM outside capture session");
                    byte[] pcm = LiveProtocol.unpack(frame.getBinaryPayload(), expectedSequence);
                    if (inputBytes + pcm.length > MAX_INPUT_BYTES) throw new IllegalArgumentException("60 second capture limit reached");
                    if (!engine.accept(pcm)) { rejectedInputBytes += pcm.length; abort("input_overflow", "Processing queue rejected audio"); return; }
                    expectedSequence++; inputBytes += pcm.length; state = "running";
                } else throw new IllegalArgumentException("Unsupported message frame");
            } catch (Exception exception) {
                if (frame.getOpCode() == WebSocketFrame.OpCode.Binary)
                    rejectedInputBytes += Math.max(0, frame.getBinaryPayload().length - 4);
                abort("protocol_error", exception.getMessage());
            }
        }
        @Override public synchronized void onReady() {
            if (closed || terminal) return;
            state = "ready"; update();
            offer(new JSONObject(LiveProtocol.fields("type", "ready", "protocol", 1, "source", "glasses", "rate", 16000, "format", "s16le")).toString());
        }
        @Override public synchronized void onAudio(byte[] pcm) {
            if (closed || terminal) { droppedOutputBytes += pcm.length; return; }
            // Keep every application message within the same strict frame bound.
            int max = FrameGuardInputStream.MAX_PAYLOAD - 4;
            for (int at = 0; at < pcm.length; at += max) {
                byte[] part = Arrays.copyOfRange(pcm, at, Math.min(pcm.length, at + max));
                outputBytes += part.length;
                if (!offer(LiveProtocol.pack(part, outputSequence++))) return;
            }
        }
        @Override public synchronized void onFinished(Map<String, Object> stats) {
            if (closed || terminal) return;
            state = "stopped";
            Map<String, Object> result = combined(stats);
            result.put("type", "stopped");
            terminal = true;
            publish(result); offer(result);
        }
        @Override public synchronized void onError(String code, String message, Map<String, Object> stats) {
            if (closed || terminal) return;
            state = "error"; terminal = true;
            Map<String, Object> result = combined(stats);
            result.put("type", "error"); result.put("code", code); result.put("message", message);
            publish(result); offer(result);
        }
        private Map<String, Object> combined(Map<String, Object> stats) {
            Map<String, Object> data = new LinkedHashMap<>(stats);
            data.put("state", state); data.put("source", "glasses");
            data.put("transport_input_bytes", inputBytes); data.put("transport_input_frames", expectedSequence);
            data.put("transport_rejected_input_bytes", rejectedInputBytes);
            data.put("transport_generated_output_bytes", outputBytes);
            data.put("transport_written_output_bytes", writtenOutputBytes);
            data.put("transport_dropped_output_bytes", droppedOutputBytes);
            if (clientStats != null) data.put("hub_client_stats", clientStats);
            data.put("live_g2_verified", false);
            return data;
        }
        private synchronized void update() { publish(combined(java.util.Collections.emptyMap())); }
        private boolean offer(Object value) {
            if (closed) return false;
            long bytes = value instanceof byte[] ? ((byte[]) value).length - 4 : 0;
            if (queuedOutputBytes + bytes <= 16000 && outgoing.offer(value)) {
                queuedOutputBytes += bytes;
                return true;
            }
            if (value instanceof byte[]) droppedOutputBytes += ((byte[]) value).length - 4;
            // Do not block the neural worker on an unresponsive client.
            abort("output_overflow", "Output consumer is too slow");
            return false;
        }
        private void writeLoop() {
            try {
                while (!closed) {
                    Object next = outgoing.poll(1, TimeUnit.SECONDS);
                    if (next instanceof byte[]) {
                        send((byte[]) next);
                        synchronized (this) {
                            writtenOutputBytes += ((byte[]) next).length - 4;
                            queuedOutputBytes -= ((byte[]) next).length - 4;
                        }
                    } else if (next instanceof String) send((String) next);
                    else if (next instanceof Map) {
                        @SuppressWarnings("unchecked") Map<String, Object> finalStats = new LinkedHashMap<>((Map<String, Object>) next);
                        synchronized (this) {
                            finalStats.put("transport_written_output_bytes", writtenOutputBytes);
                            finalStats.put("transport_dropped_output_bytes", droppedOutputBytes);
                        }
                        // The engine has finalized WAVs before its terminal callback. Persist combined
                        // transport/client receipts only after all earlier audio writes have completed.
                        if (outputDir != null) {
                            Files.write(outputDir.resolve("stats.json"), (new JSONObject(finalStats).toString() + "\n")
                                    .getBytes(java.nio.charset.StandardCharsets.UTF_8));
                            latestOutput = outputDir;
                        }
                        publish(finalStats);
                        send(new JSONObject(finalStats).toString());
                        terminalDelivered = true;
                    }
                }
            } catch (IOException exception) { abort("socket_write", "Audio connection write failed"); }
            catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
        }
        private synchronized void checkIdle() {
            if (!closed && !terminal && !state.equals("draining")
                    && System.nanoTime() - lastInputNs > TimeUnit.SECONDS.toNanos(state.equals("preparing") ? 60 : 10))
                abort("capture_timeout", "Audio connection became idle");
        }
        private synchronized void abort(String code, String message) {
            if (closed) return;
            terminal = true; state = "aborted";
            droppedOutputBytes = Math.max(droppedOutputBytes, outputBytes - writtenOutputBytes);
            Map<String, Object> data = combined(LiveProtocol.fields("type", "error", "code", code,
                    "message", message == null ? code : message));
            publish(data);
            closed = true;
            if (engine != null) engine.close();
            outgoing.clear();
            if (writer != null) writer.interrupt();
            // Close the input immediately; NanoWSD.close alone waits for a peer close frame.
            try { getHandshakeRequest().getInputStream().close(); } catch (IOException ignored) { }
            active.compareAndSet(this, null);
        }
        @Override protected void onClose(WebSocketFrame.CloseCode code, String reason, boolean remote) {
            if (!terminal || !terminalDelivered) abort("disconnected", "Audio connection closed before drain completed");
            else {
                closed = true;
                if (engine != null) engine.close();
                if (writer != null) writer.interrupt();
                active.compareAndSet(this, null);
            }
        }
        @Override protected void onException(IOException exception) { abort("socket_read", "Audio connection read failed"); }
        @Override protected void onPong(WebSocketFrame frame) { }
    }

    /** Preserve NanoWSD's tested framing while bounding allocation before payload reads. */
    private static final class GuardedSession implements IHTTPSession {
        private final IHTTPSession delegate;
        private final InputStream input;
        GuardedSession(IHTTPSession delegate) { this.delegate = delegate; input = new FrameGuardInputStream(delegate.getInputStream()); }
        public void execute() throws IOException { delegate.execute(); }
        public CookieHandler getCookies() { return delegate.getCookies(); }
        public Map<String, String> getHeaders() { return delegate.getHeaders(); }
        public InputStream getInputStream() { return input; }
        public Method getMethod() { return delegate.getMethod(); }
        @SuppressWarnings("deprecation") public Map<String, String> getParms() { return delegate.getParms(); }
        public Map<String, List<String>> getParameters() { return delegate.getParameters(); }
        public String getQueryParameterString() { return delegate.getQueryParameterString(); }
        public String getUri() { return delegate.getUri(); }
        public void parseBody(Map<String, String> files) throws IOException, ResponseException { delegate.parseBody(files); }
        public String getRemoteIpAddress() { return delegate.getRemoteIpAddress(); }
        public String getRemoteHostName() { return delegate.getRemoteHostName(); }
    }

    /** Bound NanoHTTPD's otherwise unbounded per-connection thread creation. */
    private static final class BoundedRequests implements AsyncRunner {
        private final Set<ClientHandler> handlers = ConcurrentHashMap.newKeySet();
        private final ThreadPoolExecutor pool = new ThreadPoolExecutor(4, 4, 0, TimeUnit.SECONDS,
                new ArrayBlockingQueue<>(4), runnable -> {
                    Thread thread = new Thread(runnable, "reuse-loopback-request");
                    thread.setDaemon(true); return thread;
                });
        public void exec(ClientHandler handler) {
            handlers.add(handler);
            try { pool.execute(handler); }
            catch (java.util.concurrent.RejectedExecutionException rejected) { handlers.remove(handler); handler.close(); }
        }
        public void closed(ClientHandler handler) { handlers.remove(handler); }
        public void closeAll() {
            for (ClientHandler handler : handlers) handler.close();
            handlers.clear(); pool.shutdownNow();
        }
    }
}
