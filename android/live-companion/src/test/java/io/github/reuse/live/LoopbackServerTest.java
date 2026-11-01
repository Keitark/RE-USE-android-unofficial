package io.github.reuse.live;

import org.json.JSONObject;
import org.junit.Test;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.Collections;
import java.util.concurrent.atomic.AtomicBoolean;
import static org.junit.Assert.*;

public class LoopbackServerTest {
    private static final String START = "{\"type\":\"start\",\"protocol\":1,\"source\":\"glasses\",\"rate\":16000,\"format\":\"s16le\"}";
    private static int freePort() throws Exception { try (ServerSocket socket = new ServerSocket(0)) { return socket.getLocalPort(); } }
    private static LoopbackServer server(int port, AtomicBoolean aborted) throws Exception {
        LoopbackServer server = new LoopbackServer(port, "testtoken", path -> new ByteArrayInputStream("test page".getBytes(StandardCharsets.UTF_8)),
                Files.createTempDirectory("reuse-live-server"), (path, listener) -> new LoopbackServer.Engine() {
                    long bytes;
                    public void start() { listener.onReady(); }
                    public boolean accept(byte[] pcm) { bytes += pcm.length; listener.onAudio(pcm); return true; }
                    public void finish() { listener.onFinished(Collections.singletonMap("processed_bytes", bytes)); }
                    public void close() { aborted.set(true); }
                }, status -> { });
        server.start(2000, true); return server;
    }
    @Test public void actualLoopbackHandshakeChecksAuthOriginAndSingleClient() throws Exception {
        int port = freePort(); LoopbackServer server = server(port, new AtomicBoolean());
        try {
            assertTrue(http(port, "GET /health HTTP/1.1\r\nHost: 127.0.0.1:" + port + "\r\n\r\n").startsWith("HTTP/1.1 403"));
            assertTrue(http(port, "GET /health?token=testtoken HTTP/1.1\r\nHost: attacker\r\n\r\n").startsWith("HTTP/1.1 403"));
            assertTrue(http(port, "GET /health?token=testtoken HTTP/1.1\r\nHost: 127.0.0.1:" + port + "\r\n\r\n").startsWith("HTTP/1.1 200"));
            try (Client client = new Client(port, "http://evil.example")) { assertTrue(client.handshake.startsWith("HTTP/1.1 403")); }
            try (Client first = new Client(port, "http://127.0.0.1:" + port); Client second = new Client(port, "http://127.0.0.1:" + port)) {
                assertTrue(first.handshake.startsWith("HTTP/1.1 101"));
                assertTrue(second.handshake.startsWith("HTTP/1.1 409"));
            }
        } finally { server.stop(); }
    }
    @Test public void startPcmStopDrainsInOrderAndReportsWrittenBytes() throws Exception {
        int port = freePort(); LoopbackServer server = server(port, new AtomicBoolean());
        try (Client client = new Client(port, "http://127.0.0.1:" + port)) {
            client.send(1, START.getBytes(StandardCharsets.UTF_8));
            assertEquals("ready", new JSONObject(client.readText()).getString("type"));
            byte[] pcm = new byte[160]; Arrays.fill(pcm, (byte) 23);
            client.send(2, LiveProtocol.pack(pcm, 0)); client.send(2, LiveProtocol.pack(pcm, 1));
            client.send(1, "{\"type\":\"stop\",\"lastSequence\":1,\"inputBytes\":320,\"clientStats\":{\"sourceAsserted\":\"glasses\"}}".getBytes(StandardCharsets.UTF_8));
            assertArrayEquals(pcm, LiveProtocol.unpack(client.readFrame(), 0));
            assertArrayEquals(pcm, LiveProtocol.unpack(client.readFrame(), 1));
            JSONObject stopped = new JSONObject(client.readText());
            assertEquals("stopped", stopped.getString("type"));
            assertEquals(320, stopped.getLong("processed_bytes"));
            assertEquals(320, stopped.getLong("transport_written_output_bytes"));
            assertEquals(0, stopped.getLong("transport_dropped_output_bytes"));
            assertEquals("glasses", stopped.getJSONObject("hub_client_stats").getString("sourceAsserted"));
        } finally { server.stop(); }
    }
    @Test public void sequenceGapAbortsInsteadOfSplicingState() throws Exception {
        int port = freePort(); AtomicBoolean aborted = new AtomicBoolean(); LoopbackServer server = server(port, aborted);
        try (Client client = new Client(port, "http://127.0.0.1:" + port)) {
            client.send(1, START.getBytes(StandardCharsets.UTF_8)); client.readText();
            client.send(2, LiveProtocol.pack(new byte[2], 1));
            for (int attempt = 0; attempt < 100 && !aborted.get(); attempt++) Thread.sleep(5);
            assertTrue(aborted.get()); assertEquals("protocol_error", server.snapshot().get("code"));
            assertEquals(0L, server.snapshot().get("transport_input_bytes"));
        } finally { server.stop(); }
    }
    @Test public void stopTotalMismatchIsTerminalAndDoesNotFlush() throws Exception {
        int port = freePort(); AtomicBoolean aborted = new AtomicBoolean(); LoopbackServer server = server(port, aborted);
        try (Client client = new Client(port, "http://127.0.0.1:" + port)) {
            client.send(1, START.getBytes(StandardCharsets.UTF_8)); client.readText();
            client.send(1, "{\"type\":\"stop\",\"lastSequence\":0,\"inputBytes\":2}".getBytes(StandardCharsets.UTF_8));
            for (int attempt = 0; attempt < 100 && !aborted.get(); attempt++) Thread.sleep(5);
            assertTrue(aborted.get()); assertEquals("aborted", server.snapshot().get("state"));
        } finally { server.stop(); }
    }
    @Test public void oversizedWireFrameAbortsBeforePayloadArrival() throws Exception {
        int port = freePort(); AtomicBoolean aborted = new AtomicBoolean(); LoopbackServer server = server(port, aborted);
        try (Client client = new Client(port, "http://127.0.0.1:" + port)) {
            client.send(1, START.getBytes(StandardCharsets.UTF_8)); client.readText();
            client.output.write(new byte[]{(byte) 0x82, (byte) 0xfe, (byte) 0x80, 1}); client.output.flush();
            for (int attempt = 0; attempt < 100 && !aborted.get(); attempt++) Thread.sleep(5);
            assertTrue(aborted.get()); assertEquals("socket_read", server.snapshot().get("code"));
        } finally { server.stop(); }
    }
    @Test public void slowConsumerCannotBlockEngineOnUnboundedWriterStorage() throws Exception {
        int port = freePort(); AtomicBoolean aborted = new AtomicBoolean();
        LoopbackServer server = new LoopbackServer(port, "testtoken", path -> new ByteArrayInputStream(new byte[0]),
                Files.createTempDirectory("reuse-live-slow"), (path, listener) -> new LoopbackServer.Engine() {
                    public void start() {
                        listener.onReady();
                        for (int chunk = 0; chunk < 100; chunk++) listener.onAudio(new byte[3200]);
                    }
                    public boolean accept(byte[] bytes) { return true; }
                    public void finish() { fail("Overflow must not flush as a successful capture"); }
                    public void close() { aborted.set(true); }
                }, status -> { });
        server.start(2000, true);
        try (Client client = new Client(port, "http://127.0.0.1:" + port)) {
            client.send(1, START.getBytes(StandardCharsets.UTF_8));
            for (int attempt = 0; attempt < 100 && !aborted.get(); attempt++) Thread.sleep(5);
            assertTrue(aborted.get()); assertEquals("output_overflow", server.snapshot().get("code"));
            assertTrue((Long) server.snapshot().get("transport_dropped_output_bytes") > 0);
        } finally { server.stop(); }
    }
    private static String http(int port, String request) throws Exception {
        try (Socket socket = new Socket("127.0.0.1", port)) {
            socket.setSoTimeout(2000); socket.getOutputStream().write(request.getBytes(StandardCharsets.US_ASCII));
            return header(socket.getInputStream());
        }
    }
    private static String header(InputStream stream) throws Exception {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream(); int value;
        while ((value = stream.read()) >= 0) {
            buffer.write(value); String text = buffer.toString("US-ASCII");
            if (text.endsWith("\r\n\r\n")) return text;
            if (buffer.size() > 8192) throw new AssertionError("HTTP header too long");
        }
        throw new AssertionError("Truncated HTTP header");
    }
    private static final class Client implements AutoCloseable {
        final Socket socket; final InputStream input; final OutputStream output; final String handshake;
        Client(int port, String origin) throws Exception {
            socket = new Socket("127.0.0.1", port); socket.setSoTimeout(3000); input = socket.getInputStream(); output = socket.getOutputStream();
            output.write(("GET /pcm?token=testtoken HTTP/1.1\r\nHost: 127.0.0.1:" + port + "\r\nOrigin: " + origin
                    + "\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Version: 13\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n").getBytes(StandardCharsets.US_ASCII));
            handshake = header(input);
        }
        void send(int opcode, byte[] bytes) throws Exception {
            output.write(0x80 | opcode); output.write(0x80 | (bytes.length < 126 ? bytes.length : 126));
            if (bytes.length >= 126) { output.write(bytes.length >> 8); output.write(bytes.length & 255); }
            output.write(new byte[4]); output.write(bytes); output.flush();
        }
        byte[] readFrame() throws Exception {
            int first = input.read(), count = input.read();
            if (first < 0 || count < 0) throw new AssertionError("Truncated frame");
            if ((count & 128) != 0) throw new AssertionError("Masked server frame");
            if (count == 126) count = (input.read() << 8) | input.read();
            if (count == 127) throw new AssertionError("Unexpected huge frame");
            byte[] bytes = new byte[count]; int at = 0;
            while (at < count) { int next = input.read(bytes, at, count - at); if (next < 0) throw new AssertionError("Truncated frame"); at += next; }
            return bytes;
        }
        String readText() throws Exception { return new String(readFrame(), StandardCharsets.UTF_8); }
        public void close() throws Exception { socket.close(); }
    }
}
