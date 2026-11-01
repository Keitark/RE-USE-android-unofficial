package io.github.reuse.live;

import org.json.JSONObject;
import org.junit.Test;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import static org.junit.Assert.*;

public class LiveProtocolTest {
    @Test public void authenticationRequiresHostOriginAndOneCorrectToken() {
        Map<String, String> headers = new HashMap<>();
        headers.put("host", "127.0.0.1:8765"); headers.put("origin", "http://127.0.0.1:8765");
        assertTrue(LiveProtocol.authorized(headers, Collections.singletonMap("token", List.of("test")), "test", "127.0.0.1:8765", true));
        assertFalse(LiveProtocol.authorized(headers, Collections.singletonMap("token", List.of("wrong")), "test", "127.0.0.1:8765", true));
        assertFalse(LiveProtocol.authorized(headers, Collections.singletonMap("token", List.of("test", "test")), "test", "127.0.0.1:8765", true));
        headers.put("origin", "https://example.com");
        assertFalse(LiveProtocol.authorized(headers, Collections.singletonMap("token", List.of("test")), "test", "127.0.0.1:8765", true));
        headers.remove("origin");
        assertFalse(LiveProtocol.authorized(headers, Collections.singletonMap("token", List.of("test")), "test", "127.0.0.1:8765", true));
        assertTrue(LiveProtocol.authorized(headers, Collections.singletonMap("token", List.of("test")), "test", "127.0.0.1:8765", false));
        headers.put("host", "example.com");
        assertFalse(LiveProtocol.authorized(headers, Collections.singletonMap("token", List.of("test")), "test", "127.0.0.1:8765", false));
    }
    @Test public void acceptsOnlyExplicitGlassesPcmContract() throws Exception {
        JSONObject start = new JSONObject("{\"type\":\"start\",\"protocol\":1,\"source\":\"glasses\",\"rate\":16000,\"format\":\"s16le\"}");
        LiveProtocol.validateStart(start);
        start.put("source", "phone");
        assertThrows(IllegalArgumentException.class, () -> LiveProtocol.validateStart(start));
        start.put("source", "glasses"); start.put("rate", 8000);
        assertThrows(IllegalArgumentException.class, () -> LiveProtocol.validateStart(start));
    }
    @Test public void sequenceAndPayloadPreserveExactBytes() {
        byte[] pcm = new byte[]{0, 1, -1, 127};
        assertArrayEquals(pcm, LiveProtocol.unpack(LiveProtocol.pack(pcm, 0xffffffffL), 0xffffffffL));
        assertThrows(IllegalArgumentException.class, () -> LiveProtocol.unpack(LiveProtocol.pack(pcm, 1), 0));
        assertThrows(IllegalArgumentException.class, () -> LiveProtocol.unpack(new byte[5], 0));
        assertThrows(IllegalArgumentException.class, () -> LiveProtocol.unpack(new byte[32770], 0));
        assertThrows(IllegalArgumentException.class, () -> LiveProtocol.pack(pcm, 0x100000000L));
    }
    @Test public void frameGuardRejectsHugePayloadBeforeBodyIsRead() throws Exception {
        FrameGuardInputStream stream = new FrameGuardInputStream(new ByteArrayInputStream(new byte[]{(byte) 0x82, (byte) 0xfe, (byte) 0x80, 1}));
        assertThrows(IOException.class, () -> { while (stream.read() != -1) { } });
        final FrameGuardInputStream huge = new FrameGuardInputStream(new ByteArrayInputStream(new byte[]{(byte) 0x82, (byte) 0xff}));
        assertThrows(IOException.class, () -> { while (huge.read() != -1) { } });
    }
    @Test public void frameGuardRejectsFragmentedAndUnmaskedFrames() {
        for (byte[] frame : new byte[][]{{2}, {(byte) 0x82, 0}, {(byte) 0x80}, {(byte) 0xc2}}) {
            FrameGuardInputStream stream = new FrameGuardInputStream(new ByteArrayInputStream(frame));
            assertThrows(IOException.class, () -> { while (stream.read() != -1) { } });
        }
    }
    @Test public void boundedMaskedFramesRemainByteIdenticalAcrossReads() throws Exception {
        ByteArrayOutputStream wire = new ByteArrayOutputStream();
        for (int length : new int[]{0, 2, 125, 126, 32768}) {
            wire.write(0x82); wire.write(0x80 | (length < 126 ? length : 126));
            if (length >= 126) { wire.write(length >> 8); wire.write(length & 255); }
            wire.write(new byte[4]); wire.write(new byte[length]);
        }
        byte[] original = wire.toByteArray();
        FrameGuardInputStream input = new FrameGuardInputStream(new ByteArrayInputStream(original));
        ByteArrayOutputStream result = new ByteArrayOutputStream();
        byte[] scratch = new byte[77]; int count;
        while ((count = input.read(scratch)) >= 0) result.write(scratch, 0, count);
        assertArrayEquals(original, result.toByteArray());
    }
}
