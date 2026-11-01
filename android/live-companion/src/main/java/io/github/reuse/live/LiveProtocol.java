package io.github.reuse.live;

import org.json.JSONObject;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.List;
import java.util.LinkedHashMap;
import java.util.Map;

/** Pure protocol validation, shared by the server and local contract tests. */
final class LiveProtocol {
    private LiveProtocol() {}
    static Map<String, Object> fields(Object... pairs) {
        Map<String, Object> result = new LinkedHashMap<>();
        for (int at = 0; at < pairs.length; at += 2) result.put((String) pairs[at], pairs[at + 1]);
        return result;
    }
    static boolean authorized(Map<String, String> headers, Map<String, List<String>> params,
            String token, String host, boolean websocket) {
        if (!host.equals(headers.get("host"))) return false;
        String origin = headers.get("origin");
        if (websocket && !("http://" + host).equals(origin)) return false;
        if (!websocket && origin != null && !("http://" + host).equals(origin)) return false;
        String supplied = headers.get("authorization");
        if (supplied != null && supplied.startsWith("Bearer ")) supplied = supplied.substring(7);
        else {
            List<String> values = params.get("token");
            if (values == null || values.size() != 1) return false;
            supplied = values.get(0);
        }
        return supplied != null && MessageDigest.isEqual(token.getBytes(StandardCharsets.US_ASCII),
                supplied.getBytes(StandardCharsets.US_ASCII));
    }
    static void validateStart(JSONObject object) {
        if (!"start".equals(object.optString("type")) || object.optInt("protocol", -1) != 1
                || !"glasses".equals(object.optString("source")) || object.optInt("rate", -1) != 16000
                || !"s16le".equals(object.optString("format"))) {
            throw new IllegalArgumentException("Expected protocol 1, glasses, 16000Hz mono s16le");
        }
    }
    static byte[] unpack(byte[] frame, long expected) {
        if (frame.length < 6 || frame.length > FrameGuardInputStream.MAX_PAYLOAD || (frame.length & 1) != 0)
            throw new IllegalArgumentException("Invalid PCM frame length");
        long sequence = Integer.toUnsignedLong(ByteBuffer.wrap(frame).order(ByteOrder.LITTLE_ENDIAN).getInt());
        if (expected > 0xffffffffL || sequence != expected)
            throw new IllegalArgumentException("Missing, repeated or out-of-order PCM sequence");
        byte[] pcm = new byte[frame.length - 4];
        System.arraycopy(frame, 4, pcm, 0, pcm.length);
        return pcm;
    }
    static byte[] pack(byte[] pcm, long sequence) {
        if (pcm.length == 0 || (pcm.length & 1) != 0 || pcm.length + 4 > FrameGuardInputStream.MAX_PAYLOAD
                || sequence > 0xffffffffL) throw new IllegalArgumentException("Invalid output PCM frame");
        return ByteBuffer.allocate(pcm.length + 4).order(ByteOrder.LITTLE_ENDIAN)
                .putInt((int) sequence).put(pcm).array();
    }
}
