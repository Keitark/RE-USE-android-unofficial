package io.github.reuse.live;

import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;

/** Rejects oversized/fragmented/unmasked client frames before NanoWSD allocates payload storage. */
final class FrameGuardInputStream extends FilterInputStream {
    static final int MAX_PAYLOAD = 32768;
    private int stage; // first byte, length byte, extended length, mask, payload
    private int opcode, lengthBytes, maskBytes;
    private long payloadLength, remaining;

    FrameGuardInputStream(InputStream input) { super(input); }

    @Override public int read() throws IOException {
        int value = in.read();
        if (value < 0) return value;
        consume(value);
        return value;
    }

    @Override public int read(byte[] buffer, int offset, int length) throws IOException {
        if (offset < 0 || length < 0 || offset > buffer.length - length) throw new IndexOutOfBoundsException();
        if (length == 0) return 0;
        if (stage == 4) {
            int count = in.read(buffer, offset, (int) Math.min(length, remaining));
            if (count > 0 && (remaining -= count) == 0) stage = 0;
            return count;
        }
        int value = read();
        if (value < 0) return -1;
        buffer[offset] = (byte) value;
        return 1;
    }

    private void consume(int value) throws IOException {
        if (stage == 0) {
            opcode = value & 15;
            if ((value & 0x80) == 0 || (value & 0x70) != 0 || opcode == 0
                    || !(opcode == 1 || opcode == 2 || opcode == 8 || opcode == 9 || opcode == 10)) {
                throw new IOException("Fragmented, reserved or unsupported WebSocket frame");
            }
            stage = 1;
        } else if (stage == 1) {
            if ((value & 0x80) == 0) throw new IOException("Client WebSocket frame must be masked");
            payloadLength = value & 127;
            if (opcode >= 8 && payloadLength > 125) throw new IOException("Invalid control frame size");
            if (payloadLength == 127) throw new IOException("WebSocket frame exceeds bound");
            if (payloadLength == 126) { payloadLength = 0; lengthBytes = 2; stage = 2; }
            else finishLength();
        } else if (stage == 2) {
            payloadLength = (payloadLength << 8) | value;
            if (--lengthBytes == 0) {
                if (payloadLength < 126) throw new IOException("Nonminimal WebSocket length");
                finishLength();
            }
        } else if (stage == 3) {
            if (--maskBytes == 0) { remaining = payloadLength; stage = remaining == 0 ? 0 : 4; }
        } else if (--remaining == 0) stage = 0;
    }

    private void finishLength() throws IOException {
        if (payloadLength > MAX_PAYLOAD) throw new IOException("WebSocket frame exceeds bound");
        maskBytes = 4;
        stage = 3;
    }
}
