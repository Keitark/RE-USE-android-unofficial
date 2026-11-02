package io.github.reuse.qnnprobe;

import static org.junit.Assert.*;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.Locale;
import org.json.JSONObject;
import org.junit.Test;

/** File and directory contracts only; no Android context, native runtime or microphone is opened. */
public final class ResourceComparisonRunnerTest {
    @Test public void providedCaptureDirectoryMustBeAnExistingImmediatePrivateChild() throws Exception {
        Path root = Files.createTempDirectory("reuse-capture-boundary-test"), captures = root.resolve("captures");
        Path child = captures.resolve("resource-test"), nested = child.resolve("nested"), outside = root.resolve("outside");
        Files.createDirectories(nested); Files.createDirectory(outside);
        Path file = captures.resolve("not-directory"); Files.write(file, new byte[]{1});
        try {
            assertEquals(child.toRealPath(), QnnProbeRunner.validateCaptureDirectory(captures, child));
            assertThrows(IOException.class, () -> QnnProbeRunner.validateCaptureDirectory(captures, captures));
            assertThrows(IOException.class, () -> QnnProbeRunner.validateCaptureDirectory(captures, nested));
            assertThrows(IOException.class, () -> QnnProbeRunner.validateCaptureDirectory(captures, outside));
            assertThrows(IOException.class, () -> QnnProbeRunner.validateCaptureDirectory(captures, captures.resolve("missing")));
            assertThrows(IOException.class, () -> QnnProbeRunner.validateCaptureDirectory(captures, file));
        } finally {
            Files.deleteIfExists(file); Files.deleteIfExists(nested); Files.deleteIfExists(child);
            Files.deleteIfExists(captures); Files.deleteIfExists(outside); Files.deleteIfExists(root);
        }
    }

    @Test public void publicOutputRequiresSafeNameBoundSizeHashAndFiniteFloatData() throws Exception {
        Path directory = Files.createTempDirectory("reuse-public-output-test");
        byte[] data = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putFloat(.25f).putFloat(-.5f).array();
        Path output = directory.resolve("cpu_t1_pass1-cpu-fp32.f32"); Files.write(output, data);
        JSONObject saved = new JSONObject().put("float32_file", output.getFileName().toString())
                .put("float32_sha256", hash(data)).put("samples", 2);
        JSONObject backend = new JSONObject().put("trimmed_output_samples", 2).put("saved_public_output", saved);
        try {
            assertArrayEquals(new float[]{.25f, -.5f}, ResourceComparisonRunner.loadVerifiedOutput(directory, backend), 0);
            saved.put("float32_sha256", "0".repeat(64));
            assertThrows(IOException.class, () -> ResourceComparisonRunner.loadVerifiedOutput(directory, backend));
            saved.put("float32_sha256", hash(data)).put("float32_file", "../outside.f32");
            assertThrows(IOException.class, () -> ResourceComparisonRunner.loadVerifiedOutput(directory, backend));
            saved.put("float32_file", output.getFileName().toString()).put("samples", 3);
            assertThrows(IOException.class, () -> ResourceComparisonRunner.loadVerifiedOutput(directory, backend));
            byte[] invalid = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putFloat(Float.NaN).putFloat(0).array();
            Files.write(output, invalid); saved.put("samples", 2).put("float32_sha256", hash(invalid));
            assertThrows(IOException.class, () -> ResourceComparisonRunner.loadVerifiedOutput(directory, backend));
        } finally { Files.deleteIfExists(output); Files.deleteIfExists(directory); }
    }
    private static String hash(byte[] data) throws Exception {
        StringBuilder text = new StringBuilder();
        for (byte value : MessageDigest.getInstance("SHA-256").digest(data)) text.append(String.format(Locale.ROOT, "%02x", value & 255));
        return text.toString();
    }
}
