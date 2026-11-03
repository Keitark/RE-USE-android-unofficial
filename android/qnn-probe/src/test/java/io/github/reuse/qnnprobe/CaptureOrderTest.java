package io.github.reuse.qnnprobe;

import static org.junit.Assert.assertEquals;
import java.io.File;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.FileTime;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public final class CaptureOrderTest {
    @Rule public final TemporaryFolder temporary = new TemporaryFolder();

    @Test public void newerCheckpointWinsOverLexicallyGreaterUuidWithinSameSecond() throws Exception {
        File oldPass = capture("20261102-120000-ffffffff", 1000);
        File newerInterrupted = capture("20261102-120000-00000000", 2000);
        File[] captures = {oldPass, newerInterrupted};
        CaptureOrder.newestFirst(captures);
        assertEquals(newerInterrupted, captures[0]);
        assertEquals(oldPass, captures[1]);
    }

    @Test public void equalCheckpointTimesUseDeterministicFilenameFallback() throws Exception {
        File lower = capture("20261102-120000-00000000", 2000);
        File higher = capture("20261102-120000-ffffffff", 2000);
        File[] captures = {lower, higher};
        CaptureOrder.newestFirst(captures);
        assertEquals(higher, captures[0]);
        assertEquals(lower, captures[1]);
    }

    private File capture(String name, long modifiedMs) throws Exception {
        File directory = temporary.newFolder(name);
        Path stats = directory.toPath().resolve("stats.json");
        Files.write(stats, new byte[]{'{', '}'});
        Files.setLastModifiedTime(stats, FileTime.fromMillis(modifiedMs));
        return directory;
    }
}
