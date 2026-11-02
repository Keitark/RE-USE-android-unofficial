package io.github.reuse.qnnprobe;

import java.io.File;
import java.util.Arrays;
import java.util.Comparator;

/** Checkpoint modification time determines recency; the UUID is only a deterministic tie-breaker. */
final class CaptureOrder {
    private CaptureOrder() { }

    static void newestFirst(File[] directories) {
        Arrays.sort(directories, Comparator
                .comparingLong((File directory) -> new File(directory, "stats.json").lastModified()).reversed()
                .thenComparing(File::getName, Comparator.reverseOrder()));
    }
}
