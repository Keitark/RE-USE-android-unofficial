package io.github.reuse.audio.qnn;

import static org.junit.Assert.*;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Test;

public final class OwnedEngineResourcesTest {
    @Test public void enhancerClosesBeforeBackendExactlyOnce() throws Exception {
        List<String> order = new ArrayList<>();
        OwnedEngineResources resources = new OwnedEngineResources(
                () -> order.add("enhancer"), () -> order.add("backend"));
        resources.close(); resources.close();
        assertEquals(Arrays.asList("enhancer", "backend"), order);
    }

    @Test public void firstCloseFailureIsPreservedAndBackendStillCloses() {
        List<String> order = new ArrayList<>();
        IOException first = new IOException("enhancer failed"), second = new IOException("backend failed");
        OwnedEngineResources resources = new OwnedEngineResources(
                () -> { order.add("enhancer"); throw first; },
                () -> { order.add("backend"); throw second; });
        Exception failure = assertThrows(Exception.class, resources::close);
        assertSame(first, failure);
        assertArrayEquals(new Throwable[]{second}, failure.getSuppressed());
        assertEquals(Arrays.asList("enhancer", "backend"), order);
        assertDoesNotThrow(resources);
        assertEquals(2, order.size());
    }

    @Test public void enhancerErrorDoesNotSkipBackendCleanup() {
        List<String> order = new ArrayList<>();
        LinkageError first = new LinkageError("native enhancer failure");
        OwnedEngineResources resources = new OwnedEngineResources(
                () -> { order.add("enhancer"); throw first; }, () -> order.add("backend"));
        assertSame(first, assertThrows(LinkageError.class, resources::close));
        assertEquals(Arrays.asList("enhancer", "backend"), order);
    }

    @Test public void creationFailureKeepsOriginalAndAddsCleanupFailure() {
        IOException original = new IOException("model initialization failed"), cleanup = new IOException("backend cleanup failed");
        OwnedEngineResources.closeAfterCreationFailure(() -> { throw cleanup; }, original);
        assertArrayEquals(new Throwable[]{cleanup}, original.getSuppressed());
        OwnedEngineResources.closeAfterCreationFailure(() -> { throw original; }, original);
        assertEquals(1, original.getSuppressed().length);
    }

    private static void assertDoesNotThrow(OwnedEngineResources resources) {
        try { resources.close(); }
        catch (Exception error) { throw new AssertionError("Repeated close retried failed cleanup", error); }
    }
}
