package io.github.reuse.qnn;

import java.io.IOException;

/** Process-local lease policy; native registration remains protected by the backend's shared lock. */
final class RegistrationPolicy {
    private Object owner;
    private String path;
    private int references;
    private Throwable cleanupFailure;

    boolean acquire(Object environment, String realLibraryPath) throws IOException {
        if (cleanupFailure != null)
            throw new IOException("Previous QNN registration cleanup failed; restart the app before retrying", cleanupFailure);
        if (environment == null || realLibraryPath == null || realLibraryPath.isEmpty())
            throw new IllegalArgumentException("Environment and real library path are required");
        if (references > 0 && (owner != environment || !path.equals(realLibraryPath)))
            throw new IOException("QNN is already registered for a different environment or native library path");
        boolean first = references == 0;
        if (first) { owner = environment; path = realLibraryPath; }
        references++;
        return first;
    }

    boolean release(Object environment, String realLibraryPath) {
        if (references <= 0 || owner != environment || !path.equals(realLibraryPath))
            throw new IllegalStateException("No matching QNN registration lease exists");
        references--;
        if (references != 0) return false;
        owner = null; path = null;
        return true;
    }

    void poison(Throwable failure) { cleanupFailure = failure; }
}
