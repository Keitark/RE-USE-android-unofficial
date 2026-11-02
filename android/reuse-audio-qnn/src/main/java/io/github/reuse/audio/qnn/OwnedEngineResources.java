package io.github.reuse.audio.qnn;

/** Closes the consuming enhancer before its runtime backend, once, preserving the first failure. */
final class OwnedEngineResources implements AutoCloseable {
    private final AutoCloseable enhancer, backend;
    private boolean closed;

    OwnedEngineResources(AutoCloseable enhancer, AutoCloseable backend) {
        if (enhancer == null || backend == null) throw new IllegalArgumentException("Owned resources are required");
        this.enhancer = enhancer; this.backend = backend;
    }

    @Override public synchronized void close() throws Exception {
        if (closed) return;
        closed = true;
        Throwable failure = null;
        try { enhancer.close(); }
        catch (Throwable error) { failure = error; }
        try { backend.close(); }
        catch (Throwable error) {
            if (failure == null) failure = error;
            else if (failure != error) failure.addSuppressed(error);
        }
        if (failure instanceof Exception) throw (Exception) failure;
        if (failure instanceof Error) throw (Error) failure;
        if (failure != null) throw new IllegalStateException("Resource cleanup failed", failure);
    }

    static void closeAfterCreationFailure(AutoCloseable backend, Throwable original) {
        try { backend.close(); }
        catch (Throwable cleanup) { if (cleanup != original) original.addSuppressed(cleanup); }
    }
}
