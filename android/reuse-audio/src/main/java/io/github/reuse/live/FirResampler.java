package io.github.reuse.live;

/** One exclusively owned exact 95-tap FIR stream. No Android or codec dependency. */
public final class FirResampler implements AutoCloseable {
    static { System.loadLibrary("reuse_live_dsp"); }
    private long handle = nativeCreate();

    public synchronized int downsample(float[] input, int count, float[] output) {
        check();
        return nativeProcess(handle, input, count, output, false);
    }

    public synchronized int upsample(float[] input, int count, float[] output) {
        check();
        return nativeProcess(handle, input, count, output, true);
    }

    public synchronized void reset() { check(); nativeReset(handle); }
    private void check() { if (handle == 0) throw new IllegalStateException("Resampler is closed"); }
    @Override public synchronized void close() {
        if (handle != 0) { nativeDestroy(handle); handle = 0; }
    }
    private static native long nativeCreate();
    private static native void nativeReset(long handle);
    private static native int nativeProcess(long handle, float[] input, int count, float[] output, boolean up);
    private static native void nativeDestroy(long handle);
}
