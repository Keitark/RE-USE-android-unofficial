package io.github.reuse.qnnprobe;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.json.JSONArray;
import org.json.JSONException;
import org.json.JSONObject;

/** Numerical and execution evidence for isolated test graphs, independent of Android. */
final class ProbeVerification {
    private ProbeVerification() { }

    static int countQnnKernelEvents(String text) throws IOException {
        try {
            JSONArray events = new JSONArray(text);
            int count = 0;
            for (int i = 0; i < events.length(); i++) {
                JSONObject event = events.optJSONObject(i);
                if (event == null || !"Node".equals(event.optString("cat"))
                        || !event.optString("name").endsWith("_kernel_time")) continue;
                JSONObject args = event.optJSONObject("args");
                if (args != null && "QNNExecutionProvider".equals(args.optString("provider"))) count++;
            }
            return count;
        } catch (JSONException error) {
            throw new IOException("Invalid ORT profiling JSON", error);
        }
    }

    static float[] decodeFloat32(byte[] bytes, int count) throws IOException {
        if (count <= 0 || count > 16 * 1024 * 1024 || bytes.length != (long) count * Float.BYTES)
            throw new IOException("Float32 fixture length does not match its declared shape");
        float[] result = new float[count];
        ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(result);
        for (float value : result) if (!Float.isFinite(value))
            throw new IOException("Fixture contains a non-finite sample");
        return result;
    }

    static Comparison compare(float[] actual, float[] expected, double absoluteTolerance,
                              double relativeTolerance) throws IOException {
        if (actual.length != expected.length || expected.length == 0)
            throw new IOException("Output shape/element count differs from reference");
        if (!Double.isFinite(absoluteTolerance) || !Double.isFinite(relativeTolerance)
                || absoluteTolerance < 0 || relativeTolerance < 0)
            throw new IOException("Invalid fixture comparison tolerance");
        double maximum = 0, squareError = 0;
        int outside = 0;
        for (int i = 0; i < expected.length; i++) {
            if (!Float.isFinite(actual[i]) || !Float.isFinite(expected[i]))
                return new Comparison(false, expected.length, Double.POSITIVE_INFINITY,
                        Double.POSITIVE_INFINITY, expected.length);
            double error = Math.abs((double) actual[i] - expected[i]);
            maximum = Math.max(maximum, error);
            squareError += error * error;
            if (error > absoluteTolerance + relativeTolerance * Math.abs(expected[i])) outside++;
        }
        return new Comparison(outside == 0, expected.length, maximum,
                Math.sqrt(squareError / expected.length), outside);
    }

    static boolean verified(boolean npuDeviceSelected, boolean correctnessPassed, int qnnEvents) {
        return npuDeviceSelected && correctnessPassed && qnnEvents > 0;
    }

    static final class Comparison {
        final boolean passed;
        final int elements, outsideTolerance;
        final double maximumAbsoluteError, rmse;
        Comparison(boolean passed, int elements, double maximumAbsoluteError, double rmse,
                   int outsideTolerance) {
            this.passed = passed; this.elements = elements;
            this.maximumAbsoluteError = maximumAbsoluteError;
            this.rmse = rmse; this.outsideTolerance = outsideTolerance;
        }
    }
}
