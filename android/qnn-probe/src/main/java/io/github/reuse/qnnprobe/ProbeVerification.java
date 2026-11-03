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
        return executionEvidence(text).qnnEvents;
    }

    static ExecutionEvidence executionEvidence(String text) throws IOException {
        try {
            JSONArray events = new JSONArray(text);
            int qnn = 0, cpu = 0, other = 0, missing = 0;
            for (int i = 0; i < events.length(); i++) {
                JSONObject event = events.optJSONObject(i);
                if (event == null || !"Node".equals(event.optString("cat"))
                        || !event.optString("name").endsWith("_kernel_time")) continue;
                JSONObject args = event.optJSONObject("args");
                String provider = args == null ? "" : args.optString("provider");
                if ("QNNExecutionProvider".equals(provider)) qnn++;
                else if ("CPUExecutionProvider".equals(provider)) cpu++;
                else if (provider.isEmpty()) missing++;
                else other++;
            }
            return new ExecutionEvidence(qnn, cpu, other, missing);
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
        double maximum = 0, squareError = 0, absoluteErrorSum = 0, referenceSquares = 0;
        int outside = 0, maximumIndex = 0;
        for (int i = 0; i < expected.length; i++) {
            if (!Float.isFinite(actual[i]) || !Float.isFinite(expected[i]))
                return new Comparison(false, expected.length, Double.POSITIVE_INFINITY,
                        Double.POSITIVE_INFINITY, expected.length, Double.POSITIVE_INFINITY,
                        i, actual[i], expected[i], Double.NaN, Double.NaN);
            double error = Math.abs((double) actual[i] - expected[i]);
            if (i == 0 || error > maximum) { maximum = error; maximumIndex = i; }
            squareError += error * error;
            absoluteErrorSum += error;
            referenceSquares += (double) expected[i] * expected[i];
            if (error > absoluteTolerance + relativeTolerance * Math.abs(expected[i])) outside++;
        }
        double rmse = Math.sqrt(squareError / expected.length);
        double referenceRms = Math.sqrt(referenceSquares / expected.length);
        double normalizedRmse = referenceRms > 0 ? rmse / referenceRms : rmse == 0 ? 0 : Double.NaN;
        return new Comparison(outside == 0, expected.length, maximum, rmse, outside,
                absoluteErrorSum / expected.length, maximumIndex, actual[maximumIndex], expected[maximumIndex],
                referenceRms, normalizedRmse);
    }

    static boolean verified(boolean npuDeviceSelected, boolean correctnessPassed, int qnnEvents) {
        return npuDeviceSelected && correctnessPassed && qnnEvents > 0;
    }

    static boolean verifiedCpu(boolean correctnessPassed, ExecutionEvidence evidence) {
        return correctnessPassed && evidence.cpuEvents > 0 && evidence.qnnEvents == 0
                && evidence.otherEvents == 0 && evidence.missingProviderEvents == 0;
    }

    static boolean verifiedHtp(boolean npuDeviceSelected, boolean correctnessPassed, ExecutionEvidence evidence) {
        return verified(npuDeviceSelected, correctnessPassed, evidence.qnnEvents)
                && evidence.cpuEvents == 0 && evidence.otherEvents == 0 && evidence.missingProviderEvents == 0;
    }

    static Double speedup(boolean cpuVerified, double cpuMeanMs, boolean qnnVerified, double qnnMeanMs) {
        if (!cpuVerified || !qnnVerified || !Double.isFinite(cpuMeanMs) || !Double.isFinite(qnnMeanMs)
                || cpuMeanMs <= 0 || qnnMeanMs <= 0) return null;
        double ratio = cpuMeanMs / qnnMeanMs;
        return Double.isFinite(ratio) && ratio > 0 ? ratio : null;
    }

    static final class ExecutionEvidence {
        final int qnnEvents, cpuEvents, otherEvents, missingProviderEvents;
        ExecutionEvidence(int qnnEvents, int cpuEvents, int otherEvents, int missingProviderEvents) {
            this.qnnEvents = qnnEvents; this.cpuEvents = cpuEvents;
            this.otherEvents = otherEvents; this.missingProviderEvents = missingProviderEvents;
        }
    }

    static final class Comparison {
        final boolean passed;
        final int elements, outsideTolerance, maximumErrorIndex;
        final double maximumAbsoluteError, rmse, meanAbsoluteError, maximumErrorActual,
                maximumErrorReference, referenceRms, normalizedRmse;
        Comparison(boolean passed, int elements, double maximumAbsoluteError, double rmse,
                   int outsideTolerance, double meanAbsoluteError, int maximumErrorIndex,
                   double maximumErrorActual, double maximumErrorReference,
                   double referenceRms, double normalizedRmse) {
            this.passed = passed; this.elements = elements;
            this.maximumAbsoluteError = maximumAbsoluteError;
            this.rmse = rmse; this.outsideTolerance = outsideTolerance;
            this.meanAbsoluteError = meanAbsoluteError; this.maximumErrorIndex = maximumErrorIndex;
            this.maximumErrorActual = maximumErrorActual; this.maximumErrorReference = maximumErrorReference;
            this.referenceRms = referenceRms; this.normalizedRmse = normalizedRmse;
        }
    }
}
