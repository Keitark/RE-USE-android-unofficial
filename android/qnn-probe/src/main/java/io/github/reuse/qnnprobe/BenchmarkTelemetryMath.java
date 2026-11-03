package io.github.reuse.qnnprobe;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Locale;

/** Pure parsing/accounting helpers; no Android access and no process identity collection. */
final class BenchmarkTelemetryMath {
    private BenchmarkTelemetryMath() { }

    static final class CpuTimes {
        final long[] counters;
        final long totalTicks, busyTicks;
        CpuTimes(long[] values) {
            counters = values.clone();
            long total = 0;
            for (long value : counters) total = Math.addExact(total, value);
            totalTicks = total;
            busyTicks = total - counters[3] - counters[4];
        }
    }

    /** Linux guest/guest_nice are already included in user/nice and must not be added again. */
    static CpuTimes parseProcStat(String text) {
        if (text == null) throw new IllegalArgumentException("Missing CPU counters");
        String first = text.split("\\r?\\n", 2)[0].trim();
        String[] fields = first.split("\\s+");
        if (fields.length < 5 || !"cpu".equals(fields[0]))
            throw new IllegalArgumentException("Expected aggregate cpu row");
        long[] values = new long[8];
        for (int i = 0; i < values.length && i + 1 < fields.length; i++) {
            values[i] = Long.parseLong(fields[i + 1]);
            if (values[i] < 0) throw new IllegalArgumentException("Negative CPU counter");
        }
        return new CpuTimes(values);
    }

    private static boolean increasing(CpuTimes previous, CpuTimes current) {
        if (previous == null || current == null) return false;
        for (int i = 0; i < previous.counters.length; i++)
            if (current.counters[i] < previous.counters[i]) return false;
        return current.totalTicks > previous.totalTicks;
    }

    static Double systemBusyPercent(CpuTimes previous, CpuTimes current) {
        if (!increasing(previous, current)) return null;
        return 100.0 * (current.busyTicks - previous.busyTicks)
                / (current.totalTicks - previous.totalTicks);
    }

    /** 100% means one fully busy core. A multithreaded process may legitimately exceed 100%. */
    static Double processOneCorePercent(long previousCpuMs, long currentCpuMs,
                                        long previousElapsedNs, long currentElapsedNs) {
        if (previousCpuMs < 0 || currentCpuMs < previousCpuMs || previousElapsedNs < 0
                || currentElapsedNs <= previousElapsedNs) return null;
        return 100.0 * (currentCpuMs - previousCpuMs) * 1_000_000.0
                / (currentElapsedNs - previousElapsedNs);
    }

    /** Approximate other-system work, including kernel/drivers/other foreground apps; not attribution. */
    static Double otherSystemOneCorePercent(CpuTimes previous, CpuTimes current,
                                            long processCpuDeltaMs, long elapsedDeltaNs, long ticksPerSecond) {
        if (!increasing(previous, current) || processCpuDeltaMs < 0 || elapsedDeltaNs <= 0
                || ticksPerSecond <= 0) return null;
        double busyMs = 1000.0 * (current.busyTicks - previous.busyTicks) / ticksPerSecond;
        // Jiffy quantization and slightly different read instants can disagree with process accounting.
        if (processCpuDeltaMs > busyMs + Math.max(1.0, 1000.0 / ticksPerSecond)) return null;
        return 100.0 * Math.max(0.0, busyMs - processCpuDeltaMs) * 1_000_000.0 / elapsedDeltaNs;
    }

    /** Read only the first three load averages; the trailing last-PID field is deliberately ignored. */
    static double[] parseLoadAverage(String text) {
        if (text == null) throw new IllegalArgumentException("Missing load average");
        String[] fields = text.trim().split("\\s+");
        if (fields.length < 3) throw new IllegalArgumentException("Incomplete load average");
        double[] values = new double[3];
        for (int i = 0; i < 3; i++) {
            values[i] = Double.parseDouble(fields[i]);
            if (!Double.isFinite(values[i]) || values[i] < 0)
                throw new IllegalArgumentException("Invalid load average");
        }
        return values;
    }

    static Map<String, Double> parseCpuPressure(String text) {
        if (text == null) throw new IllegalArgumentException("Missing pressure data");
        Map<String, Double> values = new LinkedHashMap<>();
        for (String line : text.trim().split("\\r?\\n")) {
            String[] fields = line.trim().split("\\s+");
            if (fields.length < 2 || !("some".equals(fields[0]) || "full".equals(fields[0])))
                throw new IllegalArgumentException("Invalid CPU pressure row");
            for (int i = 1; i < fields.length; i++) {
                String[] pair = fields[i].split("=", 2);
                if (pair.length != 2 || !("avg10".equals(pair[0]) || "avg60".equals(pair[0])
                        || "avg300".equals(pair[0]) || "total".equals(pair[0]))) continue;
                double value = Double.parseDouble(pair[1]);
                if (!Double.isFinite(value) || value < 0 || (!"total".equals(pair[0]) && value > 100))
                    throw new IllegalArgumentException("Invalid CPU pressure value");
                values.put(fields[0] + "_" + pair[0], value);
            }
        }
        if (values.isEmpty()) throw new IllegalArgumentException("Missing CPU pressure values");
        return values;
    }

    /** Explicit plausibility guard rejects vendor sentinels such as -273000 milli-Celsius. */
    static Double milliCelsius(long raw) {
        return raw >= -40_000 && raw <= 150_000 ? raw / 1000.0 : null;
    }

    static Double batteryTenthsCelsius(int raw) {
        return raw >= -400 && raw <= 1500 ? raw / 10.0 : null;
    }

    /** Xperia exposes a "socd" thermal-control zone whose value is not a temperature unit. */
    static boolean temperatureUnitVerified(String vendorType) {
        if (vendorType == null || vendorType.trim().isEmpty()) return false;
        return !"socd".equals(vendorType.trim().toLowerCase(Locale.ROOT));
    }

    static String label(String value) {
        if (value == null) return "";
        String bounded = value.length() > 96 ? value.substring(0, 96) : value;
        for (int i = 0; i < bounded.length(); i++) if (Character.isISOControl(bounded.charAt(i))) {
            StringBuilder clean = new StringBuilder(bounded);
            for (int j = i; j < clean.length(); j++) if (Character.isISOControl(clean.charAt(j))) clean.setCharAt(j, '?');
            return clean.toString();
        }
        return bounded;
    }
}
