package io.github.reuse.qnnprobe;

import static org.junit.Assert.*;
import org.junit.Test;

public final class BenchmarkTelemetryMathTest {
    @Test public void guestCountersAreNotCountedTwice() {
        BenchmarkTelemetryMath.CpuTimes value = BenchmarkTelemetryMath.parseProcStat(
                "cpu 100 20 30 400 50 6 7 8 80 10\ncpu0 1 2 3 4");
        assertEquals(621, value.totalTicks);
        assertEquals(171, value.busyTicks);
    }

    @Test public void busyPercentExcludesIdleAndIoWaitAcrossTheSameCounterWindow() {
        var before = BenchmarkTelemetryMath.parseProcStat("cpu 100 0 0 100 0 0 0 0");
        var after = BenchmarkTelemetryMath.parseProcStat("cpu 120 0 10 150 20 0 0 0");
        assertEquals(30.0, BenchmarkTelemetryMath.systemBusyPercent(before, after), 0);
    }

    @Test public void resetOrZeroCounterWindowsHaveNoInventedUtilization() {
        var before = BenchmarkTelemetryMath.parseProcStat("cpu 100 0 0 100 0 0 0 0");
        var reset = BenchmarkTelemetryMath.parseProcStat("cpu 200 0 0 50 0 0 0 0");
        assertNull(BenchmarkTelemetryMath.systemBusyPercent(before, before));
        assertNull(BenchmarkTelemetryMath.systemBusyPercent(before, reset));
        assertNull(BenchmarkTelemetryMath.systemBusyPercent(null, before));
    }

    @Test public void malformedNegativeAndOverflowCountersAreRejected() {
        for (String text : new String[]{"", "cpu0 1 2 3 4", "cpu 1 2", "cpu 1 -2 3 4",
                "cpu x 2 3 4", "cpu 9223372036854775807 1 0 0"}) {
            assertThrows(RuntimeException.class, () -> BenchmarkTelemetryMath.parseProcStat(text));
        }
    }

    @Test public void fourBasicCountersRemainSupported() {
        assertEquals(10, BenchmarkTelemetryMath.parseProcStat("cpu 1 2 3 4").totalTicks);
    }

    @Test public void processLoadMayExceedOneCore() {
        assertEquals(240.0, BenchmarkTelemetryMath.processOneCorePercent(100, 1300,
                1_000_000_000L, 1_500_000_000L), 0);
        assertEquals(0.0, BenchmarkTelemetryMath.processOneCorePercent(100, 100, 1, 500_000_001), 0);
    }

    @Test public void processLoadNeedsMonotonicTimeAndCpuCounters() {
        assertNull(BenchmarkTelemetryMath.processOneCorePercent(100, 99, 1, 2));
        assertNull(BenchmarkTelemetryMath.processOneCorePercent(0, 1, 2, 2));
        assertNull(BenchmarkTelemetryMath.processOneCorePercent(-1, 1, 1, 2));
    }

    @Test public void otherSystemWorkUsesCommonMillisecondsAndDoesNotAttributeApps() {
        var before = BenchmarkTelemetryMath.parseProcStat("cpu 100 0 0 100 0 0 0 0");
        var after = BenchmarkTelemetryMath.parseProcStat("cpu 200 0 0 100 0 0 0 0");
        // At HZ=100, 100 busy ticks are 1000 CPU-ms; 200 belong to this process.
        assertEquals(160.0, BenchmarkTelemetryMath.otherSystemOneCorePercent(before, after,
                200, 500_000_000L, 100), 0);
        assertNull(BenchmarkTelemetryMath.otherSystemOneCorePercent(before, after, 200, 0, 100));
        assertNull(BenchmarkTelemetryMath.otherSystemOneCorePercent(before, after, 200, 1, 0));
        assertNull(BenchmarkTelemetryMath.otherSystemOneCorePercent(before, after, 2000, 1_000_000_000, 100));
    }

    @Test public void loadAverageIgnoresTheLastPid() {
        assertArrayEquals(new double[]{1.2, 3.4, .5}, BenchmarkTelemetryMath.parseLoadAverage(
                "1.20 3.40 0.50 2/7000 DO_NOT_PARSE_PID"), 0);
        for (String text : new String[]{"", "1 2", "NaN 1 2", "-1 1 2", "Infinity 1 2"})
            assertThrows(RuntimeException.class, () -> BenchmarkTelemetryMath.parseLoadAverage(text));
    }

    @Test public void pressureUsesOnlyAggregatePressureFields() {
        var values = BenchmarkTelemetryMath.parseCpuPressure(
                "some avg10=1.25 avg60=2.00 avg300=3.50 total=12345\nfull avg10=0 avg60=0 avg300=0 total=0");
        assertEquals(1.25, values.get("some_avg10"), 0);
        assertEquals(12345.0, values.get("some_total"), 0);
        assertEquals(8, values.size());
        assertThrows(RuntimeException.class, () -> BenchmarkTelemetryMath.parseCpuPressure("some avg10=NaN"));
        assertThrows(RuntimeException.class, () -> BenchmarkTelemetryMath.parseCpuPressure("some avg10=101"));
    }

    @Test public void temperatureSentinelsStayInvalidRatherThanBecomingCpuReadings() {
        assertEquals(42.125, BenchmarkTelemetryMath.milliCelsius(42125), 0);
        assertEquals(-40.0, BenchmarkTelemetryMath.milliCelsius(-40000), 0);
        assertNull(BenchmarkTelemetryMath.milliCelsius(-273000));
        assertNull(BenchmarkTelemetryMath.milliCelsius(150001));
        assertNull(BenchmarkTelemetryMath.milliCelsius(Long.MAX_VALUE));
        assertEquals(31.5, BenchmarkTelemetryMath.batteryTenthsCelsius(315), 0);
        assertNull(BenchmarkTelemetryMath.batteryTenthsCelsius(Integer.MIN_VALUE));
    }

    @Test public void stageLabelsAreBoundedAndControlCharactersRemoved() {
        assertEquals("", BenchmarkTelemetryMath.label(null));
        assertEquals("a?b", BenchmarkTelemetryMath.label("a\nb"));
        assertEquals(96, BenchmarkTelemetryMath.label("a".repeat(200)).length());
    }

    @Test public void vendorControlSocdIsNotConvertedIntoASocTemperature() {
        assertFalse(BenchmarkTelemetryMath.temperatureUnitVerified("socd"));
        assertFalse(BenchmarkTelemetryMath.temperatureUnitVerified(" SOCD "));
        assertFalse(BenchmarkTelemetryMath.temperatureUnitVerified(null));
        for (String sensor : new String[]{"cpuss-0", "cpu-1-0", "gpuss-1", "soc-thermal", "battery", "skin"})
            assertTrue(sensor, BenchmarkTelemetryMath.temperatureUnitVerified(sensor));
        // A plausible numeric value alone does not establish a temperature unit.
        assertNotNull(BenchmarkTelemetryMath.milliCelsius(78));
        assertFalse(BenchmarkTelemetryMath.temperatureUnitVerified("socd"));
    }
}
