# SPDX-License-Identifier: MIT
"""Record bounded, read-only ADB-shell load telemetry beside APK diagnostics.

The shell has different permissions from an Android application. These records
must not be presented as APK-observed load or attribution to background apps.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import time

SOURCES = ("uptime", "stat", "loadavg", "cpu_pressure")
MAX_POLL_BYTES = 128 * 1024
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
MAX_RECORDS = 1201
MAX_CPUS = 256
BEGIN = "__REUSE_DEVICE_LOAD_BEGIN__ "
END = "__REUSE_DEVICE_LOAD_END__ "
SHELL_SCRIPT = """read_source() {
  printf '__REUSE_DEVICE_LOAD_BEGIN__ %s\\n' "$1"
  cat "$2" 2>&1
  result=$?
  printf '\\n__REUSE_DEVICE_LOAD_END__ %s %s\\n' "$1" "$result"
}
read_source uptime /proc/uptime
read_source stat /proc/stat
read_source loadavg /proc/loadavg
read_source cpu_pressure /proc/pressure/cpu
"""


class PollError(Exception):
    """A deliberately non-identifying transport error."""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def failure(reason):
    return {"available": False, "reason": reason}


def parse_blocks(text):
    """Parse the fixed protocol; never retain arbitrary shell output."""
    blocks = {}
    current = None
    content = []
    for line in text.replace("\r\n", "\n").splitlines():
        if line.startswith(BEGIN):
            name = line[len(BEGIN):]
            if current is not None or name not in SOURCES or name in blocks:
                raise ValueError("malformed_protocol")
            current, content = name, []
        elif line.startswith(END):
            fields = line[len(END):].split()
            if current is None or len(fields) != 2 or fields[0] != current:
                raise ValueError("malformed_protocol")
            try:
                status = int(fields[1])
            except ValueError as error:
                raise ValueError("malformed_protocol") from error
            value = "\n".join(content).strip()
            if status:
                lower = value.lower()
                reason = ("permission_denied" if "permission denied" in lower
                          else "missing" if "no such file" in lower else "read_error")
                blocks[current] = failure(reason)
            else:
                blocks[current] = {"available": True, "text": value}
            current, content = None, []
        elif current is not None:
            content.append(line)
        elif line.strip():
            raise ValueError("malformed_protocol")
    if current is not None:
        raise ValueError("incomplete_protocol")
    for name in SOURCES:
        blocks.setdefault(name, failure("missing_block"))
    return blocks


def nonnegative_number(text):
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid_nonnegative_number")
    return value


def parse_uptime(text):
    fields = text.split()
    if len(fields) != 2:
        raise ValueError("invalid_uptime")
    uptime = nonnegative_number(fields[0])
    nonnegative_number(fields[1])
    return {"seconds": uptime, "device_elapsed_realtime_ms": uptime * 1000,
            "scope": "Kernel uptime, read before the other sources; not an atomic device snapshot"}


def parse_cpu_stat(text):
    """Retain only aggregate/per-core counters, excluding guest double counting."""
    cpus = {}
    for line in text.splitlines():
        parts = line.split()
        if not parts or not re.fullmatch(r"cpu(?:[0-9]+)?", parts[0]):
            continue
        if parts[0] in cpus or len(parts) < 5 or len(cpus) >= MAX_CPUS + 1:
            raise ValueError("invalid_cpu_stat")
        try:
            values = [int(x) for x in parts[1:]]
        except ValueError as error:
            raise ValueError("invalid_cpu_stat") from error
        if any(x < 0 for x in values):
            raise ValueError("invalid_cpu_stat")
        # guest/guest_nice are already included in user/nice. Do not add twice.
        selected = (values + [0] * 8)[:8]
        total = sum(selected)
        idle = selected[3] + selected[4]
        cpus[parts[0]] = {"counters": selected, "total_ticks": total,
                         "idle_and_iowait_ticks": idle, "busy_ticks": total - idle}
    if "cpu" not in cpus:
        raise ValueError("missing_aggregate_cpu")
    return cpus


def cpu_delta(previous, current):
    if previous is None:
        return failure("baseline_required")
    if any(n < o for n, o in zip(current["counters"], previous["counters"])):
        return failure("counter_reset")
    total = current["total_ticks"] - previous["total_ticks"]
    busy = current["busy_ticks"] - previous["busy_ticks"]
    if total <= 0 or busy < 0 or busy > total:
        return failure("invalid_or_zero_interval")
    return {"available": True, "busy_percent": busy / total * 100,
            "busy_ticks_delta": busy, "total_ticks_delta": total,
            "scope": "Logical CPU accounting; idle and iowait excluded; not background-app attribution"}


def parse_loadavg(text):
    fields = text.split()
    if len(fields) != 5:
        raise ValueError("invalid_loadavg")
    loads = [nonnegative_number(x) for x in fields[:3]]
    match = re.fullmatch(r"([0-9]+)/([0-9]+)", fields[3])
    if match is None or not fields[4].isdigit():
        raise ValueError("invalid_loadavg")
    runnable, tasks = map(int, match.groups())
    if runnable > tasks or tasks < 1:
        raise ValueError("invalid_loadavg")
    # The last field is a PID. Validate but deliberately never record it.
    return {"one_minute": loads[0], "five_minutes": loads[1], "fifteen_minutes": loads[2],
            "runnable_tasks": runnable, "total_tasks": tasks,
            "scope": "Load averages and instantaneous runnable/task counts, not CPU utilization percentages"}


def parse_pressure(text):
    result = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or fields[0] not in ("some", "full") or fields[0] in result:
            raise ValueError("invalid_cpu_pressure")
        values = dict(item.split("=", 1) for item in fields[1:])
        if set(values) != {"avg10", "avg60", "avg300", "total"}:
            raise ValueError("invalid_cpu_pressure")
        averages = {k: nonnegative_number(values[k]) for k in ("avg10", "avg60", "avg300")}
        if any(v > 100 for v in averages.values()) or not values["total"].isdigit():
            raise ValueError("invalid_cpu_pressure")
        result[fields[0]] = {**averages, "total_stall_microseconds": int(values["total"])}
    if "some" not in result:
        raise ValueError("invalid_cpu_pressure")
    return {**result, "scope": "CPU pressure stall information, not app identity or CPU usage percentage"}


class Decoder:
    def __init__(self):
        self.previous_stat = None
        self.previous_uptime_ms = None

    def decode(self, text):
        blocks = parse_blocks(text)
        record = {}
        parsed = {}
        parsers = {"uptime": parse_uptime, "stat": parse_cpu_stat,
                   "loadavg": parse_loadavg, "cpu_pressure": parse_pressure}
        for name, parser in parsers.items():
            source = blocks[name]
            if not source["available"]:
                record[name] = source
                continue
            try:
                parsed[name] = parser(source["text"])
                if name != "stat":
                    record[name] = {"available": True, **parsed[name]}
            except (ValueError, IndexError, OverflowError):
                record[name] = failure("invalid_data")
        uptime_ms = parsed.get("uptime", {}).get("device_elapsed_realtime_ms")
        if "stat" in parsed:
            current = parsed["stat"]
            previous = self.previous_stat
            window = (uptime_ms - self.previous_uptime_ms
                      if uptime_ms is not None and self.previous_uptime_ms is not None else None)
            reset = window is not None and window <= 0
            values = {name: failure("device_clock_reset") if reset else
                      cpu_delta(None if previous is None else previous.get(name), value)
                      for name, value in current.items()}
            record["stat"] = {"available": True, "window_ms": None if reset else window,
                              "aggregate": values.pop("cpu"), "per_cpu": values,
                              "logical_cpu_count": len(current) - 1}
            self.previous_stat = current
            self.previous_uptime_ms = uptime_ms
        return record


def poll(adb, serial, timeout=10):
    # This fixed quoted remote command is the transport verified on the Xperia.
    # `adb exec-out sh` with a stdin script can hang on this device/host pair.
    remote = "sh -c " + shlex.quote(SHELL_SCRIPT)
    command = [str(adb), "-s", serial, "shell", remote]
    process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    try:
        stdout, _ = process.communicate(timeout=timeout)
    except BaseException:
        process.kill()
        process.communicate()
        raise
    if process.returncode:
        raise PollError("adb_exit_nonzero")
    if len(stdout) > MAX_POLL_BYTES:
        raise PollError("poll_output_limit")
    return stdout.decode("utf-8", errors="strict")


def validate_args(args):
    if not args.adb.is_file():
        raise ValueError("ADB executable is not a file")
    if not re.fullmatch(r"[A-Za-z0-9_.:\[\]-]{1,256}", args.serial):
        raise ValueError("Invalid explicit device selector")
    if not math.isfinite(args.duration) or not 0 < args.duration <= 600:
        raise ValueError("Duration must be positive and at most 600 seconds")
    if not math.isfinite(args.interval) or not .5 <= args.interval <= 5:
        raise ValueError("Interval must be between 0.5 and 5 seconds")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+", args.package):
        raise ValueError("Invalid package metadata")
    if args.output.suffix.lower() != ".jsonl":
        raise ValueError("Output must have a .jsonl extension")
    summary = Path(str(args.output) + ".summary.json")
    if args.output.exists() or summary.exists():
        raise ValueError("Output or summary already exists; choose new paths")
    if not args.output.parent.is_dir():
        raise ValueError("Output directory must already exist")
    return summary


def record(args, poller=poll):
    summary_path = validate_args(args)
    summary = {"schema_version": 1, "source": "adb_shell_device_load_sidecar",
               "package": args.package, "started_utc": utc_now(),
               "requested_duration_seconds": args.duration, "requested_interval_seconds": args.interval,
               "samples": 0, "polls_attempted": 0, "transport_errors": 0, "parse_errors": 0,
               "source_unavailable_samples": {name: 0 for name in SOURCES},
               "poll_roundtrip_total_ms": 0.0, "poll_roundtrip_max_ms": 0.0,
               "host_recorder_cpu_total_ms": 0.0,
               "observer_overhead_scope": "Roundtrip includes ADB launch, fixed device reads and host parsing; host_recorder_cpu_ms measures Python recorder CPU only, excluding ADB child and device shell CPU. Device reads themselves contribute some observed system work.",
               "scope": "ADB shell permissions, separate from APK. No other-app process names, PIDs or device serial are recorded. No background-app causal attribution.",
               "application_cpu_scope": "Use the APK's app-process CPU-time log; this sidecar records only global accounting/queue/pressure.",
               "alignment": "Match /proc/uptime device_elapsed_realtime_ms to APK elapsedRealtime timestamps; sources are sequential and host roundtrip bounds observation skew.",
               "status": "RUNNING"}
    decoder = Decoder()
    began = time.monotonic()
    deadline = began + args.duration
    next_poll = began
    bytes_written = 0
    consecutive_errors = 0
    # Both exclusive opens happen before device reads. An existing file is never replaced.
    with summary_path.open("x", encoding="utf-8") as summary_file:
        try:
            with args.output.open("x", encoding="utf-8", newline="\n") as output:
                began = time.monotonic()
                deadline = began + args.duration
                next_poll = began
                try:
                    while time.monotonic() < deadline and summary["samples"] < MAX_RECORDS:
                        if next_poll >= deadline:
                            time.sleep(max(0, deadline - time.monotonic()))
                            break
                        delay = next_poll - time.monotonic()
                        if delay > 0:
                            time.sleep(min(delay, max(0, deadline - time.monotonic())))
                        if time.monotonic() >= deadline:
                            break
                        start = time.monotonic()
                        host_cpu_start = time.process_time()
                        summary["polls_attempted"] += 1
                        item = {"schema_version": 1, "source": summary["source"],
                                "sample_index": summary["samples"], "host_utc": utc_now(),
                                "host_elapsed_ms": (start - began) * 1000,
                                "schedule_lag_ms": max(0, start - next_poll) * 1000}
                        try:
                            text = poller(args.adb, args.serial)
                            item.update(decoder.decode(text))
                            consecutive_errors = 0
                            for name in SOURCES:
                                if not item[name]["available"]:
                                    summary["source_unavailable_samples"][name] += 1
                        except (PollError, subprocess.TimeoutExpired, OSError, UnicodeError) as error:
                            reason = (str(error) if isinstance(error, PollError) and str(error) in (
                                      "adb_exit_nonzero", "poll_output_limit") else
                                      "timeout" if isinstance(error, subprocess.TimeoutExpired) else "transport_error")
                            item["transport"] = failure(reason)
                            summary["transport_errors"] += 1
                            consecutive_errors += 1
                        except ValueError:
                            item["transport"] = failure("invalid_protocol")
                            summary["parse_errors"] += 1
                            consecutive_errors += 1
                        elapsed_ms = (time.monotonic() - start) * 1000
                        host_cpu_ms = max(0.0, time.process_time() - host_cpu_start) * 1000
                        item["poll_roundtrip_ms"] = elapsed_ms
                        item["host_recorder_cpu_ms"] = host_cpu_ms
                        summary["host_recorder_cpu_total_ms"] += host_cpu_ms
                        summary["poll_roundtrip_total_ms"] += elapsed_ms
                        summary["poll_roundtrip_max_ms"] = max(summary["poll_roundtrip_max_ms"], elapsed_ms)
                        line = json.dumps(item, separators=(",", ":"), allow_nan=False) + "\n"
                        line_bytes = len(line.encode("utf-8"))
                        if bytes_written + line_bytes > MAX_OUTPUT_BYTES:
                            summary["status"] = "OUTPUT_LIMIT"
                            break
                        output.write(line); output.flush()
                        bytes_written += line_bytes
                        summary["samples"] += 1
                        if consecutive_errors >= 3:
                            summary["status"] = "TRANSPORT_OR_PROTOCOL_FAILURE"
                            break
                        # Skip missed slots; never launch a burst of catch-up shell processes.
                        next_poll += args.interval
                        if next_poll < time.monotonic():
                            next_poll = time.monotonic() + args.interval
                    else:
                        if summary["samples"] >= MAX_RECORDS:
                            summary["status"] = "SAMPLE_LIMIT"
                    if summary["status"] == "RUNNING":
                        summary["status"] = "COMPLETE"
                except KeyboardInterrupt:
                    summary["status"] = "CANCELLED_PARTIAL"
        except BaseException:
            summary["status"] = "ERROR_PARTIAL"
            raise
        finally:
            summary["ended_utc"] = utc_now()
            summary["actual_duration_seconds"] = time.monotonic() - began
            summary["jsonl_bytes"] = bytes_written
            summary["mean_poll_roundtrip_ms"] = (summary["poll_roundtrip_total_ms"] / summary["polls_attempted"]
                                                  if summary["polls_attempted"] else None)
            json.dump(summary, summary_file, indent=2, allow_nan=False); summary_file.write("\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", type=Path, required=True)
    parser.add_argument("--serial", required=True, help="Explicit selector, never written to output")
    parser.add_argument("--duration", type=float, default=240)
    parser.add_argument("--interval", type=float, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package", default="io.github.reuse.qnnprobe")
    args = parser.parse_args()
    try:
        result = record(args)
    except (ValueError, OSError) as error:
        parser.exit(2, f"Recorder error: {type(error).__name__}; check arguments/output access.\n")
    print(json.dumps({"status": result["status"], "samples": result["samples"],
                      "transport_errors": result["transport_errors"], "parse_errors": result["parse_errors"]}))
    return 0 if result["status"] in ("COMPLETE", "CANCELLED_PARTIAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
