# SPDX-License-Identifier: MIT
"""No device calls: parser/accounting/protocol/privacy checks for the sidecar."""
from __future__ import annotations

from argparse import Namespace
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import record_benchmark_device_load as load


def protocol(stat="cpu 10 0 5 85 0 0 0 0 2 0\ncpu0 10 0 5 85 0 0 0 0 2 0", uptime="100.00 700.00"):
    values = {"uptime": uptime, "stat": stat, "loadavg": "1.20 0.80 0.40 2/800 99999",
              "cpu_pressure": "some avg10=1.25 avg60=0.10 avg300=0.01 total=500000\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0"}
    return "".join(load.BEGIN + name + "\n" + text + "\n" + load.END + name + " 0\n"
                   for name, text in values.items())


class DeviceLoadRecorderTest(unittest.TestCase):
    def test_guest_fields_are_not_double_counted_and_per_core_preserved(self):
        cpus = load.parse_cpu_stat("cpu 10 1 4 80 5 2 3 1 8 1\ncpu0 6 0 1 40\nintr 999 3 2\n")
        self.assertEqual(cpus["cpu"]["total_ticks"], 106)
        self.assertEqual(cpus["cpu"]["busy_ticks"], 21)
        self.assertEqual(set(cpus), {"cpu", "cpu0"})

    def test_cpu_delta_distinguishes_baseline_idle_and_counter_reset(self):
        a = load.parse_cpu_stat("cpu 10 0 5 85")["cpu"]
        b = load.parse_cpu_stat("cpu 30 0 15 155")["cpu"]
        self.assertEqual(load.cpu_delta(a, b)["busy_percent"], 30)
        self.assertEqual(load.cpu_delta(None, b)["reason"], "baseline_required")
        self.assertEqual(load.cpu_delta(b, a)["reason"], "counter_reset")
        self.assertEqual(load.cpu_delta(a, a)["reason"], "invalid_or_zero_interval")

    def test_stat_rejects_bad_duplicate_or_missing_aggregate(self):
        for text in ("cpu0 1 2 3 4", "cpu 1 2 3 -4", "cpu a 2 3 4", "cpu 1 2 3 4\ncpu 1 2 3 4"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                load.parse_cpu_stat(text)

    def test_loadavg_is_queue_data_and_drops_pid(self):
        value = load.parse_loadavg("1.20 0.80 0.40 2/800 99999")
        self.assertEqual(value["runnable_tasks"], 2)
        self.assertEqual(value["total_tasks"], 800)
        self.assertNotIn("99999", json.dumps(value))
        self.assertIn("not CPU utilization", value["scope"])
        for text in ("nan 0 0 1/2 1", "0 0 0 3/2 1", "0 0 0 1/2"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                load.parse_loadavg(text)

    def test_pressure_units_and_missing_or_invalid_data(self):
        value = load.parse_pressure("some avg10=1.25 avg60=0 avg300=0 total=500000")
        self.assertEqual(value["some"]["total_stall_microseconds"], 500000)
        self.assertEqual(value["some"]["avg10"], 1.25)
        for text in ("full avg10=0 avg60=0 avg300=0 total=0", "some avg10=101 avg60=0 avg300=0 total=0"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                load.parse_pressure(text)

    def test_protocol_failures_never_turn_into_zero_load(self):
        text = load.BEGIN + "stat\ncat: /proc/stat: Permission denied\n" + load.END + "stat 1\n"
        value = load.Decoder().decode(text)
        self.assertEqual(value["stat"], {"available": False, "reason": "permission_denied"})
        self.assertEqual(value["uptime"]["reason"], "missing_block")
        malformed = [load.BEGIN + "stat\n1\n", load.BEGIN + "bad\n", load.END + "stat 0\n",
                     "device 'PRIVATE_SELECTOR' not found"]
        for text in malformed:
            with self.subTest(text=text), self.assertRaises(ValueError):
                load.parse_blocks(text)

    def test_decoder_uses_device_uptime_for_cross_record_window(self):
        decoder = load.Decoder()
        a = decoder.decode(protocol())
        self.assertFalse(a["stat"]["aggregate"]["available"])
        b = decoder.decode(protocol("cpu 30 0 15 155\ncpu0 30 0 15 155", "101.00 707.00"))
        self.assertEqual(b["stat"]["window_ms"], 1000)
        self.assertEqual(b["stat"]["aggregate"]["busy_percent"], 30)
        c = decoder.decode(protocol(uptime="1.00 7.00"))
        self.assertEqual(c["stat"]["aggregate"]["reason"], "device_clock_reset")

    def test_source_parse_error_preserves_other_sources(self):
        value = load.Decoder().decode(protocol(uptime="bad data"))
        self.assertEqual(value["uptime"]["reason"], "invalid_data")
        self.assertTrue(value["loadavg"]["available"])
        self.assertTrue(value["stat"]["available"])

    def arguments(self, folder):
        fake_adb = folder / "adb.exe"; fake_adb.write_bytes(b"fake")
        return Namespace(adb=fake_adb, serial="PRIVATE_SELECTOR", duration=.05, interval=.5,
                         output=folder / "new.jsonl", package="io.github.reuse.qnnprobe")

    def test_exclusive_outputs_and_argument_limits(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(Path(temporary))
            for key, bad in (("duration", 601), ("duration", float("nan")), ("interval", .1),
                             ("serial", "bad selector"), ("package", "bad;package")):
                valid = getattr(args, key); setattr(args, key, bad)
                with self.subTest(key=key), self.assertRaises(ValueError): load.validate_args(args)
                setattr(args, key, valid)
            args.output.write_text("preserve")
            with self.assertRaises(ValueError): load.validate_args(args)
            self.assertEqual(args.output.read_text(), "preserve")

    def test_transport_command_is_explicit_and_fixed_without_stdin_script(self):
        process = unittest.mock.Mock(returncode=0)
        process.communicate.return_value = (protocol().encode(), b"")
        with patch.object(load.subprocess, "Popen", return_value=process) as popen:
            self.assertIn(load.BEGIN, load.poll(Path("adb.exe"), "PRIVATE_SELECTOR"))
        command = popen.call_args.args[0]
        self.assertEqual(command[:4], ["adb.exe", "-s", "PRIVATE_SELECTOR", "shell"])
        self.assertIn("/proc/stat", command[4])
        self.assertEqual(popen.call_args.kwargs["stdin"], subprocess.DEVNULL)
        self.assertNotIn("input", process.communicate.call_args.kwargs)

    def test_cancellation_saves_partial_summary_without_serial(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(Path(temporary))
            def cancelled(adb, serial): raise KeyboardInterrupt()
            summary = load.record(args, poller=cancelled)
            saved = json.loads(Path(str(args.output) + ".summary.json").read_text())
            self.assertEqual(summary["status"], "CANCELLED_PARTIAL")
            self.assertEqual(saved["samples"], 0)
            self.assertNotIn(args.serial, json.dumps(saved))
            self.assertTrue(args.output.is_file())

    def test_successful_poll_and_failure_are_non_identifying(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.arguments(Path(temporary))
            summary = load.record(args, poller=lambda adb, serial: protocol())
            row = json.loads(args.output.read_text())
            self.assertEqual(summary["status"], "COMPLETE")
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(row["source"], "adb_shell_device_load_sidecar")
            self.assertEqual(row["uptime"]["device_elapsed_realtime_ms"], 100000)
            self.assertNotIn(args.serial, args.output.read_text())
            self.assertNotIn("99999", args.output.read_text())
            self.assertGreaterEqual(row["poll_roundtrip_ms"], 0)

    def test_transport_failures_are_saved_without_raw_adb_selector(self):
        for kind in ("exit", "timeout"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                args = self.arguments(Path(temporary))
                def failed(adb, serial):
                    if kind == "exit":
                        raise load.PollError("adb_exit_nonzero")
                    raise subprocess.TimeoutExpired([str(adb), "-s", serial], 10)
                summary = load.record(args, poller=failed)
                row = json.loads(args.output.read_text())
                self.assertEqual(summary["transport_errors"], 1)
                self.assertEqual(row["transport"]["available"], False)
                self.assertEqual(row["transport"]["reason"], "adb_exit_nonzero" if kind == "exit" else "timeout")
                self.assertNotIn(args.serial, args.output.read_text())
                self.assertNotIn(args.serial, Path(str(args.output) + ".summary.json").read_text())


if __name__ == "__main__":
    unittest.main()
