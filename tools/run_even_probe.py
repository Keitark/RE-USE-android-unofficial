"""Run/pull the separate ARM64 codec/model harness without replacing the Even app."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "io.github.reuse.evenprobe"
FILES = ("init-result.json", "probe-result.json", "enhanced.pcm16le",
         "enhanced.pcm16le.baseline.pcm16le", "enhanced.pcm16le.direct.pcm16le")
RUNTIME_PROFILES = ("default", "no_spin", "bounded_spin", "dynamic4")


def runtime_profile_launch_arguments(profile):
    if profile is None:
        return []
    if not isinstance(profile, str) or profile not in RUNTIME_PROFILES:
        raise ValueError("Unsupported runtime profile")
    return ["--es", "runtimeProfile", profile]


def runtime_profile_evidence(requested, report):
    native = report.get("diagnostics", {}).get("runtime_profile")
    effective = report.get("effective_runtime_profile", native)
    return {"requested_runtime_profile": requested,
            "effective_runtime_profile": effective, "native_runtime_profile": native,
            "runtime_profile_matches_requested": requested is None or
                (effective == requested and native == requested)}


def main(args):
    prefix = [str(args.adb), "-s", args.serial]

    def adb(*parts, check=True):
        return subprocess.run(prefix + list(parts), check=check, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=120)

    def screen_state():
        power = adb("shell", "dumpsys", "power", check=False).stdout.decode(errors="replace")
        policy = adb("shell", "dumpsys", "window", "policy", check=False).stdout.decode(errors="replace")
        wake = re.search(r"\bmWakefulness=(\w+)", power)
        display = re.search(r"\bDisplay Power:\s*state=(\w+)", power)
        keyguard = re.search(r"\b(?:mKeyguardShowing|keyguardShowing|isKeyguardShowing)=(true|false)", policy)
        if keyguard is None:
            delegate = re.search(r"KeyguardServiceDelegate(?P<section>[\s\S]{0,2500})", policy)
            if delegate:
                keyguard = re.search(r"\bshowing=(true|false)", delegate.group("section"))
        state = {"wakefulness": wake.group(1) if wake else "unknown",
                 "display_power": display.group(1) if display else "unknown",
                 "keyguard_showing": keyguard.group(1) == "true" if keyguard else None}
        state["clear_background_or_doze"] = state["wakefulness"].lower() in ("asleep", "dozing", "dreaming") or state["display_power"].upper() in ("OFF", "DOZE", "DOZE_SUSPEND")
        state["foreground_conditions_confirmed"] = (state["wakefulness"] == "Awake" and state["display_power"] in ("ON", "unknown") and state["keyguard_showing"] is False)
        return state

    prior_pid = ""
    previous_started = -1
    screen_before = screen_state()
    if args.warm and args.pull_only:
        raise ValueError("--warm and --pull-only are mutually exclusive")
    args.output.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        (args.output / name).unlink(missing_ok=True)
    if not args.pull_only:
        if screen_before["clear_background_or_doze"] and not args.allow_background:
            raise RuntimeError("Phone is Dozing/asleep or its display is off. Wake and unlock it, then retry; --allow-background is only for an intentional background test.")
        if not args.input and not args.warm:
            raise ValueError("--input must be 16 kHz mono PCM16LE")
        if args.input:
            size = args.input.stat().st_size
            if not size or size % 2:
                raise ValueError("Expected nonempty PCM16LE input")
        if args.warm:
            process = adb("shell", "pidof", PACKAGE, check=False)
            prior_pid = process.stdout.decode().strip()
            if process.returncode != 0 or not prior_pid:
                raise RuntimeError("--warm requires an already running probe process; run a cold test first")
            prior_report = adb("shell", "run-as", PACKAGE, "cat", "files/reuse/probe-result.json", check=False)
            if prior_report.returncode == 0:
                previous_started = json.loads(prior_report.stdout).get("started_at_epoch_ms", -1)
        else:
            adb("install", "-r", str(args.apk))
            adb("shell", "am", "force-stop", PACKAGE)
        if args.input:
            temporary = "/data/local/tmp/reuse-even-input.pcm16le"
            adb("push", str(args.input), temporary)
            adb("shell", "chmod", "644", temporary)
            adb("shell", "run-as", PACKAGE, "mkdir", "-p", "files/reuse")
            adb("shell", "run-as", PACKAGE, "cp", temporary, "files/reuse/input.pcm16le")
            adb("shell", "rm", "-f", temporary)
        # Clear every prior result before this run. In particular an init
        # failure must never leave old PCM looking like this run's output.
        adb("shell", "run-as", PACKAGE, "rm", "-f", *("files/reuse/" + name for name in FILES))
        launch = ["shell", "am", "start"]
        if args.warm:
            # NEW_TASK | MULTIPLE_TASK creates another standard Activity while
            # keeping the app process and its existing OrtSession alive.
            launch += ["-f", "0x18000000", "--ez", "warmRequested", "true"]
        if args.model_rate is not None:
            launch += ["--ei", "modelRate", str(args.model_rate)]
        if args.threads is not None:
            launch += ["--ei", "threads", str(args.threads)]
        launch += runtime_profile_launch_arguments(args.runtime_profile)
        if args.hold_screen:
            launch += ["--ez", "holdScreen", "true"]
        launch += ["-n", PACKAGE + "/io.github.reuse.even.ReuseProbeActivity"]
        adb(*launch)
        deadline = time.monotonic() + args.timeout
        while True:
            result = adb("shell", "run-as", PACKAGE, "cat", "files/reuse/probe-result.json", check=False)
            if result.returncode == 0:
                latest = json.loads(result.stdout)
                if latest.get("started_at_epoch_ms", -1) > previous_started:
                    break
            if time.monotonic() >= deadline:
                raise TimeoutError("Probe did not finish; inspect app logcat and initialization status")
            time.sleep(2)
    # The fresh report is mandatory. Native initialization may fail before any
    # PCM exists, which is a valid failed test receipt rather than a pull error.
    report_bytes = adb("exec-out", "run-as", PACKAGE, "cat", "files/reuse/probe-result.json").stdout
    (args.output / "probe-result.json").write_bytes(report_bytes)
    report = json.loads(report_bytes)
    raw_self_test = report.get("self_test")
    has_self_test = isinstance(raw_self_test, dict)
    self_test = raw_self_test if has_self_test else {}
    profile_evidence = runtime_profile_evidence(args.runtime_profile, report)
    availability = {"probe-result.json": {"exists_remote": True, "retrieved": True, "bytes": len(report_bytes)}}
    for name in FILES:
        if name == "probe-result.json":
            continue
        # Binary-safe and optional: an early failure produces no audio files.
        exists = adb("shell", "run-as", PACKAGE, "test", "-f", "files/reuse/" + name, check=False)
        found = exists.returncode == 0
        result = (adb("exec-out", "run-as", PACKAGE, "cat", "files/reuse/" + name, check=False)
                  if found else exists)
        record = {"exists_remote": found, "retrieved": False}
        if found and result.returncode == 0 and (name == "init-result.json" or has_self_test):
            (args.output / name).write_bytes(result.stdout)
            record.update(retrieved=True, bytes=len(result.stdout), sha256=hashlib.sha256(result.stdout).hexdigest())
        elif found and not has_self_test and name != "init-result.json":
            record["reason"] = "No native self-test in current report; PCM may belong to an older run and is not retained"
        else:
            record["reason"] = result.stderr.decode(errors="replace").strip() or "Not produced by this run"
        availability[name] = record
    if not availability["init-result.json"]["retrieved"] and isinstance(report.get("initialization"), dict):
        # The startup provider runs only once per process. After a warm run
        # clears old output files, its current status is still captured inside
        # the fresh probe receipt; retain that with explicit provenance.
        data = (json.dumps(report["initialization"], indent=2) + "\n").encode("utf-8")
        (args.output / "init-result.json").write_bytes(data)
        availability["init-result.json"].update(retrieved=True, bytes=len(data), source="probe-result.json.initialization")
    remote = adb("shell", "pm", "path", PACKAGE).stdout.decode().strip().removeprefix("package:")
    if not remote.startswith("/data/app/") or "\n" in remote:
        raise ValueError("Unexpected harness APK path")
    installed_hash = adb("shell", "sha256sum", remote).stdout.decode().split()[0]
    local_hash = hashlib.sha256(args.apk.read_bytes()).hexdigest()
    screen_after = screen_state()
    is_emulator = adb("shell", "getprop", "ro.kernel.qemu").stdout.decode().strip() == "1"
    metadata = {"package": PACKAGE, "serial": args.serial, "apk_sha256": local_hash,
                "installed_apk_sha256": installed_hash, "exact_apk_match": local_hash == installed_hash,
                "system_abi": adb("shell", "getprop", "ro.product.cpu.abi").stdout.decode().strip(),
                "packaged_native_abi": "arm64-v8a", "device_is_emulator": is_emulator,
                "warm_requested": args.warm, "prior_process_id": prior_pid,
                "current_process_id": adb("shell", "pidof", PACKAGE, check=False).stdout.decode().strip(),
                "session_reconfigured": report.get("session_reconfigured"),
                "session_reused": report.get("session_reused"),
                "requested_model_rate": args.model_rate, "requested_threads": args.threads,
                "hold_screen_requested": args.hold_screen, "hold_screen": report.get("hold_screen", False),
                "self_test_present": has_self_test, "output_availability": availability,
                "device_model": adb("shell", "getprop", "ro.product.model").stdout.decode().strip(),
                "android_version": adb("shell", "getprop", "ro.build.version.release").stdout.decode().strip(),
                "screen_before": screen_before, "screen_after": screen_after,
                "intentional_background_allowed": args.allow_background,
                "foreground_vs_doze": ("doze_or_background" if screen_before["clear_background_or_doze"] or screen_after["clear_background_or_doze"] else
                    ("foreground_conditions_confirmed" if screen_before["foreground_conditions_confirmed"] and screen_after["foreground_conditions_confirmed"] else "awake_or_unknown_keyguard")),
                "physical_phone_probe_verified": (local_hash == installed_hash and not is_emulator and report.get("ok", False)
                    and self_test.get("near_realtime_sustained", False)
                    and profile_evidence["runtime_profile_matches_requested"]),
                "live_g2_microphone_verified": False}
    metadata.update(profile_evidence)
    metadata["same_process_warm_launch"] = bool(args.warm and prior_pid == metadata["current_process_id"])
    if all(key in report for key in ("interactive_at_start", "interactive_at_finish", "keyguard_locked_at_start", "keyguard_locked_at_finish")):
        metadata["foreground_vs_doze"] = ("foreground_during_probe" if report["interactive_at_start"] and report["interactive_at_finish"]
            and not report["keyguard_locked_at_start"] and not report["keyguard_locked_at_finish"] else "noninteractive_or_locked_during_probe")
    thermal = adb("shell", "dumpsys", "thermalservice", check=False)
    (args.output / "thermal.txt").write_bytes(thermal.stdout)
    (args.output / "execution.json").write_text(json.dumps(metadata, indent=2)+"\n", encoding="utf-8")
    if not metadata["exact_apk_match"]:
        raise AssertionError("Installed harness APK differs from the retained artifact")
    if args.warm and not metadata["same_process_warm_launch"]:
        raise AssertionError("Probe app restarted during --warm; retained data is a cold measurement")
    if has_self_test and report.get("ok") and not profile_evidence["runtime_profile_matches_requested"]:
        raise AssertionError("Requested runtime profile differs from effective Java/native profile; retain this as a failed configuration test")
    if not has_self_test or not report.get("ok"):
        print(json.dumps({"ok": False, "error": report.get("error") or self_test.get("error") or
                          ("Native self-test did not execute" if not has_self_test else "Probe self-test failed"),
                          "self_test": raw_self_test, "initialization": report.get("initialization"),
                          "output_availability": availability}, indent=2))
    else:
        print(json.dumps(self_test, indent=2))
    print(f"Retained binary artifacts: {args.output}")
    return 0 if report.get("ok") and self_test.get("near_realtime_sustained") else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adb", type=Path, required=True)
    parser.add_argument("--serial", required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--apk", type=Path, default=ROOT / "android/even-probe-host/build/outputs/apk/debug/even-probe-host-debug.apk")
    parser.add_argument("--output", type=Path, default=ROOT / ".research/even-probe-final")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--pull-only", action="store_true")
    parser.add_argument("--warm", action="store_true", help="Reuse a running app process; never install or force-stop")
    parser.add_argument("--model-rate", type=int, choices=(8000, 16000))
    parser.add_argument("--threads", type=int, choices=range(1, 9))
    parser.add_argument("--runtime-profile", choices=RUNTIME_PROFILES,
                        help="Select a fixed native session profile; omitted retains the current profile")
    parser.add_argument("--allow-background", action="store_true", help="Explicitly permit a Dozing/asleep/background benchmark; never changes phone settings")
    parser.add_argument("--hold-screen", action="store_true", help="Keep the screen on while the debug result Activity stays visible; close/force-stop the probe to release")
    raise SystemExit(main(parser.parse_args()))
