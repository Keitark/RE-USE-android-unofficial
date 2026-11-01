"""Compile the native-only DSP checks; no Android/JNI or model runtime is needed."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / ".research" / "even-dsp-test"
CASES = ["fft_8000", "fft_16000", "frontend_8000", "frontend_16000",
         "analysis_8000", "analysis_16000", "resampler", "resampler_exact", "frame_queue"]


@pytest.fixture(scope="session")
def native_dsp_binary():
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("CMake is required for the independent native DSP checks")
    configure = [cmake, "-S", str(ROOT / "tests/native_even_dsp"), "-B", str(BUILD)]
    if os.name == "nt":
        configure += ["-G", "Visual Studio 17 2022", "-A", "x64"]
    else:
        configure += ["-DCMAKE_BUILD_TYPE=Release"]
    for command in (configure, [cmake, "--build", str(BUILD), "--config", "Release", "--parallel", "2"]):
        result = subprocess.run(command, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr
    binary = BUILD / ("Release/native_even_dsp_test.exe" if os.name == "nt" else "native_even_dsp_test")
    assert binary.is_file()
    return binary


@pytest.mark.parametrize("case", CASES)
def test_independent_native_dsp(native_dsp_binary, case):
    result = subprocess.run([str(native_dsp_binary), case], capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=15)
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["ok"] is True
    (BUILD / f"{case}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
