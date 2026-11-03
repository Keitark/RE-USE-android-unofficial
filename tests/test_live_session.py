"""Host JNI + actual live session lifecycle/count proof, without Android or model execution."""
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / ".research/live-session-test"


def run(command, timeout=60, **kwargs):
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace",
                            timeout=timeout, **kwargs)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


@pytest.fixture(scope="session")
def session_java():
    cmake, javac, java = (shutil.which(name) for name in ("cmake", "javac", "java"))
    if not all((cmake, javac, java)):
        pytest.skip("CMake and a JDK are needed for live JNI/session checks")
    java_home = Path(javac).resolve().parent.parent
    env = dict(os.environ, JAVA_HOME=str(java_home))
    config = [cmake, "-S", str(ROOT / "native/live"), "-B", str(BUILD / "native")]
    if os.name == "nt": config += ["-G", "Visual Studio 17 2022", "-A", "x64"]
    else: config += ["-DCMAKE_BUILD_TYPE=Release"]
    run(config, env=env)
    run([cmake, "--build", str(BUILD / "native"), "--config", "Release", "--parallel", "2"], env=env)
    stubs = BUILD / "stubs/io/github/reuse"
    stubs.mkdir(parents=True, exist_ok=True)
    (stubs / "ChunkedReuseEnhancer.java").write_text("""package io.github.reuse;
import java.nio.file.Path;
public final class ChunkedReuseEnhancer {
 public ChunkedReuseEnhancer(Path m,Path j,ReuseConfig c,Path l){throw new AssertionError("Injected engine required in host test");}
 public float[] process(float[] a){throw new AssertionError();}
 public float[] flush(){throw new AssertionError();} public void reset(){} public void close(){}
 public void verifyAcceleratorExecution(){throw new AssertionError();}
 public java.util.Map<String,Object> executionDiagnostics(){return java.util.Collections.emptyMap();}
}
""")
    classes = BUILD / "classes"
    classes.mkdir(exist_ok=True)
    production = ROOT / "android/reuse-audio/src/main/java/io/github/reuse/live"
    run([javac, "-d", str(classes), str(ROOT / "android/reuse/src/main/java/io/github/reuse/ReuseConfig.java"),
         str(ROOT / "android/reuse/src/main/java/io/github/reuse/RecommendedCpuProfile.java"), str(stubs / "ChunkedReuseEnhancer.java"),
         str(production / "FirResampler.java"), str(production / "EnhancementSession.java"),
         str(ROOT / "tests/live/SessionHarness.java"), str(ROOT / "tests/live/PublicEngineFactoryHarness.java")])
    native = BUILD / "native" / ("Release" if os.name == "nt" else "")
    return java, classes, native


@pytest.mark.parametrize("case", ["lengths", "overflow", "abort", "empty", "native", "sequence", "duration", "flush", "listener", "snapshots", "backend", "verification"])
def test_live_session_contract(session_java, case, tmp_path):
    java, classes, native = session_java
    result = run([java, f"-Djava.library.path={native}", "-cp", str(classes),
                  "io.github.reuse.live.SessionHarness", case, str(tmp_path / case)], timeout=30)
    report = json.loads(result.stdout)
    assert report["passed"] is True
    (BUILD / f"{case}.json").write_text(json.dumps(report, indent=2) + "\n")


@pytest.mark.parametrize("case", ["factory_lifecycle", "factory_failure", "factory_queue", "factory_nulls",
                                "factory_details_processing", "factory_details_bound"])
def test_public_engine_factory(session_java, case, tmp_path):
    java, classes, native = session_java
    result = run([java, f"-Djava.library.path={native}", "-cp", str(classes),
                  "io.github.reuse.externaltest.PublicEngineFactoryHarness", case, str(tmp_path / case)], timeout=30)
    report = json.loads(result.stdout)
    assert report["passed"] is True
    (BUILD / f"{case}.json").write_text(json.dumps(report, indent=2) + "\n")
