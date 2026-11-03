# SPDX-License-Identifier: MIT
"""Package a pinned experimental NNAPI GPU-only runtime with stock Java/JNI.

The native C API remains ORT 1.23.2. This is an arm64-only research artifact;
it deliberately changes CPU_DISABLED from non-CPU devices to GPU devices only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import struct
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

ORT_COMMIT = "a83fc4d58cb48eb68890dd689f94f28288cf2278"
ONNX_COMMIT = "e709452ef2bbc1d113faf678c24e6d3467696e83"
STOCK_SHA256 = "82048d1f462218adae4ba76477089ab0ba76093d84f733540066db1a8ba6b827"
MARKER = "REUSE_NNAPI_GPU_ONLY_V1"
VERSION = "1.23.2-reuse-gpuonly1"
ARTIFACT = "onnxruntime-android-gpuonly"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def validate_elf(data: bytes, *, require_marker: bool = True) -> list[int]:
    if data[:6] != b"\x7fELF\x02\x01" or struct.unpack_from("<H", data, 18)[0] != 183:
        raise ValueError("Expected an ELF64 little-endian AArch64 shared library")
    phoff = struct.unpack_from("<Q", data, 32)[0]
    phsize, phnum = struct.unpack_from("<HH", data, 54)
    alignments = []
    for index in range(phnum):
        offset = phoff + index * phsize
        if struct.unpack_from("<I", data, offset)[0] == 1:
            alignments.append(struct.unpack_from("<Q", data, offset + 48)[0])
    if not alignments or min(alignments) < 16384:
        raise ValueError(f"Native library is not 16 KiB LOAD-aligned: {alignments}")
    if require_marker and MARKER.encode() not in data:
        raise ValueError("Expected GPU-only runtime marker was not found")
    return alignments


def dynamic_dependencies(data: bytes) -> tuple[str | None, list[str]]:
    """Read ELF64 dynamic strings without relying on a host ELF loader."""
    phoff = struct.unpack_from("<Q", data, 32)[0]
    phsize, phnum = struct.unpack_from("<HH", data, 54)
    loads = []
    dynamic = None
    for index in range(phnum):
        offset = phoff + index * phsize
        kind = struct.unpack_from("<I", data, offset)[0]
        file_offset, address = struct.unpack_from("<QQ", data, offset + 8)
        file_size = struct.unpack_from("<Q", data, offset + 32)[0]
        if kind == 1:
            loads.append((address, file_offset, file_size))
        elif kind == 2:
            dynamic = (file_offset, file_size)
    if dynamic is None:
        raise ValueError("Shared library has no ELF dynamic table")
    strings = None
    needed, soname = [], None
    for offset in range(dynamic[0], sum(dynamic), 16):
        tag, value = struct.unpack_from("<qQ", data, offset)
        if tag == 0:
            break
        if tag == 1:
            needed.append(value)
        elif tag == 5:
            strings = value
        elif tag == 14:
            soname = value
    if strings is None:
        raise ValueError("Shared library has no dynamic string table")
    for address, file_offset, file_size in loads:
        if address <= strings < address + file_size:
            base = file_offset + strings - address
            break
    else:
        raise ValueError("Dynamic strings are outside ELF load segments")
    def string_at(index: int) -> str:
        offset = base + index
        return data[offset:data.index(b"\0", offset)].decode("ascii")
    return (None if soname is None else string_at(soname), [string_at(index) for index in needed])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--stock-aar", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build-config", choices=("Release", "MinSizeRel"), default="Release")
    args = parser.parse_args()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=args.source, text=True).strip()
    if commit != ORT_COMMIT:
        raise ValueError(f"Unexpected ORT source revision: {commit}")
    stock = args.stock_aar.read_bytes()
    if sha256(stock) != STOCK_SHA256:
        raise ValueError("Stock ORT 1.23.2 AAR SHA-256 does not match the pinned Maven artifact")
    native = args.native_library.read_bytes()
    cache = (args.native_library.parent / "CMakeCache.txt").read_text(encoding="utf-8")
    if f"CMAKE_BUILD_TYPE:STRING={args.build_config}\n" not in cache:
        raise ValueError("Requested packaging configuration differs from the native build cache")
    for expected in ("onnxruntime_DISABLE_CONTRIB_OPS:BOOL=OFF", "onnxruntime_DISABLE_ML_OPS:BOOL=ON",
                     "onnxruntime_USE_NNAPI_BUILTIN:BOOL=ON", "onnxruntime_BUILD_SHARED_LIB:BOOL=ON"):
        if expected + "\n" not in cache:
            raise ValueError(f"Unexpected native build feature setting: expected {expected}")
    alignments = validate_elf(native)
    native_soname, native_needed = dynamic_dependencies(native)
    if native_soname != "libonnxruntime.so":
        raise ValueError(f"Unexpected ORT SONAME: {native_soname}")
    system_libraries = {"libc.so", "libm.so", "libdl.so", "liblog.so", "libandroid.so",
                        "libneuralnetworks.so", "libz.so", "libjnigraphics.so"}
    if set(native_needed) - system_libraries:
        raise ValueError(f"ORT requires unpackaged non-system native dependencies: {native_needed}")
    with ZipFile(args.stock_aar) as archive:
        jni = archive.read("jni/arm64-v8a/libonnxruntime4j_jni.so")
    jni_alignments = validate_elf(jni, require_marker=False)
    _, jni_needed = dynamic_dependencies(jni)
    if "libonnxruntime.so" not in jni_needed or set(jni_needed) - system_libraries - {"libonnxruntime.so"}:
        raise ValueError(f"Stock JNI dependencies do not match the packaged runtime: {jni_needed}")
    patch = args.patch.read_bytes()
    source_diff = subprocess.check_output(
        ["git", "diff", "--no-ext-diff", "--binary", "--no-color", "HEAD", "--"], cwd=args.source)
    if source_diff != patch:
        raise ValueError("Tracked ORT source changes differ from the reviewed GPU-only patch")
    onnx_source = args.source / "cmake/external/onnx"
    onnx_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=onnx_source, text=True).strip()
    if onnx_commit != ONNX_COMMIT:
        raise ValueError(f"ONNX submodule revision differs from the pinned source: {onnx_commit}")
    if subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=onnx_source):
        raise ValueError("Tracked ONNX submodule source is modified")
    args.output.mkdir(parents=True, exist_ok=True)
    repo = args.output / "repository" / "com/microsoft/onnxruntime" / ARTIFACT / VERSION
    repo.mkdir(parents=True, exist_ok=True)
    aar = repo / f"{ARTIFACT}-{VERSION}.aar"
    provenance = {
        "artifact": f"com.microsoft.onnxruntime:{ARTIFACT}:{VERSION}",
        "upstream_repository": "https://github.com/microsoft/onnxruntime",
        "upstream_tag": "v1.23.2", "upstream_commit": commit,
        "onnx_submodule_commit": onnx_commit,
        "runtime_marker": MARKER, "abi": "arm64-v8a", "android_api": 29,
        "native_build_config": args.build_config, "nnapi_enabled": True,
        "contrib_ops_enabled": True, "ml_ops_enabled": False,
        "stock_aar_sha256": sha256(stock), "native_library_sha256": sha256(native),
        "patch_sha256": sha256(patch), "source_diff_sha256": sha256(source_diff),
        "source_diff_matches_reviewed_patch": True, "elf_load_alignment_bytes": alignments,
        "native_soname": native_soname, "native_needed": native_needed,
        "stock_jni_needed": jni_needed, "stock_jni_elf_load_alignment_bytes": jni_alignments,
        "unpackaged_non_system_native_dependencies": [],
        "scope": "NNAPI CPU_DISABLED explicitly selects GPU devices only. Unsupported operators remain on ORT CPU. No device execution or performance claim.",
        "physical_device_tested": False, "gpu_speedup_verified": False,
    }
    with ZipFile(args.stock_aar) as zin, ZipFile(aar, "w", compression=ZIP_DEFLATED, compresslevel=9) as zout:
        for entry in zin.infolist():
            if entry.filename.startswith("jni/") and entry.filename != "jni/" and not entry.filename.startswith("jni/arm64-v8a/"):
                continue
            data = native if entry.filename == "jni/arm64-v8a/libonnxruntime.so" else zin.read(entry)
            zout.writestr(entry, data)
        for filename, data in {
            "LICENSE-ONNXRuntime.txt": (args.source / "LICENSE").read_bytes(),
            "ThirdPartyNotices.txt": (args.source / "ThirdPartyNotices.txt").read_bytes(),
            "gpu-only-device-filter.patch": patch,
            "runtime-provenance.json": (json.dumps(provenance, indent=2) + "\n").encode(),
        }.items():
            zout.writestr("assets/reuse-gpu-runtime/" + filename, data)
    with ZipFile(aar) as archive, ZipFile(args.stock_aar) as stock_archive:
        if archive.testzip() is not None:
            raise ValueError("Packaged AAR CRC validation failed")
        for name in ("classes.jar", "jni/arm64-v8a/libonnxruntime4j_jni.so"):
            if archive.read(name) != stock_archive.read(name):
                raise ValueError(f"Stock Java/JNI must remain unchanged: {name}")
        if any(name.startswith(("jni/x86", "jni/armeabi")) for name in archive.namelist()):
            raise ValueError("An unpatched ABI was accidentally packaged")
    pom = repo / f"{ARTIFACT}-{VERSION}.pom"
    pom.write_text(f'''<?xml version="1.0" encoding="UTF-8"?>
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion><groupId>com.microsoft.onnxruntime</groupId>
  <artifactId>{ARTIFACT}</artifactId><version>{VERSION}</version><packaging>aar</packaging>
  <name>Unofficial ORT NNAPI GPU-only experiment</name>
  <description>Modified ORT 1.23.2, ARM64/API29. NNAPI CPU_DISABLED selects GPU devices only. No upstream endorsement.</description>
  <licenses><license><name>MIT</name><url>https://opensource.org/license/mit/</url></license></licenses>
</project>
''', encoding="utf-8", newline="\n")
    provenance.update(aar_sha256=sha256(aar.read_bytes()), aar_bytes=aar.stat().st_size)
    (args.output / "runtime-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    for path in (aar, pom):
        path.with_suffix(path.suffix + ".sha256").write_text(sha256(path.read_bytes()) + "\n", encoding="ascii")
    print(json.dumps({"aar": str(aar.resolve()), "sha256": provenance["aar_sha256"], "bytes": provenance["aar_bytes"]}))


if __name__ == "__main__":
    main()
