"""Prepare a separate ARM64 codec/DSP probe from frozen, read-only inputs.

No Even DEX, application loader, account code or unrelated native feature is
included. Binary outputs stay under .research; this module's sources stay small.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[2]
PLATFORM_LIBRARIES = {
    "libandroid.so", "libc.so", "libdl.so", "liblog.so", "libm.so", "libz.so",
    "libjnigraphics.so", "libnativewindow.so", "libEGL.so", "libGLESv2.so",
    "libOpenSLES.so", "libvulkan.so", "libneuralnetworks.so",
}
RUNTIME_PROFILES = ("default", "no_spin", "bounded_spin", "dynamic4")


def prepare_configuration(config_bytes, runtime_profile=None):
    """Validate the fixed profile allowlist, preserving bytes when not overridden."""
    config = json.loads(config_bytes)
    if not isinstance(config, dict):
        raise ValueError("Probe configuration must be a JSON object")
    source_profile = config.get("runtime_profile", "default")
    if not isinstance(source_profile, str) or source_profile not in RUNTIME_PROFILES:
        raise ValueError("Unsupported runtime_profile in source configuration")
    if runtime_profile is None:
        return config, config_bytes
    if not isinstance(runtime_profile, str) or runtime_profile not in RUNTIME_PROFILES:
        raise ValueError("Unsupported runtime_profile override")
    config["runtime_profile"] = runtime_profile
    return config, (json.dumps(config, indent=2) + "\n").encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def elf_dynamic(data):
    if data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected little-endian ELF64")
    if struct.unpack_from("<H", data, 18)[0] != 183:
        raise ValueError("Research probe accepts ARM64 native libraries only")
    offset = struct.unpack_from("<Q", data, 40)[0]
    entry_size, count = struct.unpack_from("<HH", data, 58)
    if entry_size < 64 or offset + count * entry_size > len(data):
        raise ValueError("Invalid ELF section header table")
    sections = [struct.unpack_from("<IIQQQQIIQQ", data, offset + i * entry_size)
                for i in range(count)]
    needed, soname = [], None
    for section in sections:
        if section[1] != 6:  # SHT_DYNAMIC
            continue
        strings = sections[section[6]]
        string_table = data[strings[4]:strings[4] + strings[5]]
        for pos in range(section[4], section[4] + section[5], 16):
            tag, value = struct.unpack_from("<qQ", data, pos)
            if tag == 0:
                break
            if tag not in (1, 14):
                continue
            text = string_table[value:string_table.index(0, value)].decode("ascii")
            if tag == 1:
                needed.append(text)
            else:
                soname = text
    return {"soname": soname, "needed": needed}


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and file_sha256(path) == sha256(data):
        return
    handle, temporary = tempfile.mkstemp(prefix="reuse-input-", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checked_model(model_dir, rate):
    metadata_path = model_dir / "manifest.json"
    graph_path = model_dir / "model.onnx"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (metadata.get("format_version") != 1 or metadata.get("sample_rate") != rate or
            metadata.get("hop_size") != rate // 50 or metadata.get("fft_size") != rate // 25 or
            metadata.get("look_ahead_frames") != 0):
        raise ValueError(f"Probe model format/rate/hop/look-ahead mismatch: {model_dir}")
    expected = metadata.get("model_sha256")
    if not isinstance(expected, str) or len(expected) != 64 or file_sha256(graph_path) != expected:
        raise ValueError(f"Probe model checksum mismatch: {model_dir}")
    return metadata


def read_custom_op(path):
    if path.suffix.lower() in (".aar", ".zip"):
        entry = "jni/arm64-v8a/libreuse_ort_ops.so"
        with zipfile.ZipFile(path) as archive:
            return archive.read(entry), str(path.resolve()) + "!" + entry
    return path.read_bytes(), str(path.resolve())


def prepare(args):
    output = args.output.resolve()
    if not output.is_relative_to((ROOT / ".research").resolve()):
        raise ValueError("Generated probe inputs must remain inside this workspace's .research directory")
    source_paths = [args.base, args.split, args.original_split, args.ort_library]
    model_packs = []
    for rate in (8000, 16000):
        selected = args.model8 if rate == 8000 else args.model16
        model_dir = selected or (args.models_dir / f"reuse-{rate//1000}k-exit3-native")
        metadata_path = model_dir / "manifest.json"
        graph_path = model_dir / "model.onnx"
        metadata = checked_model(model_dir, rate)
        model_packs.append((rate, model_dir, metadata))
        source_paths.extend((metadata_path, graph_path))
    if args.shim_override is not None:
        source_paths.append(args.shim_override)
    if args.custom_op_override is not None:
        source_paths.append(args.custom_op_override)
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    native = output / "jniLibs/arm64-v8a"
    assets = output / "assets"
    receipt = {
        "schema_version": 1,
        "purpose": "isolated ARM64 codec -> RE-USE research probe; no Even application loader/DEX",
        "application_id": "io.github.reuse.evenprobe",
        "sources": [{"path": str(path.resolve()), "sha256": file_sha256(path)}
                    for path in source_paths],
        "shim_override": args.shim_override is not None,
        "custom_op_override": args.custom_op_override is not None,
        "model_overrides": {"8000": args.model8 is not None, "16000": args.model16 is not None},
        "selected_models": {str(rate): {"directory": str(model_dir.resolve()),
                                       "model_sha256": metadata["model_sha256"],
                                       "manifest_sha256": file_sha256(model_dir / "manifest.json")}
                            for rate, model_dir, metadata in model_packs},
        "native_libraries": [], "assets": [], "runtime_validation": "not performed by preparation",
    }
    with zipfile.ZipFile(args.base) as base, zipfile.ZipFile(args.split) as split, \
            zipfile.ZipFile(args.original_split) as original:
        libraries = {}
        for name in ("liblc3.so", "liblco.so", "libreuse_ort_ops.so"):
            entry = "lib/arm64-v8a/" + name
            libraries[name] = (split.read(entry), str(args.split.resolve()) + "!" + entry)
        if args.shim_override is not None:
            libraries["liblc3.so"] = (args.shim_override.read_bytes(), str(args.shim_override.resolve()))
        if args.custom_op_override is not None:
            libraries["libreuse_ort_ops.so"] = read_custom_op(args.custom_op_override)
        libraries["libonnxruntime.so"] = (args.ort_library.read_bytes(), str(args.ort_library.resolve()))
        details = {name: elf_dynamic(data) for name, (data, _) in libraries.items()}
        if details["liblco.so"]["soname"] != "liblco.so":
            raise ValueError("Frozen delegated original LC3 must retain the patched liblco.so SONAME")
        if details["liblc3.so"]["soname"] != "liblc3.so":
            raise ValueError("The selected interposer must have the liblc3.so SONAME")
        if details["libreuse_ort_ops.so"]["soname"] != "libreuse_ort_ops.so":
            raise ValueError("The selected custom operation library must have the libreuse_ort_ops.so SONAME")
        dependencies = {dep for detail in details.values() for dep in detail["needed"]}
        if "libc++_shared.so" in dependencies:
            entry = "lib/arm64-v8a/libc++_shared.so"
            libraries["libc++_shared.so"] = (original.read(entry), str(args.original_split.resolve()) + "!" + entry)
            details["libc++_shared.so"] = elf_dynamic(libraries["libc++_shared.so"][0])
        unresolved = {dep for detail in details.values() for dep in detail["needed"]}
        unresolved -= set(libraries) | PLATFORM_LIBRARIES
        if unresolved:
            raise ValueError("Unresolved native dependencies: " + ", ".join(sorted(unresolved)))
        if native.exists():
            unexpected = {path.name for path in native.glob("*.so")} - set(libraries)
            if unexpected:
                raise ValueError("Unexpected stale native files in generated inputs: " + ", ".join(sorted(unexpected)))
        for name, (data, source) in libraries.items():
            write_atomic(native / name, data)
            receipt["native_libraries"].append({"name": name, "bytes": len(data), "sha256": sha256(data),
                                                 "source": source, **details[name]})

        required = {"assets/reuse/model.onnx", "assets/reuse/manifest.json", "assets/reuse/config.json"}
        if not required.issubset(base.namelist()):
            raise ValueError("Frozen payload APK is missing model/manifest/config assets")
        manifest = json.loads(base.read("assets/reuse/manifest.json"))
        config, config_bytes = prepare_configuration(base.read("assets/reuse/config.json"),
                                                     getattr(args, "runtime_profile", None))
        model = base.read("assets/reuse/model.onnx")
        if sha256(model) != manifest["model_sha256"]:
            raise ValueError("Frozen model checksum does not match its manifest")
        if config["sample_rate"] != manifest["sample_rate"] or config["sample_rate"] not in (8000, 16000):
            raise ValueError("Frozen configuration and model sample rates differ")
        if config.get("enabled") is not True:
            raise ValueError("Probe requires the explicitly enabled research configuration")
        default_override = args.model8 if config["sample_rate"] == 8000 else args.model16
        replacements = {}
        if default_override is not None:
            manifest = checked_model(default_override, config["sample_rate"])
            model = (default_override / "model.onnx").read_bytes()
            replacements = {"assets/reuse/model.onnx": (model, default_override / "model.onnx"),
                            "assets/reuse/manifest.json": ((default_override / "manifest.json").read_bytes(), default_override / "manifest.json"),
                            "assets/reuse/LICENSE-NVIDIA.txt": ((default_override / "LICENSE-NVIDIA.txt").read_bytes(), default_override / "LICENSE-NVIDIA.txt")}
        if getattr(args, "runtime_profile", None) is not None:
            replacements["assets/reuse/config.json"] = (config_bytes, None)
        receipt["default_model_overridden"] = default_override is not None
        receipt["runtime_profile_override"] = getattr(args, "runtime_profile", None) is not None
        receipt["runtime_profile"] = config.get("runtime_profile", "default")
        receipt["configuration"] = config
        receipt["model_sha256"] = manifest["model_sha256"]
        for entry in base.namelist():
            if entry.endswith("/") or not entry.startswith(("assets/reuse/", "assets/reuse-licenses/")):
                continue
            relative = Path(entry.removeprefix("assets/"))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe asset path in frozen APK")
            data, replacement_source = replacements.get(entry, (base.read(entry), None))
            write_atomic(assets / relative, data)
            asset_record = {"name": relative.as_posix(), "bytes": len(data), "sha256": sha256(data)}
            if replacement_source is not None:
                asset_record["source"] = str(replacement_source.resolve())
            receipt["assets"].append(asset_record)

    for name, source in (("Java-MIT.txt", ROOT / "LICENSE"),
                         ("ONNXRuntime-MIT.txt", ROOT / "native/include/LICENSE")):
        data = source.read_bytes()
        relative = Path("reuse-licenses") / name
        write_atomic(assets / relative, data)
        receipt["assets"].append({"name": relative.as_posix(), "bytes": len(data), "sha256": sha256(data)})
    for rate, model_dir, metadata in model_packs:
        for name in ("model.onnx", "manifest.json", "LICENSE-NVIDIA.txt"):
            source = model_dir / name
            data = source.read_bytes()
            relative = Path("reuse-models") / str(rate) / name
            write_atomic(assets / relative, data)
            receipt["assets"].append({"name": relative.as_posix(), "bytes": len(data), "sha256": sha256(data),
                                      "source": str(source.resolve())})
    notice = ("Isolated ARM64 research probe. No Even application DEX, account flow, or protected loader is included.\n"
              "Original codec and ONNX Runtime are frozen copies from the locally obtained app artifacts.\n"
              "Native RE-USE processing/model: NVIDIA NSCLv1; see reuse/LICENSE-NVIDIA.txt.\n"
              "Java payload: MIT. ONNX Runtime API/runtime: Microsoft MIT; see reuse-licenses.\n"
              "This app does not establish that the modified Even app launches or that physical-phone/G2 performance passes.\n")
    write_atomic(assets / "reuse-licenses/RESEARCH-NOTICE.txt", notice.encode("utf-8"))
    write_atomic(output / "inputs.json", (json.dumps(receipt, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"output": str(output), "model_sample_rate": config["sample_rate"],
                      "native_libraries": [item["name"] for item in receipt["native_libraries"]],
                      "receipt": str(output / "inputs.json")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--original-split", type=Path, required=True)
    parser.add_argument("--ort-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shim-override", type=Path,
                        help="Explicit rebuilt ARM64 shim; original delegated codec and ORT stay frozen")
    parser.add_argument("--models-dir", type=Path, default=ROOT / "models",
                        help="Both hash-verified 8/16 kHz native exit-3 model packs")
    parser.add_argument("--custom-op-override", type=Path,
                        help="ARM64 libreuse_ort_ops.so, or an AAR containing its jni/arm64-v8a entry")
    parser.add_argument("--model8", type=Path, help="Explicit 8 kHz model directory; also replaces matching default provider assets")
    parser.add_argument("--model16", type=Path, help="Explicit 16 kHz model directory; also replaces matching default provider assets")
    parser.add_argument("--runtime-profile", choices=RUNTIME_PROFILES,
                        help="Explicit native session setting profile; omitted preserves the frozen config bytes")
    prepare(parser.parse_args())
