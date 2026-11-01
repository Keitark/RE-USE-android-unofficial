"""Build an isolated, research-signed Even G2 RE-USE split-APK set.

Only accepts the hash-locked Google Play baseline. Original DEX, Dart,
resources and ONNX Runtime bytes are preserved. The only original native
change is liblc3.so's DT_SONAME string in its renamed liblco.so copy.
No app/device installation, uninstallation, authentication changes, Dart
patches, or signature/integrity-check bypasses are performed.

The .apks output is a ZIP of install-multiple APKs, not a bundletool toc.pb
archive. Extract it, inspect receipt.json, and use the included installer
only on an explicitly selected test device/profile.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
import zipfile


ROOT = Path(__file__).resolve().parents[1]
ANDROID_NS = "http://schemas.android.com/apk/res/android"
ET.register_namespace("android", ANDROID_NS)
NATIVE_DIR = "lib/arm64-v8a/"
PROVIDER = "io.github.reuse.even.ReuseInitProvider"
ACTIVITY = "io.github.reuse.even.ReuseProbeActivity"
AUTHORITY = "com.even.sg.reuse.init"
SIGNING_ENTRY = re.compile(r"^(?:META-INF/(?:MANIFEST\.MF|[^/]+\.(?:SF|RSA|DSA|EC))|stamp-cert-sha256)$", re.I)
DEX_ENTRY = re.compile(r"^classes(?:(\d+))?\.dex$")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            value.update(part)
    return value.hexdigest()


def within_artifacts(path: Path) -> Path:
    path = path.resolve()
    if not any(path.is_relative_to((ROOT / directory).resolve()) for directory in (".research", "dist")):
        raise ValueError(f"Output/work paths must stay inside {ROOT / '.research'} or {ROOT / 'dist'}: {path}")
    return path


def run(command: list[str | Path], *, env: dict | None = None) -> str:
    result = subprocess.run([str(part) for part in command], cwd=ROOT, env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", timeout=600)
    if result.returncode:
        raise RuntimeError(f"Command failed ({result.returncode}): {command[0]}\n{result.stdout[-12000:]}")
    return result.stdout


def apk_contents(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate ZIP entries in {path}")
        return {item.filename: digest(archive.read(item)) for item in archive.infolist()
                if not item.is_dir() and not SIGNING_ENTRY.match(item.filename)}


def verify_contents(before: dict[str, str], after: dict[str, str], *, changed: set[str], added: set[str], label: str) -> None:
    missing = set(before) - set(after)
    unexplained_new = set(after) - set(before) - added
    modified = {name for name in set(before) & set(after) if before[name] != after[name]}
    if missing or unexplained_new or modified - changed:
        raise ValueError(f"Unexpected payload change in {label}: missing={sorted(missing)}, "
                         f"added={sorted(unexplained_new)}, changed={sorted(modified - changed)}")
    if not added <= set(after):
        raise ValueError(f"Missing injected entries in {label}: {sorted(added - set(after))}")


def dex_classes(data: bytes) -> set[str]:
    """Read standard DEX class descriptors without executing or disassembling it."""
    if len(data) < 112 or not data.startswith(b"dex\n") or data[7] != 0:
        raise ValueError("Expected a standard DEX file for class-collision verification")
    endian = struct.unpack_from("<I", data, 40)[0]
    if endian != 0x12345678:
        raise ValueError("Unsupported DEX endianness")
    string_count, string_offset, type_count, type_offset = struct.unpack_from("<IIII", data, 56)
    class_count, class_offset = struct.unpack_from("<II", data, 96)
    for offset, count, width in ((string_offset, string_count, 4), (type_offset, type_count, 4), (class_offset, class_count, 32)):
        if offset < 0 or offset + count * width > len(data):
            raise ValueError("DEX table outside input bounds")
    strings: dict[int, str] = {}

    def read_string(index: int) -> str:
        if index >= string_count:
            raise ValueError("DEX string index outside input bounds")
        if index not in strings:
            offset = struct.unpack_from("<I", data, string_offset + 4 * index)[0]
            for _ in range(5):
                if offset >= len(data):
                    raise ValueError("Truncated DEX string")
                value = data[offset]
                offset += 1
                if value < 128:
                    break
            else:
                raise ValueError("Invalid DEX string length")
            end = data.find(b"\0", offset)
            if end < 0:
                raise ValueError("Unterminated DEX descriptor")
            strings[index] = data[offset:end].decode("utf-8", errors="strict")
        return strings[index]

    descriptors = set()
    for item in range(class_count):
        type_index = struct.unpack_from("<I", data, class_offset + item * 32)[0]
        if type_index >= type_count:
            raise ValueError("DEX type index outside input bounds")
        string_index = struct.unpack_from("<I", data, type_offset + type_index * 4)[0]
        descriptors.add(read_string(string_index))
    return descriptors


def elf_dynamic(data: bytes) -> dict:
    """Read ELF64 little-endian dynamic strings, without loading the library."""
    if len(data) < 64 or data[:6] != b"\x7fELF\x02\x01":
        raise ValueError("Expected a little-endian ELF64 library")
    if struct.unpack_from("<H", data, 18)[0] != 183:
        raise ValueError("Expected an Android AArch64 native library")
    phoff = struct.unpack_from("<Q", data, 32)[0]
    phsize, phcount = struct.unpack_from("<HH", data, 54)
    if phsize < 56 or phoff + phsize * phcount > len(data):
        raise ValueError("Invalid ELF program headers")
    headers = [struct.unpack_from("<IIQQQQQQ", data, phoff + index * phsize) for index in range(phcount)]
    dynamic = [header for header in headers if header[0] == 2]
    if len(dynamic) != 1:
        raise ValueError("Expected one ELF dynamic segment")
    segment = dynamic[0]
    tags: dict[int, list[int]] = {}
    for offset in range(segment[2], segment[2] + segment[5], 16):
        tag, value = struct.unpack_from("<QQ", data, offset)
        if tag == 0:
            break
        tags.setdefault(tag, []).append(value)
    if len(tags.get(5, [])) != 1 or len(tags.get(14, [])) != 1:
        raise ValueError("ELF has no unique DT_STRTAB / DT_SONAME")
    string_va = tags[5][0]
    loads = [header for header in headers if header[0] == 1 and header[3] <= string_va < header[3] + header[5]]
    if len(loads) != 1:
        raise ValueError("Cannot map ELF string table to file bytes")
    table = loads[0][2] + string_va - loads[0][3]

    def string_at(index: int) -> tuple[str, int]:
        start = table + index
        end = data.find(b"\0", start)
        if end < start:
            raise ValueError("Unterminated ELF dynamic string")
        return data[start:end].decode("ascii"), start

    soname, position = string_at(tags[14][0])
    return {"soname": soname, "soname_offset": position,
            "needed": [string_at(index)[0] for index in tags.get(1, [])]}


def rename_lc3_soname(data: bytes) -> tuple[bytes, dict]:
    metadata = elf_dynamic(data)
    if metadata["soname"] != "liblc3.so":
        raise ValueError(f"Unexpected original LC3 SONAME: {metadata['soname']}")
    position = metadata["soname_offset"]
    old, new = b"liblc3.so", b"liblco.so"
    result = data[:position] + new + data[position + len(old):]
    differences = [index for index, pair in enumerate(zip(data, result)) if pair[0] != pair[1]]
    if len(result) != len(data) or any(not position <= index < position + len(old) for index in differences):
        raise AssertionError("LC3 copy changed outside its SONAME string")
    if elf_dynamic(result)["soname"] != "liblco.so":
        raise AssertionError("LC3 SONAME patch verification failed")
    return result, {"old": "liblc3.so", "new": "liblco.so", "offset": position,
                    "changed_byte_offsets": differences, "sha256_before": digest(data),
                    "sha256_after": digest(result)}


def write_zip(source: Path, destination: Path, replacements: dict[str, bytes], additions: dict[str, bytes], *, stored: set[str]) -> None:
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(destination, "w", allowZip64=True) as new:
        existing = set(old.namelist())
        if set(additions) & existing:
            raise ValueError(f"Injected entries already exist: {sorted(set(additions) & existing)}")
        if not set(replacements) <= existing:
            raise ValueError("A replacement entry is absent from the source archive")
        for item in old.infolist():
            if SIGNING_ENTRY.match(item.filename):
                continue
            copied = copy.copy(item)
            if item.filename in stored:
                copied.compress_type = zipfile.ZIP_STORED
            new.writestr(copied, replacements.get(item.filename, old.read(item)))
        for name, data in additions.items():
            item = zipfile.ZipInfo(name)
            item.compress_type = zipfile.ZIP_STORED if name in stored else zipfile.ZIP_DEFLATED
            item.create_system = 3
            item.external_attr = (0o100755 if name.endswith(".so") else 0o100644) << 16
            new.writestr(item, data)


def edit_manifest(path: Path) -> dict:
    tree = ET.parse(path)
    manifest = tree.getroot()
    if manifest.get("package") != "com.even.sg":
        raise ValueError("Refusing a manifest outside com.even.sg")
    application = manifest.find("application")
    if application is None:
        raise ValueError("No application element in decoded manifest")
    a = lambda key: f"{{{ANDROID_NS}}}{key}"
    if any(element.get(a("name")) in (PROVIDER, ACTIVITY) or element.get(a("authorities")) == AUTHORITY
           for element in application):
        raise ValueError("RE-USE provider/activity already exists in original manifest")
    previous = {name: application.get(a(name)) for name in ("debuggable", "extractNativeLibs")}
    application.set(a("debuggable"), "true")
    application.set(a("extractNativeLibs"), "true")
    ET.SubElement(application, "provider", {a("name"): PROVIDER, a("authorities"): AUTHORITY,
                  a("exported"): "false", a("initOrder"): "100"})
    ET.SubElement(application, "activity", {a("name"): ACTIVITY, a("exported"): "true"})
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return {"application_attributes_before": previous,
            "application_attributes_after": {"debuggable": "true", "extractNativeLibs": "true"},
            "provider": {"class": PROVIDER, "authorities": AUTHORITY, "exported": False, "initOrder": 100},
            "probe_activity": {"class": ACTIVITY, "exported": True,
                               "purpose": "Explicit experimental debug/emulator probe"}}


def restore_manifest_resource_ids(path: Path, original: bytes, resource_names: dict[int, str]) -> int:
    """Keep exact resource IDs when apktool --no-res emits decimal references.

    This reads typed reference values from the original binary XML, aligns
    its element order with ElementTree's decoded document, and restores
    qualified symbols from the original included resource tables. It
    changes no resource table or resource file.
    """
    if original[:2] != b"\x03\x00":
        raise ValueError("Original manifest is not Android binary XML")
    root = ET.parse(path)
    elements = list(root.getroot().iter())
    element_index = 0
    strings = None
    count = 0

    def length8(data: bytes, offset: int) -> tuple[int, int]:
        value = data[offset]
        if value & 0x80:
            return ((value & 0x7F) << 8) | data[offset + 1], offset + 2
        return value, offset + 1

    def length16(data: bytes, offset: int) -> tuple[int, int]:
        value = struct.unpack_from("<H", data, offset)[0]
        if value & 0x8000:
            return ((value & 0x7FFF) << 16) | struct.unpack_from("<H", data, offset + 2)[0], offset + 4
        return value, offset + 2

    offset = struct.unpack_from("<H", original, 2)[0]
    while offset < len(original):
        kind, header, size = struct.unpack_from("<HHI", original, offset)
        if header < 8 or size < header or offset + size > len(original):
            raise ValueError("Invalid binary-XML chunk")
        if kind == 1:
            string_count, _, flags, string_start, _ = struct.unpack_from("<IIIII", original, offset + 8)
            strings = []
            for index in range(string_count):
                position = offset + string_start + struct.unpack_from("<I", original, offset + header + 4 * index)[0]
                if flags & 0x100:
                    _, position = length8(original, position)
                    byte_length, position = length8(original, position)
                    value = original[position:position + byte_length].decode("utf-8")
                else:
                    units, position = length16(original, position)
                    value = original[position:position + units * 2].decode("utf-16-le")
                strings.append(value)
        elif kind == 0x102:
            if strings is None:
                raise ValueError("Binary XML string pool missing")
            namespace, tag, attribute_start, attribute_size, attribute_count = struct.unpack_from("<IIHHH", original, offset + 16)
            expected_tag = strings[tag]
            element = elements[element_index] if element_index < len(elements) else None
            restored_sdk = False
            if expected_tag == "uses-sdk" and (element is None or element.tag.rsplit("}", 1)[-1] != "uses-sdk"):
                # apktool moves this element into apktool.yml. Restore its
                # exact original values rather than accepting linker defaults.
                element = ET.Element("uses-sdk")
                root.getroot().insert(0, element)
                restored_sdk = True
            elif element is None or element.tag.rsplit("}", 1)[-1] != expected_tag:
                raise ValueError(f"Decoded/original manifest element order differs: expected {expected_tag}, got {element.tag if element is not None else None}")
            else:
                element_index += 1
            for index in range(attribute_count):
                position = offset + 16 + attribute_start + index * attribute_size
                ns, name, _, value_size, _, value_type, value = struct.unpack_from("<IIIHBBI", original, position)
                if value_size != 8:
                    raise ValueError("Unexpected binary XML typed-value width")
                key = strings[name] if ns == 0xFFFFFFFF else "{" + strings[ns] + "}" + strings[name]
                if restored_sdk:
                    if value_type in (0x10, 0x11):
                        element.set(key, str(value))
                    elif value_type == 3:
                        element.set(key, strings[value])
                    else:
                        raise ValueError("Unsupported original uses-sdk typed attribute")
                    continue
                if value_type not in (1, 2):
                    continue
                if key not in element.attrib:
                    raise ValueError(f"Decoded manifest omitted referenced attribute {key}")
                prefix = "@" if value_type == 1 else "?"
                if value == 0 and value_type == 1:
                    element.set(key, "@null")
                elif value in resource_names:
                    element.set(key, prefix + resource_names[value])
                else:
                    raise ValueError(f"Original manifest resource ID has no verified symbol: 0x{value:08x}")
                count += 1
        offset += size
    if element_index != len(elements):
        raise ValueError("Decoded manifest has unexpected extra elements")
    root.write(path, encoding="utf-8", xml_declaration=True)
    return count


def parse_certificate(output: str) -> str:
    values = re.findall(r"Signer #\d+ certificate SHA-256 digest:\s*([0-9a-fA-F:]+)", output)
    if not values or len(set(value.replace(":", "").lower() for value in values)) != 1:
        raise ValueError("Expected exactly one APK signing certificate")
    return values[0].replace(":", "").lower()


def read_custom_op(path: Path) -> bytes:
    if path.suffix.lower() == ".aar":
        with zipfile.ZipFile(path) as archive:
            return archive.read("jni/arm64-v8a/libreuse_ort_ops.so")
    return path.read_bytes()


def locate_java(java_home: Path | None) -> tuple[Path, Path]:
    if java_home:
        home = java_home
    elif os.environ.get("JAVA_HOME"):
        home = Path(os.environ["JAVA_HOME"])
    elif shutil.which("java"):
        home = Path(shutil.which("java")).parent.parent
    else:
        raise FileNotFoundError("Pass --java-home pointing to JDK 17+")
    suffix = ".exe" if os.name == "nt" else ""
    java, keytool = home / "bin" / f"java{suffix}", home / "bin" / f"keytool{suffix}"
    if not java.is_file() or not keytool.is_file():
        raise FileNotFoundError(f"JDK tools missing under {home}")
    return java.resolve(), keytool.resolve()


def package(args: argparse.Namespace) -> dict:
    source = args.source_dir.resolve()
    lock = json.loads(args.source_lock.read_text(encoding="utf-8-sig"))
    output = within_artifacts(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a fresh reviewable artifact path: {output}")
    java, keytool = locate_java(args.java_home)
    signer = args.build_tools / "lib/apksigner.jar"
    zipalign = args.build_tools / ("zipalign.exe" if os.name == "nt" else "zipalign")
    aapt2 = args.build_tools / ("aapt2.exe" if os.name == "nt" else "aapt2")
    for tool in (args.apktool, signer, zipalign, aapt2):
        if not tool.is_file():
            raise FileNotFoundError(f"Required packaging tool missing: {tool}")
    apktool_version = run([java, "-jar", args.apktool, "--version"]).strip()
    supplied = {path.name for path in source.glob("*.apk")}
    if supplied != set(lock["apks"]):
        raise ValueError(f"Source split inventory differs from lock: got {sorted(supplied)}")
    print("Verifying locked Play APKs and signing certificates...", flush=True)
    source_certificates = {}
    for name, expected in lock["apks"].items():
        path = source / name
        if file_hash(path) != expected:
            raise ValueError(f"Locked APK SHA256 mismatch: {name}")
        certificate = parse_certificate(run([java, "-jar", signer, "verify", "--print-certs", str(path)]))
        if certificate != lock["signer_sha256"]:
            raise ValueError(f"Official Play signing certificate differs: {name}")
        source_certificates[name] = certificate
    base = source / lock["base_apk"]
    arm_split = source / lock["arm64_split"]
    original_badging = run([aapt2, "dump", "badging", base])
    match = re.search(r"^package: name='([^']+)' versionCode='(\d+)' versionName='([^']+)'", original_badging, re.M)
    if not match or (match[1], int(match[2]), match[3]) != (lock["package"], lock["version_code"], lock["version_name"]):
        raise ValueError("Actual manifest package/version differs from lock")
    original_target_sdk = re.search(r"^targetSdkVersion:'([^']+)'", original_badging, re.M)
    original_min_sdk = re.search(r"^(?:sdkVersion|minSdkVersion):'([^']+)'", original_badging, re.M)

    model_manifest = json.loads((args.model_dir / "manifest.json").read_text(encoding="utf-8-sig"))
    model = (args.model_dir / "model.onnx").read_bytes()
    if digest(model) != model_manifest["model_sha256"]:
        raise ValueError("Model bytes differ from model manifest")
    rate = model_manifest["sample_rate"]
    if rate not in (8000, 16000):
        raise ValueError("Only the verified 8/16 kHz model frontends are supported")
    configuration = {"sample_rate": rate, "threads": args.threads, "enabled": True}
    assets = {"assets/reuse/model.onnx": model,
              "assets/reuse/manifest.json": (args.model_dir / "manifest.json").read_bytes(),
              "assets/reuse/config.json": (json.dumps(configuration, indent=2) + "\n").encode(),
              "assets/reuse/LICENSE-NVIDIA.txt": (args.model_dir / "LICENSE-NVIDIA.txt").read_bytes()}
    with zipfile.ZipFile(base) as archive:
        dex_names = [name for name in archive.namelist() if DEX_ENTRY.match(name)]
        dex_names.sort(key=lambda name: int(DEX_ENTRY.match(name)[1] or 1))
        expected_names = ["classes.dex"] + [f"classes{index}.dex" for index in range(2, lock["dex_count"] + 1)]
        if dex_names != expected_names:
            raise ValueError("Original DEX inventory differs from locked count/order")
        original_dex = {name: archive.read(name) for name in dex_names}
        original_classes: set[str] = set()
        for data in original_dex.values():
            original_classes.update(dex_classes(data))
    with zipfile.ZipFile(args.payload_apk) as payload:
        payload_dex_names = [name for name in payload.namelist() if DEX_ENTRY.match(name)]
        if payload_dex_names != ["classes.dex"]:
            raise ValueError("The injection payload must contain exactly one classes.dex")
        payload_dex = payload.read("classes.dex")
    payload_classes = dex_classes(payload_dex)
    collisions = original_classes & payload_classes
    if collisions:
        raise ValueError(f"Payload/original class collisions: {sorted(collisions)}")
    if not {"L" + PROVIDER.replace(".", "/") + ";", "L" + ACTIVITY.replace(".", "/") + ";"} <= payload_classes:
        raise ValueError("Payload DEX lacks the expected provider/probe classes")
    injected_dex_name = f"classes{lock['dex_count'] + 1}.dex"

    before_arm = apk_contents(arm_split)
    with zipfile.ZipFile(arm_split) as archive:
        for name, expected in lock["native"].items():
            if digest(archive.read(name)) != expected:
                raise ValueError(f"Locked native SHA256 mismatch: {name}")
        original_lc3 = archive.read(NATIVE_DIR + "liblc3.so")
    renamed_lc3, soname_patch = rename_lc3_soname(original_lc3)
    shim, custom_op = args.shim.read_bytes(), read_custom_op(args.custom_op)
    if elf_dynamic(shim)["soname"] != "liblc3.so":
        raise ValueError("Shim SONAME must be liblc3.so")
    if elf_dynamic(custom_op)["soname"] != "libreuse_ort_ops.so":
        raise ValueError("Custom-op library SONAME must be libreuse_ort_ops.so")
    native_additions = {NATIVE_DIR + "liblco.so": renamed_lc3,
                        NATIVE_DIR + "libreuse_ort_ops.so": custom_op}
    for argument in args.native_lib:
        name, sep, path = argument.partition("=")
        native_path = Path(path if sep else name)
        name = name if sep else native_path.name
        if not re.fullmatch(r"lib[A-Za-z0-9_+.-]+\.so", name) or name == "libonnxruntime.so":
            raise ValueError(f"Invalid or protected native destination: {name}")
        entry = NATIVE_DIR + name
        if entry in before_arm or entry in native_additions:
            raise ValueError(f"Refusing native entry collision: {entry}")
        data = native_path.read_bytes()
        if elf_dynamic(data)["soname"] != name:
            raise ValueError(f"Additional library SONAME must match its destination: {name}")
        native_additions[entry] = data

    work = within_artifacts(ROOT / ".research" / f"even-patch-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}")
    work.mkdir(parents=True)
    built = work / "unsigned"
    built.mkdir()
    final = work / "signed"
    final.mkdir()
    framework = work / "framework"
    decoded = work / "base-decoded"
    print(f"Decoding manifest only with apktool {apktool_version}; original DEX/resources remain compiled...", flush=True)
    decode_log = run([java, "-jar", args.apktool, "decode", "--no-src", "--no-res", "--force-manifest",
                      "--frame-path", framework, "--output", decoded, base])
    (work / "apktool-decode.log").write_text(decode_log, encoding="utf-8")
    with zipfile.ZipFile(base) as archive:
        original_manifest_bytes = archive.read("AndroidManifest.xml")
    resource_names = {}
    for included, package_name in ((base, lock["package"]), (framework / "1.apk", "android")):
        resource_dump = run([aapt2, "dump", "resources", included])
        for resource in re.finditer(r"^\s*resource (0x[0-9a-fA-F]+) (\S+)", resource_dump, re.M):
            resource_names[int(resource[1], 16)] = package_name + ":" + resource[2]
    resource_reference_count = restore_manifest_resource_ids(decoded / "AndroidManifest.xml", original_manifest_bytes, resource_names)
    manifest_changes = edit_manifest(decoded / "AndroidManifest.xml")
    manifest_changes["original_resource_ids_preserved"] = resource_reference_count
    # An original Play source-stamp digest is signing metadata, not app code.
    # It cannot describe a newly research-signed APK. Keep it out of this build.
    for path in (decoded / "stamp-cert-sha256", decoded / "unknown/stamp-cert-sha256"):
        path.unlink(missing_ok=True)
    # Copy exact original DEX bytes explicitly; no disassembly/assembly pass.
    for name, data in original_dex.items():
        (decoded / name).write_bytes(data)
    (decoded / injected_dex_name).write_bytes(payload_dex)
    for name, data in assets.items():
        target = decoded / name
        if target.exists():
            raise ValueError(f"Original app already has a RE-USE asset: {name}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    unsigned_base = built / lock["base_apk"]
    print("Compiling only the manifest; copying all original resources and app payloads directly...", flush=True)
    compiled_manifest_apk = work / "compiled-manifest.apk"
    original_package_attributes = dict(re.findall(r"(\w+)='([^']*)'", original_badging.splitlines()[0]))
    compile_sdk_args = []
    if "compileSdkVersion" in original_package_attributes:
        compile_sdk_args += ["--compile-sdk-version-code", original_package_attributes["compileSdkVersion"]]
    if "compileSdkVersionCodename" in original_package_attributes:
        compile_sdk_args += ["--compile-sdk-version-name", original_package_attributes["compileSdkVersionCodename"]]
    build_log = run([aapt2, "link", "--manifest", decoded / "AndroidManifest.xml", "-I", framework / "1.apk", "-I", base,
                     "--allow-reserved-package-id", "--package-id", "0x7f", "--no-auto-version",
                     "--version-code", str(lock["version_code"]), "--version-name", lock["version_name"],
                     *compile_sdk_args, "-o", compiled_manifest_apk])
    (work / "manifest-compile.log").write_text(build_log, encoding="utf-8")
    with zipfile.ZipFile(compiled_manifest_apk) as archive:
        compiled_manifest = archive.read("AndroidManifest.xml")
    if compiled_manifest[:2] != b"\x03\x00":
        raise ValueError("aapt2 did not produce a binary AndroidManifest.xml")
    write_zip(base, unsigned_base, {"AndroidManifest.xml": compiled_manifest},
              {injected_dex_name: payload_dex, **assets}, stored={injected_dex_name, "assets/reuse/model.onnx"})
    before_base = apk_contents(base)
    verify_contents(before_base, apk_contents(unsigned_base), changed={"AndroidManifest.xml"},
                    added={injected_dex_name, *assets}, label="rebuilt base")
    unsigned_arm = built / lock["arm64_split"]
    write_zip(arm_split, unsigned_arm, {NATIVE_DIR + "liblc3.so": shim}, native_additions,
              stored={NATIVE_DIR + "liblc3.so", *native_additions})
    verify_contents(before_arm, apk_contents(unsigned_arm), changed={NATIVE_DIR + "liblc3.so"},
                    added=set(native_additions), label="ARM64 split")
    for name in lock["apks"]:
        if name not in (lock["base_apk"], lock["arm64_split"]):
            write_zip(source / name, built / name, {}, {}, stored=set())

    keys = within_artifacts(ROOT / ".research/even-research-signing")
    keys.mkdir(parents=True, exist_ok=True)
    keystore = keys / "reuse-research.p12"
    password_file = keys / "password.txt"
    if keystore.exists() != password_file.exists():
        raise ValueError("Incomplete local research signing identity; preserve and inspect existing key files")
    if not keystore.exists():
        password_file.write_text(uuid.uuid4().hex + uuid.uuid4().hex, encoding="ascii")
        env = dict(os.environ)
        env["REUSE_RESEARCH_APK_KEY_PASSWORD"] = password_file.read_text(encoding="ascii")
        run([keytool, "-genkeypair", "-keystore", keystore, "-storetype", "PKCS12", "-alias", "reuse-research",
             "-storepass:env", "REUSE_RESEARCH_APK_KEY_PASSWORD", "-keypass:env", "REUSE_RESEARCH_APK_KEY_PASSWORD",
             "-keyalg", "RSA", "-keysize", "2048", "-validity", "3650",
             "-dname", "CN=REUSE Local Research - Not Even Realities,OU=Experimental Local Build,O=REUSE Research,C=JP"], env=env)
    print("Aligning and signing every split with one clearly labelled local research certificate...", flush=True)
    output_apks = {}
    research_certificate = None
    for name in lock["apks"]:
        aligned = work / f"aligned-{name}"
        run([zipalign, "-f", "-P", "16", "4", built / name, aligned])
        signed = final / name
        run([java, "-jar", signer, "sign", "--ks", keystore, "--ks-key-alias", "reuse-research",
             "--ks-pass", f"file:{password_file}", "--out", signed, aligned])
        verification = run([java, "-jar", signer, "verify", "--verbose", "--print-certs", signed])
        (work / f"verify-{name}.log").write_text(verification, encoding="utf-8")
        certificate = parse_certificate(verification)
        if research_certificate is None:
            research_certificate = certificate
        if certificate != research_certificate or certificate == lock["signer_sha256"]:
            raise ValueError("All splits must use the same distinct local research certificate")
        run([zipalign, "-c", "-P", "16", "4", signed])
        original = apk_contents(source / name)
        if name == lock["base_apk"]:
            changed, added = {"AndroidManifest.xml"}, {injected_dex_name, *assets}
        elif name == lock["arm64_split"]:
            changed, added = {NATIVE_DIR + "liblc3.so"}, set(native_additions)
        else:
            changed, added = set(), set()
        verify_contents(original, apk_contents(signed), changed=changed, added=added, label=f"signed {name}")
        output_apks[name] = {"sha256": file_hash(signed), "bytes": signed.stat().st_size,
                             "signer_sha256": certificate, "verified": True, "alignment_kb": 16}
    final_badging = run([aapt2, "dump", "badging", final / lock["base_apk"]])
    final_match = re.search(r"^package: name='([^']+)' versionCode='(\d+)' versionName='([^']+)'", final_badging, re.M)
    if not final_match or final_match.groups() != match.groups():
        raise ValueError("Package/version changed unexpectedly during rebuilding")
    final_package_attributes = dict(re.findall(r"(\w+)='([^']*)'", final_badging.splitlines()[0]))
    for name in ("compileSdkVersion", "compileSdkVersionCodename", "platformBuildVersionCode", "platformBuildVersionName"):
        if final_package_attributes.get(name) != original_package_attributes.get(name):
            raise ValueError(f"Original {name} changed during manifest compilation")
    for pattern, expected in ((r"^targetSdkVersion:'([^']+)'", original_target_sdk), (r"^(?:sdkVersion|minSdkVersion):'([^']+)'", original_min_sdk)):
        actual = re.search(pattern, final_badging, re.M)
        if (actual[1] if actual else None) != (expected[1] if expected else None):
            raise ValueError("Original target/min SDK was changed")
    xml_dump = run([aapt2, "dump", "xmltree", "--file", "AndroidManifest.xml", final / lock["base_apk"]])
    for expected in (PROVIDER, ACTIVITY, AUTHORITY):
        if expected not in xml_dump:
            raise ValueError(f"Compiled manifest omitted injected component: {expected}")
    for flag in ("debuggable", "extractNativeLibs"):
        if not re.search(rf":{flag}\([^)]*\)=true", xml_dump):
            raise ValueError(f"Compiled manifest omitted application flag: {flag}")
    with zipfile.ZipFile(final / lock["base_apk"]) as archive:
        dex_hashes = {name: digest(archive.read(name)) for name in dex_names}
        if dex_hashes != {name: digest(data) for name, data in original_dex.items()}:
            raise AssertionError("An original DEX changed")
    with zipfile.ZipFile(final / lock["arm64_split"]) as archive:
        preserved_native = {name: digest(archive.read(name)) for name in lock["native"] if not name.endswith("/liblc3.so")}
        if any(value != lock["native"][name] for name, value in preserved_native.items()):
            raise AssertionError("Original Dart/ORT bytes changed")
    receipt = {
        "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "BUILT_AND_STATICALLY_VERIFIED_NOT_INSTALLED",
        "package": lock["package"], "version_name": lock["version_name"], "version_code": lock["version_code"],
        "input_source": lock["source"], "input_apk_sha256": lock["apks"],
        "official_play_signer_sha256": lock["signer_sha256"], "source_certificates_verified": source_certificates,
        "research_signer_sha256": research_certificate,
        "experimental_certificate_name": "REUSE Local Research - Not Even Realities",
        "original_dex_sha256_preserved": dex_hashes, "injected_dex": {"name": injected_dex_name, "sha256": digest(payload_dex), "classes": len(payload_classes)},
        "class_collision_check": {"original_classes": len(original_classes), "payload_classes": len(payload_classes), "collisions": 0},
        "manifest_changes": manifest_changes, "lc3_soname_patch": soname_patch,
        "original_sdk_metadata_preserved": {"package_attributes": original_package_attributes,
                                            "min_sdk": original_min_sdk[1] if original_min_sdk else None,
                                            "target_sdk": original_target_sdk[1] if original_target_sdk else None},
        "signing_metadata": "Original APK signing blocks, v1 signature entries and old source-stamp digest are replaced/removed for the distinct local research signer; application verification code is unchanged",
        "preserved_dart_and_ort_sha256": preserved_native,
        "native_additions": {name: {"sha256": digest(data), "dynamic": elf_dynamic(data)} for name, data in native_additions.items()},
        "lc3_shim": {"sha256": digest(shim), "dynamic": elf_dynamic(shim)},
        "configuration": configuration, "model_sha256": digest(model),
        "model_license": model_manifest.get("license"), "commercial_use": model_manifest.get("commercial_use", False),
        "tools": {"apktool_version": apktool_version, "apktool_sha256": file_hash(args.apktool), "build_tools": str(args.build_tools.resolve())},
        "apks": output_apks, "work_directory": str(work),
        "boundaries": {"authentication_changed": False, "signature_or_integrity_checks_bypassed": False,
                       "dart_code_changed": False, "original_onnxruntime_changed": False,
                       "installed": False, "live_g2_hook_verified": False, "phone_realtime_verified": False},
    }
    (final / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    install_lines = ["@echo off", "setlocal", "rem Experimental local research build. This does not uninstall an existing app.",
                     "rem Select the intended test device with ANDROID_SERIAL before running.",
                     'if "%ANDROID_SERIAL%"=="" (echo Set ANDROID_SERIAL to the intended test device first. & exit /b 1)',
                     'adb install-multiple -r ' + " ".join(f'"%~dp0{name}"' for name in lock["apks"]), "exit /b %errorlevel%"]
    (final / "install-test-device.bat").write_text("\r\n".join(install_lines) + "\r\n", encoding="ascii", newline="")
    (final / "README.txt").write_text(
        "EXPERIMENTAL LOCAL RESEARCH BUILD - NOT SIGNED BY EVEN REALITIES\n\n"
        "Extract this archive first. Inspect receipt.json and the certificate identity.\n"
        "Set ANDROID_SERIAL to the explicitly selected test device/emulator, then run install-test-device.bat.\n"
        "This .apks file is an ordinary ZIP, not a bundletool toc.pb archive.\n"
        "The local certificate cannot update an installed Play-signed copy. No uninstall is performed.\n"
        "Use an isolated test device/profile; preserve any existing application and data.\n"
        "No authentication, signature-verification or integrity-check bypass is included.\n"
        "Static package checks do not prove login, glasses connectivity, audio interception or real-time speed.\n"
        "The streaming model remains subject to NVIDIA's noncommercial research license.\n", encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".apks":
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(final.iterdir()):
                if path.is_file() and not path.name.endswith(".idsig"):
                    archive.write(path, path.name)
        receipt_path = output.with_suffix(".receipt.json")
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    else:
        shutil.copytree(final, output)
        receipt_path = output / "receipt.json"
    print(json.dumps({"output": str(output), "receipt": str(receipt_path), "status": receipt["status"],
                      "research_signer_sha256": research_certificate}, indent=2))
    return receipt


def main() -> None:
    local_sdk = Path(os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData/Local"))) / "Android/Sdk/build-tools/35.0.0"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / ".research/even-play-2.4.0")
    parser.add_argument("--source-lock", type=Path, default=ROOT / "tools/even-apk-lock.json")
    parser.add_argument("--payload-apk", type=Path, default=ROOT / "android/even-payload/build/outputs/apk/debug/even-payload-debug.apk")
    parser.add_argument("--shim", type=Path, default=ROOT / "native/even/build-arm64/liblc3.so")
    parser.add_argument("--custom-op", type=Path, default=ROOT / "android/reuse/build/outputs/aar/reuse-release.aar", help="ARM64 .so or AAR containing jni/arm64-v8a/libreuse_ort_ops.so")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/reuse-8k-exit3-native")
    parser.add_argument("--output", type=Path, default=ROOT / "dist/even-reuse-fast8k.apks")
    parser.add_argument("--native-lib", action="append", default=[], metavar="NAME=PATH", help="Optional future native dependency; never overwrites original libraries")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--apktool", type=Path, default=Path("F:/dev/EvenG2/apk/_tools/apktool.jar"))
    parser.add_argument("--build-tools", type=Path, default=local_sdk)
    parser.add_argument("--java-home", type=Path)
    args = parser.parse_args()
    if not 1 <= args.threads <= 8:
        parser.error("--threads must be between 1 and 8")
    try:
        package(args)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, zipfile.BadZipFile) as error:
        print(f"Packaging stopped: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
