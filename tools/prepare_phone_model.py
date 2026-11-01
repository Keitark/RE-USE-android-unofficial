"""Extract only the verified research model from a checksum-verified demo APK.

No Android installation or code execution is performed. No microphone data is
read. Download the APK and its SHA256SUMS from the same published release first.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA256 = "f950dcfb9321191b2bcb3ab9afa3e100c40e5c64dd1de1d30849be4c1aacce32"
DEFAULT_OUTPUT = ROOT / "models/reuse-8k-exit3-native-chunk5-sequence-pad-softplus-temporal-channelnorm-frequency"
NAMES = {"model.onnx": 10_000_000, "manifest.json": 200_000, "LICENSE-NVIDIA.txt": 100_000}


def extract(apk, expected_apk_sha256, output):
    apk, output = Path(apk), Path(output)
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_apk_sha256):
        raise ValueError("Expected APK SHA-256 must contain exactly 64 hexadecimal digits")
    actual = hashlib.sha256(apk.read_bytes()).hexdigest()
    if actual != expected_apk_sha256.lower():
        raise ValueError("APK SHA-256 mismatch; no model files were written")
    files = {}
    with zipfile.ZipFile(apk) as archive:
        for name, limit in NAMES.items():
            entry = "assets/model/" + name
            if sum(info.filename == entry for info in archive.infolist()) != 1:
                raise ValueError("Missing or duplicated model asset: " + name)
            info = archive.getinfo(entry)
            if info.file_size > limit:
                raise ValueError("Unexpected model asset size: " + name)
            files[name] = archive.read(entry)
    manifest = json.loads(files["manifest.json"])
    digest = hashlib.sha256(files["model.onnx"]).hexdigest()
    if digest != MODEL_SHA256 or manifest.get("model_sha256") != digest:
        raise ValueError("Pinned model SHA-256 mismatch; no model files were written")
    if manifest.get("sample_rate") != 8000 or manifest.get("frames_per_call") != 5:
        raise ValueError("The demo requires the pinned 8 kHz five-frame model")
    if not files["LICENSE-NVIDIA.txt"].strip():
        raise ValueError("The bundled model license is empty")
    # Verify every input before creating or changing the destination.
    for name, data in files.items():
        target = output / name
        if target.exists() and target.read_bytes() != data:
            raise ValueError("Refusing to overwrite a different existing model asset: " + str(target))
    output.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        target = output / name
        if not target.exists():
            temporary = output / (name + ".part")
            temporary.write_bytes(data)
            temporary.replace(target)
    return {"apk_sha256": actual, "model_sha256": digest, "output": str(output),
            "license": manifest.get("license"), "commercial_use": manifest.get("commercial_use")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apk", type=Path, required=True)
    parser.add_argument("--expected-apk-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    try:
        result = extract(args.apk, args.expected_apk_sha256, args.output)
    except (OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
        parser.exit(1, str(error) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
