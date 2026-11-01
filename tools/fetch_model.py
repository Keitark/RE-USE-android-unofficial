"""Fetch a hash-locked NVIDIA research checkpoint; never run downloaded code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "tools/source-lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(destination: Path, include_reference: bool = False) -> dict:
    lock = json.loads(LOCK.read_text(encoding="utf-8-sig"))
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in lock["files"].items():
        if not include_reference and name not in ("config.json", "model.safetensors"):
            continue
        target = destination / name
        if not target.is_file() or sha256(target) != expected:
            target.parent.mkdir(parents=True, exist_ok=True)
            url = f'https://huggingface.co/{lock["repository"]}/resolve/{lock["revision"]}/{name}'
            temporary = target.with_suffix(target.suffix + ".download")
            print(f"Downloading pinned {name}", flush=True)
            try:
                urllib.request.urlretrieve(url, temporary)
                if sha256(temporary) != expected:
                    raise RuntimeError(f"SHA256 mismatch for {name}; no file accepted")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
        print(f"Verified {name}", flush=True)
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".research/upstream")
    parser.add_argument("--reference", action="store_true", help="Also fetch reference source and sample audio for parity tests")
    args = parser.parse_args()
    print("Checkpoint terms: NVIDIA NSCLv1; noncommercial research/education only.")
    fetch(args.output, args.reference)
