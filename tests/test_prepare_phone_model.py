"""Synthetic APK integrity tests; no model weights or microphone audio needed."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from tools import prepare_phone_model


@pytest.fixture
def model_assets(monkeypatch):
    model = b"synthetic model for extraction contract"
    digest = hashlib.sha256(model).hexdigest()
    monkeypatch.setattr(prepare_phone_model, "MODEL_SHA256", digest)
    return {
        "model.onnx": model,
        "manifest.json": json.dumps({"model_sha256": digest, "sample_rate": 8000,
                                     "frames_per_call": 5, "license": "test license",
                                     "commercial_use": False}).encode(),
        "LICENSE-NVIDIA.txt": b"synthetic license notice\n",
    }


def apk_file(tmp_path, assets, duplicates=()):
    path = tmp_path / "demo.apk"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in assets.items():
            archive.writestr("assets/model/" + name, data)
        for name in duplicates:
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("assets/model/" + name, assets[name])
        archive.writestr("assets/private-audio.wav", b"must never be extracted")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def test_verified_model_assets_extract_and_repeat_without_changing_files(tmp_path, model_assets):
    apk, digest = apk_file(tmp_path, model_assets)
    output = tmp_path / "model"
    result = prepare_phone_model.extract(apk, digest.upper(), output)
    assert result["apk_sha256"] == digest
    assert result["model_sha256"] == hashlib.sha256(model_assets["model.onnx"]).hexdigest()
    assert result["commercial_use"] is False
    assert {path.name for path in output.iterdir()} == set(model_assets)
    for name, data in model_assets.items():
        assert (output / name).read_bytes() == data
    assert prepare_phone_model.extract(apk, digest, output) == result


@pytest.mark.parametrize("expected", ["0" * 64, "not a hash", "a" * 63])
def test_bad_apk_checksum_is_rejected_before_creating_destination(tmp_path, model_assets, expected):
    apk, _ = apk_file(tmp_path, model_assets)
    output = tmp_path / "model"
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_phone_model.extract(apk, expected, output)
    assert not output.exists()


@pytest.mark.parametrize("corruption", ["model", "manifest_sha", "rate", "frames", "license"])
def test_unexpected_model_content_is_rejected_before_writes(tmp_path, model_assets, corruption):
    if corruption == "model":
        model_assets["model.onnx"] += b"changed"
    elif corruption == "license":
        model_assets["LICENSE-NVIDIA.txt"] = b" \n"
    else:
        manifest = json.loads(model_assets["manifest.json"])
        manifest[{"manifest_sha": "model_sha256", "rate": "sample_rate", "frames": "frames_per_call"}[corruption]] = {
            "manifest_sha": "0" * 64, "rate": 16000, "frames": 1,
        }[corruption]
        model_assets["manifest.json"] = json.dumps(manifest).encode()
    apk, digest = apk_file(tmp_path, model_assets)
    output = tmp_path / "model"
    with pytest.raises(ValueError):
        prepare_phone_model.extract(apk, digest, output)
    assert not output.exists()


@pytest.mark.parametrize("name", ["model.onnx", "manifest.json", "LICENSE-NVIDIA.txt"])
def test_missing_model_asset_is_rejected(tmp_path, model_assets, name):
    del model_assets[name]
    apk, digest = apk_file(tmp_path, model_assets)
    output = tmp_path / "model"
    with pytest.raises(ValueError, match="Missing or duplicated"):
        prepare_phone_model.extract(apk, digest, output)
    assert not output.exists()


def test_duplicate_asset_is_rejected_even_with_verified_apk_checksum(tmp_path, model_assets):
    apk, digest = apk_file(tmp_path, model_assets, duplicates=["model.onnx"])
    output = tmp_path / "model"
    with pytest.raises(ValueError, match="Missing or duplicated"):
        prepare_phone_model.extract(apk, digest, output)
    assert not output.exists()


def test_oversize_compressed_asset_is_rejected_before_destination_write(tmp_path, model_assets):
    model_assets["model.onnx"] = b"\0" * (prepare_phone_model.NAMES["model.onnx"] + 1)
    apk, digest = apk_file(tmp_path, model_assets)
    output = tmp_path / "model"
    with pytest.raises(ValueError, match="Unexpected model asset size"):
        prepare_phone_model.extract(apk, digest, output)
    assert not output.exists()


def test_different_existing_asset_is_preserved_and_nothing_else_is_written(tmp_path, model_assets):
    apk, digest = apk_file(tmp_path, model_assets)
    output = tmp_path / "model"
    output.mkdir()
    target = output / "LICENSE-NVIDIA.txt"
    target.write_bytes(b"existing notice")
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        prepare_phone_model.extract(apk, digest, output)
    assert list(output.iterdir()) == [target]
    assert target.read_bytes() == b"existing notice"
