from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import settings as settings


def test_directory_constants_are_exported():
    for name in ("DATA_DIR", "OUTPUT_DIR", "MANUAL_DATA_DIR", "PUBLISH_DIR"):
        value = getattr(settings, name)
        assert isinstance(value, Path)
        assert settings.d[name] == value


def test_ensure_dir_creates_primary(tmp_path):
    target = tmp_path / "primary"

    resolved = settings._ensure_dir(target, fallback=tmp_path / "fallback", label="TEST")

    assert resolved == target.resolve()
    assert target.exists()


def test_ensure_dir_falls_back_when_primary_unavailable(tmp_path, monkeypatch):
    target = tmp_path / "primary"
    fallback = tmp_path / "fallback"

    original_mkdir = Path.mkdir

    def fake_mkdir(self, *args, **kwargs):
        if self == target.resolve():
            raise PermissionError("denied")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", fake_mkdir)

    with pytest.warns(RuntimeWarning):
        resolved = settings._ensure_dir(target, fallback=fallback, label="TEST")

    assert resolved == fallback.resolve()
    assert fallback.exists()


def test_create_dirs_updates_module_globals(tmp_path):
    # Use a disposable base directory so create_dirs does not touch the repo paths.
    original_values = {
        key: settings.d[key]
        for key in ("BASE_DIR", "DATA_DIR", "OUTPUT_DIR", "MANUAL_DATA_DIR", "PUBLISH_DIR")
    }
    original_ensure = settings._ensure_dir

    settings.d["BASE_DIR"] = tmp_path
    settings.d["DATA_DIR"] = tmp_path / "primary_data"
    settings.d["OUTPUT_DIR"] = tmp_path / "primary_output"
    settings.d["MANUAL_DATA_DIR"] = tmp_path / "primary_manual"
    settings.d["PUBLISH_DIR"] = tmp_path / "primary_publish"

    results = {}

    def fake_ensure(path, fallback, label):
        # Force DATA_DIR to fall back while letting other directories resolve normally.
        chosen = fallback if label == "DATA_DIR" else path
        results[label] = Path(chosen)
        return Path(chosen)

    try:
        settings._ensure_dir = fake_ensure  # type: ignore[assignment]
        settings.create_dirs()

        assert settings.DATA_DIR == results["DATA_DIR"]
        assert settings.OUTPUT_DIR == results["OUTPUT_DIR"]
        assert settings.MANUAL_DATA_DIR == results["MANUAL_DATA_DIR"]
        assert settings.PUBLISH_DIR == results["PUBLISH_DIR"]
        assert settings.d["DATA_DIR"] == results["DATA_DIR"]
        assert settings.d["OUTPUT_DIR"] == results["OUTPUT_DIR"]
        assert settings.d["MANUAL_DATA_DIR"] == results["MANUAL_DATA_DIR"]
        assert settings.d["PUBLISH_DIR"] == results["PUBLISH_DIR"]
    finally:
        settings._ensure_dir = original_ensure  # type: ignore[assignment]
        for key, value in original_values.items():
            settings.d[key] = value
        settings.create_dirs()

