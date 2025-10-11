from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import settings as settings


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

