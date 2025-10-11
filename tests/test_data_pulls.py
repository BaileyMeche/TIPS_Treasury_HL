from __future__ import annotations

import io
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pull_fed_tips_yield_curve as tips_module  # noqa: E402
import pull_fed_yield_curve as yield_module  # noqa: E402


class _DummyResponse:
    def __init__(self, content: bytes, status_code: int = 200):
        self.content = content
        self.status_code = status_code


def _patch_requests(monkeypatch: pytest.MonkeyPatch, module, content_factory: Callable[[], bytes]):
    def _fake_get(url: str):  # pragma: no cover - trivial forwarding
        return _DummyResponse(content_factory())

    monkeypatch.setattr(module.requests, "get", _fake_get)


def _build_yield_curve_csv(start: str, periods: int) -> bytes:
    date_index = pd.date_range(start=start, periods=periods, freq="90D")
    data = {
        f"SVENY{str(i).zfill(2)}": range(periods)
        for i in range(1, 31)
    }
    df = pd.DataFrame(data, index=date_index)

    buffer = io.StringIO()
    df.to_csv(buffer, index_label="Date", date_format="%m/%d/%Y")
    metadata = "\n".join(["metadata"] * 9)
    return (metadata + "\n" + buffer.getvalue()).encode("utf-8")


def _build_tips_curve_csv(start: str, periods: int) -> bytes:
    dates = pd.date_range(start=start, periods=periods, freq="90D")
    df = pd.DataFrame(
        {
            "Date": dates,
            "TIPSY02": range(periods),
            "TIPSY05": range(periods),
            "TIPSY10": range(periods),
            "TIPSY20": range(periods),
            "TIPSY30": range(periods),
        }
    )

    buffer = io.StringIO()
    df.to_csv(buffer, index=False, date_format="%Y-%m-%d")
    metadata = "\n".join(["metadata"] * 18)
    return (metadata + "\n" + buffer.getvalue()).encode("utf-8")


def test_pull_fed_yield_curve_retains_three_years(monkeypatch: pytest.MonkeyPatch):
    cutoff_date = pd.Timestamp("2021-01-01")
    today = datetime(2024, 1, 1)

    def _content() -> bytes:
        return _build_yield_curve_csv("2019-01-01", 24)

    _patch_requests(monkeypatch, yield_module, _content)

    df_all, df_subset = yield_module.pull_fed_yield_curve(years=3, today=today)

    assert df_all.index.min() >= cutoff_date
    assert df_subset.index.min() >= cutoff_date
    assert set(df_subset.columns) == {f"SVENY{str(i).zfill(2)}" for i in range(1, 31)}


def test_pull_fed_tips_yield_curve_retains_three_years(monkeypatch: pytest.MonkeyPatch):
    cutoff_date = pd.Timestamp("2021-01-01")
    today = datetime(2024, 1, 1)

    def _content() -> bytes:
        return _build_tips_curve_csv("2019-01-01", 24)

    _patch_requests(monkeypatch, tips_module, _content)

    df = tips_module.pull_fed_tips_yield_curve(years=3, today=today)

    assert df["date"].min() >= cutoff_date
    assert list(df.columns[:5]) == [
        "date",
        "TIPSY02",
        "TIPSY05",
        "TIPSY10",
        "TIPSY20",
    ]
    assert df["date"].is_monotonic_increasing
