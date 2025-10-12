from datetime import date
from pathlib import Path
import sys
from urllib.error import HTTPError

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from planC import infl_curve_public as module  # noqa: E402


def test_clevelandfed_json_fallback(monkeypatch, tmp_path):
    def fail_download(*args, **kwargs):  # pragma: no cover - exercised via fallback
        raise RuntimeError("download failed")

    monkeypatch.setattr(module, "download_clevelandfed_term_structure", fail_download)

    sample = pd.DataFrame(
        [
            {"date": date(2024, 1, 2), "maturity": 2.0, "inflation": 0.02},
            {"date": date(2024, 1, 2), "maturity": 5.0, "inflation": 0.022},
            {"date": date(2024, 1, 3), "maturity": 2.0, "inflation": 0.021},
            {"date": date(2024, 1, 3), "maturity": 5.0, "inflation": 0.023},
        ]
    )

    class DummyClient(module.ClevelandFedClient):  # pragma: no cover - simple shim
        def fetch_curve(self, query):
            return sample

    monkeypatch.setattr(module, "ClevelandFedClient", DummyClient)

    with pytest.warns(RuntimeWarning):
        panel = module.load_clevelandfed_zero_coupon_inflation(
            tmp_path, [2, 5], refresh=True
        )

    assert {"inf_swap_2y", "inf_swap_5y"}.issubset(panel.columns)
    assert len(panel) == 2
    parquet_path = tmp_path / "clevelandfed" / "clevelandfed_inflation_term_structure.parquet"
    assert parquet_path.exists()


def test_clevelandfed_sample_fallback(monkeypatch, tmp_path):
    def fail_download(*args, **kwargs):  # pragma: no cover - exercised via fallback
        raise RuntimeError("download failed")

    def fail_api(*args, **kwargs):  # pragma: no cover - exercised via fallback
        raise HTTPError(url="https://example.com", code=404, msg="not found", hdrs=None, fp=None)

    monkeypatch.setattr(module, "download_clevelandfed_term_structure", fail_download)
    monkeypatch.setattr(module, "_load_panel_from_api", fail_api)

    with pytest.warns(RuntimeWarning):
        panel = module.load_clevelandfed_zero_coupon_inflation(
            tmp_path, [2, 5, 10, 20], refresh=True
        )

    assert {"inf_swap_2y", "inf_swap_5y", "inf_swap_10y", "inf_swap_20y"}.issubset(
        panel.columns
    )
    assert len(panel) >= 10
