"""Tests for the planC_fullsample_enrich_safe module."""

from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("pandas")

import pandas as pd

from .planC_fullsample_enrich_safe import PlanCSafeConfig, run_planC_fullsample_enrich_safe


def test_run_planC_safe_with_sample_data(tmp_path):
    output_dir = tmp_path / "planC_safe"
    config = PlanCSafeConfig(
        years=[2020],
        quarters=[1],
        use_sample_data=True,
        output_dir=output_dir,
        throttle_seconds=0.0,
    )

    results = run_planC_fullsample_enrich_safe(config)

    basis = results["synthetic_basis"]
    assert isinstance(basis, pd.DataFrame)
    assert not basis.empty
    assert {"date", "pv_nom", "pv_tips", "basis_px_per100"}.issubset(basis.columns)

    enriched = results["panel_enriched"]
    assert isinstance(enriched, pd.DataFrame)
    assert {"date", "futures_involved", "ats", "capacity"}.issubset(enriched.columns)

    assert (output_dir / "synthetic_basis.csv").exists()
    assert (output_dir / "panel_enriched.csv").exists()
    assert (output_dir / "half_life_overall.csv").exists()
    assert (output_dir / "qa_checks.json").exists()
