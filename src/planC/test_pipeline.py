"""Unit tests for the Plan C pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from planC import pipeline
from planC import discount_curves
from planC import proof_of_concept
from planC.crsp_tsy import make_sample_treasury_yields
from planC.infl_curve_public import make_sample_inflation_curve
from planC.wrds_trace import make_sample_trace_data, summarise_trades


def test_discount_table_building():
    dates = pd.date_range("2023-01-01", periods=3, freq="D")
    maturities = [1, 3, 5]
    nominal = make_sample_treasury_yields(dates, maturities)
    inflation = make_sample_inflation_curve(dates, maturities)
    table = discount_curves.build_discount_table(nominal, inflation)
    assert {"nominal_yield", "real_yield", "nominal_discount", "real_discount"}.issubset(table.columns)
    assert not table.empty


def test_pipeline_sample_run(tmp_path):
    out_dir = tmp_path / "planC"
    result = pipeline.run_pipeline(
        "2023-01-01",
        "2023-01-05",
        output_dir=out_dir,
        use_sample_data=True,
    )
    assert set(pipeline.OUTPUT_FILES).issubset(result.keys())
    for filename in pipeline.OUTPUT_FILES.values():
        assert (out_dir / filename).exists()
    basis = result["synthetic_basis"]
    assert "basis" in basis.columns
    assert not basis.empty


def test_cli_with_sample_data(tmp_path):
    output_dir = tmp_path / "cli"
    args = [
        "--start",
        "2023-01-01",
        "--end",
        "2023-01-03",
        "--sample-data",
        "--output-dir",
        str(output_dir),
    ]
    result = pipeline.main(args)
    for filename in pipeline.OUTPUT_FILES.values():
        assert (output_dir / filename).exists()
    assert "basis" in result["synthetic_basis"].columns


def test_trace_summary_from_sample():
    dates = pd.date_range("2023-01-01", periods=2, freq="D")
    trades = make_sample_trace_data(dates)
    summary = summarise_trades(trades)
    assert (summary["trade_count"] > 0).all()
    assert len(summary) == len(dates)


def test_proof_of_concept_runner(tmp_path):
    output_dir = tmp_path / "poc"
    report_path = tmp_path / "planC_proof_of_concept.md"
    artefacts = proof_of_concept.run_proof_of_concept(
        "2023-01-01",
        "2023-01-05",
        output_dir=output_dir,
        report_path=report_path,
        live_data=False,
    )
    assert report_path.exists()
    assert artefacts["summary_path"].exists()
    assert not (output_dir / "basis_timeseries.png").exists()
    summary_df = pd.read_csv(artefacts["summary_path"])
    assert "metric" in summary_df.columns
    assert artefacts["results"]["synthetic_basis"].shape[0] > 0
