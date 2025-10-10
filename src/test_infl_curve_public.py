import numpy as np
import pandas as pd
import pytest

from pathlib import Path

from planC.infl_curve_public import (
    ClevelandFedInflationCurve,
    load_clevelandfed_zero_coupon_inflation,
    parse_clevelandfed_term_structure,
)


def _build_sample_excel(path: Path) -> None:
    rows = [
        ["", "Expected Inflation", "Expected Inflation", "Inflation Uncertainty"],
        ["Date", "1-Year", "2-Year", "1-Year"],
        [pd.Timestamp("2024-01-02"), 2.5, 2.7, 0.9],
        [pd.Timestamp("2024-01-03"), 2.4, 2.6, 0.8],
    ]
    df = pd.DataFrame(rows)
    df.to_excel(path, index=False, header=False, sheet_name="Term Structure Data")


def test_parse_clevelandfed_term_structure(tmp_path):
    excel_path = tmp_path / "sample.xlsx"
    _build_sample_excel(excel_path)

    panel = parse_clevelandfed_term_structure(excel_path)

    assert list(panel.columns) == [1.0, 2.0]
    assert panel.index.tolist() == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert panel.loc[pd.Timestamp("2024-01-02"), 1.0] == pytest.approx(0.025)


def test_clevelandfed_curve_interpolation():
    dates = pd.to_datetime(["2024-01-02"])
    data = pd.DataFrame(
        [[0.02, 0.025, 0.03]],
        index=dates,
        columns=[1.0, 5.0, 10.0],
    )
    curve = ClevelandFedInflationCurve(data)

    rates = curve.zero_coupon("2024-01-02", [2.0, 7.0])
    assert rates[2.0] == pytest.approx(0.02125)
    assert rates[7.0] == pytest.approx(0.027)

    with pytest.raises(ValueError):
        curve.zero_coupon("2024-01-02", [0.5])


def test_clevelandfed_curve_panel_skip_missing():
    dates = pd.to_datetime(["2024-01-02", "2024-01-03"])
    data = pd.DataFrame(
        {
            1.0: [0.02, 0.021],
            5.0: [0.025, np.nan],
            10.0: [0.03, 0.031],
        },
        index=dates,
    )
    curve = ClevelandFedInflationCurve(data)

    panel = curve.build_panel([2.0, 10.0])
    assert panel.shape[0] == 1
    assert panel.iloc[0]["inf_swap_2y"] == pytest.approx(0.02125)


def test_load_clevelandfed_zero_coupon_inflation_local(tmp_path):
    source_path = tmp_path / "source.xlsx"
    _build_sample_excel(source_path)

    df = load_clevelandfed_zero_coupon_inflation(
        tmp_path, [1, 2], refresh=True, urls=[source_path.as_uri()]
    )

    assert set(df.columns) == {"date", "inf_swap_1y", "inf_swap_2y"}
    assert not df[["inf_swap_1y", "inf_swap_2y"]].isna().any().any()
