import pandas as pd
import pytest

from manual_inflation_swaps import load_manual_inflation_swaps


def test_load_manual_inflation_swaps_structure():
    df = load_manual_inflation_swaps()

    assert 'date' in df.columns
    expected_columns = {f"inf_swap_{tenor}y" for tenor in (1, 2, 3, 4, 5, 10, 20, 30)}
    assert expected_columns.issuperset(set(df.columns) - {'date'})
    assert pd.api.types.is_datetime64_any_dtype(df['date'])


def test_load_manual_inflation_swaps_decimal_conversion():
    df = load_manual_inflation_swaps()

    sample = df.iloc[0]
    assert sample['inf_swap_2y'] == pytest.approx(0.015827, abs=1e-6)
    assert sample['inf_swap_10y'] == pytest.approx(0.028239, abs=1e-6)
