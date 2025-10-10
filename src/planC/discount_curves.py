"""Discount-curve construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Literal

import numpy as np
import pandas as pd


@dataclass
class CurveInterpolator:
    """Simple linear interpolator for discount factors."""

    maturities: np.ndarray
    values: np.ndarray

    def __call__(self, maturity: float) -> float:
        if maturity <= 0:
            return 1.0
        return float(np.interp(maturity, self.maturities, self.values))


def build_discount_table(
    nominal_yields: pd.DataFrame,
    inflation_curve: pd.DataFrame,
) -> pd.DataFrame:
    """Combine nominal yields with inflation expectations."""

    df = nominal_yields.merge(
        inflation_curve,
        on=["date", "maturity"],
        how="left",
        suffixes=("_nominal", "_inflation"),
    )
    df["inflation"] = df["inflation"].fillna(method="ffill").fillna(method="bfill").fillna(0.0)
    df = df.rename(columns={"yield": "nominal_yield"})
    df["real_yield"] = df["nominal_yield"] - df["inflation"]
    df["nominal_discount"] = np.exp(-df["nominal_yield"] * df["maturity"])
    df["real_discount"] = np.exp(-df["real_yield"] * df["maturity"])
    return df.sort_values(["date", "maturity"]).reset_index(drop=True)


def _make_curve(df: pd.DataFrame, valuation_date: date, column: Literal["nominal_discount", "real_discount"]) -> CurveInterpolator:
    subset = df.loc[df["date"] == pd.Timestamp(valuation_date).date()]
    if subset.empty:
        raise ValueError(f"No discount data available for {valuation_date}")
    subset = subset.sort_values("maturity")
    maturities = subset["maturity"].to_numpy(dtype=float)
    values = subset[column].to_numpy(dtype=float)
    if maturities[0] > 0:
        maturities = np.insert(maturities, 0, 0.0)
        values = np.insert(values, 0, 1.0)
    return CurveInterpolator(maturities=maturities, values=values)


def nominal_curve(df: pd.DataFrame, valuation_date: date) -> CurveInterpolator:
    return _make_curve(df, valuation_date, "nominal_discount")


def real_curve(df: pd.DataFrame, valuation_date: date) -> CurveInterpolator:
    return _make_curve(df, valuation_date, "real_discount")


def make_sample_discount_table(
    dates: Iterable[pd.Timestamp],
    maturities: Iterable[int],
    *,
    base_nominal: float = 0.02,
) -> pd.DataFrame:
    """Create a deterministic discount table for smoke tests."""

    out = []
    for idx, dt in enumerate(sorted({pd.Timestamp(d).date() for d in dates})):
        for maturity in sorted(maturities):
            nominal = base_nominal + 0.0005 * maturity + 0.0001 * np.sin(idx)
            inflation = 0.018 + 0.0001 * np.cos(idx + maturity)
            out.append(
                {
                    "date": dt,
                    "maturity": float(maturity),
                    "nominal_yield": nominal,
                    "inflation": inflation,
                    "real_yield": nominal - inflation,
                    "nominal_discount": np.exp(-nominal * maturity),
                    "real_discount": np.exp(-(nominal - inflation) * maturity),
                }
            )
    return pd.DataFrame(out)


__all__ = [
    "CurveInterpolator",
    "build_discount_table",
    "nominal_curve",
    "real_curve",
    "make_sample_discount_table",
]
