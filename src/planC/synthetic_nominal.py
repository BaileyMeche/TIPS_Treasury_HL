"""Utilities for constructing synthetic nominal pricing."""

from __future__ import annotations

import pandas as pd


def compute_basis(nominal_cashflows: pd.DataFrame, tips_cashflows: pd.DataFrame) -> pd.DataFrame:
    """Return the synthetic nominal versus TIPS basis by valuation date."""

    nominal = (
        nominal_cashflows.groupby("valuation_date")["present_value"].sum().rename("nominal_present_value")
    )
    tips = tips_cashflows.groupby("valuation_date")["present_value"].sum().rename("tips_present_value")
    combined = nominal.to_frame().join(tips, how="outer").fillna(0.0)
    combined["basis"] = combined["nominal_present_value"] - combined["tips_present_value"]
    return combined.reset_index()


def combine_cashflows(nominal_cashflows: pd.DataFrame, tips_cashflows: pd.DataFrame) -> pd.DataFrame:
    """Return both cash-flow legs stacked together with a "leg" column."""

    nominal = nominal_cashflows.copy()
    nominal["leg"] = "nominal"
    tips = tips_cashflows.copy()
    tips["leg"] = "tips"
    return pd.concat([nominal, tips], ignore_index=True, sort=False)


__all__ = ["compute_basis", "combine_cashflows"]
