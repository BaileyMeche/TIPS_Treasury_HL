"""Nominal Treasury cash-flow construction."""

from __future__ import annotations

from datetime import date
from typing import Iterable

import pandas as pd

from . import discount_curves


def build_cashflows(
    discount_table: pd.DataFrame,
    valuation_date: date,
    *,
    coupon_rate: float,
    years_to_maturity: float,
    frequency: int = 2,
    face_value: float = 100.0,
) -> pd.DataFrame:
    """Create discounted nominal cash flows."""

    curve = discount_curves.nominal_curve(discount_table, valuation_date)
    periods = int(round(years_to_maturity * frequency))
    if periods <= 0:
        periods = frequency
    valuation_date = pd.Timestamp(valuation_date).date()
    rows = []
    for n in range(1, periods + 1):
        maturity_years = n / frequency
        payment_date = (pd.Timestamp(valuation_date) + pd.DateOffset(months=int(12 / frequency * n))).date()
        cash_flow = face_value * coupon_rate / frequency
        if n == periods:
            cash_flow += face_value
        discount_factor = curve(maturity_years)
        present_value = cash_flow * discount_factor
        rows.append(
            {
                "valuation_date": valuation_date,
                "payment_date": payment_date,
                "period": n,
                "cash_flow": cash_flow,
                "discount_factor": discount_factor,
                "present_value": present_value,
            }
        )
    return pd.DataFrame(rows)


def concatenate_cashflows(flows: Iterable[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(list(flows), ignore_index=True) if flows else pd.DataFrame()


__all__ = ["build_cashflows", "concatenate_cashflows"]
