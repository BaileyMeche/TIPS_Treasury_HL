"""Utilities for working with TRACE data via WRDS.

The module keeps direct dependencies on the :mod:`wrds` client contained in a
small wrapper so that it can easily be mocked in tests.  In production the
functions will use the live WRDS connection, while unit tests can inject
pre-built data frames.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional, Sequence

import pandas as pd


class WRDSTraceError(RuntimeError):
    """Raised when an error occurs while talking to WRDS."""


@dataclass(frozen=True)
class TraceQuery:
    """Description of the TRACE pull to perform."""

    start_date: date
    end_date: date
    cusips: Optional[Sequence[str]] = None


class WRDSTraceClient:
    """Thin wrapper around :mod:`wrds` tailored to the TRACE schema."""

    def __init__(
        self,
        *,
        schema: str = "trace",
        table: str = "ct",
        connection_factory=None,
    ) -> None:
        self.schema = schema
        self.table = table
        self._connection = None
        self._connection_factory = connection_factory

    def _connect(self):
        if self._connection is None:
            if self._connection_factory is not None:
                self._connection = self._connection_factory()
            else:  # pragma: no cover - exercised in integration usage
                try:
                    import wrds

                    self._connection = wrds.Connection()
                except Exception as exc:  # pragma: no cover - best effort logging
                    raise WRDSTraceError("Failed to create WRDS connection") from exc
        return self._connection

    def fetch_trades(self, query: TraceQuery) -> pd.DataFrame:
        """Return TRACE trades between ``start_date`` and ``end_date``.

        Parameters
        ----------
        query:
            The :class:`TraceQuery` describing the slice of TRACE data.
        """

        sql = self._build_sql(query)
        try:
            conn = self._connect()
            df = conn.raw_sql(sql)
        except Exception as exc:  # pragma: no cover - depends on WRDS
            raise WRDSTraceError("TRACE pull failed") from exc

        if df.empty:
            return df

        df = df.copy()
        if "trade_dt" in df:
            df["trade_dt"] = pd.to_datetime(df["trade_dt"]).dt.date
        if "coupon" in df:
            df["coupon"] = df["coupon"].astype(float)
        if "maturity" in df:
            df["maturity"] = pd.to_datetime(df["maturity"]).dt.date
            df["years_to_maturity"] = (
                (pd.to_datetime(df["maturity"]) - pd.to_datetime(df["trade_dt"]))
                .dt.days
                .div(365.25)
            )
        return df

    # ------------------------------------------------------------------
    def _build_sql(self, query: TraceQuery) -> str:
        where_parts = [
            f"trade_dt between '{query.start_date:%Y-%m-%d}' and '{query.end_date:%Y-%m-%d}'"
        ]
        if query.cusips:
            cusips = ", ".join(f"'{c}'" for c in query.cusips)
            where_parts.append(f"cusip_id in ({cusips})")
        where_clause = " and ".join(where_parts)
        columns = [
            "trade_dt",
            "cusip_id as cusip",
            "price",
            "par",
            "coupon",
            "maturity",
        ]
        sql = (
            f"select {', '.join(columns)}\n"
            f"from {self.schema}.{self.table}\n"
            f"where {where_clause}"
        )
        return sql


# ---------------------------------------------------------------------------
# Utility helpers used by the pipeline and its unit tests


def make_sample_trace_data(dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """Return a deterministic sample TRACE-like data set."""

    records = []
    cusips = ["9128285M8", "912828Z60"]
    coupons = [0.02, 0.0225]
    maturities = [5.0, 7.0]
    for idx, dt in enumerate(sorted({pd.Timestamp(d).date() for d in dates})):
        for cusip, coupon, maturity in zip(cusips, coupons, maturities):
            records.append(
                {
                    "trade_dt": dt,
                    "cusip": cusip,
                    "price": 100 + idx,
                    "par": 1000000,
                    "coupon": coupon,
                    "maturity": pd.Timestamp(dt) + pd.DateOffset(years=int(maturity)),
                    "years_to_maturity": maturity,
                }
            )
    return pd.DataFrame.from_records(records)


def summarise_trades(trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate TRACE trades to the valuation-day level."""

    if trades.empty:
        return pd.DataFrame(
            columns=["trade_dt", "avg_coupon", "avg_years_to_maturity", "trade_count"]
        )

    grouped = (
        trades.groupby("trade_dt")
        .agg(
            avg_coupon=("coupon", "mean"),
            avg_years_to_maturity=("years_to_maturity", "mean"),
            trade_count=("cusip", "count"),
        )
        .reset_index()
        .sort_values("trade_dt")
    )
    return grouped


__all__ = [
    "TraceQuery",
    "WRDSTraceClient",
    "WRDSTraceError",
    "make_sample_trace_data",
    "summarise_trades",
]
