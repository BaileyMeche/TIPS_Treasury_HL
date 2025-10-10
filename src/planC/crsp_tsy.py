"""Pull nominal Treasury data from WRDS CRSP."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from .wrds_trace import WRDSTraceError


@dataclass(frozen=True)
class TreasuryQuery:
    start_date: date
    end_date: date
    maturities: Optional[Sequence[int]] = None


class CRSPTreasuryClient:
    """Wrapper around the CRSP daily Treasury tables."""

    def __init__(self, connection_factory=None, *, schema: str = "crsp", table: str = "tfcdi"):
        self.schema = schema
        self.table = table
        self._connection_factory = connection_factory
        self._connection = None

    def _connect(self):
        if self._connection is None:
            if self._connection_factory is not None:
                self._connection = self._connection_factory()
            else:  # pragma: no cover - requires WRDS
                try:
                    import wrds

                    self._connection = wrds.Connection()
                except Exception as exc:  # pragma: no cover - integration
                    raise WRDSTraceError("Failed to connect to WRDS for CRSP") from exc
        return self._connection

    def fetch_nominal_yields(self, query: TreasuryQuery) -> pd.DataFrame:
        sql = self._build_sql(query)
        try:
            conn = self._connect()
            df = conn.raw_sql(sql)
        except Exception as exc:  # pragma: no cover - depends on WRDS
            raise WRDSTraceError("CRSP Treasury pull failed") from exc

        if df.empty:
            return df

        df = df.rename(columns={"date": "obs_date"})
        df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.date
        maturity_columns = [c for c in df.columns if c.lower().startswith("yld")]
        records = []
        for _, row in df.iterrows():
            for maturity_col in maturity_columns:
                maturity_years = int(maturity_col.strip("yld")) / 12
                if query.maturities and int(maturity_years) not in query.maturities:
                    continue
                records.append(
                    {
                        "date": row["obs_date"],
                        "maturity": maturity_years,
                        "yield": row[maturity_col] / 100.0,
                    }
                )
        return pd.DataFrame.from_records(records)

    def _build_sql(self, query: TreasuryQuery) -> str:
        where_clause = (
            f"date between '{query.start_date:%Y-%m-%d}' and '{query.end_date:%Y-%m-%d}'"
        )
        columns = ["date", "yld1m", "yld3m", "yld6m", "yld1", "yld2", "yld5", "yld7", "yld10"]
        return f"select {', '.join(columns)} from {self.schema}.{self.table} where {where_clause}"


def make_sample_treasury_yields(dates: Iterable[pd.Timestamp], maturities: Sequence[int]) -> pd.DataFrame:
    """Create a deterministic set of nominal yields for use in tests."""

    maturities = sorted(maturities)
    out = []
    for idx, dt in enumerate(sorted({pd.Timestamp(d).date() for d in dates})):
        for maturity in maturities:
            base = 0.015 + 0.0005 * maturity
            seasonality = 0.0003 * np.sin(idx)
            out.append(
                {
                    "date": dt,
                    "maturity": float(maturity),
                    "yield": base + seasonality,
                }
            )
    return pd.DataFrame(out)


__all__ = ["CRSPTreasuryClient", "TreasuryQuery", "make_sample_treasury_yields"]
