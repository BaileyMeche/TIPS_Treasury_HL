"""Tools for loading the TIPS Daily Index Ratio series."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional

import pandas as pd

from .wrds_trace import WRDSTraceError


@dataclass(frozen=True)
class TipsIndexQuery:
    start_date: date
    end_date: date


class TIPSIndexClient:
    """Wrapper around the WRDS TIPS index table."""

    def __init__(self, connection_factory=None, *, schema: str = "wrdsapps", table: str = "tips_daily_index"):
        self.schema = schema
        self.table = table
        self._connection_factory = connection_factory
        self._connection = None

    def _connect(self):
        if self._connection is None:
            if self._connection_factory is not None:
                self._connection = self._connection_factory()
            else:  # pragma: no cover - integration only
                try:
                    import wrds

                    self._connection = wrds.Connection()
                except Exception as exc:  # pragma: no cover - integration only
                    raise WRDSTraceError("Failed to connect to WRDS for TIPS index") from exc
        return self._connection

    def fetch_index(self, query: TipsIndexQuery) -> pd.DataFrame:
        sql = (
            "select caldt as date, index_ratio "
            f"from {self.schema}.{self.table} "
            f"where caldt between '{query.start_date:%Y-%m-%d}' and '{query.end_date:%Y-%m-%d}'"
        )
        try:
            conn = self._connect()
            df = conn.raw_sql(sql)
        except Exception as exc:  # pragma: no cover - integration only
            raise WRDSTraceError("TIPS index pull failed") from exc

        if df.empty:
            return df
        df["date"] = pd.to_datetime(df["date"]).dt.date
        return df


def make_sample_tips_index(dates: Iterable[pd.Timestamp]) -> pd.DataFrame:
    """Produce deterministic index ratios used during testing."""

    out = []
    for idx, dt in enumerate(sorted({pd.Timestamp(d).date() for d in dates})):
        out.append({"date": dt, "index_ratio": 1.0 + 0.0002 * idx})
    return pd.DataFrame(out)


__all__ = ["TIPSIndexClient", "TipsIndexQuery", "make_sample_tips_index"]
