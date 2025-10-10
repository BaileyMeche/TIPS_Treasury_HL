"""Cleveland Fed inflation expectations ingestion."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import requests


@dataclass(frozen=True)
class InflationCurveQuery:
    start_date: date
    end_date: date
    horizons: Sequence[int]


class ClevelandFedClient:
    """Pulls public Cleveland Fed inflation expectation curves."""

    BASE_URL = (
        "https://www.clevelandfed.org/-/media/project/clevelandfedtenant/"
        "clevelandfedsite/api/inflation-expectations/inflation-expectations.json"
    )

    def __init__(self, session: Optional[requests.Session] = None):
        self.session = session or requests.Session()

    def fetch_curve(self, query: InflationCurveQuery) -> pd.DataFrame:
        response = self.session.get(self.BASE_URL, timeout=30)
        response.raise_for_status()
        payload = response.json()
        observations = payload.get("observations", [])
        records = []
        start = pd.Timestamp(query.start_date)
        end = pd.Timestamp(query.end_date)
        for obs in observations:
            obs_date = pd.to_datetime(obs["date"]).date()
            if obs_date < start.date() or obs_date > end.date():
                continue
            for horizon in query.horizons:
                key = f"expInflation{horizon}Yr"
                if key not in obs:
                    continue
                records.append(
                    {
                        "date": obs_date,
                        "maturity": float(horizon),
                        "inflation": float(obs[key]) / 100.0,
                    }
                )
        return pd.DataFrame.from_records(records)


def make_sample_inflation_curve(dates: Iterable[pd.Timestamp], horizons: Sequence[int]) -> pd.DataFrame:
    """Deterministic inflation curve used for tests and the novendor pipeline."""

    out = []
    for idx, dt in enumerate(sorted({pd.Timestamp(d).date() for d in dates})):
        for horizon in sorted(horizons):
            base = 0.02 - 0.0002 * np.log1p(horizon)
            cycle = 0.0001 * np.cos(idx)
            out.append(
                {
                    "date": dt,
                    "maturity": float(horizon),
                    "inflation": base + cycle,
                }
            )
    return pd.DataFrame(out)


__all__ = ["ClevelandFedClient", "InflationCurveQuery", "make_sample_inflation_curve"]
