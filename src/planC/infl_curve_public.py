"""Utilities for working with the Cleveland Fed inflation expectations curve.

This module provides helpers to download and parse the Cleveland Fed term
structure of inflation expectations and exposes an interpolation interface
that mirrors the zero-coupon conventions documented by ICE for USD inflation
swaps. Rates are returned in decimal form (e.g., 0.02 for 2 percent).
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import requests

__all__ = [
    "ClevelandFedInflationCurve",
    "download_clevelandfed_term_structure",
    "load_clevelandfed_term_structure",
    "load_clevelandfed_zero_coupon_inflation",
]

_ENV_URL_KEY = "CLEVELANDFED_TERM_STRUCTURE_URL"
_DEFAULT_RAW_FILENAME = "ie-term-structure-data.xlsx"
_DEFAULT_PARQUET_FILENAME = "clevelandfed_inflation_term_structure.parquet"


def _storage_dir(data_dir: Path) -> Path:
    base = Path(data_dir) / "clevelandfed"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _candidate_urls() -> List[str]:
    explicit = os.getenv(_ENV_URL_KEY)
    if explicit:
        return [explicit]

    year = datetime.utcnow().year
    base_paths = [
        "https://www.clevelandfed.org/-/media/project/clevelandfedten/fedcom/sections/data/inflation-expectations/latest/ie-term-structure-data.xlsx",
        f"https://www.clevelandfed.org/-/media/project/clevelandfedten/fedcom/sections/data/inflation-expectations/{year}/ie-term-structure-data.xlsx",
        f"https://www.clevelandfed.org/-/media/project/clevelandfedten/fedcom/sections/data/inflation-expectations/{year - 1}/ie-term-structure-data.xlsx",
        "https://www.clevelandfed.org/-/media/content/indicators-and-data/inflation-expectations/ie-term-structure-data.xlsx",
        "https://www.clevelandfed.org/~/media/project/clevelandfedten/fedcom/sections/data/inflation-expectations/latest/ie-term-structure-data.xlsx",
    ]
    # Preserve order but drop duplicates.
    return list(dict.fromkeys(base_paths))


def _download_candidate(url: str, destination: Path, timeout: int = 30) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    parsed = urlparse(url)
    request: object
    if parsed.scheme in {"http", "https"}:
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        stream_kwargs = {"timeout": timeout}
    else:
        # Allow loading from local files for testing.
        request = url
        stream_kwargs = {}

    with urlopen(request, **stream_kwargs) as resp:  # type: ignore[arg-type]
        data = resp.read()
    destination.write_bytes(data)


def download_clevelandfed_term_structure(
    data_dir: Path | str,
    *,
    urls: Optional[Sequence[str]] = None,
    force: bool = False,
) -> Path:
    """Download the Cleveland Fed term-structure Excel workbook.

    Parameters
    ----------
    data_dir:
        Directory where the downloaded file should be stored.
    urls:
        Optional override for the download URL list. By default a list of
        Cleveland Fed endpoints is attempted until one succeeds.
    force:
        If ``True`` the file will be downloaded even when it already exists.

    Returns
    -------
    Path
        Location of the downloaded Excel file.

    Raises
    ------
    RuntimeError
        If none of the candidate URLs returned data.
    """

    base = _storage_dir(Path(data_dir))
    destination = base / _DEFAULT_RAW_FILENAME
    if destination.exists() and not force:
        return destination

    errors: List[str] = []
    for candidate in urls or _candidate_urls():
        try:
            _download_candidate(candidate, destination)
            return destination
        except (HTTPError, URLError, OSError) as exc:
            errors.append(f"{candidate}: {exc}")
    raise RuntimeError(
        "Unable to download the Cleveland Fed inflation expectations term "
        "structure. Tried the following URLs: "
        + "; ".join(errors)
        + ". You can manually download the workbook from "
        "https://www.clevelandfed.org/indicators-and-data/inflation-expectations "
        "and place it at "
        + str(destination)
        + "."
    )


def _combine_headers(raw: pd.DataFrame, header_row: int) -> List[str]:
    header = raw.iloc[header_row].fillna("").astype(str).tolist()
    if header_row == 0:
        return [col.strip() for col in header]

    parent = (
        raw.iloc[:header_row]
        .ffill()
        .iloc[-1]
        .fillna("")
        .astype(str)
        .tolist()
    )
    combined: List[str] = []
    for top, bottom in zip(parent, header):
        top_clean = top.strip()
        bottom_clean = bottom.strip()
        if not bottom_clean:
            combined.append(top_clean)
        elif top_clean and top_clean.lower() != "date":
            combined.append(f"{top_clean}__{bottom_clean}")
        else:
            combined.append(bottom_clean)
    return combined


def _find_header_row(raw: pd.DataFrame) -> Optional[int]:
    for idx in range(len(raw)):
        cell = str(raw.iloc[idx, 0]).strip().lower()
        if cell == "date":
            return idx
    return None


_EXPECTED_PATTERN = re.compile(r"expected\s*inflation", re.IGNORECASE)
_YEAR_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:year|yr|y)", re.IGNORECASE)
_MONTH_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*(?:month|mo|m)", re.IGNORECASE)
_COMPACT_YEAR_PATTERN = re.compile(r"expinf\s*(\d+(?:\.\d+)?)", re.IGNORECASE)


def _infer_maturity(label: str) -> Optional[float]:
    lowered = label.lower()
    year_match = _YEAR_PATTERN.search(lowered)
    if year_match:
        return float(year_match.group(1))
    month_match = _MONTH_PATTERN.search(lowered)
    if month_match:
        return float(month_match.group(1)) / 12.0
    compact = _COMPACT_YEAR_PATTERN.search(lowered)
    if compact:
        return float(compact.group(1))
    digits = re.search(r"(\d+(?:\.\d+)?)", lowered)
    if digits:
        return float(digits.group(1))
    return None


def parse_clevelandfed_term_structure(excel_path: Path) -> pd.DataFrame:
    """Parse the Cleveland Fed workbook into a zero-coupon panel."""

    if not excel_path.exists():
        raise FileNotFoundError(excel_path)

    xls = pd.ExcelFile(excel_path)
    for sheet in xls.sheet_names:
        raw = xls.parse(sheet_name=sheet, header=None)
        header_row = _find_header_row(raw)
        if header_row is None:
            continue
        columns = _combine_headers(raw, header_row)
        data = raw.iloc[header_row + 1 :].copy()
        data.columns = columns
        data = data.dropna(how="all")

        date_cols = [c for c in data.columns if str(c).strip().lower() == "date"]
        if not date_cols:
            continue
        date_col = date_cols[0]

        expected_cols = [
            col for col in data.columns if _EXPECTED_PATTERN.search(str(col))
        ]
        if not expected_cols:
            continue

        subset = data[[date_col] + expected_cols].copy()
        subset[date_col] = pd.to_datetime(subset[date_col], errors="coerce")
        subset = subset.dropna(subset=[date_col])
        for col in expected_cols:
            subset[col] = pd.to_numeric(subset[col], errors="coerce")
        subset = subset.dropna(how="all", subset=expected_cols)
        if subset.empty:
            continue

        maturity_map: dict[str, float] = {}
        for col in expected_cols:
            maturity = _infer_maturity(str(col))
            if maturity is not None:
                maturity_map[col] = maturity

        if not maturity_map:
            continue

        numeric = subset[list(maturity_map)].copy()
        if not numeric.empty:
            values = np.abs(numeric.to_numpy(dtype=float)).ravel()
            max_abs = np.nanmax(values) if values.size else np.nan
            if np.isnan(max_abs):
                max_abs = 0.0
            if max_abs > 1.0:
                numeric = numeric / 100.0
        subset[list(maturity_map)] = numeric

        panel = subset.rename(columns=maturity_map).set_index(date_col)
        panel.columns = panel.columns.astype(float)
        panel = panel.groupby(panel.columns, axis=1).mean()
        panel = panel.sort_index(axis=1).sort_index()
        panel.index = panel.index.tz_localize(None)
        return panel

    raise RuntimeError(
        "Unable to locate an \"Expected Inflation\" table in the Cleveland Fed "
        "term-structure workbook."
    )


@dataclass
class ClevelandFedInflationCurve:
    """Daily zero-coupon inflation curve constructed from Cleveland Fed data."""

    zero_coupon_rates: pd.DataFrame

    def available_maturities(self) -> List[float]:
        return list(self.zero_coupon_rates.columns.astype(float))

    def _prepare_row(self, date: pd.Timestamp) -> pd.Series:
        try:
            row = self.zero_coupon_rates.loc[date]
        except KeyError:
            raise KeyError(f"No Cleveland Fed curve available for {date:%Y-%m-%d}")
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return row.astype(float)

    def zero_coupon(
        self,
        date: pd.Timestamp | str,
        maturities: Sequence[float],
        *,
        allow_partial: bool = False,
    ) -> pd.Series:
        ts = pd.to_datetime(date)
        row = self._prepare_row(ts)
        available = row.dropna()
        if len(available) < 2:
            raise ValueError(
                "At least two maturities are required to interpolate zero-coupon "
                "inflation rates."
            )
        x = available.index.to_numpy(dtype=float)
        y = available.to_numpy(dtype=float)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        unique_x, unique_idx = np.unique(x, return_index=True)
        x = unique_x
        y = y[unique_idx]

        target = np.array([float(m) for m in maturities], dtype=float)
        mask = (target >= x.min()) & (target <= x.max())
        if not allow_partial and not mask.all():
            raise ValueError(
                "Requested maturities fall outside the supported range. "
                f"Available range: [{x.min():.2f}, {x.max():.2f}] years."
            )

        result = np.full(target.shape, np.nan, dtype=float)
        if mask.any():
            result[mask] = np.interp(target[mask], x, y)
        return pd.Series(result, index=maturities, dtype=float)

    def build_panel(
        self,
        maturities: Sequence[float],
        *,
        allow_partial: bool = False,
    ) -> pd.DataFrame:
        target = np.array(sorted({float(m) for m in maturities}), dtype=float)
        records: List[dict[str, float | pd.Timestamp]] = []
        for date, row in self.zero_coupon_rates.sort_index().iterrows():
            available = row.dropna()
            if len(available) < 2:
                continue
            x = available.index.to_numpy(dtype=float)
            y = available.to_numpy(dtype=float)
            order = np.argsort(x)
            x = x[order]
            y = y[order]
            unique_x, unique_idx = np.unique(x, return_index=True)
            x = unique_x
            y = y[unique_idx]
            mask = (target >= x.min()) & (target <= x.max())
            if not allow_partial and not mask.all():
                if mask.any():
                    continue
                continue
            values = np.full(target.shape, np.nan, dtype=float)
            if mask.any():
                values[mask] = np.interp(target[mask], x, y)
            record: dict[str, float | pd.Timestamp] = {"date": date}
            for maturity, value in zip(target, values):
                record[_format_maturity_label(maturity)] = float(value)
            records.append(record)
        panel = pd.DataFrame.from_records(records)
        if panel.empty:
            return panel
        panel = panel.sort_values("date").reset_index(drop=True)
        return panel


def _format_maturity_label(maturity: float) -> str:
    if maturity >= 1:
        if math.isclose(maturity % 1, 0):
            return f"inf_swap_{int(round(maturity))}y"
        return f"inf_swap_{str(maturity).replace('.', 'p')}y"
    months = int(round(maturity * 12))
    return f"inf_swap_{months}m"


def load_clevelandfed_term_structure(
    data_dir: Path | str,
    *,
    refresh: bool = False,
    urls: Optional[Sequence[str]] = None,
) -> ClevelandFedInflationCurve:
    """Load (and optionally download) the Cleveland Fed inflation curve."""

    data_dir = Path(data_dir)
    base = _storage_dir(data_dir)
    parquet_path = base / _DEFAULT_PARQUET_FILENAME
    excel_path = base / _DEFAULT_RAW_FILENAME

    if refresh or not parquet_path.exists():
        if refresh or not excel_path.exists():
            excel_path = download_clevelandfed_term_structure(
                data_dir, urls=urls, force=refresh
            )
        try:
            panel = parse_clevelandfed_term_structure(excel_path)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Cleveland Fed term-structure file is missing. Expected to find "
                f"{excel_path}."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "Unable to parse the Cleveland Fed inflation expectations "
                f"workbook at {excel_path}: {exc}"
            ) from exc
        panel.to_parquet(parquet_path, compression="snappy")
    else:
        panel = pd.read_parquet(parquet_path)
        panel.index = pd.to_datetime(panel.index)

    panel = panel.sort_index().sort_index(axis=1)
    panel.columns = panel.columns.astype(float)
    return ClevelandFedInflationCurve(panel)


def load_clevelandfed_zero_coupon_inflation(
    data_dir: Path | str,
    maturities: Sequence[float],
    *,
    refresh: bool = False,
    allow_partial: bool = False,
    urls: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Return zero-coupon inflation rates for the requested maturities."""

    curve = load_clevelandfed_term_structure(
        data_dir, refresh=refresh, urls=urls
    )
    panel = curve.build_panel(maturities, allow_partial=allow_partial)
    if panel.empty:
        available = ", ".join(f"{m:.0f}y" for m in curve.available_maturities())
        raise RuntimeError(
            "Cleveland Fed inflation expectations data does not cover the "
            "requested maturities. Available maturities include: "
            f"{available}."
        )
    return panel
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
