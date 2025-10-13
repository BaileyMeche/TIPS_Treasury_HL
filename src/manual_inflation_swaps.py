"""Helpers for loading manually curated Treasury inflation swap data."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

from settings import MANUAL_DATA_DIR

__all__: Iterable[str] = ["load_manual_inflation_swaps"]


_COLUMN_PATTERN = re.compile(r"USSWIT(\d+)")


def _rename_columns(columns: Iterable[str]) -> dict[str, str]:
    """Return a mapping from Bloomberg-style column names to pipeline names."""

    rename_map: dict[str, str] = {}
    for column in columns:
        match = _COLUMN_PATTERN.search(column)
        if match:
            tenor = int(match.group(1))
            rename_map[column] = f"inf_swap_{tenor}y"
    return rename_map


def load_manual_inflation_swaps(
    manual_data_dir: Path | str = MANUAL_DATA_DIR,
    *,
    filename: str = "treasury_inflation_swaps.csv",
) -> pd.DataFrame:
    """Load manually maintained Treasury inflation swap rates.

    Parameters
    ----------
    manual_data_dir:
        Directory containing ``filename``. Defaults to :data:`MANUAL_DATA_DIR` from
        :mod:`settings`.
    filename:
        Name of the CSV file that stores the manually curated Bloomberg export.

    Returns
    -------
    pandas.DataFrame
        A tidy DataFrame with a ``date`` column and inflation swap columns named
        ``inf_swap_{tenor}y`` that contain decimal rates (e.g., ``0.025`` for
        ``2.5`` percent).
    """

    manual_data_dir = Path(manual_data_dir)
    csv_path = manual_data_dir / filename
    if not csv_path.exists():
        raise FileNotFoundError(
            "Manual Treasury inflation swap data not found at"
            f" {csv_path}. Ensure the Bloomberg export is available."
        )

    df = pd.read_csv(csv_path)

    if "Dates" not in df.columns:
        raise ValueError("Expected a 'Dates' column in the manual swap CSV export.")

    df = df.rename(columns={"Dates": "date"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)

    rename_map = _rename_columns(df.columns)
    if not rename_map:
        raise ValueError("Could not identify any inflation swap tenor columns in the CSV file.")

    df = df.rename(columns=rename_map)

    swap_columns = sorted(
        rename_map.values(), key=lambda col: int(col.rsplit("_", 1)[-1][:-1])
    )

    for column in swap_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce") / 100.0

    columns_to_keep = ["date"] + swap_columns
    return df[columns_to_keep]
