from datetime import datetime
from io import BytesIO
from pathlib import Path

import pandas as pd
import requests

from settings import config

DATA_DIR = config("DATA_DIR")

# Define the URL for the TIPS yield data
TIPS_URL = "https://www.federalreserve.gov/data/yield-curve-tables/feds200805.csv"


def _cutoff_date(years: int, today: datetime | None = None) -> pd.Timestamp:
    if today is None:
        today_ts = pd.Timestamp.today().normalize()
    else:
        today_ts = pd.Timestamp(today).normalize()
    return today_ts - pd.DateOffset(years=years)


def pull_fed_tips_yield_curve(years: int = 3, today: datetime | None = None):
    """
    Download and process the latest zero-coupon TIPS yield curve from the Federal Reserve.

    Expected CSV structure:
    - Metadata in the first 19 rows (to be skipped).
    - 'Date' column in YMD format.
    - TIPS yield columns named 'TIPSY02', 'TIPSY05', 'TIPSY10', 'TIPSY20'.
    - Yield values are in percentage terms and must be converted to decimals.

    Returns:
        pd.DataFrame: Processed TIPS yield data.
    """
    # Fetch the data from the Federal Reserve
    response = requests.get(TIPS_URL)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch TIPS data: HTTP {response.status_code}")
    
    # Read CSV while skipping the first 19 rows (metadata)
    df = pd.read_csv(BytesIO(response.content), skiprows=18)
    if "Date" not in df.columns:
        raise ValueError("Expected a 'Date' column in the TIPS yield data")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    cutoff = _cutoff_date(years, today)
    df = df[df["Date"] >= cutoff].copy()

    df = df.rename(columns={"Date": "date"})
    df = df.sort_values("date")

    return df

def save_tips_yield_curve(df, data_dir):
    """
    Save the TIPS yield curve DataFrame to a parquet file.
    """
    path = Path(data_dir) / "fed_tips_yield_curve.parquet"
    df.to_parquet(path, index=False)

def load_tips_yield_curve(data_dir):
    """
    Load the TIPS yield curve DataFrame from a parquet file.
    Selects and renames the following columns:
    
    Source columns: TIPSY02, TIPSY05, TIPSY10, TIPSY20, TIPSY30
    Target columns: ['TIPS_Treasury_02Y', 'TIPS_Treasury_05Y', 'TIPS_Treasury_10Y', 'TIPS_Treasury_20Y']
    
    Note: TIPSY30 is ignored since only four target columns are provided.
    """
    path = Path(data_dir) / "fed_tips_yield_curve.parquet"
    df = pd.read_parquet(path)

    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in cached TIPS data")

    # Select only the required columns (ignoring TIPSY30)
    selected_cols = ["TIPSY02", "TIPSY05", "TIPSY10", "TIPSY20"]
    df = df[["date"] + selected_cols]

    # Rename the selected columns as specified.
    rename_mapping = {
        "TIPSY02": "TIPS_Treasury_02Y",
        "TIPSY05": "TIPS_Treasury_05Y",
        "TIPSY10": "TIPS_Treasury_10Y",
        "TIPSY20": "TIPS_Treasury_20Y",
    }
    df = df.rename(columns=rename_mapping)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date")

    return df

# Example usage
if __name__ == "__main__":
    tips_df = pull_fed_tips_yield_curve()
    save_tips_yield_curve(tips_df, DATA_DIR)
    # To load the data later
    #loaded_tips_df = load_tips_yield_curve(DATA_DIR)
