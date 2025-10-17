# macro_pull.py
import pandas as pd
import requests
from settings import config

def fetch_cpi_fed():
    """
    Example: fetch CPI or core inflation from FRED (via their API).
    You’ll need a FRED API key in config.
    """
    FRED_KEY = config("FRED_API_KEY")
    # e.g. CPIAUCSL series
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": "CPIAUCSL",
        "api_key": FRED_KEY,
        "file_type": "json"
    }
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json().get("observations", [])
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.set_index("date")

# You can add more macro / fiscal pulls: e.g. federal debt, deficit, survey inflation expectations, etc.
