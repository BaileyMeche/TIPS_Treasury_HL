# finra_agg_api.py
import requests
import pandas as pd
from settings import config

# Base URL per FINRA API docs
BASE = "https://api.finra.org/data"
GROUP = "FixedIncomeMarket"
# endpoint names from FINRA docs: “treasuryDailyAggregates”, “treasuryMonthlyAggregates”, “treasuryWeeklyAggregates”
ENDPOINTS = {
    "daily": "treasuryDailyAggregates",
    "monthly": "treasuryMonthlyAggregates",
    "weekly": "treasuryWeeklyAggregates",
}

def fetch_agg_treasury(freq="daily", start_date=None, end_date=None):
    """
    Fetch TRACE Treasury aggregate data from FINRA API for freq in ["daily","monthly","weekly"].
    Returns DataFrame or raises error.
    """
    if freq not in ENDPOINTS:
        raise ValueError("freq must be one of daily, monthly, weekly")
    dataset = ENDPOINTS[freq]
    url = f"{BASE}/group/{GROUP}/name/{dataset}"
    params = {}
    if start_date:
        params["startDate"] = start_date
    if end_date:
        params["endDate"] = end_date
    # Public API may require api key; but docs show public access for aggregates. 
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    j = resp.json()
    df = pd.DataFrame(j.get("results", []))
    return df
