# finra_agg_treasury.py (updated stub)

import requests
import pandas as pd
from settings import config
import time

BASE_URL = "https://api.finra.org/data"
GROUP = "FixedIncomeMarket"
ENDPOINTS = {"daily": "treasuryDailyAggregates", "monthly": "treasuryMonthlyAggregates"}

def get_finra_access_token():
    client_id = config("FINRA_CLIENT_ID")
    client_secret = config("FINRA_CLIENT_SECRET")
    token_url = config("FINRA_TOKEN_URL")  # e.g. "https://ews.fip.finra.org/fip/rest/ews/oauth2/access_token"
    resp = requests.post(token_url, data={"grant_type": "client_credentials"},
                         auth=(client_id, client_secret))
    resp.raise_for_status()
    return resp.json()["access_token"]

def fetch_treasury_aggregates(freq="daily", start_date=None, end_date=None, product_category=None):
    if freq not in ENDPOINTS:
        raise ValueError("freq must be daily or monthly")
    dataset = ENDPOINTS[freq]
    url = f"{BASE_URL}/group/{GROUP}/name/{dataset}"
    params = {}
    if start_date: params["startDate"] = start_date
    if end_date: params["endDate"] = end_date
    if product_category: params["productCategory"] = product_category

    token = get_finra_access_token()
    headers = {"Authorization": f"Bearer {token}"}
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    j = resp.json()
    df = pd.DataFrame(j.get("results", []))
    if freq == "daily" and "tradeDate" in df.columns:
        df["tradeDate"] = pd.to_datetime(df["tradeDate"])
    elif freq == "monthly" and "beginningOfTheMonthDate" in df.columns:
        df["beginningOfTheMonthDate"] = pd.to_datetime(df["beginningOfTheMonthDate"])
    return df
