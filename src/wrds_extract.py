# wrds_extract.py

import wrds
import pandas as pd
from settings import config

WRDS_USERNAME = config("WRDS_USERNAME")

def get_wrds_connection():
    return wrds.Connection(wrds_username=WRDS_USERNAME)

def pull_trace_treasury_trades(start_date: str, end_date: str, limit: int = None) -> pd.DataFrame:
    """
    Pull TRACE (Treasury) trade-level data from WRDS, if available.
    You will need to adjust lib and table names to match your account.
    """
    conn = get_wrds_connection()
    # Example candidate location — adjust if different
    lib = "trace_enhanced"
    tbl = "trace_enhanced"  # this is commonly the “enhanced” TRACE table including many securities
    # Build query
    q = f"SELECT * FROM {lib}.{tbl} WHERE trd_exctn_dt BETWEEN '{start_date}' AND '{end_date}'"
    if limit is not None:
        q += f" LIMIT {limit}"
    try:
        df = conn.raw_sql(q, date_cols=["trd_exctn_dt", "report_dt", "trade_time"])
    finally:
        conn.close()
    return df

def pull_crsp_treasury_quotes(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pull CRSP Treasury quotes (daily) from WRDS.
    Adjust library and table names as needed.
    """
    conn = get_wrds_connection()
    lib = "crsp"
    tbl = "tfz_dly_ts2"
    q = f"SELECT * FROM {lib}.{tbl} WHERE caldt BETWEEN '{start_date}' AND '{end_date}'"
    try:
        df = conn.raw_sql(q, date_cols=["caldt"])
    finally:
        conn.close()
    return df

def pull_fisd_treasury_auction(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pull FISD Treasury auction data (issue metadata) from WRDS.
    Adjust library / table names as needed.
    """
    conn = get_wrds_connection()
    lib = "fisd"
    tbl = "fisd_treasury"
    q = f"SELECT * FROM {lib}.{tbl} WHERE auction_date BETWEEN '{start_date}' AND '{end_date}'"
    try:
        df = conn.raw_sql(q, date_cols=["auction_date"])
    finally:
        conn.close()
    return df
