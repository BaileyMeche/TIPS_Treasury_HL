# wrds_pull.py
import wrds
import pandas as pd
from settings import config

WRDS_USERNAME = config("WRDS_USERNAME")

def pull_crsp_treasury_panel(start_date=None, end_date=None):
    """
    Pull CRSP Treasury quotes, yields, header, debt, and merge into a panel.
    (This is your existing master panel.)
    """
    from data_pull.crsp_treasury_pull import pull_bmquotes, pull_bmyield, pull_header, pull_debt_outstanding
    dfq = pull_bmquotes(start_date, end_date)
    dfy = pull_bmyield(start_date, end_date)
    hdr = pull_header()
    debt = pull_debt_outstanding(start_date, end_date)
    df = pd.merge(dfq, dfy, on=["crspid", "qdate"], how="outer")
    df = pd.merge(df, hdr, on="crspid", how="left")
    df = pd.merge(df, debt, on=["crspid", "qdate"], how="left")
    return df

# You can add other WRDS pulls if e.g. TRACE-TS is in WRDS:
def pull_wrds_trace_ts(start_date, end_date):
    """
    Attempt to pull TRACE-TS trades from WRDS, if library exists.
    Returns None if not accessible.
    """
    from trace_wrds_pull import pull_trace_ts_trades
    return pull_trace_ts_trades(start_date, end_date)
