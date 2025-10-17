# trace_wrds_pull.py
import wrds
import pandas as pd
from settings import config

WRDS_USERNAME = config("WRDS_USERNAME")

CANDIDATES = [
    "finra.trace_ts",
    "finra.trace_ts_trades",
    "trace.ts_treasury",
    "trace.trace_treasury_trade",
]

def find_trace_ts_table():
    conn = wrds.Connection(wrds_username=WRDS_USERNAME)
    libs = conn.list_libraries()
    for full in CANDIDATES:
        lib, tbl = full.split(".", 1)
        if lib in libs:
            try:
                conn.get_table(library=lib, table=tbl, obs=1)
                conn.close()
                return lib, tbl
            except Exception:
                pass
    conn.close()
    return None, None

def pull_trace_ts_trades(start_date, end_date, cusip=None):
    lib, tbl = find_trace_ts_table()
    if lib is None:
        print("TRACE TS not found in WRDS")
        return None
    query = f"SELECT * FROM {lib}.{tbl} WHERE execution_dt BETWEEN '{start_date}' AND '{end_date}'"
    if cusip:
        query += f" AND cusip = '{cusip}'"
    conn = wrds.Connection(wrds_username=WRDS_USERNAME)
    df = conn.raw_sql(query, date_cols=["execution_dt", "report_dt", "trade_time"])  # adjust date_cols as needed
    conn.close()
    return df
