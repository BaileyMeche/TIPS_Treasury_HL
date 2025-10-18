# trace_wrds_pull2.py

import wrds
import pandas as pd
import logging
from settings import config

WRDS_USERNAME = config("WRDS_USERNAME")

CANDIDATES = [
    "finra.trace_treasury",
    "trace.trace_treasury",
    "trace_ts.treasury",
    "finra.trace_ts_trades",
    "trace.trace_ts",
]

def find_trace_ts_table():
    conn = wrds.Connection(wrds_username=WRDS_USERNAME)
    libs = conn.list_libraries()
    for full in CANDIDATES:
        lib, tbl = full.split(".", 1)
        if lib in libs:
            try:
                conn.get_table(library=lib, table=tbl, obs=1)
                logging.info(f"Found candidate TRACE TS table: {lib}.{tbl}")
                conn.close()
                return lib, tbl
            except Exception as e:
                logging.debug(f"Candidate {lib}.{tbl} failed: {e}")
    conn.close()
    logging.warning("No TRACE TS table found in WRDS among candidates")
    return None, None

def pull_trace_ts_trades(start_date, end_date, cusip=None, limit=None):
    lib, tbl = find_trace_ts_table()
    if lib is None:
        print("TRACE TS not found in WRDS")
        return None

    query = f"SELECT * FROM {lib}.{tbl} WHERE execution_dt BETWEEN '{start_date}' AND '{end_date}'"
    if cusip:
        query += f" AND cusip = '{cusip}'"
    if limit:
        query += f" LIMIT {limit}"

    conn = wrds.Connection(wrds_username=WRDS_USERNAME)
    try:
        df = conn.raw_sql(query, date_cols=["execution_dt","report_dt","trade_time"])
        print(f"TRACE TS pull succeeded: {lib}.{tbl}, rows = {len(df)}")
    except Exception as e:
        print(f"Error fetching {lib}.{tbl}: {e}")
        df = None
    finally:
        conn.close()
    return df
