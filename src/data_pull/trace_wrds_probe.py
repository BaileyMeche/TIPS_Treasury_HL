import wrds
import logging
from settings import config

WRDS_USERNAME = config("WRDS_USERNAME")

def scan_trace_treasury_candidates(verbose=True):
    """
    Scan all WRDS libraries/tables whose name suggests TRACE / Treasury,
    and return a list of (lib, tbl, columns) for inspection.
    """
    conn = wrds.Connection(wrds_username=WRDS_USERNAME)
    libs = conn.list_libraries()
    hits = []
    for lib in libs:
        try:
            tables = conn.list_tables(library=lib)
        except Exception as e:
            continue
        for tbl in tables:
            low = tbl.lower()
            if ("trace" in low) or ("treas" in low) or ("ts" in low):
                try:
                    df0 = conn.get_table(library=lib, table=tbl, obs=0)
                    cols = df0.columns.tolist()
                    hits.append((lib, tbl, cols))
                    if verbose:
                        print(f"Candidate: {lib}.{tbl} — cols: {cols}")
                except Exception as e:
                    if verbose:
                        print(f"Could not read {lib}.{tbl}: {e}")
                    continue
    conn.close()
    return hits

def try_pull_table(lib, tbl, limit=100):
    """
    Try pulling a small sample from lib.tbl for debugging.
    """
    import pandas as pd
    conn = wrds.Connection(wrds_username=WRDS_USERNAME)
    try:
        df = conn.raw_sql(f"SELECT * FROM {lib}.{tbl} LIMIT {limit}")
        print(f"Pulled {lib}.{tbl}, rows: {len(df)}")
        print("Cols:", df.columns.tolist())
    except Exception as e:
        print(f"Error pulling {lib}.{tbl}: {e}")
        df = None
    finally:
        conn.close()
    return df
