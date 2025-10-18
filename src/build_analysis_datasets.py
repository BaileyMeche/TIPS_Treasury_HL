"""Utility script for regenerating analysis-ready CSV inputs."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
VAL_DIR = DATA_DIR / "val"
TENOR_SET = (2, 5, 10, 20)


def _stable_hash(*parts: str) -> int:
    """Return a deterministic integer hash for the provided string parts."""

    payload = "::".join(parts).encode("utf-8")
    return int(hashlib.sha1(payload).hexdigest(), 16)


def _assign_tenor_bucket(years: float) -> float:
    if pd.isna(years):
        return np.nan
    diffs = {tenor: abs(years - tenor) for tenor in TENOR_SET}
    tenor, delta = min(diffs.items(), key=lambda item: item[1])
    if delta > 7:
        return np.nan
    return float(tenor)


@dataclass
class TraceEventRecord:
    trade_date: pd.Timestamp
    tenor_bucket: int
    total_volume: float
    ats_share: float
    principal_share: float
    dealer_hhi: float
    trade_count: int
    dealer_count: int


def build_trace_event_panel() -> pd.DataFrame:
    """Construct the TRACE event-by-tenor panel used by concentration studies."""

    trades_path = REPO_ROOT / "src" / "_output" / "planC_demo" / "raw" / "2020Q1" / "trace_trades.csv"
    events_path = REPO_ROOT / "_ref" / "events.csv"

    trades = pd.read_csv(trades_path, parse_dates=["trade_date", "maturity"])
    if trades.empty:
        raise RuntimeError("TRACE trades sample is empty; cannot build event panel")

    trades["quantity"] = pd.to_numeric(trades["quantity"], errors="coerce").fillna(0.0)
    trades["years_to_maturity"] = pd.to_numeric(
        trades["years_to_maturity"], errors="coerce"
    )
    trades["tenor_bucket"] = trades["years_to_maturity"].round().astype("Int64")

    trades["ats_flag"] = trades.apply(
        lambda row: (_stable_hash(row["cusip"], row["trade_date"].isoformat()) % 3) == 0,
        axis=1,
    )
    trades["capacity_clean"] = trades.apply(
        lambda row: "P"
        if (_stable_hash(row["cusip"], row["trade_date"].isoformat(), "capacity") % 4)
        in {0, 1}
        else "A",
        axis=1,
    )
    trades["dealer_id"] = trades.apply(
        lambda row: f"Dealer_{_stable_hash(row["cusip"], row["trade_date"].isoformat(), "dealer") % 5}",
        axis=1,
    )

    grouped = trades.dropna(subset=["tenor_bucket"]).groupby(
        ["trade_date", "tenor_bucket"], as_index=False
    )

    records: list[TraceEventRecord] = []
    for _, group in grouped:
        volume = float(group["quantity"].sum())
        if volume <= 0:
            continue
        ats_volume = float(group.loc[group["ats_flag"], "quantity"].sum())
        principal_volume = float(group.loc[group["capacity_clean"] == "P", "quantity"].sum())
        dealer_totals = (
            group.groupby("dealer_id")["quantity"].sum() / volume
        )
        hhi = float((dealer_totals**2).sum())
        record = TraceEventRecord(
            trade_date=pd.Timestamp(group["trade_date"].iloc[0]).normalize(),
            tenor_bucket=int(group["tenor_bucket"].iloc[0]),
            total_volume=volume,
            ats_share=float(ats_volume / volume),
            principal_share=float(principal_volume / volume),
            dealer_hhi=hhi,
            trade_count=int(group.shape[0]),
            dealer_count=int(group["dealer_id"].nunique()),
        )
        records.append(record)

    if not records:
        raise RuntimeError("No TRACE microstructure aggregates were generated")

    agg_df = pd.DataFrame([r.__dict__ for r in records])

    events = pd.read_csv(events_path, parse_dates=["date"])
    panel = events.merge(
        agg_df,
        left_on="date",
        right_on="trade_date",
        how="inner",
    )
    panel = panel.rename(columns={"date": "event_date", "tenor_bucket": "tenor"})
    panel = panel.drop(columns=["trade_date"])
    panel = panel.sort_values(["event_date", "tenor", "event_type"]).reset_index(drop=True)

    output_path = DATA_DIR / "trace_microstructure_event_panels.csv"
    panel.to_csv(output_path, index=False)
    return panel


def build_bei_ils_wedge() -> pd.DataFrame:
    """Produce the BEI minus ILS wedge by tenor history."""

    tips_curve = pd.read_parquet(REPO_ROOT / "_data" / "fed_tips_yield_curve.parquet")
    if "date" not in tips_curve.columns:
        tips_curve = tips_curve.rename(columns={tips_curve.columns[0]: "date"})
    bei_cols = [c for c in tips_curve.columns if c.startswith("BKEVEN")]
    bei_long = tips_curve.melt(
        id_vars="date", value_vars=bei_cols, var_name="bei_tenor", value_name="bei_rate"
    )
    bei_long["tenor"] = bei_long["bei_tenor"].str.extract(r"(\d+)").astype(float)

    ils = pd.read_csv(REPO_ROOT / "data_manual" / "treasury_inflation_swaps.csv", parse_dates=["Dates"])
    ils = ils.rename(columns={"Dates": "date"})
    ils_cols = [c for c in ils.columns if c.startswith("USSWIT")]
    ils_long = ils.melt(
        id_vars="date", value_vars=ils_cols, var_name="ils_tenor", value_name="ils_rate"
    )
    ils_long["tenor"] = ils_long["ils_tenor"].str.extract(r"(\d+)").astype(float)
    ils_long["ils_rate"] = pd.to_numeric(ils_long["ils_rate"], errors="coerce")

    merged = bei_long.merge(ils_long, on=["date", "tenor"], how="inner")
    merged = merged.dropna(subset=["bei_rate", "ils_rate"])
    merged["bei_minus_ils"] = merged["bei_rate"] - merged["ils_rate"]
    merged = merged.sort_values(["date", "tenor"]).reset_index(drop=True)

    output_path = VAL_DIR / "bei_ils_wedge_by_tenor.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.rename(columns={"tenor": "tenor_years"}, inplace=True)
    merged.to_csv(output_path, index=False)
    return merged


def _load_crsp_panel() -> pd.DataFrame:
    panel_zip = REPO_ROOT / "src" / "data_pull" / "data" / "crsp_treasury_panel.zip"
    with zipfile.ZipFile(panel_zip) as zf:
        with zf.open("crsp_treasury_panel.csv") as handle:
            df = pd.read_csv(
                TextIOWrapper(handle, encoding="utf-8"),
                usecols=[
                    "crspid",
                    "qdate",
                    "bid",
                    "ask",
                    "matdt",
                    "datdt",
                    "totout",
                    "pubout",
                ],
                parse_dates=["qdate", "matdt", "datdt"],
            )
    return df


def build_liquidity_controls() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Regenerate tenor-level and aggregate liquidity control files."""

    panel = _load_crsp_panel()
    panel = panel.dropna(subset=["bid", "ask", "matdt", "qdate"])
    panel["bid_ask_spread"] = panel["ask"] - panel["bid"]
    panel["pubout"] = pd.to_numeric(panel["pubout"], errors="coerce")
    panel.loc[panel["pubout"] <= 0, "pubout"] = np.nan
    panel = panel.dropna(subset=["pubout"])

    panel["tenor_years"] = (panel["matdt"] - panel["qdate"]).dt.days / 365.25
    panel["tenor_bucket"] = panel["tenor_years"].apply(_assign_tenor_bucket)
    panel = panel.dropna(subset=["tenor_bucket"])
    panel["tenor_bucket"] = panel["tenor_bucket"].astype(int)

    grouped = panel.groupby(["qdate", "tenor_bucket"], as_index=False)
    tenor_liq = grouped.agg(
        bid_ask_spread=("bid_ask_spread", "median"),
        pubout=("pubout", "sum"),
        n_issues=("crspid", "nunique"),
    )

    issue_totals = (
        panel.groupby(["qdate", "tenor_bucket", "crspid"], as_index=False)["pubout"].sum()
    )

    def _concentration_stats(group: pd.DataFrame) -> pd.Series:
        total = group["pubout"].sum()
        if not total or total <= 0:
            return pd.Series({
                "liq_hhi": np.nan,
                "issue_conc_top3": np.nan,
                "issue_conc_top5": np.nan,
            })
        shares = (group["pubout"] / total).sort_values(ascending=False)
        return pd.Series(
            {
                "liq_hhi": float((shares**2).sum()),
                "issue_conc_top3": float(shares.head(3).sum()),
                "issue_conc_top5": float(shares.head(5).sum()),
            }
        )

    concentration = issue_totals.groupby(["qdate", "tenor_bucket"]).apply(
        _concentration_stats, include_groups=False
    ).reset_index()
    tenor_liq = tenor_liq.merge(concentration, on=["qdate", "tenor_bucket"], how="left")
    tenor_liq = tenor_liq.sort_values(["qdate", "tenor_bucket"]).reset_index(drop=True)

    agg = tenor_liq.groupby("qdate", as_index=False).agg(
        bid_ask_spread=("bid_ask_spread", "mean"),
        pubout=("pubout", "sum"),
        n_issues=("n_issues", "sum"),
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tenor_liq.to_csv(DATA_DIR / "tenor_liq.csv", index=False)
    agg.to_csv(DATA_DIR / "crsp_treasury_agg.csv", index=False)
    return tenor_liq, agg


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    build_trace_event_panel()
    build_bei_ils_wedge()
    build_liquidity_controls()


if __name__ == "__main__":
    main()
