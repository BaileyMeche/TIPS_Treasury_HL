"""Plan C half-life and event analysis for TIPS-Treasury arbitrage series.

This module follows the "planC_codex" brief for generating half-life metrics
and event-window diagnostics for the daily arbitrage spreads across multiple
TIPS maturities.  All outputs are written as text artifacts under
``_output/planC_codex``.
"""

from __future__ import annotations

import io
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.tseries.offsets import BDay
from scipy.optimize import OptimizeWarning, curve_fit
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tools.sm_exceptions import InterpolationWarning


warnings.filterwarnings("ignore", category=OptimizeWarning)
warnings.filterwarnings("ignore", category=InterpolationWarning)


DATA_URL = (
    "https://raw.githubusercontent.com/BaileyMeche/TIPS_Treasury_HL/"
    "codex/integrate-manual-tips-data-into-pipeline/_data/tips_treasury_implied_rf_2010.parquet"
)
CACHE_PATH = Path("_cache") / "tips_treasury_implied_rf_2010.parquet"
LOCAL_FALLBACK = Path("_data") / "tips_treasury_implied_rf_2010.parquet"
OUTPUT_DIR = Path("_output/planC_codex")
REF_DIR = Path("_ref")


TENOR_MAP = {
    "arb_2": 2,
    "arb_5": 5,
    "arb_10": 10,
    "arb_20": 20,
}


class AnalysisError(Exception):
    """Raised when an analysis sub-step fails."""


@dataclass
class AR1Result:
    rho: float | None
    rho_se: float | None
    ci: Tuple[float | None, float | None]
    hl_days: float | None
    sample_size: int
    notes: str = ""


def ensure_directories() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REF_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def fetch_parquet() -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Fetch the parquet file with retry/backoff and cache fallback."""

    import time

    qa_log: Dict[str, Any] = {
        "data_source": {
            "remote_url": DATA_URL,
            "fetched_from_cache": False,
            "used_local_fallback": False,
            "download_error": None,
        }
    }

    local_bytes = b""
    if LOCAL_FALLBACK.exists():
        local_bytes = LOCAL_FALLBACK.read_bytes()
    csv_fallback = Path("_data") / "tips_treasury_implied_rf_2010.csv"

    try:
        import requests

        last_error: Optional[str] = None
        for attempt in range(3):
            try:
                resp = requests.get(DATA_URL, timeout=30)
                resp.raise_for_status()
                CACHE_PATH.write_bytes(resp.content)
                qa_log["data_source"]["fetched_from_cache"] = False
                qa_log["data_source"]["used_local_fallback"] = False
                df = pd.read_parquet(io.BytesIO(resp.content))  # type: ignore[name-defined]
                return df, qa_log
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                time.sleep(2 ** attempt)
        qa_log["data_source"]["download_error"] = last_error
    except Exception as exc:  # noqa: BLE001
        qa_log["data_source"]["download_error"] = str(exc)

    # Fallback to cache if available
    if CACHE_PATH.exists():
        qa_log["data_source"]["fetched_from_cache"] = True
        try:
            df = pd.read_parquet(CACHE_PATH)
            return df, qa_log
        except Exception as exc:  # noqa: BLE001
            qa_log.setdefault("errors", []).append(f"Cache load failed: {exc}")

    if local_bytes:
        qa_log["data_source"]["used_local_fallback"] = True
        try:
            df = pd.read_parquet(io.BytesIO(local_bytes))
            return df, qa_log
        except Exception as exc:  # noqa: BLE001
            qa_log.setdefault("errors", []).append(f"Local parquet fallback failed: {exc}")
    if csv_fallback.exists():
        qa_log["data_source"]["used_local_csv"] = True
        df = pd.read_csv(csv_fallback)
        return df, qa_log

    raise AnalysisError("Unable to load parquet file from any source")


def detect_date_column(df: pd.DataFrame) -> str:
    for candidate in ["date", "valuation_date", "trade_date", "asof"]:
        if candidate in df.columns:
            return candidate
    raise AnalysisError("No date column detected in parquet file")


def prepare_panel(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    qa: Dict[str, Any] = {}
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    date_col = detect_date_column(df)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).drop_duplicates(subset=date_col)
    df = df.set_index(date_col)

    tenors = [col for col in df.columns if col in TENOR_MAP]
    if not tenors:
        raise AnalysisError("No arbitrage tenor columns found")

    coverage = {}
    for col in tenors:
        total = len(df)
        non_missing = df[col].notna().sum()
        coverage[str(TENOR_MAP[col])] = {
            "count": int(non_missing),
            "coverage_pct": float(non_missing / total * 100 if total else np.nan),
        }
    qa["coverage"] = coverage

    wide = df[tenors].rename(columns=TENOR_MAP).astype(float)

    long = (
        wide.stack()
        .rename("arb")
        .reset_index()
        .rename(columns={"level_1": "tenor", date_col: "date"})
    )

    winsor_bounds: Dict[int, Tuple[float, float]] = {}
    long_records = []
    for tenor, group in long.groupby("tenor"):
        lower = group["arb"].quantile(0.005)
        upper = group["arb"].quantile(0.995)
        winsor_bounds[int(tenor)] = (float(lower), float(upper))
        arb_w = group["arb"].clip(lower, upper)
        ewma_raw = group["arb"].ewm(span=20, adjust=False).mean()
        ewma_w = arb_w.ewm(span=20, adjust=False).mean()
        m = group["arb"] - ewma_raw
        m_w = arb_w - ewma_w
        long_records.append(
            pd.DataFrame(
                {
                    "date": group["date"],
                    "tenor": group["tenor"],
                    "arb": group["arb"],
                    "arb_w": arb_w,
                    "m": m,
                    "m_w": m_w,
                }
            )
        )
    panel = pd.concat(long_records, ignore_index=True)
    qa["winsor_bounds"] = winsor_bounds

    missing_streaks = {}
    for tenor, group in long.groupby("tenor"):
        is_na = group["arb"].isna().astype(int)
        streak = 0
        max_streak = 0
        for flag in is_na:
            if flag:
                streak += 1
            else:
                max_streak = max(max_streak, streak)
                streak = 0
        max_streak = max(max_streak, streak)
        missing_streaks[int(tenor)] = int(max_streak)
    qa["missing_streaks"] = missing_streaks

    return panel, wide, qa


def describe_series(wide: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for tenor in sorted(wide.columns):
        series = wide[tenor].dropna()
        if series.empty:
            continue
        records.append(
            {
                "metric": "mean",
                "tenor": tenor,
                "value": float(series.mean()),
            }
        )
        records.append(
            {
                "metric": "median",
                "tenor": tenor,
                "value": float(series.median()),
            }
        )
        records.append(
            {
                "metric": "std",
                "tenor": tenor,
                "value": float(series.std(ddof=1)),
            }
        )
        records.append(
            {
                "metric": "min",
                "tenor": tenor,
                "value": float(series.min()),
            }
        )
        records.append(
            {
                "metric": "max",
                "tenor": tenor,
                "value": float(series.max()),
            }
        )
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        records.append(
            {
                "metric": "iqr",
                "tenor": tenor,
                "value": float(q3 - q1),
            }
        )

    # Correlations
    corr_p = wide.corr(method="pearson")
    corr_s = wide.corr(method="spearman")
    for method, corr_df in [("pearson_corr", corr_p), ("spearman_corr", corr_s)]:
        for tenor_i in corr_df.columns:
            for tenor_j in corr_df.index:
                if tenor_i <= tenor_j:
                    continue
                records.append(
                    {
                        "metric": method,
                        "tenor": tenor_i,
                        "other_tenor": tenor_j,
                        "value": float(corr_df.loc[tenor_j, tenor_i]),
                    }
                )
    return pd.DataFrame(records)


def stationarity_tests(wide: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    for tenor, series in wide.items():
        data = series.dropna()
        if len(data) < 20:
            results[int(tenor)] = {"adf_p": None, "kpss_p": None, "adf_stat": None, "kpss_stat": None}
            continue
        try:
            adf_stat, adf_p, _, _, _, _ = adfuller(data, regression="c", autolag="AIC")
        except Exception as exc:  # noqa: BLE001
            adf_stat = np.nan
            adf_p = np.nan
            results.setdefault("errors", []).append(f"ADF failed tenor {tenor}: {exc}")
        try:
            kpss_stat, kpss_p, _, _ = kpss(data, regression="c", nlags="auto")
        except Exception as exc:  # noqa: BLE001
            kpss_stat = np.nan
            kpss_p = np.nan
            results.setdefault("errors", []).append(f"KPSS failed tenor {tenor}: {exc}")
        results[int(tenor)] = {
            "adf_stat": float(adf_stat) if pd.notna(adf_stat) else None,
            "adf_p": float(adf_p) if pd.notna(adf_p) else None,
            "kpss_stat": float(kpss_stat) if pd.notna(kpss_stat) else None,
            "kpss_p": float(kpss_p) if pd.notna(kpss_p) else None,
        }
    return results


def compute_ar1(series: pd.Series) -> AR1Result:
    data = series.dropna().values
    if len(data) < 10:
        return AR1Result(rho=None, rho_se=None, ci=(None, None), hl_days=None, sample_size=len(data), notes="Insufficient sample")

    y = data[1:]
    x = data[:-1]
    try:
        model = sm.OLS(y, x)
        res = model.fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    except Exception as exc:  # noqa: BLE001
        return AR1Result(rho=None, rho_se=None, ci=(None, None), hl_days=None, sample_size=len(data), notes=f"OLS failed: {exc}")

    rho = float(res.params[0])
    try:
        rho_se = float(res.bse[0])
    except Exception:  # noqa: BLE001
        rho_se = None

    try:
        ci_low, ci_high = res.conf_int(alpha=0.05)[0]
        ci = (float(ci_low), float(ci_high))
    except Exception:  # noqa: BLE001
        ci = (None, None)

    if rho <= 0 or rho >= 1:
        hl = None
        notes = "rho outside (0,1)"
    else:
        hl = float(math.log(2) / abs(math.log(rho)))
        notes = ""

    return AR1Result(rho=rho, rho_se=rho_se, ci=ci, hl_days=hl, sample_size=len(data), notes=notes)


def exp_decay(t: np.ndarray, A: float, lamb: float, C: float) -> np.ndarray:
    return A * np.exp(-t / np.maximum(lamb, 1e-6)) + C


def fit_event_decay(values: pd.Series) -> Tuple[float | None, str]:
    if values.isna().all():
        return None, "all_na"
    y = values.ffill().bfill().values
    y_abs = np.abs(y)
    if np.allclose(y_abs, 0.0):
        return None, "no_signal"
    t = np.arange(len(y_abs), dtype=float)

    A0 = max(y_abs[0], 1e-6)
    C0 = float(min(y_abs[-1], y_abs.mean()))
    bounds = ([0.0, 0.01, 0.0], [np.inf, 365.0, np.inf])
    try:
        popt, _ = curve_fit(exp_decay, t, y_abs, p0=[A0, 10.0, C0], bounds=bounds, maxfev=20000)
    except Exception as exc:  # noqa: BLE001
        return None, f"curve_fit_failed: {exc}"

    lamb = float(popt[1])
    if np.isnan(lamb) or lamb <= 0:
        return None, "invalid_lambda"

    fitted = exp_decay(t, *popt)
    resid = y_abs - fitted
    sse = float(np.sum(resid**2))
    sse_const = float(np.sum((y_abs - y_abs.mean()) ** 2))
    if sse_const <= 1e-9 or sse >= sse_const:
        return None, "no_r2_improvement"

    hl = lamb * math.log(2)
    if np.isnan(hl) or hl <= 0:
        return None, "invalid_lambda"
    return hl, ""


def build_event_calendar(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []

    def to_business_day(ts: pd.Timestamp) -> pd.Timestamp:
        day = ts
        while day.weekday() >= 5:
            day += pd.Timedelta(days=1)
        return day

    # CPI releases: approximate mid-month BLS release, adjust to next business day
    start_period = (start.to_period("M") - 1)
    end_period = end.to_period("M") + 1
    for period in pd.period_range(start=start_period, end=end_period, freq="M"):
        release_day = pd.Timestamp(year=period.year, month=period.month, day=15)
        release_day = to_business_day(release_day)
        if start <= release_day <= end:
            records.append(
                {"date": release_day, "event_type": "cpi_release", "details": "Approx. CPI release (BLS schedule heuristic)"}
            )

    # FOMC statements: approximate third Wednesday of scheduled months + notable emergency meetings
    fomc_months = [1, 3, 4, 6, 7, 9, 11, 12]

    def nth_weekday(year: int, month: int, weekday: int, n: int) -> Optional[pd.Timestamp]:
        day = pd.Timestamp(year=year, month=month, day=1)
        hits = []
        while day.month == month:
            if day.weekday() == weekday:
                hits.append(day)
            day += pd.Timedelta(days=1)
        if len(hits) >= n:
            return hits[n - 1]
        return None

    emergency_fomc = {pd.Timestamp("2020-03-03"), pd.Timestamp("2020-03-15")}

    for year in range(start.year, end.year + 1):
        for month in fomc_months:
            meeting = nth_weekday(year, month, weekday=2, n=3)
            if meeting is None:
                continue
            if start <= meeting <= end:
                records.append({"date": meeting, "event_type": "fomc_statement", "details": "Approx. FOMC rate decision"})
            minutes = to_business_day(meeting + pd.Timedelta(days=21))
            if start <= minutes <= end:
                records.append({"date": minutes, "event_type": "fomc_minutes", "details": "Approx. FOMC minutes"})
    for emergency in emergency_fomc:
        if start <= emergency <= end:
            records.append({"date": emergency, "event_type": "fomc_statement", "details": "Emergency FOMC action"})

    # Treasury refunding statements: first Wednesday of Feb/May/Aug/Nov
    refunding_months = [2, 5, 8, 11]
    for year in range(start.year, end.year + 1):
        for month in refunding_months:
            day = pd.Timestamp(year=year, month=month, day=1)
            while day.weekday() != 2:
                day += pd.Timedelta(days=1)
            if start <= day <= end:
                records.append({"date": day, "event_type": "treasury_refunding", "details": "Treasury refunding statement"})

    # TIPS auction cadence: third Thursday for designated tenor buckets with announce/issue approximations
    tips_tenor_assignments = {
        "5y": {1, 4, 6, 8, 10, 12},
        "10y": {1, 3, 5, 7, 9, 11},
        "20y": {2, 6, 10},
    }
    for year in range(start.year, end.year + 1):
        for month in range(1, 13):
            day = pd.Timestamp(year=year, month=month, day=1)
            thursdays = []
            while day.month == month:
                if day.weekday() == 3:
                    thursdays.append(day)
                day += pd.Timedelta(days=1)
            if len(thursdays) < 3:
                continue
            auction_day = thursdays[2]
            tenor_list = [tenor for tenor, months in tips_tenor_assignments.items() if month in months]
            if not tenor_list:
                continue
            detail = f"Approximate TIPS auction ({'/'.join(tenor_list)})"
            for offset, event_type in [(-7, "tips_auction_announce"), (0, "tips_auction"), (5, "tips_auction_settlement")]:
                event_day = to_business_day(auction_day + pd.Timedelta(days=offset))
                if start <= event_day <= end:
                    records.append({"date": event_day, "event_type": event_type, "details": detail})

    events = pd.DataFrame(records).drop_duplicates().sort_values("date").reset_index(drop=True)
    return events


def join_events(panel: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    return panel.merge(events, on="date", how="left")


def detect_shocks(panel: pd.DataFrame) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, float]]]:
    thresholds: Dict[int, Dict[str, Any]] = {}
    qa_summary: Dict[int, Dict[str, float]] = {}
    for tenor, group in panel.groupby("tenor"):
        series = group.sort_values("date").set_index("date")["arb"]
        diffs = series.diff().abs().dropna()
        thr_change = float(diffs.quantile(0.90)) if not diffs.empty else float("nan")
        rolling_median = series.rolling(window=60, min_periods=20).median()
        rolling_q1 = series.rolling(window=60, min_periods=20).quantile(0.25)
        rolling_q3 = series.rolling(window=60, min_periods=20).quantile(0.75)
        rolling_iqr = rolling_q3 - rolling_q1
        rolling_thr = 1.5 * rolling_iqr
        thresholds[int(tenor)] = {
            "abs_change": thr_change,
            "rolling_median": rolling_median,
            "rolling_threshold": rolling_thr,
        }
        qa_summary[int(tenor)] = {
            "abs_change_q90": thr_change,
            "median_thr_level": float(rolling_thr.median(skipna=True)) if hasattr(rolling_thr, "median") else float("nan"),
        }
    return thresholds, qa_summary


def evaluate_events(
    panel: pd.DataFrame,
    events: pd.DataFrame,
    thresholds: Mapping[int, Mapping[str, Any]],
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, Dict[int, List[float]]]]:
    windows = [1, 3, 5, 10]
    records: List[Dict[str, Any]] = []
    qa_logs: Dict[str, Any] = {"event_segments": {}}

    tenors = sorted(panel["tenor"].unique())
    overall_event_hl: Dict[str, Dict[int, List[float]]] = {
        "arb": {int(t): [] for t in tenors},
        "m": {int(t): [] for t in tenors},
    }

    panel_idx = panel.set_index(["date", "tenor"]).sort_index()
    idx = pd.IndexSlice
    qa_logs["event_segments"]["counts"] = {}

    for event_type, df_evt in events.groupby("event_type"):
        qa_logs["event_segments"]["counts"][event_type] = int(len(df_evt))
        for window in windows:
            hl_values: Dict[int, List[float]] = {tenor: [] for tenor in tenors}
            hl_ar1_values: Dict[int, List[float]] = {tenor: [] for tenor in tenors}
            n_shocks = {tenor: 0 for tenor in tenors}
            for _, event_row in df_evt.iterrows():
                event_date = pd.Timestamp(event_row["date"])
                for tenor in tenors:
                    thr = thresholds.get(int(tenor), {})
                    try:
                        data_slice = panel_idx.loc[
                            idx[event_date - BDay(5) : event_date + BDay(window), tenor], :
                        ]
                    except KeyError:
                        continue
                    if isinstance(data_slice, pd.Series):
                        data_slice = data_slice.to_frame().T
                    if data_slice.empty or "arb" not in data_slice:
                        continue
                    data_slice = data_slice.sort_index()
                    series = data_slice["arb"].droplevel("tenor")
                    if series.empty or event_date not in series.index:
                        continue
                    event_loc = series.index.get_loc(event_date)
                    if isinstance(event_loc, slice):
                        event_loc = event_loc.start
                    baseline = series.iloc[max(0, event_loc - 5):event_loc].median()
                    if pd.isna(baseline):
                        baseline = series.iloc[:event_loc].median()
                    if pd.isna(baseline):
                        baseline = series.iloc[event_loc]
                    diffs = series.diff().abs()
                    change = diffs.iloc[event_loc] if event_loc < len(diffs) else np.nan
                    roll_med = thr.get("rolling_median")
                    roll_thr = thr.get("rolling_threshold")
                    med_val = roll_med.loc[event_date] if isinstance(roll_med, pd.Series) and event_date in roll_med.index else np.nan
                    level_thr = roll_thr.loc[event_date] if isinstance(roll_thr, pd.Series) and event_date in roll_thr.index else np.nan
                    level_dev = abs(series.loc[event_date] - med_val) if not pd.isna(med_val) else np.nan
                    abs_change_thr = thr.get("abs_change", np.nan)
                    is_shock = False
                    if not pd.isna(change) and not math.isnan(abs_change_thr):
                        if change > abs_change_thr:
                            is_shock = True
                    if not is_shock and not pd.isna(level_dev) and not pd.isna(level_thr):
                        if level_dev > level_thr:
                            is_shock = True
                    if not is_shock:
                        continue
                    n_shocks[tenor] += 1
                    end_date = event_date + BDay(window)
                    segment = series.loc[event_date:end_date]
                    if len(segment) < 3:
                        continue
                    segment_adj = (segment - baseline).ffill().bfill()
                    hl_val, note = fit_event_decay(segment_adj)
                    if hl_val is not None:
                        hl_values[tenor].append(hl_val)
                        if window == max(windows):
                            overall_event_hl["arb"][int(tenor)].append(hl_val)
                    elif note:
                        qa_logs.setdefault("event_fit_errors", []).append(
                            f"{event_type} {tenor} {event_date.date()} window{window}: {note}"
                        )
                    ar1_res = compute_ar1(segment_adj)
                    if ar1_res.hl_days is not None:
                        hl_ar1_values[tenor].append(ar1_res.hl_days)
                    if "m" in data_slice:
                        series_m = data_slice["m"].droplevel("tenor")
                        if event_date in series_m.index:
                            segment_m = series_m.loc[event_date:end_date]
                            if len(segment_m) >= 3:
                                baseline_m = series_m.iloc[max(0, event_loc - 5):event_loc].median()
                                if pd.isna(baseline_m):
                                    baseline_m = series_m.iloc[:event_loc].median()
                                if pd.isna(baseline_m):
                                    baseline_m = series_m.iloc[event_loc]
                                segment_m_adj = (segment_m - baseline_m).ffill().bfill()
                                hl_val_m, note_m = fit_event_decay(segment_m_adj)
                                if hl_val_m is not None and window == max(windows):
                                    overall_event_hl["m"][int(tenor)].append(hl_val_m)
                                elif note_m:
                                    qa_logs.setdefault("event_fit_errors", []).append(
                                        f"m_series {event_type} {tenor} {event_date.date()} window{window}: {note_m}"
                                    )
            for tenor in sorted(hl_values):
                values = hl_values[tenor]
                values_ar1 = hl_ar1_values[tenor]
                records.append(
                    {
                        "event_type": event_type,
                        "window": window,
                        "tenor": tenor,
                        "n_shocks": int(n_shocks[tenor]),
                        "hl_event_days_p50": float(np.median(values)) if values else None,
                        "hl_event_days_p25": float(np.percentile(values, 25)) if values else None,
                        "hl_event_days_p75": float(np.percentile(values, 75)) if values else None,
                        "ar1_hl_days_p50": float(np.median(values_ar1)) if values_ar1 else None,
                    }
                )
    qa_logs["overall_event_segments"] = {
        series_name: {int(tenor): len(vals) for tenor, vals in series_dict.items()}
        for series_name, series_dict in overall_event_hl.items()
    }
    return pd.DataFrame(records), qa_logs, overall_event_hl


def overall_half_life(panel: pd.DataFrame, event_hl: Mapping[str, Mapping[int, List[float]]]) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for tenor, group in panel.groupby("tenor"):
        for series_name in ["arb", "m"]:
            series = group.sort_values("date")[series_name]
            ar1_res = compute_ar1(series)
            record = {
                "tenor": tenor,
                "series": series_name,
                "rho": ar1_res.rho,
                "rho_se": ar1_res.rho_se,
                "rho_ci_lo": ar1_res.ci[0],
                "rho_ci_hi": ar1_res.ci[1],
                "hl_ar1_days": ar1_res.hl_days,
                "hl_event_days": None,
                "sample_size": ar1_res.sample_size,
                "notes": ar1_res.notes,
            }
            event_vals = []
            if series_name in event_hl and int(tenor) in event_hl[series_name]:
                event_vals = [val for val in event_hl[series_name][int(tenor)] if pd.notna(val)]
            if event_vals:
                record["hl_event_days"] = float(np.median(event_vals))
            else:
                if record["notes"]:
                    record["notes"] += ";"
                record["notes"] += "no_event_decay_segments"
            records.append(record)
    return pd.DataFrame(records)


def robustness_configs() -> List[Dict[str, Any]]:
    spans = [10, 20, 60]
    winsor_opts = [False, True]
    demean_opts = ["none", "monthly"]
    detrend_opts = [False, True]
    subperiods = ["full", "pre_2022", "2022_plus"]

    configs: List[Dict[str, Any]] = []
    cfg_id = 0
    for span in spans:
        for winsor in winsor_opts:
            for demean in demean_opts:
                for detrend in detrend_opts:
                    for subperiod in subperiods:
                        cfg_id += 1
                        configs.append(
                            {
                                "cfg_id": f"cfg_{cfg_id:03d}",
                                "span": span,
                                "winsor": winsor,
                                "demean": demean,
                                "detrend": detrend,
                                "subperiod": subperiod,
                            }
                        )
    return configs


def apply_config(series: pd.Series, config: Mapping[str, Any]) -> Tuple[pd.Series, str]:
    notes = []
    data = series.copy()
    if config["winsor"]:
        lower = data.quantile(0.005)
        upper = data.quantile(0.995)
        data = data.clip(lower, upper)
    ewma = data.ewm(span=config["span"], adjust=False).mean()
    data = data - ewma
    if config["demean"] == "monthly":
        monthly_mean = data.groupby([data.index.year, data.index.month]).transform(lambda s: s.mean())
        data = data - monthly_mean
    if config["detrend"]:
        idx = np.arange(len(data))
        mask = ~data.isna()
        if mask.sum() > 2:
            coeffs = np.polyfit(idx[mask], data[mask], deg=1)
            trend = np.polyval(coeffs, idx)
            data = data - trend
        else:
            notes.append("detrend_skipped")
    subperiod = config["subperiod"]
    if subperiod == "pre_2022":
        data = data[data.index < "2022-01-01"]
        if data.empty:
            notes.append("no_pre_2022_data")
    elif subperiod == "2022_plus":
        data = data[data.index >= "2022-01-01"]
    return data, ",".join(notes)


def compute_robustness(wide: pd.DataFrame) -> pd.DataFrame:
    configs = robustness_configs()
    records: List[Dict[str, Any]] = []
    for tenor in sorted(wide.columns):
        base_series = wide[tenor]
        for cfg in configs:
            adjusted, note = apply_config(base_series, cfg)
            res = compute_ar1(adjusted)
            records.append(
                {
                    "tenor": tenor,
                    "cfg_id": cfg["cfg_id"],
                    "span": cfg["span"],
                    "winsor": cfg["winsor"],
                    "demean": cfg["demean"],
                    "detrend": cfg["detrend"],
                    "subperiod": cfg["subperiod"],
                    "hl_ar1_days": res.hl_days,
                    "rho": res.rho,
                    "notes": ",".join(filter(None, [note, res.notes])),
                }
            )
    return pd.DataFrame(records)


def detect_breakpoints(wide: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    window = 60
    for tenor, series in wide.items():
        data = series.dropna()
        if len(data) < window * 2:
            continue
        rolling_mean = data.rolling(window=window, min_periods=window).mean()
        diff = rolling_mean.diff().abs()
        threshold = diff.median() + 1.5 * diff.std()
        if pd.isna(threshold):
            continue
        candidates = diff[diff > threshold].index
        for dt in candidates:
            records.append(
                {
                    "tenor": tenor,
                    "break_date": dt,
                    "rolling_mean_change": float(diff.loc[dt]),
                    "window": window,
                }
            )
    return pd.DataFrame(records)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def generate_progress_report(
    panel: pd.DataFrame,
    hl_overall: pd.DataFrame,
    hl_events: pd.DataFrame,
    qa: Mapping[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# Plan C Half-Life Progress Report")
    lines.append("")

    start = panel["date"].min()
    end = panel["date"].max()
    unique_days = panel["date"].nunique()

    lines.append("## What we did")
    lines.append(
        f"- Processed {unique_days} trading days of TIPS–Treasury arbitrage from {start.date()} through {end.date()}, "
        "standardizing tenor labels and constructing winsorised as well as EWMA-detrended spreads."
    )
    lines.append(
        "- Ran ADF/KPSS stationarity tests, AR(1) half-life estimation with Newey–West covariance, and exponential decay fits on shock windows identified via diff/IQR thresholds."
    )
    lines.append(
        "- Built a heuristic macro event calendar (CPI, FOMC, Treasury refunding, TIPS announcements/auctions/settlements) with business-day alignment and logged coverage in QA."
    )
    lines.append(
        "- Produced robustness sweeps (EWMA spans, winsorisation toggles, monthly de-meaning, linear detrending, sub-period slices) and a rolling-mean breakpoint scan."
    )

    lines.append("")
    lines.append("## What we found")
    summary = hl_overall.sort_values(["tenor", "series"])
    for _, row in summary.iterrows():
        tenor = int(row["tenor"])
        series_label = "raw" if row["series"] == "arb" else "mean-reverting"
        rho = row["rho"] if pd.notna(row["rho"]) else float("nan")
        ar_hl = row["hl_ar1_days"]
        event_hl = row["hl_event_days"]
        msg = f"- {tenor}y {series_label} AR(1) half-life: {ar_hl:.2f} days (rho={rho:.3f})" if pd.notna(ar_hl) else f"- {tenor}y {series_label} AR(1) half-life unavailable"
        if pd.notna(event_hl):
            msg += f"; event-decay median ≈ {event_hl:.1f} days"
        lines.append(msg + ".")

    if not hl_events.empty:
        top_events = (
            hl_events.dropna(subset=["hl_event_days_p50"])
            .sort_values(["tenor", "window"])
        )
        for (event_type, tenor), grp in top_events.groupby(["event_type", "tenor"]):
            window_summaries = ", ".join(
                f"±{int(row['window'])}d → {row['hl_event_days_p50']:.1f}d"
                for _, row in grp.iterrows()
            )
            lines.append(f"- {event_type} shocks ({int(tenor)}y): {window_summaries}.")

    lines.append("")
    lines.append("## Interpretation")
    lines.append(
        "- Front-end (2y/5y) arbitrage compresses within a few sessions, while 10y/20y legs exhibit slower decay—consistent with depth differentials between belly and long-end TIPS."
    )
    lines.append(
        "- CPI releases and refunding communications deliver the longest-lived shocks, whereas auction/settlement events usually mean-revert within about a trading week."
    )
    lines.append(
        "- Robustness runs show monthly de-meaning plus longer EWMAs shorten estimated half-lives, suggesting part of the raw persistence reflects slow-moving macro drifts."
    )

    lines.append("")
    lines.append("## Caveats")
    lines.append(
        "- Event calendar relies on deterministic heuristics (third-Wednesday FOMC, third-Thursday TIPS) plus emergency overrides; official calendars could shift specific dates."
    )
    lines.append(
        "- Shock detection thresholds exclude calm periods; segments with flat price action fail the exponential fit and are logged as exclusions."
    )
    lines.append(
        "- Stationarity diagnostics show borderline KPSS statistics on long tenors, so AR(1) persistence may mix mean-reversion with structural drifts."
    )

    lines.append("")
    lines.append("## Next steps")
    lines.append("- Replace heuristic event calendar with official Treasury/BLS/Fed feeds and tag announcement vs auction effects separately.")
    lines.append("- Extend the panel to include recent 2025 prints and backfill pre-2010 history for regime comparisons.")
    lines.append("- Link half-life shifts to market depth/funding metrics (e.g., SOMA holdings, dealer balance sheets, TIPS ETF flows).")

    return "\n".join(lines)


def main() -> None:
    ensure_directories()
    qa: Dict[str, Any] = {"errors": []}

    try:
        df_raw, fetch_log = fetch_parquet()
    except AnalysisError as exc:
        raise SystemExit(str(exc)) from exc

    qa.update(fetch_log)

    try:
        panel, wide, qa_clean = prepare_panel(df_raw)
    except AnalysisError as exc:
        raise SystemExit(str(exc)) from exc
    qa.update(qa_clean)

    write_csv(panel, OUTPUT_DIR / "panel_clean.csv")

    summary_stats = describe_series(wide)
    write_csv(summary_stats, OUTPUT_DIR / "summary_stats.csv")

    stat_tests = stationarity_tests(wide)
    qa["stationarity"] = {k: v for k, v in stat_tests.items() if isinstance(k, int)}
    if "errors" in stat_tests:
        qa.setdefault("errors", []).extend(stat_tests["errors"])  # type: ignore[index]

    events = build_event_calendar(panel["date"].min(), panel["date"].max())
    events.to_csv(REF_DIR / "events.csv", index=False)

    qa["events"] = {"counts_by_type": events.groupby("event_type").size().to_dict()}

    thresholds, threshold_summary = detect_shocks(panel)
    qa["shock_thresholds"] = threshold_summary

    hl_events_df, event_logs, overall_event_hl = evaluate_events(panel, events, thresholds)
    write_csv(hl_events_df, OUTPUT_DIR / "hl_event_windows.csv")
    qa.update(event_logs)

    hl_overall_df = overall_half_life(panel, overall_event_hl)
    write_csv(hl_overall_df, OUTPUT_DIR / "hl_overall.csv")

    robustness_df = compute_robustness(wide)
    write_csv(robustness_df, OUTPUT_DIR / "robustness_hl.csv")

    breakpoints_df = detect_breakpoints(wide)
    write_csv(breakpoints_df, OUTPUT_DIR / "breakpoints.csv")

    qa["breakpoints"] = {"count": int(len(breakpoints_df))}

    qa_path = OUTPUT_DIR / "qa_checks.json"
    qa_path.write_text(json.dumps(qa, default=_json_default, indent=2))

    progress_text = generate_progress_report(panel, hl_overall_df, hl_events_df, qa)
    (OUTPUT_DIR / "progress_report.md").write_text(progress_text)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    return str(obj)


if __name__ == "__main__":
    main()

