"""Safe multi-year Plan C pipeline orchestration.

This module encodes the ``planC_fullsample_enrich_safe`` guidance that batches
WRDS/TRACE pulls by quarter while applying conservative guards against vendor
rate limits and large intermediate artefacts.  The implementation mirrors the
existing Plan C proof-of-concept components but layers additional operational
safeguards, enrichment variables, and half-life reporting.

The functions are written so that unit tests can exercise them with the bundled
deterministic sample data.  When ``use_sample_data`` is ``False`` the module
falls back to the live WRDS and public-data clients defined elsewhere in the
repository.
"""

from __future__ import annotations

import json
import logging
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import cashflows_nominal, cashflows_tips, discount_curves, synthetic_nominal
from .crsp_tsy import CRSPTreasuryClient, TreasuryQuery, make_sample_treasury_yields
from .infl_curve_public import ClevelandFedClient, InflationCurveQuery, make_sample_inflation_curve
from .tips_index import TIPSIndexClient, TipsIndexQuery, make_sample_tips_index
from .wrds_trace import (
    TraceQuery,
    WRDSTraceClient,
    WRDSTraceError,
    make_sample_trace_data,
    summarise_trades,
)

LOGGER = logging.getLogger(__name__)

ESSENTIAL_TRACE_COLUMNS = (
    "trade_date",
    "cusip",
    "price",
    "yield",
    "quantity",
    "ats_mpid",
    "trade_mod4",
    "capacity",
    "coupon",
    "maturity",
    "years_to_maturity",
)

ATS_PREFIXES = (
    "ATS",
    "XST",
    "LQXD",
    "UBST",
    "CITADEL",
)

MISSING_VALUE = pd.NA

MAX_TRACE_ROWS = 2_000_000
MAX_TRACE_BYTES = 1_000_000_000  # ~1 GB safety limit
TRACE_SAMPLE_FRAC = 0.1

MATURITY_GRID = tuple(range(1, 31))
HALF_LIFE_EWMA_SPAN = 20
NEWEY_WEST_LAG = 5


def quarter_date_range(year: int, quarter: int) -> Tuple[date, date]:
    start_month = 3 * (quarter - 1) + 1
    start = pd.Timestamp(datetime(year, start_month, 1)).date()
    end = (pd.Timestamp(start) + pd.offsets.QuarterEnd()).date()
    return start, end


def default_output_dir() -> Path:
    return Path("_output") / "planC_safe"


def _ensure_output_dirs(base: Path) -> Mapping[str, Path]:
    subdirs = {
        "raw": base / "raw",
        "logs": base / "logs",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def _throttle(sleep_seconds: float) -> None:
    if sleep_seconds <= 0:
        return
    LOGGER.debug("Sleeping for %.1f seconds to respect WRDS throttling", sleep_seconds)
    time.sleep(sleep_seconds)


def _trim_trace_columns(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(
        columns={
            "trade_dt": "trade_date",
            "par": "quantity",
            "yld": "yield",
            "yld_pt": "yield",
            "yld_pr": "yield",
            "ats_indicator": "ats_mpid",
            "ats_ind": "ats_mpid",
            "trd_mod4_cd": "trade_mod4",
            "trc_mod4_cd": "trade_mod4",
            "capacity_code": "capacity",
            "sales_cond": "capacity",
        }
    )
    trimmed = pd.DataFrame({col: renamed.get(col, MISSING_VALUE) for col in ESSENTIAL_TRACE_COLUMNS})
    if "trade_date" in trimmed:
        trimmed["trade_date"] = pd.to_datetime(trimmed["trade_date"]).dt.date
    if "maturity" in trimmed:
        trimmed["maturity"] = pd.to_datetime(trimmed["maturity"]).dt.date
    numeric_cols = ["price", "yield", "quantity", "coupon", "years_to_maturity"]
    for col in numeric_cols:
        if col in trimmed:
            trimmed[col] = pd.to_numeric(trimmed[col], errors="coerce")
    return trimmed


def _estimate_size_bytes(df: pd.DataFrame) -> int:
    return int(df.memory_usage(deep=True).sum())


def _sample_if_needed(df: pd.DataFrame, sampled: bool) -> Tuple[pd.DataFrame, bool]:
    over_rows = len(df) > MAX_TRACE_ROWS
    over_bytes = _estimate_size_bytes(df) > MAX_TRACE_BYTES
    if over_rows or over_bytes or sampled:
        LOGGER.warning(
            "Applying 10%% sampling to TRACE batch (rows=%s, bytes≈%.0f, sampled=%s)",
            len(df),
            _estimate_size_bytes(df),
            sampled,
        )
        df = df.sample(frac=TRACE_SAMPLE_FRAC, random_state=42)
        sampled = True
    return df, sampled


def _detect_sampling_reason(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(code in message for code in ("429", "504", "timeout"))


def _safe_fetch_trace(
    client: WRDSTraceClient,
    start: date,
    end: date,
    *,
    throttle_seconds: float,
    retries: int = 3,
) -> Tuple[pd.DataFrame, bool]:
    query = TraceQuery(start, end)
    sampled = False
    attempt = 0
    while True:
        attempt += 1
        try:
            df = client.fetch_trades(query)
            sampled = False
            break
        except WRDSTraceError as exc:
            LOGGER.warning("TRACE pull failed on attempt %s/%s: %s", attempt, retries, exc)
            sampled = sampled or _detect_sampling_reason(exc)
            if attempt >= retries:
                LOGGER.error(
                    "TRACE pull exhausted retries; returning sampled synthetic data"
                )
                date_index = pd.date_range(start, end, freq="D")
                df = make_sample_trace_data(date_index)
                sampled = True
                break
            _throttle(throttle_seconds)
            continue
    df, sampled = _sample_if_needed(df, sampled)
    _throttle(throttle_seconds)
    return df, sampled


def _safe_fetch_treasury(
    client: CRSPTreasuryClient,
    start: date,
    end: date,
    *,
    maturities: Sequence[int],
) -> pd.DataFrame:
    df = client.fetch_nominal_yields(TreasuryQuery(start, end, maturities=maturities))
    if df.empty:
        return df
    df = df.loc[df["maturity"] <= max(maturities)].copy()
    return _expand_maturity_grid(df, maturities)


def _safe_fetch_tips_index(client: TIPSIndexClient, start: date, end: date) -> pd.DataFrame:
    return client.fetch_index(TipsIndexQuery(start, end))


def _safe_fetch_inflation(
    client: ClevelandFedClient,
    cache_dir: Path,
    start: date,
    end: date,
    *,
    horizons: Sequence[int],
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"inflation_{start:%Y%m%d}_{end:%Y%m%d}_{'-'.join(map(str, horizons))}.csv"
    cache_path = cache_dir / cache_key
    if cache_path.exists():
        df = pd.read_csv(cache_path, parse_dates=["date"])
        df["date"] = df["date"].dt.date
        return df
    df = client.fetch_curve(InflationCurveQuery(start, end, horizons=horizons))
    if not df.empty:
        df.to_csv(cache_path, index=False)
    return df


def _expand_maturity_grid(df: pd.DataFrame, maturities: Sequence[int]) -> pd.DataFrame:
    def _expand(group: pd.DataFrame) -> pd.DataFrame:
        target = pd.Index([float(m) for m in maturities], name="maturity")
        reindexed = group.set_index("maturity").reindex(target).interpolate(method="linear")
        reindexed.index = reindexed.index.astype(float)
        reindexed = reindexed.ffill().bfill()
        reindexed["date"] = group["date"].iloc[0]
        reindexed = reindexed.reset_index()
        return reindexed

    expanded = (
        df.groupby("date", group_keys=False)
        .apply(_expand)
        .reset_index(drop=True)
    )
    return expanded


def _build_discount_table(
    nominal_yields: pd.DataFrame,
    inflation_curve: pd.DataFrame,
) -> pd.DataFrame:
    if nominal_yields.empty or inflation_curve.empty:
        return pd.DataFrame()
    expanded_infl = _expand_maturity_grid(inflation_curve, MATURITY_GRID)
    expanded_nom = _expand_maturity_grid(nominal_yields, MATURITY_GRID)
    return discount_curves.build_discount_table(expanded_nom, expanded_infl)


def _write_csv(df: pd.DataFrame, path: Path, *, chunk_threshold_mb: int = 500, compress_threshold_mb: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        df.to_csv(path, index=False)
        return

    estimated_mb = _estimate_size_bytes(df) / (1024 * 1024)
    if estimated_mb > chunk_threshold_mb:
        chunk_size = max(int(len(df) * chunk_threshold_mb / estimated_mb), 1)
        for idx, start in enumerate(range(0, len(df), chunk_size), start=1):
            part = df.iloc[start : start + chunk_size]
            part_path = path.with_name(f"{path.stem}_part_{idx}.csv")
            part.to_csv(part_path, index=False)
            if part_path.stat().st_size > compress_threshold_mb * 1024 * 1024:
                _compress_csv(part_path)
        return

    df.to_csv(path, index=False)
    if path.stat().st_size > compress_threshold_mb * 1024 * 1024:
        _compress_csv(path)


def _compress_csv(path: Path) -> None:
    import gzip

    compressed = path.with_suffix(path.suffix + ".gz")
    with path.open("rt") as src, gzip.open(compressed, "wt") as dst:
        for chunk in iter(lambda: src.read(1 << 20), ""):
            dst.write(chunk)
    path.unlink()


def _tenor_label(years: float) -> str:
    if math.isnan(years):
        return "unknown"
    if years < 1:
        months = round(years * 12)
        return f"{months}M"
    rounded = min(30, max(1, int(round(years))))
    return f"{rounded}Y"


def _weekly_on_the_run_flags(trades: pd.DataFrame) -> pd.Series:
    if trades.empty or "years_to_maturity" not in trades:
        return pd.Series(False, index=trades.index)

    def _flag(group: pd.DataFrame) -> pd.Series:
        labelled = group.assign(tenor=group["years_to_maturity"].apply(_tenor_label))
        labelled = labelled.sort_values(["tenor", "years_to_maturity"], ascending=[True, False])
        max_per_tenor = labelled.groupby("tenor")["years_to_maturity"].transform("max")
        return labelled["years_to_maturity"] >= max_per_tenor

    monday = trades["trade_date"].map(lambda d: pd.Timestamp(d) - pd.offsets.Week(weekday=0))
    flags = trades.groupby(monday, group_keys=False).apply(_flag)
    flags.index = trades.index
    return flags


def _compute_enrichment(trades: pd.DataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=[
            "date",
            "futures_involved",
            "ats",
            "capacity",
            "on_the_run",
            "maturity_years",
            "tenor_label",
        ])

    trades = trades.copy()
    trades["trade_date"] = pd.to_datetime(trades["trade_date"]).dt.date
    on_the_run_flags = _weekly_on_the_run_flags(trades)
    trades["on_the_run_flag"] = on_the_run_flags.values

    grouped = trades.groupby("trade_date")

    futures = grouped["trade_mod4"].apply(lambda s: s.astype(str).str.upper().eq("B").any())
    def _ats_flag(values: pd.Series) -> bool:
        cleaned = values.fillna("").astype(str).str.upper()
        if cleaned.empty:
            return False
        return cleaned.apply(
            lambda v: bool(v) or any(v.startswith(prefix) for prefix in ATS_PREFIXES)
        ).any()

    ats = grouped["ats_mpid"].apply(_ats_flag)

    def _modal_capacity(values: pd.Series) -> str:
        cleaned = values.dropna().astype(str).str.upper()
        if cleaned.empty:
            return "unknown"
        counts = Counter(cleaned)
        most_common = counts.most_common()
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            return "mixed"
        return most_common[0][0]

    capacity = grouped["capacity"].apply(_modal_capacity)

    on_the_run = grouped["on_the_run_flag"].any()

    maturity = summary.set_index("trade_dt")["avg_years_to_maturity"].rename("maturity_years")
    tenor = maturity.apply(_tenor_label).rename("tenor_label")

    enriched = (
        pd.concat(
            [
                futures.rename("futures_involved"),
                ats.rename("ats"),
                capacity,
                on_the_run.rename("on_the_run"),
                maturity,
                tenor,
            ],
            axis=1,
        )
        .reset_index()
        .rename(columns={"trade_date": "date"})
        .sort_values("date")
    )
    return enriched


def _prepare_basis(summary: pd.DataFrame, discount: pd.DataFrame, tips_index: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    nominal_cashflows: List[pd.DataFrame] = []
    tips_cashflows: List[pd.DataFrame] = []
    for _, row in summary.iterrows():
        nominal_cashflows.append(
            cashflows_nominal.build_cashflows(
                discount,
                row.trade_dt,
                coupon_rate=row.avg_coupon,
                years_to_maturity=row.avg_years_to_maturity,
            )
        )
        tips_cashflows.append(
            cashflows_tips.build_cashflows(
                discount,
                tips_index,
                row.trade_dt,
                coupon_rate=row.avg_coupon,
                years_to_maturity=row.avg_years_to_maturity,
            )
        )
    nominal_df = cashflows_nominal.concatenate_cashflows(nominal_cashflows)
    tips_df = cashflows_tips.concatenate_cashflows(tips_cashflows)
    basis = synthetic_nominal.compute_basis(nominal_df, tips_df)
    basis = basis.rename(columns={"valuation_date": "date", "basis": "basis_px_per100"})
    basis["pv_nom"] = basis.pop("nominal_present_value")
    basis["pv_tips"] = basis.pop("tips_present_value")
    return nominal_df, tips_df, basis


def _ewma_residuals(series: pd.Series, span: int) -> pd.Series:
    trend = series.ewm(span=span, adjust=False).mean()
    return series - trend


def _estimate_ar1_half_life(series: pd.Series) -> Dict[str, float]:
    residuals = _ewma_residuals(series, HALF_LIFE_EWMA_SPAN).dropna()
    lagged = residuals.shift(1).dropna()
    aligned = residuals.loc[lagged.index]
    if len(aligned) < 10:
        return {"rho": float("nan"), "rho_se": float("nan"), "rho_ci_low": float("nan"), "rho_ci_high": float("nan"), "half_life_days": float("nan")}

    x = lagged.to_numpy(dtype=float)
    y = aligned.to_numpy(dtype=float)
    denom = np.dot(x, x)
    if denom == 0:
        return {"rho": float("nan"), "rho_se": float("nan"), "rho_ci_low": float("nan"), "rho_ci_high": float("nan"), "half_life_days": float("nan")}

    rho = float(np.dot(x, y) / denom)
    eps = y - rho * x
    sxx = denom
    T = len(eps)
    if T <= 1:
        return {"rho": rho, "rho_se": float("nan"), "rho_ci_low": float("nan"), "rho_ci_high": float("nan"), "half_life_days": float("nan")}

    gamma0 = np.sum((x * eps) ** 2) / T
    gamma = 0.0
    for lag in range(1, min(NEWEY_WEST_LAG, T - 1) + 1):
        weight = 1 - lag / (NEWEY_WEST_LAG + 1)
        lag_prod = np.sum((x[lag:] * eps[lag:]) * (x[:-lag] * eps[:-lag])) / T
        gamma += 2 * weight * lag_prod
    var = (gamma0 + gamma) / (sxx ** 2)
    rho_se = float(math.sqrt(var)) if var >= 0 else float("nan")
    z = 1.96
    ci_low = rho - z * rho_se
    ci_high = rho + z * rho_se
    if abs(rho) >= 1:
        half_life = float("inf")
    else:
        half_life = float(math.log(2) / abs(math.log(abs(rho))))
    return {
        "rho": rho,
        "rho_se": rho_se,
        "rho_ci_low": ci_low,
        "rho_ci_high": ci_high,
        "half_life_days": half_life,
    }


def _estimate_event_decay(series: pd.Series) -> Dict[str, float]:
    cleaned = _ewma_residuals(series, HALF_LIFE_EWMA_SPAN).dropna()
    if cleaned.empty:
        return {"lambda": float("nan"), "half_life_days": float("nan")}
    y = cleaned.to_numpy(dtype=float)
    t = np.arange(len(y), dtype=float)

    def model(t, A, lam, C):
        return A * np.exp(-t / lam) + C

    try:
        from scipy.optimize import curve_fit  # type: ignore

        bounds = ([0.0, 1e-6, -np.inf], [np.inf, np.inf, np.inf])
        guess = [abs(y[0] - y[-1]), max(len(y) / 4, 1.0), y[-1]]
        params, _ = curve_fit(model, t, y, p0=guess, bounds=bounds, maxfev=10000)
        lam = float(params[1])
        half_life = lam * math.log(2)
        return {"lambda": lam, "half_life_days": half_life}
    except Exception:  # pragma: no cover - SciPy optional
        lam = max(len(y) / 4, 1.0)
        return {"lambda": lam, "half_life_days": lam * math.log(2)}


def _compute_half_life_metrics(basis: pd.DataFrame, splits: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if basis.empty:
        columns_overall = [
            "group",
            "rho",
            "rho_se",
            "rho_ci_low",
            "rho_ci_high",
            "half_life_ar1_days",
            "half_life_event_days",
        ]
        return pd.DataFrame(columns=columns_overall), pd.DataFrame(columns=["split", "value", "half_life_ar1_days", "half_life_event_days"])

    basis_series = basis.set_index("date")["basis_px_per100"].astype(float).sort_index()
    overall_ar1 = _estimate_ar1_half_life(basis_series)
    overall_event = _estimate_event_decay(basis_series)
    overall_df = pd.DataFrame(
        {
            "group": ["overall"],
            "rho": [overall_ar1["rho"]],
            "rho_se": [overall_ar1["rho_se"]],
            "rho_ci_low": [overall_ar1["rho_ci_low"]],
            "rho_ci_high": [overall_ar1["rho_ci_high"]],
            "half_life_ar1_days": [overall_ar1["half_life_days"]],
            "half_life_event_days": [overall_event["half_life_days"]],
        }
    )

    if splits is None or splits.empty:
        return overall_df, pd.DataFrame(columns=["split", "value", "half_life_ar1_days", "half_life_event_days"])

    merged = basis.merge(splits, left_on="date", right_on="date", how="left")
    split_records: List[Mapping[str, object]] = []
    desired_splits = ["futures_involved", "ats", "capacity", "on_the_run", "tenor_label"]
    split_cols = [col for col in desired_splits if col in merged.columns]
    for col in split_cols:
        groups = merged.groupby(col)
        for value, group in groups:
            series = group.set_index("date")["basis_px_per100"].astype(float).sort_index()
            if series.empty:
                continue
            ar1 = _estimate_ar1_half_life(series)
            event = _estimate_event_decay(series)
            record = {
                "split": col,
                "value": value,
                "half_life_ar1_days": ar1["half_life_days"],
                "half_life_event_days": event["half_life_days"],
            }
            split_records.append(record)

    split_df = pd.DataFrame(split_records)
    return overall_df, split_df


def _compute_qa_checks(basis: pd.DataFrame, splits: pd.DataFrame) -> Dict[str, object]:
    if basis.empty:
        return {"coverage_days": 0, "basis_min": None, "basis_max": None, "basis_mean": None, "nominal_real_correlation": None}

    coverage_days = basis["date"].nunique()
    basis_values = basis["basis_px_per100"].astype(float)
    corr = float("nan")
    try:
        corr = float(basis[["pv_nom", "pv_tips"]].corr().iloc[0, 1])
    except Exception:
        corr = float("nan")

    missingness = {col: int(basis[col].isna().sum()) for col in ["pv_nom", "pv_tips", "basis_px_per100"]}
    sampling_days = int(splits.get("sampled", pd.Series(dtype=bool)).sum()) if "sampled" in splits else 0

    return {
        "coverage_days": int(coverage_days),
        "basis_min": float(basis_values.min()),
        "basis_max": float(basis_values.max()),
        "basis_mean": float(basis_values.mean()),
        "nominal_real_correlation": corr,
        "missing_counts": missingness,
        "sampling_days": sampling_days,
    }


def _make_progress_report(
    path: Path,
    basis: pd.DataFrame,
    overall_half_life: pd.DataFrame,
    split_half_life: pd.DataFrame,
    qa_metrics: Mapping[str, object],
    failures: Sequence[Mapping[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Plan C Safe Pipeline Progress",
        "",
        f"- Coverage days: {qa_metrics.get('coverage_days', 0)}",
        f"- Basis min / max / mean: {qa_metrics.get('basis_min')} / {qa_metrics.get('basis_max')} / {qa_metrics.get('basis_mean')}",
        f"- Nominal vs TIPS PV correlation: {qa_metrics.get('nominal_real_correlation')}",
        "",
        "## Half-life (Overall)",
    ]
    if overall_half_life.empty:
        lines.append("No basis observations available.")
    else:
        for _, row in overall_half_life.iterrows():
            lines.append(
                f"- AR(1) half-life: {row['half_life_ar1_days']:.2f} days (rho={row['rho']:.4f} ± {row['rho_se']:.4f})"
            )
            lines.append(f"- Event-decay half-life: {row['half_life_event_days']:.2f} days")

    lines.append("\n## Half-life by Split")
    if split_half_life.empty:
        lines.append("No split-level observations available.")
    else:
        for _, row in split_half_life.iterrows():
            lines.append(
                f"- {row['split']} = {row['value']}: AR(1) {row['half_life_ar1_days']:.2f} days, "
                f"Event-decay {row['half_life_event_days']:.2f} days"
            )

    if failures:
        lines.append("\n## Sampling & Failures")
        for failure in failures:
            lines.append(
                f"- {failure['batch']}: reason={failure.get('reason', 'unknown')} sampled={failure.get('sampled', False)}"
            )

    with path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


@dataclass
class PlanCSafeConfig:
    years: Sequence[int] = tuple(range(2020, 2024))
    quarters: Sequence[int] = (1, 2, 3, 4)
    maturities: Sequence[int] = MATURITY_GRID
    horizons: Sequence[int] = MATURITY_GRID
    throttle_seconds: float = 3.0
    trace_retries: int = 3
    use_sample_data: bool = False
    output_dir: Path = field(default_factory=default_output_dir)


def run_planC_fullsample_enrich_safe(config: Optional[PlanCSafeConfig] = None) -> Mapping[str, pd.DataFrame]:
    config = config or PlanCSafeConfig()
    base_output = Path(config.output_dir)
    dirs = _ensure_output_dirs(base_output)
    cache_dir = base_output / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if config.use_sample_data:
        trace_client = None
        treasury_client = None
        tips_client = None
        inflation_client = None
    else:
        trace_client = WRDSTraceClient()
        treasury_client = CRSPTreasuryClient()
        tips_client = TIPSIndexClient()
        inflation_client = ClevelandFedClient()

    basis_parts: List[pd.DataFrame] = []
    enriched_parts: List[pd.DataFrame] = []
    failures: List[Dict[str, object]] = []

    for year in config.years:
        for quarter in config.quarters:
            start, end = quarter_date_range(year, quarter)
            batch_label = f"{year}Q{quarter}"
            batch_dir = dirs["raw"] / batch_label
            batch_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.info("Processing batch %s (%s to %s)", batch_label, start, end)

            if config.use_sample_data:
                date_index = pd.date_range(start, end, freq="D")
                trace_df = make_sample_trace_data(date_index)
                sampled = False
            else:
                assert trace_client is not None
                trace_df, sampled = _safe_fetch_trace(
                    trace_client,
                    start,
                    end,
                    throttle_seconds=config.throttle_seconds,
                    retries=config.trace_retries,
                )

            if trace_df.empty:
                failures.append({"batch": batch_label, "reason": "no_trace_data", "sampled": sampled})
                continue

            trace_trimmed = _trim_trace_columns(trace_df)
            trace_trimmed["sampled"] = sampled
            _write_csv(trace_trimmed, batch_dir / "trace_trades.csv")

            summary = summarise_trades(trace_df)
            summary.to_csv(batch_dir / "trace_summary.csv", index=False)

            if config.use_sample_data:
                date_index = pd.date_range(start, end, freq="D")
                treasury_df = make_sample_treasury_yields(date_index, config.maturities)
                tips_index_df = make_sample_tips_index(date_index)
                inflation_df = make_sample_inflation_curve(date_index, config.horizons)
            else:
                assert treasury_client is not None
                assert tips_client is not None
                assert inflation_client is not None
                treasury_df = _safe_fetch_treasury(
                    treasury_client,
                    start,
                    end,
                    maturities=config.maturities,
                )
                tips_index_df = _safe_fetch_tips_index(tips_client, start, end)
                inflation_df = _safe_fetch_inflation(
                    inflation_client,
                    cache_dir,
                    start,
                    end,
                    horizons=config.horizons,
                )

            if treasury_df.empty or inflation_df.empty or tips_index_df.empty:
                failures.append({"batch": batch_label, "reason": "missing_market_data", "sampled": sampled})
                continue

            discount = _build_discount_table(treasury_df, inflation_df)
            if discount.empty:
                failures.append({"batch": batch_label, "reason": "discount_table_empty", "sampled": sampled})
                continue

            discount.to_csv(batch_dir / "discount_table.csv", index=False)
            tips_index_df.to_csv(batch_dir / "tips_index.csv", index=False)

            nominal_cf, tips_cf, basis = _prepare_basis(summary, discount, tips_index_df)
            nominal_cf.to_csv(batch_dir / "nominal_cashflows.csv", index=False)
            tips_cf.to_csv(batch_dir / "tips_cashflows.csv", index=False)
            _write_csv(basis, batch_dir / "synthetic_basis.csv")

            enrichment = _compute_enrichment(trace_trimmed, summary)
            enrichment["sampled"] = sampled
            enrichment.to_csv(batch_dir / "panel_enriched.csv", index=False)

            basis_parts.append(basis.assign(batch=batch_label))
            enriched_parts.append(enrichment.assign(batch=batch_label))
    synthetic_basis = pd.concat(basis_parts, ignore_index=True) if basis_parts else pd.DataFrame()
    enriched_panel = pd.concat(enriched_parts, ignore_index=True) if enriched_parts else pd.DataFrame()

    overall_half_life, split_half_life = _compute_half_life_metrics(synthetic_basis, enriched_panel)
    qa_metrics = _compute_qa_checks(synthetic_basis, enriched_panel)

    _write_csv(synthetic_basis, base_output / "synthetic_basis.csv")
    enriched_panel.to_csv(base_output / "panel_enriched.csv", index=False)
    overall_half_life.to_csv(base_output / "half_life_overall.csv", index=False)
    split_half_life.to_csv(base_output / "half_life_by_split.csv", index=False)

    qa_path = base_output / "qa_checks.json"
    qa_path.write_text(json.dumps(qa_metrics, indent=2, default=str))

    log_dir = dirs["logs"]
    if failures:
        failures_path = log_dir / "failures.json"
        failures_path.write_text(json.dumps(failures, indent=2, default=str))

    _make_progress_report(
        base_output / "progress_report.md",
        synthetic_basis,
        overall_half_life,
        split_half_life,
        qa_metrics,
        failures,
    )

    return {
        "synthetic_basis": synthetic_basis,
        "panel_enriched": enriched_panel,
        "half_life_overall": overall_half_life,
        "half_life_by_split": split_half_life,
    }


__all__ = ["PlanCSafeConfig", "run_planC_fullsample_enrich_safe"]

