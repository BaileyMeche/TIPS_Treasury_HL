"""Top level orchestration for the Plan C pipeline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import pandas as pd

from . import cashflows_nominal, cashflows_tips, discount_curves, synthetic_nominal
from .crsp_tsy import CRSPTreasuryClient, TreasuryQuery, make_sample_treasury_yields
from .infl_curve_public import (
    ClevelandFedClient,
    InflationCurveQuery,
    make_sample_inflation_curve,
)
from .tips_index import TIPSIndexClient, TipsIndexQuery, make_sample_tips_index
from .wrds_trace import (
    TraceQuery,
    WRDSTraceClient,
    make_sample_trace_data,
    summarise_trades,
)

try:  # pragma: no cover - optional dependency
    from settings import config as project_config
except Exception:  # pragma: no cover - optional dependency
    project_config = None

LOGGER = logging.getLogger(__name__)
DEFAULT_MATURITIES = (1, 3, 5, 7, 10)
DEFAULT_HORIZONS = (1, 3, 5, 7, 10)
OUTPUT_FILES: Mapping[str, str] = {
    "trace": "trace_trades.csv",
    "trace_summary": "trace_summary.csv",
    "treasury_yields": "treasury_yields.csv",
    "inflation_curve": "inflation_curve.csv",
    "discount_table": "discount_table.csv",
    "tips_index": "tips_index.csv",
    "nominal_cashflows": "nominal_cashflows.csv",
    "tips_cashflows": "tips_cashflows.csv",
    "synthetic_basis": "synthetic_basis.csv",
}


def default_output_dir() -> Path:
    if project_config is not None:  # pragma: no branch - simple guard
        try:
            return Path(project_config("OUTPUT_DIR")) / "planC"
        except Exception:  # pragma: no cover - fallback in case config fails
            pass
    return Path("_output") / "planC"


def run_pipeline(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    *,
    output_dir: Optional[Path | str] = None,
    maturities: Sequence[int] = DEFAULT_MATURITIES,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    use_sample_data: bool = False,
    trace_client: Optional[WRDSTraceClient] = None,
    treasury_client: Optional[CRSPTreasuryClient] = None,
    tips_client: Optional[TIPSIndexClient] = None,
    inflation_client: Optional[ClevelandFedClient] = None,
) -> Dict[str, pd.DataFrame]:
    """Run the Plan C pipeline and return the intermediate artefacts."""

    start = pd.Timestamp(start_date).date()
    end = pd.Timestamp(end_date).date()
    if start > end:
        raise ValueError("start_date must be before end_date")
    LOGGER.info("Running Plan C pipeline from %s to %s", start, end)
    date_index = pd.date_range(start, end, freq="D")

    if use_sample_data:
        trace_df = make_sample_trace_data(date_index)
        treasury_df = make_sample_treasury_yields(date_index, maturities)
        tips_df = make_sample_tips_index(date_index)
        inflation_df = make_sample_inflation_curve(date_index, horizons)
    else:
        trace_client = trace_client or WRDSTraceClient()
        treasury_client = treasury_client or CRSPTreasuryClient()
        tips_client = tips_client or TIPSIndexClient()
        inflation_client = inflation_client or ClevelandFedClient()

        trace_df = trace_client.fetch_trades(TraceQuery(start, end))
        treasury_df = treasury_client.fetch_nominal_yields(
            TreasuryQuery(start, end, maturities=maturities)
        )
        tips_df = tips_client.fetch_index(TipsIndexQuery(start, end))
        inflation_df = inflation_client.fetch_curve(
            InflationCurveQuery(start, end, horizons=horizons)
        )

    discount_table = discount_curves.build_discount_table(treasury_df, inflation_df)
    trace_summary = summarise_trades(trace_df)

    nominal_cashflows = []
    tips_cashflows_list = []
    for _, row in trace_summary.iterrows():
        nominal_cashflows.append(
            cashflows_nominal.build_cashflows(
                discount_table,
                row.trade_dt,
                coupon_rate=row.avg_coupon,
                years_to_maturity=row.avg_years_to_maturity,
            )
        )
        tips_cashflows_list.append(
            cashflows_tips.build_cashflows(
                discount_table,
                tips_df,
                row.trade_dt,
                coupon_rate=row.avg_coupon,
                years_to_maturity=row.avg_years_to_maturity,
            )
        )

    nominal_cf_df = cashflows_nominal.concatenate_cashflows(nominal_cashflows)
    tips_cf_df = cashflows_tips.concatenate_cashflows(tips_cashflows_list)
    synthetic_basis = synthetic_nominal.compute_basis(nominal_cf_df, tips_cf_df)

    results: Dict[str, pd.DataFrame] = {
        "trace": trace_df,
        "trace_summary": trace_summary,
        "treasury_yields": treasury_df,
        "inflation_curve": inflation_df,
        "discount_table": discount_table,
        "tips_index": tips_df,
        "nominal_cashflows": nominal_cf_df,
        "tips_cashflows": tips_cf_df,
        "synthetic_basis": synthetic_basis,
    }

    if output_dir is not None:
        write_outputs(results, output_dir)

    return results


def write_outputs(results: Mapping[str, pd.DataFrame], output_dir: Path | str) -> None:
    """Persist pipeline artefacts to ``output_dir`` as CSV files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    for key, filename in OUTPUT_FILES.items():
        df = results.get(key)
        if df is None:
            continue
        path = output_path / filename
        LOGGER.info("Writing %s", path)
        df.to_csv(path, index=False)


def parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Plan C data pipeline")
    parser.add_argument("--start", required=False, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", required=False, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--output-dir", default=None, help="Directory for CSV outputs")
    parser.add_argument(
        "--sample-data",
        action="store_true",
        default=False,
        help="Use bundled synthetic data instead of live vendor pulls",
    )
    parser.add_argument(
        "--novendor",
        action="store_true",
        default=False,
        help="Alias for --sample-data",
    )
    parser.add_argument(
        "--maturities",
        nargs="*",
        type=int,
        default=list(DEFAULT_MATURITIES),
        help="Treasury maturities (in years) to include",
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        type=int,
        default=list(DEFAULT_HORIZONS),
        help="Inflation expectation horizons (in years)",
    )
    return parser.parse_args(args=args)


def resolve_dates(args: argparse.Namespace) -> tuple[pd.Timestamp, pd.Timestamp]:
    if args.start is None or args.end is None:
        if project_config is not None:  # pragma: no branch - simple guard
            try:
                start = project_config("START_DATE").date()
                end = project_config("END_DATE").date()
                return pd.Timestamp(start), pd.Timestamp(end)
            except Exception:  # pragma: no cover - fallback
                pass
        raise SystemExit("--start and --end are required when settings.py is unavailable")
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    return start, end


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, pd.DataFrame]:
    logging.basicConfig(level=logging.INFO)
    args = parse_args(argv)
    sample = args.sample_data or args.novendor
    start, end = resolve_dates(args)
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    results = run_pipeline(
        start,
        end,
        output_dir=output_dir,
        maturities=args.maturities,
        horizons=args.horizons,
        use_sample_data=sample,
    )
    LOGGER.info("Synthetic basis head:\n%s", results["synthetic_basis"].head())
    return results


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
