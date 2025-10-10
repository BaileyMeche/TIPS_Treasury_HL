"""Utilities to demonstrate the Plan C pipeline end-to-end with artefacts."""

from __future__ import annotations

import argparse
import logging
import numbers
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import pandas as pd

from .pipeline import DEFAULT_HORIZONS, DEFAULT_MATURITIES, default_output_dir, run_pipeline

LOGGER = logging.getLogger(__name__)

SUMMARY_NAME = "basis_summary.csv"
REPORT_NAME = "planC_proof_of_concept.md"


def _ensure_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _build_summary_table(basis: pd.DataFrame) -> pd.DataFrame:
    stats = {
        "metric": [
            "start_date",
            "end_date",
            "observations",
            "mean_basis",
            "median_basis",
            "std_basis",
            "min_basis",
            "max_basis",
        ],
        "value": [
            basis["valuation_date"].min().date().isoformat(),
            basis["valuation_date"].max().date().isoformat(),
            int(len(basis)),
            float(basis["basis"].mean()),
            float(basis["basis"].median()),
            float(basis["basis"].std(ddof=0)),
            float(basis["basis"].min()),
            float(basis["basis"].max()),
        ],
    }
    return pd.DataFrame(stats)


def _write_report(
    summary: pd.DataFrame,
    report_path: Path,
    *,
    use_sample_data: bool,
) -> None:
    _ensure_path(report_path)
    lines = [
        "# Plan C Proof-of-Concept",
        "",
        "This proof-of-concept run executes the Plan C pipeline end-to-end and "
        "produces artefacts that can be shared with stakeholders.",
        "",
        f"*Data mode:* {'Sample synthetic inputs' if use_sample_data else 'Live data pulls'}",
        "",
        "## Summary statistics",
    ]
    for _, row in summary.iterrows():
        value = row["value"]
        if isinstance(value, numbers.Real) and not isinstance(value, bool):
            value_str = f"{value:,.4f}" if not float(value).is_integer() else f"{int(value)}"
        else:
            value_str = str(value)
        lines.append(f"- **{row['metric'].replace('_', ' ').title()}:** {value_str}")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "The synthetic nominal leg and the TIPS leg are constructed using the "
            "Plan C modules. The resulting basis illustrates how the toolkit "
            "can surface mispricing dynamics even when only public proxies are available.",
            "This proof-of-concept configuration focuses on reproducible CSV outputs "
            "and omits plot generation to keep the artefact footprint lightweight.",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def run_proof_of_concept(
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    *,
    output_dir: Optional[Path | str] = None,
    report_path: Optional[Path | str] = None,
    maturities: Sequence[int] = DEFAULT_MATURITIES,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    live_data: bool = False,
) -> Dict[str, object]:
    """Execute the pipeline and generate a lightweight report."""

    output = Path(output_dir) if output_dir is not None else default_output_dir() / "proof_of_concept"
    report = Path(report_path) if report_path is not None else Path("reports") / REPORT_NAME

    LOGGER.info(
        "Running Plan C proof-of-concept from %s to %s (sample=%s)",
        start_date,
        end_date,
        not live_data,
    )

    results = run_pipeline(
        start_date,
        end_date,
        output_dir=output,
        maturities=maturities,
        horizons=horizons,
        use_sample_data=not live_data,
    )

    basis = results["synthetic_basis"].copy()
    basis["valuation_date"] = pd.to_datetime(basis["valuation_date"])
    basis = basis.sort_values("valuation_date")

    summary = _build_summary_table(basis)
    summary_path = output / SUMMARY_NAME
    _ensure_path(summary_path)
    summary.to_csv(summary_path, index=False)

    _write_report(summary, report, use_sample_data=not live_data)

    return {
        "results": results,
        "summary_path": summary_path,
        "report_path": report,
        "output_dir": output,
    }


def _parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Plan C proof-of-concept workflow")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-01-05", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV outputs (defaults to _output/planC/proof_of_concept)",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="Path to the Markdown report (defaults to reports/planC_proof_of_concept.md)",
    )
    parser.add_argument(
        "--maturities",
        nargs="*",
        type=int,
        default=list(DEFAULT_MATURITIES),
        help="Treasury maturities (years)",
    )
    parser.add_argument(
        "--horizons",
        nargs="*",
        type=int,
        default=list(DEFAULT_HORIZONS),
        help="Inflation expectation horizons (years)",
    )
    parser.add_argument(
        "--live-data",
        action="store_true",
        help="Use live data sources instead of bundled sample inputs",
    )
    return parser.parse_args(args=args)


def main(argv: Optional[Sequence[str]] = None) -> Mapping[str, object]:  # pragma: no cover - CLI wrapper
    logging.basicConfig(level=logging.INFO)
    args = _parse_args(argv)
    output_dir = Path(args.output_dir) if args.output_dir else None
    report_path = Path(args.report) if args.report else None
    artefacts = run_proof_of_concept(
        args.start,
        args.end,
        output_dir=output_dir,
        report_path=report_path,
        maturities=args.maturities,
        horizons=args.horizons,
        live_data=args.live_data,
    )
    LOGGER.info("Proof-of-concept artefacts written to %s", artefacts["output_dir"])
    LOGGER.info("Report available at %s", artefacts["report_path"])
    return artefacts


__all__ = ["run_proof_of_concept", "main"]
