"""Utilities to demonstrate the Plan C pipeline end-to-end with artefacts."""

from __future__ import annotations

import argparse
import logging
import numbers
import os
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import pandas as pd

from .pipeline import DEFAULT_HORIZONS, DEFAULT_MATURITIES, default_output_dir, run_pipeline

try:  # pragma: no cover - optional dependency for figure styling
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - guard against missing matplotlib
    raise RuntimeError(
        "matplotlib is required to generate the proof-of-concept figures"
    ) from exc

LOGGER = logging.getLogger(__name__)

FIGURE_NAME = "basis_timeseries.png"
SUMMARY_NAME = "basis_summary.csv"
REPORT_NAME = "planC_proof_of_concept.md"


def _ensure_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _create_figure(basis: pd.DataFrame, figure_path: Path) -> None:
    """Create a basis time-series chart."""

    _ensure_path(figure_path)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(basis["valuation_date"], basis["basis"], marker="o", linestyle="-", linewidth=1.0)
    ax.set_title("Synthetic Nominal vs TIPS Basis (Proof-of-Concept)")
    ax.set_xlabel("Valuation date")
    ax.set_ylabel("Basis (price units)")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)


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
    figure_path: Path,
    report_path: Path,
    *,
    use_sample_data: bool,
) -> None:
    _ensure_path(report_path)
    figure_rel = os.path.relpath(figure_path, report_path.parent)
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
            "## Basis time series",
            "",
            f"![Synthetic basis time series]({figure_rel})",
            "",
            "The synthetic nominal leg and the TIPS leg are constructed using the "
            "Plan C modules. The resulting basis illustrates how the toolkit "
            "can surface mispricing dynamics even when only public proxies are available.",
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

    figure_path = output / FIGURE_NAME
    _create_figure(basis, figure_path)

    summary = _build_summary_table(basis)
    summary_path = output / SUMMARY_NAME
    _ensure_path(summary_path)
    summary.to_csv(summary_path, index=False)

    _write_report(summary, figure_path, report, use_sample_data=not live_data)

    return {
        "results": results,
        "figure_path": figure_path,
        "summary_path": summary_path,
        "report_path": report,
    }


def _parse_args(args: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Plan C proof-of-concept workflow")
    parser.add_argument("--start", default="2023-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default="2023-01-05", help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for CSV outputs and figures (defaults to _output/planC/proof_of_concept)",
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
    LOGGER.info("Proof-of-concept artefacts written to %s", artefacts["figure_path"].parent)
    LOGGER.info("Report available at %s", artefacts["report_path"])
    return artefacts


__all__ = ["run_proof_of_concept", "main"]
