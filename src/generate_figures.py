from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from settings import DATA_DIR, OUTPUT_DIR


def load_tips_treasury_data(
    file_path: str | Path | None = None, *, filter_columns: bool = True
):
    """
    Load TIPS-Treasury arbitrage data from parquet file.

    Parameters:
        file_path (str | Path | None): Path to the parquet file. If ``None``,
            defaults to :data:`DATA_DIR / "tips_treasury_implied_rf.parquet"`.
        filter_columns (bool): If True, return only arbitrage columns

    Returns:
        pd.DataFrame: DataFrame with the requested data
    """
    try:
        resolved_path = Path(file_path) if file_path is not None else DATA_DIR / "tips_treasury_implied_rf.parquet"

        # Read the parquet file
        df = pd.read_parquet(resolved_path)

        # Set the date as index
        if 'date' in df.columns:
            df.index = df['date']

        # Extract only arbitrage columns if requested
        if filter_columns:
            arb_cols = [col for col in df.columns if col.startswith('arb_')]
            df = df[arb_cols]

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# Function to calculate AR(1) coefficient for an entire series
def ar1_coefficient(series):
    # Drop NaN values
    series = series.dropna()

    if len(series) <= 1:
        return np.nan

    y = series[1:].values
    X = add_constant(series[:-1].values)

    try:
        model = OLS(y, X).fit()
        return model.params[1]  # AR(1) coefficient
    except:
        return np.nan

def generate_summary_statistics(test_df, start_date=None, end_date=None, save_path=None):
    """
    Generate summary statistics for the TIPS-Treasury arbitrage data.

    Parameters:
        test_df (pd.DataFrame): DataFrame containing arbitrage data
        start_date (str): Start date in format 'YYYY-MM-DD' (optional)
        end_date (str): End date in format 'YYYY-MM-DD' (optional)
        save_path (str): Path to save the summary statistics as a CSV file

    Returns:
        pd.DataFrame: Summary statistics with renamed indices and formatted values
    """
    if start_date and end_date:
        df = test_df.loc[start_date:end_date].copy()
    elif start_date:
        df = test_df.loc[start_date:].copy()
    elif end_date:
        df = test_df.loc[:end_date].copy()
    else:
        df = test_df.copy()

    arb_cols = [col for col in df.columns if col.startswith('arb_') and not col.endswith('_AR1')]

    summary = pd.DataFrame()

    col_name_map = {
        'arb_2': 'TIPS-Treasury 2Y',
        'arb_5': 'TIPS-Treasury 5Y',
        'arb_10': 'TIPS-Treasury 10Y',
        'arb_20': 'TIPS-Treasury 20Y'
    }

    for col in arb_cols:
        series = df[col]

        ar1_val = ar1_coefficient(series)

        min_val = max(0, series.min())

        stats = {
            'Mean': round(series.mean()),
            'p50': round(series.median()),
            'Std. Dev': round(series.std()),
            'Min': round(min_val),
            'Max': round(series.max()),
            'AR1': round(ar1_val, 3),  # Keep AR1 to 2 decimal places
            'First': series.first_valid_index().strftime('%b-%Y') if not pd.isna(series.first_valid_index()) else 'N/A',
            'Last': series.last_valid_index().strftime('%b-%Y') if not pd.isna(series.last_valid_index()) else 'N/A',
            'N': int(series.count())
        }

        col_name = col_name_map.get(col, col)

        summary[col_name] = pd.Series(stats)

    if save_path:
        summary.to_csv(save_path)

    return summary.T


def plot_tips_treasury_spreads(
    data_df,
    start_date=None,
    end_date=None,
    figsize=(12, 6),
    style="dark",
    save_path=None,
    threshold: float = 0.0,
    highlight_events: Mapping[str, Iterable[tuple[pd.Timestamp, pd.Timestamp]]] | None = None,
    shade_threshold_exceedances: bool = True,
):
    """
    Plot TIPS-Treasury spreads over time.

    Parameters:
        data_df (pd.DataFrame): DataFrame containing arbitrage data
        start_date (str): Start date in format 'YYYY-MM-DD' (optional)
        end_date (str): End date in format 'YYYY-MM-DD' (optional)
        figsize (tuple): Figure size as (width, height)
        style (str): Seaborn style theme
        save_path (str): If provided, save figure to this path

    Returns:
        matplotlib.figure.Figure: The figure object
    """
    sns.set_theme(style=style)

    date_filter = slice(start_date, end_date)

    legend_name_map = {
        "arb_2": "2Y",
        "arb_5": "5Y",
        "arb_10": "10Y",
        "arb_20": "20Y"
    }

    subset = data_df.loc[date_filter]
    fig, ax = plt.subplots(figsize=figsize)

    palette = sns.color_palette("colorblind", len(subset.columns))
    for color, column in zip(palette, subset.columns):
        series = subset[column]
        ax.plot(series.index, series.values, label=legend_name_map.get(column, column), color=color)

        if highlight_events:
            for start, end in highlight_events.get(column, []):
                ax.axvspan(start, end, color=color, alpha=0.15)

        if shade_threshold_exceedances:
            mask = series > threshold
            if mask.any():
                ax.fill_between(
                    series.index,
                    threshold,
                    series.where(mask),
                    where=mask,
                    color=color,
                    alpha=0.1,
                )

    title_dates = f"({start_date[:4] if start_date else ''}-{end_date[:4] if end_date else ''})"
    ax.set_title(f'TIPS Treasury Rates {title_dates}', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Spread (bps)', fontsize=14)
    ax.grid(True, axis='y')

    ax.axhline(threshold, color='black', linestyle='--', linewidth=0.8, label=f'Threshold ({threshold:.1f} bps)')

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=4)

    plt.tight_layout()

    # Save if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)

    return fig


def identify_mispricing_events(data_df: pd.DataFrame, *, threshold: float = 0.0) -> pd.DataFrame:
    """Identify contiguous stretches where arbitrage spreads exceed ``threshold``.

    Parameters
    ----------
    data_df:
        DataFrame containing ``arb_*`` columns.
    threshold:
        Basis point level that a spread must exceed to be considered a mispricing
        event.

    Returns
    -------
    pandas.DataFrame
        Table with columns ``series``, ``start``, ``end``, ``duration_days``,
        ``max_spread`` and ``mean_spread`` summarising each event.
    """

    events: list[dict[str, object]] = []
    arb_columns = [col for col in data_df.columns if col.startswith('arb_')]

    for column in arb_columns:
        series = data_df[column].dropna()
        if series.empty:
            continue

        mask = (series > threshold).fillna(False)
        if not mask.any():
            continue

        starts = mask & ~mask.shift(fill_value=False)
        ends = mask & ~mask.shift(-1, fill_value=False)

        for start, end in zip(series.index[starts], series.index[ends]):
            window = series.loc[start:end]
            events.append(
                {
                    'series': column,
                    'start': start,
                    'end': end,
                    'duration_days': (end - start).days + 1,
                    'max_spread': window.max(),
                    'mean_spread': window.mean(),
                    'threshold': threshold,
                }
            )

    return pd.DataFrame(events)


def _prepare_event_windows(events: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    windows: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    if events.empty:
        return windows

    for series_name, group in events.groupby('series'):
        windows[series_name] = list(zip(group['start'], group['end']))
    return windows


def build_mispricing_report(
    summary_stats: pd.DataFrame,
    events: pd.DataFrame,
    *,
    figure_paths: Iterable[Path],
    output_path: Path,
    threshold: float,
) -> None:
    """Render a lightweight HTML report for mispricing diagnostics."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary_html = summary_stats.to_html(classes='table table-striped', float_format=lambda x: f"{x:,.2f}")

    if events.empty:
        events_html = '<p>No mispricing events exceeded the specified threshold.</p>'
    else:
        display_events = events.copy()
        display_events['start'] = display_events['start'].dt.strftime('%Y-%m-%d')
        display_events['end'] = display_events['end'].dt.strftime('%Y-%m-%d')
        events_html = display_events.to_html(index=False, classes='table table-bordered', float_format=lambda x: f"{x:,.2f}")

    figures_html = ''.join(
        f'<figure><img src="{Path(path).name}" alt="{Path(path).stem}" style="max-width: 100%; height: auto;" />'
        f'<figcaption>{Path(path).stem.replace("_", " ")}</figcaption></figure>'
        for path in figure_paths
    )

    html = f"""
    <html>
        <head>
            <meta charset="utf-8" />
            <title>TIPS-Treasury Mispricing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2rem; }}
                h1, h2 {{ color: #2F4F4F; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 1.5rem; }}
                th, td {{ border: 1px solid #ddd; padding: 0.5rem; text-align: center; }}
                th {{ background-color: #f2f2f2; }}
                figure {{ margin: 2rem 0; }}
                figcaption {{ text-align: center; font-style: italic; margin-top: 0.5rem; }}
            </style>
        </head>
        <body>
            <h1>TIPS-Treasury Mispricing Diagnostic</h1>
            <p>Threshold for defining a mispricing event: <strong>{threshold:.1f} basis points</strong>.</p>
            <h2>Summary Statistics</h2>
            {summary_html}
            <h2>Mispricing Events</h2>
            {events_html}
            <h2>Figures</h2>
            {figures_html}
        </body>
    </html>
    """

    output_path.write_text(html, encoding='utf-8')

if __name__ == '__main__':
    data_path = DATA_DIR / "tips_treasury_implied_rf.parquet"
    base_fig_path = OUTPUT_DIR / "tips_treasury_spreads.png"
    mispricing_fig_path = OUTPUT_DIR / "tips_treasury_spreads_mispricing.png"
    summary_stats_path = OUTPUT_DIR / "tips_treasury_summary_stats.csv"
    events_csv_path = OUTPUT_DIR / "mispricing_events.csv"
    report_path = OUTPUT_DIR / "mispricing_report.html"

    arb_data = load_tips_treasury_data(file_path=data_path)
    if arb_data is None:
        raise SystemExit("TIPS-Treasury data could not be loaded. Ensure the compute step has completed successfully.")

    summary_stats = generate_summary_statistics(arb_data, save_path=summary_stats_path)

    mispricing_threshold = 0.0
    events = identify_mispricing_events(arb_data, threshold=mispricing_threshold)
    events.to_csv(events_csv_path, index=False)

    base_fig = plot_tips_treasury_spreads(
        arb_data,
        save_path=base_fig_path,
        threshold=mispricing_threshold,
        shade_threshold_exceedances=False,
    )
    plt.close(base_fig)

    event_windows = _prepare_event_windows(events)
    mispricing_fig = plot_tips_treasury_spreads(
        arb_data,
        save_path=mispricing_fig_path,
        threshold=mispricing_threshold,
        highlight_events=event_windows,
        shade_threshold_exceedances=True,
    )
    plt.close(mispricing_fig)

    build_mispricing_report(
        summary_stats,
        events,
        figure_paths=[base_fig_path, mispricing_fig_path],
        output_path=report_path,
        threshold=mispricing_threshold,
    )

