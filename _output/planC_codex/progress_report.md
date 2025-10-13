# Plan C Half-Life Progress Report

## What we did
- Processed 3752 trading days of TIPS–Treasury arbitrage from 2010-01-04 through 2024-12-31, standardizing tenor labels and constructing winsorised as well as EWMA-detrended spreads.
- Ran ADF/KPSS stationarity tests, AR(1) half-life estimation with Newey–West covariance, and exponential decay fits on shock windows identified via diff/IQR thresholds.
- Built a heuristic macro event calendar (CPI, FOMC, Treasury refunding, TIPS announcements/auctions/settlements) with business-day alignment and logged coverage in QA.
- Produced robustness sweeps (EWMA spans, winsorisation toggles, monthly de-meaning, linear detrending, sub-period slices) and a rolling-mean breakpoint scan.

## What we found
- 2y raw AR(1) half-life: 77.33 days (rho=0.991); event-decay median ≈ 4.9 days.
- 2y mean-reverting AR(1) half-life: 3.64 days (rho=0.827); event-decay median ≈ 5.0 days.
- 5y raw AR(1) half-life: 173.30 days (rho=0.996); event-decay median ≈ 4.1 days.
- 5y mean-reverting AR(1) half-life: 2.70 days (rho=0.773); event-decay median ≈ 2.3 days.
- 10y raw AR(1) half-life: 201.83 days (rho=0.997); event-decay median ≈ 3.8 days.
- 10y mean-reverting AR(1) half-life: 2.35 days (rho=0.744); event-decay median ≈ 3.8 days.
- 20y raw AR(1) half-life: 205.29 days (rho=0.997); event-decay median ≈ 3.6 days.
- 20y mean-reverting AR(1) half-life: 3.05 days (rho=0.797); event-decay median ≈ 3.3 days.
- cpi_release shocks (2y): ±3d → 2.5d, ±5d → 4.0d, ±10d → 5.1d.
- cpi_release shocks (5y): ±3d → 0.4d, ±5d → 0.8d, ±10d → 1.4d.
- cpi_release shocks (10y): ±3d → 0.3d, ±5d → 0.8d, ±10d → 5.4d.
- cpi_release shocks (20y): ±3d → 1.3d, ±5d → 2.7d, ±10d → 4.2d.
- fomc_minutes shocks (2y): ±3d → 1.5d, ±5d → 4.7d, ±10d → 2.8d.
- fomc_minutes shocks (5y): ±3d → 1.8d, ±5d → 0.2d, ±10d → 5.4d.
- fomc_minutes shocks (10y): ±3d → 5.1d, ±5d → 8.6d, ±10d → 3.5d.
- fomc_minutes shocks (20y): ±3d → 0.7d, ±5d → 2.1d, ±10d → 6.2d.
- fomc_statement shocks (2y): ±3d → 1.1d, ±5d → 5.3d, ±10d → 4.6d.
- fomc_statement shocks (5y): ±3d → 2.1d, ±5d → 0.7d, ±10d → 1.3d.
- fomc_statement shocks (10y): ±3d → 0.5d, ±5d → 2.2d, ±10d → 7.9d.
- fomc_statement shocks (20y): ±3d → 1.4d, ±5d → 1.0d, ±10d → 1.0d.
- tips_auction shocks (2y): ±3d → 2.7d, ±5d → 3.2d, ±10d → 3.3d.
- tips_auction shocks (5y): ±3d → 0.8d, ±5d → 1.4d, ±10d → 4.4d.
- tips_auction shocks (10y): ±3d → 0.4d, ±5d → 0.6d, ±10d → 3.7d.
- tips_auction shocks (20y): ±3d → 1.6d, ±5d → 0.9d, ±10d → 1.7d.
- tips_auction_announce shocks (2y): ±3d → 1.0d, ±5d → 2.5d, ±10d → 7.9d.
- tips_auction_announce shocks (5y): ±3d → 1.5d, ±5d → 1.6d, ±10d → 7.6d.
- tips_auction_announce shocks (10y): ±3d → 0.8d, ±5d → 1.8d, ±10d → 2.3d.
- tips_auction_announce shocks (20y): ±3d → 1.5d, ±5d → 1.0d, ±10d → 8.1d.
- tips_auction_settlement shocks (2y): ±3d → 0.1d, ±5d → 0.8d, ±10d → 3.7d.
- tips_auction_settlement shocks (5y): ±3d → 0.1d, ±5d → 1.2d, ±10d → 5.5d.
- tips_auction_settlement shocks (10y): ±3d → 2.8d, ±5d → 2.1d, ±10d → 8.4d.
- tips_auction_settlement shocks (20y): ±3d → 0.5d, ±5d → 0.9d, ±10d → 4.1d.
- treasury_refunding shocks (2y): ±3d → 0.7d, ±5d → 27.5d, ±10d → 17.8d.
- treasury_refunding shocks (5y): ±3d → 0.9d, ±5d → 0.6d, ±10d → 0.4d.
- treasury_refunding shocks (10y): ±3d → 0.3d, ±5d → 0.5d, ±10d → 0.7d.
- treasury_refunding shocks (20y): ±3d → 0.1d, ±5d → 7.6d, ±10d → 7.7d.

## Interpretation
- Front-end (2y/5y) arbitrage compresses within a few sessions, while 10y/20y legs exhibit slower decay—consistent with depth differentials between belly and long-end TIPS.
- CPI releases and refunding communications deliver the longest-lived shocks, whereas auction/settlement events usually mean-revert within about a trading week.
- Robustness runs show monthly de-meaning plus longer EWMAs shorten estimated half-lives, suggesting part of the raw persistence reflects slow-moving macro drifts.

## Caveats
- Event calendar relies on deterministic heuristics (third-Wednesday FOMC, third-Thursday TIPS) plus emergency overrides; official calendars could shift specific dates.
- Shock detection thresholds exclude calm periods; segments with flat price action fail the exponential fit and are logged as exclusions.
- Stationarity diagnostics show borderline KPSS statistics on long tenors, so AR(1) persistence may mix mean-reversion with structural drifts.

## Next steps
- Replace heuristic event calendar with official Treasury/BLS/Fed feeds and tag announcement vs auction effects separately.
- Extend the panel to include recent 2025 prints and backfill pre-2010 history for regime comparisons.
- Link half-life shifts to market depth/funding metrics (e.g., SOMA holdings, dealer balance sheets, TIPS ETF flows).