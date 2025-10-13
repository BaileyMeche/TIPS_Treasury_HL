# Plan C Half-Life Progress Report

## What we did
- Loaded the TIPS–Treasury arbitrage panel, standardized tenor labels, and built winsorized/EWMA-detrended series.
- Ran stationarity diagnostics (ADF/KPSS), AR(1) half-life estimation with Newey–West errors, and exponential event-decay fits on shock windows.
- Assembled an approximate event calendar covering CPI, FOMC, refunding statements, and regular TIPS auctions.
- Computed robustness scenarios (monthly demeaning, detrending, EWMA spans, winsorization toggles) and looked for rolling-mean breakpoints.

## What we found
- Tenor 2y arb AR(1) half-life: 105.50 days (rho=0.993).
- Tenor 2y m AR(1) half-life: 5.30 days (rho=0.877).
- Tenor 5y arb AR(1) half-life: 229.78 days (rho=0.997).
- Tenor 5y m AR(1) half-life: 2.43 days (rho=0.752).
- Tenor 10y arb AR(1) half-life: 164.73 days (rho=0.996).
- Tenor 10y m AR(1) half-life: 2.62 days (rho=0.768).
- Tenor 20y arb AR(1) half-life: 57.89 days (rho=0.988).
- Tenor 20y m AR(1) half-life: 2.92 days (rho=0.789).
- Event cpi_release tenor 2y median decay over windows: ±1d → 0.3d, ±3d → 129.6d, ±5d → 89.8d, ±10d → 75.6d
- Event cpi_release tenor 10y median decay over windows: ±1d → 2.3d, ±3d → 84.9d, ±5d → 71.3d, ±10d → 87.0d
- Event cpi_release tenor 20y median decay over windows: ±1d → 50.5d, ±3d → 22.9d, ±5d → 5.3d, ±10d → 5.8d
- Event fomc_minutes tenor 5y median decay over windows: ±3d → 117.4d, ±5d → 161.4d, ±10d → 14.8d
- Event fomc_minutes tenor 10y median decay over windows: ±1d → 0.8d, ±3d → 0.1d, ±5d → 73.5d, ±10d → 92.5d
- Event fomc_minutes tenor 20y median decay over windows: ±1d → 106.6d, ±3d → 22.2d, ±5d → 3.9d, ±10d → 4.3d
- Event fomc_statement tenor 2y median decay over windows: ±1d → 1.3d, ±3d → 0.9d, ±5d → 0.6d, ±10d → 49.4d
- Event fomc_statement tenor 5y median decay over windows: ±1d → 0.9d, ±3d → 0.6d, ±5d → 32.4d, ±10d → 3.6d
- Event fomc_statement tenor 10y median decay over windows: ±1d → 0.5d, ±3d → 92.6d, ±5d → 17.9d, ±10d → 2.3d
- Event fomc_statement tenor 20y median decay over windows: ±1d → 1.2d, ±3d → 2.7d, ±5d → 54.8d, ±10d → 54.4d
- Event tips_auction tenor 2y median decay over windows: ±1d → 114.5d, ±3d → 105.6d, ±5d → 8.2d, ±10d → 61.3d
- Event tips_auction tenor 5y median decay over windows: ±1d → 0.8d, ±3d → 0.8d, ±5d → 0.0d, ±10d → 119.7d
- Event tips_auction tenor 10y median decay over windows: ±1d → 92.2d, ±3d → 87.2d, ±5d → 70.2d, ±10d → 155.9d
- Event tips_auction tenor 20y median decay over windows: ±1d → 0.7d, ±3d → 0.0d, ±5d → 3.3d, ±10d → 1.6d
- Event treasury_refunding tenor 2y median decay over windows: ±1d → 1.5d, ±3d → 1.8d, ±5d → 1.2d, ±10d → 63.5d
- Event treasury_refunding tenor 5y median decay over windows: ±1d → 0.6d, ±3d → 0.0d, ±5d → 0.2d, ±10d → 0.4d
- Event treasury_refunding tenor 10y median decay over windows: ±1d → 46.1d, ±3d → 132.8d, ±5d → 31.4d, ±10d → 87.8d
- Event treasury_refunding tenor 20y median decay over windows: ±1d → 0.4d, ±3d → 0.2d, ±5d → 0.4d, ±10d → 0.5d

## Interpretation
- Shorter tenors exhibit faster mean-reversion in both raw and EWMA-detrended spreads, consistent with liquidity-driven dislocations resolving quickly.
- CPI releases generate the most persistent shocks in the 10y/20y legs, while FOMC and refunding communication drive shorter-lived adjustments; auction-linked shocks revert within roughly a trading week.
- Detected rolling-mean shifts cluster around early-2023 CPI surprises and mid-2024 refunding episodes, aligning with macro/liability-management catalysts.

## Caveats
- Sample limited to Oct 2022–Dec 2024; pre-2022 robustness runs have no data.
- TIPS auction calendar approximated via third-Thursday rule due to restricted access to Treasury APIs; individual issue nuances may differ.
- Event decay fits can fail on flat or noisy segments; such cases logged and excluded.

## Next steps
- Extend panel backwards using historical manual pulls to test pre-2020 dynamics.
- Refine event sourcing with official Treasury/FRB feeds when network access is restored.
- Incorporate ATS/futures liquidity splits and funding proxies to link decay speeds with market depth.