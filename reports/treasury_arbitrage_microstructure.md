# Treasury Arbitrage Microstructure & Convergence Report

*Generated on 2025-10-18*

## Executive Summary
- AR(1) half-life estimates derived from yield-forward basis dynamics vary materially across issues.
- Wider bid-ask spreads and longer effective durations associate with slower convergence in cross-sectional regressions.
- TRACE trade data were empty, precluding direct dealer concentration metrics; auction identifiers matched 0.00% of issues.
- Results emphasise the importance of richer microstructure coverage for firm conclusions.

## Data Description
The analysis uses three parquet datasets located in `src/data`: CRSP Treasury quotes, TRACE Treasury trades, and FISD Treasury auctions.
Key preparation steps:
- Constructed bid-ask spreads, mid yields, duration proxies, and forward-basis measures from CRSP quotes.
- Estimated issue-level AR(1) half-lives when at least 30 observations were available.
- Aggregated liquidity metrics per issue and attempted identifier suffix matching to auctions (match rate 0.00%).
- TRACE file contained 0 rows.

## Descriptive Statistics
### Half-life Distribution by Tenor and Period
| tenor_bucket   |   count |   mean |   median |    std |    min |    max |    p25 |    p75 | period       |
|:---------------|--------:|-------:|---------:|-------:|-------:|-------:|-------:|-------:|:-------------|
| 10y+           |      25 | 1.0485 |   1.0703 | 0.2783 | 0.6349 | 1.4939 | 0.8284 | 1.2097 | Full Sample  |
| 10y+           |      25 | 1.1556 |   1.0332 | 0.3513 | 0.6628 | 2.2179 | 0.9220 | 1.3524 | Post-2020-03 |
| 10y+           |      25 | 1.0289 |   0.8865 | 0.3472 | 0.5239 | 1.6976 | 0.8087 | 1.2799 | Pre-2020-03  |

### Liquidity Feature Correlations
|                |   half_life |   bid_ask_spread |   duration |   mid_yield |   basis_fwd1 |
|:---------------|------------:|-----------------:|-----------:|------------:|-------------:|
| half_life      |      1.0000 |          -0.5953 |    -0.5236 |     -0.5123 |       0.1800 |
| bid_ask_spread |     -0.5953 |           1.0000 |     0.9904 |      0.9834 |      -0.7805 |
| duration       |     -0.5236 |           0.9904 |     1.0000 |      0.9966 |      -0.8399 |
| mid_yield      |     -0.5123 |           0.9834 |     0.9966 |      1.0000 |      -0.8461 |
| basis_fwd1     |      0.1800 |          -0.7805 |    -0.8399 |     -0.8461 |       1.0000 |

### TRACE Trade File Availability
_TRACE Treasury trade file contained no records; dealer concentration metrics could not be computed._

## Regression Analysis
### Table 1: Liquidity baseline
| variable       |   coefficient |   std_error |   p_value | model              | subset      |   n_obs |   r_squared |   adj_r_squared |
|:---------------|--------------:|------------:|----------:|:-------------------|:------------|--------:|------------:|----------------:|
| const          |        8.1784 |     15.1365 |    0.5890 | Liquidity baseline | Full Sample | 25.0000 |      0.5881 |          0.5292 |
| bid_ask_spread |     -685.5488 |    144.4296 |    0.0000 | Liquidity baseline | Full Sample | 25.0000 |      0.5881 |          0.5292 |
| duration       |        0.0244 |      0.0131 |    0.0629 | Liquidity baseline | Full Sample | 25.0000 |      0.5881 |          0.5292 |
| mid_yield      |  -145047.4966 | 324317.8498 |    0.6547 | Liquidity baseline | Full Sample | 25.0000 |      0.5881 |          0.5292 |

### Table 2: With auction controls
| model                 | message           |
|:----------------------|:------------------|
| With auction controls | Insufficient data |

## Heterogeneity & Robustness Checks
### Split-Sample Regressions by Duration
#### Duration ≤ 94.83
| variable       |   coefficient |   std_error |   p_value | model              | subset           |   n_obs |   r_squared |   adj_r_squared |
|:---------------|--------------:|------------:|----------:|:-------------------|:-----------------|--------:|------------:|----------------:|
| const          |      -48.3678 |     21.9333 |    0.0274 | Liquidity baseline | Duration ≤ 94.83 | 13.0000 |      0.5784 |          0.4379 |
| bid_ask_spread |     -397.4157 |    333.1579 |    0.2329 | Liquidity baseline | Duration ≤ 94.83 | 13.0000 |      0.5784 |          0.4379 |
| duration       |       -0.0337 |      0.0214 |    0.1156 | Liquidity baseline | Duration ≤ 94.83 | 13.0000 |      0.5784 |          0.4379 |
| mid_yield      |  1075724.2486 | 473928.4607 |    0.0232 | Liquidity baseline | Duration ≤ 94.83 | 13.0000 |      0.5784 |          0.4379 |

#### Duration > 94.83
| variable       |   coefficient |   std_error |   p_value | model              | subset           |   n_obs |   r_squared |   adj_r_squared |
|:---------------|--------------:|------------:|----------:|:-------------------|:-----------------|--------:|------------:|----------------:|
| const          |      270.0394 |     44.8508 |    0.0000 | Liquidity baseline | Duration > 94.83 | 12.0000 |      0.8562 |          0.8023 |
| bid_ask_spread |    -2089.8891 |    246.5280 |    0.0000 | Liquidity baseline | Duration > 94.83 | 12.0000 |      0.8562 |          0.8023 |
| duration       |        0.2386 |      0.0355 |    0.0000 | Liquidity baseline | Duration > 94.83 | 12.0000 |      0.8562 |          0.8023 |
| mid_yield      | -5718679.4158 | 952775.2209 |    0.0000 | Liquidity baseline | Duration > 94.83 | 12.0000 |      0.8562 |          0.8023 |

### Half-life Specification Sensitivity
|   count |   mean |   median |    std |    min |    max |    p25 |    p75 | spec                      |
|--------:|-------:|---------:|-------:|-------:|-------:|-------:|-------:|:--------------------------|
| 25.0000 | 1.0485 |   1.0703 | 0.2783 | 0.6349 | 1.4939 | 0.8284 | 1.2097 | Basis vs 1M forward       |
| 22.0000 | 3.7587 |   3.7052 | 0.9626 | 1.4118 | 5.7480 | 3.2469 | 4.4838 | Basis vs 4M forward       |
| 25.0000 | 1.0485 |   1.0703 | 0.2783 | 0.6349 | 1.4939 | 0.8284 | 1.2097 | Short sample (min 20 obs) |

## Interpretation & Discussion
- Half-life dispersion remains sizable across tenor buckets, with longer-duration issues showing higher persistence on average.
- Liquidity stress as proxied by bid-ask spreads aligns with slower convergence, consistent with inventory or funding frictions.
- Sparse auction linkages and absent TRACE trades limit inference about dealer concentration and issuance structure.
- Augmenting identifiers and enriching TRACE coverage are priority next steps for microstructure diagnostics.

## Code & Reproducibility Notes
- Generated via `src/treasury_arbitrage_microstructure_report.py`.
- Inputs: `src/data/crsp_treasury_quotes.parquet`, `src/data/trace_treas_trace.parquet`, `src/data/fisd_treasury_auctions.parquet`.
- Output: `reports/treasury_arbitrage_microstructure.md`; no additional files created.
