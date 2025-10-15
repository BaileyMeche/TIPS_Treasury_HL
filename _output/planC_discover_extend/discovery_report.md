# Discovery Report

## Trading Gaps
Identified gaps larger than five days across tenors:
|   tenor |   count |   max |
|--------:|--------:|------:|
|       2 |       4 |     7 |

## Low-Volatility Plateaus
Detected extended periods with low 20-day rolling volatility (bottom decile).
|   tenor |   count |   max |    mean |
|--------:|--------:|------:|--------:|
|       2 |      39 |    78 | 12.6667 |
|       5 |      42 |    56 | 11.4286 |
|      10 |      40 |    43 | 12.05   |
|      20 |      34 |    97 | 14.6176 |

## Event Decay Fit Shortfalls
Event-type/window combinations with fewer than five valid half-life estimates across tenors:
| event_type              |   window |   n_valid |
|:------------------------|---------:|----------:|
| cpi_release             |        1 |         0 |
| cpi_release             |        3 |         4 |
| cpi_release             |        5 |         4 |
| cpi_release             |       10 |         4 |
| fomc_minutes            |        1 |         0 |
| fomc_minutes            |        3 |         4 |
| fomc_minutes            |        5 |         4 |
| fomc_minutes            |       10 |         4 |
| fomc_statement          |        1 |         0 |
| fomc_statement          |        3 |         4 |
| fomc_statement          |        5 |         4 |
| fomc_statement          |       10 |         4 |
| tips_auction            |        1 |         0 |
| tips_auction            |        3 |         4 |
| tips_auction            |        5 |         4 |
| tips_auction            |       10 |         4 |
| tips_auction_announce   |        1 |         0 |
| tips_auction_announce   |        3 |         4 |
| tips_auction_announce   |        5 |         4 |
| tips_auction_announce   |       10 |         4 |
| tips_auction_settlement |        1 |         0 |
| tips_auction_settlement |        3 |         4 |
| tips_auction_settlement |        5 |         4 |
| tips_auction_settlement |       10 |         4 |
| treasury_refunding      |        1 |         0 |
| treasury_refunding      |        3 |         4 |
| treasury_refunding      |        5 |         4 |
| treasury_refunding      |       10 |         4 |

## AR(1) vs Event Half-Life Comparison
|   tenor |      rho |   hl_ar1_days |   hl_event_days | flag_inconsistent   |
|--------:|---------:|--------------:|----------------:|:--------------------|
|       2 | 0.991076 |       77.3284 |         4.92441 | True                |
|       5 | 0.996008 |      173.305  |         4.12687 | True                |
|      10 | 0.996572 |      201.828  |         3.76124 | True                |
|      20 | 0.996629 |      205.287  |         3.62832 | True                |

## Breakpoint Alignment
Breakpoints coinciding with events within ±2 days (by type):
| event_type_list         |   matching_breaks |
|:------------------------|------------------:|
| tips_auction_announce   |               306 |
| tips_auction_settlement |               287 |
| tips_auction            |               263 |
| fomc_statement          |               253 |
| cpi_release             |               239 |
| fomc_minutes            |               186 |
| treasury_refunding      |                74 |

## Stationarity Diagnostics
|   tenor |   adf_stat |       adf_p |   kpss_stat |    kpss_p |
|--------:|-----------:|------------:|------------:|----------:|
|       2 |   -7.0055  | 7.1396e-10  |   0.0666349 | 0.1       |
|       5 |   -5.20622 | 8.53856e-06 |   1.03598   | 0.01      |
|      10 |   -4.2147  | 0.000622143 |   0.357284  | 0.0955673 |
|      20 |   -4.63325 | 0.00011212  |   2.54593   | 0.01      |

## Recommendations
- Supplement low-volatility regimes with liquidity measures (e.g., TRACE ATS flags) to distinguish calm funding from data gaps.
- Extend event decay models to incorporate auction settlement and refunding sequences, which currently lack stable fits.
- Gather repo and futures positioning data to reconcile AR(1) persistence with short-lived event half-lives, especially where ρ ≈ 1.
- Tie breakpoint detection to exact event timestamps (CPI release time, FOMC statements) to confirm causality.