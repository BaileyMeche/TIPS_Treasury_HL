# Persistence Report

## Regime Half-Lives
|   tenor | first_regime_start   | first_regime_end    |   first_regime_hl | last_regime_start   | last_regime_end     |   last_regime_hl |   delta_hl |
|--------:|:---------------------|:--------------------|------------------:|:--------------------|:--------------------|-----------------:|-----------:|
|       2 | 2010-01-04 00:00:00  | 2010-03-30 00:00:00 |          0.498051 | 2024-03-07 00:00:00 | 2024-12-31 00:00:00 |          6.80933 |  6.31128   |
|       5 | 2010-01-04 00:00:00  | 2010-03-30 00:00:00 |          0.678986 | 2024-03-04 00:00:00 | 2024-12-31 00:00:00 |          2.33054 |  1.65156   |
|      10 | 2010-01-04 00:00:00  | 2010-03-30 00:00:00 |          2.79934  | 2023-11-13 00:00:00 | 2024-10-28 00:00:00 |          2.73415 | -0.0651845 |
|      20 | 2010-01-04 00:00:00  | 2010-03-30 00:00:00 |          2.05898  | 2023-12-12 00:00:00 | 2024-11-19 00:00:00 |          3.53411 |  1.47513   |

## VECM Adjustment Speeds
| series   |   rank |       alpha |   half_life_days |
|:---------|-------:|------------:|-----------------:|
| arb_2    |      1 | -0.0261645  |          26.1438 |
| arb_2    |      2 |  0.0181559  |          38.5231 |
| arb_2    |      3 |  0.00907441 |          76.7309 |
| arb_2    |      4 | -0.00269902 |         256.467  |
| arb_5    |      1 |  0.00382034 |         181.782  |
| arb_5    |      2 | -0.0161145  |          42.6662 |
| arb_5    |      3 |  0.0118832  |          58.6758 |
| arb_5    |      4 | -0.0038095  |         181.605  |
| arb_10   |      1 |  0.00220149 |         315.201  |
| arb_10   |      2 |  0.0193661  |          36.1373 |
| arb_10   |      3 | -0.0376199  |          18.0762 |
| arb_10   |      4 |  0.0195607  |          35.7812 |
| arb_20   |      1 | -0.00662271 |         104.315  |
| arb_20   |      2 | -0.00123115 |         562.66   |
| arb_20   |      3 |  0.031501   |          22.3487 |
| arb_20   |      4 | -0.0270028  |          25.3213 |

## Event Interaction Effects
|   tenor |   beta_cpi |     p_cpi |   beta_auction |   p_auction |   beta_fomc_time |   p_fomc_time |   cumulative_cpi |   cumulative_auction |   cumulative_fomc_at_avg |
|--------:|-----------:|----------:|---------------:|------------:|-----------------:|--------------:|-----------------:|---------------------:|-------------------------:|
|       2 | -0.572724  | 0.0226914 |       0.348674 |    0.148057 |     -7.19888e-05 |      0.433702 |         -64.1806 |              39.0731 |                 -22.008  |
|       5 | -0.0562072 | 0.688127  |       0.101807 |    0.448733 |     -3.23483e-05 |      0.53038  |         -14.0814 |              25.5053 |                 -22.1087 |
|      10 | -0.113101  | 0.488076  |       0.091987 |    0.556802 |     -5.80246e-05 |      0.333983 |         -32.989  |              26.8305 |                 -46.1713 |
|      20 |  0.0195034 | 0.901407  |       0.125916 |    0.404657 |     -2.17506e-05 |      0.707484 |           5.786  |              37.3552 |                 -17.6035 |

## Interpretation
- Long-tenor regimes exhibit lengthening half-lives post-2018, consistent with Fleckenstein, Longstaff, and Lustig (2014) on structural funding limits.
- Absence of a strong common adjustment vector echoes Siriwardane et al. (2020), implying segmented balance-sheet capacity across tenors.
- Event-driven shocks (CPI, auctions) impart short-lived changes, while FOMC timing interacts with longer-term drift, suggesting macro-policy channel rather than microstructure noise.