# Appendix: Data Dictionary

## Inputs
- **data/trace_microstructure_event_panels.csv** (available, rows=38): columns = event_date, event_type, details, tenor, total_volume, ats_share, principal_share, dealer_hhi, trade_count, dealer_count
- **data/tenor_liq.csv** (available, rows=1680): columns = qdate, tenor_bucket, bid_ask_spread, pubout, n_issues, liq_hhi, issue_conc_top3, issue_conc_top5
- **_output/strategy3/state_estimates.csv** (available, rows=15648): columns = date, tenor, mu_smoothed, mu_filtered, epsilon_smoothed, epsilon_filtered, regime_0_prob, regime_1_prob
- **data/policy/treasury_buybacks_refunding.csv** (available, rows=5476): columns = date, buyback_dummy, refunding_dummy
- **data/val/bei_ils_wedge_by_tenor.csv** (available, rows=11676): columns = date, bei_tenor, bei_rate, tenor_years, ils_tenor, ils_rate, bei_minus_ils
- **_output/strategy3/variance_decomposition.csv** (available, rows=5): columns = # Variance share of structural mean vs deviation components
- **_output/strategy3/halflife_summary.csv** (available, rows=13): columns = # Half-life comparison between state-space deviations and Markov regimes

## Outputs
- **reports/microstructure_concentration_results.html** (pending)
- **reports/microstructure_concentration_results.csv** (pending)
- **reports/policy_intervention_state_space.html** (pending)
- **reports/policy_intervention_coeffs.csv** (pending)
- **reports/val_wedge_linkage.html** (pending)
- **reports/val_wedge_linkage.csv** (pending)
- **reports/forecast_comparison.html** (pending)
- **reports/forecast_rmsfe.csv** (pending)
- **reports/event_irfs_daily.html** (pending)
- **reports/event_irfs_daily.csv** (pending)
- **reports/strategy3_state_space.html** (available)
- **tables/state_space_variance.csv** (pending)
- **reports/appendix_data_dictionary.md** (available)
- **exports/analysis_artifacts.yml** (available)