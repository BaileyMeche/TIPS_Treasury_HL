# Next Data Targets

## Feature Enhancements
- TRACE ATS flags to proxy non-dealer liquidity and differentiate dark pool activity during low-volatility regimes.
- Federal Reserve H.4.1 repo balances and SOFR-OIS spreads to gauge funding stress around breakpoints.
- TIPS on-the-run indicators and WI auction data to contextualize auction-driven shocks.

## Suggested WRDS Queries
```sql
SELECT trade_dt, cusip_id, ats_indicator, volume
FROM trace.enhanced_trade
WHERE trade_dt BETWEEN '2010-01-01' AND '2024-12-31'
  AND security_type = 'TIPS';
```

```python
import wrds
db = wrds.Connection()
repo = db.raw_sql("""
SELECT asofdate, primary_credit_borrowing, other_credit_extensions
FROM frb.h41
WHERE asofdate BETWEEN '2010-01-01' AND '2024-12-31';
""")
```

## Immediate Computations
- Merge TRACE ATS participation rates with detected low-volatility segments to test whether dealer internalization drives calm periods.
- Overlay repo spread data onto breakpoint chronology to confirm funding-driven regime shifts.
- Compute on/off-the-run spread differentials using Treasury WI auction files to isolate liquidity-driven arbitrage persistence.