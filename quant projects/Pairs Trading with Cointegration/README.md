# Project 03: Pairs Trading with Cointegration

## Goal

Find cointegrated pairs, estimate a rolling hedge ratio, and backtest a z-score mean-reversion strategy.

## Run

```bash
python3 'quant projects/Pairs Trading with Cointegration/main.py'
```

## Outputs

- `outputs/cointegration_tests.csv`
- `outputs/backtest.csv`
- `outputs/performance_metrics.csv`

## What This Shows

- Difference between correlation and cointegration
- Rolling OLS hedge-ratio estimation
- Entry and exit logic using spread z-scores

## Limitation

The fallback cointegration test is approximate. A full research version should use `statsmodels` and transaction-cost modelling.
