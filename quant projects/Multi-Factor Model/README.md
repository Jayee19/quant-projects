# Project 05: Multi-Factor Model

## Goal

Build a monthly equity factor model using momentum, reversal, value, and quality signals, then evaluate long-short performance and information coefficient.

## Assumptions

The universe contains 12 Indian equity tickers. Momentum, reversal, value, and quality are converted to percentile ranks and combined into an equal-weight composite score. Each month, the strategy goes long the top three names and short the bottom three names. Static synthetic fundamentals are used for value and quality.

## Data Sources

The price loader attempts to use `yfinance` adjusted close data. If live data is unavailable, it uses deterministic synthetic equity paths. The committed output was generated from synthetic fallback data, and the value and quality fundamentals are deterministic synthetic series generated in `main.py`.

## Methodology

The script resamples prices to month-end, computes monthly returns and forward returns, builds factor ranks, creates a composite score, and records monthly long and short books. It evaluates the long-short return stream with annualised Sharpe and measures rank correlation between factor scores and next-month returns using information coefficient.

## Results

The committed run produced an annualised Sharpe of about -0.30, mean monthly return of about -0.38%, mean information coefficient of about 0.0065, and IC information ratio of about 0.023. These results show the workflow and diagnostics, but the synthetic sample does not produce a strong positive factor edge.

## What The Model Cannot Do

This model cannot prove that the factors have persistent real-market alpha. It ignores transaction costs, turnover limits, borrow constraints, liquidity, sector neutrality, risk-model constraints, realistic fundamentals, and out-of-sample validation across regimes. It is a research scaffold for factor testing, not a deployable portfolio construction engine.

## How To Run

```bash
python3 'quant projects/Multi-Factor Model/main.py'
```

## Outputs

- `outputs/monthly_long_short_backtest.csv`
- `outputs/monthly_factor_scores.csv`
- `outputs/factor_summary.csv`
