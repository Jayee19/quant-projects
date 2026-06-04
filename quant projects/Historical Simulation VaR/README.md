# Project 01: Historical Simulation VaR Calculator

## Goal

Estimate daily downside risk for a five-stock Indian equity portfolio using historical VaR, Expected Shortfall, and parametric VaR.

## Assumptions

The portfolio uses RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, and ICICIBANK.NS with weights of 25%, 20%, 20%, 20%, and 15%. Returns are daily percentage returns, portfolio weights are fixed through the sample, and VaR is reported as a positive loss number. The analysis assumes the historical return sample is a useful proxy for near-term risk.

## Data Sources

The project attempts to download adjusted close prices through `yfinance`. If live market data is unavailable, it uses deterministic synthetic equity price paths with market, sector, idiosyncratic, crisis, and recovery components. The committed outputs were generated from the synthetic fallback data, as recorded in `outputs/assumptions.csv`.

## Methodology

The script converts prices to daily returns, computes weighted portfolio returns, and estimates VaR at 90%, 95%, and 99% confidence levels. Historical VaR uses empirical quantiles, Expected Shortfall averages losses beyond the VaR threshold, and parametric VaR uses the normal approximation from the portfolio mean and standard deviation.

## Results

At 95% confidence, the daily historical VaR is about 1.89%, Expected Shortfall is about 2.36%, and parametric VaR is about 1.97%. At 99% confidence, historical VaR rises to about 2.60% and Expected Shortfall rises to about 3.08%, showing that tail-loss averages are meaningfully larger than the VaR cutoff itself.

## What The Model Cannot Do

This model cannot predict future crashes or explain the full severity of losses once the VaR threshold is breached. It treats the historical sample as representative, ignores intraday liquidity, transaction costs, changing portfolio weights, and nonlinear exposures, and can understate risk when future market regimes are different from the observed or simulated sample.

## How To Run

```bash
python3 'quant projects/Historical Simulation VaR/main.py'
```

## Outputs

- `outputs/var_summary.csv`
- `outputs/portfolio_returns.csv`
- `outputs/assumptions.csv`
