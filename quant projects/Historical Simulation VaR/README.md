# Project 01: Historical Simulation VaR Calculator

## Goal

Build a five-stock portfolio and estimate daily loss risk using historical VaR, Expected Shortfall, and parametric VaR.

## Run

```bash
python3 'quant projects/Historical Simulation VaR/main.py'
```

## Outputs

- `outputs/var_summary.csv`
- `outputs/portfolio_returns.csv`
- `outputs/assumptions.csv`

## What This Shows

- VaR as a quantile of historical portfolio returns
- Expected Shortfall as the average loss beyond the VaR threshold
- Why historical and parametric VaR can disagree

## Limitation

VaR does not explain how severe losses can be after the threshold is breached.
