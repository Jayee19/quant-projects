# Project 09: Monte Carlo + Variance Reduction

## Goal

Price options with Monte Carlo simulation and compare standard simulation, antithetic variates, and control variates.

## Run

```bash
python3 'quant projects/Monte Carlo Option Pricing with Variance Reduction/main.py'
```

## Outputs

- `outputs/pricing_summary.csv`
- `outputs/convergence_analysis.csv`

## What This Shows

- GBM path simulation
- European call pricing versus Black-Scholes
- Asian option pricing
- Variance reduction and convergence analysis

## Limitation

The model uses plain GBM and does not include stochastic volatility, jumps, or calibration to a volatility surface.
