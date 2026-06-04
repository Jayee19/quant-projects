# Project 09: Monte Carlo Option Pricing with Variance Reduction

## Goal

Price options with Monte Carlo simulation and compare standard sampling against antithetic variates and a control variate.

## Assumptions

The base case uses spot 100, strike 105, risk-free rate 5.5%, volatility 24%, and one-year maturity. The underlying follows geometric Brownian motion, discounting uses a constant risk-free rate, and simulations use fixed random seeds for reproducibility.

## Data Sources

This project does not use external data. All inputs are deterministic model parameters in `main.py`, and the benchmark European call price comes from the Black-Scholes formula.

## Methodology

The script simulates terminal prices for a European call, estimates discounted payoffs, and compares standard Monte Carlo, antithetic variates, and a stock-price control variate against the Black-Scholes benchmark. It also prices an Asian call with path simulation and records convergence across multiple path counts.

## Results

The Black-Scholes benchmark call price is about 9.835. Standard Monte Carlo estimates about 9.975 with standard error about 0.117, antithetic variates estimate about 9.901 with smaller absolute error, and the control variate estimates about 9.841 with standard error about 0.051 and absolute error about 0.006. The Asian option estimate is about 4.509.

## What The Model Cannot Do

This model cannot price real exotic books without richer dynamics, calibration, variance reduction design, path-dependent state handling, and validation against market prices. It assumes constant volatility and rates, ignores dividends and transaction costs, and does not address model risk from stochastic volatility, jumps, or early exercise.

## How To Run

```bash
python3 'quant projects/Monte Carlo Option Pricing with Variance Reduction/main.py'
```

## Outputs

- `outputs/pricing_summary.csv`
- `outputs/convergence_analysis.csv`
