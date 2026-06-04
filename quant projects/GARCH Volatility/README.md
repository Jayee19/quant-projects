# Project 04: GARCH Volatility

## Goal

Model volatility clustering with a GARCH(1,1) process and estimate a dynamic one-day 99% VaR.

## Assumptions

Returns are treated as daily log returns with conditional volatility that follows a GARCH(1,1) process. The grid search restricts alpha and beta to stable values where alpha plus beta is below one. The VaR estimate uses a normal innovation assumption and the next-step conditional variance.

## Data Sources

The project uses deterministic synthetic NIFTY-style returns generated in `main.py`. The simulation embeds persistent volatility through known GARCH parameters, making it suitable for demonstrating ARCH effects and conditional volatility estimation without requiring external data.

## Methodology

The script simulates daily returns, runs an ARCH LM test on squared returns, fits GARCH(1,1) parameters by grid-search likelihood minimization, writes the conditional volatility series, and calculates next-day 99% dynamic VaR from the fitted variance.

## Results

The ARCH LM statistic is about 151.76 with an approximate p-value near 0.0, indicating strong volatility clustering. The fitted parameters are alpha 0.10, beta 0.88, persistence 0.98, and one-day 99% GARCH VaR of about 1.75%.

## What The Model Cannot Do

This model cannot fully capture jumps, leverage effects, fat-tailed innovations, regime shifts, or structural breaks in volatility. It also does not forecast liquidity stress or portfolio-level nonlinear exposures, so its VaR output should be interpreted as a conditional volatility estimate rather than a complete risk-control system.

## How To Run

```bash
python3 'quant projects/GARCH Volatility/main.py'
```

## Outputs

- `outputs/nifty_returns.csv`
- `outputs/conditional_volatility.csv`
- `outputs/arch_lm_test.csv`
- `outputs/garch_parameters.csv`
