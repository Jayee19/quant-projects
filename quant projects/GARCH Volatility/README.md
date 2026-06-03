# Project 04: GARCH(1,1) Volatility

## Goal

Model volatility clustering and use conditional volatility for a dynamic one-day VaR forecast.

## Run

```bash
python3 'quant projects/GARCH Volatility/main.py'
```

## Outputs

- `outputs/nifty_returns.csv`
- `outputs/arch_lm_test.csv`
- `outputs/conditional_volatility.csv`
- `outputs/garch_parameters.csv`

## What This Shows

- ARCH effects and volatility clustering
- GARCH parameters: omega, alpha, beta
- Persistence as alpha plus beta

## Limitation

The transparent grid fit is useful for learning, but production GARCH should use full maximum-likelihood optimization and residual diagnostics.
