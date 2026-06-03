# Project 08: Implied Volatility Surface

## Goal

Generate a synthetic options market and back out implied volatility across strikes and maturities.

## Run

```bash
python3 'quant projects/Implied Volatility Surface/main.py'
```

## Outputs

- `outputs/implied_vol_surface.csv`
- `outputs/surface_summary.csv`

## What This Shows

- Black-Scholes inversion by bisection
- Smile, skew, and term structure
- Why lower-strike equity puts often have higher implied volatility

## Limitation

The market prices are synthetic. A trading-desk version should ingest real option chains and clean bad quotes.
