# Project 02: Black-Scholes + Greeks

## Goal

Price European calls and puts from scratch and calculate Delta, Gamma, Theta, Vega, and Rho.

## Run

```bash
python3 'quant projects/Black-Scholes + Greeks/main.py'
```

## Outputs

- `outputs/price_vs_spot.csv`
- `outputs/greeks.csv`
- `outputs/delta_sensitivity.csv`
- `outputs/put_call_parity.csv`

## What This Shows

- Black-Scholes pricing logic
- Put-call parity as a sanity check
- How Greeks change with spot, strike, volatility, and maturity

## Limitation

The model assumes constant volatility, continuous hedging, and frictionless markets.
