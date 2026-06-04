# Project 02: Black-Scholes + Greeks

## Goal

Price European call and put options from scratch and calculate Delta, Gamma, Theta, Vega, and Rho.

## Assumptions

The base case uses spot 100, strike 100, risk-free rate 6.5%, volatility 22%, and maturity of 0.5 years. Options are European style, volatility and interest rates are constant, markets are frictionless, and the underlying follows geometric Brownian motion.

## Data Sources

This project does not use external market data. It uses deterministic model inputs defined in `main.py`, then generates pricing grids, Greek tables, delta sensitivity, and put-call parity checks from the Black-Scholes formula.

## Methodology

The script implements closed-form Black-Scholes pricing for calls and puts and calculates the five main Greeks analytically. It also prices options across multiple spot and strike scenarios, tests delta sensitivity across volatility levels, and checks put-call parity as a numerical sanity check.

## Results

In the base case, the call price is about 7.83 and the put price is about 4.63. The call Delta is about 0.613, Gamma is about 0.0246, Vega per 1% volatility move is about 0.271, and the put-call parity gap is 0.0, confirming internal pricing consistency.

## What The Model Cannot Do

This model cannot price American exercise, discrete dividends, jumps, stochastic volatility, volatility smiles, market frictions, or liquidity effects. It is best used as a clean analytical benchmark, not as a complete production options pricing model for real market books.

## How To Run

```bash
python3 'quant projects/Black-Scholes + Greeks/main.py'
```

## Outputs

- `outputs/price_vs_spot.csv`
- `outputs/greeks.csv`
- `outputs/delta_sensitivity.csv`
- `outputs/put_call_parity.csv`
