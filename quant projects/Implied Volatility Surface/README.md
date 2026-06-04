# Project 08: Implied Volatility Surface

## Goal

Construct an implied volatility surface by inverting option prices across strikes and maturities.

## Assumptions

The base market uses spot 100 and risk-free rate 5.5%. Strikes range from 75 to 125, maturities range from 0.08 to 2.0 years, and synthetic market prices are generated from a known volatility smile and skew. Below-spot strikes use puts and at-the-money or above-spot strikes use calls.

## Data Sources

This project does not use external option-chain data. It uses deterministic synthetic implied volatilities generated in `main.py`, prices options with Black-Scholes, then recovers implied volatility through bisection.

## Methodology

The script generates synthetic market volatility for each strike and maturity, converts it into a market option price, and then solves for implied volatility from that price. It writes the full surface and a compact summary of minimum volatility, maximum volatility, at-the-money one-year volatility, and one-year skew.

## Results

The committed surface has minimum implied volatility of about 18.71% and maximum implied volatility of about 26.41%. The one-year at-the-money implied volatility is about 20.50%, while the one-year 75-strike implied volatility is about 25.38%, giving a low-strike minus ATM skew of about 4.88 percentage points.

## What The Model Cannot Do

This model cannot infer a real exchange-traded volatility surface without real option-chain quotes, bid-ask filtering, arbitrage checks, dividend assumptions, and liquidity controls. It demonstrates inversion mechanics and surface summarization, but it does not enforce full no-arbitrage constraints or handle noisy production market data.

## How To Run

```bash
python3 'quant projects/Implied Volatility Surface/main.py'
```

## Outputs

- `outputs/implied_vol_surface.csv`
- `outputs/surface_summary.csv`
