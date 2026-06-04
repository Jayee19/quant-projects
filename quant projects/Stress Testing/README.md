# Project 07: Stress Testing

## Goal

Stress test a five-stock Indian equity portfolio under historical-style and hypothetical shock scenarios.

## Assumptions

The portfolio notional is INR 50,000,000 with fixed weights in RELIANCE.NS, TCS.NS, HDFCBANK.NS, INFY.NS, and ICICIBANK.NS. Scenario shocks are applied linearly to position notionals, and PnL is calculated as notional times shocked return. No rebalancing, hedging, liquidity adjustment, or transaction cost is applied.

## Data Sources

The project attempts to load recent adjusted close prices through `yfinance` and falls back to deterministic synthetic equity prices when live data is unavailable. Scenario shocks are hard-coded in `main.py` and are grouped into historical-style events and hypothetical stress cases.

## Methodology

The script builds current position notionals, applies each scenario shock at the ticker level, aggregates portfolio PnL, calculates portfolio return under each stress, and identifies the largest loss driver for every scenario.

## Results

The largest loss is the 2008 global financial crisis scenario, with portfolio PnL of about INR -15.47 million and return of about -30.94%. The COVID crash scenario loses about INR -12.12 million, and the hypothetical rate spike plus equity selloff loses about INR -11.67 million. RELIANCE.NS is the largest loss driver in most scenarios, while TCS.NS drives the IT margin squeeze scenario.

## What The Model Cannot Do

This model cannot capture nonlinear instruments, option convexity, intraday gaps, liquidity haircuts, funding stress, margin calls, or second-order contagion effects. The scenario shocks are illustrative rather than calibrated to a full macro risk engine, so the results are useful for portfolio stress storytelling but not sufficient for enterprise stress testing.

## How To Run

```bash
python3 'quant projects/Stress Testing/main.py'
```

## Outputs

- `outputs/current_positions.csv`
- `outputs/scenario_position_pnl.csv`
- `outputs/scenario_summary.csv`
- `outputs/largest_loss_driver.csv`
- `outputs/assumptions.csv`
