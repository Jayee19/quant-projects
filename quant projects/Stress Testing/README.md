# Project 07: Portfolio Stress Testing

## Goal

Apply historical and hypothetical shocks to a Rs.5 crore equity portfolio and attribute losses by position.

## Run

```bash
python3 'quant projects/Stress Testing/main.py'
```

## Outputs

- `outputs/current_positions.csv`
- `outputs/scenario_position_pnl.csv`
- `outputs/scenario_summary.csv`
- `outputs/largest_loss_driver.csv`
- `outputs/assumptions.csv`

## What This Shows

- Historical crisis replay
- Hypothetical scenario design
- P&L attribution under stress

## Limitation

The stress test is linear and does not include liquidity gaps, transaction costs, or derivative convexity.
