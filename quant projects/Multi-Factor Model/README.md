# Project 05: Multi-Factor Model

## Goal

Create a long-short equity strategy using momentum, reversal, value, and quality factor scores.

## Run

```bash
python3 'quant projects/Multi-Factor Model/main.py'
```

## Outputs

- `outputs/monthly_long_short_backtest.csv`
- `outputs/monthly_factor_scores.csv`
- `outputs/factor_summary.csv`

## What This Shows

- Cross-sectional factor ranking
- Top-versus-bottom portfolio construction
- Information Coefficient and IC Information Ratio

## Limitation

Value and quality data are synthetic fallbacks. A live version should use real fundamentals.
