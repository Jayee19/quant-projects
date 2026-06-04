# Project 03: Pairs Trading with Cointegration

## Goal

Identify cointegrated asset pairs and backtest a z-score mean-reversion trading strategy on the strongest pair.

## Assumptions

The project assumes that a stable linear relationship between two log-price series can create a tradable spread. Entry signals are triggered when the spread z-score moves beyond +/-2.0, and exits occur when the absolute z-score falls below 0.5. The backtest uses rolling hedge ratios and does not include trading costs.

## Data Sources

The project uses deterministic synthetic pair data generated inside `main.py`. The synthetic universe contains BANK_A/BANK_B, IT_A/IT_B, and ENERGY_A/ENERGY_B, with two pairs designed to be strongly cointegrated and one noisier pair for comparison.

## Methodology

The script estimates hedge ratios with OLS on log prices, runs an ADF-style residual stationarity test, selects the pair with the lowest approximate p-value, and backtests a rolling spread strategy. It reports cointegration diagnostics, daily strategy returns, Sharpe ratio, drawdown, win rate, and trade count.

## Results

The committed run finds all three synthetic pairs cointegrated at the 5% level, with IT_A/IT_B and BANK_A/BANK_B showing very strong residual stationarity. The selected backtest produced a Sharpe ratio of about -0.40, max drawdown of about -26.9%, win rate of about 17.0%, and 29 position changes, which is a useful reminder that statistical cointegration alone does not guarantee a profitable strategy.

## What The Model Cannot Do

This model cannot prove that a real pair will remain cointegrated after deployment. It does not account for borrow costs, slippage, transaction costs, shorting constraints, market impact, latency, changing hedge ratios under stress, or crowded-trade unwind risk. It should be treated as a research workflow, not a live trading system.

## How To Run

```bash
python3 'quant projects/Pairs Trading with Cointegration/main.py'
```

## Outputs

- `outputs/cointegration_tests.csv`
- `outputs/backtest.csv`
- `outputs/performance_metrics.csv`
