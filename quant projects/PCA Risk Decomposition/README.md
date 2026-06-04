# Project 10: PCA Risk Decomposition

## Goal

Decompose portfolio risk using marginal VaR, component VaR, and principal component analysis.

## Assumptions

The portfolio uses 10 Indian equity tickers with fixed weights that are normalized before calculation. Daily returns are used to estimate the covariance matrix, 99% VaR is based on a normal approximation, and PCA is applied to centered daily returns.

## Data Sources

The price loader attempts to use `yfinance` adjusted close prices and falls back to deterministic synthetic equity paths when live data is unavailable. The committed output was generated from synthetic fallback data, as recorded in `outputs/portfolio_risk_summary.csv`.

## Methodology

The script estimates the covariance matrix, computes portfolio volatility, 99% VaR, marginal VaR, component VaR, standalone VaR, and diversification ratio. It then runs eigen decomposition on the return covariance matrix to summarize principal components and ticker loadings.

## Results

The committed run reports daily portfolio volatility of about 1.08% and 99% portfolio VaR of about 2.52%. Component VaR sums back to total VaR with negligible numerical error, and the diversification ratio is about 1.33. The first principal component explains about 57.19% of variance, indicating a dominant common risk factor in the portfolio.

## What The Model Cannot Do

This model cannot guarantee that PCA factors are economically stable or interpretable across market regimes. It depends on the covariance estimate, assumes linear relationships, ignores nonlinear payoffs and liquidity effects, and can miss tail dependence that appears only during stress periods.

## How To Run

```bash
python3 'quant projects/PCA Risk Decomposition/main.py'
```

## Outputs

- `outputs/risk_contributions.csv`
- `outputs/pca_summary.csv`
- `outputs/pca_loadings.csv`
- `outputs/portfolio_risk_summary.csv`
