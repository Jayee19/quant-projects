# Project 10: PCA Risk Decomposition

## Goal

Decompose portfolio risk using marginal VaR, component VaR, diversification ratio, and PCA.

## Run

```bash
python3 'quant projects/PCA Risk Decomposition/main.py'
```

## Outputs

- `outputs/risk_contributions.csv`
- `outputs/pca_summary.csv`
- `outputs/pca_loadings.csv`
- `outputs/portfolio_risk_summary.csv`

## What This Shows

- Marginal and component VaR
- Component VaR sum check
- PCA loadings and explained variance
- Market-like first principal component

## Limitation

PCA is statistical, not causal. Components can change across market regimes.
