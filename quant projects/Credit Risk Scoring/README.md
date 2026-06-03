# Project 06: Credit Risk Scoring

## Goal

Estimate Probability of Default and calculate Expected Loss as `PD x LGD x EAD`.

## Run

```bash
python3 'quant projects/Credit Risk Scoring/main.py'
```

## Outputs

- `outputs/loan_book_with_expected_loss.csv`
- `outputs/cross_validation_auc.csv`
- `outputs/model_comparison.csv`
- `outputs/portfolio_expected_loss.csv`

## What This Shows

- Credit feature engineering
- AUC-based model comparison
- Expected Loss at loan and portfolio level

## Limitation

The loan book is simulated. A bank-grade model needs validation, monitoring, calibration checks, and governance.
