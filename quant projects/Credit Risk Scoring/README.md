# Project 06: Credit Risk Scoring

## Goal

Estimate Probability of Default, compare credit scoring models, and calculate Expected Loss as `PD x LGD x EAD`.

## Assumptions

The loan book has 5,000 simulated borrowers. Default probability is driven by credit score, debt-to-income, employment years, late payments, loan-to-income, and existing credit lines. LGD and EAD are simulated borrower-level quantities, and model quality is evaluated with five-fold cross-validated AUC.

## Data Sources

The project uses a deterministic synthetic loan book generated in `main.py`. It does not use private borrower data or external credit bureau data. The synthetic dataset is designed to mimic common credit-risk relationships while remaining safe to publish.

## Methodology

The script simulates borrower features, default outcomes, LGD, and EAD, then compares logistic regression, random-forest-style stumps, and gradient-boosted stumps. The best model by mean cross-validation AUC is used to score the full loan book and calculate loan-level and portfolio-level expected loss.

## Results

Logistic regression is the best model in the committed run with mean cross-validation AUC of about 0.791. The simulated portfolio default rate is about 8.28%, mean model PD is about 8.38%, mean LGD is about 42.97%, total EAD is about INR 3.40 billion, and portfolio expected loss is about INR 146.31 million.

## What The Model Cannot Do

This model cannot replace a bank-grade credit model. It is trained on simulated data, lacks real underwriting labels, fairness review, calibration monitoring, reject inference, macroeconomic overlays, documentation governance, and regulatory validation. It demonstrates the mechanics of PD and expected loss, not production credit approval.

## How To Run

```bash
python3 'quant projects/Credit Risk Scoring/main.py'
```

## Outputs

- `outputs/loan_book_with_expected_loss.csv`
- `outputs/cross_validation_auc.csv`
- `outputs/model_comparison.csv`
- `outputs/portfolio_expected_loss.csv`
