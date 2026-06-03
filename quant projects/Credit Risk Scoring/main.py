"""Project 06: Credit risk scoring model with PD, LGD, EAD and Expected Loss."""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from project_metrics import auc_score, kfold_indices
from project_models import fit_boosted_stumps, fit_logistic_regression, fit_stump_forest


FEATURES = [
    "credit_score",
    "debt_to_income",
    "employment_years",
    "late_payments_12m",
    "loan_to_income",
    "existing_credit_lines",
]


def simulate_loan_book(n_rows: int = 5000, seed: int = 606) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income = rng.lognormal(mean=np.log(900000), sigma=0.45, size=n_rows)
    loan_amount = rng.lognormal(mean=np.log(650000), sigma=0.55, size=n_rows)
    credit_score = np.clip(rng.normal(690, 62, n_rows), 350, 850)
    debt_to_income = np.clip(rng.beta(2.2, 5.5, n_rows), 0.02, 0.85)
    employment_years = np.clip(rng.gamma(2.2, 2.2, n_rows), 0, 30)
    late_payments = rng.poisson(np.clip(1.9 - (credit_score - 600) / 180, 0.05, 4.5))
    existing_lines = rng.poisson(4.5, n_rows)
    loan_to_income = loan_amount / income

    logit = (
        -4.25
        - 0.009 * (credit_score - 650)
        + 3.2 * debt_to_income
        - 0.08 * employment_years
        + 0.42 * late_payments
        + 0.55 * loan_to_income
        + 0.035 * existing_lines
    )
    probability_default = 1 / (1 + np.exp(-logit))
    default = rng.binomial(1, probability_default)
    lgd = np.clip(rng.normal(0.42, 0.12, n_rows) + 0.10 * default, 0.05, 0.95)
    ead = loan_amount * rng.uniform(0.78, 1.02, n_rows)

    return pd.DataFrame(
        {
            "income": income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "debt_to_income": debt_to_income,
            "employment_years": employment_years,
            "late_payments_12m": late_payments,
            "existing_credit_lines": existing_lines,
            "loan_to_income": loan_to_income,
            "true_pd": probability_default,
            "default": default,
            "lgd": lgd,
            "ead": ead,
        }
    )


def _evaluate_model(name: str, factory, x: np.ndarray, y: np.ndarray) -> tuple[pd.DataFrame, object]:
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(kfold_indices(len(y), k=5, seed=606), start=1):
        model = factory(x[train_idx], y[train_idx])
        prediction = model.predict_proba(x[test_idx])
        fold_rows.append({"model": name, "fold": fold, "auc": auc_score(y[test_idx], prediction)})
    final_model = factory(x, y)
    return pd.DataFrame(fold_rows), final_model


def run_analysis(output_dir: str = "quant projects/Credit Risk Scoring/outputs") -> dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    loans = simulate_loan_book()
    x = loans[FEATURES].to_numpy(dtype=float)
    y = loans["default"].to_numpy(dtype=float)

    factories = {
        "logistic_regression": lambda x_train, y_train: fit_logistic_regression(x_train, y_train),
        "random_forest_stumps": lambda x_train, y_train: fit_stump_forest(x_train, y_train),
        "gradient_boosted_stumps": lambda x_train, y_train: fit_boosted_stumps(x_train, y_train),
    }

    cv_tables = []
    fitted_models = {}
    for name, factory in factories.items():
        cv, model = _evaluate_model(name, factory, x, y)
        cv_tables.append(cv)
        fitted_models[name] = model

    cv_results = pd.concat(cv_tables, ignore_index=True)
    comparison = (
        cv_results.groupby("model")["auc"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "mean_cv_auc", "std": "std_cv_auc"})
        .sort_values("mean_cv_auc", ascending=False)
    )
    best_name = comparison.index[0]
    loans["model_pd"] = fitted_models[best_name].predict_proba(x)
    loans["expected_loss"] = loans["model_pd"] * loans["lgd"] * loans["ead"]

    portfolio_el = pd.DataFrame(
        {
            "metric": ["best_model", "default_rate", "portfolio_expected_loss", "mean_pd", "mean_lgd", "total_ead"],
            "value": [
                best_name,
                float(loans["default"].mean()),
                float(loans["expected_loss"].sum()),
                float(loans["model_pd"].mean()),
                float(loans["lgd"].mean()),
                float(loans["ead"].sum()),
            ],
        }
    )

    loans.to_csv(os.path.join(output_dir, "loan_book_with_expected_loss.csv"), index=False)
    cv_results.to_csv(os.path.join(output_dir, "cross_validation_auc.csv"), index=False)
    comparison.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=True)
    portfolio_el.to_csv(os.path.join(output_dir, "portfolio_expected_loss.csv"), index=False)
    return {"comparison": comparison, "expected_loss": portfolio_el, "cv_results": cv_results}


def main() -> None:
    result = run_analysis()
    print("\nCredit Model AUC Comparison")
    print(result["comparison"].round(4))
    print("\nPortfolio Expected Loss")
    print(result["expected_loss"])


if __name__ == "__main__":
    main()
