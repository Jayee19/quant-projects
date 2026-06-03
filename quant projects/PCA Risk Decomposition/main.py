"""Project 10: Portfolio risk decomposition via marginal VaR, component VaR, and PCA."""

from __future__ import annotations

import os
import pathlib
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from project_data import daily_returns, load_prices


TICKERS = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
    "LT.NS",
    "SBIN.NS",
    "AXISBANK.NS",
    "HINDUNILVR.NS",
    "BHARTIARTL.NS",
]

WEIGHTS = np.array([0.18, 0.14, 0.13, 0.10, 0.11, 0.09, 0.07, 0.06, 0.07, 0.05])


def run_analysis(output_dir: str = "quant projects/PCA Risk Decomposition/outputs") -> dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    prices, assumption = load_prices(TICKERS, years=5, seed=1001)
    returns = daily_returns(prices).reindex(columns=TICKERS).dropna()
    covariance = returns.cov().to_numpy()
    weights = WEIGHTS[: returns.shape[1]]
    weights = weights / weights.sum()

    portfolio_volatility = float(np.sqrt(np.dot(weights, np.dot(covariance, weights))))
    z_99 = abs(NormalDist().inv_cdf(0.01))
    total_var = z_99 * portfolio_volatility
    marginal_var = z_99 * np.dot(covariance, weights) / portfolio_volatility
    component_var = weights * marginal_var
    component_pct = component_var / total_var
    standalone_var = z_99 * returns.std(ddof=1).to_numpy()
    diversification_ratio = float(np.dot(weights, returns.std(ddof=1).to_numpy()) / portfolio_volatility)

    risk_table = pd.DataFrame(
        {
            "ticker": returns.columns,
            "weight": weights,
            "standalone_99pct_var": standalone_var,
            "marginal_var": marginal_var,
            "component_var": component_var,
            "component_var_pct": component_pct,
        }
    ).sort_values("component_var", ascending=False)

    centered_returns = returns - returns.mean()
    covariance_matrix = centered_returns.cov().to_numpy()
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    explained = eigenvalues / eigenvalues.sum()
    pc_summary = pd.DataFrame(
        {
            "principal_component": [f"PC{i + 1}" for i in range(len(eigenvalues))],
            "eigenvalue": eigenvalues,
            "explained_variance_pct": explained,
            "cumulative_explained_variance_pct": np.cumsum(explained),
        }
    )
    loadings = pd.DataFrame(
        {
            "ticker": returns.columns,
            "pc1_loading": eigenvectors[:, 0],
            "pc2_loading": eigenvectors[:, 1],
        }
    )
    summary = pd.DataFrame(
        {
            "metric": [
                "portfolio_daily_volatility",
                "portfolio_99pct_var",
                "component_var_sum",
                "component_var_sum_check",
                "diversification_ratio",
                "pc1_explained_variance_pct",
                "data_source",
            ],
            "value": [
                portfolio_volatility,
                total_var,
                component_var.sum(),
                abs(component_var.sum() - total_var),
                diversification_ratio,
                explained[0],
                f"{assumption.source}: {assumption.note}",
            ],
        }
    )

    risk_table.to_csv(os.path.join(output_dir, "risk_contributions.csv"), index=False)
    pc_summary.to_csv(os.path.join(output_dir, "pca_summary.csv"), index=False)
    loadings.to_csv(os.path.join(output_dir, "pca_loadings.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "portfolio_risk_summary.csv"), index=False)
    return {"risk_table": risk_table, "pc_summary": pc_summary, "loadings": loadings, "summary": summary}


def main() -> None:
    result = run_analysis()
    print("\nRisk Contributions")
    print(result["risk_table"].round(5))
    print("\nPortfolio/PCA Summary")
    print(result["summary"])


if __name__ == "__main__":
    main()
