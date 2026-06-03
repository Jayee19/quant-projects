"""Project 01: Historical Simulation VaR Calculator."""

from __future__ import annotations

import os
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quant_projects.data import DEFAULT_INDIAN_EQUITIES, daily_returns, load_prices, portfolio_returns, save_table
from quant_projects.metrics import expected_shortfall, historical_var, parametric_var


CONFIDENCE_LEVELS = [0.90, 0.95, 0.99]
WEIGHTS = [0.25, 0.20, 0.20, 0.20, 0.15]


def run_analysis(output_dir: str = "quant projects/Historical Simulation VaR/outputs") -> pd.DataFrame:
    prices, assumption = load_prices(DEFAULT_INDIAN_EQUITIES, years=5, seed=101)
    returns = daily_returns(prices)
    portfolio = portfolio_returns(returns, WEIGHTS[: returns.shape[1]])

    rows = []
    for confidence in CONFIDENCE_LEVELS:
        hist_var = historical_var(portfolio, confidence)
        cvar = expected_shortfall(portfolio, confidence)
        pvar = parametric_var(portfolio, confidence)
        rows.append(
            {
                "confidence": confidence,
                "historical_var_daily_loss_pct": hist_var,
                "expected_shortfall_daily_loss_pct": cvar,
                "parametric_var_daily_loss_pct": pvar,
                "hist_minus_parametric_gap_pct": hist_var - pvar,
            }
        )

    summary = pd.DataFrame(rows).set_index("confidence")
    summary.attrs["data_source"] = assumption.source
    summary.attrs["assumption_note"] = assumption.note

    save_table(summary, output_dir, "var_summary.csv")
    save_table(portfolio.to_frame(), output_dir, "portfolio_returns.csv")
    assumptions = pd.DataFrame(
        {
            "item": ["tickers", "weights", "data_source", "limitation"],
            "value": [
                ", ".join(returns.columns),
                ", ".join(f"{w:.2%}" for w in WEIGHTS[: returns.shape[1]]),
                f"{assumption.source}: {assumption.note}",
                "VaR is a quantile estimate; it does not describe losses beyond the threshold.",
            ],
        }
    )
    os.makedirs(output_dir, exist_ok=True)
    assumptions.to_csv(os.path.join(output_dir, "assumptions.csv"), index=False)
    return summary


def main() -> None:
    summary = run_analysis()
    print("\nHistorical Simulation VaR")
    print(summary.round(5))
    print("\nInterpretation: Expected Shortfall is larger than VaR because it averages deeper tail losses.")


if __name__ == "__main__":
    main()
