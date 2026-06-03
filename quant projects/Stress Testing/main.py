"""Project 07: Portfolio stress testing framework."""

from __future__ import annotations

import os
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quant_projects.data import DEFAULT_INDIAN_EQUITIES, load_prices


NOTIONAL_INR = 50_000_000
WEIGHTS = pd.Series(
    {
        "RELIANCE.NS": 0.28,
        "TCS.NS": 0.20,
        "HDFCBANK.NS": 0.22,
        "INFY.NS": 0.16,
        "ICICIBANK.NS": 0.14,
    }
)

SCENARIOS = {
    "2008_global_financial_crisis": {
        "type": "historical",
        "shocks": {
            "RELIANCE.NS": -0.36,
            "TCS.NS": -0.24,
            "HDFCBANK.NS": -0.31,
            "INFY.NS": -0.21,
            "ICICIBANK.NS": -0.42,
        },
    },
    "2020_covid_crash": {
        "type": "historical",
        "shocks": {
            "RELIANCE.NS": -0.25,
            "TCS.NS": -0.18,
            "HDFCBANK.NS": -0.28,
            "INFY.NS": -0.17,
            "ICICIBANK.NS": -0.34,
        },
    },
    "2022_rate_hike_repricing": {
        "type": "historical",
        "shocks": {
            "RELIANCE.NS": -0.13,
            "TCS.NS": -0.16,
            "HDFCBANK.NS": -0.11,
            "INFY.NS": -0.19,
            "ICICIBANK.NS": -0.14,
        },
    },
    "hypothetical_rate_spike_plus_equity_selloff": {
        "type": "hypothetical",
        "shocks": {
            "RELIANCE.NS": -0.22,
            "TCS.NS": -0.20,
            "HDFCBANK.NS": -0.27,
            "INFY.NS": -0.19,
            "ICICIBANK.NS": -0.30,
        },
    },
    "hypothetical_it_margin_squeeze": {
        "type": "hypothetical",
        "shocks": {
            "RELIANCE.NS": -0.08,
            "TCS.NS": -0.26,
            "HDFCBANK.NS": -0.07,
            "INFY.NS": -0.29,
            "ICICIBANK.NS": -0.09,
        },
    },
}


def run_analysis(output_dir: str = "quant projects/Stress Testing/outputs") -> dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    prices, assumption = load_prices(DEFAULT_INDIAN_EQUITIES, years=5, seed=707)
    latest_prices = prices.iloc[-1]
    positions = pd.DataFrame(
        {
            "ticker": WEIGHTS.index,
            "weight": WEIGHTS.values,
            "notional_inr": WEIGHTS.values * NOTIONAL_INR,
            "latest_price": latest_prices.reindex(WEIGHTS.index).values,
        }
    )

    pnl_rows = []
    for scenario_name, config in SCENARIOS.items():
        for ticker, shock in config["shocks"].items():
            notional = float(WEIGHTS[ticker] * NOTIONAL_INR)
            pnl_rows.append(
                {
                    "scenario": scenario_name,
                    "scenario_type": config["type"],
                    "ticker": ticker,
                    "shock_return": shock,
                    "position_notional_inr": notional,
                    "pnl_inr": notional * shock,
                }
            )
    pnl = pd.DataFrame(pnl_rows)
    scenario_summary = (
        pnl.groupby(["scenario", "scenario_type"])["pnl_inr"]
        .sum()
        .to_frame("portfolio_pnl_inr")
        .assign(portfolio_return=lambda frame: frame["portfolio_pnl_inr"] / NOTIONAL_INR)
        .reset_index()
    )
    attribution = (
        pnl.assign(abs_loss=lambda frame: frame["pnl_inr"].abs())
        .sort_values(["scenario", "abs_loss"], ascending=[True, False])
        .groupby("scenario")
        .head(1)[["scenario", "ticker", "pnl_inr"]]
        .rename(columns={"ticker": "largest_loss_driver"})
    )
    assumptions = pd.DataFrame(
        {
            "item": ["notional", "data_source", "limitation"],
            "value": [
                f"INR {NOTIONAL_INR:,.0f}",
                f"{assumption.source}: {assumption.note}",
                "Linear equity stress test ignores transaction costs, liquidity gaps, and option convexity.",
            ],
        }
    )

    positions.to_csv(os.path.join(output_dir, "current_positions.csv"), index=False)
    pnl.to_csv(os.path.join(output_dir, "scenario_position_pnl.csv"), index=False)
    scenario_summary.to_csv(os.path.join(output_dir, "scenario_summary.csv"), index=False)
    attribution.to_csv(os.path.join(output_dir, "largest_loss_driver.csv"), index=False)
    assumptions.to_csv(os.path.join(output_dir, "assumptions.csv"), index=False)
    return {"summary": scenario_summary, "attribution": attribution, "positions": positions}


def main() -> None:
    result = run_analysis()
    print("\nStress Test Scenario Summary")
    print(result["summary"].round(4))
    print("\nLargest Loss Drivers")
    print(result["attribution"])


if __name__ == "__main__":
    main()
