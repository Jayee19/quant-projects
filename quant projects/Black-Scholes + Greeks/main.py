"""Project 02: Black-Scholes options pricer with all five Greeks."""

from __future__ import annotations

import os
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quant_projects.data import save_table
from quant_projects.options import black_scholes_greeks, black_scholes_price


BASE_CASE = {
    "spot": 100.0,
    "strike": 100.0,
    "rate": 0.065,
    "volatility": 0.22,
    "maturity": 0.5,
}


def run_analysis(output_dir: str = "quant projects/Black-Scholes + Greeks/outputs") -> dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    spot = BASE_CASE["spot"]
    strike = BASE_CASE["strike"]
    rate = BASE_CASE["rate"]
    vol = BASE_CASE["volatility"]
    maturity = BASE_CASE["maturity"]

    call = black_scholes_price(spot, strike, rate, vol, maturity, "call")
    put = black_scholes_price(spot, strike, rate, vol, maturity, "put")
    parity_gap = call - put - (spot - strike * pow(2.718281828459045, -rate * maturity))

    price_rows = []
    for scenario_spot in range(70, 131, 5):
        for scenario_strike in [90, 100, 110]:
            price_rows.append(
                {
                    "spot": scenario_spot,
                    "strike": scenario_strike,
                    "call_price": black_scholes_price(scenario_spot, scenario_strike, rate, vol, maturity, "call"),
                    "put_price": black_scholes_price(scenario_spot, scenario_strike, rate, vol, maturity, "put"),
                }
            )
    price_grid = pd.DataFrame(price_rows)

    greek_rows = []
    for option_type in ["call", "put"]:
        greek_rows.append(
            {
                "option_type": option_type,
                "price": black_scholes_price(spot, strike, rate, vol, maturity, option_type),
                **black_scholes_greeks(spot, strike, rate, vol, maturity, option_type),
            }
        )
    greeks = pd.DataFrame(greek_rows).set_index("option_type")

    sensitivity_rows = []
    for scenario_vol in [0.12, 0.18, 0.22, 0.30, 0.45]:
        for moneyness_spot in [90, 100, 110]:
            sensitivity_rows.append(
                {
                    "spot": moneyness_spot,
                    "volatility": scenario_vol,
                    "delta": black_scholes_greeks(
                        moneyness_spot, strike, rate, scenario_vol, maturity, "call"
                    )["delta"],
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)

    parity = pd.DataFrame(
        {
            "metric": ["call", "put", "put_call_parity_gap"],
            "value": [call, put, parity_gap],
        }
    )

    save_table(price_grid, output_dir, "price_vs_spot.csv")
    save_table(greeks, output_dir, "greeks.csv")
    save_table(sensitivity, output_dir, "delta_sensitivity.csv")
    parity.to_csv(os.path.join(output_dir, "put_call_parity.csv"), index=False)

    return {"price_grid": price_grid, "greeks": greeks, "sensitivity": sensitivity, "parity": parity}


def main() -> None:
    result = run_analysis()
    print("\nBlack-Scholes Base-Case Greeks")
    print(result["greeks"].round(5))
    print("\nPut-call parity gap should be close to zero:")
    print(result["parity"].round(8))


if __name__ == "__main__":
    main()
