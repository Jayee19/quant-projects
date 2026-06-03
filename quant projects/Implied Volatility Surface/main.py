"""Project 08: Implied volatility surface construction."""

from __future__ import annotations

import os
import pathlib
import sys

import pandas as pd

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from project_options import black_scholes_price, implied_volatility_bisection


SPOT = 100.0
RATE = 0.055
STRIKES = [75, 85, 95, 100, 105, 115, 125]
MATURITIES = [0.08, 0.25, 0.50, 1.00, 2.00]


def synthetic_market_iv(strike: float, maturity: float) -> float:
    moneyness = strike / SPOT
    smile = 0.18 + 0.38 * (moneyness - 1) ** 2
    put_skew = 0.10 * max(1 - moneyness, 0)
    term_premium = 0.025 * maturity**0.5
    return smile + put_skew + term_premium


def run_analysis(output_dir: str = "quant projects/Implied Volatility Surface/outputs") -> pd.DataFrame:
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for maturity in MATURITIES:
        for strike in STRIKES:
            true_iv = synthetic_market_iv(strike, maturity)
            option_type = "put" if strike < SPOT else "call"
            market_price = black_scholes_price(SPOT, strike, RATE, true_iv, maturity, option_type)
            implied_iv = implied_volatility_bisection(market_price, SPOT, strike, RATE, maturity, option_type)
            rows.append(
                {
                    "strike": strike,
                    "maturity_years": maturity,
                    "option_type_used": option_type,
                    "market_price": market_price,
                    "implied_volatility": implied_iv,
                    "true_synthetic_volatility": true_iv,
                    "reconstruction_error": implied_iv - true_iv,
                }
            )
    surface = pd.DataFrame(rows)
    summary = pd.DataFrame(
        {
            "metric": ["min_iv", "max_iv", "atm_1y_iv", "lowest_strike_1y_iv", "skew_1y_low_minus_atm"],
            "value": [
                surface["implied_volatility"].min(),
                surface["implied_volatility"].max(),
                surface.query("strike == 100 and maturity_years == 1.0")["implied_volatility"].iloc[0],
                surface.query("strike == 75 and maturity_years == 1.0")["implied_volatility"].iloc[0],
                surface.query("strike == 75 and maturity_years == 1.0")["implied_volatility"].iloc[0]
                - surface.query("strike == 100 and maturity_years == 1.0")["implied_volatility"].iloc[0],
            ],
        }
    )
    surface.to_csv(os.path.join(output_dir, "implied_vol_surface.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "surface_summary.csv"), index=False)
    return surface


def main() -> None:
    surface = run_analysis()
    print("\nImplied Volatility Surface")
    print(surface.pivot(index="strike", columns="maturity_years", values="implied_volatility").round(4))


if __name__ == "__main__":
    main()
