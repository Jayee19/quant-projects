"""Project 09: Monte Carlo option pricing with variance reduction."""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from project_options import black_scholes_price


PARAMS = {
    "spot": 100.0,
    "strike": 105.0,
    "rate": 0.055,
    "volatility": 0.24,
    "maturity": 1.0,
}


def _terminal_prices(n_paths: int, seed: int, antithetic: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    spot, rate, vol, maturity = PARAMS["spot"], PARAMS["rate"], PARAMS["volatility"], PARAMS["maturity"]
    if antithetic:
        half = n_paths // 2
        z = rng.normal(size=half)
        z = np.concatenate([z, -z])
    else:
        z = rng.normal(size=n_paths)
    drift = (rate - 0.5 * vol**2) * maturity
    diffusion = vol * np.sqrt(maturity) * z
    return spot * np.exp(drift + diffusion)


def european_call_mc(n_paths: int = 20000, seed: int = 909, antithetic: bool = False) -> dict[str, float]:
    terminal = _terminal_prices(n_paths, seed, antithetic=antithetic)
    discount = np.exp(-PARAMS["rate"] * PARAMS["maturity"])
    payoff = discount * np.maximum(terminal - PARAMS["strike"], 0)
    return {"price": float(payoff.mean()), "standard_error": float(payoff.std(ddof=1) / np.sqrt(len(payoff)))}


def european_call_control_variate(n_paths: int = 20000, seed: int = 909) -> dict[str, float]:
    terminal = _terminal_prices(n_paths, seed, antithetic=False)
    discount = np.exp(-PARAMS["rate"] * PARAMS["maturity"])
    payoff = discount * np.maximum(terminal - PARAMS["strike"], 0)
    control = discount * terminal
    known_control_mean = PARAMS["spot"]
    beta = np.cov(payoff, control, ddof=1)[0, 1] / np.var(control, ddof=1)
    adjusted = payoff - beta * (control - known_control_mean)
    return {
        "price": float(adjusted.mean()),
        "standard_error": float(adjusted.std(ddof=1) / np.sqrt(len(adjusted))),
        "control_beta": float(beta),
    }


def asian_call_mc(n_paths: int = 20000, n_steps: int = 126, seed: int = 910) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    spot, rate, vol, maturity = PARAMS["spot"], PARAMS["rate"], PARAMS["volatility"], PARAMS["maturity"]
    dt = maturity / n_steps
    z = rng.normal(size=(n_paths, n_steps))
    increments = (rate - 0.5 * vol**2) * dt + vol * np.sqrt(dt) * z
    paths = spot * np.exp(np.cumsum(increments, axis=1))
    average_price = paths.mean(axis=1)
    payoff = np.exp(-rate * maturity) * np.maximum(average_price - PARAMS["strike"], 0)
    return {"price": float(payoff.mean()), "standard_error": float(payoff.std(ddof=1) / np.sqrt(n_paths))}


def run_analysis(output_dir: str = "quant projects/Monte Carlo Option Pricing with Variance Reduction/outputs") -> dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    bs_price = black_scholes_price(
        PARAMS["spot"], PARAMS["strike"], PARAMS["rate"], PARAMS["volatility"], PARAMS["maturity"], "call"
    )
    standard = european_call_mc()
    antithetic = european_call_mc(antithetic=True)
    control = european_call_control_variate()
    asian = asian_call_mc()

    summary = pd.DataFrame(
        [
            {"method": "black_scholes_closed_form", "price": bs_price, "standard_error": 0.0, "absolute_error": 0.0},
            {"method": "standard_monte_carlo", **standard, "absolute_error": abs(standard["price"] - bs_price)},
            {"method": "antithetic_variates", **antithetic, "absolute_error": abs(antithetic["price"] - bs_price)},
            {"method": "control_variate", **control, "absolute_error": abs(control["price"] - bs_price)},
            {"method": "asian_option_standard_mc", **asian, "absolute_error": np.nan},
        ]
    )

    convergence_rows = []
    for paths in [1000, 2500, 5000, 10000, 20000]:
        standard_run = european_call_mc(n_paths=paths, seed=paths)
        anti_run = european_call_mc(n_paths=paths, seed=paths, antithetic=True)
        convergence_rows.append(
            {
                "paths": paths,
                "standard_price": standard_run["price"],
                "standard_abs_error": abs(standard_run["price"] - bs_price),
                "antithetic_price": anti_run["price"],
                "antithetic_abs_error": abs(anti_run["price"] - bs_price),
                "variance_reduction_ratio": (standard_run["standard_error"] / anti_run["standard_error"]) ** 2,
            }
        )
    convergence = pd.DataFrame(convergence_rows)
    summary.to_csv(os.path.join(output_dir, "pricing_summary.csv"), index=False)
    convergence.to_csv(os.path.join(output_dir, "convergence_analysis.csv"), index=False)
    return {"summary": summary, "convergence": convergence}


def main() -> None:
    result = run_analysis()
    print("\nMonte Carlo Pricing Summary")
    print(result["summary"].round(6))
    print("\nConvergence")
    print(result["convergence"].round(6))


if __name__ == "__main__":
    main()
