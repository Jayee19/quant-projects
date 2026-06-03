"""Project 04: GARCH(1,1) volatility modelling and dynamic VaR."""

from __future__ import annotations

import os
import pathlib
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

PROJECT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from project_data import business_dates, save_table


def _garch_returns(periods: int = 1260, seed: int = 404) -> pd.Series:
    rng = np.random.default_rng(seed)
    omega, alpha, beta = 0.000002, 0.08, 0.90
    variance = np.empty(periods)
    returns = np.empty(periods)
    variance[0] = omega / (1 - alpha - beta)
    returns[0] = rng.normal(0, np.sqrt(variance[0]))
    for t in range(1, periods):
        variance[t] = omega + alpha * returns[t - 1] ** 2 + beta * variance[t - 1]
        returns[t] = rng.normal(0.00025, np.sqrt(variance[t]))
    return pd.Series(returns, index=business_dates(periods, start="2019-01-01"), name="nifty_log_return")


def _chi_square_survival_wilson_hilferty(x: float, df: int) -> float:
    if x <= 0:
        return 1.0
    z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / np.sqrt(2 / (9 * df))
    return float(1 - NormalDist().cdf(z))


def arch_lm_test(returns: pd.Series, lags: int = 5) -> dict[str, float]:
    squared = returns.dropna().to_numpy() ** 2
    y = squared[lags:]
    x_lags = [squared[lags - lag : -lag] for lag in range(1, lags + 1)]
    x = np.column_stack([np.ones_like(y), *x_lags])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    fitted = np.dot(x, beta)
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    lm_stat = len(y) * r_squared
    return {"lm_stat": float(lm_stat), "df": float(lags), "approx_p_value": _chi_square_survival_wilson_hilferty(lm_stat, lags)}


def fit_garch_grid(returns: pd.Series) -> tuple[dict[str, float], pd.Series]:
    values = returns.dropna().to_numpy()
    demeaned = values - values.mean()
    sample_variance = float(np.var(demeaned))
    best = {"neg_log_likelihood": np.inf, "omega": np.nan, "alpha": np.nan, "beta": np.nan}
    best_variance = np.full(len(demeaned), sample_variance)

    for alpha in np.linspace(0.03, 0.18, 16):
        for beta in np.linspace(0.72, 0.97, 26):
            if alpha + beta >= 0.995:
                continue
            omega = sample_variance * (1 - alpha - beta)
            variance = np.empty(len(demeaned))
            variance[0] = sample_variance
            for t in range(1, len(demeaned)):
                variance[t] = omega + alpha * demeaned[t - 1] ** 2 + beta * variance[t - 1]
            neg_ll = float(0.5 * np.sum(np.log(variance) + demeaned**2 / variance))
            if neg_ll < best["neg_log_likelihood"]:
                best = {
                    "neg_log_likelihood": neg_ll,
                    "omega": float(omega),
                    "alpha": float(alpha),
                    "beta": float(beta),
                    "persistence_alpha_plus_beta": float(alpha + beta),
                }
                best_variance = variance

    conditional_vol = pd.Series(np.sqrt(best_variance), index=returns.dropna().index, name="conditional_volatility")
    return best, conditional_vol


def run_analysis(output_dir: str = "quant projects/GARCH Volatility/outputs") -> dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    returns = _garch_returns()
    lm_result = pd.DataFrame([arch_lm_test(returns)])
    params, conditional_vol = fit_garch_grid(returns)

    z_99 = NormalDist().inv_cdf(0.01)
    next_variance = params["omega"] + params["alpha"] * (returns.iloc[-1] - returns.mean()) ** 2 + params["beta"] * conditional_vol.iloc[-1] ** 2
    dynamic_var = -(returns.mean() + z_99 * np.sqrt(next_variance))
    parameter_table = pd.DataFrame([{**params, "one_day_99pct_garch_var": float(dynamic_var)}])

    save_table(returns.to_frame(), output_dir, "nifty_returns.csv")
    save_table(conditional_vol.to_frame(), output_dir, "conditional_volatility.csv")
    lm_result.to_csv(os.path.join(output_dir, "arch_lm_test.csv"), index=False)
    parameter_table.to_csv(os.path.join(output_dir, "garch_parameters.csv"), index=False)
    return {"arch_lm": lm_result, "parameters": parameter_table, "volatility": conditional_vol.to_frame()}


def main() -> None:
    result = run_analysis()
    print("\nARCH LM Test")
    print(result["arch_lm"].round(5))
    print("\nGARCH(1,1) Grid-Fit Parameters")
    print(result["parameters"].round(6))


if __name__ == "__main__":
    main()
