"""Project 03: Pairs trading with cointegration and a z-score backtest."""

from __future__ import annotations

import os
import pathlib
import sys
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quant_projects.data import business_dates, save_table
from quant_projects.metrics import max_drawdown, ols_alpha_beta, rolling_zscore, sharpe_ratio, win_rate


PAIRS = [("BANK_A", "BANK_B"), ("IT_A", "IT_B"), ("ENERGY_A", "ENERGY_B")]


def _synthetic_pairs(periods: int = 900, seed: int = 303) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = business_dates(periods, start="2020-01-01")
    data = {}

    for i, (left, right) in enumerate(PAIRS):
        common_trend = np.cumsum(rng.normal(0.00025, 0.012, periods))
        spread = np.zeros(periods)
        persistence = 0.72 if i < 2 else 0.98
        for t in range(1, periods):
            spread[t] = persistence * spread[t - 1] + rng.normal(0, 0.018 + 0.004 * i)

        left_log = common_trend + rng.normal(0, 0.006, periods)
        if i < 2:
            right_log = 0.05 + 0.95 * common_trend + spread
        else:
            right_log = np.cumsum(rng.normal(0.0003, 0.015, periods))

        data[left] = 100 * np.exp(left_log)
        data[right] = 95 * np.exp(right_log)

    return pd.DataFrame(data, index=dates)


def _adf_on_residuals(residuals: pd.Series) -> tuple[float, float]:
    clean = residuals.dropna().to_numpy()
    delta = np.diff(clean)
    lagged = clean[:-1]
    x = np.column_stack([np.ones_like(lagged), lagged])
    beta = np.linalg.lstsq(x, delta, rcond=None)[0]
    fitted = np.dot(x, beta)
    errors = delta - fitted
    sigma2 = np.dot(errors, errors) / max(len(delta) - x.shape[1], 1)
    covariance = sigma2 * np.linalg.pinv(np.dot(x.T, x))
    se = np.sqrt(np.diag(covariance))
    t_stat = float(beta[1] / se[1]) if se[1] > 0 else np.nan

    if t_stat < -3.43:
        p_value = 0.01
    elif t_stat < -2.86:
        p_value = 0.05
    elif t_stat < -2.57:
        p_value = 0.10
    else:
        p_value = float(1 - NormalDist().cdf(abs(t_stat)))
    return t_stat, p_value


def _test_cointegration(prices: pd.DataFrame, pair: tuple[str, str]) -> dict[str, float | str]:
    left, right = pair
    alpha, beta = ols_alpha_beta(np.log(prices[left]), np.log(prices[right]))
    residuals = np.log(prices[right]) - alpha - beta * np.log(prices[left])
    t_stat, p_value = _adf_on_residuals(residuals)
    return {
        "pair": f"{left}/{right}",
        "hedge_ratio_full_sample": beta,
        "adf_t_stat": t_stat,
        "approx_p_value": p_value,
        "cointegrated_at_5pct": p_value <= 0.05,
    }


def _backtest_pair(prices: pd.DataFrame, pair: tuple[str, str], window: int = 60) -> tuple[pd.DataFrame, pd.DataFrame]:
    left, right = pair
    log_left = np.log(prices[left])
    log_right = np.log(prices[right])

    hedge_ratio = pd.Series(index=prices.index, dtype=float)
    for i in range(window, len(prices)):
        _, beta = ols_alpha_beta(log_left.iloc[i - window : i], log_right.iloc[i - window : i])
        hedge_ratio.iloc[i] = beta
    hedge_ratio = hedge_ratio.ffill()

    spread = log_right - hedge_ratio * log_left
    zscore = rolling_zscore(spread, window)

    position = pd.Series(0.0, index=prices.index)
    current = 0.0
    for date, z_value in zscore.items():
        if np.isnan(z_value):
            position.loc[date] = current
            continue
        if current == 0 and z_value > 2.0:
            current = -1.0
        elif current == 0 and z_value < -2.0:
            current = 1.0
        elif current != 0 and abs(z_value) < 0.5:
            current = 0.0
        position.loc[date] = current

    returns = prices[[left, right]].pct_change().fillna(0)
    pair_return = returns[right] - hedge_ratio.fillna(1.0) * returns[left]
    strategy_return = position.shift(1).fillna(0) * pair_return

    backtest = pd.DataFrame(
        {
            "hedge_ratio": hedge_ratio,
            "spread": spread,
            "zscore": zscore,
            "position": position,
            "strategy_return": strategy_return,
        }
    ).dropna()
    metrics = pd.DataFrame(
        {
            "metric": ["sharpe_ratio", "max_drawdown", "win_rate", "trades"],
            "value": [
                sharpe_ratio(backtest["strategy_return"]),
                max_drawdown(backtest["strategy_return"]),
                win_rate(backtest["strategy_return"]),
                int((backtest["position"].diff().abs() > 0).sum()),
            ],
        }
    )
    return backtest, metrics


def run_analysis(output_dir: str = "quant projects/Pairs Trading with Cointegration/outputs") -> dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    prices = _synthetic_pairs()
    cointegration = pd.DataFrame([_test_cointegration(prices, pair) for pair in PAIRS])
    best_pair_name = cointegration.sort_values("approx_p_value").iloc[0]["pair"]
    best_pair = tuple(best_pair_name.split("/"))  # type: ignore[assignment]
    backtest, metrics = _backtest_pair(prices, best_pair)

    save_table(cointegration, output_dir, "cointegration_tests.csv")
    save_table(backtest, output_dir, "backtest.csv")
    metrics.to_csv(os.path.join(output_dir, "performance_metrics.csv"), index=False)
    return {"cointegration": cointegration, "backtest": backtest, "metrics": metrics}


def main() -> None:
    result = run_analysis()
    print("\nCointegration Tests")
    print(result["cointegration"].round(4))
    print("\nBest Pair Backtest Metrics")
    print(result["metrics"].round(4))


if __name__ == "__main__":
    main()
