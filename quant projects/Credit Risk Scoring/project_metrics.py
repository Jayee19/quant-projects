"""Risk, backtest, and model-evaluation metrics."""

from __future__ import annotations

from statistics import NormalDist
from typing import Iterable

import numpy as np
import pandas as pd


STANDARD_NORMAL = NormalDist()


def historical_var(returns: Iterable[float], confidence: float = 0.99) -> float:
    series = pd.Series(returns).dropna()
    return float(-series.quantile(1 - confidence))


def expected_shortfall(returns: Iterable[float], confidence: float = 0.99) -> float:
    series = pd.Series(returns).dropna()
    threshold = series.quantile(1 - confidence)
    return float(-series[series <= threshold].mean())


def parametric_var(returns: Iterable[float], confidence: float = 0.99) -> float:
    series = pd.Series(returns).dropna()
    z = STANDARD_NORMAL.inv_cdf(1 - confidence)
    return float(-(series.mean() + z * series.std(ddof=1)))


def sharpe_ratio(returns: Iterable[float], annualisation: int = 252) -> float:
    series = pd.Series(returns).dropna()
    std = series.std(ddof=1)
    if np.isclose(std, 0):
        return 0.0
    return float(np.sqrt(annualisation) * series.mean() / std)


def max_drawdown(returns: Iterable[float]) -> float:
    series = pd.Series(returns).dropna()
    wealth = (1 + series).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return float(drawdown.min())


def win_rate(returns: Iterable[float]) -> float:
    series = pd.Series(returns).dropna()
    if series.empty:
        return 0.0
    return float((series > 0).mean())


def spearman_corr(x: Iterable[float], y: Iterable[float]) -> float:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3:
        return np.nan
    return float(frame["x"].rank().corr(frame["y"].rank()))


def rolling_zscore(series: pd.Series, window: int = 60) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=1)
    return (series - mean) / std


def ols_alpha_beta(x: Iterable[float], y: Iterable[float]) -> tuple[float, float]:
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    matrix = np.column_stack([np.ones_like(x_array), x_array])
    alpha, beta = np.linalg.lstsq(matrix, y_array, rcond=None)[0]
    return float(alpha), float(beta)


def auc_score(y_true: Iterable[int], y_score: Iterable[float]) -> float:
    frame = pd.DataFrame({"y": list(y_true), "score": list(y_score)}).dropna()
    positives = frame["y"].sum()
    negatives = len(frame) - positives
    if positives == 0 or negatives == 0:
        return np.nan
    ranks = frame["score"].rank(method="average")
    positive_rank_sum = ranks[frame["y"] == 1].sum()
    auc = (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    return float(auc)


def kfold_indices(n_rows: int, k: int = 5, seed: int = 7) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    indices = np.arange(n_rows)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)
    result = []
    for i, test_idx in enumerate(folds):
        train_idx = np.concatenate([fold for j, fold in enumerate(folds) if j != i])
        result.append((train_idx, test_idx))
    return result
