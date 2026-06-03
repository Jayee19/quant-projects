"""Data utilities with live-data support and deterministic synthetic fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


DEFAULT_INDIAN_EQUITIES = [
    "RELIANCE.NS",
    "TCS.NS",
    "HDFCBANK.NS",
    "INFY.NS",
    "ICICIBANK.NS",
]


@dataclass(frozen=True)
class DataAssumption:
    source: str
    note: str


def business_dates(periods: int, start: str = "2019-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=periods)


def normalise_weights(weights: Sequence[float], n_assets: int | None = None) -> np.ndarray:
    weights_array = np.asarray(weights, dtype=float)
    if n_assets is not None and len(weights_array) != n_assets:
        raise ValueError(f"Expected {n_assets} weights, got {len(weights_array)}")
    total = weights_array.sum()
    if np.isclose(total, 0):
        raise ValueError("Weights must not sum to zero")
    return weights_array / total


def synthetic_price_paths(
    tickers: Sequence[str],
    periods: int = 1260,
    start: str = "2019-01-01",
    seed: int = 7,
    base_price: float = 100.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_assets = len(tickers)
    dates = business_dates(periods=periods, start=start)
    market = rng.normal(0.00035, 0.0105, size=periods)
    sector = rng.normal(0.00015, 0.0065, size=(periods, 3))
    prices = {}

    for i, ticker in enumerate(tickers):
        beta = 0.75 + 0.12 * (i % 5)
        sector_beta = 0.35 + 0.05 * (i % 3)
        drift = 0.00012 + 0.00004 * (i % 4)
        idio_vol = 0.008 + 0.0015 * (i % 4)
        shocks = rng.normal(0, idio_vol, size=periods)
        log_returns = drift + beta * market + sector_beta * sector[:, i % 3] + shocks

        crisis_idx = np.arange(260, min(periods, 285))
        if len(crisis_idx):
            log_returns[crisis_idx] -= 0.012 + 0.002 * (i % 3)
        recovery_idx = np.arange(286, min(periods, 330))
        if len(recovery_idx):
            log_returns[recovery_idx] += 0.004

        prices[ticker] = base_price * (1 + 0.08 * i) * np.exp(np.cumsum(log_returns))

    return pd.DataFrame(prices, index=dates)


def load_prices(
    tickers: Sequence[str],
    years: int = 5,
    seed: int = 7,
    prefer_yfinance: bool = True,
) -> tuple[pd.DataFrame, DataAssumption]:
    periods = years * 252
    if prefer_yfinance:
        try:
            import yfinance as yf  # type: ignore

            raw = yf.download(
                list(tickers),
                period=f"{years}y",
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            prices = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw
            prices = prices.dropna(how="all").ffill().dropna()
            if len(prices) > 60 and set(tickers).intersection(prices.columns):
                prices = prices.reindex(columns=tickers).dropna(axis=1, how="all")
                return prices, DataAssumption(
                    source="yfinance",
                    note=f"Adjusted close data for {len(prices.columns)} assets over {len(prices)} rows.",
                )
        except Exception:
            pass

    prices = synthetic_price_paths(tickers=tickers, periods=periods, seed=seed)
    return prices, DataAssumption(
        source="synthetic",
        note="Deterministic synthetic equity data used because live market data was unavailable.",
    )


def daily_returns(prices: pd.DataFrame, log: bool = False) -> pd.DataFrame:
    if log:
        return np.log(prices / prices.shift(1)).dropna()
    return prices.pct_change().dropna()


def portfolio_returns(returns: pd.DataFrame, weights: Iterable[float]) -> pd.Series:
    weights_array = normalise_weights(list(weights), returns.shape[1])
    values = returns.to_numpy(dtype=float, copy=True)
    return pd.Series(np.dot(values, weights_array), index=returns.index, name="portfolio_return")


def save_table(df: pd.DataFrame, output_dir: str, filename: str) -> str:
    import os

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=True)
    return path
