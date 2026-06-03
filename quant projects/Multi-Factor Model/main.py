"""Project 05: Multi-factor model with monthly IC and long-short backtest."""

from __future__ import annotations

import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from quant_projects.data import load_prices, save_table
from quant_projects.metrics import sharpe_ratio, spearman_corr


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
    "MARUTI.NS",
    "SUNPHARMA.NS",
]


def _static_fundamentals(tickers: list[str]) -> tuple[pd.Series, pd.Series]:
    rng = np.random.default_rng(505)
    value = pd.Series(rng.normal(0.7, 0.18, len(tickers)), index=tickers, name="book_to_market")
    quality = pd.Series(rng.normal(0.18, 0.05, len(tickers)), index=tickers, name="gross_profit_to_assets")
    return value.clip(0.2, 1.2), quality.clip(0.05, 0.35)


def _rank_pct(row: pd.Series, ascending: bool = True) -> pd.Series:
    return row.rank(pct=True, ascending=ascending)


def run_analysis(output_dir: str = "quant projects/Multi-Factor Model/outputs") -> dict[str, pd.DataFrame]:
    os.makedirs(output_dir, exist_ok=True)
    prices, assumption = load_prices(TICKERS, years=7, seed=505)
    monthly_prices = prices.resample(pd.offsets.MonthEnd()).last()
    monthly_returns = monthly_prices.pct_change()
    forward_returns = monthly_returns.shift(-1)

    momentum = monthly_prices.shift(1) / monthly_prices.shift(13) - 1
    reversal = -monthly_returns.shift(1)
    value, quality = _static_fundamentals(list(monthly_prices.columns))

    portfolio_rows = []
    score_rows = []
    for date in monthly_returns.index:
        factors = pd.DataFrame(
            {
                "momentum": momentum.loc[date],
                "reversal": reversal.loc[date],
                "value": value,
                "quality": quality,
            }
        ).dropna()
        if len(factors) < 6 or date not in forward_returns.index:
            continue

        ranked = pd.DataFrame(
            {
                "momentum_rank": _rank_pct(factors["momentum"], ascending=True),
                "reversal_rank": _rank_pct(factors["reversal"], ascending=True),
                "value_rank": _rank_pct(factors["value"], ascending=True),
                "quality_rank": _rank_pct(factors["quality"], ascending=True),
            }
        )
        score = ranked.mean(axis=1)
        next_month = forward_returns.loc[date, score.index].dropna()
        score = score.loc[next_month.index]
        if len(score) < 6:
            continue

        long_names = score.nlargest(3).index
        short_names = score.nsmallest(3).index
        long_short_return = float(next_month.loc[long_names].mean() - next_month.loc[short_names].mean())
        ic = spearman_corr(score, next_month)
        portfolio_rows.append(
            {
                "date": date,
                "long_book": ", ".join(long_names),
                "short_book": ", ".join(short_names),
                "long_short_return": long_short_return,
                "information_coefficient": ic,
            }
        )
        for ticker, value_score in score.items():
            score_rows.append({"date": date, "ticker": ticker, "composite_score": value_score})

    portfolio = pd.DataFrame(portfolio_rows).set_index("date")
    scores = pd.DataFrame(score_rows)
    summary = pd.DataFrame(
        {
            "metric": [
                "annualised_sharpe",
                "mean_monthly_return",
                "mean_ic",
                "ic_information_ratio",
                "data_source",
            ],
            "value": [
                sharpe_ratio(portfolio["long_short_return"], annualisation=12),
                float(portfolio["long_short_return"].mean()),
                float(portfolio["information_coefficient"].mean()),
                float(portfolio["information_coefficient"].mean() / portfolio["information_coefficient"].std(ddof=1)),
                f"{assumption.source}: {assumption.note}",
            ],
        }
    )

    save_table(portfolio, output_dir, "monthly_long_short_backtest.csv")
    scores.to_csv(os.path.join(output_dir, "monthly_factor_scores.csv"), index=False)
    summary.to_csv(os.path.join(output_dir, "factor_summary.csv"), index=False)
    return {"portfolio": portfolio, "scores": scores, "summary": summary}


def main() -> None:
    result = run_analysis()
    print("\nFactor Model Summary")
    print(result["summary"])


if __name__ == "__main__":
    main()
