"""Black-Scholes pricing, Greeks, and implied volatility inversion."""

from __future__ import annotations

import math
from statistics import NormalDist


N = NormalDist()
SQRT_2PI = math.sqrt(2 * math.pi)


def normal_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def normal_cdf(x: float) -> float:
    return N.cdf(x)


def d1_d2(spot: float, strike: float, rate: float, volatility: float, maturity: float) -> tuple[float, float]:
    if spot <= 0 or strike <= 0 or volatility <= 0 or maturity <= 0:
        raise ValueError("spot, strike, volatility, and maturity must be positive")
    d1 = (
        math.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity
    ) / (volatility * math.sqrt(maturity))
    d2 = d1 - volatility * math.sqrt(maturity)
    return d1, d2


def black_scholes_price(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    option_type: str = "call",
) -> float:
    d1, d2 = d1_d2(spot, strike, rate, volatility, maturity)
    discount = math.exp(-rate * maturity)
    if option_type == "call":
        return spot * normal_cdf(d1) - strike * discount * normal_cdf(d2)
    if option_type == "put":
        return strike * discount * normal_cdf(-d2) - spot * normal_cdf(-d1)
    raise ValueError("option_type must be 'call' or 'put'")


def black_scholes_greeks(
    spot: float,
    strike: float,
    rate: float,
    volatility: float,
    maturity: float,
    option_type: str = "call",
) -> dict[str, float]:
    d1, d2 = d1_d2(spot, strike, rate, volatility, maturity)
    discount = math.exp(-rate * maturity)
    pdf_d1 = normal_pdf(d1)
    if option_type == "call":
        delta = normal_cdf(d1)
        theta = (
            -spot * pdf_d1 * volatility / (2 * math.sqrt(maturity))
            - rate * strike * discount * normal_cdf(d2)
        )
        rho = strike * maturity * discount * normal_cdf(d2)
    elif option_type == "put":
        delta = normal_cdf(d1) - 1
        theta = (
            -spot * pdf_d1 * volatility / (2 * math.sqrt(maturity))
            + rate * strike * discount * normal_cdf(-d2)
        )
        rho = -strike * maturity * discount * normal_cdf(-d2)
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    gamma = pdf_d1 / (spot * volatility * math.sqrt(maturity))
    vega = spot * pdf_d1 * math.sqrt(maturity)
    return {
        "delta": float(delta),
        "gamma": float(gamma),
        "theta_annual": float(theta),
        "theta_daily": float(theta / 365),
        "vega_per_1pct": float(vega / 100),
        "rho_per_1pct": float(rho / 100),
    }


def implied_volatility_bisection(
    market_price: float,
    spot: float,
    strike: float,
    rate: float,
    maturity: float,
    option_type: str = "call",
    lower: float = 0.0001,
    upper: float = 3.0,
    tolerance: float = 1e-6,
    max_iter: int = 100,
) -> float:
    intrinsic = max(0.0, spot - strike) if option_type == "call" else max(0.0, strike - spot)
    if market_price < intrinsic:
        raise ValueError("market_price is below intrinsic value")

    lo, hi = lower, upper
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        price = black_scholes_price(spot, strike, rate, mid, maturity, option_type)
        if abs(price - market_price) < tolerance:
            return float(mid)
        if price > market_price:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))
