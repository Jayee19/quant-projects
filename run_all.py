"""Run every project folder in the roadmap and regenerate CSV outputs."""

from __future__ import annotations

import importlib.util
import pathlib
import time


ROOT = pathlib.Path(__file__).resolve().parent

PROJECTS = [
    ("01 Historical VaR", "quant projects/Historical Simulation VaR/main.py"),
    ("02 Black-Scholes + Greeks", "quant projects/Black-Scholes + Greeks/main.py"),
    ("03 Pairs Trading", "quant projects/Pairs Trading with Cointegration/main.py"),
    ("04 GARCH Volatility", "quant projects/GARCH Volatility/main.py"),
    ("05 Factor Model", "quant projects/Multi-Factor Model/main.py"),
    ("06 Credit Scoring", "quant projects/Credit Risk Scoring/main.py"),
    ("07 Stress Testing", "quant projects/Stress Testing/main.py"),
    ("08 Implied Vol Surface", "quant projects/Implied Volatility Surface/main.py"),
    ("09 Monte Carlo Option Pricing", "quant projects/Monte Carlo Option Pricing with Variance Reduction/main.py"),
    ("10 PCA Risk Decomposition", "quant projects/PCA Risk Decomposition/main.py"),
]


def load_project(script_path: str):
    path = ROOT / script_path
    module_name = script_path.replace("/", "_").replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    start = time.time()
    for label, script_path in PROJECTS:
        project_start = time.time()
        module = load_project(script_path)
        module.run_analysis()
        print(f"[ok] {label} ({time.time() - project_start:.2f}s)")
    print(f"\nAll projects completed in {time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
