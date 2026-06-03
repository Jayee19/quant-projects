"""Smoke test every roadmap script without requiring optional packages."""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROJECTS = [
    "quant projects/Historical Simulation VaR/main.py",
    "quant projects/Black-Scholes + Greeks/main.py",
    "quant projects/Pairs Trading with Cointegration/main.py",
    "quant projects/GARCH Volatility/main.py",
    "quant projects/Multi-Factor Model/main.py",
    "quant projects/Credit Risk Scoring/main.py",
    "quant projects/Stress Testing/main.py",
    "quant projects/Implied Volatility Surface/main.py",
    "quant projects/Monte Carlo Option Pricing with Variance Reduction/main.py",
    "quant projects/PCA Risk Decomposition/main.py",
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
    with tempfile.TemporaryDirectory() as tmpdir:
        for index, script_path in enumerate(PROJECTS, start=1):
            module = load_project(script_path)
            result = module.run_analysis(output_dir=f"{tmpdir}/project_{index:02d}")
            assert result is not None, script_path
            print(f"[smoke-ok] {script_path}")


if __name__ == "__main__":
    main()
