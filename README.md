# Quant Finance Project Portfolio

Ten standalone quant finance projects for market risk, derivatives, quant research, credit risk, stress testing, and quant development.

Each project has its own folder under `quant projects/` with:

- `main.py` for the runnable analysis
- `README.md` for the project-specific explanation
- `outputs/` generated locally when the script is run

The private interview-practice notes are intentionally kept outside the public repo in `private/`, which is ignored by git.

## Projects

| # | Project | Folder |
|---|---|---|
| 01 | Historical Simulation VaR | `quant projects/Historical Simulation VaR/` |
| 02 | Black-Scholes + Greeks | `quant projects/Black-Scholes + Greeks/` |
| 03 | Pairs Trading with Cointegration | `quant projects/Pairs Trading with Cointegration/` |
| 04 | GARCH Volatility | `quant projects/GARCH Volatility/` |
| 05 | Multi-Factor Model | `quant projects/Multi-Factor Model/` |
| 06 | Credit Risk Scoring | `quant projects/Credit Risk Scoring/` |
| 07 | Stress Testing | `quant projects/Stress Testing/` |
| 08 | Implied Volatility Surface | `quant projects/Implied Volatility Surface/` |
| 09 | Monte Carlo Option Pricing with Variance Reduction | `quant projects/Monte Carlo Option Pricing with Variance Reduction/` |
| 10 | PCA Risk Decomposition | `quant projects/PCA Risk Decomposition/` |

## Quick Start

```bash
python3 run_all.py
python3 tests/smoke_test.py
```

To run one project:

```bash
python3 'quant projects/Historical Simulation VaR/main.py'
```

The baseline code runs with `numpy` and `pandas`. Optional packages in `requirements.txt` can be installed later for live data, formal econometrics, richer ML libraries, and plotting.

## Notes

- Generated outputs are ignored by git and can be regenerated locally.
- Synthetic fallback data is deterministic, so the repo remains runnable without network access.
- The code is educational and portfolio-oriented, not investment advice or production risk infrastructure.
