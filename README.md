# Latent Synchronization Risk and Value-at-Risk Underestimation

**A Random Matrix Theory Approach to Spectral Co-movement Detection**

Working paper (SSRN) applying Random Matrix Theory (RMT) and Marchenko–Pastur eigenvalue
filtering to detect latent market synchronization risk, and using it to build an
Eigenvalue-Adjusted VaR model that outperforms standard VaR benchmarks out-of-sample.

## Abstract

Conventional VaR models assume constant correlation between assets, so they miss the
tail risk that emerges when cross-asset co-movement spikes during stress. This paper
introduces the **Synchronization Factor (SF)** — a daily scalar derived from the leading
eigenvalue of a rolling residual correlation matrix relative to its Marchenko–Pastur upper
bound — across 169 S&P 500 stocks over 26 years (1999–2025, 6,776 trading days). A
calibration parameter **κ\* = 0.020** is fixed on in-sample data (1999–2012) via Kupiec
(1995) likelihood-ratio optimization and locked before out-of-sample testing (2013–2025).

**Key result:** the Eigenvalue-Adjusted VaR model (Model D) is the only one of four VaR
models to land in the **Basel III Green Zone** out-of-sample (1.28% violation rate, Kupiec
p = 0.117), while Normal VaR, Historical Simulation, and EWMA-Filtered Historical
Simulation all remain in the Red Zone (2.20%, 2.05%, 1.84% respectively).

## Repository Contents

```
latent-synchronization-risk-var/
├── Latent_Synchronization_Risk___VaR_Underestimation.pdf   # Full paper
├── VAR_underestimation_code.py                             # Full replication code
├── README.md
├── requirements.txt
└── images/       (optional: key figures — scree plot, SF distribution, violation chart)
```

## Methodology Summary

- **Universe:** 169 S&P 500 stocks (of 200 candidates) passing a 99.5% data-completeness filter, spanning 20 GICS sectors
- **Synchronization Factor (SF):** ratio of the leading eigenvalue of a rolling 15-day correlation matrix to the Marchenko–Pastur upper bound (λ⁺)
- **Four VaR models compared:**
  - Model A — Normal (parametric Gaussian) VaR
  - Model B — Historical Simulation
  - Model C — EWMA-Filtered Historical Simulation (Hull & White, 1998)
  - Model D — **Eigenvalue-Adjusted VaR** (Normal VaR inflated by SF when SF > 1)
- **Calibration:** κ grid search on IS data (1999–2012) via Kupiec (1995) LR test; locked before OOS evaluation
- **Backtesting:** Kupiec (1995) unconditional coverage + Christoffersen (1998) conditional coverage/independence tests, per Basel III zones
- **Robustness checks:** event-window sensitivity (10/15/20/30 days), crisis-period sub-analysis (GFC, Eurozone Crisis, Oil Crash, COVID-19, Ukraine War)

## How to Run

```bash
pip install -r requirements.txt
python VAR_underestimation_code.py
```

The script pulls adjusted close prices via `yfinance` for the 1999–2025 window. The
`arch` package is optional (used for supplementary GARCH-based volatility checks); the
core RMT/VaR pipeline runs without it.

## Key Findings

| Model | OOS Violation Rate | Kupiec p | Basel Zone |
|---|---|---|---|
| A — Normal VaR | 2.20% | < 0.001 | RED |
| B — Historical Simulation | 2.05% | < 0.001 | RED |
| C — EWMA-Filtered HS | 1.84% | < 0.001 | RED |
| **D — Eigenvalue-Adjusted** | **1.28%** | **0.117** | **GREEN** |

- SF is significantly elevated during the GFC, Eurozone Crisis, Oil Crash, and Ukraine War, and reaches its third-highest peak during COVID-19 (Cohen's d = 1.455, p < 0.001, calm vs. stress regimes)
- High persistence: ACF(1) = 0.959, half-life ≈ 17 trading days
- All models fail the Christoffersen independence test — violations cluster in time even where the Eigenvalue-Adjusted model gets the *unconditional* frequency right
- COVID-19 remains a blind spot: the SF is designed to catch endogenous, leverage-driven contagion, not exogenous shocks like a pandemic-triggered crash

## Limitations

- Survivorship bias from the 99.5% completeness filter (excludes delisted/bankrupt tickers)
- Violation clustering unresolved by the level-based SF adjustment
- T < N rank-deficient correlation matrices (mitigated by using only the leading eigenvalue, not the full inverse)
- Exogenous shocks (e.g., COVID-19) fall outside the model's designed scope

## Citation

Singh, Y. (2026). *Latent Synchronization Risk and Value-at-Risk Underestimation: A Random
Matrix Theory Approach to Spectral Co-movement Detection.* SSRN Working Paper.

## Author

Yuvraj Singh — Graduate Researcher,
Department of Humanities & Social Sciences, IIT Roorkee.
