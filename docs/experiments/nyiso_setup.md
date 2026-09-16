# Experiment 04 — NYISO training setup

**Status:** registered before any NYISO agent is trained or evaluated.

## Scope
Applies to NYISO agents only. Every choice below was made from walk-forward calibration
data (the OU fits, seasonal forecasts and inventory values of the 333 NYISO windows); no
NYISO agent result existed.

- **CISO is unchanged.** CISO agents keep the frozen configuration (tag `rl-config-frozen`).
  `scripts/check_nyiso_setup.py` §A verifies that CISO training and replay environments
  reproduce the frozen tag exactly.
- **PPO hyperparameters are the frozen `src/config.py` values**, used unchanged.

## Why NYISO needs its own setup
1. **The inventory value q varies widely.** q spans 21–81 across NYISO eval weeks (CISO:
   35–54). CISO agents never observe q; they were trained with a fixed value. In NYISO a fixed
   training q would misplace the charge threshold (1−η)q/σ by 0.08σ at the median and 0.50σ at
   p95. The DP receives each week's q, so the agent should too.
2. **Price level, volatility and forecast amplitude co-move.** Log-correlations: q~σ_stat
   +0.79, q~std(f) +0.86, σ_stat~std(f) +0.73. The CISO prior samples these independently,
   which in NYISO would generate combinations that never occur (e.g. q ≈ 20 with σ_stat ≈ 130).

## Training scenario prior — `src/rl/priors.py`, `NYIS_PRIOR`
Draw q first; draw everything that scales with price as a ratio to q. In the NYISO windows
these ratios are close to independent of q and of each other (|r| ≤ 0.25).

| component | distribution | bounds | observed p1–p99 |
|---|---|---|---|
| q | log-uniform | 19 – 90 | 21.0 – 81.3 |
| half-life (h) | log-uniform, then θ ≥ 0.01 as in the pipeline | 8 – 127 | 10.2 – 69.3 |
| σ / q (diffusion σ) | log-uniform | 0.06 – 0.40 | 0.065 – 0.369 |
| std(f) / q | log-uniform | 0.15 – 0.62 | 0.158 – 0.598 |
| log(mean f / q) | normal, mean 0 | sd 0.33 | observed sd 0.333 |
| μ / σ_stat | uniform | ±0.9 | ±0.90 (the μ_eff clip) |

Construction notes:
- **Diffusion σ rather than σ_stat.** σ_stat/q correlates +0.59 with half-life, mechanically,
  because σ_stat = σ/√(2θ). σ/q correlates −0.07, so sampling σ reproduces the observed link
  without modeling it.
- **The pipeline's θ ≥ 0.01 clamp is applied**, and the half-life upper bound (127h) is set so
  the clamp binds for 21.9% of draws, matching 22% of NYISO windows.
- **Seasonal forecasts are redrawn if negative.** Real NYISO forecasts never go below 0
  (min(f)/q ≥ 0.098 at p1); independent draws would go negative 6.4% of the time.
- Forecast shape: daily harmonics 1–3 as for CISO, rescaled to the sampled level and std.

## Observation — 18 features
- The 17 CISO features, with the three calibration features standardized under the NYISO
  prior (`NYIS_NORM`: mean and std over 2,000,000 prior draws).
- **Plus log(q / σ_stat)**, standardized: the scale of the gap between the charge and
  discharge break-even prices relative to price noise. Price is already observed relative to q.

## Validation against real windows (`scripts/check_nyiso_setup.py`)
| | prior p1 / p50 / p99 | real p1 / p50 / p99 |
|---|---|---|
| q | 19.3 / 41.6 / 88.6 | 21.0 / 42.1 / 81.3 |
| half-life (h) | 8.2 / 32.3 / 69.3 | 10.2 / 32.8 / 69.3 |
| σ_stat | 5.1 / 30.0 / 168.2 | 5.0 / 29.4 / 127.1 |
| σ / q | 0.061 / 0.154 / 0.393 | 0.065 / 0.121 / 0.369 |
| std(f) / q | 0.152 / 0.294 / 0.607 | 0.158 / 0.199 / 0.598 |
| log(mean f / q) | −0.70 / +0.03 / +0.77 | −0.68 / −0.08 / +0.75 |
| min(f) / q | 0.025 / 0.567 / 1.748 | 0.098 / 0.486 / 1.721 |

- θ at clamp: prior 22.1%, real 21.6%. Negative forecasts: 0% in both.
- Co-movement (prior | real): q~σ_stat +0.57 | +0.79; q~std(f) +0.75 | +0.86;
  σ_stat~std(f) +0.43 | +0.73; half-life~σ_stat/q +0.55 | +0.59.
- Real windows standardized with `NYIS_NORM`: means within ±0.2, stds 0.71–0.92, max |z| 2.40.

## Known limitations
- **Flat densities between p1–p99 bounds over-weight large ratios.** Median std(f)/q is 0.29
  under the prior against 0.20 in reality, and median σ/q 0.15 against 0.12. Co-movement is
  reproduced in sign but weaker. This is the same bounds-only method used for CISO, kept
  deliberately: fitting density shapes would use more information from the evaluation period.
- **Bounds use all 333 windows**, so agents evaluated on early weeks were trained on a range
  partly informed by later calibrations. Same caveat as CISO; disclosed rather than corrected.
- **q is observed in NYISO but not CISO.** RL results should not be compared across markets
  without noting this.

## Training-time monitoring and evaluation
- Frozen PPO configuration. The OU multi-calibration evaluation for NYISO (20 cases drawn from
  `NYIS_PRIOR`) is monitored as a check that training works. It selects nothing.
- Historical evaluation follows `docs/experiments/scoring_rule.md`, using
  `ENV_KWARGS["NYIS"]` for the observation settings.
