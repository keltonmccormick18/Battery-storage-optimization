# Experiment 05 — Bootstrap training source

**Status:** registered before any bootstrap agent is trained or evaluated.

## Question
Does training on real residual paths beat training on OU-simulated ones, evaluated on the
same historical weeks? The OU model cannot produce spikes or sustained level drift, and the
README already identifies sustained trends as the source of most of the DP's shortfall
against perfect foresight. This is the experiment that tests whether that matters.

## Design
Shape comes from the data; scale and labels come from the market's prior. Each episode:

1. Draws q, the calibration (θ, μ, σ) and the seasonal forecast **from the same prior the OU
   source uses** — the frozen sampler for CISO, `NYIS_PRIOR` for NYISO.
2. Takes a contiguous 336-hour block of real residuals from the pool.
3. Rescales it by a single global factor, σ_stat(drawn) / σ_stat(pool), and centres it on μ.
4. Gives the agent the **drawn** calibration, never a fit to the block.

Rationale:
- **One global scale factor, not per block.** Blocks keep their real relative dispersion:
  some weeks calm, some violent, exactly as evaluation weeks differ.
- **Labels from the prior** make the calibration observations identically distributed for the
  OU and bootstrap agents, so the only difference between them is the path shape.
- **The label is deliberately wrong for the path**, as it is at evaluation, where the DP's
  calibration comes from a year-long fit and the realized week does something else. The OU
  agent never meets that mismatch; this one trains under it.
- **OU fits to individual blocks are not used.** They give half-lives around 5–7.5h, while
  evaluation calibrations come from year-long fits (median 11h CISO, 33h NYISO).
- **Contiguous blocks**, not stitched shorter ones, because preserving week-long trends is the point.

## Pool and leakage boundary
Residuals of the hours before the first evaluation week — the first training window, 8,760
hours — against a seasonal model fitted to exactly those hours.

| market | pool | ends | first scored week | q_pool | σ_stat(pool) |
|---|---|---|---|---|---|
| CISO | 8,760 h | 2024-02-20 07:00 | 2024-02-20 08:00 | 55.0 | 31.2 |
| NYISO | 8,760 h | 2020-01-01 04:00 | 2020-01-01 05:00 | 28.9 | 12.0 |

No hour an agent trains on postdates any week it is scored on, for every week.

## Validation (`scripts/check_bootstrap_source.py`)
| | bootstrap | OU | real weeks |
|---|---|---|---|
| CISO innovation excess kurtosis | 401.5 | 2.3 | 115.3 |
| CISO daily drift autocorrelation | +0.63 | +0.28 | +0.56 |
| CISO realized 168h std ÷ labelled σ_stat | 0.36 | 0.84 | 0.51 |
| NYISO innovation excess kurtosis | 26.0 | 1.4 | 123.0 |
| NYISO daily drift autocorrelation | +0.58 | +0.46 | +0.62 |
| NYISO realized 168h std ÷ labelled σ_stat | 0.38 | 0.72 | 0.34 |

Also verified: identical q, calibration and forecast from the same seed (only the path differs);
environments finite, deterministic and free of mask violations in both markets.

## Known limitations
- **Shape comes from a single year.** The leakage rule allows only the first training window.
  CISO's pool year is more spike-heavy than the average evaluation week (kurtosis 401 against
  115); NYISO's 2019 is calmer (26 against 123) and is the cheapest year in the sample
  (mean price 28.9 against 84.4 in 2022). Prior-drawn scale covers the volatility range, but
  tail thickness and trend structure come from that one regime.
- **Roughly 26 independent two-week stretches** underlie the ~8,400 overlapping blocks.
- **Training-time monitoring still uses OU cases**, so a bootstrap agent may score lower there
  without that meaning anything. The historical evaluation is the comparison that counts.
- An alternative design — pooling several years and scoring only later weeks — would enlarge
  the pool at the cost of evaluation weeks. Not chosen; worth revisiting as a robustness check.

## Evaluation
Trained per market with seeds 100–104 and the frozen config, scored under
`docs/experiments/scoring_rule.md` on the same weeks as the OU agents, named `rl_boot_s<seed>`.
The comparison of interest is bootstrap against OU on identical weeks, by paired weekly
differences with the registered block bootstrap.
