# Experiment 05 — intraday shape forecast

**Status: specification frozen 2026-09-24, before any method other than the forecast-only
optimal has been scored under it.**

Read the status carefully, because this document is doing two different things at once:

- **The forecast itself was chosen after an exploratory sweep on the evaluation weeks.** Four
  variants (7-day, 28-day, 28-day split by day type, and a 50/50 blend with the Fourier shape)
  were scored on the same 114 CISO and 333 NYISO weeks used for evaluation, and the best was
  kept. Its effect on the ceiling is therefore an optimistic, in-sample number. It is reported
  as an exploratory finding and must not be quoted as an out-of-sample estimate.
- **How the methods compare under it has not been looked at.** That is the part registered
  here, with falsifiable predictions, before the DP, the baselines or the agents are run.

## Why
The registered comparison established that all price-residual information is worth 18.4% (CISO)
and 29.0% (NYISO) of perfect foresight, that the DP captures none of it, and that even a perfect
two-hour-ahead forecast is worth nothing because a 4-hour battery commits over 12-18 hours
(docs/experiments/baselines.md). That leaves the deterministic forecast as the only lever worth
pulling: perfect knowledge of the within-day *shape* closes 87% (CISO) and 70% (NYISO) of the
remaining gap, while perfect knowledge of the price *level* is worth about $5/week.

## Specification
Frozen. `src/windows.shape_profile` and `src/windows.apply_shape`, with `SHAPE_DAYS = 28`.

- For each window, take the 28 days of prices ending at `train_end`. Subtract a centred 25-hour
  moving average from each hour, leaving the intraday shape with the level removed.
- Average those deviations by hour of day, separately for weekdays and weekends, giving a
  (2, 24) profile. If a day type has fewer than 24 observations, fall back to all days.
- The forecast is the Fourier seasonal curve's centred 25-hour moving average — its level
  trajectory — plus that profile. The Fourier fit keeps the level; the recent history supplies
  the shape.
- Causal: the profile uses only prices strictly before `train_end`, which is where the training
  window ends and the scored week begins. No evaluation-week data enters it.
- The same correction is applied to the pseudo-out-of-sample residuals used for OU calibration,
  so theta and sigma describe residuals from the forecast the policy actually uses.

Everything else is unchanged: walk-forward windows, the scoring rule, the replay harness, the
validity and no-lookahead gates, the block bootstrap, and the equivalence margins.

## Registered predictions
Recorded before running. The point of the re-run is that these can fail.

1. **The DP will remain at or below the new forecast-only ceiling in both markets.** The residual
   channel was worthless under the old forecast; a better deterministic forecast does not add
   information about the residual, so it should stay worthless. If the DP clears the new ceiling
   by a significant margin, the conclusion that modelling the residual does not pay is wrong.
2. **The fixed schedule will stay within roughly 5% of the new ceiling**, as it was under the
   old one (95.2% and 95.3%).
3. **The DP-versus-schedule verdict will stay inconclusive in both markets.**
4. **The OU calibration will barely move**, because the residual is dominated by the slow level,
   which this change does not touch.

## Reporting
Scored under docs/experiments/scoring_rule.md, both markets, all weeks, through the same harness.
Written to `results/historical_{market}_shape.csv`, leaving the registered results untouched, so
the two forecasts can be compared week by week. Every method is reported regardless of outcome,
and the predictions above are reported as met or failed.

## The RL agents
Both agents were retrained under this forecast on 2026-09-25, with the widened forecast-spread
priors specified below. Results are in the final section.

## Training under this forecast
Measured rather than assumed. Comparing the 114 CISO and 333 NYISO calibrations under both
forecasts, at p1 / median / p99: **q, mean(f)/q, half-life, sigma_stat, sigma/q and
mu/sigma_stat are unchanged to two decimals.** The OU calibration priors and both sets of
observation-normalisation constants (CISO_NORM, NYIS_NORM) therefore stay exactly as registered,
and `scripts/check_nyiso_setup.py` still passes.

One quantity does move: **the spread of the forecast itself**, because the shape forecast is
sharper and more variable than the Fourier curve.

| | old forecast | shape forecast | training support | weeks outside it, old -> shape |
|---|---|---|---|---|
| CISO std(f), p1-p99 | 8.6 - 17.8 | 5.4 - 27.1 | 7.1 - 22.1 | 0.9% -> **21.1%** |
| NYISO std(f)/q, p1-p99 | 0.16 - 0.60 | 0.10 - 0.68 | 0.15 - 0.62 | 0.9% -> **13.5%** |

Training on the old prior would leave a fifth of CISO evaluation weeks outside the distribution
the agent ever saw — the same failure this project already documented, where a mismatched prior
cost more than the choice of algorithm. So two bounds widen, and nothing else:

- `src.rl.sources.AMP_RANGE["shape"] = (7.0, 42.0)`, against the frozen `(10.0, 30.0)`.
- `src.rl.priors.NYIS_PRIOR_SHAPE` sets `f_std_over_q = (0.08, 0.80)`, against `(0.15, 0.62)`.

Both give 0.0% of weeks outside the training support under either forecast. The frozen base path
is untouched and bit-identical, selected by `--forecast base` (the default).

The bootstrap pool is also rebuilt: `src.rl.bootstrap.build_pool(market, shape=True)` subtracts
the shape forecast rather than the plain Fourier curve, so the pool carries residuals of the
forecast the policy will actually face. Pool residual std moves 31.2 -> 32.2 (CISO) and
12.0 -> 12.1 (NYISO); `scripts/check_bootstrap_source.py` passes on both.

Agents trained this way are scored with `scripts/eval_historical.py --shape-days 28`, which
writes to `results/historical_{market}_shape.csv` alongside the DP and baselines above.

---

## Results (run 2026-09-24)

`scripts/run_shape_forecast.py`, then `scripts/compare_forecasts.py`. Perfect foresight is
identical under both forecasts, as it must be, which the comparison script asserts.

| method | CISO old f | CISO shape f | change (95% CI) | NYISO old f | NYISO shape f | change (95% CI) |
|---|---|---|---|---|---|---|
| perfect foresight | $18,314 | $18,314 | — | $16,674 | $16,674 | — |
| forecast_optimal | $14,938 | **$16,428** | +$1,490 [+1,014, +2,034] | $11,839 | **$13,266** | +$1,427 [+1,015, +1,862] |
| DP | $14,559 | **$15,787** | +$1,228 [+659, +1,797] | $10,918 | **$12,375** | +$1,457 [+872, +2,107] |
| schedule | $14,216 | $15,029 | +$813 [+533, +1,108] | $11,279 | $11,762 | +$483 [+121, +879] |
| threshold | $5,240 | $5,240 | +$0 | $3,424 | $3,424 | +$0 |

Value capture rises from 79.5% to 86.2% (CISO) and 65.5% to 74.2% (NYISO) for the DP, and the
residual-blind ceiling from 81.6% to 89.7% and 71.0% to 79.6%. The threshold rule is unchanged
to the cent, as it must be — it never reads the forecast. That is the internal control on the
re-run.

### Registered predictions
| # | prediction | CISO | NYISO |
|---|---|---|---|
| 1 | DP stays at or below the new ceiling | **met** — −$641, CI [−1,249, −161] | **met** — −$892, CI [−1,482, −335] |
| 2 | schedule within ~5% of the new ceiling | **failed** — 91.5% | **failed** — 88.7% |
| 3 | DP vs schedule still inconclusive | met — −$758, CI [−1,576, +90] | **failed** — −$613, CI [−1,189, −10], schedule now underperforms |
| 4 | OU calibration barely moves | met — sigma_stat 21.6→21.4, half-life 17→18 h | met — 34.6→34.0, 39→39 h |

### What the failures mean
Predictions 2 and 3 failed together and for one reason: **the schedule's parity with the DP was
an artifact of a weak forecast.** The schedule collapses the forecast into a single repeating
daily pattern — the four cheapest and four dearest hour positions, averaged over the week — so a
sharper, day-type-aware forecast is information it structurally cannot use. The DP and the
forecast-only optimal read the full hourly path. Improving the forecast therefore widens the gap
between policies that use all of it and policies that use a daily average of it, and it restores
a reason to run an optimiser rather than a timetable. On NYISO the schedule now underperforms the
DP, though only just: the interval's upper end is −$10.

Prediction 1 held in both markets, which is the substantive result. A better deterministic
forecast lifts every forecast-using method by roughly $1,200-1,500 a week, but the DP still
sits below what is achievable while ignoring the residual entirely. Modelling the residual
remains worth less than nothing; the earlier conclusion survives a much better forecast.

### Caveats
- The forecast specification was selected on these same weeks (see Status). The +$1,400-1,500
  ceiling gain is in-sample and optimistic. The *comparison* between methods under it was not
  looked at before the predictions above were recorded.
- The RL agents are not included, for the reason given under Not covered.
- `results/historical_{market}_shape.csv` is a parallel result set. The registered results in
  `results/historical_{market}.csv` are unchanged and remain the reference for every claim in
  the README that is not explicitly labelled as coming from this experiment.

---

## Validation of the specification (2026-09-24)

The Status section flagged that the forecast was selected on the evaluation weeks. Two checks
bound how much that selection could be worth.

### 1. The gain on the half of the weeks not used for tuning
The variant sweep tuned on the first half of the weeks. Restricting the paired comparison to the
second half:

| | CISO full sample | CISO second half | NYISO full sample | NYISO second half |
|---|---|---|---|---|
| forecast_optimal | +$1,490 [+1,014, +2,034] | **+$1,028 [+493, +1,516]** | +$1,427 [+1,015, +1,862] | **+$1,678 [+1,087, +2,300]** |
| DP | +$1,228 [+659, +1,797] | +$864 [+340, +1,396] | +$1,457 [+872, +2,107] | +$1,539 [+598, +2,602] |
| schedule | +$813 [+533, +1,108] | +$524 [+244, +792] | +$483 [+121, +879] | +$337 [−175, +964] |

The gain survives in both markets with intervals clear of zero. CISO shrinks by about 30%,
NYISO grows. Treat the CISO full-sample number as the optimistic end of the range.

### 2. Rolling-origin selection of the discretionary choices
The spec makes two discretionary choices: a 28-day lookback and a weekday/weekend split. Both
were re-made causally — at each week, whichever of ten candidates (lookbacks 7/14/28/56/91, with
and without the day-type split) had earned the most on weeks strictly before it — and compared
with the registered fixed choice. The causal arm never sees the future. 26-week burn-in.

| | CISO (88 weeks) | NYISO (307 weeks) |
|---|---|---|
| registered spec, fixed | $13,786 (88.0% of PF) | $14,055 (79.6% of PF) |
| causally selected each week | $13,760 (87.9%) | $14,031 (79.5%) |
| best fixed choice in hindsight | $13,849 | $14,058 |
| causal − registered | −$26, CI [−160, +108] | −$24, CI [−187, +125] |
| registered spec's rank of 10, in hindsight | 2nd | 2nd |

**The discretionary choices are worth at most about $60 a week, against a gain of $1,000-1,700.**
Selection risk is bounded at roughly 4% of the effect. The registered spec ranks second of ten
in both markets and sits within $63 (CISO) and $3 (NYISO) of the best choice available with
hindsight.

The causal rule usually prefers a 14-day lookback (81 of 88 CISO weeks, 205 of 307 NYISO), not
28. **The spec is not being changed to match.** The difference is inside its own confidence
interval, the registered results already exist under 28 days, and re-specifying after seeing
which alternative a procedure favours is the behaviour this document exists to prevent. The
day-type split is confirmed: the causal rule selects a day-type variant in 88 of 88 CISO weeks
and 281 of 307 NYISO weeks.

### Still not validated
No weeks are genuinely untouched: every week in both markets was used to build the walk-forward
windows. The price CSVs end 2026-05-07 (CISO) and 2026-06-01 (NYISO), so a true out-of-sample
test is available by re-extracting from MotherDuck and scoring the weeks since. What the two
checks above establish is narrower but sufficient for the decision at hand: the specification's
free choices are not where the gain comes from.


---

## Results with the RL agents (2026-09-25)

Twenty runs — two markets, two training sources, five seeds — at 2,000,000 timesteps each under
`--forecast shape`, then scored with `--shape-days 28`. All twenty manifests record
`status: finished`, `git_dirty: false` at commit `a6ae81c`. The DP and baseline rows are
bit-identical to those written before the agents were added, no method beats perfect foresight,
and there are no mask violations.

| method | CISO | NYISO |
|---|---|---|
| perfect foresight | $18,314 (100%) | $16,674 (100%) |
| forecast_optimal | $16,428 (89.7%) | $13,266 (79.6%) |
| DP | $15,787 (86.2%) | $12,375 (74.2%) |
| schedule | $15,029 (82.1%) | $11,762 (70.5%) |
| RL, OU-trained | $14,889 (81.3%) | $10,830 (64.9%) |
| RL, bootstrap-trained | $14,656 (80.0%) | $10,303 (61.8%) |
| threshold | $5,240 (28.6%) | $3,424 (20.5%) |

Against the DP, seeds averaged, same paired block bootstrap:

| | CISO | NYISO |
|---|---|---|
| RL, OU-trained | −$898 [−1,561, −174], underperforms | −$1,545 [−1,998, −1,127], underperforms |
| RL, bootstrap-trained | −$1,131 [−1,789, −412], underperforms | −$2,072 [−2,653, −1,534], underperforms |
| bootstrap vs OU, head to head | −$233 [−361, −108] | −$527 [−771, −298] |

### What it changes
1. **Nothing reorders.** Every method gains $1,000-1,800 a week and the ranking is unchanged.
   Model-free RL still loses to model-based control in all four comparisons, every interval
   excluding zero. That result was not an artifact of a weak forecast.
2. **The widened prior worked.** The agents gained +$1,417 and +$1,013 (CISO) and +$1,585 and
   +$1,848 (NYISO) — in NYISO more than the DP gained — so they exploited the sharper forecast
   rather than ignoring it. Had the old prior been kept, a fifth of CISO evaluation weeks would
   have fallen outside the training distribution.
3. **Prediction 1 extends to the agents.** Neither clears the residual-blind ceiling; they sit
   8 to 15 points of value capture below it. Every method in this project that models the price
   residual still scores below a policy that ignores it.
4. **One verdict flips.** Bootstrap training is now worse than OU training in *both* markets. It
   was inconclusive on CISO under the old forecast (+$171, CI [−51, +442]); it is now −$233, CI
   [−361, −108]. "Training on real price paths did not help" strengthens from mixed to
   consistently negative.

The equivalence margins registered in docs/experiments/scoring_rule.md are ±$291 and ±$218, set
at 2% of the DP mean under the old forecast. Under this forecast 2% would be $316 and $247;
`check_margin` prints the discrepancy and the registered values continue to apply, since
re-deriving a margin from the results it will judge would defeat its purpose.
