# Experiment 03 — Rule-based baselines

**Status:** registered before any baseline is scored on historical data.

## Purpose
Floors for the DP/RL comparison, chosen so each isolates one source of value. Both
rules are parameter-free: nothing is tuned on any data, so nothing can be fit to the
evaluation weeks.

## Definitions
One action per hour; any action infeasible at the current SOC becomes hold.

**schedule — timing from the forecast, no reaction to prices**
- For week w, average the seasonal forecast f_w at each hour position h = t mod 24
  over the 336-hour episode.
- Charge at the 4 positions with the lowest average, discharge at the 4 highest, hold
  otherwise, every day. 4 = S_max / u_max, the battery's duration.
- Never observes realized prices.

**threshold — reaction to price, no timing**
- Charge if p_t < η·q_w, discharge if p_t > q_w, hold otherwise, where p_t is the
  realized price at hour t and q_w is the week's inventory value.
- η·q_w and q_w are the break-even prices under the valuation every method is scored
  with; this is the greedy policy on the RL shaping reward.
- Never uses the forecast shape or any future price.

## What the comparison isolates
| method | seasonal timing | reacts to realized price | plans ahead under uncertainty |
|---|---|---|---|
| schedule | yes | no | no |
| threshold | no | yes | no |
| DP | yes | yes | yes (model-based) |
| RL | yes | yes | yes (learned) |

## Scoring and reporting
Scored under docs/experiments/scoring_rule.md, both markets, all weeks, through the
same replay harness as DP and RL. Both baselines are reported regardless of how they
compare. Definitions are not changed after results exist; any variant is reported as
a separate, labeled follow-up.

---

# Follow-up — forecast-only optimal

**Status: added 2026-09-23, after every other method had been scored.** This is a
diagnostic benchmark, not a registered claim, and it is labeled as such wherever it is
reported. It was added because the registered comparison left an obvious question
unanswered: the fixed schedule finished statistically indistinguishable from the DP in
both markets, and nothing in the registered set said how much of the achievable value
needs the price residual at all.

## Definitions
**forecast_optimal — the ceiling for any residual-blind policy**
- Backward induction over the week's seasonal forecast f_w, terminal value q_w · SOC,
  then executed against realized prices. Closed-loop in SOC, blind to price: it never
  reads X.
- f_w is fit on the training window, so it is known before the week starts. The policy
  is implementable, not an oracle.
- Planned over the 168 scored hours, which makes its objective exactly the scoring rule
  (as for perfect foresight, and unlike the DP, which plans over the full 336-hour
  episode).
- Any policy that ignores the residual scores at or below this, so it — not the fixed
  schedule — is the bar the DP must clear to show that modelling the residual pays.

**forecast_optimal_336 — horizon-matched variant**
- Identical, but planned over all 336 hours like the DP. The gap between the two is the
  cost of the longer planning horizon, which is not a cost of modelling the residual.

## Results
Mean weekly score, all weeks, same harness and scoring rule as every other method.

| | CISO (114 wk) | NYISO (333 wk) |
|---|---|---|
| perfect foresight | $18,314 (100%) | $16,674 (100%) |
| **forecast_optimal** | **$14,938 (81.6%)** | **$11,839 (71.0%)** |
| forecast_optimal_336 | $14,847 (81.1%) | $11,646 (69.8%) |
| DP | $14,559 (79.5%) | $10,918 (65.5%) |
| schedule | $14,216 (77.6%) | $11,279 (67.6%) |

Paired weekly differences, circular block bootstrap, same statistic as the registered
comparison (the verdict wording is registered for comparisons against the DP; applying
it here is an analogy):

| comparison | CISO | NYISO |
|---|---|---|
| forecast_optimal − DP | +$379, CI [−22, +893], inconclusive | +$922, CI [+260, +1,626], outperforms DP |
| forecast_optimal_336 − DP | +$289, CI [−150, +842], inconclusive | +$728, CI [+91, +1,417], outperforms DP |
| schedule − forecast_optimal | −$722, CI [−1,330, −171] | −$560, CI [−1,029, −121] |

## What it shows
1. **Most of the achievable value is deterministic.** All residual information is worth
   18.4% (CISO) and 29.0% (NYISO) of perfect foresight. The rest is diurnal shape that
   the seasonal forecast already supplies.
2. **The fixed schedule is not a weak baseline.** It captures 95% of the residual-blind
   ceiling in both markets, which is why it finished level with the DP.
3. **The DP's residual modelling has negative measured value.** It lands below the
   ceiling in both markets, significantly so on NYISO. Horizon-matched, the shortfall is
   $289 (CISO) and $728 (NYISO) per week.

## Why the residual model does not pay
The fitted OU forecasts the raw residual well — CISO RMSE $8.18 vs $17.03 unconditional
one hour ahead. But a 4-hour battery commits over 12–18 hours (trough to peak), and
arbitrage depends on price *spreads*, not levels: adding a constant to every price of a
day changes no decision. Splitting the residual into its centred 24-hour local level and
the deviation from it, on the deviation — the only part arbitrage can monetise — the
fitted OU is **worse than assuming the residual away** at every horizon past ~4 hours:

| RMSE ($/MWh), level removed | h=1 | h=4 | h=12 | h=18 |
|---|---|---|---|---|
| CISO — fitted OU | 8.13 | 13.64 | 15.39 | 13.09 |
| CISO — unconditional mean | 11.37 | 11.46 | 11.61 | 11.63 |
| NYISO — fitted OU | 7.94 | 16.20 | 17.42 | 15.67 |
| NYISO — unconditional mean | 12.31 | 12.33 | 12.24 | 11.84 |

The fitted half-lives (17 h CISO, 39 h NYISO) are far longer than the ~4–6 h at which the
level-removed residual actually decorrelates, so the DP extrapolates the current deviation
across its whole planning horizon and acts on a signal that has already died. Sweeping the
assumed half-life in the near-deterministic limit confirms the direction: on CISO, a DP
that treats the residual as permanent reproduces forecast_optimal (+$15), the fitted value
gives −$409, and 4.2 h gives −$1,710. See figures/residual_forecast_skill.png and
figures/value_ceiling.png.

## Caveats
- Post-hoc. It does not change any registered verdict; it adds context to them.
- The no-lookahead gate is vacuous for this policy: it responds to changed future prices
  in 0% of cases because it never reads prices. The gate is passed by construction.
- The DP's 336-hour planning horizon accounts for roughly a quarter of its shortfall
  against the 168-hour ceiling; forecast_optimal_336 is reported so that the two causes
  are not conflated.
