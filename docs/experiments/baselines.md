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
