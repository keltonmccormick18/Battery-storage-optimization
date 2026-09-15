# Experiment 02 — Historical scoring rule

**Status:** registered before any trained agent is evaluated on historical data.

## Per-week score (every method: DP, perfect foresight, RL agents, baselines)
    score_w = Σ_{t=0}^{167} u_t · p_t · dt  +  q_w · SOC_168
- SOC_0 = 0. Planners still run the full 336-hour episode; only hours 0–167 are scored.
- q_w = mean price over week w's training window: the terminal value in the DP
  backward induction, the RL shaping potential, and the perfect-foresight terminal
  value. No information from after hour 168 is used.
- All methods except perfect foresight are replayed through BatteryEnv + ReplaySource.

Rejected: cash only (rewards liquidating before the cutoff; perfect foresight stops
being an upper bound, which happened in 93 of 333 NYISO weeks); forcing SOC_168 = 0
(changes the control problem the policies solve); valuing inventory at realized
later prices (uses post-cutoff information); carrying SOC across weeks (requires
restructuring the walk-forward backtest).

## Benchmark
PF_w = score_w for perfect foresight over hours 0–167 with terminal value q_w. This
is its optimization objective, so PF_w ≥ score_w for any feasible policy on the same
action set and SOC lattice.

## Metrics, reported per market, never pooled
- Mean weekly score
- Value capture = Σ_w score_w / Σ_w PF_w (headline); median per-week ratio (secondary).
  Mean of per-week ratios is not reported.
- Weekly Sharpe = mean / std of score_w, untrimmed. The old [5:-5] trimmed Sharpe
  appears only as a labeled continuity figure; trimming 5 weeks per tail removes 9% of
  CISO weeks but 3% of NYISO weeks, so it is not comparable across markets.
- Win rate = share of weeks with score_w > 0; worst week.

## Primary comparison: agent vs DP, per market and per training source
- Δ_w = (mean over seeds of agent score_w) − DP score_w, paired by week.
- Report mean Δ in dollars and as % of DP mean score, plus the share of weeks the agent beats DP.
- 95% CI: circular block bootstrap over weeks, block length ⌈n^(1/3)⌉
  (CISO 5, NYISO 7), 10,000 resamples, percentile interval, rng seed 0.
- This CI reflects week-to-week market uncertainty, not training-seed uncertainty.
  Seed variability is reported separately as the min–max of per-seed mean Δ.

## Claims
- **Outperforms DP:** CI lower bound > 0.
- **Underperforms DP:** CI upper bound < 0.
- **Equivalent to DP:** CI lies entirely within ±2% of DP mean weekly score
  (CISO ±$291, NYISO ±$218).
- Otherwise **inconclusive**, reported as such.

## Validity
A result is invalid (fix and rerun, do not report) if any week has mask violations
> 0, or any method's score_w exceeds PF_w by more than $0.01.

## Consequence for existing results
DP and perfect foresight are rescored under this rule. README value capture is
corrected: CISO 82% → 79.5%; NYISO 72% → 65.5% (ratio of totals), and NYISO
mean-of-ratios 86% → 69.5%. Previous figures used cash-only scoring.