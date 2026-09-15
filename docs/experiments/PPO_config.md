# Experiment 01 — PPO configuration freeze (CISO)

**Status:** registered, not yet run
**Registered:** see commit timestamp. Nothing below was written after results existed.

## Purpose
Choose the PPO configuration using only OU-simulated evaluation, before any
historical data is touched. After this experiment the configuration is frozen
(`src/rl/config.py`, tag `rl-config-frozen`) and used unchanged for all
historical evaluation.

## Motivation (from the prior 2M-step run, n_steps=1008)
- `train/explained_variance` ≈ 0.99 from 400k → critic is not the bottleneck; no gamma candidate.
- `train/approx_kl` ≈ 1e-3, `train/clip_fraction` ≈ 0.1 → update steps already small; LR decay dropped.
- Eval still improving at 1.7M → update count and training length are the open questions.

## Candidates
Shared: MaskablePPO, MlpPolicy, net_arch [128,128], gamma 1.0, gae_lambda 0.95,
ent_coef 0.01, n_epochs 10, constant learning rate 3e-4, 16 DummyVecEnv envs,
RandomizedOUSource + sample_f with CISO priors (src/rl/sources.py at this commit).

| | n_steps | batch_size | total_timesteps |
|---|---|---|---|
| A | 1008 | 504 | 2,000,000 |
| B | 512 | 512 | 3,000,000 |

Both use ~39k gradient steps per 2M timesteps; B makes ~2× as many policy updates.

## Seeds
Training seeds 0 and 1 for each candidate (4 runs).

## Decision metric
`eval/mc_pct_of_dp` = 100 × Σ agent revenue / Σ DP revenue over the
multi-calibration eval set:
- 20 (calibration, seasonal curve) cases drawn from the CISO prior, `default_rng(999)`
- 10 episodes per case, episode seeds `20000 + 100·case + k`
- deterministic policy, evaluated every 100,000 timesteps

**Score** at T timesteps = mean of the last 5 evals ending at T
(e.g. T = 2M → evals at 1.6M, 1.7M, 1.8M, 1.9M, 2.0M), averaged over the two seeds.

Logged but **diagnostic only, not used in the decision:**
`eval/sw_pct_of_dp`, `eval/mc_worst_case_pct`, `eval/sw_paired_diff_*`.

## Rules
1. **n_steps:** adopt B if Score(B, 2M) ≥ Score(A, 2M) − 1.0 pp; otherwise adopt A.
2. **total_timesteps** (only if B adopted): 3M if Score(B, 3M) ≥ Score(B, 2M) + 1.0 pp; otherwise 2M.
   If A is adopted, total_timesteps = 2M (A was not run longer).
3. **Seed disagreement:** if the per-seed differences Score(B,2M) − Score(A,2M) have
   opposite signs and both exceed 1.0 pp in magnitude, run seed 2 for both candidates
   and apply rules 1–2 to three-seed means.

## Validity
- Any run with `eval/mask_violations` > 0 at any eval is invalid: fix the bug, rerun the same seed.
- A crashed run is rerun with the same seed.
- No historical evaluation of any agent before this decision is recorded.

## Scoring procedure
```python
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def mc_series(run_dir, tag="eval/mc_pct_of_dp"):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0}); ea.Reload()
    return np.array([e.value for e in sorted(ea.Scalars(tag), key=lambda e: e.step)])

def score(run_dir, T, eval_freq=100_000, k=5):
    v = mc_series(run_dir)
    n = T // eval_freq                      # the n-th eval is at n × eval_freq
    assert len(v) >= n, f"{run_dir}: {len(v)} evals, need {n}"
    return v[n - k:n].mean()

A2 = np.mean([score(f"/content/runs/A_s{s}_1", 2_000_000) for s in (0, 1)])
B2 = np.mean([score(f"/content/runs/B_s{s}_1", 2_000_000) for s in (0, 1)])
B3 = np.mean([score(f"/content/runs/B_s{s}_1", 3_000_000) for s in (0, 1)])
```

## Outcome
Runs completed 2026-09-15. All valid: A 20 evals, B 30 evals, 0 mask violations.

| | seed 0 | seed 1 | mean |
|---|---|---|---|
| A @ 2M | 94.63 | 94.92 | 94.78 |
| B @ 2M | 94.25 | 93.70 | 93.98 |
| B @ 3M | 95.34 | 94.16 | 94.75 |

Per-seed B − A at 2M: −0.38, −1.22 (same sign; Rule 3 not triggered).

- Rule 1: 93.98 ≥ 94.78 − 1.0 = 93.78 → **adopt B** (n_steps 512, batch_size 512).
- Rule 2: 94.75 < 93.98 + 1.0 = 94.98 → **total_timesteps 2M**.

**Interpretation.** B was selected by the pre-registered tolerance, not because it
outperformed A: it scored 0.80 pp below A at 2M, lower on both seeds. With two
seeds the configurations are not distinguishable. Extending B to 3M added 0.77 pp,
below threshold, suggesting the remaining ~5% gap to DP reflects function
approximation rather than training budget.
