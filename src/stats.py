"""Paired comparisons between methods, per docs/experiments/scoring_rule.md.

Weekly scores are strongly autocorrelated (lag-1 ~0.6 in both markets), so the
confidence interval comes from a circular block bootstrap over weeks rather than
an i.i.d. interval, which would be far too narrow.
"""
import re

import numpy as np
import pandas as pd

N_BOOT = 10_000
BOOT_SEED = 0
ALPHA = 0.05

# Equivalence margins fixed in docs/experiments/scoring_rule.md (2% of DP mean weekly score).
MARGIN = {"CISO": 291.0, "NYIS": 218.0}


def block_length(n):
    """Registered rule: ceil(n ** 1/3) -- 5 weeks for CISO, 7 for NYISO."""
    return int(np.ceil(n ** (1 / 3)))


def block_bootstrap_ci(x, block=None, n_boot=N_BOOT, seed=BOOT_SEED, alpha=ALPHA):
    """Percentile CI for the mean of x under a circular block bootstrap."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    block = block or block_length(n)
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n     # wrap around
    means = x[idx.reshape(n_boot, -1)[:, :n]].mean(axis=1)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def classify(lo, hi, margin):
    """The four verdicts allowed by the scoring rule."""
    if lo > 0:
        return "outperforms DP"
    if hi < 0:
        return "underperforms DP"
    if -margin < lo and hi < margin:
        return "equivalent to DP"
    return "inconclusive"


def method_groups(table):
    """Group seed variants: rl_ou_s0, rl_ou_s1, ... -> rl_ou. dp and pf are excluded."""
    groups = {}
    for m in sorted(table.method.unique()):
        if m in ("dp", "pf"):
            continue
        groups.setdefault(re.sub(r"_s\d+$", "", m), []).append(m)
    return groups


def paired_comparison(table, methods, market, baseline="dp"):
    """Paired weekly comparison of one policy against a baseline, both averaged over their seeds.

    baseline is a method name or, for a head-to-head between two trained policies, a list of
    the baseline's seed methods. The equivalence margin is the one registered for comparisons
    against the DP; applying it to another baseline is an analogy, not a registered rule.
    """
    wide = table.pivot(index="week_idx", columns="method", values="score")
    base_methods = [baseline] if isinstance(baseline, str) else list(baseline)
    base = wide[base_methods].mean(axis=1)
    delta = (wide[methods].mean(axis=1) - base).to_numpy()
    lo, hi = block_bootstrap_ci(delta)
    margin = MARGIN[market]
    per_seed = [float((wide[m] - base).mean()) for m in methods]
    return {
        "weeks": len(delta),
        "seeds": len(methods),
        "mean_delta": float(delta.mean()),
        "pct_of_baseline": float(delta.mean() / base.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "beats_baseline": float((delta > 0).mean()),
        "seed_min": min(per_seed),
        "seed_max": max(per_seed),
        "block": block_length(len(delta)),
        "verdict": classify(lo, hi, margin),
    }


def check_margin(table, market, baseline="dp"):
    """The registered margin should still be 2% of the DP mean; warn if the data moved."""
    dp_mean = table[table.method == baseline].score.mean()
    expected = 0.02 * dp_mean
    if abs(expected - MARGIN[market]) > 1.0:
        return (f"registered margin ${MARGIN[market]:,.0f} is no longer 2% of the DP mean "
                f"(${expected:,.0f}); the registered value is the one that applies")
    return None
