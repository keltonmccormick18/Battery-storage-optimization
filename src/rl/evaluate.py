"""Historical evaluation on walk-forward windows.

Every method except perfect foresight is replayed through the same
BatteryEnv + ReplaySource path, so score differences come from decisions, not
from separate simulators. Scoring follows docs/experiments/scoring_rule.md:

    score = cash revenue over hours 0-167 + q * SOC at hour 168
"""
import numpy as np
import pandas as pd

from src.rl.env import BatteryEnv
from src.rl.sources import ReplaySource
from src.simulation import simulate, perfect_foresight, solve_dp
from src.windows import ROOT, BASE_PARAMS, SOC_GRID, EVAL_WINDOW


def window_params(w):
    return {**BASE_PARAMS, "q": w["q"]}


def replay_env(w):
    source = ReplaySource(w["X_eval_resid"], calib=(w["theta"], w["mu_eff"], w["sigma"]))
    return BatteryEnv(source, w["f_eval"], window_params(w), f_sampler=None)


def dp_action(policy, w, env):
    x_idx = np.argmin(np.abs(w["X_grid"] - env.X[env.t]))
    s_idx = np.argmin(np.abs(SOC_GRID - env.soc))
    u = policy[env.t, x_idx, s_idx]
    return 0 if u > 0 else (2 if u < 0 else 1)


def pf_outcome(w):
    """Perfect foresight over the scored week: (cash revenue, SOC at hour 168).

    Same price window and q as notebooks/03. Its objective is exactly the score,
    so it bounds every feasible policy from above.
    """
    return perfect_foresight(w["prices"][:EVAL_WINDOW], SOC_GRID, window_params(w), return_soc=True)


def result_row(market, w, method, revenue, soc_end, pf_score, violations):
    score = revenue + w["q"] * soc_end
    return {
        "market": market,
        "week_idx": w["week_idx"],
        "eval_start": w["eval_start"],
        "eval_start_ts": w["eval_start_ts"],
        "method": method,
        "revenue": float(revenue),          # cash, hours 0-167
        "soc_end": float(soc_end),          # MWh held at hour 168
        "score": float(score),              # revenue + q * soc_end
        "pf_score": float(pf_score),
        "value_capture": float(score / pf_score) if pf_score > 0 else float("nan"),
        "theta": float(w["theta"]),
        "sigma_stat": float(w["sigma_stat"]),
        "mu_eff": float(w["mu_eff"]),
        "q": float(w["q"]),
        "mask_violations": int(violations),
    }


def evaluate_dp(market, windows):
    """DP and perfect foresight on every window.

    Returns the results table and, for the reproduction gate, the DP cash revenue
    computed by the original simulate() path.
    """
    rows, sim_revenues = [], []
    for w in windows:
        policy = solve_dp(w)

        env = replay_env(w)
        env.reset(seed=0)
        soc_end = float("nan")
        for _ in range(env.T):
            if env.t == EVAL_WINDOW:
                soc_end = env.soc
            env.step(dp_action(policy, w, env))

        pf_cash, pf_soc = pf_outcome(w)
        pf_score = pf_cash + w["q"] * pf_soc
        rows.append(result_row(market, w, "dp", env.total_revenue, soc_end, pf_score,
                               env.n_mask_violations))
        rows.append(result_row(market, w, "pf", pf_cash, pf_soc, pf_score, 0))
        sim_revenues.append(simulate(policy, w["f_eval"], w["X_eval_resid"], w["X_grid"],
                                     SOC_GRID, window_params(w), T_eval=EVAL_WINDOW)[0])
    return pd.DataFrame(rows), np.array(sim_revenues)


def check_validity(table, tol=0.01):
    """Raise if any row breaks the validity rule: mask violations, or beating perfect foresight."""
    masked = table[table.mask_violations > 0]
    if len(masked):
        raise AssertionError(f"{len(masked)} rows with mask violations: {sorted(masked.method.unique())}")
    over = table[table.score > table.pf_score + tol]
    if len(over):
        raise AssertionError(f"{len(over)} rows score above perfect foresight, worst by "
                             f"${(over.score - over.pf_score).max():,.2f}: {sorted(over.method.unique())}")


def summarize(table, method):
    """Headline metrics for one method in one market."""
    t = table[table.method == method]
    s = t.score.values
    return {
        "weeks": len(t),
        "mean_score": s.mean(),
        "value_capture": s.sum() / t.pf_score.sum(),
        "median_weekly_vc": float(np.median(t.value_capture)),
        "sharpe": s.mean() / s.std(),
        "win_rate": float(np.mean(s > 0)),
        "worst_week": s.min(),
    }


def save_results(table, market, results_dir=None):
    """Write rows to results/historical_{market}.csv, replacing any earlier rows for the same methods."""
    path = (results_dir or ROOT / "results") / f"historical_{market}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path)
        table = pd.concat([old[~old["method"].isin(table["method"].unique())], table],
                          ignore_index=True)
    table.sort_values(["method", "week_idx"]).to_csv(path, index=False)
    return path
