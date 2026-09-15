"""Historical evaluation on walk-forward windows.

Every method is scored through the same BatteryEnv + ReplaySource path, so
revenue differences come from decisions, not from separate simulators.
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


def pf_revenue(w):
    """Same definition as notebooks/03: realized prices over the eval week, q from training."""
    return perfect_foresight(w["prices"][:EVAL_WINDOW], SOC_GRID, window_params(w))


def result_row(market, w, method, revenue, pf, violations, soc_end=float("nan")):
    return {
        "market": market,
        "week_idx": w["week_idx"],
        "eval_start": w["eval_start"],
        "eval_start_ts": w["eval_start_ts"],
        "method": method,
        "revenue": float(revenue),
        "pf_revenue": float(pf),
        "value_capture": float(revenue / pf) if pf > 0 else 0.0,
        "theta": float(w["theta"]),
        "sigma_stat": float(w["sigma_stat"]),
        "mu_eff": float(w["mu_eff"]),
        "q": float(w["q"]),
        "mask_violations": int(violations),
        "soc_end": float(soc_end),   # SOC at hour 168; not credited in revenue
    }


def evaluate_dp(market, windows):
    """DP and perfect foresight on every window.

    Returns the results table and, for the reproduction gate, the DP revenue
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

        pf = pf_revenue(w)
        rows.append(result_row(market, w, "dp", env.total_revenue, pf, env.n_mask_violations, soc_end))
        rows.append(result_row(market, w, "pf", pf, pf, 0))
        sim_revenues.append(simulate(policy, w["f_eval"], w["X_eval_resid"], w["X_grid"],
                                     SOC_GRID, window_params(w), T_eval=EVAL_WINDOW)[0])
    return pd.DataFrame(rows), np.array(sim_revenues)


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