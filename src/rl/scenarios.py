"""Market-aware training scenarios and training-time evaluation cases.

No SB3 imports, so everything here runs and is tested locally; src/rl/training.py adds
the SB3 wrappers. Settings are registered in docs/experiments/nyiso_setup.md (NYISO) and
match experiment 01 exactly for CISO.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from src.windows import ROOT, BASE_PARAMS, SOC_GRID
from src.price_model import fit_seasonal_fourier, build_fourier_features, estimate_ou_params
from src.optimization import build_transition_matrix, get_optimal_policy
from src.rl.env import BatteryEnv
from src.rl.sources import OUSource, RandomizedOUSource, sample_calib, sample_f
from src.rl.priors import NYIS_PRIOR, ENV_KWARGS, draw_scenario, make_episode_sampler
from src.rl.evaluate import rollout_batch
from src.rl.bootstrap import make_bootstrap_sampler

MARKETS = ("CISO", "NYIS")
SOURCES = ("ou", "bootstrap")
EPISODE_HOURS = 336
MC_SEED = 999                     # multi-calibration cases, as in experiment 01
N_MC_CASES = 20
EPISODES_PER_CASE = 10
N_SW_EPISODES = 200


@lru_cache(maxsize=None)
def ciso_gate2():
    """The CISO Gate 2 window used in experiment 01 (same computation as scripts/run_config.py).

    Returns (params, f_eval, theta, mu_eff, sigma). Its q is the fixed q CISO agents train with.
    """
    ciso = pd.read_csv(ROOT / "data" / "prices_CISO.csv")
    ciso["hour"] = pd.to_datetime(ciso["hour"])
    ciso["hour_of_day"] = ciso["hour"].dt.hour
    ciso["dow"] = ciso["hour"].dt.dayofweek
    ciso["month"] = ciso["hour"].dt.month
    train_window, eval_window = 24 * 365, 24 * 7
    mid = len(ciso) // 2

    full_train = ciso.iloc[mid - train_window:mid].copy()
    seasonal_model, feature_cols = fit_seasonal_fourier(full_train)
    eval_data = ciso.iloc[mid:mid + eval_window * 2].copy()
    f_eval = seasonal_model.predict(build_fourier_features(eval_data, feature_cols)).values
    q = full_train["price_usd_mwh"].mean()

    split = mid - int(train_window * 0.25)
    inner, cols_inner = fit_seasonal_fourier(ciso.iloc[mid - train_window:split].copy())
    val = ciso.iloc[split:mid].copy()
    val_resid = val["price_usd_mwh"].values - inner.predict(build_fourier_features(val, cols_inner)).values
    theta, _, sigma = estimate_ou_params(pd.Series(val_resid - val_resid.mean()))
    theta = max(theta, 0.01)
    sigma_stat = sigma / np.sqrt(2 * theta)

    lookback = ciso.iloc[mid - eval_window:mid].copy()
    L_prev = (lookback["price_usd_mwh"].values
              - seasonal_model.predict(build_fourier_features(lookback, feature_cols)).values).mean()
    L_prev = np.clip(L_prev, -1.5 * sigma_stat, 1.5 * sigma_stat)
    mu_eff = 0.6 * L_prev

    params = {**BASE_PARAMS, "q": q}
    return params, f_eval, theta, mu_eff, sigma


def training_params(market):
    """CISO trains with the fixed Gate 2 q. NYISO redraws q every episode; this value is a placeholder."""
    if market == "CISO":
        return dict(ciso_gate2()[0])
    return {**BASE_PARAMS, "q": 42.0}


def scenario_drawer(market, params):
    """Draw q, an OU calibration and a seasonal forecast from the market's training prior.

    The OU and bootstrap sources share this, so they differ only in the price path.
    """
    if market == "CISO":
        q = params["q"]

        def draw(rng, T):
            return {"q": q, "calib": sample_calib(rng), "f": sample_f(rng, T, q)}
    elif market == "NYIS":
        def draw(rng, T):
            return draw_scenario(rng, NYIS_PRIOR, T)
    else:
        raise ValueError(f"unknown market {market!r}")
    return draw


def make_training_env(market, source, params):
    """One raw (unwrapped) training environment."""
    if source not in SOURCES:
        raise ValueError(f"unknown training source {source!r}; available: {SOURCES}")
    if market not in MARKETS:
        raise ValueError(f"unknown market {market!r}")
    if source == "bootstrap":
        sampler = make_bootstrap_sampler(market, scenario_drawer(market, params))
        return BatteryEnv(None, np.zeros(EPISODE_HOURS), params, episode_sampler=sampler,
                          **ENV_KWARGS[market])
    if market == "CISO":
        f_init = sample_f(np.random.default_rng(), EPISODE_HOURS, params["q"])   # replaced at first reset
        return BatteryEnv(RandomizedOUSource(), f_init, params, f_sampler=sample_f)
    if market == "NYIS":
        return BatteryEnv(None, np.zeros(EPISODE_HOURS), params,
                          episode_sampler=make_episode_sampler(NYIS_PRIOR), **ENV_KWARGS["NYIS"])
    raise ValueError(f"unknown market {market!r}")


# ---------------- training-time evaluation cases ----------------

def dp_lookup(policy, X_grid, env):
    x_idx = np.argmin(np.abs(X_grid - env.X[env.t]))
    s_idx = np.argmin(np.abs(SOC_GRID - env.soc))
    u = policy[env.t, x_idx, s_idx]
    return 0 if u > 0 else (2 if u < 0 else 1)


def case_envs(case):
    return [BatteryEnv(case["source"], case["f"], case["params"], f_sampler=None, **case["env_kwargs"])
            for _ in case["seeds"]]


def make_case(label, calib, f, q, seeds, env_kwargs, keep_policy=False):
    """An OU evaluation case with its DP reference (cash revenue and SOC at hour 168 per seed).

    The DP policy (~44 MB) is discarded unless keep_policy is set.
    """
    theta, mu, sigma = calib
    params = {**BASE_PARAMS, "q": q}
    sigma_stat = sigma / np.sqrt(2 * theta)
    X_grid = np.linspace(mu - 5 * sigma_stat, mu + 5 * sigma_stat, 200)
    policy = get_optimal_policy(build_transition_matrix(theta, mu, sigma, X_grid), f, X_grid, SOC_GRID, params)
    case = {"label": label, "source": OUSource(theta, mu, sigma), "f": f, "params": params,
            "seeds": list(seeds), "env_kwargs": env_kwargs}
    out = rollout_batch(case_envs(case), lambda obs, masks, envs: [dp_lookup(policy, X_grid, e) for e in envs],
                        case["seeds"])
    case["dp_revenue"], case["dp_soc_end"] = out["revenue"], out["soc_end"]
    if keep_policy:
        case["policy"], case["X_grid"] = policy, X_grid
    return case


def build_eval_cases(market, params, n_cases=N_MC_CASES, keep_policy=False):
    """Multi-calibration cases drawn from the market's training prior; plus, for CISO only,
    the Gate 2 single-window case kept for continuity with experiment 01."""
    env_kwargs = ENV_KWARGS[market]
    rng = np.random.default_rng(MC_SEED)
    mc = []
    for c in range(n_cases):
        if market == "CISO":
            calib = sample_calib(rng)
            f = sample_f(rng, EPISODE_HOURS, params["q"])
            q = params["q"]
        else:
            s = draw_scenario(rng, NYIS_PRIOR, EPISODE_HOURS)
            calib, f, q = s["calib"], s["f"], s["q"]
        seeds = [20_000 + 100 * c + k for k in range(EPISODES_PER_CASE)]
        mc.append(make_case(f"mc{c}", calib, f, q, seeds, env_kwargs, keep_policy))
    sw = None
    if market == "CISO":
        _, f_eval, theta, mu_eff, sigma = ciso_gate2()
        sw = make_case("gate2", (theta, mu_eff, sigma), f_eval, params["q"], range(N_SW_EPISODES),
                       env_kwargs, keep_policy)
    return {"market": market, "mc": mc, "sw": sw}


def evaluate_cases(cases, predict_fn):
    """Score a policy on every case, paired with the cached DP references.

    mc_score_pct_of_dp uses the registered scoring rule (cash + q * SOC_168); mc_pct_of_dp is
    cash only, the metric logged in experiment 01, kept for comparable training curves.
    """
    mc = cases["mc"]
    n_eps = {len(c["seeds"]) for c in mc}
    assert len(n_eps) == 1, "all multi-calibration cases need the same number of episodes"
    envs = [e for c in mc for e in case_envs(c)]
    out = rollout_batch(envs, predict_fn, [s for c in mc for s in c["seeds"]])

    rev = out["revenue"].reshape(len(mc), -1)
    soc = out["soc_end"].reshape(len(mc), -1)
    dp_rev = np.stack([c["dp_revenue"] for c in mc])
    dp_soc = np.stack([c["dp_soc_end"] for c in mc])
    q = np.array([c["params"]["q"] for c in mc])[:, None]
    score, dp_score = rev + q * soc, dp_rev + q * dp_soc
    metrics = {
        "mc_score_pct_of_dp": float(100 * (score.sum() / dp_score.sum())),
        "mc_worst_case_score_pct": float((100 * (score.mean(axis=1) / dp_score.mean(axis=1))).min()),
        "mc_pct_of_dp": float(100 * (rev.sum() / dp_rev.sum())),
        "mask_violations": int(out["violations"].sum()),
    }
    sw = cases["sw"]
    if sw is not None:
        o = rollout_batch(case_envs(sw), predict_fn, sw["seeds"])
        diff = o["revenue"] - sw["dp_revenue"]
        metrics.update({
            "sw_pct_of_dp": float(100 * (o["revenue"].mean() / sw["dp_revenue"].mean())),
            "sw_paired_diff_mean": float(diff.mean()),
            "sw_paired_diff_se": float(diff.std(ddof=1) / np.sqrt(len(diff))),
        })
        metrics["mask_violations"] += int(o["violations"].sum())
    return metrics
