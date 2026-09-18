"""Local checks for production training (no SB3 needed).

A. The CISO Gate 2 window reproduces experiment 01's calibration.
B. The CISO training environment matches experiment 01's construction.
C. CISO evaluation cases: DP references identical to experiment 01's callback computation.
D. evaluate_cases scores the DP itself at exactly 100% (both markets): batching order,
   per-case q, SOC at hour 168, pairing.
E. NYISO: 18 observations, q redrawn per episode and per case, no Gate 2 window.
F. Cost of one evaluation's environment stepping, with a cheap stand-in policy.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.optimization import build_transition_matrix, get_optimal_policy
from src.rl.env import BatteryEnv
from src.rl.sources import OUSource, RandomizedOUSource, sample_calib, sample_f
from src.rl.scenarios import (ciso_gate2, training_params, make_training_env, build_eval_cases,
                              evaluate_cases, case_envs, dp_lookup, N_MC_CASES, EPISODES_PER_CASE)
from src.baselines import threshold_predict_fn

SOC_GRID = np.linspace(0, 100, 81)


def section(t):
    print(f"\n== {t}", flush=True)


def check_gate2():
    section("A. CISO Gate 2 window")
    params, f_eval, theta, mu_eff, sigma = ciso_gate2()
    got = (round(theta, 4), round(mu_eff, 2), round(sigma, 2), round(params["q"], 2))
    assert got == (0.3021, -0.23, 22.20, 43.78), got
    print(f"   PASS: theta={got[0]} mu_eff={got[1]} sigma={got[2]} q={got[3]} (experiment 01 printed the same)")


def rollout(env, seed, n=2):
    rng = np.random.default_rng(seed + 1000)
    log = []
    for e in range(n):
        obs, _ = env.reset(seed=seed if e == 0 else None); log.append(obs)
        for _ in range(env.T):
            m = np.array(env.action_masks())
            obs, r, *_ = env.step(int(rng.choice(np.flatnonzero(m)))); log.append(np.append(obs, r))
    return log


def check_ciso_env():
    section("B. CISO training environment vs experiment 01 construction")
    params = training_params("CISO")
    legacy = BatteryEnv(RandomizedOUSource(), sample_f(np.random.default_rng(), 336, params["q"]),
                        params, f_sampler=sample_f)                  # src/rl/train.py make_env_fn
    new = make_training_env("CISO", "ou", params)
    for seed in (0, 3):
        a, b = rollout(legacy, seed), rollout(new, seed)
        assert all(np.array_equal(x, y) for x, y in zip(a, b)), f"differs at seed {seed}"
    print(f"   PASS: identical observations and rewards (2 seeds x 2 episodes x 336 steps); q={params['q']:.2f}")


def check_ciso_references(cases):
    section("C. CISO evaluation cases vs experiment 01's callback")
    params = training_params("CISO")

    def legacy_run_dp(src, f, Xg, pol, seed):                         # Gate2EvalCallback._run_dp
        env = BatteryEnv(src, f, params, f_sampler=None)
        env.reset(seed=seed)
        for _ in range(env.T):
            x_idx = np.argmin(np.abs(Xg - env.X[env.t]))
            s_idx = np.argmin(np.abs(SOC_GRID - env.soc))
            u = pol[env.t, x_idx, s_idx]
            env.step(0 if u > 0 else (2 if u < 0 else 1))
        return env.total_revenue

    _, f_eval, theta, mu, sigma = ciso_gate2()
    ss = sigma / np.sqrt(2 * theta)
    Xg = np.linspace(mu - 5 * ss, mu + 5 * ss, 200)
    pol = get_optimal_policy(build_transition_matrix(theta, mu, sigma, Xg), f_eval, Xg, SOC_GRID, params)
    src = OUSource(theta, mu, sigma)
    sw = np.array([legacy_run_dp(src, f_eval, Xg, pol, s) for s in range(200)])
    assert np.array_equal(sw, cases["sw"]["dp_revenue"]), "Gate 2 DP references differ"

    rng = np.random.default_rng(999)
    for c in range(20):
        th, m, sg = sample_calib(rng)
        f = sample_f(rng, 336, params["q"])
        ss = sg / np.sqrt(2 * th)
        Xg = np.linspace(m - 5 * ss, m + 5 * ss, 200)
        pol = get_optimal_policy(build_transition_matrix(th, m, sg, Xg), f, Xg, SOC_GRID, params)
        src = OUSource(th, m, sg)
        ref = np.array([legacy_run_dp(src, f, Xg, pol, 20_000 + 100 * c + k) for k in range(10)])
        assert np.array_equal(ref, cases["mc"][c]["dp_revenue"]), f"case {c} DP references differ"
    print("   PASS: all 20 multi-calibration cases and the 200-episode Gate 2 window identical")


def dp_as_policy(cases):
    lookups = {
        "mc": [(c["policy"], c["X_grid"]) for c in cases["mc"] for _ in c["seeds"]],
        "sw": [(cases["sw"]["policy"], cases["sw"]["X_grid"])] * len(cases["sw"]["seeds"]) if cases["sw"] else [],
    }
    def predict(obs, masks, envs):
        table = lookups["mc"] if len(envs) == len(lookups["mc"]) else lookups["sw"]
        return [dp_lookup(p, xg, e) for (p, xg), e in zip(table, envs)]
    return predict


def check_metrics(market):
    params = training_params(market)
    cases = build_eval_cases(market, params, n_cases=3, keep_policy=True)
    m = evaluate_cases(cases, dp_as_policy(cases))
    assert m["mc_score_pct_of_dp"] == 100.0 and m["mc_pct_of_dp"] == 100.0, m
    assert m["mc_worst_case_score_pct"] == 100.0 and m["mask_violations"] == 0, m
    if cases["sw"]:
        assert m["sw_pct_of_dp"] == 100.0 and m["sw_paired_diff_mean"] == 0.0, m
    print(f"   PASS: {market} -- DP scored as the policy gives exactly 100% on every metric "
          f"({len(cases['mc'])} cases{' + Gate 2' if cases['sw'] else ''})")
    return cases


def check_nyiso(cases):
    section("E. NYISO environments and cases")
    env = make_training_env("NYIS", "ou", training_params("NYIS"))
    qs = []
    for s in range(4):
        obs, _ = env.reset(seed=s); qs.append(env.q)
        assert obs.shape == (18,) and np.isfinite(obs).all()
    case_q = [c["params"]["q"] for c in cases["mc"]]
    assert len(set(case_q)) == len(case_q) and cases["sw"] is None
    assert case_envs(cases["mc"][0])[0].reset(seed=0)[0].shape == (18,)
    print(f"   PASS: 18 observations; q per episode {', '.join(f'{q:.1f}' for q in qs)}; "
          f"q per case {', '.join(f'{q:.1f}' for q in case_q)}; no Gate 2 window")


def main():
    check_gate2()
    check_ciso_env()
    t0 = time.time(); full = {"CISO": build_eval_cases("CISO", training_params("CISO"))}
    build_ciso = time.time() - t0
    check_ciso_references(full["CISO"])
    section("D. Metric pipeline")
    check_metrics("CISO")
    nyis_small = check_metrics("NYIS")
    check_nyiso(nyis_small)

    section("F. Cost of one evaluation (environment stepping, stand-in policy)")
    t0 = time.time(); full["NYIS"] = build_eval_cases("NYIS", training_params("NYIS")); build_nyis = time.time() - t0
    for market, built in [("CISO", build_ciso), ("NYIS", build_nyis)]:
        t0 = time.time(); m = evaluate_cases(full[market], threshold_predict_fn())
        n_ep = N_MC_CASES * EPISODES_PER_CASE + (200 if full[market]["sw"] else 0)
        print(f"   {market}: cases built in {built:.0f}s; one evaluation ({n_ep} episodes) {time.time() - t0:.1f}s; "
              f"threshold stand-in scores {m['mc_score_pct_of_dp']:.1f}% of DP")
    print("\nall checks passed")


if __name__ == "__main__":
    main()
