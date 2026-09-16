"""Checks for the NYISO training setup (docs/experiments/nyiso_setup.md).

A. CISO is untouched: with default settings, training and replay environments produce
   identical observations, rewards and masks to the frozen tag rl-config-frozen.
B. The NYISO prior reproduces the real joint distribution of the walk-forward calibrations.
C. NYIS_NORM matches a fresh Monte Carlo of the prior, and standardizes real windows sensibly.
D. NYISO environments run: 18 observations, finite, deterministic, no mask violations.
E. NYISO baseline scores are unchanged under the NYISO environment settings.
"""
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.windows import get_windows, BASE_PARAMS
from src.rl.env import BatteryEnv
from src.rl.priors import (NYIS_PRIOR, NYIS_NORM, ENV_KWARGS, THETA_MIN, draw_scenario,
                           make_episode_sampler, norm_constants)
from src.rl.evaluate import replay_env, evaluate_agent
from src.baselines import BASELINES

TAG = "rl-config-frozen"

DUMPER = r'''
import sys, numpy as np
root, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
from src.rl.env import BatteryEnv
from src.rl.sources import RandomizedOUSource, ReplaySource, sample_f
params = {"u_max": 25, "eta": 0.85, "S_max": 100, "S_0": 0, "dt": 1, "q": 43.78}
rec = {}
def rollout(env, reset_seed, action_seed, tag):
    obs, _ = env.reset(seed=reset_seed)
    rng = np.random.default_rng(action_seed)
    O, R, M, V = [obs], [], [], []
    for _ in range(env.T):
        m = np.array(env.action_masks()); M.append(m)
        obs, r, term, trunc, info = env.step(int(rng.choice(np.flatnonzero(m))))
        O.append(obs); R.append(r); V.append(info["revenue"])
    for k, v in {"obs": O, "rew": R, "mask": M, "rev": V, "X": env.X, "f": env.f}.items():
        rec[f"{tag}_{k}"] = np.array(v)
train = BatteryEnv(RandomizedOUSource(), sample_f(np.random.default_rng(123), 336, params["q"]),
                   params, f_sampler=sample_f)
rollout(train, 0, 10, "train_seeded")
rollout(train, None, 11, "train_continued")
rollout(train, 5, 12, "train_reseeded")
t = np.arange(336)
f = 40 + 12 * np.sin(2 * np.pi * t / 24)
X = 8 * np.sin(2 * np.pi * t / 37) + np.linspace(-5, 5, 336)
rollout(BatteryEnv(ReplaySource(X, calib=(0.05, -2.0, 6.0)), f, params), 0, 13, "replay")
np.savez(out, **rec)
'''


def section(title):
    print(f"\n== {title}")


def check_ciso_invariance():
    section(f"A. CISO environments identical to {TAG}")
    archive = subprocess.run(["git", "-C", str(ROOT), "archive", TAG], check=True, capture_output=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        frozen = Path(tmp) / "frozen"; frozen.mkdir()
        tarfile.open(fileobj=io.BytesIO(archive)).extractall(frozen)
        outs = {}
        for label, root in [("frozen", frozen), ("current", ROOT)]:
            outs[label] = Path(tmp) / f"{label}.npz"
            subprocess.run([sys.executable, "-c", DUMPER, str(root), str(outs[label])], check=True,
                           capture_output=True)
        a, b = np.load(outs["frozen"]), np.load(outs["current"])
        assert sorted(a.files) == sorted(b.files)
        bad = [k for k in a.files if not np.array_equal(a[k], b[k])]
        assert not bad, f"CISO behavior changed in: {bad}"
    steps = a["train_seeded_obs"].shape[0] - 1
    print(f"   PASS: {len(a.files)} arrays identical (4 rollouts x {steps} steps: obs, rewards, masks, "
          f"revenue, prices; seeded, continued and reseeded resets; replay env)")


def scenario_stats(q, theta, sigma, mu, f):
    ss = sigma / np.sqrt(2 * theta)
    return {
        "q": q,
        "half-life (h)": np.log(2) / theta,
        "sigma_stat": ss,
        "sigma/q": sigma / q,
        "std(f)/q": np.array([x.std() for x in f]) / q,
        "log mean(f)/q": np.log(np.array([x.mean() for x in f]) / q),
        "mu/sigma_stat": mu / ss,
        "min(f)/q": np.array([x.min() for x in f]) / q,
    }, ss


def check_prior(windows, n=20_000):
    section(f"B. NYISO prior ({n:,} draws) vs {len(windows)} walk-forward windows")
    rng = np.random.default_rng(2024)
    draws = [draw_scenario(rng, NYIS_PRIOR) for _ in range(n)]
    P, pss = scenario_stats(np.array([d["q"] for d in draws]),
                            np.array([d["calib"][0] for d in draws]),
                            np.array([d["calib"][2] for d in draws]),
                            np.array([d["calib"][1] for d in draws]), [d["f"] for d in draws])
    R, rss = scenario_stats(np.array([w["q"] for w in windows]), np.array([w["theta"] for w in windows]),
                            np.array([w["sigma"] for w in windows]), np.array([w["mu_eff"] for w in windows]),
                            [w["f_eval"] for w in windows])
    pct = lambda v: " / ".join(f"{np.percentile(v, p):>7.3f}" for p in (1, 50, 99))
    print(f"   {'':15s} {'prior p1 / p50 / p99':>27s}    {'real p1 / p50 / p99':>27s}")
    for k in P:
        print(f"   {k:15s} {pct(P[k])}    {pct(R[k])}")
    clamp_p = np.mean([d["calib"][0] <= THETA_MIN for d in draws])
    clamp_r = np.mean([w["theta"] <= THETA_MIN + 1e-12 for w in windows])
    print(f"   theta at clamp: prior {clamp_p:.1%}, real {clamp_r:.1%} | min(f) < 0: prior "
          f"{np.mean(P['min(f)/q'] < 0):.1%}, real {np.mean(R['min(f)/q'] < 0):.1%}")

    def corr(S, ss):
        std_f = S["std(f)/q"] * S["q"]
        c = lambda x, y: np.corrcoef(np.log(x), np.log(y))[0, 1]
        return {"q ~ sigma_stat": c(S["q"], ss), "q ~ std(f)": c(S["q"], std_f),
                "sigma_stat ~ std(f)": c(ss, std_f), "half-life ~ sigma_stat/q": c(S["half-life (h)"], ss / S["q"])}
    cp, cr = corr(P, pss), corr(R, rss)
    print("   log-correlations the prior should reproduce (prior | real):")
    for k in cp:
        print(f"     {k:26s} {cp[k]:+.2f} | {cr[k]:+.2f}")
    assert np.all(P["min(f)/q"] >= NYIS_PRIOR["f_floor_over_q"])
    print("   PASS: no forecast below the floor")


def check_norm(windows):
    section("C. Normalization constants")
    fresh = norm_constants(NYIS_PRIOR, n=1_000_000, seed=1)
    for k, (m, s) in NYIS_NORM.items():
        fm, fs = fresh[k]
        assert abs(fm - m) < 0.01 and abs(fs - s) < 0.01, f"{k}: stored ({m}, {s}) vs fresh ({fm:.4f}, {fs:.4f})"
    print("   PASS: NYIS_NORM within 0.01 of an independent Monte Carlo (1,000,000 draws, seed 1)")
    z = np.array([replay_env(w, ENV_KWARGS["NYIS"]).reset(seed=0)[0][14:18] for w in windows])
    for i, k in enumerate(NYIS_NORM):
        print(f"   real windows, {k:24s} mean {z[:, i].mean():+.2f}  std {z[:, i].std():.2f}  max |z| {np.abs(z[:, i]).max():.2f}")


def rollout(env, reset_seed, action_seed, episodes=1):
    rng = np.random.default_rng(action_seed)
    obs_log, violations = [], 0
    for e in range(episodes):
        obs, _ = env.reset(seed=reset_seed if e == 0 else None)
        obs_log.append(obs)
        for _ in range(env.T):
            m = np.array(env.action_masks())
            obs, r, *_rest, info = env.step(int(rng.choice(np.flatnonzero(m))))
            assert np.isfinite(r)
            obs_log.append(obs)
        violations += info["mask_violations"]
    return np.array(obs_log), violations


def check_envs(windows):
    section("D. NYISO environments")
    params = {**BASE_PARAMS, "q": 42.0}                     # q is redrawn every reset
    make = lambda: BatteryEnv(None, np.zeros(336), params,
                              episode_sampler=make_episode_sampler(NYIS_PRIOR), **ENV_KWARGS["NYIS"])
    env = make()
    obs, viol = rollout(env, 7, 1, episodes=5)
    assert env.observation_space.shape == (18,) and obs.shape[1] == 18
    assert np.isfinite(obs).all() and viol == 0
    obs2, _ = rollout(make(), 7, 1, episodes=5)
    assert np.array_equal(obs, obs2), "training env is not deterministic under a fixed seed"
    q_draws = []
    for s in range(5):
        e = make(); e.reset(seed=s); q_draws.append(e.q)
    print(f"   PASS: training env -- 18 obs, 5 episodes finite, deterministic, 0 mask violations; "
          f"q across resets {', '.join(f'{x:.1f}' for x in q_draws)}")

    for w in windows[:3]:
        e = replay_env(w, ENV_KWARGS["NYIS"])
        o, _ = e.reset(seed=0)
        ss = w["sigma"] / np.sqrt(2 * w["theta"])
        expected = (np.log(w["q"] / ss) - NYIS_NORM["log_q_over_sigma_stat"][0]) / NYIS_NORM["log_q_over_sigma_stat"][1]
        assert o.shape == (18,) and np.isfinite(o).all() and np.isclose(o[17], expected, atol=1e-5)
    print("   PASS: replay env on real windows -- 18 obs, finite, q feature matches its definition")


def check_baselines_unchanged(windows):
    section("E. NYISO baseline scores under the NYISO environment settings")
    stored = pd.read_csv(ROOT / "results" / "historical_NYIS.csv", float_precision="round_trip")
    for name, factory in BASELINES.items():
        table, _ = evaluate_agent("NYIS", windows, factory(), name, env_kwargs=ENV_KWARGS["NYIS"])
        s = stored[stored.method == name].set_index("week_idx").score
        diff = float(np.abs(table.set_index("week_idx").score - s.loc[table.week_idx]).max())
        assert diff == 0.0, f"{name}: scores changed by up to {diff}"
        print(f"   PASS: {name} -- {len(table)} weeks, identical scores")


def main():
    windows = get_windows("NYIS")
    check_ciso_invariance()
    check_prior(windows)
    check_norm(windows)
    check_envs(windows)
    check_baselines_unchanged(windows)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
