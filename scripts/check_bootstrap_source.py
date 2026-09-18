"""Checks for the bootstrap training source (docs/experiments/bootstrap_source.md).

A. The pool ends before the first evaluation week (no training data postdates any scored week).
B. Given the same seed, the OU and bootstrap sources draw identical q, calibration and
   forecast -- only the price path differs.
C. Rescaling preserves what the bootstrap is for: fat tails and sustained level drift.
D. Realized dispersion relative to the labelled sigma matches what evaluation weeks show.
E. Environments run clean in both markets, and blocks vary.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.stats import kurtosis

from src.windows import load_windows, EVAL_WINDOW, TRAIN_WINDOW
from src.rl.bootstrap import get_pool, make_bootstrap_sampler
from src.rl.priors import ou_path
from src.rl.scenarios import scenario_drawer, training_params, make_training_env, EPISODE_HOURS

N_EPISODES = 300


def episodes(sampler, n=N_EPISODES, seed=5):
    rng = np.random.default_rng(seed)
    return [sampler(rng, EPISODE_HOURS) for _ in range(n)]


def ou_sampler(drawer):
    def sample(rng, T):
        e = drawer(rng, T)
        e["X"] = ou_path(*e["calib"], T, rng)
        return e
    return sample


def innovation_kurtosis(eps):
    z = []
    for e in eps:
        theta, mu, sigma = e["calib"]
        b = np.exp(-theta)
        z.append((e["X"][1:] - (mu * (1 - b) + b * e["X"][:-1])) / (sigma / np.sqrt(2 * theta)))
    return float(kurtosis(np.concatenate(z)))


def daily_drift_autocorr(paths):
    ac = []
    for x in paths:
        d = x[:14 * 24].reshape(14, 24).mean(axis=1)
        if d.std() > 0:
            ac.append(np.corrcoef(d[:-1], d[1:])[0, 1])
    return float(np.mean(ac))


def dispersion_ratio(paths, sigma_stats):
    return float(np.median([x[:EVAL_WINDOW].std() / s for x, s in zip(paths, sigma_stats)]))


def main():
    for market in ("CISO", "NYIS"):
        print(f"\n===== {market}")
        pool = get_pool(market)
        windows = load_windows(ROOT / "results" / f"windows_{market}.npz")
        params = training_params(market)
        drawer = scenario_drawer(market, params)

        # A. leakage boundary
        assert int(pool["n_hours"]) == TRAIN_WINDOW == windows[0]["eval_start"]
        print(f"  A. pool {int(pool['n_hours'])} h ending {pool['last_hour']}; first scored week starts "
              f"{windows[0]['eval_start_ts']}  PASS")

        # B. same scenario, different path
        boot = make_bootstrap_sampler(market, drawer, pool)
        ou = ou_sampler(drawer)
        # one fresh rng per episode: the two sources consume different amounts of randomness
        # after the scenario draw, so a shared stream would only agree on the first episode.
        pairs = [(boot(np.random.default_rng(100 + k), EPISODE_HOURS),
                  ou(np.random.default_rng(100 + k), EPISODE_HOURS)) for k in range(50)]
        assert all(np.array_equal(x["f"], y["f"]) and x["calib"] == y["calib"] and x["q"] == y["q"]
                   for x, y in pairs)
        assert not any(np.array_equal(x["X"], y["X"]) for x, y in pairs)
        print("  B. same seed -> identical q, calibration and forecast; different price path  PASS")

        # C. shape preserved through the rescaling
        be, oe = episodes(boot), episodes(ou_sampler(drawer))
        real = [w["X_eval_resid"][:EPISODE_HOURS] for w in windows]
        print(f"  C. innovation excess kurtosis: bootstrap {innovation_kurtosis(be):8.1f} | "
              f"OU {innovation_kurtosis(oe):5.1f} | real weeks {innovation_kurtosis([{'X': w['X_eval_resid'], 'calib': (w['theta'], w['mu_eff'], w['sigma'])} for w in windows]):8.1f}")
        print(f"     daily drift autocorrelation: bootstrap {daily_drift_autocorr([e['X'] for e in be]):+.2f} | "
              f"OU {daily_drift_autocorr([e['X'] for e in oe]):+.2f} | real weeks {daily_drift_autocorr(real):+.2f}")

        # D. dispersion relative to the labelled sigma
        lab = lambda eps: [e["calib"][2] / np.sqrt(2 * e["calib"][0]) for e in eps]
        real_ratio = dispersion_ratio([w["X_eval_resid"] for w in windows],
                                      [w["sigma"] / np.sqrt(2 * w["theta"]) for w in windows])
        print(f"  D. median (realized 168h std / labelled sigma_stat): bootstrap "
              f"{dispersion_ratio([e['X'] for e in be], lab(be)):.2f} | OU {dispersion_ratio([e['X'] for e in oe], lab(oe)):.2f} "
              f"| real weeks {real_ratio:.2f}")

        # E. environment
        env = make_training_env(market, "bootstrap", params)
        expected = 18 if market == "NYIS" else 17
        rng = np.random.default_rng(3)
        obs_log, viol = [], 0
        for ep in range(3):
            o, _ = env.reset(seed=ep); obs_log.append(o)
            assert o.shape == (expected,) and np.isfinite(o).all()
            for _ in range(env.T):
                m = np.array(env.action_masks())
                o, r, *_rest, info = env.step(int(rng.choice(np.flatnonzero(m))))
                assert np.isfinite(r)
            viol += info["mask_violations"]
        env2 = make_training_env(market, "bootstrap", params)
        assert np.array_equal(np.stack([env2.reset(seed=ep)[0] for ep in range(3)]), np.stack(obs_log))
        starts = {boot(np.random.default_rng(s), EPISODE_HOURS)["block_start"] for s in range(50)}
        assert viol == 0
        print(f"  E. env: {expected} observations, finite, deterministic, {viol} mask violations; "
              f"{len(starts)} distinct block starts in 50 draws  PASS")

        t0 = time.time(); episodes(boot, n=200)
        t1 = time.time(); episodes(ou_sampler(drawer), n=200)
        print(f"  F. 200 episodes: bootstrap {t1 - t0:.2f}s | OU {time.time() - t1:.2f}s")

    print("\nall checks passed")


if __name__ == "__main__":
    main()
