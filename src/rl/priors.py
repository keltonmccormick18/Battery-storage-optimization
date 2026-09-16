"""Training-scenario priors by market (docs/experiments/nyiso_setup.md).

CISO keeps the frozen sample_calib / sample_f in src/rl/sources.py and the observation
normalization it was trained with (tag rl-config-frozen). Nothing here changes CISO.

NYISO draws a whole scenario: the price level q first, then everything that scales with
price as a ratio to q. In the NYISO walk-forward windows the raw quantities co-move
strongly (log-correlations +0.79 to +0.86), while q, half-life, sigma/q, std(f)/q and
mean(f)/q are close to independent (|r| <= 0.25). Sampling the diffusion coefficient
sigma rather than sigma_stat matters: sigma_stat/q correlates +0.59 with half-life
(mechanically, since sigma_stat = sigma / sqrt(2 theta)); sigma/q correlates -0.07.
"""
import numpy as np

from src.rl.env import CISO_NORM

THETA_MIN = 0.01          # the clamp in src/windows.build_windows

# Bounds bracket p1-p99 of the 333 NYISO walk-forward calibrations, slightly widened,
# the same method used for the CISO bounds. Values in comments are observed p1-p99.
NYIS_PRIOR = {
    "q": (19.0, 90.0),              # loguniform; observed 21.0-81.3
    "half_life": (8.0, 127.0),      # loguniform, then theta >= THETA_MIN as in the pipeline.
                                    # Upper bound sets the clamp mass to 21.9% (observed 22%).
    "sigma_over_q": (0.06, 0.40),   # loguniform, diffusion sigma; observed 0.065-0.369
    "f_std_over_q": (0.15, 0.62),   # loguniform; observed 0.158-0.598
    "log_level_sd": 0.33,           # log(mean f / q) ~ N(0, sd); observed std 0.333
    "mu_over_sigma_stat": 0.9,      # uniform(-0.9, 0.9), as for CISO and the mu_eff clip
    "f_floor_over_q": 0.0,          # redraw forecasts that go below this (never negative in NYISO)
}

# Standardization for the calibration observations under NYIS_PRIOR: mean and std of each
# raw feature over 2,000,000 prior draws (seed 0). Real NYISO windows standardized this
# way have means within +/-0.2 and stds 0.71-0.92. Recompute if NYIS_PRIOR changes
# (scripts/check_nyiso_setup.py verifies they still match).
NYIS_NORM = {
    "log_theta": (-3.7619, 0.7088),
    "mu_ratio": (0.0, 0.5197),
    "log_sigma_stat": (3.3911, 0.7919),
    "log_q_over_sigma_stat": (0.3312, 0.6525),
}

# BatteryEnv settings per market, shared by training and historical evaluation.
ENV_KWARGS = {
    "CISO": {},
    "NYIS": {"norm": NYIS_NORM, "observe_q": True},
}


def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def ou_path(theta, mu, sigma, T, rng, dt=1):
    """Same exact-discretization recursion as OUSource.sample."""
    X = np.zeros(T)
    X[0] = mu + rng.normal(0, sigma / np.sqrt(2 * theta))
    b = np.exp(-theta * dt)
    noise_std = np.sqrt((sigma**2 / (2 * theta)) * (1 - b**2))
    for t in range(1, T):
        X[t] = mu * (1 - b) + b * X[t - 1] + rng.normal(0, noise_std)
    return X


def seasonal_curve(rng, T, q, prior, max_tries=100):
    """Daily shape with harmonics 1-3, scaled to a sampled level and std; redrawn if below the floor."""
    t = np.arange(T)
    for _ in range(max_tries):
        shape = np.sin(2 * np.pi * t / 24 + rng.uniform(0, 2 * np.pi))
        for k in (2, 3):
            shape += rng.uniform(0.1, 0.5) / k * np.sin(2 * np.pi * k * t / 24 + rng.uniform(0, 2 * np.pi))
        shape = (shape - shape.mean()) / shape.std()
        level = q * np.exp(rng.normal(0, prior["log_level_sd"]))
        f = level + _loguniform(rng, *prior["f_std_over_q"]) * q * shape
        if f.min() >= prior["f_floor_over_q"] * q:
            return f
    raise RuntimeError("seasonal_curve: floor not met; check f_floor_over_q against the prior")


def draw_scenario(rng, prior, T=336):
    """One training scenario: q, OU calibration (theta, mu, sigma), and seasonal forecast f."""
    q = _loguniform(rng, *prior["q"])
    theta = max(np.log(2) / _loguniform(rng, *prior["half_life"]), THETA_MIN)
    sigma = q * _loguniform(rng, *prior["sigma_over_q"])
    sigma_stat = sigma / np.sqrt(2 * theta)
    mu = rng.uniform(-prior["mu_over_sigma_stat"], prior["mu_over_sigma_stat"]) * sigma_stat
    return {"q": q, "calib": (theta, mu, sigma), "f": seasonal_curve(rng, T, q, prior)}


def make_episode_sampler(prior):
    """Callable for BatteryEnv(episode_sampler=...): returns q, f, X and calib for one episode."""
    def sample(rng, T):
        s = draw_scenario(rng, prior, T)
        s["X"] = ou_path(*s["calib"], T, rng)
        return s
    return sample


def norm_constants(prior, n=2_000_000, seed=0):
    """Mean and std of each raw calibration feature under the prior (vectorized Monte Carlo)."""
    rng = np.random.default_rng(seed)
    lu = lambda lo, hi: np.exp(rng.uniform(np.log(lo), np.log(hi), n))
    q = lu(*prior["q"])
    theta = np.maximum(np.log(2) / lu(*prior["half_life"]), THETA_MIN)
    sigma = q * lu(*prior["sigma_over_q"])
    ss = sigma / np.sqrt(2 * theta)
    a = prior["mu_over_sigma_stat"]
    raw = {
        "log_theta": np.log(theta),
        "mu_ratio": rng.uniform(-a, a, n),
        "log_sigma_stat": np.log(ss),
        "log_q_over_sigma_stat": np.log(q / ss),
    }
    return {k: (float(v.mean()), float(v.std())) for k, v in raw.items()}
