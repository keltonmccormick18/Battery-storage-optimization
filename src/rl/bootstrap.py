"""Bootstrap training source: real residual paths instead of simulated ones.

Shape comes from the data, scale and labels from the market's prior:

  * a contiguous 336-hour block of real residuals supplies the path shape -- fat tails,
    spikes and the sustained level drifts an OU process cannot produce;
  * q, the calibration (theta, mu, sigma) and the seasonal forecast are drawn from the
    same prior the OU source uses, so the two agents differ only in path shape;
  * the block is rescaled by one global factor, sigma_stat_drawn / sigma_stat_pool, so
    blocks keep their real relative dispersion (some weeks calm, some violent).

The agent is given the drawn calibration, not a fit to the block. That mismatch is
deliberate: at evaluation the DP's calibration comes from a year-long fit and the realized
week does something else, and this trains under the same misspecification.

The pool is residuals from before the first evaluation week, so nothing an agent trains on
postdates any week it is scored on. See docs/experiments/bootstrap_source.md.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from src.price_model import fit_seasonal_fourier, build_fourier_features, estimate_ou_params
from src.windows import (ROOT, TRAIN_WINDOW, SHAPE_DAYS, add_calendar,
                         apply_shape, shape_profile)
from src.rl.priors import THETA_MIN

_META_KEYS = ("train_window", "theta", "sigma", "sigma_stat", "mean", "q_pool", "n_hours")


def build_pool(market, shape=False):
    """Residuals of the hours before the first evaluation week, with one OU fit over all of them.

    shape: subtract the intraday shape forecast instead of the plain Fourier curve, so the pool
    carries the residuals of the forecast the policy will actually face.
    """
    data = add_calendar(pd.read_csv(ROOT / "data" / f"prices_{market}.csv"))
    pool_df = data.iloc[:TRAIN_WINDOW].copy()
    model, cols = fit_seasonal_fourier(pool_df)
    f_pool = model.predict(build_fourier_features(pool_df, cols)).values
    if shape:
        prices = pool_df["price_usd_mwh"].values
        hod, dow = pool_df["hour_of_day"].values, pool_df["dow"].values
        # profile from the pool's own last SHAPE_DAYS, the only history available to it
        f_pool = apply_shape(f_pool, hod, dow,
                             shape_profile(prices, hod, dow, len(prices), SHAPE_DAYS))
    resid = pool_df["price_usd_mwh"].values - f_pool
    theta, _, sigma = estimate_ou_params(pd.Series(resid - resid.mean()))
    theta = max(theta, THETA_MIN)
    return {
        "resid": resid,
        "mean": float(resid.mean()),
        "theta": float(theta),
        "sigma": float(sigma),
        "sigma_stat": float(sigma / np.sqrt(2 * theta)),
        "q_pool": float(pool_df["price_usd_mwh"].mean()),
        "train_window": float(TRAIN_WINDOW),
        "n_hours": float(len(resid)),
        "first_hour": str(pool_df["hour"].iloc[0]),
        "last_hour": str(pool_df["hour"].iloc[-1]),
    }


@lru_cache(maxsize=None)
def get_pool(market, cache_dir=None, shape=False):
    """Load the cached pool, building it on first use. Shared by all envs in a process."""
    path = Path(cache_dir or ROOT / "results") / f"pool_{market}{'_shape' if shape else ''}.npz"
    if path.exists():
        with np.load(path) as z:
            pool = {k: z[k] for k in z.files}
        if pool["train_window"] == TRAIN_WINDOW:
            return {k: (v.item() if v.ndim == 0 else v) for k, v in pool.items()}
    pool = build_pool(market, shape)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **pool)
    return pool


def make_bootstrap_sampler(market, drawer, pool=None, shape=False):
    """episode_sampler for BatteryEnv: prior scenario, real residual block scaled to it.

    drawer(rng, T) -> {"q", "calib", "f"} is the market's scenario prior, shared with the
    OU source so the two differ only in the price path.
    """
    pool = pool or get_pool(market, shape=shape)
    resid, base, sigma_stat_pool = pool["resid"], pool["mean"], pool["sigma_stat"]

    def sample(rng, T):
        episode = drawer(rng, T)
        theta, mu, sigma = episode["calib"]
        scale = (sigma / np.sqrt(2 * theta)) / sigma_stat_pool
        start = int(rng.integers(0, len(resid) - T + 1))
        episode["X"] = mu + (resid[start:start + T] - base) * scale
        episode["block_start"] = start
        return episode

    return sample
