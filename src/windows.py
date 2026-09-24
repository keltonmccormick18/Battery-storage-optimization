"""Walk-forward windows shared by the DP backtest and RL evaluation.

One window per evaluation week: everything known at train_end (calibration,
seasonal forecast, terminal value q) plus the realized prices used for scoring.
Windows deliberately hold no DP policy -- each is ~44 MB.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.price_model import fit_seasonal_fourier, build_fourier_features, estimate_ou_params

ROOT = Path(__file__).resolve().parents[1]

TRAIN_WINDOW = 24 * 365
EVAL_WINDOW = 24 * 7
BUFFER = 24 * 7
STEP = 24 * 7
BETA = 0.6
N_X = 200
SHAPE_DAYS = 28        # lookback for the recent intraday shape (docs/experiments/shape_forecast.md)

SOC_GRID = np.linspace(0, 100, 81)
BASE_PARAMS = {"u_max": 25, "eta": 0.85, "S_max": 100, "S_0": 0, "dt": 1}

_META_BASE = [TRAIN_WINDOW, EVAL_WINDOW, BUFFER, STEP, BETA, N_X]


def _meta(shape_days):
    return np.array(_META_BASE + [shape_days or 0], dtype=float)


_INTS = ["week_idx", "eval_start"]
_FLOATS = ["q", "theta", "sigma", "sigma_stat", "mu_eff"]
_ARRAYS = ["X_grid", "f_eval", "X_eval_resid", "prices"]


def _local_level(v, k=25):
    """Centred 25-hour moving average: the slow price level with the intraday shape removed."""
    pad = k // 2
    return np.convolve(np.pad(v, pad, mode="edge"), np.ones(k) / k, "same")[pad:-pad]


def shape_profile(prices, hod, dow, end, days=SHAPE_DAYS):
    """Mean intraday shape over the `days` days ending at `end`, as a (2, 24) weekday/weekend
    table of deviations from the local level. Uses only data before `end`, so it is causal.

    Deviation from the local level, never the level itself: the level is worth about $5/week
    (docs/experiments/shape_forecast.md), the shape is worth most of the gap to perfect foresight.
    """
    lo = max(0, end - 24 * days)
    dev = prices[lo:end] - _local_level(prices[lo:end])
    h, d = hod[lo:end], dow[lo:end]
    prof = np.zeros((2, 24))
    for wk in (0, 1):
        m = (d >= 5) == bool(wk)
        if m.sum() < 24:                       # too few of this day type: fall back to all days
            m = np.ones(len(d), bool)
        prof[wk] = (pd.Series(dev[m]).groupby(h[m]).mean()
                    .reindex(range(24)).interpolate().bfill().ffill().values)
    return prof


def apply_shape(f_seasonal, hod, dow, prof):
    """Keep a seasonal forecast's level trajectory, replace its intraday shape."""
    return _local_level(np.asarray(f_seasonal, dtype=float)) + prof[(dow >= 5).astype(int), hod]


def add_calendar(data):
    if "hour_of_day" in data.columns:
        return data
    data = data.copy()
    data["hour"] = pd.to_datetime(data["hour"])
    data["hour_of_day"] = data["hour"].dt.hour
    data["dow"] = data["hour"].dt.dayofweek
    data["month"] = data["hour"].dt.month
    return data


def build_windows(data, shape_days=None):
    """shape_days: if set, replace the Fourier intraday shape with the mean shape of the last
    `shape_days` days (docs/experiments/shape_forecast.md). None reproduces the registered
    windows exactly."""
    data = add_calendar(data)
    all_prices = data["price_usd_mwh"].values
    all_hod = data["hour_of_day"].values
    all_dow = data["dow"].values
    windows = []
    for start in range(0, len(data) - TRAIN_WINDOW - EVAL_WINDOW - BUFFER, STEP):
        train_end = start + TRAIN_WINDOW
        split = start + int(TRAIN_WINDOW * 0.75)

        # OU calibration on pseudo-out-of-sample residuals (last 25% of the training window)
        inner, inner_cols = fit_seasonal_fourier(data.iloc[start:split].copy())
        validation = data.iloc[split:train_end].copy()
        val_f = inner.predict(build_fourier_features(validation, inner_cols)).values
        if shape_days:      # residuals must come from the same forecast the policy will use
            val_f = apply_shape(val_f, validation["hour_of_day"].values,
                                validation["dow"].values,
                                shape_profile(all_prices, all_hod, all_dow, split, shape_days))
        val_resid = validation["price_usd_mwh"].values - val_f
        val_resid = val_resid - val_resid.mean()
        theta, _, sigma = estimate_ou_params(pd.Series(val_resid))
        theta = max(theta, 0.01)
        sigma_stat = sigma / np.sqrt(2 * theta)

        # Seasonal forecast and terminal value from the full training window
        full_train = data.iloc[start:train_end].copy()
        seasonal, cols = fit_seasonal_fourier(full_train)

        # mu: last training week's level, clipped and shrunk
        prof = (shape_profile(all_prices, all_hod, all_dow, train_end, shape_days)
                if shape_days else None)

        lookback = data.iloc[train_end - EVAL_WINDOW:train_end].copy()
        look_f = seasonal.predict(build_fourier_features(lookback, cols)).values
        if shape_days:
            look_f = apply_shape(look_f, lookback["hour_of_day"].values, lookback["dow"].values, prof)
        L_prev = (lookback["price_usd_mwh"].values - look_f).mean()
        L_prev = np.clip(L_prev, -1.5 * sigma_stat, 1.5 * sigma_stat)
        mu_eff = BETA * L_prev

        eval_data = data.iloc[train_end:train_end + EVAL_WINDOW + BUFFER].copy()
        f_eval = seasonal.predict(build_fourier_features(eval_data, cols)).values
        if shape_days:
            f_eval = apply_shape(f_eval, eval_data["hour_of_day"].values,
                                 eval_data["dow"].values, prof)
        prices = eval_data["price_usd_mwh"].values

        windows.append({
            "week_idx": len(windows),
            "eval_start": train_end,
            "eval_start_ts": str(data["hour"].iloc[train_end]),
            "q": full_train["price_usd_mwh"].mean(),
            "theta": theta,
            "sigma": sigma,
            "sigma_stat": sigma_stat,
            "mu_eff": mu_eff,
            "X_grid": np.linspace(mu_eff - 5 * sigma_stat, mu_eff + 5 * sigma_stat, N_X),
            "f_eval": f_eval,
            "X_eval_resid": prices - f_eval,
            "prices": prices,
        })
    return windows


def save_windows(windows, path, shape_days=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        meta=_meta(shape_days),
        eval_start_ts=np.array([w["eval_start_ts"] for w in windows]),
        **{k: np.array([w[k] for w in windows]) for k in _INTS + _FLOATS + _ARRAYS},
    )


def load_windows(path, shape_days=None):
    with np.load(path) as z:
        meta = z["meta"]
        if len(meta) == len(_META_BASE):        # windows written before shape forecasts existed
            meta = np.append(meta, 0.0)
        if not np.array_equal(meta, _meta(shape_days)):
            raise ValueError(f"{path} was built with different window settings; delete and rebuild")
        cols = {k: z[k] for k in z.files}
    windows = []
    for i in range(len(cols["week_idx"])):
        w = {k: cols[k][i] for k in _FLOATS + _ARRAYS}
        w.update({k: int(cols[k][i]) for k in _INTS})
        w["eval_start_ts"] = str(cols["eval_start_ts"][i])
        windows.append(w)
    return windows


def get_windows(market, cache_dir=None, shape_days=None):
    """Load cached windows for a market, building and caching them on first use.

    shape_days caches separately, so the registered windows and the shape-forecast windows
    can coexist.
    """
    suffix = f"_shape{shape_days}" if shape_days else ""
    path = Path(cache_dir or ROOT / "results") / f"windows_{market}{suffix}.npz"
    if path.exists():
        return load_windows(path, shape_days)
    windows = build_windows(pd.read_csv(ROOT / "data" / f"prices_{market}.csv"), shape_days)
    save_windows(windows, path, shape_days)
    return windows