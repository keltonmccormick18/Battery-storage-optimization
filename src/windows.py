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

SOC_GRID = np.linspace(0, 100, 81)
BASE_PARAMS = {"u_max": 25, "eta": 0.85, "S_max": 100, "S_0": 0, "dt": 1}

_META = np.array([TRAIN_WINDOW, EVAL_WINDOW, BUFFER, STEP, BETA, N_X], dtype=float)
_INTS = ["week_idx", "eval_start"]
_FLOATS = ["q", "theta", "sigma", "sigma_stat", "mu_eff"]
_ARRAYS = ["X_grid", "f_eval", "X_eval_resid", "prices"]


def add_calendar(data):
    if "hour_of_day" in data.columns:
        return data
    data = data.copy()
    data["hour"] = pd.to_datetime(data["hour"])
    data["hour_of_day"] = data["hour"].dt.hour
    data["dow"] = data["hour"].dt.dayofweek
    data["month"] = data["hour"].dt.month
    return data


def build_windows(data):
    data = add_calendar(data)
    windows = []
    for start in range(0, len(data) - TRAIN_WINDOW - EVAL_WINDOW - BUFFER, STEP):
        train_end = start + TRAIN_WINDOW
        split = start + int(TRAIN_WINDOW * 0.75)

        # OU calibration on pseudo-out-of-sample residuals (last 25% of the training window)
        inner, inner_cols = fit_seasonal_fourier(data.iloc[start:split].copy())
        validation = data.iloc[split:train_end].copy()
        val_resid = (validation["price_usd_mwh"].values
                     - inner.predict(build_fourier_features(validation, inner_cols)).values)
        val_resid = val_resid - val_resid.mean()
        theta, _, sigma = estimate_ou_params(pd.Series(val_resid))
        theta = max(theta, 0.01)
        sigma_stat = sigma / np.sqrt(2 * theta)

        # Seasonal forecast and terminal value from the full training window
        full_train = data.iloc[start:train_end].copy()
        seasonal, cols = fit_seasonal_fourier(full_train)

        # mu: last training week's level, clipped and shrunk
        lookback = data.iloc[train_end - EVAL_WINDOW:train_end].copy()
        L_prev = (lookback["price_usd_mwh"].values
                  - seasonal.predict(build_fourier_features(lookback, cols)).values).mean()
        L_prev = np.clip(L_prev, -1.5 * sigma_stat, 1.5 * sigma_stat)
        mu_eff = BETA * L_prev

        eval_data = data.iloc[train_end:train_end + EVAL_WINDOW + BUFFER].copy()
        f_eval = seasonal.predict(build_fourier_features(eval_data, cols)).values
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


def save_windows(windows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        meta=_META,
        eval_start_ts=np.array([w["eval_start_ts"] for w in windows]),
        **{k: np.array([w[k] for w in windows]) for k in _INTS + _FLOATS + _ARRAYS},
    )


def load_windows(path):
    with np.load(path) as z:
        if not np.array_equal(z["meta"], _META):
            raise ValueError(f"{path} was built with different window settings; delete and rebuild")
        cols = {k: z[k] for k in z.files}
    windows = []
    for i in range(len(cols["week_idx"])):
        w = {k: cols[k][i] for k in _FLOATS + _ARRAYS}
        w.update({k: int(cols[k][i]) for k in _INTS})
        w["eval_start_ts"] = str(cols["eval_start_ts"][i])
        windows.append(w)
    return windows


def get_windows(market, cache_dir=None):
    """Load cached windows for a market, building and caching them on first use."""
    path = Path(cache_dir or ROOT / "results") / f"windows_{market}.npz"
    if path.exists():
        return load_windows(path)
    windows = build_windows(pd.read_csv(ROOT / "data" / f"prices_{market}.csv"))
    save_windows(windows, path)
    return windows