import os
os.environ.setdefault("OMP_NUM_THREADS", "1")  
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np, pandas as pd, torch
torch.set_num_threads(1)
from sb3_contrib import MaskablePPO
from src.price_model import fit_seasonal_fourier, build_fourier_features, estimate_ou_params
from src.rl.train import make_vec_env, Gate2EvalCallback

COMMON = dict(gamma=1.0, learning_rate=3e-4, n_epochs=10, gae_lambda=0.95,
              ent_coef=0.01, policy_kwargs=dict(net_arch=[128, 128]), device="cpu")
CONFIGS = {
    "A": dict(n_steps=1008, batch_size=504, total_timesteps=2_000_000),
    "B": dict(n_steps=512,  batch_size=512, total_timesteps=3_000_000),
}

def gate2_window(ciso):
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

    params = {"u_max": 25, "eta": 0.85, "S_max": 100, "S_0": 0, "dt": 1, "q": q}
    return params, f_eval, theta, mu_eff, sigma

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--logdir", default="/content/runs")
    ap.add_argument("--ckptdir", default="/content/drive/MyDrive/battery_rl/ckpt")
    ap.add_argument("--steps", type=int, default=None, help="override total_timesteps (smoke tests)")
    ap.add_argument("--eval-freq", type=int, default=100_000)
    a = ap.parse_args()
    cfg, name = CONFIGS[a.config], f"{a.config}_s{a.seed}"
    os.makedirs(a.ckptdir, exist_ok=True)   # fail fast on a bad path, before training

    ciso = pd.read_csv(ROOT / "data" / "prices_CISO.csv")
    params, f_eval, theta, mu_eff, sigma = gate2_window(ciso)
    print(f"[{name}] theta={theta:.4f} mu_eff={mu_eff:.2f} sigma={sigma:.2f} q={params['q']:.2f}", flush=True)

    venv = make_vec_env(params, n_envs=16)
    callback = Gate2EvalCallback(f_eval=f_eval, params=params, theta=theta, mu=mu_eff,
                                 sigma=sigma, n_eval_episodes=200, eval_freq=a.eval_freq)
    model = MaskablePPO("MlpPolicy", venv, n_steps=cfg["n_steps"], batch_size=cfg["batch_size"],
                        seed=a.seed, verbose=0, tensorboard_log=a.logdir, **COMMON)
    model.learn(total_timesteps=a.steps or cfg["total_timesteps"], callback=callback, tb_log_name=name)

    model.save(f"{a.ckptdir}/{name}_final")
    venv.close()

if __name__ == "__main__":
    main()