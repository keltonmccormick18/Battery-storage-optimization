import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.price_model import fit_seasonal_fourier, build_fourier_features, fit_seasonal, estimate_ou_params
from src.optimization import build_transition_matrix, get_optimal_policy


def simulate(policy, f, X_actual, X_grid, soc_grid, params, T_eval=None):
    if T_eval is None:
        T_eval = len(f)
    
    S_max = params["S_max"]
    eta = params["eta"]
    dt = params["dt"]
    S_0 = params["S_0"]
    
    S_trajectory = np.zeros(T_eval + 1)
    S_trajectory[0] = S_0
    revenue = 0
    
    for t in range(T_eval):
        x_idx = np.argmin(np.abs(X_grid - X_actual[t]))
        s_idx = np.argmin(np.abs(soc_grid - S_trajectory[t]))
        
        u = policy[t, x_idx, s_idx]
        revenue += u * (f[t] + X_actual[t]) * dt
        
        if u > 0:
            S_next = S_trajectory[t] - u * dt
        elif u < 0:
            S_next = S_trajectory[t] + eta * abs(u) * dt
        else:
            S_next = S_trajectory[t]
        S_trajectory[t + 1] = np.clip(S_next, 0, S_max)
    
    return revenue, S_trajectory

def walk_forward_backtest(data):

    if "hour_of_day" not in data.columns:
        data = data.copy()
        data["hour"] = pd.to_datetime(data["hour"])
        data["hour_of_day"] = data["hour"].dt.hour
        data["dow"] = data["hour"].dt.dayofweek
        data["month"] = data["hour"].dt.month
    
    train_window = 24 * 365
    eval_window = 24  * 7
    buffer = 24 * 7
    step = 24 * 7
    
    X_grid = np.linspace(-100,200,80)
    soc_grid = np.linspace(0, 100, 81)
    
    params = {"u_max":25, "eta":0.85, "S_max":100, "S_0": 0, "dt":1}
    
    results = []
    prices_all = data["price_usd_mwh"].values
    hours_all = data["hour"].values
    
    for start in range(0, len(data) - train_window - eval_window - buffer, step):
        train_end = start + train_window
        split = start + int(train_window * 0.75)
        
        # Fit seasonal on first 75% for inner CV
        seasonal_first = data.iloc[start:split].copy()
        seasonal_model_inner, feature_cols_inner = fit_seasonal_fourier(seasonal_first)
        
        # Pseudo-OOS residuals on last 25%
        validation = data.iloc[split:train_end].copy()
        val_features = build_fourier_features(validation, feature_cols_inner)
        val_resid = validation["price_usd_mwh"].values - seasonal_model_inner.predict(val_features).values
        val_resid = val_resid - val_resid.mean()
        
        # Estimate OU on pseudo-OOS residuals
        theta, mu, sigma = estimate_ou_params(pd.Series(val_resid))
        theta = max(theta, 0.01)
        
        # Refit seasonal on full training window
        full_train = data.iloc[start:train_end].copy()
        seasonal_model, feature_cols = fit_seasonal_fourier(full_train)
        
        # Compute mu from last week's residual against full-window model
        lookback = data.iloc[train_end - eval_window:train_end].copy()
        lookback_features = build_fourier_features(lookback, feature_cols)
        L_prev = (lookback["price_usd_mwh"].values - seasonal_model.predict(lookback_features).values).mean()
        
        BETA = 0.6
        mu_eff = BETA * L_prev
        
        # Build transition matrix with corrected mu
        trans = build_transition_matrix(theta, mu_eff, sigma, X_grid)
        
        # Eval predictions
        eval_data = data.iloc[train_end:train_end + eval_window + buffer].copy()
        eval_features = build_fourier_features(eval_data, feature_cols)
        f_eval = seasonal_model.predict(eval_features).values
        X_eval_resid = eval_data["price_usd_mwh"].values - f_eval
        
        params["q"] = full_train["price_usd_mwh"].mean()
        
        # Solve and simulate
        policy = get_optimal_policy(trans, f_eval, X_grid, soc_grid, params)
        rev, traj = simulate(policy, f_eval, X_eval_resid, X_grid, soc_grid, params, T_eval=eval_window)
        
        results.append({
            "eval_start": train_end,
            "revenue": rev,
            "theta": theta,
            "sigma": sigma,
            "mu_eff": mu_eff
        })
        
        print(f"Week{len(results):3d} | rev = ${rev:>10,.0f} | theta = {theta:.4f} | sigma={sigma:.2f} | mu={mu_eff:.2f}")
        
    revenues = [r["revenue"] for r in results]
    print(f"Mean weekly revenue: ${np.mean(revenues):,.0f}")
    print(f"Std weekly revenue:  ${np.std(revenues):,.0f}")
    print(f"Sharpe (weekly):     {np.mean(revenues) / np.std(revenues):.2f}")
    print(f"Worst week:          ${np.min(revenues):,.0f}")
    print(f"% positive weeks:    {100 * np.mean(np.array(revenues) > 0):.0f}%")
    print(f"Unique revenues: {len(set([round(r['revenue']) for r in results]))}")
    return results

def perfect_foresight(prices, soc_grid, params):
    T = len(prices)
    N_s = len(soc_grid)
    ds = soc_grid[1] - soc_grid[0]
    
    u_max = params["u_max"]
    eta = params["eta"]
    S_max = params["S_max"]
    q = params["q"]
    dt = params["dt"]
    
    V = q * soc_grid
    policy = np.zeros((T, N_s))
    
    S_next_discharge = soc_grid - u_max * dt
    S_next_charge = soc_grid + eta * u_max * dt
    
    j_lo_d = np.clip(np.floor((S_next_discharge - soc_grid[0]) / ds).astype(int), 0, N_s - 2)
    w_d = (S_next_discharge - soc_grid[0]) / ds - j_lo_d
    discharge_feasible = S_next_discharge >= 0
    
    j_lo_c = np.clip(np.floor((S_next_charge - soc_grid[0]) / ds).astype(int), 0, N_s - 2)
    w_c = (S_next_charge - soc_grid[0]) / ds - j_lo_c
    charge_feasible = S_next_charge <= S_max
    
    for t in range(T - 1, -1, -1):
        price = prices[t]
        
        future_d = (1 - w_d) * V[j_lo_d] + w_d * V[j_lo_d + 1]
        val_discharge = np.where(discharge_feasible, price * u_max * dt + future_d, -np.inf)
        
        future_c = (1 - w_c) * V[j_lo_c] + w_c * V[j_lo_c + 1]
        val_charge = np.where(charge_feasible, price * (-u_max) * dt + future_c, -np.inf)
        
        val_hold = V
        
        all_vals = np.stack([val_discharge, val_hold, val_charge], axis=0)
        best_idx = np.argmax(all_vals, axis=0)
        
        V = np.max(all_vals, axis=0)
        actions = np.array([u_max, 0, -u_max])
        policy[t] = actions[best_idx]
    
    # Simulate
    soc = params["S_0"]
    rev = 0
    for t in range(T):
        j_idx = np.argmin(np.abs(soc_grid - soc))
        u = policy[t, j_idx]
        rev += u * prices[t] * dt
        if u > 0:
            soc -= u * dt
        elif u < 0:
            soc += eta * abs(u) * dt
        soc = np.clip(soc, 0, S_max)
    return rev

