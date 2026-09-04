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
    train_window = 24 * 365
    eval_window = 24  * 7
    buffer = 24 * 7
    step = 24 * 7
    
    X_grid = np.linspace(-100,200,80)
    soc_grid = np.linspace(0, 100, 41)
    
    params = {"u_max":25, "eta":0.85, "S_max":100, "S_0": 0, "dt":1}
    
    results = []
    prices_all = data["price_usd_mwh"].values
    hours_all = data["hour"].values
    
    for start in range(0, len(prices_all) - train_window - eval_window - buffer, step):
        train_end = start + train_window
        split = start + int(train_window * 0.75)
        
        # Fit seasonal on first half
        seasonal_first = data.iloc[start:split].copy()
        seasonal_model_inner, feature_cols = fit_seasonal_fourier(seasonal_first)
        
        # Pseudo-OOS residuals on second half
        validation = data.iloc[split:train_end].copy()
        val_features = build_fourier_features(validation, feature_cols)
        val_resid = validation["price_usd_mwh"].values - seasonal_model_inner.predict(val_features).values
        
        # Estimate OU on pseudo-OOS residuals
        val_resid = val_resid - val_resid.mean()  # remove bias before OU estimation
        
        theta, mu, sigma = estimate_ou_params(pd.Series(val_resid))
    
        trans = build_transition_matrix(data, theta, mu, sigma, X_grid)  # ADD THIS
        
        # Refit seasonal on full training window for eval
        full_train = data.iloc[start:train_end].copy()
        seasonal_model, feature_cols = fit_seasonal_fourier(full_train)
        
        # Eval as normal
        eval_data = data.iloc[train_end:train_end + eval_window + buffer].copy()
        eval_features = build_fourier_features(eval_data, feature_cols)
        f_eval = seasonal_model.predict(eval_features).values
        X_eval_resid = eval_data["price_usd_mwh"].values - f_eval
    
        params["q"] = full_train["price_usd_mwh"].mean()
    
        policy = get_optimal_policy(trans, f_eval, X_grid, soc_grid, params)
    
        # evaluate on unseen data, append to results.
        #simulate first eval window:
        rev, traj = simulate(policy, f_eval, X_eval_resid, X_grid, soc_grid, params, T_eval = eval_window)
    
        results.append({
            "eval_start":train_end,
            "revenue": rev,
            "theta": theta,
            "sigma" :sigma
        })
    
        print(f"Week{len(results):3d} | " f"rev = ${rev:>10,.0f} | " f"theta = {theta:.4f} |" f"sigma={sigma:.2f}")
    
    revenues = [r["revenue"] for r in results]
    print(f"Mean weekly revenue: ${np.mean(revenues):,.0f}")
    print(f"Std weekly revenue:  ${np.std(revenues):,.0f}")
    print(f"Sharpe (weekly):     {np.mean(revenues) / np.std(revenues):.2f}")
    print(f"Worst week:          ${np.min(revenues):,.0f}")
    print(f"% positive weeks:    {100 * np.mean(np.array(revenues) > 0):.0f}%")
    print(f"Unique revenues: {len(set([round(r['revenue']) for r in results]))}")
    return results
