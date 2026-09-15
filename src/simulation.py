import numpy as np
from src.optimization import build_transition_matrix, get_optimal_policy
from src.dynamics import battery_step
from src.windows import build_windows, SOC_GRID, BASE_PARAMS, EVAL_WINDOW

def simulate(policy, f, X_actual, X_grid, soc_grid, params, T_eval=None):
    if T_eval is None:
        T_eval = len(f)
    
    S_0 = params["S_0"]
    
    soc = S_0
    S_trajectory = np.zeros(T_eval + 1)
    S_trajectory[0] = soc
    revenue = 0
    
    for t in range(T_eval):
        x_idx = np.argmin(np.abs(X_grid - X_actual[t]))
        s_idx = np.argmin(np.abs(soc_grid - soc))
        
        u = policy[t, x_idx, s_idx]
        full_price = f[t] + X_actual[t]

        soc, rev = battery_step(soc, u , full_price, params)
        revenue += rev

        S_trajectory[t + 1] = soc

    return revenue, S_trajectory

def solve_dp(w):
    """Backward-induction policy for one window. ~44 MB: use it, don't store it."""
    params = {**BASE_PARAMS, "q": w["q"]}
    trans = build_transition_matrix(w["theta"], w["mu_eff"], w["sigma"], w["X_grid"])
    return get_optimal_policy(trans, w["f_eval"], w["X_grid"], SOC_GRID, params)

def walk_forward_backtest(data, windows=None, verbose=True):
    if windows is None:
        windows = build_windows(data)

    results = []
    for w in windows:
        params = {**BASE_PARAMS, "q": w["q"]}
        policy = solve_dp(w)
        rev, traj = simulate(policy, w["f_eval"], w["X_eval_resid"], w["X_grid"],
                             SOC_GRID, params, T_eval=EVAL_WINDOW)
        results.append({
            "eval_start": w["eval_start"],
            "revenue": rev,
            "theta": w["theta"],
            "sigma": w["sigma"],
            "mu_eff": w["mu_eff"],
            "sigma_stat": w["sigma_stat"],
            "X_grid": w["X_grid"],
            "f_eval": w["f_eval"],
            "X_eval_resid": w["X_eval_resid"],
        })
        if verbose:
            print(f"Week{len(results):3d} | rev = ${rev:>10,.0f} | theta = {w['theta']:.4f} | "
                  f"sigma={w['sigma']:.2f} | mu={w['mu_eff']:.2f}")

    if verbose:
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
        full_price = prices[t]
        soc, r = battery_step(soc, u, full_price, params)
        rev += r
    return rev

