import numpy as np
from scipy.stats import norm

def build_transition_matrix(theta, mu, sigma, X_grid, dt=1):
    N_x = len(X_grid)
    dx = X_grid[1] - X_grid[0]
    
    trans = np.zeros((N_x, N_x))
    b = np.exp(-theta * dt)
    ou_var = (sigma**2 / (2 * theta)) * (1 - b**2)
    ou_std = np.sqrt(ou_var)
    
    for i in range(N_x):
        mean_next = mu * (1 - b) + b * X_grid[i]
        probs = norm.pdf(X_grid, mean_next, ou_std) * dx
        probs /= probs.sum()
        trans[i, :] = probs
    
    return trans


def get_optimal_policy(trans, f, X_grid, soc_grid, params):
    T = len(f)
    N_x = len(X_grid)
    N_s = len(soc_grid)
    ds = soc_grid[1] - soc_grid[0]
    
    u_max = params["u_max"]
    eta = params["eta"]
    S_max = params["S_max"]
    q = params["q"]
    dt = params["dt"]
    
    V = np.zeros((N_x, N_s))
    V[:, :] = q * soc_grid[np.newaxis, :]
    policy = np.zeros((T, N_x, N_s))
    
    # Precompute SOC transitions and interpolation weights
    S_next_discharge = soc_grid - u_max * dt
    S_next_charge = soc_grid + eta * u_max * dt
    
    # Discharge interpolation
    j_frac_d = (S_next_discharge - soc_grid[0]) / ds
    j_lo_d = np.clip(np.floor(j_frac_d).astype(int), 0, N_s - 2)
    w_d = j_frac_d - j_lo_d
    discharge_feasible = S_next_discharge >= 0
    
    # Charge interpolation
    j_frac_c = (S_next_charge - soc_grid[0]) / ds
    j_lo_c = np.clip(np.floor(j_frac_c).astype(int), 0, N_s - 2)
    w_c = j_frac_c - j_lo_c
    charge_feasible = S_next_charge <= S_max
    
    for t in range(T - 1, -1, -1):
        f_t = f[t]
        EV = trans @ V  # (N_x, N_s)
        
        full_prices = f_t + X_grid  # (N_x,)
        
        # Future value for discharge at each SOC
        future_d = (1 - w_d) * EV[:, j_lo_d] + w_d * EV[:, j_lo_d + 1]  # (N_x, N_s)
        rev_d = full_prices[:, np.newaxis] * u_max * dt
        val_discharge = np.where(discharge_feasible, rev_d + future_d, -np.inf)
        
        # Future value for charge at each SOC
        future_c = (1 - w_c) * EV[:, j_lo_c] + w_c * EV[:, j_lo_c + 1]  # (N_x, N_s)
        rev_c = full_prices[:, np.newaxis] * (-u_max) * dt
        val_charge = np.where(charge_feasible, rev_c + future_c, -np.inf)
        
        # Hold
        val_hold = EV
        
        # Stack and pick best
        all_vals = np.stack([val_discharge, val_hold, val_charge], axis=0)  # (3, N_x, N_s)
        best_idx = np.argmax(all_vals, axis=0)  # (N_x, N_s)
        
        V = np.max(all_vals, axis=0)
        actions = np.array([u_max, 0, -u_max])
        policy[t] = actions[best_idx]
    
    return policy
