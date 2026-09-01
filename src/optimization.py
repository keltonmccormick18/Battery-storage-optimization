import numpy as np
from scipy.stats import norm

def build_transition_matrix(data, theta, mu, sigma, X_grid, dt = 1):

    eta = 0.85 #round-trip efficiency
    q =  data["price_usd_mwh"].mean()    # q is terminal value = long-term avg of price

    #storage parameters
    S_max = 100
    S_0 = 0
    u_max  = 25 #(= beta = -alpha)

    # grid sizes:
    N_x = 80
    N_s = 41

    X = np.linspace(-100, 200, N_x)
    dx = X[1]-X[0]

    soc_grid = np.linspace(0, S_max, N_s)
    ds = soc_grid[1] - soc_grid[0]

    T_extended = 24*14
    T = 24 * 7

    #precompute OU transition matrix...
    #P(X_k at t+1 | X_t at t) for all i, k.
    trans = np.zeros((N_x, N_x))

    b = np.exp(-theta * dt)
    ou_var = (sigma**2 / (2*theta)) * (1-b**2)
    ou_std = np.sqrt(ou_var)

    for i in range(N_x):
        mean_next = mu * (1-b) + b * X[i]
        probs = norm.pdf(X,mean_next, ou_std) * dx
        probs /= probs.sum() #normalization
        trans[i,:] = probs

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
    
    actions = [u_max, 0, -u_max]
    
    V = np.zeros((N_x,N_s))
    V[:,:] = q * soc_grid[np.newaxis, :]
    policy = np.zeros((T, N_x, N_s))
    
    for t in range(T - 1, -1, -1):
        f_t = f[t]
        EV = trans @ V
        V_new = np.full((N_x, N_s), -np.inf)
        
        for i in range(N_x):
            full_price = f_t + X_grid[i]
            
            for j in range(N_s):
                best_val = -np.inf
                best_u = 0
                
                for u in actions:
                    if u > 0:
                        S_next = soc_grid[j] - u * dt
                    elif u < 0:
                        S_next = soc_grid[j] + eta * abs(u) * dt
                    else:
                        S_next = soc_grid[j]
                    
                    if S_next < 0 or S_next > S_max:
                        continue
                    
                    revenue = u * full_price * dt
                    
                    j_frac = (S_next - soc_grid[0]) / ds
                    j_lo = int(np.floor(j_frac))
                    j_hi = min(j_lo + 1, N_s - 1)
                    w = j_frac - j_lo
                    future = (1 - w) * EV[i, j_lo] + w * EV[i, j_hi]
                    
                    total = revenue + future
                    if total > best_val:
                        best_val = total
                        best_u = u
                
                V_new[i, j] = best_val
                policy[t, i, j] = best_u
        
        V = V_new
    
    return policy


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
    
    for t in range(T - 1, -1, -1):
        V_new = np.full(N_s, -np.inf)
        for j in range(N_s):
            best_val = -np.inf
            best_u = 0
            for u in [u_max, 0, -u_max]:
                if u > 0:
                    S_next = soc_grid[j] - u * dt
                elif u < 0:
                    S_next = soc_grid[j] + eta * abs(u) * dt
                else:
                    S_next = soc_grid[j]
                if S_next < 0 or S_next > S_max:
                    continue
                revenue = u * prices[t] * dt
                j_frac = (S_next - soc_grid[0]) / ds
                j_lo = int(np.floor(j_frac))
                j_lo = min(j_lo, N_s - 2)
                j_hi = j_lo + 1
                w = j_frac - j_lo
                future = (1 - w) * V[j_lo] + w * V[j_hi]
                total = revenue + future
                if total > best_val:
                    best_val = total
                    best_u = u
            V_new[j] = best_val
            policy[t, j] = best_u
        V = V_new
    
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
