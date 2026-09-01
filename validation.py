train_window = 24 * 365
eval_window = 24  * 7
buffer = 24 * 7
step = 24 * 7

X_grid = np.linspace(-100,200,80)
soc_grid = np.linspace(0, 100, 41)

params = {"u_max":25, "eta":0.85, "S_max":100, "S_0": 0, "dt":1}

results_ciso = []
prices_all = CISO["price_usd_mwh"].values
hours_all = CISO["hour"].values

for start in range(0, len(prices_all) - train_window - eval_window - buffer, step):
    train_end = start + train_window
    split = start + int(train_window * 0.75)
    
    # Fit seasonal on first half
    seasonal_first = CISO.iloc[start:split].copy()
    seasonal_model_inner, feature_cols = fit_seasonal_fourier(seasonal_first)
    
    # Pseudo-OOS residuals on second half
    validation = CISO.iloc[split:train_end].copy()
    val_features = build_fourier_features(validation, feature_cols)
    val_resid = validation["price_usd_mwh"].values - seasonal_model_inner.predict(val_features).values
    
    # Estimate OU on pseudo-OOS residuals
    val_resid = val_resid - val_resid.mean()  # remove bias before OU estimation
    
    theta, mu, sigma = estimate_ou_params(pd.Series(val_resid))

    trans = build_transition_matrix(CISO, theta, mu, sigma, X_grid)  # ADD THIS
    
    # Refit seasonal on full training window for eval
    full_train = CISO.iloc[start:train_end].copy()
    seasonal_model, feature_cols = fit_seasonal_fourier(full_train)
    
    # Eval as normal
    eval_data = CISO.iloc[train_end:train_end + eval_window + buffer].copy()
    eval_features = build_fourier_features(eval_data, feature_cols)
    f_eval = seasonal_model.predict(eval_features).values
    X_eval_resid = eval_data["price_usd_mwh"].values - f_eval

    params["q"] = full_train["price_usd_mwh"].mean()

    policy = get_optimal_policy(trans, f_eval, X_grid, soc_grid, params)

    # evaluate on unseen data, append to results.
    #simulate first eval window:
    rev, traj = simulate(policy, f_eval, X_eval_resid, X_grid, soc_grid, params, T_eval = eval_window)

    results_ciso.append({
        "eval_start":train_end,
        "revenue": rev,
        "theta": theta,
        "sigma" :sigma
    })

    print(f"Week{len(results_ciso):3d} | " f"rev = ${rev:>10,.0f} | " f"theta = {theta:.4f} |" f"sigma={sigma:.2f}")

revenues = [r["revenue"] for r in results_ciso]
print(f"Mean weekly revenue: ${np.mean(revenues):,.0f}")
print(f"Std weekly revenue:  ${np.std(revenues):,.0f}")
print(f"Sharpe (weekly):     {np.mean(revenues) / np.std(revenues):.2f}")
print(f"Worst week:          ${np.min(revenues):,.0f}")
print(f"% positive weeks:    {100 * np.mean(np.array(revenues) > 0):.0f}%")
print(f"Unique revenues: {len(set([round(r['revenue']) for r in results_ciso]))}")
