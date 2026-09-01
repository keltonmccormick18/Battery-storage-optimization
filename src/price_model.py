def build_fourier_features(data, columns, n_harmonics=4):
    X = pd.DataFrame(index=data.index)
    X["const"] = 1.0
    
    hours = data["hour_of_day"].values
    months = data["month"].values
    
    for k in range(1, n_harmonics + 1):
        X[f"hour_sin_{k}"] = np.sin(2 * np.pi * k * hours / 24)
        X[f"hour_cos_{k}"] = np.cos(2 * np.pi * k * hours / 24)
        X[f"month_sin_{k}"] = np.sin(2 * np.pi * k * months / 12)
        X[f"month_cos_{k}"] = np.cos(2 * np.pi * k * months / 12)
    
    dow_dummies = pd.get_dummies(data["dow"], prefix="dow",
                                 drop_first=True, dtype=float)
    X = pd.concat([X, dow_dummies], axis=1)
    X = X.reindex(columns=columns, fill_value=0)
    return X

def fit_seasonal_fourier(data, n_harmonics=4, alpha=1.0):
    X = pd.DataFrame(index=data.index)
    X["const"] = 1.0
    
    hours = data["hour_of_day"].values
    months = data["month"].values
    
    for k in range(1, n_harmonics + 1):
        X[f"hour_sin_{k}"] = np.sin(2 * np.pi * k * hours / 24)
        X[f"hour_cos_{k}"] = np.cos(2 * np.pi * k * hours / 24)
        X[f"month_sin_{k}"] = np.sin(2 * np.pi * k * months / 12)
        X[f"month_cos_{k}"] = np.cos(2 * np.pi * k * months / 12)
    
    dow_dummies = pd.get_dummies(data["dow"], prefix="dow",
                                 drop_first=True, dtype=float)
    X = pd.concat([X, dow_dummies], axis=1)
    
    model = sm.OLS(data["price_usd_mwh"], X).fit()
    return model, X.columns

def ou_neg_loglik(params, X, dt):
    #takes a time series X, params for the OU process, and calculates the log-likelihood 
    #of those parameters for the process producing the associated time series X. step size dt.
    theta, mu, sigma = params
    n = len(X) - 1
    b = np.exp(-theta * dt)
    predicted = mu * (1-b) + b * X[:-1]
    variance = (sigma**2 / (2*theta)) * (1-b**2)
    negloglik = 0.5 * n * np.log(2*np.pi * variance) + 0.5 * np.sum((X[1:]-predicted)**2/variance)
    return negloglik

def estimate_ou_params(residuals, dt = 1):
    result = minimize(ou_neg_loglik, x0 = [1.0, 0.0, 10.0], 
                  args = (residuals.values, 1.0),
                  method = "L-BFGS-B",
                  bounds = [(1e-4,10.0), (None,None), (1e-4,500.0)])
    theta, mu, sigma = result.x
    return theta,mu,sigma


def fit_seasonal(data):
  dummies = pd.get_dummies(data[["hour_of_day","dow","month"]], columns = ["hour_of_day","dow","month"], drop_first = True, dtype=float)
  dummies = sm.add_constant(dummies)

  seasonal_model = sm.OLS(data["price_usd_mwh"], dummies).fit()
  return seasonal_model
