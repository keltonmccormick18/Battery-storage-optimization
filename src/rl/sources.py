import numpy as np

def sample_calib(rng):
    """Draw one (theta, mu, sigma) from wide priors bracketing the observed
    walk-forward calibration range. Deliberately not fitted to the results."""
    half_life  = np.exp(rng.uniform(np.log(2.0), np.log(240.0)))   # hours
    theta      = np.log(2) / half_life
    sigma_stat = np.exp(rng.uniform(np.log(4.0), np.log(140.0)))   # $/MWh
    sigma      = sigma_stat * np.sqrt(2 * theta)
    mu         = rng.uniform(-0.9, 0.9) * sigma_stat
    return theta, mu, sigma

class OUSource:
    """Sample residual paths from calibrated OU process"""

    def __init__(self, theta, mu, sigma, dt=1):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.calib = (theta, mu, sigma)

    def sample(self, T=336, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        X = np.zeros(T)
        X[0] = self.mu + rng.normal(0, self.sigma/np.sqrt(2*self.theta))

        b = np.exp(-self.theta * self.dt)
        noise_std = np.sqrt((self.sigma**2 / (2* self.theta)) * (1-b**2))

        for t in range(1,T):
            X[t] = self.mu * (1-b) + b * X[t-1] + rng.normal(0, noise_std)
        
        return X
    
class BootstrapSource:
    """resamples contiguous blocks from historical residuals"""

    def __init__(self, hist_residuals, calib, block_size=336):
        self.hist = hist_residuals
        self.block_size = block_size
        self.calib = calib

    def sample(self, T=336, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        
        max_start = len(self.hist)-T
        start = rng.integers(0,max_start)
        return self.hist[start:start + T].copy()



class ReplaySource:
    """ replay one exact historical residual path; deterministic"""

    def __init__(self, X_actual, calib):
        self.X = X_actual.copy()
        self.calib = calib

    def sample(self, T=336, rng=None):
        return self.X[:T].copy()
    
class RandomizedOUSource:
    """OU source, difference is that Randomized redraws calib = (theta, mu, sigma) every episode"""
    def __init__(self, dt=1):
        self.dt = dt
        self.calib = None

    def sample(self, T=336, rng=None):
        if rng is None:
            rng = np.random.default_rng()

        theta, mu, sigma = sample_calib(rng)
        self.calib = (theta,mu,sigma)
        
        X = np.zeros(T)
        X[0] = mu + rng.normal(0, sigma/np.sqrt(2*theta))

        b = np.exp(-theta * self.dt)
        noise_std = np.sqrt((sigma**2 / (2* theta)) * (1-b**2))

        for t in range(1,T):
            X[t] = mu * (1-b) + b * X[t-1] + rng.normal(0, noise_std)
        
        return X
    
def sample_f(rng, T = 336, q = 42.0):
    """generates a randomized seasonal price curve
    """
    level = q * (1 + rng.uniform(-0.3, 0.3))
    amp = np.exp(rng.uniform(np.log(3),np.log(60)))
    t = np.arange(T)

    f = level + amp * np.sin(2 * np.pi * t / 24 + rng.uniform(0, 2 * np.pi))

    for k in (2, 3):
        f += amp * rng.uniform(0.1, 0.5) / k * np.sin(
            2 * np.pi * k * t / 24 + rng.uniform(0, 2 * np.pi)
        )
    
    return f