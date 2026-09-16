import numpy as np
import gymnasium as gym
from gymnasium import spaces
from src.dynamics import battery_step


# Derived from sample_calib's priors:
#   log(theta)      ~ U(ln(ln2/240), ln(ln2/2)) = U(-5.85, -1.06)
#   log(sigma_stat) ~ U(ln 4, ln 140)           = U( 1.39,  4.94)
#   mu/sigma_stat   ~ U(-0.9, 0.9)
# Uniform on [a,b] -> mean (a+b)/2, std (b-a)/sqrt(12).
# Update these if you change the bounds in sample_calib.
_LOGTH_M, _LOGTH_S = -2.92, 1.07
_LOGSS_M, _LOGSS_S =  3.00, 0.53
_MUR_S = 0.52

# The constants above as a normalization dict: the default, so CISO agents see exactly
# the observations they were trained on. Other markets pass their own (src/rl/priors.py).
CISO_NORM = {
    "log_theta": (_LOGTH_M, _LOGTH_S),
    "mu_ratio": (0.0, _MUR_S),
    "log_sigma_stat": (_LOGSS_M, _LOGSS_S),
}
CALIB_FEATURES = ["log_theta", "mu_ratio", "log_sigma_stat"]


def raw_calib_features(calib, q):
    theta, mu, sigma = calib
    sigma_stat = sigma / np.sqrt(2 * theta)
    return {
        "log_theta": np.log(theta),
        "mu_ratio": mu / sigma_stat,
        "log_sigma_stat": np.log(sigma_stat),
        "log_q_over_sigma_stat": np.log(q / sigma_stat),
    }

class BatteryEnv(gym.Env):
    """
    battery storage dispatch environment.

    336-step episodes; reward scored on first 168 -- for the buffer to prevent terminal condition from distorting behaviour.
    3 discrete actions: discharge (0), hold (1), charge (2).

    Optional, all off by default (the frozen CISO configuration):
      episode_sampler(rng, T) -> {"q", "f", "X", "calib"}: draw a whole scenario each reset,
          replacing source and f_sampler (NYISO training).
      norm: standardization constants for the calibration observations.
      observe_q: append log(q / sigma_stat), the scale of the gap between the charge and
          discharge break-even prices -- 18 observations instead of 17.
    """
    def __init__(self, source, f, params, sigma_pred = None, f_sampler = None,
                 episode_sampler = None, norm = None, observe_q = False):
        super().__init__()
        self.episode_sampler = episode_sampler
        self.norm = CISO_NORM if norm is None else norm
        self.observe_q = observe_q
        self.calib_features = CALIB_FEATURES + (["log_q_over_sigma_stat"] if observe_q else [])

        self.source = source
        self.f = f
        self.f_sampler = f_sampler
        self.params = params
        self.u_max = params["u_max"]
        self.S_max = params["S_max"]
        self.eta = params["eta"]
        self.q = params["q"]
        self.dt = params["dt"]
        self.T = len(f)
        self.KS = np.array([1, 2, 3, 6, 12, 24])
        idx = np.clip(np.arange(self.T)[:, None] + self.KS[None, :], 0, len(f) - 1)
        self.f_fwd = f[idx] - f[:self.T, None]  
        self.eval_window = 168
        self.calib_obs = None
        self.n_mask_violations = 0

        self.sigma_pred = sigma_pred if sigma_pred is not None else 1.0
        self.reward_scale = 1.0 / (self.u_max * self.sigma_pred)

        self.action_space = spaces.Discrete(3)

        self.observation_space = spaces.Box(
            low = -np.inf, high = np.inf, shape = (14 + len(self.calib_features),), dtype = np.float32
        )
        
        self.t = 0
        self.soc = params["S_0"]
        self.X = None
        self.total_revenue = 0
                                        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.t = 0
        self.soc = self.params["S_0"]

        if self.episode_sampler is not None:
            episode = self.episode_sampler(self.np_random, self.T)
            self.q = episode["q"]
            self.f = episode["f"]
            idx = np.clip(np.arange(self.T)[:, None] + self.KS[None, :], 0, self.T - 1)
            self.f_fwd = self.f[idx] - self.f[:, None]
            self.X = episode["X"]
            calib = episode["calib"]
        else:
            if self.f_sampler is not None:
                self.f = self.f_sampler(rng = self.np_random, T = self.T, q = self.q)
                idx = np.clip(np.arange(self.T)[:, None] + self.KS[None, :], 0, self.T - 1)
                self.f_fwd = self.f[idx] - self.f[:, None]
            self.X = self.source.sample(T=self.T, rng=self.np_random)
            calib = self.source.calib

        theta, mu, sigma = calib
        self.sigma_pred = sigma / np.sqrt(2 * theta)
        self.reward_scale = 1.0 / (self.u_max * self.sigma_pred)
        self.calib_obs = self._calib_obs(calib)

        self.total_revenue = 0
        self.n_mask_violations = 0

        return self._obs(), self._info()

    
    def step(self, action):
        #map disc action to cont.
        action_map = {0:self.u_max, 1:0, 2: -self.u_max}
        #mask infeasible actions
        u = {0: self.u_max, 1: 0.0, 2: -self.u_max}[action]
        if not self.action_masks()[action]:
            u = 0.0
            self.n_mask_violations += 1

        full_price = self.f[self.t] + self.X[self.t]
        soc_prev = self.soc
        soc_next, revenue = battery_step(self.soc, u, full_price, self.params)

        if self.t < self.eval_window:
            self.total_revenue += revenue
        
        self.soc = soc_next
        self.t += 1

        terminated = self.t >= self.T
        truncated = False

        shaping = self.q * (soc_next - soc_prev)
        reward = (revenue + shaping) * self.reward_scale

        return self._obs(), reward, terminated, truncated, self._info()
    
    def _calib_obs(self, calib):
        raw = raw_calib_features(calib, self.q)
        return np.array([(raw[k] - self.norm[k][0]) / self.norm[k][1] for k in self.calib_features],
                        dtype=np.float32)
    
    def _obs(self):
        t = min(self.t, self.T-1)
        hour = self.t % 24
        X_t = self.X[t] if self.X is not None else 0.0

        head = np.array([
            self.soc / self.S_max,                      # 0
            X_t / self.sigma_pred,                      # 1
            np.sin(2 * np.pi * hour / 24),              # 2
            np.cos(2 * np.pi * hour / 24),              # 3
            (self.f[t] - self.q) / self.sigma_pred,     # 4
        ], dtype=np.float32)

        tail = np.array([
            (self.T - self.t) / self.T,                 # 11
            float(self.soc <= 0.0),                     # 12
            float(self.soc >= self.S_max),              # 13
        ], dtype=np.float32)

        return np.concatenate([
            head,                                       # 5
            self.f_fwd[t] / self.sigma_pred,            # 6  -> 5..10
            tail,                                       # 3
            self.calib_obs,                             # 3 or 4 -> 14..16 (17 if observe_q)
        ]).astype(np.float32)
    
    def _info(self):
        return {
            "revenue" : self.total_revenue,
            "soc" : self.soc,
            "t" : self.t,
            "mask_violations": self.n_mask_violations
        }

    
    def action_masks(self):
        masks = [True, True, True]

        if self.soc - self.u_max * self.dt < 0:
            masks[0] = False #don't discharge
        if self.soc + self.eta * self.u_max * self.dt > self.S_max:
            masks[2] = False #don't charge

        return masks
