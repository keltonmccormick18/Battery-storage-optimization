import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.rl.env import BatteryEnv
from src.rl.sources import OUSource, RandomizedOUSource, sample_f
from stable_baselines3.common.callbacks import BaseCallback


def mask_fn(env):
    return env.action_masks()

def make_env_fn(params):
    def _init():
        source = RandomizedOUSource()
        f_init = sample_f(np.random.default_rng(), 336, params["q"])
        env = BatteryEnv(source, f_init, params, f_sampler = sample_f)
        return ActionMasker(env, mask_fn)
    return _init

def make_vec_env(params, n_envs = 16):
    return DummyVecEnv([make_env_fn(params) for _ in range(n_envs)])

def make_eval_env(source, f_eval, params):
    env = BatteryEnv(source, f_eval, params, f_sampler=None)
    return ActionMasker(env, mask_fn)




class Gate2EvalCallback(BaseCallback):
    """
    Periodically evaluate the agent against OUSource with fixed
    Gate 2 calibration. Logs mean revenue to TensorBoard.
    """
    
    def __init__(self, f_eval, params, theta, mu, sigma, 
                 n_eval_episodes=200, eval_freq=10_000, verbose=1):
        super().__init__(verbose)
        self.f_eval = f_eval
        self.params = params
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean = -np.inf
    
    def _on_step(self):
        if self.num_timesteps % self.eval_freq != 0:
            return True
        
        source = OUSource(theta=self.theta, mu=self.mu, sigma=self.sigma)
        revenues = []
        violations = []
        
        for ep in range(self.n_eval_episodes):
            env = BatteryEnv(source, self.f_eval, self.params, f_sampler=None)
            obs, info = env.reset(seed=ep)
            done = False
            
            while not done:
                masks = np.array(env.action_masks())
                action, _ = self.model.predict(obs, deterministic=True,
                                            action_masks=masks)
                obs, reward, done, truncated, info = env.step(int(action))
                done = done or truncated
            
        revenues.append(info["revenue"])
        violations.append(info["mask_violations"])
        
        mean_rev = np.mean(revenues)
        std_rev = np.std(revenues)
        mean_violations = np.mean(violations)
        
        self.logger.record("eval/mean_revenue", mean_rev)
        self.logger.record("eval/std_revenue", std_rev)
        self.logger.record("eval/min_revenue", np.min(revenues))
        self.logger.record("eval/mask_violations", mean_violations)
        self.logger.record("eval/pct_of_dp", mean_rev / 18982 * 100)
        
        if mean_rev > self.best_mean:
            self.best_mean = mean_rev
            self.model.save("best_model")
        
        if self.verbose:
            print(f"Step {self.n_calls:>8d} | "
                  f"Mean rev: ${mean_rev:>10,.0f} | "
                  f"Std: ${std_rev:>6,.0f} | "
                  f"% of DP: {mean_rev / 18982 * 100:.1f}% | "
                  f"Violations: {mean_violations:.1f}")
        
        return True