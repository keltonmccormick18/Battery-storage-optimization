import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.rl.env import BatteryEnv
from src.rl.sources import OUSource, RandomizedOUSource, sample_f
from stable_baselines3.common.callbacks import BaseCallback
from src.optimization import build_transition_matrix, get_optimal_policy


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

        sigma_stat = sigma / np.sqrt(2 * theta)
        self.X_grid   = np.linspace(mu - 5*sigma_stat, mu + 5*sigma_stat, 200)
        self.soc_grid = np.linspace(0, 100, 81)
        trans = build_transition_matrix(theta, mu, sigma, self.X_grid)
        self.dp_policy = get_optimal_policy(trans, f_eval, self.X_grid, self.soc_grid, params)

        src = OUSource(theta=theta, mu=mu, sigma=sigma)
        self.dp_revs = np.array([
            self._run_episode(BatteryEnv(src, f_eval, params, f_sampler=None), ep,
                              policy=self.dp_policy)[0]
            for ep in range(n_eval_episodes)
        ])

    def _run_episode(self, env, seed, policy=None):
        obs, info = env.reset(seed=seed)
        done = False
        while not done:
            if policy is None:
                masks = np.array(env.action_masks())
                action, _ = self.model.predict(obs, deterministic=True, action_masks=masks)
                a = int(action)
            else:
                x_idx = np.argmin(np.abs(self.X_grid - env.X[env.t]))
                s_idx = np.argmin(np.abs(self.soc_grid - env.soc))
                u = policy[env.t, x_idx, s_idx]
                a = 0 if u > 0 else (2 if u < 0 else 1)
            obs, reward, done, truncated, info = env.step(a)
            done = done or truncated
        return info["revenue"], info["mask_violations"]
    
    def _on_step(self):
        if self.n_calls % max(1, self.eval_freq // self.training_env.num_envs) != 0:
            return True
        
        source = OUSource(theta=self.theta, mu=self.mu, sigma=self.sigma)
        revenues = []
        violations = []
        
        for ep in range(self.n_eval_episodes):
            env = BatteryEnv(source, self.f_eval, self.params, f_sampler=None)
            r, v = self._run_episode(env, ep)
            revenues.append(r); violations.append(v)

        agent_revs = np.array(revenues)
        mean_rev = np.mean(revenues)
        std_rev = np.std(revenues)
        mean_violations = np.mean(violations)
        diff = agent_revs - self.dp_revs
        pct  = 100 * agent_revs.mean() / self.dp_revs.mean()
        
        self.logger.record("eval/mean_revenue", mean_rev)
        self.logger.record("eval/std_revenue", std_rev)
        self.logger.record("eval/min_revenue", np.min(revenues))
        self.logger.record("eval/mask_violations", mean_violations)
        self.logger.record("eval/pct_of_dp", pct)
        self.logger.record("eval/paired_diff_mean", diff.mean())
        
        if mean_rev > self.best_mean:
            self.best_mean = mean_rev
            self.model.save("best_model")
        
        if self.verbose:
            print(f"Step {self.num_timesteps:>9,d} | "
                  f"Agent: ${mean_rev:>9,.0f} | DP: ${self.dp_revs.mean():>9,.0f} | "
                  f"% of DP: {pct:5.1f}% | paired Δ: ${diff.mean():>+9,.0f} | "
                  f"Viol: {mean_violations:.0f}")
        
        return True