import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from src.rl.env import BatteryEnv
from src.rl.sources import OUSource, RandomizedOUSource, sample_f, sample_calib
from stable_baselines3.common.callbacks import BaseCallback
from src.optimization import build_transition_matrix, get_optimal_policy
import os


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
        self.f_eval, self.params = f_eval, params
        self.theta, self.mu, self.sigma = theta, mu, sigma
        self.n_eval_episodes, self.eval_freq = n_eval_episodes, eval_freq
        self.best_mean = -np.inf
        self.soc_grid = np.linspace(0, 100, 81)

        # single window (Gate 2)
        ss = sigma / np.sqrt(2 * theta)
        Xg = np.linspace(mu - 5*ss, mu + 5*ss, 200)
        pol = get_optimal_policy(build_transition_matrix(theta, mu, sigma, Xg),
                                 f_eval, Xg, self.soc_grid, params)
        src = OUSource(theta, mu, sigma)
        seeds = list(range(n_eval_episodes))
        self.sw_case = (src, f_eval, seeds,
                        np.array([self._run_dp(src, f_eval, Xg, pol, s) for s in seeds]))
        # multi-calibration, drawn from the training prior
        rng = np.random.default_rng(999)
        self.mc_cases = []
        for c in range(20):
            th, m, sg = sample_calib(rng)                
            f = sample_f(rng, 336, params["q"])
            ss = sg / np.sqrt(2 * th)
            Xg = np.linspace(m - 5*ss, m + 5*ss, 200)
            pol = get_optimal_policy(build_transition_matrix(th, m, sg, Xg),
                                     f, Xg, self.soc_grid, params)
            src = OUSource(th, m, sg)
            seeds = [20_000 + 100*c + k for k in range(10)]   # distinct per case
            dp = np.array([self._run_dp(src, f, Xg, pol, s) for s in seeds])
            self.mc_cases.append((src, f, seeds, dp))

    def _run_dp(self, src, f, Xg, pol, seed):
        env = BatteryEnv(src, f, self.params, f_sampler=None)
        env.reset(seed=seed)
        for _ in range(env.T):
            x_idx = np.argmin(np.abs(Xg - env.X[env.t]))
            s_idx = np.argmin(np.abs(self.soc_grid - env.soc))
            u = pol[env.t, x_idx, s_idx]
            env.step(0 if u > 0 else (2 if u < 0 else 1))
        return env.total_revenue
    
    def _agent_revs_batched(self, cases):
        envs, seeds = [], []
        for src, f, seed_list, _dp in cases:
            for s in seed_list:
                envs.append(BatteryEnv(src, f, self.params, f_sampler=None))
                seeds.append(s)
        obs = np.stack([e.reset(seed=s)[0] for e, s in zip(envs, seeds)])
        for _ in range(envs[0].T):
            masks = np.stack([e.action_masks() for e in envs])
            actions, _ = self.model.predict(obs, deterministic=True, action_masks=masks)
            obs = np.stack([e.step(int(a))[0] for e, a in zip(envs, actions)])
        revs = np.array([e.total_revenue for e in envs]).reshape(len(cases), -1)
        viol = sum(e.n_mask_violations for e in envs)
        return revs, viol

    def _on_step(self):
        if self.n_calls % max(1, self.eval_freq // self.training_env.num_envs) != 0:
            return True

        sw_agent, sw_viol = self._agent_revs_batched([self.sw_case])
        mc_agent, mc_viol = self._agent_revs_batched(self.mc_cases)

        sw_agent = sw_agent.ravel()
        sw_dp = self.sw_case[3]
        mc_dp = np.stack([c[3] for c in self.mc_cases])               # (20, 10)

        mc_pct = 100 * mc_agent.sum() / mc_dp.sum()                   # decision metric
        mc_case_pct = 100 * mc_agent.mean(axis=1) / mc_dp.mean(axis=1)
        sw_pct = 100 * sw_agent.mean() / sw_dp.mean()
        diff = sw_agent - sw_dp
        diff_se = diff.std(ddof=1) / np.sqrt(len(diff))
        violations = sw_viol + mc_viol

        self.logger.record("eval/mc_pct_of_dp", mc_pct)
        self.logger.record("eval/mc_worst_case_pct", mc_case_pct.min())
        self.logger.record("eval/sw_pct_of_dp", sw_pct)
        self.logger.record("eval/sw_agent_mean", sw_agent.mean())
        self.logger.record("eval/sw_paired_diff_mean", diff.mean())
        self.logger.record("eval/sw_paired_diff_se", diff_se)
        self.logger.record("eval/mask_violations", violations)

        if self.logger.dir is not None:
            self.model.save(os.path.join(self.logger.dir, f"ckpt_{self.num_timesteps}"))
            if mc_pct > self.best_mean:                                # best by decision metric
                self.best_mean = mc_pct
                self.model.save(os.path.join(self.logger.dir, "best_mc"))

        if self.verbose:
            print(f"Step {self.num_timesteps:>9,d} | "
                  f"MC: {mc_pct:5.1f}% (worst case {mc_case_pct.min():5.1f}%) | "
                  f"SW: {sw_pct:5.1f}% | "
                  f"paired Δ: ${diff.mean():>+8,.0f} ± {diff_se:,.0f} | "
                  f"Viol: {violations}", flush=True)

        return True