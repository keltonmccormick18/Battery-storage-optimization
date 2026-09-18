"""SB3 wiring for production training (docs/experiments/nyiso_setup.md, src/config.py).

Everything that can be computed without SB3 lives in src/rl/scenarios.py and is tested
locally by scripts/check_training_setup.py. The experiment-01 training code in
src/rl/train.py is kept unchanged for reproducibility.
"""
import os

import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from src.rl.evaluate import sb3_predict_fn
from src.rl.scenarios import make_training_env, evaluate_cases


def mask_fn(env):
    return env.action_masks()


def make_vec_env(market, source, params, n_envs):
    def factory():
        return ActionMasker(make_training_env(market, source, params), mask_fn)
    return DummyVecEnv([factory for _ in range(n_envs)])


class MarketEvalCallback(BaseCallback):
    """Evaluate on the market's OU cases every eval_freq timesteps and checkpoint the model.

    Monitoring only: it confirms training works and selects nothing.
    """

    def __init__(self, cases, eval_freq, verbose=1):
        super().__init__(verbose)
        self.cases = cases
        self.eval_freq = eval_freq
        self.best_score_pct = -np.inf

    def _on_step(self):
        if self.n_calls % max(1, self.eval_freq // self.training_env.num_envs) != 0:
            return True

        m = evaluate_cases(self.cases, sb3_predict_fn(self.model))
        for k, v in m.items():
            self.logger.record(f"eval/{k}", v)

        if self.logger.dir is not None:
            self.model.save(os.path.join(self.logger.dir, f"ckpt_{self.num_timesteps}"))
            if m["mc_score_pct_of_dp"] > self.best_score_pct:
                self.best_score_pct = m["mc_score_pct_of_dp"]
                self.model.save(os.path.join(self.logger.dir, "best_mc"))

        if self.verbose:
            line = (f"Step {self.num_timesteps:>9,d} | score {m['mc_score_pct_of_dp']:5.1f}% of DP "
                    f"(worst case {m['mc_worst_case_score_pct']:5.1f}%) | cash {m['mc_pct_of_dp']:5.1f}%")
            if "sw_pct_of_dp" in m:
                line += (f" | SW {m['sw_pct_of_dp']:5.1f}% (paired ${m['sw_paired_diff_mean']:+,.0f} "
                         f"± {m['sw_paired_diff_se']:,.0f})")
            print(line + f" | Viol: {m['mask_violations']}", flush=True)
        return True
