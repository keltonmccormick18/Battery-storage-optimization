"""Rule-based baselines for the historical comparison (docs/experiments/baselines.md).

Both are parameter-free, use only information available at decision time, and have
the predict_fn(obs, masks, envs) -> actions interface, so they run through the same
harness as the DP and the agents. Actions: 0 discharge, 1 hold, 2 charge.
"""
import numpy as np

DISCHARGE, HOLD, CHARGE = 0, 1, 2


def _apply_masks(desired, masks):
    """Replace any infeasible action with hold."""
    feasible = masks[np.arange(len(desired)), desired]
    return np.where(feasible, desired, HOLD)


def schedule_predict_fn():
    """Fixed daily schedule from the week's seasonal forecast.

    Average the forecast at each hour position (t mod 24) over the episode, then charge
    at the 4 cheapest positions and discharge at the 4 dearest, every day.
    4 = S_max / u_max, the battery's duration. Never observes realized prices.
    """
    def predict(obs, masks, envs):
        profile = np.stack([e.f[:(e.T // 24) * 24].reshape(-1, 24).mean(axis=0) for e in envs])
        rank = np.argsort(np.argsort(profile, axis=1, kind="stable"), axis=1)   # 0 = cheapest
        k = np.array([int(round(e.S_max / e.u_max)) for e in envs])
        r = rank[np.arange(len(envs)), np.array([e.t % 24 for e in envs])]
        desired = np.where(r < k, CHARGE, np.where(r >= 24 - k, DISCHARGE, HOLD))
        return _apply_masks(desired, masks)
    return predict


def threshold_predict_fn():
    """Act whenever the current price beats the value of stored energy.

    Charge if price < eta * q, discharge if price > q, otherwise hold. Charging stores
    eta MWh per MWh bought, worth eta * q; discharging gives up q per MWh sold -- so these
    are the break-even prices under the q valuation every method is scored with, and this
    is the greedy policy on the RL shaping reward. Uses only the current realized price.
    """
    def predict(obs, masks, envs):
        price = np.array([e.f[e.t] + e.X[e.t] for e in envs])
        q = np.array([e.q for e in envs])
        eta = np.array([e.eta for e in envs])
        desired = np.where(price < eta * q, CHARGE, np.where(price > q, DISCHARGE, HOLD))
        return _apply_masks(desired, masks)
    return predict


BASELINES = {"schedule": schedule_predict_fn, "threshold": threshold_predict_fn}
