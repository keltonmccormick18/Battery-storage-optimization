"""Rule-based baselines for the historical comparison (docs/experiments/baselines.md).

All three are parameter-free, use only information available at decision time, and have
the predict_fn(obs, masks, envs) -> actions interface, so they run through the same
harness as the DP and the agents. Actions: 0 discharge, 1 hold, 2 charge.
"""
import numpy as np

from src.simulation import deterministic_plan
from src.windows import SOC_GRID, EVAL_WINDOW

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


def forecast_optimal_predict_fn(horizon=EVAL_WINDOW):
    """Optimal plan under the seasonal forecast alone: the ceiling for any residual-blind policy.

    Backward induction over f (the week's forecast, fit on the training window and so known
    before the week starts), then executed against realised prices. Closed-loop in SOC,
    blind to price: it never reads X. Any policy that ignores the residual scores at or below
    this, so it -- not the fixed schedule -- is the bar the DP has to clear to show that
    modelling the residual is worth anything.

    Planned over `horizon` hours with terminal value q * SOC, which is exactly the scored
    objective (like perfect foresight, and unlike the DP, which plans over the full 336-hour
    episode). Horizon matters: the 336-hour variant scores $91/week less on CISO and
    $194/week less on NYISO, so part of the DP's shortfall against this baseline is its
    longer planning horizon, not its residual model. docs/experiments/baselines.md
    separates the two.

    Added after the pre-registered comparison was run: a diagnostic benchmark, not a
    registered claim. See docs/experiments/baselines.md.
    """
    plans = {}

    def predict(obs, masks, envs):
        desired = np.empty(len(envs), dtype=int)
        for i, e in enumerate(envs):
            if e.t == 0:                      # one solve per episode, on that episode's forecast
                params = {"u_max": e.u_max, "S_max": e.S_max, "eta": e.eta, "q": e.q, "dt": e.dt}
                plans[id(e)] = deterministic_plan(e.f[:min(horizon, e.T)], SOC_GRID, params)
            plan = plans[id(e)]
            if e.t >= len(plan):              # past the scored week: nothing left to optimise
                desired[i] = HOLD
                continue
            u = plan[e.t, np.argmin(np.abs(SOC_GRID - e.soc))]
            desired[i] = DISCHARGE if u > 0 else (CHARGE if u < 0 else HOLD)
        return _apply_masks(desired, masks)
    return predict


def forecast_optimal_336_predict_fn():
    """Horizon-matched variant of forecast_optimal: plans over the whole 336-hour episode,
    as the DP does, instead of the 168 scored hours. The difference between the two is the
    cost of the DP's longer planning horizon, which is not a cost of modelling the residual.
    """
    return forecast_optimal_predict_fn(horizon=336)


BASELINES = {"schedule": schedule_predict_fn, "threshold": threshold_predict_fn,
             "forecast_optimal": forecast_optimal_predict_fn,
             "forecast_optimal_336": forecast_optimal_336_predict_fn}
