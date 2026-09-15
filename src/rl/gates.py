"""Checks that a policy scored by the historical harness is scored fairly.

Each check takes make_predict_fn(windows) -> predict_fn, so the same check works
for trained agents (which ignore the window list) and for the DP (which needs it).
"""
import numpy as np

from src.rl.evaluate import replay_batch, dp_predict_fn


def check_determinism(windows, make_predict_fn):
    """Two identical replays must produce identical actions and revenue."""
    a = replay_batch(windows, make_predict_fn(windows), record_actions=True)
    b = replay_batch(windows, make_predict_fn(windows), record_actions=True)
    if not (np.array_equal(a["actions"], b["actions"]) and np.array_equal(a["revenue"], b["revenue"])):
        raise AssertionError("policy is not deterministic on replay")


def perturb_future(w, cut, rng):
    """Copy of window w with large noise added to every residual after hour `cut`."""
    x = w["X_eval_resid"].copy()
    x[cut + 1:] += rng.normal(0.0, 3.0 * w["sigma_stat"], len(x) - cut - 1)
    return {**w, "X_eval_resid": x, "prices": w["f_eval"] + x}


def check_no_lookahead(windows, make_predict_fn, cuts=(24, 84, 150), seed=0):
    """Actions through hour `cut` must not change when prices after `cut` change.

    Returns the share of cases where actions *after* the cut did change. If that is
    zero, the perturbation never mattered and the check proved nothing.
    """
    rng = np.random.default_rng(seed)
    cases = [(i, c) for i in range(len(windows)) for c in cuts]
    base_ws = [windows[i] for i, _ in cases]
    pert_ws = [perturb_future(windows[i], c, rng) for i, c in cases]
    base = replay_batch(base_ws, make_predict_fn(base_ws), record_actions=True)["actions"]
    pert = replay_batch(pert_ws, make_predict_fn(pert_ws), record_actions=True)["actions"]

    for k, (i, c) in enumerate(cases):
        before_b, before_p = base[:c + 1, k], pert[:c + 1, k]
        if not np.array_equal(before_b, before_p):
            t = int(np.argmax(before_b != before_p))
            raise AssertionError(f"lookahead: week {windows[i]['week_idx']}, action at hour {t} "
                                 f"changed when only prices after hour {c} changed")
    return float(np.mean([not np.array_equal(base[c + 1:, k], pert[c + 1:, k])
                          for k, (_, c) in enumerate(cases)]))


def peeking_dp_predict_fn(windows):
    """Negative control: the DP, overridden by the price 24 hours ahead. Must fail check_no_lookahead."""
    honest = dp_predict_fn(windows)

    def predict(obs, masks, envs):
        a = honest(obs, masks, envs)
        for k, e in enumerate(envs):
            ahead = e.X[min(e.t + 24, e.T - 1)]
            if ahead > e.X[e.t]:
                a[k] = 1                  # hold: prices rise later
            elif masks[k, 0]:
                a[k] = 0                  # discharge: prices fall later
        return a
    return predict
