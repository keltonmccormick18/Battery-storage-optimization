"""Re-run the comparison under the intraday shape forecast (docs/experiments/shape_forecast.md).

    python scripts/run_shape_forecast.py                    # both markets, DP + PF + baselines
    python scripts/run_shape_forecast.py --markets CISO

Writes results/historical_{market}_shape.csv, leaving the registered results untouched, so the
two forecasts can be compared week by week. The RL agents are not re-run: their training priors
are calibrated to the old residual distribution and a fair re-run means retraining.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.windows import get_windows, SHAPE_DAYS
from src.rl.evaluate import evaluate_dp, evaluate_agent, save_results, check_validity, summarize
from src.rl.gates import check_determinism, check_no_lookahead
from src.baselines import BASELINES
from src.provenance import git_state

METHODS = ["dp", "forecast_optimal", "schedule", "threshold", "forecast_optimal_336"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=["CISO", "NYIS"])
    ap.add_argument("--check-weeks", type=int, default=6)
    a = ap.parse_args()

    for market in a.markets:
        commit, dirty = git_state()
        label = f"{market}_shape"
        windows = get_windows(market, shape_days=SHAPE_DAYS)
        print(f"\n== {label}: {len(windows)} weeks, {SHAPE_DAYS}-day shape profile", flush=True)

        table, sim = evaluate_dp(label, windows)
        d_sim = np.abs(table[table.method == "dp"].sort_values("week_idx").revenue.values - sim).max()
        assert d_sim < 0.01, f"{label}: env replay differs from simulate() by up to ${d_sim:.4f}"
        check_validity(table)
        save_results(table, label)
        print(f"   dp and pf scored, env replay matches simulate() to ${d_sim:.1e}", flush=True)

        for name in sorted(BASELINES):
            predict_fn = BASELINES[name]()
            sample = [windows[i] for i in
                      np.linspace(0, len(windows) - 1, a.check_weeks).round().astype(int)]
            check_determinism(sample, lambda ws: predict_fn)
            check_no_lookahead(sample, lambda ws: predict_fn)
            t, _ = evaluate_agent(label, windows, predict_fn, name)
            check_validity(t)
            save_results(t, label)
            print(f"   {name} scored", flush=True)

        meta = {
            "market": market, "label": label, "shape_days": SHAPE_DAYS,
            "weeks": len(windows), "methods": METHODS,
            "git_commit": commit, "git_dirty": dirty,
            "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (ROOT / "results" / f"meta_{label}.json").write_text(json.dumps(meta, indent=2))

        import pandas as pd
        full = pd.read_csv(ROOT / "results" / f"historical_{label}.csv", float_precision="round_trip")
        pf = summarize(full, "pf")["mean_score"]
        print(f"   {'method':22s} {'mean score':>12s} {'of PF':>8s}")
        for m in ["pf"] + METHODS:
            s = summarize(full, m)
            print(f"   {m:22s} ${s['mean_score']:>11,.0f} {s['mean_score']/pf:>8.1%}")


if __name__ == "__main__":
    main()
