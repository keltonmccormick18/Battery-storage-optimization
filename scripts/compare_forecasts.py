"""Old forecast against the intraday shape forecast, and the predictions registered for it.

    python scripts/compare_forecasts.py

Reads results/historical_{market}.csv and results/historical_{market}_shape.csv. Same paired
block bootstrap as the registered comparison (docs/experiments/scoring_rule.md).
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.stats import block_bootstrap_ci, paired_comparison, MARGIN
from src.windows import get_windows, SHAPE_DAYS

SHARED = ["dp", "forecast_optimal", "schedule", "threshold"]


def load(market, shape):
    p = ROOT / "results" / f"historical_{market}{'_shape' if shape else ''}.csv"
    return pd.read_csv(p, float_precision="round_trip")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=["CISO", "NYIS"])
    a = ap.parse_args()

    for market in a.markets:
        old, new = load(market, False), load(market, True)
        wo = old.pivot(index="week_idx", columns="method", values="score")
        wn = new.pivot(index="week_idx", columns="method", values="score")
        pf_o, pf_n = wo["pf"].mean(), wn["pf"].mean()
        assert np.allclose(wo["pf"], wn["pf"]), f"{market}: perfect foresight moved between runs"

        print(f"\n== {market}: {len(wn)} weeks | perfect foresight ${pf_n:,.0f} (unchanged)")
        print(f"   {'method':20s} {'old f':>10s} {'shape f':>10s} {'change':>10s} "
              f"{'95% CI':>20s} {'of PF':>14s}")
        for m in SHARED:
            d = (wn[m] - wo[m]).to_numpy()
            lo, hi = block_bootstrap_ci(d)
            print(f"   {m:20s} ${wo[m].mean():>9,.0f} ${wn[m].mean():>9,.0f} "
                  f"${d.mean():>+9,.0f} {f'[{lo:+,.0f}, {hi:+,.0f}]':>20s} "
                  f"{wo[m].mean()/pf_o:>6.1%} -> {wn[m].mean()/pf_n:<5.1%}")

        ceil = wn["forecast_optimal"]
        gap = (wn["dp"] - ceil).to_numpy()
        lo, hi = block_bootstrap_ci(gap)
        sched = float(wn["schedule"].mean() / ceil.mean())
        r = paired_comparison(new, ["schedule"], market)

        print(f"\n   registered predictions (docs/experiments/shape_forecast.md)")
        p1 = hi < MARGIN[market]
        print(f"   1. DP stays at or below the new ceiling      "
              f"DP - ceiling ${gap.mean():+,.0f}, CI [{lo:+,.0f}, {hi:+,.0f}]   "
              f"{'MET' if p1 else 'FAILED'}")
        print(f"   2. schedule within ~5% of the new ceiling    "
              f"{sched:.1%} of it   {'MET' if sched > 0.95 else 'FAILED'}")
        print(f"   3. DP vs schedule still inconclusive         "
              f"${r['mean_delta']:+,.0f}, CI [{r['ci_low']:+,.0f}, {r['ci_high']:+,.0f}]"
              f" -> {r['verdict']}   {'MET' if r['verdict'] == 'inconclusive' else 'FAILED'}")

        w0, w1 = get_windows(market), get_windows(market, shape_days=SHAPE_DAYS)
        ss = [np.mean([w["sigma_stat"] for w in ws]) for ws in (w0, w1)]
        hl = [np.mean([np.log(2) / w["theta"] for w in ws]) for ws in (w0, w1)]
        moved = max(abs(ss[1] / ss[0] - 1), abs(hl[1] / hl[0] - 1))
        print(f"   4. OU calibration barely moves               "
              f"sigma_stat {ss[0]:.1f} -> {ss[1]:.1f}, half-life {hl[0]:.0f}h -> {hl[1]:.0f}h   "
              f"{'MET' if moved < 0.10 else 'FAILED'}")


if __name__ == "__main__":
    main()
