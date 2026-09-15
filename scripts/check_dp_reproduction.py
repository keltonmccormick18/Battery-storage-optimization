"""Gate: refactored windows + env replay reproduce the frozen DP backtest exactly,
and scored results satisfy the validity rule in docs/experiments/scoring_rule.md.

Run scripts/capture_dp_reference.py first.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.windows import get_windows
from src.rl.evaluate import evaluate_dp, save_results, check_validity, summarize

# README figures under the old cash-only scoring, kept to confirm reproduction.
README_CASH = {
    "CISO": {"mean": 13_717, "trimmed_sharpe": 1.75, "vc_mean": 0.75, "vc_ratio": 0.82},
    "NYIS": {"mean": 10_229, "trimmed_sharpe": 1.51, "vc_mean": 0.86, "vc_ratio": 0.72},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=["CISO", "NYIS"])
    a = ap.parse_args()

    for market in a.markets:
        ref = pd.read_csv(ROOT / "results" / f"reference_dp_{market}.csv")
        table, sim = evaluate_dp(market, get_windows(market))
        dp = table[table.method == "dp"].sort_values("week_idx")
        pf = table[table.method == "pf"].sort_values("week_idx")

        assert len(dp) == len(ref), f"{market}: {len(dp)} weeks vs {len(ref)} in reference"
        assert (dp.eval_start.values == ref.eval_start.values).all(), f"{market}: weeks misaligned"
        d_ref = np.abs(dp.revenue.values - ref.revenue.values).max()
        d_sim = np.abs(dp.revenue.values - sim).max()
        assert d_ref < 0.01, f"{market}: differs from frozen DP by up to ${d_ref:.4f}"
        assert d_sim < 0.01, f"{market}: env replay differs from simulate() by up to ${d_sim:.4f}"
        check_validity(table)

        rev, pf_cash = dp.revenue.values, pf.revenue.values
        trimmed = np.sort(rev)[5:-5]
        cash = {
            "mean": rev.mean(),
            "trimmed_sharpe": trimmed.mean() / trimmed.std(),
            "vc_mean": np.mean([r / p if p > 0 else 0.0 for r, p in zip(rev, pf_cash)]),
            "vc_ratio": rev.sum() / pf_cash.sum(),
        }

        path = save_results(table, market)
        print(f"{market}: PASS  {len(dp)} weeks | max |diff| vs frozen ${d_ref:.2e}, "
              f"vs simulate ${d_sim:.2e} | no week beats perfect foresight | wrote {path.name}")
        print("  cash-only scoring (reproduces README):")
        for k, v in cash.items():
            print(f"    {k:18s} {v:12.4f}   README {README_CASH[market][k]}")
        print("  scoring rule (cash + q * SOC_168):")
        for k, v in summarize(table, "dp").items():
            print(f"    {k:18s} {v:12.4f}")
        print(f"    {'pf_mean_score':18s} {pf.score.mean():12.4f}")


if __name__ == "__main__":
    main()
