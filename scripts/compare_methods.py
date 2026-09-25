"""Paired comparison of every scored method against the DP, per market.

    python scripts/compare_methods.py

Reads results/historical_{market}.csv and applies docs/experiments/scoring_rule.md:
paired weekly differences, circular block bootstrap CI, and the registered verdicts.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from src.stats import paired_comparison, method_groups, check_margin, MARGIN, N_BOOT
from src.rl.evaluate import summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=["CISO", "NYIS"])
    ap.add_argument("--suffix", default="", help="e.g. _shape to read historical_{market}_shape.csv")
    ap.add_argument("--head-to-head", nargs=2, default=["rl_boot", "rl_ou"],
                    metavar=("A", "B"), help="paired comparison of two policies, A against B")
    a_args = ap.parse_args()
    a = a_args

    for market in a.markets:
        table = pd.read_csv(ROOT / "results" / f"historical_{market}{a.suffix}.csv",
                            float_precision="round_trip")
        dp = summarize(table, "dp")
        pf = summarize(table, "pf")
        warning = check_margin(table, market)

        print(f"== {market}{a.suffix}: {dp['weeks']} weeks | DP mean score ${dp['mean_score']:,.0f} "
              f"({dp['value_capture']:.1%} of perfect foresight, ${pf['mean_score']:,.0f})")
        if warning:
            print(f"   note: {warning}")
        print(f"   {N_BOOT:,} circular block bootstrap resamples, "
              f"equivalence margin ±${MARGIN[market]:,.0f}")
        print(f"   {'method':12s} {'capture':>8s} {'mean vs DP':>11s} {'% of DP':>8s} "
              f"{'95% CI':>20s} {'beats DP':>9s}  verdict")
        for label, methods in method_groups(table).items():
            r = paired_comparison(table, methods, market)
            s = summarize(table, methods[0]) if len(methods) == 1 else None
            capture = f"{s['value_capture']:.1%}" if s else ""
            ci = f"[{r['ci_low']:+,.0f}, {r['ci_high']:+,.0f}]"
            seeds = f" ({r['seeds']} seeds: {r['seed_min']:+,.0f} to {r['seed_max']:+,.0f})" if r["seeds"] > 1 else ""
            print(f"   {label:12s} {capture:>8s} ${r['mean_delta']:>+10,.0f} {r['pct_of_baseline']:>+8.1%} "
                  f"{ci:>20s} {r['beats_baseline']:>9.0%}  {r['verdict']}{seeds}")

        groups = method_groups(table)
        h_a, h_b = a_args.head_to_head
        if h_a in groups and h_b in groups:
            r = paired_comparison(table, groups[h_a], market, baseline=groups[h_b])
            ci = f"[{r['ci_low']:+,.0f}, {r['ci_high']:+,.0f}]"
            verdict = r["verdict"].replace("DP", h_b)
            print(f"   head to head: {h_a} vs {h_b}  ${r['mean_delta']:>+9,.0f} ({r['pct_of_baseline']:+.1%})  "
                  f"95% CI {ci}  {h_a} ahead in {r['beats_baseline']:.0%} of weeks  ->  {verdict}")
        elif h_a in groups or h_b in groups:
            print(f"   head to head {h_a} vs {h_b}: skipped, only one of them is in the results")
        print()


if __name__ == "__main__":
    main()
