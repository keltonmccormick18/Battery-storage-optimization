"""Local checks for the batched agent-evaluation path. No SB3 needed.

1. Batched replay driven by the DP policy reproduces evaluate_dp exactly.
2. Replay is deterministic.
3. check_no_lookahead passes an honest policy...
4. ...and catches one that reads future prices (negative control).

Run scripts/check_dp_reproduction.py first; it writes the DP rows compared against.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src.windows import get_windows
from src.rl.evaluate import evaluate_agent, dp_predict_fn, check_validity
from src.rl.gates import check_determinism, check_no_lookahead, peeking_dp_predict_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="+", default=["CISO", "NYIS"])
    ap.add_argument("--weeks", type=int, default=6, help="weeks per market; each DP policy is ~44 MB")
    a = ap.parse_args()

    for market in a.markets:
        stored = pd.read_csv(ROOT / "results" / f"historical_{market}.csv")
        stored = stored[stored.method == "dp"].set_index("week_idx")
        windows = get_windows(market)
        ws = [windows[i] for i in np.linspace(0, len(windows) - 1, a.weeks).round().astype(int)]

        table, _ = evaluate_agent(market, ws, dp_predict_fn(ws), "dp_batched")
        check_validity(table)
        t = table.set_index("week_idx")
        diff = float(np.abs(t.score - stored.score.loc[t.index]).max())
        assert diff < 0.01, f"{market}: batched replay differs from evaluate_dp by ${diff:.4f}"
        print(f"{market}: batched replay matches evaluate_dp on weeks {list(t.index)} (max |diff| ${diff:.1e})")

        check_determinism(ws[:3], dp_predict_fn)
        print("  deterministic")

        changed = check_no_lookahead(ws[:3], dp_predict_fn)
        print(f"  no lookahead; perturbation changed later actions in {changed:.0%} of cases")

        try:
            check_no_lookahead(ws[:3], peeking_dp_predict_fn)
        except AssertionError as e:
            print(f"  negative control caught: {e}")
        else:
            raise AssertionError("check_no_lookahead missed a policy that reads future prices")


if __name__ == "__main__":
    main()
