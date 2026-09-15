"""Record per-week DP revenue from the frozen, pre-refactor implementation.

Exports the repo at a git ref to a temp directory and runs that commit's own
walk_forward_backtest, a few weeks at a time so the policies it keeps in memory
stay small. The reproduction gate compares the refactored code against this.
"""
import argparse
import contextlib
import gc
import io
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TRAIN, EVAL, BUFFER, STEP = 24 * 365, 24 * 7, 24 * 7, 24 * 7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="rl-config-frozen")
    ap.add_argument("--repo", default=str(ROOT))
    ap.add_argument("--markets", nargs="+", default=["CISO", "NYIS"])
    ap.add_argument("--chunk", type=int, default=5, help="weeks per call (~44 MB of policies each)")
    ap.add_argument("--max-weeks", type=int, default=None, help="quick check on the first N weeks")
    ap.add_argument("--out-dir", default=str(ROOT / "results"))
    a = ap.parse_args()

    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(["git", "-C", a.repo, "archive", a.ref],
                             check=True, capture_output=True).stdout

    with tempfile.TemporaryDirectory() as tmp:
        tarfile.open(fileobj=io.BytesIO(archive)).extractall(tmp)
        sys.path.insert(0, tmp)
        from src.simulation import walk_forward_backtest   # the frozen implementation

        for market in a.markets:
            data = pd.read_csv(ROOT / "data" / f"prices_{market}.csv")
            n_total = len(range(0, len(data) - TRAIN - EVAL - BUFFER, STEP))
            if a.max_weeks:
                n_total = min(n_total, a.max_weeks)

            rows = []
            for k in range(0, n_total, a.chunk):
                n = min(a.chunk, n_total - k)
                offset = k * STEP
                chunk = data.iloc[offset:offset + TRAIN + EVAL + BUFFER + STEP * n].reset_index(drop=True)
                with contextlib.redirect_stdout(io.StringIO()):
                    res = walk_forward_backtest(chunk)
                assert len(res) == n, f"{market} chunk {k}: got {len(res)} weeks, expected {n}"
                rows += [{"week_idx": k + i, "eval_start": r["eval_start"] + offset,
                          "revenue": r["revenue"]} for i, r in enumerate(res)]
                del res
                gc.collect()

            ref = pd.DataFrame(rows)
            path = out_dir / f"reference_dp_{market}.csv"
            ref.to_csv(path, index=False, float_format="%.10f")
            print(f"{market}: {len(ref)} weeks from {a.ref}, mean ${ref.revenue.mean():,.0f} -> {path}")


if __name__ == "__main__":
    main()