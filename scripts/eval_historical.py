"""Evaluate a trained agent or a rule-based baseline on every historical walk-forward week.

    python scripts/eval_historical.py --market CISO --model <path>.zip --name rl_ou_s0
    python scripts/eval_historical.py --market CISO --baseline schedule

Checks the agent for determinism and lookahead on a sample of weeks first. Rows are
written to results/historical_{market}.csv only if the full evaluation passes the
validity rule in docs/experiments/scoring_rule.md.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.windows import get_windows
from src.rl.evaluate import evaluate_agent, sb3_predict_fn, check_validity, save_results, summarize
from src.rl.gates import check_determinism, check_no_lookahead
from src.baselines import BASELINES


def git_state():
    try:
        commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                                    capture_output=True, text=True).stdout.strip())
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return "unknown", None


def run(market, predict_fn, name, model_path=None, check_weeks=6, save_actions=False, results_dir=None):
    results_dir = Path(results_dir or ROOT / "results")
    windows = get_windows(market)

    sample = [windows[i] for i in np.linspace(0, len(windows) - 1, check_weeks).round().astype(int)]
    same_policy = lambda ws: predict_fn          # an agent's policy doesn't depend on the week list
    check_determinism(sample, same_policy)
    changed = check_no_lookahead(sample, same_policy)
    print(f"[{name}] {market}: deterministic and no lookahead on {check_weeks} sample weeks "
          f"(later actions responded to changed prices in {changed:.0%} of cases; "
          f"0% means the policy ignores prices)", flush=True)

    table, actions = evaluate_agent(market, windows, predict_fn, name, record_actions=save_actions)
    check_validity(table)

    path = save_results(table, market, results_dir)
    if save_actions:
        np.savez_compressed(results_dir / f"actions_{market}_{name}.npz",
                            actions=actions, week_idx=table.week_idx.values)
    commit, dirty = git_state()
    meta = {
        "market": market,
        "method": name,
        "weeks": len(table),
        "model_path": str(model_path) if model_path else None,
        "model_sha256": hashlib.sha256(Path(model_path).read_bytes()).hexdigest() if model_path else None,
        "git_commit": commit,
        "git_dirty": dirty,
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (results_dir / f"meta_{market}_{name}.json").write_text(json.dumps(meta, indent=2))

    s = summarize(table, name)
    print(f"[{name}] {market}: {s['weeks']} weeks | mean score ${s['mean_score']:,.0f} | "
          f"value capture {s['value_capture']:.1%} | Sharpe {s['sharpe']:.2f} | "
          f"win rate {s['win_rate']:.0%} | wrote {path}")
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=["CISO", "NYIS"])
    policy = ap.add_mutually_exclusive_group(required=True)
    policy.add_argument("--model", help="path to a MaskablePPO .zip")
    policy.add_argument("--baseline", choices=sorted(BASELINES), help="rule from src/baselines.py")
    ap.add_argument("--name", help="method label, e.g. rl_ou_s0; required with --model, "
                                   "baselines default to their own name")
    ap.add_argument("--check-weeks", type=int, default=6)
    ap.add_argument("--save-actions", action="store_true")
    ap.add_argument("--results-dir", default=None, help="default: results/ in the repo")
    a = ap.parse_args()

    name = a.name or a.baseline
    if name is None:
        ap.error("--name is required with --model")
    if name in {"dp", "pf"} or (a.model and name in BASELINES):
        ap.error(f"--name {name} would overwrite rows belonging to another method")

    if a.baseline:
        predict_fn = BASELINES[a.baseline]()
    else:
        if not Path(a.model).exists():
            ap.error(f"model not found: {a.model} (is Google Drive mounted?)")
        from sb3_contrib import MaskablePPO          # imported here so baselines run without SB3
        predict_fn = sb3_predict_fn(MaskablePPO.load(a.model, device="cpu"))

    run(a.market, predict_fn, name, model_path=a.model, check_weeks=a.check_weeks,
        save_actions=a.save_actions, results_dir=a.results_dir)


if __name__ == "__main__":
    main()
