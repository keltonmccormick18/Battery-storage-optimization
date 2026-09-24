"""Production training: one agent per market, training source and seed, with the frozen config.

    python scripts/train.py --market NYIS --source ou --seed 100

Writes tensorboard logs and a checkpoint at every evaluation to --logdir, and the final model
plus a manifest (config, git state, versions, timings, model hash) to --ckptdir.
Use seeds distinct from experiment 01 (0 and 1), e.g. 100-104: the runs that chose the
configuration should not also be the ones that report results.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")      # must precede the torch import
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
torch.set_num_threads(1)
import sb3_contrib
import stable_baselines3
from sb3_contrib import MaskablePPO

from src.config import PPO, N_ENVS, TOTAL_TIMESTEPS, EVAL_FREQ
from src.provenance import git_state
from src.rl.scenarios import MARKETS, SOURCES, training_params, build_eval_cases
from src.rl.training import make_vec_env, MarketEvalCallback


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--market", required=True, choices=MARKETS)
    ap.add_argument("--source", required=True, choices=SOURCES)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, default=None, help="override TOTAL_TIMESTEPS (smoke tests only)")
    ap.add_argument("--forecast", default="base", choices=["base", "shape"],
                    help="training prior for the seasonal forecast; 'shape' matches the intraday "
                         "shape forecast (docs/experiments/shape_forecast.md)")
    ap.add_argument("--eval-freq", type=int, default=EVAL_FREQ)
    ap.add_argument("--logdir", default="/content/runs")
    ap.add_argument("--ckptdir", default="/content/drive/MyDrive/battery_rl/models")
    a = ap.parse_args()

    name = f"rl_{a.source}_{a.market}_s{a.seed}" + ("_shape" if a.forecast == "shape" else "")
    ckptdir = Path(a.ckptdir)
    ckptdir.mkdir(parents=True, exist_ok=True)          # fail fast on a bad path
    total = a.steps or TOTAL_TIMESTEPS
    commit, dirty = git_state()
    manifest_path = ckptdir / f"{name}_manifest.json"
    manifest = {
        "name": name, "market": a.market, "source": a.source, "seed": a.seed,
        "forecast": a.forecast,
        "total_timesteps": total, "smoke_test": a.steps is not None,
        "eval_freq": a.eval_freq, "n_envs": N_ENVS, "ppo": PPO,
        "git_commit": commit, "git_dirty": dirty,
        "versions": {"python": sys.version.split()[0], "torch": torch.__version__,
                     "stable_baselines3": stable_baselines3.__version__, "sb3_contrib": sb3_contrib.__version__},
        "started_at": now(), "status": "running",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    if dirty:
        print(f"[{name}] WARNING: uncommitted source changes; the manifest records git_dirty=true", flush=True)

    params = training_params(a.market)
    t0 = time.time()
    cases = build_eval_cases(a.market, params, forecast=a.forecast)
    print(f"[{name}] {len(cases['mc'])} evaluation cases"
          f"{' + Gate 2 window' if cases['sw'] else ''} built in {time.time() - t0:.0f}s; "
          f"training {total:,} timesteps", flush=True)

    venv = make_vec_env(a.market, a.source, params, N_ENVS, a.forecast)
    model = MaskablePPO("MlpPolicy", venv, seed=a.seed, verbose=0, tensorboard_log=a.logdir, **PPO)
    t0 = time.time()
    model.learn(total_timesteps=total, callback=MarketEvalCallback(cases, a.eval_freq), tb_log_name=name)
    venv.close()

    final = ckptdir / f"{name}_final.zip"
    model.save(final)
    manifest.update({
        "status": "finished", "finished_at": now(), "train_seconds": round(time.time() - t0),
        "final_model": str(final), "final_model_sha256": hashlib.sha256(final.read_bytes()).hexdigest(),
    })
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"[{name}] done in {manifest['train_seconds'] / 60:.0f} min -> {final}", flush=True)


if __name__ == "__main__":
    main()
