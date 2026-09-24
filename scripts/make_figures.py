"""Figures for the README and writeup, built from results/.

    python scripts/make_figures.py

Writes figures/*.png. The training-curve figure needs results/training_curves.csv, exported
from the tensorboard logs; it is skipped if that file is absent.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.stats import method_groups, paired_comparison
from src.rl.priors import NYIS_PRIOR, draw_scenario
from src.rl.scenarios import training_params
from src.rl.sources import sample_calib, sample_f
from src.windows import load_windows

FIG = ROOT / "figures"
MARKETS = [("CISO", "CISO"), ("NYISO", "NYIS")]
LABEL = {"pf": "perfect foresight", "dp": "DP (stochastic control)", "schedule": "fixed daily schedule",
         "rl_boot": "RL, bootstrap-trained", "rl_ou": "RL, OU-trained", "threshold": "price threshold",
         "forecast_optimal": "forecast-only optimal", "forecast_optimal_336": "forecast-only optimal (336h plan)"}
COLOR = {"pf": "#8c8c8c", "dp": "#1f77b4", "schedule": "#2ca02c",
         "rl_boot": "#d62728", "rl_ou": "#ff7f0e", "threshold": "#9467bd",
         "forecast_optimal": "#17becf", "forecast_optimal_336": "#9edae5"}

plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150, "font.size": 9,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "legend.frameon": False})


def load(code):
    t = pd.read_csv(ROOT / "results" / f"historical_{code}.csv", float_precision="round_trip")
    wide = t.pivot(index="week_idx", columns="method", values="score")
    groups = {"dp": ["dp"], "pf": ["pf"], **method_groups(t)}
    series = pd.DataFrame({g: wide[cols].mean(axis=1) for g, cols in groups.items()})
    dates = pd.to_datetime(t[t.method == "dp"].sort_values("week_idx").eval_start_ts.values)
    return t, series, dates


def fig_paired(order=("forecast_optimal", "schedule", "rl_boot", "rl_ou")):
    """Mean weekly difference with its 95% interval: the registered statistic, zoomed in.

    Weekly differences scatter across roughly +/-$5k, so plotting them on the same axis buries
    an interval a few hundred dollars wide.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    for ax, (name, code) in zip(axes, MARKETS):
        t, s_, _ = load(code)
        groups = method_groups(t)
        rows = [(LABEL[m], paired_comparison(t, groups.get(m, [m]), code), COLOR[m]) for m in order]
        rows.append(("bootstrap − OU", paired_comparison(t, groups["rl_boot"], code, baseline=groups["rl_ou"]),
                     "#7f7f7f"))
        for i, (label, r, colour) in enumerate(rows):
            y = len(rows) - i
            ax.hlines(y, r["ci_low"] / 1000, r["ci_high"] / 1000, color=colour, lw=3, alpha=0.85)
            ax.plot(r["mean_delta"] / 1000, y, "D", color=colour, ms=6,
                    markeredgecolor="white", markeredgewidth=0.8, zorder=3)
            ax.text(1.02, y, f"{r['mean_delta']/1000:+.2f}k", va="center", fontsize=7.5, color="#333",
                    transform=ax.get_yaxis_transform())      # right-hand column, clear of the bars
        ax.axvline(0, color="black", lw=1)
        ax.axhline(1.5, color="#cccccc", lw=0.8, ls=":")          # separates the head-to-head row
        ax.set_yticks(range(len(rows), 0, -1), [r[0] for r in rows])
        ax.set_ylim(0.4, len(rows) + 0.6)
        ax.set_xlabel("mean weekly difference  ($000)")
        ax.set_title(f"{name} · {len(s_)} weeks", loc="left")
    axes[0].text(0.0, -0.42, "Diamond: mean paired weekly difference. Bar: 95% circular block-bootstrap CI. "
                 "Top four rows are against the DP;\nbottom row compares the two RL agents. "
                 "Price threshold omitted (−\$9.3k CISO, −\$7.5k NYISO).",
                 transform=axes[0].transAxes, fontsize=7, color="#555")
    fig.tight_layout(); fig.savefig(FIG / "paired_vs_dp.png", bbox_inches="tight"); plt.close(fig)


def fig_cumulative(order=("pf", "forecast_optimal", "dp", "schedule", "rl_boot", "rl_ou", "threshold")):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (name, code) in zip(axes, MARKETS):
        _, s, dates = load(code)
        for m in order:
            ax.plot(dates, s[m].cumsum() / 1e6, color=COLOR[m], lw=1.4,
                    label=LABEL[m] if code == "CISO" else None)
        ax.set_ylabel("cumulative score ($m)"); ax.set_title(f"{name} · {len(s)} weeks", loc="left")
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / "cumulative_pnl.png", bbox_inches="tight"); plt.close(fig)


def prior_draws(code, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    params = training_params(code)
    out = []
    for _ in range(n):
        if code == "CISO":
            theta, mu, sigma = sample_calib(rng)
            f, q = sample_f(rng, 336, params["q"]), params["q"]
        else:
            sc = draw_scenario(rng, NYIS_PRIOR, 336)
            theta, mu, sigma = sc["calib"]; f, q = sc["f"], sc["q"]
        out.append((np.log(2) / theta, sigma / np.sqrt(2 * theta) / q, f.std() / q))
    return np.array(out)


def fig_prior_vs_actual():
    names = ["half-life (hours)", "σ_stat / q", "std(f) / q"]
    fig, axes = plt.subplots(2, 3, figsize=(9.5, 5))
    for row, (name, code) in enumerate(MARKETS):
        draws = prior_draws(code)
        W = load_windows(ROOT / "results" / f"windows_{code}.npz")
        real = np.array([[np.log(2) / w["theta"],
                          w["sigma"] / np.sqrt(2 * w["theta"]) / w["q"],
                          w["f_eval"].std() / w["q"]] for w in W])
        for col in range(3):
            ax = axes[row, col]
            lo = min(draws[:, col].min(), real[:, col].min())
            hi = max(draws[:, col].max(), real[:, col].max())
            bins = np.logspace(np.log10(lo), np.log10(hi), 40)
            ax.hist(draws[:, col], bins=bins, density=True, color="#1f77b4", alpha=0.45, label="training prior")
            ax.hist(real[:, col], bins=bins, density=True, color="#d62728", alpha=0.45, label="walk-forward weeks")
            ax.set_xscale("log"); ax.set_yticks([])
            ticks = [t for t in ([2, 5, 10, 20, 50, 100], [0.2, 0.3, 0.5, 1, 2], [0.15, 0.2, 0.3, 0.5, 1])[col]
                     if lo <= t <= hi]
            ax.set_xticks(ticks, [f"{t:g}" for t in ticks])     # plain labels; log minor ticks collide
            ax.xaxis.set_minor_formatter(plt.NullFormatter())
            if row == 1:
                ax.set_xlabel(names[col])
            if col == 0:
                ax.set_ylabel(f"{name}\ndensity")
    axes[0, 2].legend(fontsize=7.5)
    fig.suptitle("Training priors against the calibrations actually met at evaluation", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "prior_vs_actual.png", bbox_inches="tight"); plt.close(fig)


def fig_training_curves():
    path = ROOT / "results" / "training_curves.csv"
    if not path.exists():
        print("  skipped training_curves.png (results/training_curves.csv not found)")
        return
    df = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    for run, g in df.groupby("run"):
        g = g.sort_values("step")
        ax.plot(g.step / 1e6, g.value, lw=1.5, label=run)
    ax.set_xlabel("training steps (millions)"); ax.set_ylabel("% of DP on OU evaluation")
    ax.set_title("Training under mismatched vs matched priors", loc="left")
    ax.legend(fontsize=8)
    note = df.source.iloc[0] if "source" in df else ""
    metric = df.metric.iloc[0] if "metric" in df else ""
    ax.text(0.0, -0.26, f"Metric: {metric}.\nSource: {note}.", transform=ax.transAxes,
            fontsize=6.5, color="#555")
    fig.tight_layout(); fig.savefig(FIG / "training_curves.png", bbox_inches="tight"); plt.close(fig)


def fig_residual_skill(horizons=(1, 2, 3, 4, 6, 8, 12, 18, 24)):
    """Why modelling the residual does not pay: the OU model's skill is in the wrong component.

    Top row: RMSE of the h-step-ahead forecast of the raw residual. Bottom row: the same for
    the residual with its centred 24-hour local level removed -- the only part arbitrage can
    monetise, since adding a constant to every price of the day changes no charge/discharge
    decision. The fitted OU beats the unconditional mean on the raw residual and loses to it
    on the level-removed part at every horizon the battery plans over.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.2), sharex=True)
    for col, (name, code) in enumerate(MARKETS):
        W = load_windows(ROOT / "results" / f"windows_{code}.npz")
        raw = {k: [] for k in ("OU model (fitted)", "hold current value", "unconditional mean")}
        lvl = {k: [] for k in raw}
        for h in horizons:
            acc = {k: ([], []) for k in raw}
            for w in W:
                X = w["X_eval_resid"][:168]
                ou = w["mu_eff"] + (X - w["mu_eff"]) * np.exp(-w["theta"] * h)
                dl = lambda v: v - np.convolve(np.pad(v, 12, mode="edge"), np.ones(25) / 25, "same")[12:-12]
                Xd, oud = dl(X), dl(ou)
                for k, (r_, l_) in zip(raw, [(ou[:-h], oud[:-h]), (X[:-h], Xd[:-h]),
                                             (np.zeros(len(X) - h), np.zeros(len(X) - h))]):
                    acc[k][0].append((X[h:] - r_) ** 2)
                    acc[k][1].append((Xd[h:] - l_) ** 2)
            for k in raw:
                raw[k].append(np.sqrt(np.concatenate(acc[k][0]).mean()))
                lvl[k].append(np.sqrt(np.concatenate(acc[k][1]).mean()))
        hl = np.mean([np.log(2) / w["theta"] for w in W])
        for row, (curves, title) in enumerate([(raw, "raw residual"),
                                               (lvl, "level removed — the part arbitrage can use")]):
            ax = axes[row, col]
            ax.axvspan(12, 18, color="#cccccc", alpha=0.35, lw=0)
            for (k, v), c, ls in zip(curves.items(), ("#1f77b4", "#ff7f0e", "#2ca02c"), ("-", "--", ":")):
                ax.plot(horizons, v, ls, color=c, lw=1.6, marker="o", ms=3,
                        label=k if (code == "CISO" and row == 0) else None)
            ax.set_title(f"{name} · {title}" + (f" · fitted half-life {hl:.0f} h" if row == 0 else ""),
                         loc="left", fontsize=8.5)
            ax.set_ylabel("RMSE ($/MWh)")
            if row == 1:
                ax.set_xlabel("forecast horizon (hours ahead)")
    axes[0, 0].legend(loc="lower right", fontsize=7.5)
    axes[1, 0].text(0.0, -0.36, "Shaded: the 12-18 hour trough-to-peak horizon a 4-hour battery commits "
                    "over. Top: the OU looks skilful, but it is\npredicting the slow price level. Bottom: on "
                    "deviations from that level it is worse than assuming the residual away.",
                    transform=axes[1, 0].transAxes, fontsize=7, color="#555")
    fig.tight_layout(); fig.savefig(FIG / "residual_forecast_skill.png", bbox_inches="tight"); plt.close(fig)


def fig_value_ceiling(order=("pf", "forecast_optimal", "dp", "schedule")):
    """How much of the achievable value needs the residual at all.

    The forecast-only optimal is the best any residual-blind policy can do, so the gap above
    it is all the money the residual is worth, and the gap between it and the DP is what
    modelling the residual actually earned.
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, (name, code) in zip(axes, MARKETS):
        _, s, _ = load(code)
        means = [s[m].mean() for m in order]
        pf = means[0]
        y = np.arange(len(order))[::-1]
        ax.barh(y, means, color=[COLOR[m] for m in order], height=0.6, alpha=0.9)
        for yi, v in zip(y, means):
            ax.text(v + pf * 0.015, yi, f"${v:,.0f}   {v / pf:.1%}", va="center", fontsize=7.5, color="#333")
        ax.axvline(means[1], color="#17becf", lw=1, ls="--")
        ax.set_yticks(y, [LABEL[m] for m in order])
        ax.set_xlim(0, pf * 1.32)
        ax.set_xlabel("mean weekly score ($)")
        ax.set_title(f"{name} · {len(s)} weeks", loc="left")
    axes[0].text(0.0, -0.34, "Dashed line: the ceiling for any policy that ignores the price residual. "
                 "The residual is worth 18% (CISO) and 29%\n(NYISO) of perfect foresight; the DP "
                 "captures none of it, landing below the ceiling in both markets.",
                 transform=axes[0].transAxes, fontsize=7, color="#555")
    fig.tight_layout(); fig.savefig(FIG / "value_ceiling.png", bbox_inches="tight"); plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    fig_paired(); fig_cumulative(); fig_prior_vs_actual(); fig_training_curves()
    fig_residual_skill(); fig_value_ceiling()
    for p in sorted(FIG.glob("*.png")):
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
