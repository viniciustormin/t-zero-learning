#!/usr/bin/env python3
"""Build the SAC report figures from the metric mirrors in sweep_results_sac/.

Writes ``report/figs_sac/<fig>.png``. Same visual language as
``tools/make_figures_ppo.py`` (surface, ink, grid, categorical hues in fixed
order, font scaling by placement width). Thick line = mean over seeds, band =
envelope; ``per_seed`` figures draw each seed as its own line instead
(solid / dashed / dotted = seed 1 / 2 / 3). Numbers live in the report's
tables, which come from ``tools/summarize_sac.py``.

    python tools/make_figures_sac.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "sweep_results_sac"
OUT = REPO_ROOT / "report" / "figs_sac"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4e0"
# categorical slots 1..5 (blue, orange, aqua, yellow, magenta), fixed order
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
DASH = {1: "-", 2: "--", 3: ":"}

RETURN = "charts/episodic_return_mean_last100"
EP_LEN = "charts/episodic_length_mean_last100"
ALPHA = "losses/alpha"
ENTROPY = "losses/entropy"
Q1 = "losses/qf1_values"
QMR = "charts/q_minus_return"
UPDATES = "charts/num_updates"
SPS = "charts/SPS"

# Part 3: no candidate beat the seed spread, so the final config is the initial
# one (configs/sac_hopper_tuned.yml == sac_hopper): its runs are base/1-3.
# r1_ups4 came closest (see tuning_log_sac.md).
FINAL = "base"
FINAL_LABEL = "SAC final (= base)"
CLOSEST = "r1_ups4"

SMOOTH = {RETURN: 1.0, EP_LEN: 1.0, ALPHA: 0.1, ENTROPY: 0.05, Q1: 0.2, QMR: 0.15, UPDATES: 1.0, SPS: 1.0}
GRID_N = 400
REF_W = 9.4


def panel(metric, title, yscale="linear", ylim=None, hline=None, x="global_step"):
    return {"metric": metric, "title": title, "yscale": yscale, "ylim": ylim, "hline": hline, "x": x}


FIGS = {
    # Part 3: the final config is the initial one, so it goes next to the
    # candidate that came closest
    "part3": {
        "per_seed": True,
        "seeds": (1, 2, 3),
        "series": [(FINAL, "base (inicial = final)"), (CLOSEST, "updates_per_step 4")],
        "legend_ncol": 3,
        "ncols": 2,
        "place": 0.62,
        "panels": [
            panel(RETURN, "Retorno de treino (últ. 100 ep.)"),
            panel(EP_LEN, "Duração do episódio (últ. 100 ep.)"),
        ],
    },
    # Part 4.1: fixed temperature
    "abl_alpha": {
        "seeds": (1, 2),
        "series": [(FINAL, f"{FINAL_LABEL} (autotune)"), ("abl_alpha", "alpha fixo = 0,09")],
        "ncols": 3,
        "panels": [
            panel(ALPHA, "alpha (log)", yscale="log"),
            panel(ENTROPY, "entropia (pontilhado: alvo −3)", hline=-3.0),
            panel(RETURN, "Retorno (últ. 100 ep.)"),
        ],
    },
    # Part 4.2: single critic
    "abl_q": {
        "seeds": (1, 2),
        "series": [(FINAL, f"{FINAL_LABEL} (min Q1, Q2)"), ("abl_1q", "um crítico (só Q1)")],
        "ncols": 3,
        "panels": [
            panel(QMR, "q_minus_return (Q − G)", hline=0.0),
            panel(Q1, "qf1_values"),
            panel(RETURN, "Retorno (últ. 100 ep.)"),
        ],
    },
    # Part 5: SAC vs PPO
    "sac_ppo": {
        "seeds": (1, 2),
        "series": [(FINAL, FINAL_LABEL), (CLOSEST, "SAC updates_per_step 4"), ("ppo_hopper", "PPO")],
        "ncols": 4,
        "panels": [
            panel(RETURN, "Retorno × passos (seeds 1–2)"),
            panel(RETURN, "Retorno × tempo (seed 1, isolado; log)", x="_wall"),
            panel(SPS, "charts/SPS (seed 1, isolado; log)", yscale="log", x="_wall_steps"),
            panel(UPDATES, "passos de gradiente (log)", yscale="log"),
        ],
        # wall-clock panels read these isolated, one-at-a-time runs instead
        "wall_tags": {FINAL: "bench_sac_final", CLOSEST: "bench_sac_ups4", "ppo_hopper": "bench_ppo"},
    },
}


def load_runs() -> dict[str, dict[int, list[dict]]]:
    runs: dict[str, dict[int, list[dict]]] = {}
    for path in sorted(RESULTS.glob("*__seed*/metrics.jsonl")):
        tag, _, seed = path.parent.name.rpartition("__seed")
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if rows:
            runs.setdefault(tag, {})[int(seed)] = rows
    return runs


def series(rows, metric, x="global_step"):
    pts = [(r[x], r[metric]) for r in rows if metric in r and x in r and not any(k.startswith("eval/") for k in r)]
    pts = [(a, b) for a, b in pts if b is not None and math.isfinite(b)]
    if not pts:
        return np.array([]), np.array([])
    xs, ys = zip(*sorted(pts))
    return np.asarray(xs, float), np.asarray(ys, float)


def ema(y, alpha):
    if alpha >= 1.0 or y.size == 0:
        return y
    out, acc = np.empty_like(y), y[0]
    for i, v in enumerate(y):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def style(ax, spec, k):
    ax.set_title(spec["title"], fontsize=8 * k, color=INK, pad=6 * k, loc="left")
    ax.set_yscale(spec["yscale"])
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6 * k, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=7 * k, colors=INK_2, length=3 * k, width=0.6 * k)
    if spec["yscale"] == "log":
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}")
        ax.yaxis.set_minor_formatter(lambda v, _: "")
    if spec["x"] == "_wall":
        ax.set_xscale("log")
        ax.set_xlabel("tempo de relógio (s, 1 núcleo, log)", fontsize=7 * k, color=INK_2)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:g}")
        ax.xaxis.set_minor_formatter(lambda v, _: "")
    else:
        ax.set_xlabel("passos de ambiente", fontsize=7 * k, color=INK_2)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1e3:.0f}k" if v else "0")
    if spec["hline"] is not None:
        ax.axhline(spec["hline"], color=INK_2, linewidth=0.7 * k, linestyle=":", zorder=1)
    if spec["ylim"]:
        ax.set_ylim(*spec["ylim"])


def build(fid, runs):
    spec = FIGS[fid]
    panels, ncols = spec["panels"], spec["ncols"]
    nrows = math.ceil(len(panels) / ncols)
    fig_w = 3.0 * ncols + 0.4
    k = (fig_w / REF_W) / spec.get("place", 1.0)
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, 2.45 * nrows), facecolor=SURFACE, squeeze=False)
    flat = axes.ravel()
    handles, labels = [], []
    for ax, ps in zip(flat, panels):
        style(ax, ps, k)
        metric, xkey = ps["metric"], ps["x"]
        # _wall_steps: the isolated runs, plotted against env steps
        isolated = xkey in ("_wall", "_wall_steps")
        xkey = "global_step" if xkey == "_wall_steps" else xkey
        for i, (tag, label) in enumerate(spec["series"]):
            color = CAT[i]
            if isolated:
                src = spec.get("wall_tags", {}).get(tag)
                seeds = runs.get(src, {}) if src else {}
            else:
                seeds = {s: r for s, r in runs.get(tag, {}).items() if s in spec["seeds"]}
            if not seeds:
                continue
            curves = [(s, *series(rows, metric, xkey)) for s, rows in sorted(seeds.items())]
            curves = [(s, x, ema(y, SMOOTH[metric])) for s, x, y in curves if x.size > 1]
            if not curves:
                continue
            if spec.get("per_seed"):
                for s, x, y in curves:
                    (line,) = ax.plot(x, y, color=color, linewidth=1.4 * k, linestyle=DASH[s], zorder=3)
                    if ax is flat[0] and s == min(spec["seeds"]):
                        handles.append(line)
                        labels.append(label)
                continue
            hi = max(x[-1] for _, x, _ in curves)
            grid = np.linspace(min(x[0] for _, x, _ in curves), hi, GRID_N)
            stack = np.vstack([np.interp(grid, x, y, left=np.nan, right=np.nan) for _, x, y in curves])
            if stack.shape[0] > 1:
                ax.fill_between(grid, np.nanmin(stack, 0), np.nanmax(stack, 0), color=color,
                                alpha=0.16, linewidth=0, zorder=2)
            with np.errstate(all="ignore"):
                mean = np.nanmean(stack, 0)
            (line,) = ax.plot(grid, mean, color=color, linewidth=1.8 * k, zorder=3)
            if ax is flat[0]:
                handles.append(line)
                labels.append(label)
    if spec.get("per_seed"):
        for s in spec["seeds"]:
            handles.append(Line2D([], [], color=INK_2, lw=1.4 * k, ls=DASH[s]))
            labels.append(f"seed {s}")
    ncol = spec.get("legend_ncol", len(labels))
    legend_rows = math.ceil(len(labels) / ncol)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, fontsize=7.5 * k,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.02), handlelength=1.8)
    fig.tight_layout(rect=(0, 0.11 * legend_rows / nrows, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{fid}.png"
    fig.savefig(path, dpi=220, facecolor=SURFACE)
    plt.close(fig)
    return path


def main():
    runs = load_runs()
    for fid, spec in FIGS.items():
        tags = [t for t, _ in spec["series"]] + list(spec.get("wall_tags", {}).values())
        missing = [t for t in tags if t not in runs]
        print(f"{fid}: {build(fid, runs).relative_to(REPO_ROOT)}" + (f"  (missing {missing})" if missing else ""))


if __name__ == "__main__":
    main()
