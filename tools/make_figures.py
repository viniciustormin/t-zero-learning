#!/usr/bin/env python3
"""Build the report figures (and the eval table) from the sweep's metric mirrors.

Reads ``sweep_results/<tag>__seed<N>/metrics.jsonl`` and writes one PNG per
question into ``report/figs/`` plus ``report/summary.json``.

Design notes: the three series of every sweep are *ordered* values of one
hyperparameter, so they take an ordinal one-hue ramp (light = smallest) rather
than three unrelated hues — the reader sees the ordering in the color. Thick
line = mean over seeds, thin translucent lines = the individual seeds.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "sweep_results"
OUT = REPO_ROOT / "report" / "figs"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4e0"
# Ordinal blue ramp, validated light-mode (steps 250 / 450 / 700).
RAMP = ["#86b6ef", "#2a78d6", "#0d366b"]

RETURN = "charts/episodic_return_mean_last100"
TD_LOSS = "losses/td_loss"
Q_VALUES = "losses/q_values"
EPSILON = "charts/epsilon"

# question -> (ordered [tag, legend label], [(metric, panel title, y-scale)])
QUESTIONS = {
    "q1": {
        "series": [
            ("q1_tnf1", "target_network_frequency = 1"),
            ("base", "500 (baseline)"),
            ("q1_tnf5000", "5000"),
        ],
        "panels": [
            (RETURN, "Retorno episódico (média dos últimos 100)", "linear"),
            (TD_LOSS, "TD loss", "log"),
            (Q_VALUES, "Q médio predito", "linear"),
        ],
    },
    "q2": {
        "series": [
            ("q2_buf500", "buffer_size = 500"),
            ("base", "10 000 (baseline)"),
            ("q2_buf500000", "500 000"),
        ],
        "panels": [
            (RETURN, "Retorno episódico (média dos últimos 100)", "linear"),
            (Q_VALUES, "Q médio predito", "linear"),
        ],
    },
    "q3": {
        "series": [
            ("q3_ef0.05", "exploration_fraction = 0.05"),
            ("base", "0.5 (baseline)"),
            ("q3_ef0.9", "0.9"),
        ],
        "panels": [
            (EPSILON, "Epsilon efetivo", "linear"),
            (RETURN, "Retorno episódico (média dos últimos 100)", "linear"),
            (Q_VALUES, "Q médio predito", "linear"),
        ],
    },
}

SMOOTH = {RETURN: 0.10, TD_LOSS: 0.04, Q_VALUES: 0.10, EPSILON: 1.0}
GRID_N = 400


def load_runs() -> dict[str, list[dict]]:
    """tag -> [rows-of-one-seed, ...], seeds in ascending order."""
    runs: dict[str, list[tuple[int, list[dict]]]] = {}
    for meta_path in sorted(RESULTS.glob("*/meta.json")):
        run_dir = meta_path.parent
        name = run_dir.name
        tag, _, seed_part = name.rpartition("__seed")
        rows = [
            json.loads(line)
            for line in (run_dir / "metrics.jsonl").read_text().splitlines()
            if line.strip()
        ]
        runs.setdefault(tag, []).append((int(seed_part), rows))
    return {tag: [rows for _, rows in sorted(v)] for tag, v in runs.items()}


def series(rows: list[dict], metric: str) -> tuple[np.ndarray, np.ndarray]:
    pts = [(r["global_step"], r[metric]) for r in rows if metric in r and "global_step" in r]
    pts = [(s, v) for s, v in pts if v is not None and math.isfinite(v)]
    if not pts:
        return np.array([]), np.array([])
    steps, vals = zip(*sorted(pts))
    return np.asarray(steps, dtype=float), np.asarray(vals, dtype=float)


def ema(y: np.ndarray, alpha: float) -> np.ndarray:
    if alpha >= 1.0 or y.size == 0:
        return y
    out = np.empty_like(y)
    acc = y[0]
    for i, v in enumerate(y):
        acc = alpha * v + (1 - alpha) * acc
        out[i] = acc
    return out


def resample(rows_per_seed, metric, grid):
    curves = []
    for rows in rows_per_seed:
        x, y = series(rows, metric)
        if x.size < 2:
            continue
        curves.append(ema(np.interp(grid, x, y, left=np.nan, right=y[-1]), SMOOTH[metric]))
    return curves


def style_axes(ax, title, yscale):
    ax.set_title(title, fontsize=8, color=INK, pad=6, loc="left")
    ax.set_yscale(yscale)
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=7, colors=INK_2, length=3, width=0.6)
    ax.set_xlabel("passos de ambiente", fontsize=7, color=INK_2)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")


def build(qid, runs, total_steps):
    spec = QUESTIONS[qid]
    panels = spec["panels"]
    grid = np.linspace(0, total_steps, GRID_N)
    fig, axes = plt.subplots(
        1, len(panels), figsize=(3.0 * len(panels) + 0.4, 2.45), facecolor=SURFACE
    )
    axes = np.atleast_1d(axes)

    handles, labels = [], []
    for ax, (metric, title, yscale) in zip(axes, panels):
        style_axes(ax, title, yscale)
        for (tag, label), color in zip(spec["series"], RAMP):
            curves = resample(runs.get(tag, []), metric, grid)
            if not curves:
                continue
            for c in curves:  # individual seeds, recessive
                ax.plot(grid, c, color=color, linewidth=0.7, alpha=0.30, zorder=2)
            mean = np.nanmean(np.vstack(curves), axis=0)
            (line,) = ax.plot(grid, mean, color=color, linewidth=1.8, zorder=3, label=label)
            if ax is axes[0]:
                handles.append(line)
                labels.append(label)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        fontsize=7.5,
        labelcolor=INK,
        bbox_to_anchor=(0.5, -0.02),
        handlelength=1.6,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{qid}.png"
    fig.savefig(path, dpi=220, facecolor=SURFACE)
    plt.close(fig)
    return path


def summarize(runs):
    out = {}
    for tag, seeds in sorted(runs.items()):
        per_seed = []
        for rows in seeds:
            ev = [r for r in rows if "eval/mean_return" in r]
            ret = series(rows, RETURN)[1]
            per_seed.append(
                {
                    "eval_mean": ev[-1]["eval/mean_return"] if ev else None,
                    "eval_std": ev[-1].get("eval/std_return") if ev else None,
                    "final_train_return": float(ret[-1]) if ret.size else None,
                    "final_q": float(series(rows, Q_VALUES)[1][-1]) if series(rows, Q_VALUES)[1].size else None,
                }
            )
        evals = [s["eval_mean"] for s in per_seed if s["eval_mean"] is not None]
        out[tag] = {
            "seeds": per_seed,
            "eval_mean_over_seeds": float(np.mean(evals)) if evals else None,
        }
    return out


def main():
    runs = load_runs()
    total = 0
    for seeds in runs.values():
        for rows in seeds:
            steps = [r["global_step"] for r in rows if "global_step" in r]
            total = max(total, max(steps) if steps else 0)
    print(f"{sum(len(v) for v in runs.values())} runs, {len(runs)} configs, {total} steps")
    for qid in QUESTIONS:
        missing = [t for t, _ in QUESTIONS[qid]["series"] if t not in runs]
        if missing:
            print(f"  {qid}: SKIP (missing {missing})")
            continue
        print(f"  {qid}: {build(qid, runs, total)}")
    summary = summarize(runs)
    (REPO_ROOT / "report").mkdir(exist_ok=True)
    (REPO_ROOT / "report" / "summary.json").write_text(json.dumps(summary, indent=2))
    for tag, s in summary.items():
        evs = [f"{x['eval_mean']:.0f}" if x["eval_mean"] is not None else "-" for x in s["seeds"]]
        print(f"    {tag:16} eval per seed: {evs}")


if __name__ == "__main__":
    main()
