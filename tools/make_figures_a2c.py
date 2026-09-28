#!/usr/bin/env python3
"""Build the A2C report figures (and the eval table) from the sweep's metric mirrors.

Reads ``sweep_results_a2c/<tag>__seed<N>/metrics.jsonl`` and writes one PNG per
question into ``report/figs_a2c/`` plus ``report/summary_a2c.json``.

Same visual language as the DQN report (``tools/make_figures.py``): ordinal
one-hue ramp so the reader sees the ordering of the swept hyperparameter in the
colour, thick line = mean over seeds, translucent band = envelope over seeds.

Two A2C-specific departures:

- ``losses/policy_loss`` is drawn **raw and for one seed only**. The assignment
  asks what the *noise* of that chart says about the gradient being followed, so
  smoothing it, or averaging two seeds, would destroy the evidence.
- ``losses/explained_variance`` is unbounded below (it reaches -1e4 with a
  5-transition batch), so its panel is clipped; the text reports the medians.
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
RESULTS = REPO_ROOT / "sweep_results_a2c"
OUT = REPO_ROOT / "report" / "figs_a2c"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4e0"
# Ordinal blue ramps, same validated set the DQN report used.
RAMPS = {
    2: ["#86b6ef", "#0d366b"],
    3: ["#86b6ef", "#2a78d6", "#0d366b"],
    4: ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
}

RETURN = "charts/episodic_return_mean_last100"
POLICY_LOSS = "losses/policy_loss"
VALUE_LOSS = "losses/value_loss"
EXPL_VAR = "losses/explained_variance"
ENTROPY = "losses/entropy"
ADV_STD = "charts/advantage_std"
ADV_MEAN = "charts/advantage_mean"
SPS = "charts/SPS"

LN2 = math.log(2.0)

# EMA alpha per metric; 1.0 = no smoothing.
SMOOTH = {
    RETURN: 0.10,
    POLICY_LOSS: 1.0,
    VALUE_LOSS: 0.06,
    EXPL_VAR: 0.06,
    ENTROPY: 0.12,
    ADV_STD: 0.08,
    ADV_MEAN: 0.08,
    SPS: 1.0,
}
GRID_N = 400

# SPS measured one run at a time (tools/bench_sps_a2c.py), because the sweep
# runs 6 jobs at once and its charts/SPS is contaminated by CPU contention.
BENCH_PATH = REPO_ROOT / "report" / "sps_bench_a2c.json"
BENCH_SPS: dict[str, float] = (
    json.loads(BENCH_PATH.read_text()) if BENCH_PATH.exists() else {}
)


# Reference figure: the DQN report's 3-panel row at width 100%. Font sizes are
# scaled by (fig_width / REF_W) / place, so a 2-, 3- or 4-panel figure, placed
# at any width in the Typst source, renders its labels at the same point size
# on the page.
REF_W = 9.4


def panel(metric, title, yscale="linear", ylim=None, mode="mean", hline=None):
    return {"metric": metric, "title": title, "yscale": yscale, "ylim": ylim,
            "mode": mode, "hline": hline}


def rolling_std(y: np.ndarray, window: int = 25) -> np.ndarray:
    """Std of ``y`` over a trailing window — the noise of the chart, as a curve."""
    if y.size < 2:
        return y
    out = np.empty_like(y)
    for i in range(y.size):
        out[i] = np.std(y[max(0, i - window + 1): i + 1])
    return np.maximum(out, 1e-3)


# question -> ordered series [(tag, legend label, linestyle)] + panels
QUESTIONS = {
    "q1": {
        "series": [
            ("q1_envs1", "num_envs = 1", "-"),
            ("q1_envs1_steps40", "1, num_steps=40 (batch do baseline)", "--"),
            ("base", "8 (baseline)", "-"),
            ("q1_envs32", "32", "-"),
        ],
        "ncols": 4,
        "panels": [
            panel(RETURN, "Retorno (últ. 100 ep.)"),
            panel(POLICY_LOSS, "policy_loss bruto (seed 1)", ylim=(-25, 15), mode="raw_seed0"),
            panel(POLICY_LOSS, "ruído: desv.-padrão móvel", yscale="log", mode="rollstd"),
            panel(SPS, "SPS (run isolado)", mode="bench"),
        ],
    },
    "q2": {
        "series": [
            ("q2_steps1", "num_steps = 1", "-"),
            ("base", "5 (baseline)", "-"),
            ("q2_steps32", "32", "-"),
            ("q2_steps128", "128", "-"),
        ],
        "ncols": 3,
        "panels": [
            panel(RETURN, "Retorno (últ. 100 ep.)"),
            panel(VALUE_LOSS, "value_loss (symlog)", yscale="symlog"),
            panel(EXPL_VAR, "explained_variance (eixo cortado em −1)", ylim=(-1.05, 1.05), hline=0.0),
        ],
    },
    "q3": {
        "series": [
            ("q3_ent0", "ent_coef = 0", "-"),
            ("base", "0.01 (baseline)", "-"),
            ("q3_ent0.1", "0.1", "-"),
        ],
        "ncols": 2,
        "place": 0.78,
        "panels": [
            panel(RETURN, "Retorno episódico (média dos últimos 100)"),
            panel(ENTROPY, "entropy (ln 2 ≈ 0,69 = uniforme)", hline=LN2),
        ],
    },
    "q4": {
        "series": [
            ("base", "use_baseline = true (baseline)", "-"),
            ("q4_nobaseline", "false — peso = retorno R", "-"),
        ],
        "ncols": 4,
        "panels": [
            panel(RETURN, "Retorno (últ. 100 ep.)"),
            panel(ENTROPY, "entropy (ln 2 = uniforme)", hline=LN2),
            panel(ADV_MEAN, "advantage_mean (peso médio)", hline=0.0),
            panel(ADV_STD, "advantage_std"),
        ],
    },
    "lunar": {
        "series": [
            ("ll_cartpole_hp", "HP do CartPole (num_steps=5)", "-"),
            ("ll_tuned", "ajustado: num_steps=128", "-"),
            ("ll_lr3e-4", "ajustado + lr=3e-4", "-"),
        ],
        "ncols": 3,
        "panels": [
            panel(RETURN, "Retorno (últ. 100 ep.) — resolvido = 200", hline=200.0),
            panel(VALUE_LOSS, "value_loss (symlog)", yscale="symlog"),
            panel(EXPL_VAR, "explained_variance", ylim=(-1.05, 1.05), hline=0.0),
        ],
    },
}


def load_runs() -> dict[str, list[list[dict]]]:
    """tag -> [rows-of-one-seed, ...], seeds in ascending order."""
    runs: dict[str, list[tuple[int, list[dict]]]] = {}
    for meta_path in sorted(RESULTS.glob("*/meta.json")):
        run_dir = meta_path.parent
        tag, _, seed_part = run_dir.name.rpartition("__seed")
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
        ys = ema(y, SMOOTH[metric])
        curves.append(np.interp(grid, x, ys, left=np.nan, right=ys[-1]))
    return curves


def style_axes(ax, spec, k=1.0):
    ax.set_title(spec["title"], fontsize=8 * k, color=INK, pad=6 * k, loc="left")
    if spec["yscale"] == "symlog":
        ax.set_yscale("symlog", linthresh=1.0)
    else:
        ax.set_yscale(spec["yscale"])
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6 * k, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(labelsize=7 * k, colors=INK_2, length=3 * k, width=0.6 * k)
    ax.set_xlabel("passos de ambiente", fontsize=7 * k, color=INK_2)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
    if spec["hline"] is not None:
        ax.axhline(spec["hline"], color=INK_2, linewidth=0.6 * k, linestyle=":", zorder=1)


def build(qid, runs, total_steps):
    spec = QUESTIONS[qid]
    panels = spec["panels"]
    ncols = spec["ncols"]
    nrows = math.ceil(len(panels) / ncols)
    grid = np.linspace(0, total_steps, GRID_N)
    fig_w = 3.0 * ncols + 0.4
    k = (fig_w / REF_W) / spec.get("place", 1.0)
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(fig_w, 2.45 * nrows),
        facecolor=SURFACE,
        squeeze=False,
    )
    flat = axes.ravel()
    for ax in flat[len(panels):]:
        ax.set_visible(False)

    ramp = RAMPS[len(spec["series"])]
    handles, labels = [], []
    for ax, pspec in zip(flat, panels):
        style_axes(ax, pspec, k)
        metric = pspec["metric"]
        for (tag, label, ls), color in zip(spec["series"], ramp):
            rows_per_seed = runs.get(tag, [])
            if not rows_per_seed:
                continue
            if pspec["mode"] == "bench":
                # A per-configuration constant measured outside the parallel
                # sweep: a bar, not a curve over an env-steps axis it does not have.
                v = BENCH_SPS.get(tag)
                if v is None:
                    continue
                i = [t for t, _, _ in spec["series"]].index(tag)
                bar = ax.bar(i, v, color=color, width=0.68, zorder=3, label=label)
                ax.annotate(f"{v:,.0f}".replace(",", " "), (i, v), textcoords="offset points",
                            xytext=(0, 2 * k), ha="center", fontsize=6.5 * k, color=INK_2)
                line = bar
            elif pspec["mode"] == "raw_seed0":
                x, y = series(rows_per_seed[0], metric)
                if x.size < 2:
                    continue
                (line,) = ax.plot(x, y, color=color, linewidth=0.55 * k, linestyle=ls,
                                  alpha=0.9, zorder=3, label=label)
            elif pspec["mode"] == "rollstd":
                stack = []
                for rows in rows_per_seed:
                    x, y = series(rows, metric)
                    if x.size < 2:
                        continue
                    rs = ema(rolling_std(y), 0.05)
                    stack.append(np.interp(grid, x, rs, left=np.nan, right=rs[-1]))
                if not stack:
                    continue
                (line,) = ax.plot(grid, np.nanmean(np.vstack(stack), axis=0), color=color,
                                  linewidth=1.8 * k, linestyle=ls, zorder=3, label=label)
            else:
                curves = resample(rows_per_seed, metric, grid)
                if not curves:
                    continue
                stack = np.vstack(curves)
                if stack.shape[0] > 1:
                    ax.fill_between(grid, np.nanmin(stack, axis=0), np.nanmax(stack, axis=0),
                                    color=color, alpha=0.18, linewidth=0, zorder=2)
                (line,) = ax.plot(grid, np.nanmean(stack, axis=0), color=color,
                                  linewidth=1.8 * k, linestyle=ls, zorder=3, label=label)
            if ax is flat[0]:
                handles.append(line)
                labels.append(label)
        if pspec["mode"] == "bench":
            ax.set_xlabel("")
            ax.set_xticks(range(len(spec["series"])))
            ax.set_xticklabels([t.replace("q1_envs", "").replace("base", "8")
                                .replace("1_steps40", "1×40") for t, _, _ in spec["series"]],
                               fontsize=7 * k, color=INK_2)
            ax.set_xlim(-0.6, len(spec["series"]) - 0.4)
            ax.grid(axis="x", visible=False)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v / 1000:.0f}k")
        if pspec["ylim"]:
            ax.set_ylim(*pspec["ylim"])

    fig.legend(
        handles, labels, loc="lower center", ncol=len(labels), frameon=False,
        fontsize=7.5 * k, labelcolor=INK, bbox_to_anchor=(0.5, -0.02), handlelength=1.8,
    )
    fig.tight_layout(rect=(0, 0.10 / nrows, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{qid}.png"
    fig.savefig(path, dpi=220, facecolor=SURFACE)
    plt.close(fig)
    return path


def _tail(rows, metric, n=20):
    y = series(rows, metric)[1]
    return float(np.mean(y[-n:])) if y.size else None


def _median(rows, metric):
    y = series(rows, metric)[1]
    return float(np.median(y)) if y.size else None


def summarize(runs):
    out = {}
    for tag, seeds in sorted(runs.items()):
        per_seed = []
        for rows in seeds:
            ev = [r for r in rows if "eval/mean_return" in r]
            pl = series(rows, POLICY_LOSS)[1]
            per_seed.append(
                {
                    "eval_mean": ev[-1]["eval/mean_return"] if ev else None,
                    "eval_std": ev[-1].get("eval/std_return") if ev else None,
                    "final_train_return": _tail(rows, RETURN),
                    "final_entropy": _tail(rows, ENTROPY),
                    "median_explained_variance": _median(rows, EXPL_VAR),
                    "median_value_loss": _median(rows, VALUE_LOSS),
                    "final_advantage_mean": _tail(rows, ADV_MEAN),
                    "final_advantage_std": _tail(rows, ADV_STD),
                    "policy_loss_std": float(np.std(pl)) if pl.size else None,
                    "final_sps": _tail(rows, SPS, n=1),
                }
            )
        evals = [s["eval_mean"] for s in per_seed if s["eval_mean"] is not None]
        out[tag] = {
            "seeds": per_seed,
            "eval_mean_over_seeds": float(np.mean(evals)) if evals else None,
        }
    return out


def horizon(runs, tags) -> int:
    """Largest global_step among *these* runs.

    Per question, not global: the CartPole sweep runs to 500k and the
    LunarLander extra to 5M, so one shared x-grid would squeeze every CartPole
    curve into the leftmost tenth of its panel.
    """
    total = 0
    for tag in tags:
        for rows in runs.get(tag, []):
            steps = [r["global_step"] for r in rows if "global_step" in r]
            total = max(total, max(steps) if steps else 0)
    return total


def main():
    runs = load_runs()
    print(f"{sum(len(v) for v in runs.values())} runs, {len(runs)} configs")
    for qid in QUESTIONS:
        tags = [t for t, _, _ in QUESTIONS[qid]["series"]]
        missing = [t for t in tags if t not in runs]
        if missing:
            print(f"  {qid}: SKIP (missing {missing})")
            continue
        print(f"  {qid}: {build(qid, runs, horizon(runs, tags))}")
    summary = summarize(runs)
    (REPO_ROOT / "report").mkdir(exist_ok=True)
    (REPO_ROOT / "report" / "summary_a2c.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'config':18} {'eval (seeds)':>14} {'ret':>10} {'ent':>12} {'expvar med':>14} {'pl std':>8} {'SPS':>7}")
    for tag, s in summary.items():
        j = lambda k, f: "/".join(f.format(x[k]) if x[k] is not None else "-" for x in s["seeds"])
        print(f"{tag:18} {j('eval_mean','{:.0f}'):>14} {j('final_train_return','{:.0f}'):>10} "
              f"{j('final_entropy','{:.3f}'):>12} {j('median_explained_variance','{:.2f}'):>14} "
              f"{j('policy_loss_std','{:.1f}'):>8} {j('final_sps','{:.0f}'):>7}")


if __name__ == "__main__":
    main()
