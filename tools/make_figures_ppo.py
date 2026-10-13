#!/usr/bin/env python3
"""Build the PPO report figures from the metric mirrors in sweep_results_ppo/.

Writes ``report/figs_ppo/<fig>.png`` and ``report/summary_ppo.json``. Same
visual language as ``tools/make_figures_a2c.py`` (surface, ink, grid, font
scaling by placement width), but the series here are *arms*, not points of an
ordered sweep, so they get categorical hues in fixed order (validated with the
dataviz palette validator; legend always present, numbers also in the report's
tables). Thick line = mean over seeds, band = envelope; ``per_seed`` figures
draw each seed as its own line instead (solid = seed 1, dashed = seed 2).

    python tools/make_figures_ppo.py
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
RESULTS = REPO_ROOT / "sweep_results_ppo"
OUT = REPO_ROOT / "report" / "figs_ppo"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e5e4e0"
# categorical slots 1..5 (blue, orange, aqua, yellow, magenta), fixed order
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

RETURN = "charts/episodic_return_mean_last100"
KL = "losses/approx_kl"
CLIPFRAC = "losses/clipfrac"
ENTROPY = "losses/entropy"
EXPL_VAR = "losses/explained_variance"
GRAD_NORM = "losses/grad_norm"
VALUE_LOSS = "losses/value_loss"
UPDATES = "charts/num_updates"
SPS = "charts/SPS"

# num_envs * num_steps of the A2C arms (configs/a2c_lunarlander.yml: 16 x 128;
# configs/a2c_cartpole.yml: 8 x 5)
A2C_BATCH = {"a2c_ll": 2048, "bench_a2c": 2048, "a2c_cp": 40, "bench_a2c_cp": 40}

SMOOTH = {RETURN: 0.15, KL: 0.08, CLIPFRAC: 0.08, ENTROPY: 0.1, EXPL_VAR: 0.06,
          GRAD_NORM: 0.06, VALUE_LOSS: 0.06, UPDATES: 1.0, SPS: 1.0}
GRID_N = 400
REF_W = 9.4


def panel(metric, title, yscale="linear", ylim=None, hline=None, x="global_step"):
    return {"metric": metric, "title": title, "yscale": yscale, "ylim": ylim, "hline": hline, "x": x}


FIGS = {
    # Part 4, round 0 -> 1: the base's diagnostics next to the one-change arms
    "diag": {
        "series": [
            ("base", "base"),
            ("r1_gn5", "max_grad_norm 5"),
            ("r1_vf0.1", "vf_coef 0.1"),
            ("r1_ep10", "update_epochs 10"),
            ("r1_lr1e-3", "learning_rate 1e-3"),
        ],
        "ncols": 4,
        "panels": [
            panel(RETURN, "Retorno (últ. 100 ep.)", hline=200.0),
            panel(KL, "approx_kl (log)", yscale="log"),
            panel(CLIPFRAC, "clipfrac"),
            panel(GRAD_NORM, "grad_norm antes do corte (log)", yscale="log", hline=0.5),
        ],
    },
    # Part 4, rounds 2 -> 4: the path to the final config, at the 2M budget
    "path": {
        "series": [
            ("r2_lr_ep10", "lr 1e-3 + ep 10, 1M (r2)"),
            ("r3_2m", "… 2M (r3)"),
            ("r4_kl02", "… 2M + target_kl 0.02 (r4)"),
            ("r3_lr5e-4", "… 2M, lr 5e-4 (r3)"),
            ("final", "… 2M + num_envs 16 (final)"),
        ],
        "ncols": 3,
        "panels": [
            panel(RETURN, "Retorno (últ. 100 ep.)", hline=200.0, ylim=(-50, 290)),
            panel(KL, "approx_kl (log)", yscale="log"),
            panel(ENTROPY, "entropy"),
        ],
    },
    # Part 4, item 4: initial vs final config, each seed its own line
    "final": {
        "per_seed": True,
        "seeds": (1, 2),
        "series": [("base", "config inicial"), ("final", "config final")],
        "ncols": 2,
        "place": 0.62,
        "panels": [
            panel(RETURN, "Retorno (últ. 100 ep.) — seeds 1 e 2", hline=200.0),
            panel(EXPL_VAR, "explained_variance", ylim=(-0.1, 1.02)),
        ],
    },
    # Part 5: best PPO vs our A2C, same 2M budget
    "a2c": {
        "series": [
            ("final", "PPO final"),
            ("a2c_ll", "A2C ajustado"),
            ("a2c_cp", "A2C, HP do CartPole"),
        ],
        "ncols": 4,
        "panels": [
            panel(RETURN, "Retorno × passos", hline=200.0),
            panel(RETURN, "Retorno × tempo (seed 1, run isolado)", hline=200.0, x="_wall"),
            panel(UPDATES, "passos de gradiente acumulados (log)", yscale="log"),
            panel(ENTROPY, "entropy"),
        ],
        # wall-clock panel reads these isolated, one-at-a-time runs instead
        "wall_tags": {"final": "bench_ppo", "a2c_ll": "bench_a2c", "a2c_cp": "bench_a2c_cp"},
    },
    # Part 5 mechanism: the final config with one of PPO's changes switched off
    "abl": {
        "series": [
            ("final", "PPO final"),
            ("abl_reuse", "sem reuso (1 época, 1 minibatch)"),
            ("abl_gae", "sem GAE (λ = 1)"),
            ("abl_clip", "sem clipping (ε = 100)"),
        ],
        "ncols": 3,
        "panels": [
            panel(RETURN, "Retorno (últ. 100 ep.)", hline=200.0),
            panel(KL, "approx_kl (log)", yscale="log"),
            panel(EXPL_VAR, "explained_variance", ylim=(-0.1, 1.02)),
        ],
    },
}


def load_runs() -> dict[str, dict[int, list[dict]]]:
    runs: dict[str, dict[int, list[dict]]] = {}
    for path in sorted(RESULTS.glob("*__seed*/metrics.jsonl")):
        tag, _, seed = path.parent.name.rpartition("__seed")
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
        if rows:
            # A2C does not log charts/num_updates; it takes exactly one gradient
            # step per iteration of num_envs * num_steps transitions
            batch = A2C_BATCH.get(tag)
            if batch:
                for r in rows:
                    if "global_step" in r and RETURN in r:
                        r[UPDATES] = r["global_step"] / batch
            runs.setdefault(tag, {})[int(seed)] = rows
    return runs


def series(rows, metric, x="global_step"):
    pts = [(r[x], r[metric]) for r in rows if metric in r and x in r]
    pts = [(a, b) for a, b in pts if b is not None and math.isfinite(b) and (b > 0 or metric != KL)]
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
    if spec["x"] == "_wall":
        ax.set_xlabel("tempo de relógio (s, 1 núcleo)", fontsize=7 * k, color=INK_2)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}")
    else:
        ax.set_xlabel("passos de ambiente", fontsize=7 * k, color=INK_2)
        ax.xaxis.set_major_formatter(lambda v, _: f"{v / 1e6:.1f}M" if v else "0")
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
        for i, (tag, label) in enumerate(spec["series"]):
            color = CAT[i]
            src = spec.get("wall_tags", {}).get(tag, tag) if xkey == "_wall" else tag
            seeds = runs.get(src, {})
            if spec.get("seeds"):
                seeds = {s: r for s, r in seeds.items() if s in spec["seeds"]}
            elif xkey != "_wall":
                seeds = {s: r for s, r in seeds.items() if s in (1, 2)}
            if not seeds:
                continue
            curves = [(s, *series(rows, metric, xkey)) for s, rows in sorted(seeds.items())]
            curves = [(s, x, ema(y, SMOOTH[metric])) for s, x, y in curves if x.size > 1]
            if not curves:
                continue
            if spec.get("per_seed"):
                for s, x, y in curves:
                    (line,) = ax.plot(x, y, color=color, linewidth=1.5 * k,
                                      linestyle="-" if s == min(spec["seeds"]) else "--", zorder=3)
                    if ax is flat[0] and s == min(spec["seeds"]):
                        handles.append(line)
                        labels.append(label)
                continue
            hi = max(x[-1] for _, x, _ in curves)
            grid = np.linspace(0, hi, GRID_N)
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
        from matplotlib.lines import Line2D
        handles += [Line2D([], [], color=INK_2, lw=1.5 * k), Line2D([], [], color=INK_2, lw=1.5 * k, ls="--")]
        labels += ["seed 1", "seed 2"]
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False, fontsize=7.5 * k,
               labelcolor=INK, bbox_to_anchor=(0.5, -0.02), handlelength=1.8)
    fig.tight_layout(rect=(0, 0.11 / nrows, 1, 1))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{fid}.png"
    fig.savefig(path, dpi=220, facecolor=SURFACE)
    plt.close(fig)
    return path


def first_step(rows, thr):
    for r in rows:
        if RETURN in r and r[RETURN] >= thr:
            return r["global_step"]
    return None


def summarize(runs):
    out = {}
    for tag, seeds in sorted(runs.items()):
        per = {}
        for s, rows in sorted(seeds.items()):
            ret = series(rows, RETURN)[1]
            ev = [r["eval/mean_return"] for r in rows if "eval/mean_return" in r]
            walls = [r["_wall"] for r in rows if "_wall" in r]
            upd = series(rows, UPDATES)[1]
            # drop = largest fall of the return from its running max (after first reaching 100)
            drop = None
            if ret.size:
                runmax = np.maximum.accumulate(ret)
                ok = runmax >= 100
                drop = float(np.max((runmax - ret)[ok])) if ok.any() else None
            per[s] = {
                "final_return": float(ret[-1]) if ret.size else None,
                "max_return": float(ret.max()) if ret.size else None,
                "eval": ev[-1] if ev else None,
                "steps_to": {t: first_step(rows, t) for t in (0, 100, 200)},
                "wall_min": max(walls) / 60 if walls else None,
                "num_updates": float(upd[-1]) if upd.size else None,
                "sps_last": series(rows, SPS)[1][-1].item() if series(rows, SPS)[1].size else None,
                "max_drop_after_100": drop,
            }
        out[tag] = per
    return out


def main():
    runs = load_runs()
    for fid, spec in FIGS.items():
        missing = [t for t, _ in spec["series"] if t not in runs]
        print(f"{fid}: {build(fid, runs)}" + (f"  (missing {missing})" if missing else ""))
    summary = summarize(runs)
    (REPO_ROOT / "report" / "summary_ppo.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{'run':16} {'seed':>4} {'ret':>6} {'max':>6} {'eval':>6} {'t0':>6} {'t100':>6} {'t200':>6} "
          f"{'min':>5} {'updates':>8} {'sps':>6} {'drop':>6}")
    f = lambda v, p="{:.0f}": "-" if v is None else p.format(v)
    for tag, per in summary.items():
        for s, d in per.items():
            st = d["steps_to"]
            print(f"{tag:16} {s:>4} {f(d['final_return']):>6} {f(d['max_return']):>6} {f(d['eval']):>6} "
                  f"{f(st[0] and st[0] / 1e6, '{:.2f}'):>6} {f(st[100] and st[100] / 1e6, '{:.2f}'):>6} "
                  f"{f(st[200] and st[200] / 1e6, '{:.2f}'):>6} {f(d['wall_min'], '{:.1f}'):>5} "
                  f"{f(d['num_updates']):>8} {f(d['sps_last']):>6} {f(d['max_drop_after_100']):>6}")


if __name__ == "__main__":
    main()
