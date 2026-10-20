#!/usr/bin/env python3
"""One line per run in sweep_results_sac/: the numbers the tuning decisions read.

For each run: steps, final return_last100, eval/mean_return (mean action) and
eval/mean_return_sampled, alpha at the start / end and the step it first drops
below 0.1 / 0.05 / 0.02, the policy entropy against its target, Q1 values,
q_minus_return, qf1_loss (diagnostics averaged over the last 20% of the logged
rows), SPS, num_updates and wall time. Then one line per arm: mean (min-max)
over its seeds of eval and ret100. Then, for Part 5, the env steps to reach a
few ret100 thresholds (every run) and, for the isolated ``bench_*`` runs,
ret100 at fixed wall-clock times and the time to the thresholds, plus the
transition reuse (gradient steps x batch / transitions). Every number in the
report comes from here.

    python tools/summarize_sac.py               # all runs
    python tools/summarize_sac.py base r1_      # tags starting with these
    python tools/summarize_sac.py --json        # also write report/summary_sac.json
    python tools/summarize_sac.py --trace base  # + every 10k steps, and window stats
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "sweep_results_sac"
RET = "charts/episodic_return_mean_last100"
LEN = "charts/episodic_length_mean_last100"
QMR = "charts/q_minus_return"
ALPHA = "losses/alpha"
ALPHA_THR = (0.1, 0.05, 0.02)
LAST = {
    "ent": "losses/entropy",
    "tgt": "charts/target_entropy",
    "q1": "losses/qf1_values",
    "qmr": "charts/q_minus_return",
    "qret": "charts/soft_return_ep_start",
    "qloss": "losses/qf1_loss",
}


def load(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def last_frac(rows: list[dict], key: str, frac: float = 0.2) -> float:
    vals = np.array([r[key] for r in rows if key in r and r[key] == r[key]], dtype=float)
    if not len(vals):
        return float("nan")
    return float(vals[-max(1, int(len(vals) * frac)):].mean())


def last(rows: list[dict], key: str):
    vals = [r[key] for r in rows if key in r]
    return vals[-1] if vals else float("nan")


def summarize(rows: list[dict]) -> dict:
    train = [r for r in rows if not any(k.startswith("eval/") for k in r)]
    out = {"steps": max((r.get("global_step", 0) for r in rows), default=0)}
    out["ret"] = last(train, RET)
    out["eval"] = last(rows, "eval/mean_return")
    out["eval_s"] = last(rows, "eval/mean_return_sampled")
    alphas = [(r["global_step"], r[ALPHA]) for r in train if ALPHA in r]
    out["alpha0"] = alphas[0][1] if alphas else float("nan")
    out["alpha1"] = alphas[-1][1] if alphas else float("nan")
    out["alpha_half2"] = (float(np.median([a for s, a in alphas if s >= out["steps"] / 2]))
                          if alphas else float("nan"))
    for thr in ALPHA_THR:
        s = next((s for s, a in alphas if a < thr), None)
        out[f"a<{thr}"] = f"{s / 1e3:.0f}k" if s else "-"
    for short, key in LAST.items():
        out[short] = last_frac(train, key)
    out["sps"] = last(train, "charts/SPS")
    out["updates"] = last(train, "charts/num_updates")
    out["wall"] = max((r.get("_wall", 0) for r in rows), default=0) / 60
    return out


THRESHOLDS = (300, 500, 1000)
WALL_AT = (20, 60, 120, 225)  # s: PPO's total, 1 and 2 min, SAC final's total
BATCH = {"ppo": 128}  # PPO minibatch (ppo_pendulum: 4 x 1024 / 32); SAC uses sac.batch_size = 256


def first_reach(rows: list[dict], thr: float, key: str):
    for r in rows:
        if RET in r and r[RET] >= thr:
            return r[key]
    return None


def ret_at_wall(rows: list[dict], t: float):
    vals = [r[RET] for r in rows if RET in r and r.get("_wall", 1e18) <= t]
    return vals[-1] if vals else None


def nearest(rows: list[dict], step: int) -> dict:
    return min(rows, key=lambda r: abs(r["global_step"] - step))


def window(rows: list[dict], key: str, lo: int, hi: int) -> np.ndarray:
    return np.array([r[key] for r in rows if key in r and lo <= r["global_step"] < hi], dtype=float)


def trace(name: str, rows: list[dict]) -> None:
    """The run every 10k steps (Q-G: mean of the episodes ending within +-5k),
    alpha's first logs, and entropy / alpha / Q-G over training windows."""
    train = [r for r in rows if ALPHA in r]
    print(f"\n--- {name}")
    print(f"{'step':>6} {'ret100':>7} {'len':>5} {'alpha':>7} {'ent':>6} {'Q1':>6} {'Q−G':>6} {'G0':>6} {'qf1L':>6}")
    for step in range(10000, 100001, 10000):
        r = nearest(train, step)
        qs = [x for x in rows if QMR in x and abs(x["global_step"] - step) < 5000]
        qmr = np.mean([x[QMR] for x in qs]) if qs else float("nan")
        g0 = np.mean([x[LAST["qret"]] for x in qs]) if qs else float("nan")
        print(f"{r['global_step']:6d} {r.get(RET, float('nan')):7.1f} {r.get(LEN, float('nan')):5.0f} {r[ALPHA]:7.4f} "
              f"{r.get(LAST['ent'], float('nan')):6.2f} {r[LAST['q1']]:6.1f} {qmr:6.1f} {g0:6.1f} {r[LAST['qloss']]:6.2f}")
    first = [train[i] for i in (0, 10, 30, 50) if i < len(train)]
    print("alpha, first logs: " + ", ".join(f"{r['global_step'] / 1e3:.1f}k={r[ALPHA]:.3f}" for r in first))
    ent = LAST["ent"]
    for lo, hi in ((20000, 50000), (50000, 100001)):
        e, a, q = window(train, ent, lo, hi), window(train, ALPHA, lo, hi), window(rows, QMR, lo, hi)
        if not (len(e) and len(a) and len(q)):
            continue
        print(f"[{lo // 1000}k, {min(hi, 100000) // 1000}k): entropy mean {e.mean():.2f} min {e.min():.2f} | "
              f"alpha median {np.median(a):.3f} | Q−G mean {q.mean():.0f} (n={len(q)})")


def fmt_spread(vals: list[float]) -> str:
    vals = [v for v in vals if v == v]
    if not vals:
        return "-"
    return f"{np.mean(vals):7.1f} ({min(vals):.1f}–{max(vals):.1f})"


def main() -> int:
    argv = sys.argv[1:]
    write_json = "--json" in argv
    show_trace = "--trace" in argv
    prefixes = [a for a in argv if a not in ("--json", "--trace")]
    runs = sorted(p for p in RESULTS_DIR.glob("*__seed*") if p.is_dir())
    if prefixes:
        runs = [p for p in runs if any(p.name.startswith(x) for x in prefixes)]

    hdr = (f"{'run':26} {'steps':>6} {'ret100':>7} {'eval':>7} {'evalS':>7} | {'alpha 0→1':>13} {'α½₂':>6} "
           + " ".join(f"{'α<' + str(t):>6}" for t in ALPHA_THR)
           + f" | {'ent':>6} {'tgt':>5} {'Q1':>7} {'G0':>7} {'Q−G':>7} {'qf1L':>7} | {'sps':>5} {'upd':>7} {'min':>5}")
    print(hdr)
    per_run, arms = {}, defaultdict(list)
    for p in runs:
        rows = load(p)
        if not rows:
            print(f"{p.name:26} (no metrics yet)")
            continue
        s = summarize(rows)
        per_run[p.name] = s
        arms[p.name.split("__seed")[0]].append(s)
        print(f"{p.name:26} {s['steps'] / 1e3:5.0f}k {s['ret']:7.1f} {s['eval']:7.1f} {s['eval_s']:7.1f} | "
              f"{s['alpha0']:.4f}→{s['alpha1']:.4f} {s['alpha_half2']:6.4f} "
              + " ".join(f"{s[f'a<{t}']:>6}" for t in ALPHA_THR)
              + f" | {s['ent']:6.2f} {s['tgt']:5.1f} {s['q1']:7.1f} {s['qret']:7.1f} {s['qmr']:7.1f} {s['qloss']:7.2f} | "
              f"{s['sps']:5.0f} {s['updates']:7.0f} {s['wall']:5.1f}")

    print(f"\n{'arm':18} {'n':>2}  {'eval: mean (min–max)':>26}  {'ret100: mean (min–max)':>26}  {'evalS: mean (min–max)':>26}")
    arm_out = {}
    # Parts 4-5 compare against seeds 1-2 only: add that subset for 3-seed arms
    for arm in list(arms):
        names = sorted(n for n in per_run if n.split("__seed")[0] == arm)
        if len(names) > 2:
            arms[f"{arm}[s1-2]"] = [per_run[n] for n in names if n.endswith(("__seed1", "__seed2"))]
    for arm, ss in arms.items():
        arm_out[arm] = {k: [s[k] for s in ss] for k in ("eval", "ret", "eval_s")}
        print(f"{arm:18} {len(ss):2}  {fmt_spread(arm_out[arm]['eval']):>26}  {fmt_spread(arm_out[arm]['ret']):>26}"
              f"  {fmt_spread(arm_out[arm]['eval_s']):>26}")

    print(f"\n{'run':26} " + " ".join(f"{'steps→' + str(t):>11}" for t in THRESHOLDS)
          + f" {'transitions':>11} {'reuse':>6}")
    reach = {}
    for p in runs:
        rows = load(p)
        if not rows or p.name not in per_run:
            continue
        train = [r for r in rows if RET in r]
        reach[p.name] = {t: first_reach(train, t, "global_step") for t in THRESHOLDS}
        s = per_run[p.name]
        transitions = s["steps"]
        batch = BATCH["ppo"] if "ppo" in p.name else 256
        s["reuse"] = s["updates"] * batch / transitions if transitions else float("nan")
        print(f"{p.name:26} " + " ".join(f"{(f'{v / 1e3:.1f}k' if v else '-'):>11}" for v in reach[p.name].values())
              + f" {transitions:11d} {s['reuse']:6.1f}")

    benches = [p for p in runs if p.name.startswith("bench_")]
    if benches:
        print(f"\n{'isolated run':26} {'total s':>8} " + " ".join(f"{'ret@' + str(t) + 's':>9}" for t in WALL_AT)
              + " " + " ".join(f"{'s→' + str(t):>7}" for t in THRESHOLDS))
        for p in benches:
            rows = load(p)
            train = [r for r in rows if RET in r]
            total = max(r.get("_wall", 0) for r in rows)
            at = [ret_at_wall(train, t) for t in WALL_AT]
            secs = [first_reach(train, t, "_wall") for t in THRESHOLDS]
            per_run[p.name]["ret_at_wall"] = dict(zip(WALL_AT, at))
            per_run[p.name]["wall_to"] = dict(zip(THRESHOLDS, secs))
            print(f"{p.name:26} {total:8.0f} " + " ".join(f"{(f'{v:.1f}' if v is not None else '-'):>9}" for v in at)
                  + " " + " ".join(f"{(f'{v:.0f}' if v else '-'):>7}" for v in secs))

    if show_trace:
        for p in runs:
            rows = load(p)
            if rows and p.name in per_run and any(ALPHA in r for r in rows):
                trace(p.name, rows)

    if write_json:
        out = REPO_ROOT / "report" / "summary_sac.json"
        out.write_text(json.dumps({"runs": per_run, "arms": arm_out}, indent=1, default=float))
        print(f"\nwrote {out.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
