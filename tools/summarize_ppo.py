#!/usr/bin/env python3
"""One line per run in sweep_results_ppo/: the numbers the tuning decisions read.

For each run: final return_last100, eval/mean_return, and the diagnostics
averaged over the first / last 20% of training (approx_kl, clipfrac, entropy,
explained_variance, value_loss), plus wall time and steps-to-threshold.

    python tools/summarize_ppo.py               # all runs
    python tools/summarize_ppo.py base r1_      # tags starting with these
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "sweep_results_ppo"
DIAG = {
    "kl": "losses/approx_kl",
    "clip": "losses/clipfrac",
    "ent": "losses/entropy",
    "ev": "losses/explained_variance",
    "vl": "losses/value_loss",
}
RET = "charts/episodic_return_mean_last100"


def load(run_dir: Path) -> list[dict]:
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_step(rows: list[dict], thr: float):
    for r in rows:
        if RET in r and r[RET] >= thr:
            return r["global_step"]
    return None


def summarize(rows: list[dict]) -> dict:
    train = [r for r in rows if RET in r or "losses/approx_kl" in r]
    out = {"steps": max((r.get("global_step", 0) for r in rows), default=0)}
    rets = [r[RET] for r in train if RET in r]
    out["ret"] = rets[-1] if rets else float("nan")
    out["ret_max"] = max(rets) if rets else float("nan")
    ev = [r["eval/mean_return"] for r in rows if "eval/mean_return" in r]
    out["eval"] = ev[-1] if ev else float("nan")
    n = len(train)
    for short, key in DIAG.items():
        vals = np.array([r[key] for r in train if key in r and r[key] == r[key]], dtype=float)
        k = max(1, len(vals) // 5)
        out[f"{short}0"] = vals[:k].mean() if len(vals) else float("nan")
        out[f"{short}1"] = vals[-k:].mean() if len(vals) else float("nan")
    out["wall"] = max((r.get("_wall", 0) for r in rows), default=0) / 60
    out["sps"] = [r["charts/SPS"] for r in train if "charts/SPS" in r][-1] if n else 0
    for thr in (0, 100, 200):
        s = first_step(train, thr)
        out[f"t{thr}"] = f"{s / 1e6:.2f}M" if s else "-"
    return out


def main() -> int:
    prefixes = sys.argv[1:]
    runs = sorted(p for p in RESULTS_DIR.glob("*__seed*") if p.is_dir())
    if prefixes:
        runs = [p for p in runs if any(p.name.startswith(x) for x in prefixes)]
    hdr = (f"{'run':30} {'steps':>6} {'ret100':>7} {'max':>6} {'eval':>7} | {'kl 0→1':>13} {'clip 0→1':>11} "
           f"{'ent 0→1':>11} {'ev 0→1':>11} {'vloss 0→1':>13} | {'t0':>5} {'t100':>5} {'t200':>5} {'min':>5} {'sps':>5}")
    print(hdr)
    for p in runs:
        rows = load(p)
        if not rows:
            print(f"{p.name:30} (no metrics yet)")
            continue
        s = summarize(rows)
        print(f"{p.name:30} {s['steps'] / 1e6:5.2f}M {s['ret']:7.1f} {s['ret_max']:6.1f} {s['eval']:7.1f} | "
              f"{s['kl0']:.4f}→{s['kl1']:.4f} {s['clip0']:.2f}→{s['clip1']:.2f} "
              f"{s['ent0']:.2f}→{s['ent1']:.2f} {s['ev0']:.2f}→{s['ev1']:.2f} "
              f"{s['vl0']:6.1f}→{s['vl1']:6.1f} | {s['t0']:>5} {s['t100']:>5} {s['t200']:>5} "
              f"{s['wall']:5.1f} {s['sps']:5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
