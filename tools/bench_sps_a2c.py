#!/usr/bin/env python3
"""Measure charts/SPS for the Q1 configurations, one run at a time.

The sweep runs six jobs concurrently, so its ``charts/SPS`` measures "share of a
contended CPU", not the throughput of the configuration. This script re-runs the
four Q1 points **sequentially**, with the same thread budget, for a short
100k-step slice, and writes ``report/sps_bench_a2c.json`` (tag -> steps/second)
for tools/make_figures_a2c.py to plot.

    python tools/bench_sps_a2c.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT = REPO_ROOT / "report" / "sps_bench_a2c.json"
STEPS = 100_000
THREADS = 2

CONFIGS = {
    "q1_envs1": ["num_envs=1"],
    "q1_envs1_steps40": ["num_envs=1", "a2c.num_steps=40"],
    "base": [],
    "q1_envs32": ["num_envs=32"],
}


def main() -> int:
    results: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as tmp:
        for tag, overrides in CONFIGS.items():
            metrics = Path(tmp) / tag / "metrics.jsonl"
            metrics.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env.update(
                {
                    "OMP_NUM_THREADS": str(THREADS),
                    "MKL_NUM_THREADS": str(THREADS),
                    "TZ_METRICS_OUT": str(metrics),
                    "TZ_RUN_TAG": f"bench_{tag}",
                }
            )
            cmd = [
                sys.executable, str(REPO_ROOT / "tools" / "train_csv.py"),
                "--config", "a2c_cartpole", "--override",
                f"exp_name=bench_{tag}", f"total_timesteps={STEPS}",
                "capture_video=false", "save_model=false", "track=true", *overrides,
            ]
            t0 = time.time()
            proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, capture_output=True, text=True)
            if proc.returncode != 0:
                print(f"{tag}: FAILED\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
                return 1
            rows = [json.loads(l) for l in metrics.read_text().splitlines() if l.strip()]
            sps = [r["charts/SPS"] for r in rows if "charts/SPS" in r]
            wall = STEPS / (time.time() - t0)
            # Report the in-run charts/SPS, not the wall clock: a 100k-step slice
            # spends ~3 s on imports and wandb init, a fixed cost that would swamp
            # the throughput difference we are trying to measure.
            results[tag] = float(sps[-1]) if sps else round(wall, 1)
            print(f"{tag:18} {results[tag]:8.0f} SPS (in-run)   wall incl. startup: {wall:.0f}")

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
