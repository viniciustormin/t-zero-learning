#!/usr/bin/env python3
"""Run the DQN assignment sweep: 7 configurations x 2 seeds, in parallel.

Every run goes through ``tools/train_csv.py`` so its metrics land in
``sweep_results/<tag>__seed<N>/metrics.jsonl`` regardless of wandb's mood, and
carries ``exp_name=<tag>`` so the run is identifiable by sweep value rather
than by the harness's ``CartPole-v1__empty__<seed>__<timestamp>`` name.

    python tools/run_sweep.py                 # full sweep
    python tools/run_sweep.py --pilot         # one short run, to check plumbing
    python tools/run_sweep.py --only base q1_tnf1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "sweep_results"

# tag -> overrides that differ from configs/dqn_cartpole.yml.
# The baseline (no overrides) is the third point of all three sweeps.
CONFIGS: dict[str, list[str]] = {
    "base": [],
    # Q1 — target network sync: a target recomputed every step vs. one that is
    # refreshed only 100 times in the whole run.
    "q1_tnf1": ["dqn.target_network_frequency=1"],
    "q1_tnf5000": ["dqn.target_network_frequency=5000"],
    # Q2 — replay buffer: ~2 episodes of recent history vs. never evicting.
    "q2_buf500": ["dqn.buffer_size=500"],
    "q2_buf500000": ["dqn.buffer_size=500000"],
    # Q3 — exploration schedule: epsilon floored almost immediately vs. still
    # meaningfully random at 400k steps.
    "q3_ef0.05": ["dqn.exploration_fraction=0.05"],
    "q3_ef0.9": ["dqn.exploration_fraction=0.9"],
}

SEEDS = (1, 2)


def build_jobs(tags: list[str], seeds: tuple[int, ...], extra: list[str]) -> list[dict]:
    jobs = []
    for tag in tags:
        for seed in seeds:
            name = f"{tag}__seed{seed}"
            run_dir = RESULTS_DIR / name
            overrides = [f"exp_name={tag}", f"seed={seed}", "capture_video=false"]
            overrides += CONFIGS[tag] + extra
            jobs.append(
                {
                    "name": name,
                    "tag": tag,
                    "seed": seed,
                    "run_dir": run_dir,
                    "cmd": [
                        sys.executable,
                        str(REPO_ROOT / "tools" / "train_csv.py"),
                        "--config",
                        "dqn_cartpole",
                        "--override",
                        *overrides,
                    ],
                }
            )
    return jobs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6, help="max concurrent runs")
    ap.add_argument("--threads", type=int, default=2, help="torch threads per run")
    ap.add_argument("--only", nargs="*", default=None, help="subset of config tags")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--pilot", action="store_true", help="short plumbing check")
    args = ap.parse_args()

    tags = args.only or list(CONFIGS)
    unknown = set(tags) - set(CONFIGS)
    if unknown:
        print(f"unknown tags: {sorted(unknown)}; valid: {list(CONFIGS)}")
        return 1

    extra: list[str] = []
    seeds = tuple(args.seeds)
    if args.pilot:
        extra = ["total_timesteps=20000", "dqn.learning_starts=2000"]
        tags, seeds = ["base"], (1,)

    jobs = build_jobs(tags, seeds, extra)
    print(f"{len(jobs)} runs, up to {args.jobs} at a time, {args.threads} threads each")

    running: list[tuple[dict, subprocess.Popen, object]] = []
    queue = list(jobs)
    failures: list[str] = []
    started = time.time()

    while queue or running:
        while queue and len(running) < args.jobs:
            job = queue.pop(0)
            job["run_dir"].mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env.update(
                {
                    "OMP_NUM_THREADS": str(args.threads),
                    "MKL_NUM_THREADS": str(args.threads),
                    "TZ_METRICS_OUT": str(job["run_dir"] / "metrics.jsonl"),
                    "TZ_RUN_TAG": job["name"],
                }
            )
            log = (job["run_dir"] / "train.log").open("w")
            proc = subprocess.Popen(
                job["cmd"], cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
            )
            running.append((job, proc, log))
            print(f"[{time.time() - started:7.0f}s] start  {job['name']}")

        time.sleep(5)

        still = []
        for job, proc, log in running:
            code = proc.poll()
            if code is None:
                still.append((job, proc, log))
                continue
            log.close()
            mark = "ok" if code == 0 else f"FAILED(exit {code})"
            if code != 0:
                failures.append(job["name"])
            print(f"[{time.time() - started:7.0f}s] {mark:16} {job['name']}")
        running = still

    print(f"\ndone in {(time.time() - started) / 60:.1f} min")
    if failures:
        print(f"failed runs: {failures}  (see sweep_results/<name>/train.log)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
