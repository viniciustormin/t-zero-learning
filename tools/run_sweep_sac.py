#!/usr/bin/env python3
"""Run the SAC assignment arms (Part 3 tuning, Part 4 ablations, Part 5 PPO), in parallel.

Same plumbing as ``tools/run_sweep_ppo.py``: every run goes through
``tools/train_csv.py`` so its metrics land in
``sweep_results_sac/<tag>__seed<N>/metrics.jsonl`` (wandb is offline), and
carries ``exp_name=<tag>``. An arm is a base config plus overrides; the arms
are listed in the order the tuning actually went (see report/tuning_log_sac.md).

SAC is CPU-bound (one gradient step per env step, batch 256): a run that lands
on an efficiency core is several times slower, so ``--jobs`` defaults to, and
is capped at, the number of performance cores.

    python tools/run_sweep_sac.py --only base pend:1 ppo_hopper:1,2
    python tools/run_sweep_sac.py --only bench_sac_base --seeds 1 --jobs 1
    python tools/run_sweep_sac.py --list
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "sweep_results_sac"

PPO_HOPPER = ["env_id=Hopper-v5", "total_timesteps=100000", "ppo_continuous_action.gamma=0.99", "track=true"]

# tag -> (config, overrides)
ARMS: dict[str, tuple[str, list[str]]] = {
    # --- Parts 1-2 sanity check: eval/mean_return must end above -200
    "pend": ("sac_pendulum", []),
    # --- round 0: the untouched starting point (CleanRL defaults, 100k steps)
    "base": ("sac_hopper", []),
    # --- Part 5: the repo's continuous PPO as is, only env / budget / gamma changed
    "ppo_hopper": ("ppo_pendulum", PPO_HOPPER),
    # --- round 1 (one change per arm). base/2 and base/3 still gain ~+130
    # ret100 per 10k steps at 100k: learning is limited by gradient steps, not
    # data. Cost grows with updates_per_step (ups4 first in the queue).
    "r1_ups4": ("sac_hopper", ["sac.updates_per_step=4"]),
    "r1_ups2": ("sac_hopper", ["sac.updates_per_step=2"]),
    # alpha needs ~5k gradient steps to leave its init (0.90 at 5.1k -> 0.10 at
    # 10.1k, all 3 seeds); learning_starts 1000 gives 4% of the budget back
    "r1_ls1k": ("sac_hopper", ["sac.learning_starts=1000"]),
    # base/1 collapses at 70-80k (ret100 705 -> 68) with q_minus_return ~195
    # and qf1_loss spikes 52-57: a slower critic
    "r1_qlr3e-4": ("sac_hopper", ["sac.q_lr=3.0e-4"]),
    # --- round 1 verdict: no candidate beats the seed spread (ups4 comes
    # closest on ret100, lower on eval), so the final config is the initial
    # one: configs/sac_hopper_tuned.yml, whose seeds 1-2 are base/1, base/2.
    # --- Part 4 ablations on the final config, seeds 1-2. alpha = 0.09: median
    # over seeds of base's alpha in the 2nd half (0.0999 / 0.0911 / 0.0739)
    "abl_alpha": ("sac_hopper_tuned", ["sac.autotune=false", "sac.alpha=0.09"]),
    "abl_1q": ("sac_hopper_tuned", ["sac.clipped_double_q=false"]),
    # --- Part 5 wall clock: the arms above share the CPU with up to 7 other
    # runs, so their charts/SPS and _wall measure contention. These repeat
    # seed 1, one at a time:  --only bench_sac_final bench_sac_ups4 bench_ppo --seeds 1 --jobs 1
    "bench_sac_final": ("sac_hopper_tuned", []),
    # the closest candidate, to show what updates_per_step costs in time
    "bench_sac_ups4": ("sac_hopper", ["sac.updates_per_step=4"]),
    "bench_ppo": ("ppo_pendulum", PPO_HOPPER),
}

SEEDS = (1, 2, 3)


def perf_cores() -> int:
    """Physical performance cores (Apple silicon); all cores elsewhere."""
    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True, check=True)
        return max(1, int(out.stdout.strip()))
    except (OSError, ValueError, subprocess.CalledProcessError):
        return os.cpu_count() or 1


def parse_spec(spec: str, default_seeds: tuple[int, ...]) -> tuple[str, tuple[int, ...]]:
    """``tag`` (the --seeds) or ``tag:1,2`` (its own seeds)."""
    tag, _, seeds = spec.partition(":")
    return tag, tuple(int(s) for s in seeds.split(",")) if seeds else default_seeds


def build_jobs(specs: list[tuple[str, tuple[int, ...]]], extra: list[str]) -> list[dict]:
    jobs = []
    for tag, seeds in specs:
        config, overrides = ARMS[tag]
        for seed in seeds:
            name = f"{tag}__seed{seed}"
            ov = [f"exp_name={tag}", f"seed={seed}", "capture_video=false", *overrides, *extra]
            jobs.append({
                "name": name,
                "run_dir": RESULTS_DIR / name,
                "cmd": [sys.executable, str(REPO_ROOT / "tools" / "train_csv.py"),
                        "--config", config, "--override", *ov],
            })
    return jobs


def main() -> int:
    max_jobs = perf_cores()
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=max_jobs, help=f"max concurrent runs (<= {max_jobs} P-cores)")
    ap.add_argument("--threads", type=int, default=1, help="torch threads per run")
    ap.add_argument("--only", nargs="*", default=None, help="subset of arm tags (tag or tag:1,2)")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--steps", type=int, default=None, help="override total_timesteps")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    args.jobs = max(1, min(args.jobs, max_jobs))

    if args.list:
        for tag, (config, ov) in ARMS.items():
            print(f"{tag:24} {config:22} {' '.join(ov)}")
        return 0

    specs = [parse_spec(x, tuple(args.seeds)) for x in (args.only or list(ARMS))]
    unknown = {tag for tag, _ in specs} - set(ARMS)
    if unknown:
        print(f"unknown tags: {sorted(unknown)}; valid: {list(ARMS)}")
        return 1
    extra = [f"total_timesteps={args.steps}"] if args.steps else []

    queue = build_jobs(specs, extra)
    print(f"{len(queue)} runs, up to {args.jobs} at a time, {args.threads} threads each", flush=True)
    running, failures, started = [], [], time.time()

    while queue or running:
        while queue and len(running) < args.jobs:
            job = queue.pop(0)
            job["run_dir"].mkdir(parents=True, exist_ok=True)
            env = dict(os.environ)
            env.update({
                "OMP_NUM_THREADS": str(args.threads),
                "MKL_NUM_THREADS": str(args.threads),
                "TZ_METRICS_OUT": str(job["run_dir"] / "metrics.jsonl"),
                "TZ_RUN_TAG": job["name"],
            })
            log = (job["run_dir"] / "train.log").open("w")
            proc = subprocess.Popen(job["cmd"], cwd=REPO_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
            running.append((job, proc, log))
            print(f"[{time.time() - started:7.0f}s] start  {job['name']}", flush=True)

        time.sleep(5)
        still = []
        for job, proc, log in running:
            code = proc.poll()
            if code is None:
                still.append((job, proc, log))
                continue
            log.close()
            if code != 0:
                failures.append(job["name"])
            print(f"[{time.time() - started:7.0f}s] {'ok' if code == 0 else f'FAILED({code})':12} {job['name']}", flush=True)
        running = still

    print(f"\ndone in {(time.time() - started) / 60:.1f} min")
    if failures:
        print(f"failed runs: {failures}  (see sweep_results_sac/<name>/train.log)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
