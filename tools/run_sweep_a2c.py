#!/usr/bin/env python3
"""Run the A2C assignment sweep: 10 configurations x 2 seeds, in parallel.

Every run goes through ``tools/train_csv.py`` so its metrics land in
``sweep_results_a2c/<tag>__seed<N>/metrics.jsonl`` regardless of wandb's mood,
and carries ``exp_name=<tag>`` so the run is identifiable by sweep value rather
than by the harness's ``CartPole-v1__empty__<seed>__<timestamp>`` name.

    python tools/run_sweep_a2c.py                      # full CartPole sweep (Q1-Q4)
    python tools/run_sweep_a2c.py --pilot              # one short run, to check plumbing
    python tools/run_sweep_a2c.py --only base q1_envs1
    python tools/run_sweep_a2c.py --suite lunar        # the LunarLander extra
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "sweep_results_a2c"

# tag -> overrides that differ from configs/a2c_cartpole.yml.
# The baseline (no overrides) is a point of all four sweeps.
CONFIGS: dict[str, list[str]] = {
    "base": [],
    # Q1 - parallel actors. batch_size = num_envs * num_steps, so num_envs also
    # changes how many transitions enter an update and how many updates fit in
    # 500k steps. `q1_envs1_steps40` is the control that holds both fixed
    # (1x40 = 8x5 = 40): whatever separates it from `base` is decorrelation
    # between actors, not batch size.
    "q1_envs1": ["num_envs=1"],
    "q1_envs32": ["num_envs=32"],
    "q1_envs1_steps40": ["num_envs=1", "a2c.num_steps=40"],
    # Q2 - n-step horizon: pure bootstrap (1) -> nearly Monte Carlo (128).
    "q2_steps1": ["a2c.num_steps=1"],
    "q2_steps32": ["a2c.num_steps=32"],
    "q2_steps128": ["a2c.num_steps=128"],
    # Q3 - entropy bonus: none, baseline 0.01, and one large enough to keep the
    # policy near-uniform (ln 2 on CartPole).
    "q3_ent0": ["a2c.ent_coef=0"],
    "q3_ent0.1": ["a2c.ent_coef=0.1"],
    # Q4 - baseline ablation: the policy-gradient weight becomes the raw return.
    "q4_nobaseline": ["a2c.use_baseline=false"],
}

# The LunarLander extra: configs/a2c_lunarlander.yml is the tuned starting
# point, and `ll_cartpole_hp` puts the *untouched CartPole* hyperparameters on
# LunarLander -- the control that says which of the changes were actually
# needed, rather than asserting it.
LUNAR_CONFIGS: dict[str, list[str]] = {
    "ll_tuned": [],
    # The control that justifies the config: the *untouched CartPole*
    # hyperparameters dropped onto LunarLander.
    "ll_cartpole_hp": [
        "num_envs=8", "a2c.num_steps=5", "a2c.vf_coef=0.5",
        "a2c.max_grad_norm=0.5", "network_kwargs.hidden_layers_size=64",
    ],
    # n-step horizon — the knob the pilot found decisive.
    "ll_steps5": ["a2c.num_steps=5"],
    "ll_steps32": ["a2c.num_steps=32"],
    "ll_steps256": ["a2c.num_steps=256"],
    # the reward-scale mitigations we assumed we would need
    "ll_vf0.5": ["a2c.vf_coef=0.5"],
    "ll_clip0.5": ["a2c.max_grad_norm=0.5"],
    # capacity / throughput
    "ll_lr3e-4": ["a2c.learning_rate=3.0e-4"],
    "ll_envs32": ["num_envs=32"],
    "ll_h64": ["network_kwargs.hidden_layers_size=64"],
}

SUITES = {
    "cartpole": ("a2c_cartpole", CONFIGS),
    "lunar": ("a2c_lunarlander", LUNAR_CONFIGS),
}

SEEDS = (1, 2)


def build_jobs(config: str, configs: dict, tags: list[str], seeds: tuple[int, ...], extra: list[str]) -> list[dict]:
    jobs = []
    for tag in tags:
        for seed in seeds:
            name = f"{tag}__seed{seed}"
            run_dir = RESULTS_DIR / name
            overrides = [f"exp_name={tag}", f"seed={seed}", "capture_video=false"]
            overrides += configs[tag] + extra
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
                        config,
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
    ap.add_argument("--suite", choices=sorted(SUITES), default="cartpole")
    ap.add_argument("--steps", type=int, default=None, help="override total_timesteps")
    args = ap.parse_args()

    config, configs = SUITES[args.suite]
    tags = args.only or list(configs)
    unknown = set(tags) - set(configs)
    if unknown:
        print(f"unknown tags: {sorted(unknown)}; valid: {list(configs)}")
        return 1

    extra: list[str] = []
    seeds = tuple(args.seeds)
    if args.steps:
        extra.append(f"total_timesteps={args.steps}")
    if args.pilot:
        extra = ["total_timesteps=20000"]
        tags, seeds = [tags[0]], (1,)

    jobs = build_jobs(config, configs, tags, seeds, extra)
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
