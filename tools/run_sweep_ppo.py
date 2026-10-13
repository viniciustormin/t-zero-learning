#!/usr/bin/env python3
"""Run the PPO assignment arms (Part 4 tuning, Part 5 A2C comparison), in parallel.

Same plumbing as ``tools/run_sweep_a2c.py``: every run goes through
``tools/train_csv.py`` so its metrics land in
``sweep_results_ppo/<tag>__seed<N>/metrics.jsonl`` (wandb is offline), and
carries ``exp_name=<tag>``. An arm is a base config plus overrides; the arms
are listed in the order the tuning actually went (see report/tuning_log_ppo.md).

    python tools/run_sweep_ppo.py --only base
    python tools/run_sweep_ppo.py --only r1_2m r1_envs16 --seeds 1
    python tools/run_sweep_ppo.py --list
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "sweep_results_ppo"

# tag -> (config, overrides)
ARMS: dict[str, tuple[str, list[str]]] = {
    # the untouched starting point (CleanRL ppo.py defaults, 1M steps)
    "base": ("ppo_lunarlander", []),
    # --- round 1 (1M): base has approx_kl ~1e-4, clipfrac 0, and grad_norm ~6
    # against max_grad_norm 0.5 on 100% of the steps, value_loss ~50: the
    # critic's gradient fills the norm budget and the clip shrinks the policy
    # step ~12x. One change per arm.
    "r1_gn5": ("ppo_lunarlander", ["ppo.max_grad_norm=5.0"]),
    "r1_vf0.1": ("ppo_lunarlander", ["ppo.vf_coef=0.1"]),
    "r1_ep10": ("ppo_lunarlander", ["ppo.update_epochs=10"]),
    "r1_lr1e-3": ("ppo_lunarlander", ["ppo.learning_rate=1.0e-3"]),
    # --- round 2: round 1 refuted the clipping story (gn5 / vf0.1 no better:
    # actor and critic are separate MLPs and Adam undoes a global rescale);
    # what helped is moving the policy more per rollout (lr1e-3, ep10). All
    # arms keep lr=1e-3.
    "r2_lr_ep10": ("ppo_lunarlander", ["ppo.learning_rate=1.0e-3", "ppo.update_epochs=10"]),
    "r2_lr_mb16": ("ppo_lunarlander", ["ppo.learning_rate=1.0e-3", "ppo.num_minibatches=16"]),
    "r2_lr_g999": ("ppo_lunarlander", ["ppo.learning_rate=1.0e-3", "ppo.gamma=0.999"]),
    "r2_lr_2m": ("ppo_lunarlander", ["ppo.learning_rate=1.0e-3", "total_timesteps=2000000"]),
    # --- round 3: lr1e-3 + ep10 reaches 200 at ~0.28M but does not hold it:
    # drops (seed 1 down to 34 at 0.71M) line up with approx_kl spikes
    # (0.018-0.025) and entropy climbing back up. Stabilize, at the 2M budget.
    "r3_2m": ("ppo_lunarlander", ["ppo.learning_rate=1.0e-3", "ppo.update_epochs=10", "total_timesteps=2000000"]),
    "r3_envs16": ("ppo_lunarlander", ["ppo.learning_rate=1.0e-3", "ppo.update_epochs=10", "total_timesteps=2000000",
                                      "num_envs=16"]),
    "r3_kl02": ("ppo_lunarlander", ["ppo.learning_rate=1.0e-3", "ppo.update_epochs=10", "total_timesteps=2000000",
                                    "ppo.target_kl=0.02"]),
    "r3_lr5e-4": ("ppo_lunarlander", ["ppo.learning_rate=5.0e-4", "ppo.update_epochs=10", "total_timesteps=2000000"]),
    # --- round 4: r3_2m and r3_envs16 both meet the criterion; envs16 holds
    # >= 200 from ~1M on with no drop. Close the target_kl question (YAML: the
    # CLI override cannot set an Optional float) and check the final config on
    # two extra seeds (3, 4: --seeds 3 4).
    "r4_kl02": ("ppo_lunarlander_targetkl", []),
    "final": ("ppo_lunarlander_tuned", []),
    # --- round 5: `final` meets the criterion but with no margin: the same
    # trained weights (final seed 1 == r3_envs16 seed 1, identical curves) got
    # eval 204.1 and then 197.6. eval is 10 *sampled* episodes, std ~110 when
    # one or two crash. Entropy ends at ~0.5: make the sampled policy sharper,
    # or weight the whole ~300-step landing episode.
    "r5_ent0": ("ppo_lunarlander_tuned", ["ppo.ent_coef=0.0"]),
    "r5_g999": ("ppo_lunarlander_tuned", ["ppo.gamma=0.999"]),
    # --- Part 5: our A2C on LunarLander, same 2M-step budget as the final PPO ---
    # the assignment's starting point: a2c_cartpole, only the env changed
    "a2c_cp": ("a2c_cartpole", ["env_id=LunarLander-v3", "total_timesteps=2000000"]),
    # the config our A2C LunarLander pilot selected (num_steps=128 decides it)
    "a2c_ll": ("a2c_lunarlander", ["total_timesteps=2000000"]),
    # --- round 5: `final` meets the criterion but with no margin: the same
    # trained weights (final seed 1 == r3_envs16 seed 1, identical curves) got
    # eval 204.1 and then 197.6. eval is 10 *sampled* episodes, std ~110 when
    # one or two crash. Entropy ends at ~0.5: make the sampled policy sharper,
    # or weight the whole ~300-step landing episode.
    "r5_ent0": ("ppo_lunarlander_tuned", ["ppo.ent_coef=0.0"]),
    "r5_g999": ("ppo_lunarlander_tuned", ["ppo.gamma=0.999"]),
    # --- Part 5 mechanism: switch off one of PPO's four changes at a time on
    # the final config. (The ratio cannot go alone: with 1 epoch and 1
    # minibatch it is exactly 1 and the loss is A2C's; that is abl_reuse.)
    "abl_reuse": ("ppo_lunarlander_tuned", ["ppo.update_epochs=1", "ppo.num_minibatches=1"]),
    "abl_gae": ("ppo_lunarlander_tuned", ["ppo.gae_lambda=1.0"]),
    # clip_coef also bounds the value clip (clip_vloss), so both go
    "abl_clip": ("ppo_lunarlander_tuned", ["ppo.clip_coef=100.0"]),
    # --- Part 5 wall clock: the arms above share the CPU with up to 9 other
    # runs, so their charts/SPS and _wall measure contention. These repeat
    # seed 1 of each, one at a time:  --only bench_ppo bench_a2c bench_a2c_cp --seeds 1 --jobs 1
    "bench_ppo": ("ppo_lunarlander_tuned", []),
    "bench_a2c": ("a2c_lunarlander", ["total_timesteps=2000000"]),
    "bench_a2c_cp": ("a2c_cartpole", ["env_id=LunarLander-v3", "total_timesteps=2000000"]),
}

SEEDS = (1, 2)


def build_jobs(tags: list[str], seeds: tuple[int, ...], extra: list[str]) -> list[dict]:
    jobs = []
    for tag in tags:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10, help="max concurrent runs")
    ap.add_argument("--threads", type=int, default=1, help="torch threads per run")
    ap.add_argument("--only", nargs="*", default=None, help="subset of arm tags")
    ap.add_argument("--seeds", nargs="*", type=int, default=list(SEEDS))
    ap.add_argument("--steps", type=int, default=None, help="override total_timesteps")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for tag, (config, ov) in ARMS.items():
            print(f"{tag:24} {config:22} {' '.join(ov)}")
        return 0

    tags = args.only or list(ARMS)
    unknown = set(tags) - set(ARMS)
    if unknown:
        print(f"unknown tags: {sorted(unknown)}; valid: {list(ARMS)}")
        return 1
    extra = [f"total_timesteps={args.steps}"] if args.steps else []

    queue = build_jobs(tags, tuple(args.seeds), extra)
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
        print(f"failed runs: {failures}  (see sweep_results_ppo/<name>/train.log)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
