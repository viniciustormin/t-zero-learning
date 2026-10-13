#!/usr/bin/env python3
"""``train.py`` with every ``wandb.log`` row mirrored to a local JSONL file.

The DQN assignment sweep must survive wandb being offline or unavailable: an
offline run keeps its history only inside the binary ``.wandb`` file, which no
public API reads back.  Mirroring each logged row to disk costs nothing and
makes the report figures reproducible from the run directory alone.

Command line is exactly ``train.py``'s.  Two env vars steer the mirror:

    TZ_METRICS_OUT   path of the JSONL file to append metric rows to (required)
    TZ_RUN_TAG       human-readable name of this sweep configuration (optional)

Alongside the JSONL a ``meta.json`` is written with the tag, the wandb run url
(when online) and the final summary, so figures can label runs by sweep value
instead of by the harness's timestamped run name.
"""
import json
import numbers
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import wandb  # noqa: E402
from wandb.sdk.wandb_run import Run  # noqa: E402

import train  # noqa: E402


def _as_number(value):
    """Python number for anything scalar-numeric, else ``None``.

    ``isinstance(v, float)`` is not enough: ``np.float32`` (what
    ``losses/explained_variance`` is, since it comes from ``np.var`` over a
    float32 tensor) is *not* a subclass of ``float``, so a plain isinstance
    filter silently drops the metric from the mirror.  ``np.float64`` is, which
    is why this only bites some metrics.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Number):
        try:
            return int(value) if isinstance(value, numbers.Integral) else float(value)
        except (TypeError, ValueError):
            return None
    return None


def _install_mirror() -> None:
    out_path = Path(os.environ["TZ_METRICS_OUT"]).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path = out_path.parent / "meta.json"
    tag = os.environ.get("TZ_RUN_TAG", "")

    sink = out_path.open("w", buffering=1)
    state = {"url": None, "rows": 0}
    t0 = time.time()
    # Patch the *method*, not ``wandb.log``: ``wandb.init`` rebinds the module
    # attribute to the freshly created run's bound method, which would silently
    # undo a module-level patch.
    original_log = Run.log

    def logged_url() -> str | None:
        run = getattr(wandb, "run", None)
        return getattr(run, "url", None) if run is not None else None

    def write_meta() -> None:
        meta_path.write_text(
            json.dumps(
                {
                    "tag": tag,
                    "argv": sys.argv[1:],
                    "wandb_url": state["url"],
                    "metrics_file": out_path.name,
                    "rows": state["rows"],
                },
                indent=2,
            )
        )

    def mirrored_log(self, data, step=None, **kwargs):
        if isinstance(data, dict):
            row = {k: n for k, v in data.items() if (n := _as_number(v)) is not None}
            if step is not None:
                row.setdefault("global_step", step)
            # wall-clock seconds since launch: wandb's "Relative Time" axis
            row["_wall"] = time.time() - t0
            sink.write(json.dumps(row) + "\n")
            state["rows"] += 1
            if state["url"] is None:
                state["url"] = logged_url()
                if state["url"]:
                    write_meta()
        return original_log(self, data, step=step, **kwargs)

    Run.log = mirrored_log
    write_meta()
    import atexit

    def finish():
        state["url"] = state["url"] or logged_url()
        write_meta()
        sink.close()

    atexit.register(finish)


if __name__ == "__main__":
    os.chdir(REPO_ROOT)
    _install_mirror()
    train.main()
