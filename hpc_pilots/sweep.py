"""Run the acquisition grid in parallel on one machine.

The domain layer -- the pathogen and method landscapes -- has no Slurm wiring and was run
by hand, one command per cell, which is why the recorded sweep was partial: two of four
method landscapes complete, one policy short by four pools, the pathogen landscapes
covering the first ten pools of the nineteen and twenty available. A grid that has to be
driven by hand gets covered unevenly, and the gaps then look like choices.

Each cell writes its own self-contained artifact, so cells are independent and a failure
costs one cell rather than the sweep. Failures are collected and reported rather than
raised: a sweep that stops at the first bad cell has to be restarted from the beginning,
and the usual cause is one landscape missing from disk.

This is CPU work -- a few seconds per cell, dominated by fitting a small forest -- so
local processes are the right tool. The GPU arm, where a local model serves the agent
policy, is the part that wants a remote accelerator.
"""
import concurrent.futures
import contextlib
import io
import json
import time

import numpy as np

from .arms import arm as arm_spec
from .common import ROOT, provenance, write_json

POLICIES = ("random", "diverse", "greedy", "ucb")


def available_pools(arm, landscape=None):
    """How many disjoint evaluation pools the staged landscape actually holds.

    Read from the splits on disk rather than from the config, because `pool_geometry`
    shrinks the pool count to keep pools disjoint and the config records what was
    requested, not what was built. A landscape too small for two disjoint pools is staged
    with an empty split and reports zero, which is how it drops out of the grid rather
    than failing in it.
    """
    path = ROOT / arm_spec(arm).staged(landscape) / "splits.npz"
    if not path.exists():
        return 0
    with np.load(path) as splits:
        evaluation = splits["evaluation"]
        return int(evaluation.shape[0]) if evaluation.ndim == 2 and evaluation.shape[1] else 0


def grid(arms=("binding", "target", "method", "display"), controls=True, policies=POLICIES,
         max_pools=None):
    """Every cell the staged data supports, as (arm, landscape, pool, policy, delay, control)."""
    tasks = []
    for arm in arms:
        a = arm_spec(arm)
        cfg = json.loads((ROOT / a.config_path()).read_text())
        block = cfg[a.section] if a.section else cfg
        landscapes = list(cfg["landscapes"]) if a.per_landscape else [None]
        delays = block["feedback_delays"]
        for landscape in landscapes:
            pools = available_pools(arm, landscape)
            if max_pools:
                pools = min(pools, max_pools)
            for pool in range(pools):
                for delay in delays:
                    for policy in policies:
                        tasks.append((arm, landscape, pool, policy, delay, None))
                        # The falsification arm only needs the undelayed cells: it asks
                        # whether the policy is learning at all, not how delay affects it.
                        if controls and delay == 0:
                            tasks.append((arm, landscape, pool, policy, delay, "permuted"))
    return tasks


def execute(task):
    """Run one cell. Never raises: a failed cell is reported, not fatal to the sweep."""
    arm, landscape, pool, policy, delay, control = task
    started = time.perf_counter()
    try:
        from .binding import run_arm
        # One code path for every arm. The result line each cell prints is unreadable
        # across thousands of cells, so stdout is captured and discarded; the artifact
        # is the record.
        with contextlib.redirect_stdout(io.StringIO()):
            run_arm(arm, landscape, pool, policy, delay, False, control)
    except Exception as error:
        return {"task": task, "ok": False, "seconds": time.perf_counter() - started,
                "error": f"{type(error).__name__}: {error}"[:300]}
    return {"task": task, "ok": True, "seconds": time.perf_counter() - started}


def run_sweep(arms=("binding", "target", "method", "display"), workers=8, controls=True,
              max_pools=None, dry_run=False):
    tasks = grid(arms, controls=controls, max_pools=max_pools)
    counts = {}
    for arm, landscape, _, _, _, control in tasks:
        key = f"{arm}:{landscape or 'gb1'}{'/permuted' if control else ''}"
        counts[key] = counts.get(key, 0) + 1
    if dry_run:
        print(json.dumps({"cells": len(tasks), "by_group": counts}, indent=2))
        return
    print(json.dumps({"cells": len(tasks), "workers": workers}), flush=True)
    started = time.perf_counter()
    results, done = [], 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(execute, tasks, chunksize=1):
            results.append(result)
            done += 1
            if done % 50 == 0 or done == len(tasks):
                failed = sum(1 for r in results if not r["ok"])
                print(f"{done}/{len(tasks)} cells, {failed} failed, "
                      f"{time.perf_counter() - started:.0f}s elapsed", flush=True)
    failures = [r for r in results if not r["ok"]]
    report = {"cells": len(tasks), "completed": len(results) - len(failures),
              "failed": len(failures), "by_group": counts,
              "wall_seconds": round(time.perf_counter() - started, 1),
              "cell_seconds_mean": round(sum(r["seconds"] for r in results) / len(results), 2)
              if results else None,
              "failures": [{"task": list(r["task"]), "error": r["error"]} for r in failures[:50]],
              "provenance": provenance()}
    write_json(ROOT / "runs/sweep_report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    if failures:
        print(f"first failures:\n" + "\n".join(
            f"  {r['task']}: {r['error']}" for r in failures[:5]))
