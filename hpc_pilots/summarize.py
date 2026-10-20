import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import wilcoxon

from .common import ROOT, config, write_json


def headroom(labelled_fraction, top_fraction):
    """The largest recall multiple over random the geometry permits, and the recall ceiling.

    Two things cap recall, and the binding one is whichever is tighter. A policy cannot
    recover more hits than it has labels (ceiling `labelled / top_fraction`), and it
    cannot recover more than exist (ceiling 1). Random recall is about the labelled
    fraction, so the attainable multiple is `min(1 / top_fraction, 1 / labelled)`.

    The summariser previously reported `1 / labelled` unconditionally. That equals the
    bound only while `labelled >= top_fraction`, which holds at the terminal checkpoint
    of every configured landscape and fails at the earlier ones -- at 8 labels in a 512
    pool at 5 percent it claims 64x of headroom where 20x is available, and it is the
    early checkpoints whose raw multiples look most impressive.
    """
    ceiling = min(1.0, labelled_fraction / top_fraction)
    return {"recall_ceiling": ceiling,
            "headroom_multiple": min(1 / top_fraction, 1 / labelled_fraction)}


def summarize_targets(directory):
    """Pathogen landscapes are summarised per landscape and never pooled across endpoints."""
    rows = []
    for path in (ROOT / "runs/targets").glob("*.json"):
        data = json.loads(path.read_text())
        if data.get("smoke") or data.get("status") != "complete":
            continue
        for row in data["metrics"]:
            rows.append({"landscape": data["landscape"], "target": data["target"],
                         "endpoint": data["endpoint"], "policy": data["policy"],
                         "pool": data["pool"], "delay": data["delay"],
                         "file": str(path.relative_to(ROOT)), **row})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(directory / "target_metrics.csv", index=False)
    for landscape, part in df.groupby("landscape"):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        for policy, p in part.groupby("policy"):
            summary = p.groupby("labels")[["top_recall", "returned_top_recall",
                                           "returned_enrichment"]].mean()
            for ax, metric in zip(axes, summary.columns):
                ax.plot(summary.index, summary[metric], marker="o", label=policy)
                ax.set(xlabel="Total queried measurements", ylabel=metric.replace("_", " "))
        axes[2].axhline(1.0, color="0.6", lw=0.8, ls="--")
        axes[0].legend(fontsize=8)
        endpoint = part.endpoint.iloc[0]
        fig.suptitle(f"{part.target.iloc[0]} acquisition replay — endpoint: {endpoint}", fontsize=10)
        fig.savefig(directory / f"target_{landscape}.png", dpi=180)
        plt.close(fig)


def summarize_method(directory):
    """Method-development landscapes, summarised per landscape and never pooled.

    These differ from each other in pool geometry as well as endpoint, and the multiple
    over random is bounded by the labelled fraction of the pool, so the headroom column
    is carried alongside every comparison rather than left for the reader to reconstruct.
    """
    rows = []
    for path in (ROOT / "runs/method").glob("*.json"):
        data = json.loads(path.read_text())
        if data.get("smoke") or data.get("status") != "complete":
            continue
        for row in data["metrics"]:
            rows.append({"landscape": data["landscape"], "protein": data["protein"],
                         "interface": data["interface"], "endpoint": data["endpoint"],
                         "pool_size": data["pool_size"], "top_fraction": data["top_fraction"],
                         "policy": data["policy"],
                         "pool": data["pool"], "delay": data["delay"],
                         "file": str(path.relative_to(ROOT)), **row})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(directory / "method_metrics.csv", index=False)
    final = df[(df.labels == df.labels.max()) & (df.delay == 0)]
    summary = []
    for landscape, part in final.groupby("landscape"):
        wide = part.pivot_table(index="pool", columns="policy", values="top_recall")
        if "random" not in wide:
            continue
        labelled = part.labels.iloc[0] / part.pool_size.iloc[0]
        bound = headroom(labelled, part.top_fraction.iloc[0])
        baseline = wide["random"].mean()
        for policy in [p for p in ("greedy", "ucb", "diverse") if p in wide]:
            gain = wide[policy] - wide["random"]
            nonzero = gain[gain != 0]
            summary.append({
                "landscape": landscape, "protein": part.protein.iloc[0],
                "endpoint": part.endpoint.iloc[0], "pool_size": part.pool_size.iloc[0],
                "labelled_fraction": round(labelled, 4),
                "top_fraction": part.top_fraction.iloc[0],
                "recall_ceiling": round(bound["recall_ceiling"], 4),
                "headroom_multiple": round(bound["headroom_multiple"], 1),
                "policy": policy, "pools": len(gain),
                "random_recall": round(baseline, 4),
                "policy_recall": round(wide[policy].mean(), 4),
                "multiple_over_random": round(wide[policy].mean() / baseline, 2)
                if baseline else None,
                # Normalised against the recall the budget can actually reach, not against
                # 1.0: a policy cannot recover more hits than it has labels to spend, so
                # dividing by (1 - random) understates the gain at early checkpoints.
                "normalised_gain": round((wide[policy].mean() - baseline)
                                         / (bound["recall_ceiling"] - baseline), 4)
                if bound["recall_ceiling"] > baseline else None,
                "mean_paired_gain": round(gain.mean(), 4),
                "paired_sd": round(gain.std(), 4),
                "pools_improved": int((gain > 0).sum()),
                # Pools are one seeded partition of a single assay table, so this is a
                # paired test over sub-samples and not a replication across experiments.
                "wilcoxon_p": round(float(wilcoxon(nonzero).pvalue), 4)
                if len(nonzero) >= 6 else None})
    if summary:
        pd.DataFrame(summary).to_csv(directory / "method_policy_gain.csv", index=False)


def summarize_delay(directory):
    """Isolate the effect of delayed feedback, matched on label count and on metric.

    The selected-versus-returned ratio cannot do this, and reporting it as though it
    could was wrong. `returned` is always a subset of `selected`, and recall divides
    both by the same pool-wide hit count, so the ratio is monotone in set inclusion:
    it is bounded below by 1 for every policy in every run, and what it measures is the
    share of the budget still in flight. With a 128-label budget, batches of 8 and a
    three-round delay, 24 labels are still pending at the last checkpoint and
    128/104 = 1.23 falls out of the arithmetic before any policy acts.

    The `random` policy settles it. It never reads a returned label, so its selections
    are identical at every delay -- and it reproduces the same ratio band as the
    informed policies. A quantity a feedback-blind policy can produce is not a
    measurement of how feedback delay degrades a decision.

    This instead holds the budget and the metric fixed and asks whether a delay changed
    what was selected. The random arm is carried through as the null band: its paired
    difference must be exactly zero, and anything else there is a harness artefact.
    """
    rows = []
    for experiment in ("binding", "targets", "method"):
        for path in (ROOT / "runs" / experiment).glob("*.json"):
            data = json.loads(path.read_text())
            if data.get("smoke") or data.get("status") != "complete" or data.get("control"):
                continue
            final = data["metrics"][-1]
            rows.append({"experiment": experiment, "landscape": data.get("landscape", "gb1"),
                         "policy": data["policy"], "pool": data.get("pool", 0),
                         "delay": data.get("delay", 0), "labels": final["labels"],
                         "selected_recall": final["top_recall"],
                         "returned_recall": final.get("returned_top_recall"),
                         "pending_labels": final.get("pending_labels")})
    if not rows:
        return
    df = pd.DataFrame(rows)
    wide = df.pivot_table(index=["experiment", "landscape", "policy", "pool", "labels"],
                          columns="delay", values="selected_recall")
    delays = [d for d in wide.columns if d != 0]
    if 0 not in wide.columns or not delays:
        return
    summary = []
    for delay in delays:
        paired = wide[[0, delay]].dropna()
        for (experiment, landscape, policy), part in paired.groupby(level=[0, 1, 2]):
            difference = part[delay] - part[0]
            nonzero = difference[difference != 0]
            summary.append({
                "experiment": experiment, "landscape": landscape, "policy": policy,
                "delay": delay, "pools": len(difference),
                # The random arm reads no returned label, so it is the null band rather
                # than a comparison: a non-zero difference here indicts the harness.
                "is_null_band": policy == "random",
                "selected_recall_delay0": round(part[0].mean(), 4),
                "selected_recall_delayed": round(part[delay].mean(), 4),
                "mean_paired_difference": round(difference.mean(), 4),
                "paired_sd": round(difference.std(), 4) if len(difference) > 1 else None,
                "delayed_better": int((difference > 0).sum()),
                "delayed_worse": int((difference < 0).sum()),
                "identical": int((difference == 0).sum()),
                "wilcoxon_p": round(float(wilcoxon(nonzero).pvalue), 4)
                if len(nonzero) >= 6 else None})
    frame = pd.DataFrame(summary)
    frame.to_csv(directory / "delay_effect.csv", index=False)
    informed = frame[~frame.is_null_band]
    pooled = df[df.delay.isin([0] + delays)]
    ratio = (pooled.selected_recall / pooled.returned_recall).replace([float("inf")], pd.NA)
    overall = {
        "matched_comparison": "selected recall at equal label count, delay 0 versus delayed",
        "cells": int(len(informed)), "pools_compared": int(informed.pools.sum()),
        "mean_paired_difference": round(float(
            (informed.mean_paired_difference * informed.pools).sum() / informed.pools.sum()), 4),
        "cells_delayed_better": int((informed.mean_paired_difference > 0).sum()),
        "cells_delayed_worse": int((informed.mean_paired_difference < 0).sum()),
        "null_band_max_abs_difference": round(float(
            frame[frame.is_null_band].mean_paired_difference.abs().max()), 6),
        "selected_over_returned_ratio_mean": round(float(ratio.dropna().mean()), 4),
        "selected_over_returned_ratio_below_one": int((ratio.dropna() < 1).sum()),
        "interpretation": "The selected-over-returned ratio is an identity, not an effect: "
                          "returned is a subset of selected and recall shares a fixed "
                          "denominator, so the ratio never falls below 1 and the "
                          "feedback-blind random policy reproduces it. Read "
                          "mean_paired_difference for the effect of delay on decisions, and "
                          "read it against null_band_max_abs_difference, which must be 0."}
    write_json(directory / "delay_effect.json", overall)


def summarize_emergency(directory):
    rows = []
    for path in (ROOT / "runs/emergency").glob("*.json"):
        data = json.loads(path.read_text())
        if data.get("smoke") or data.get("status") != "complete":
            continue
        rows.append({"file": str(path.relative_to(ROOT)), "packet": data["family_packet"],
                     **data["metrics"]})
    if rows:
        pd.DataFrame(rows).to_csv(directory / "emergency_metrics.csv", index=False)


def summarize():
    directory = ROOT / "runs/summary"
    directory.mkdir(parents=True, exist_ok=True)
    binding, structure = [], []
    for experiment, destination in (("binding", binding), ("structure", structure)):
        for path in (ROOT / "runs" / experiment).glob("*.json"):
            data = json.loads(path.read_text())
            # A control arm is a falsification check, never a result: it must not pool.
            if data.get("smoke") or data.get("status") != "complete" or data.get("control"):
                continue
            for row in data["metrics"]:
                base = {"policy": data["policy"], "file": str(path.relative_to(ROOT))}
                if experiment == "binding":
                    base.update({"pool": data["pool"], "delay": data["delay"]})
                base.update({k: v for k, v in row.items() if k != "per_protein"})
                destination.append(base)
    if binding:
        df = pd.DataFrame(binding)
        df.to_csv(directory / "binding_metrics.csv", index=False)
        for delay, part in df.groupby("delay"):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
            for policy, p in part.groupby("policy"):
                summary = p.groupby("labels")[["best_score", "top_recall"]].agg(["mean", "std", "count"])
                for ax, metric in zip(axes, ("best_score", "top_recall")):
                    mean = summary[(metric, "mean")]
                    # Descriptive variability across pools, not biological confidence intervals.
                    sd = summary[(metric, "std")].fillna(0)
                    ax.plot(mean.index, mean, marker="o", label=policy)
                    ax.fill_between(mean.index, mean - sd, mean + sd, alpha=0.12)
                    ax.set(xlabel="Total queried measurements", ylabel=metric.replace("_", " "))
            axes[0].legend(fontsize=8)
            fig.suptitle(f"GB1 selection replay; synthetic feedback delay = {delay} rounds")
            fig.savefig(directory / f"binding_delay{delay}.png", dpi=180)
            plt.close(fig)
    if structure:
        df = pd.DataFrame(structure)
        df.to_csv(directory / "structure_metrics.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for policy, p in df.groupby("policy"):
            axes[0].plot(p.processed_frames, p.mean_rmse_nm2, label=policy)
            axes[1].plot(p.estimated_acquisition_cpu_seconds, p.mean_top_overlap, label=policy)
        axes[0].set(xlabel="Analyzed trajectory frames", ylabel="Mean SASA RMSE (nm²)")
        axes[1].set(xlabel="Replayed measured analysis CPU seconds", ylabel="Top-10 residue overlap")
        axes[0].legend(fontsize=8)
        fig.suptitle("Structural evidence allocation; held-out MD replicate reference")
        fig.savefig(directory / "structure_allocation.png", dpi=180)
        plt.close(fig)
    summarize_targets(directory)
    summarize_method(directory)
    summarize_delay(directory)
    summarize_emergency(directory)
    bc, sc = config()["binding"], config()["structure"]
    expected_binding = len(bc["policies"]) * bc["evaluation_pools"] * len(bc["feedback_delays"])
    # Hosted-model runs are extra comparisons; only the configured grid counts toward completion.
    completed_binding = len({r["file"] for r in binding if r["policy"] in bc["policies"]})
    completed_structure = len({r["file"] for r in structure if r["policy"] in sc["policies"]})
    status = {"binding_expected": expected_binding, "binding_completed": completed_binding,
              "structure_expected": len(sc["policies"]), "structure_completed": completed_structure,
              "complete": completed_binding == expected_binding and completed_structure == len(sc["policies"]),
              "interpretation": "Pilot results; no claim of non-canonical, antiviral, or full-system efficacy"}
    write_json(directory / "status.json", status)
    print(json.dumps(status))
