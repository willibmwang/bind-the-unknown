import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .common import ROOT, config, provenance, write_json


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


def with_bounds(frame):
    """Attach the attainable ceiling and `attainment` to every checkpoint row.

    Recall at a budget is not comparable across environment instances on its own. Two
    landscapes with different pool sizes and hit thresholds have different attainable
    maxima, so the same raw recall can mean a policy recovered almost everything it could
    have or almost nothing. Carrying the ceiling beside every number, at every checkpoint,
    is what stops a raw figure being read without the bound that limits it.

    `attainment` = recall / min(1, (n/N)/f) is the share of the reachable hits that were
    reached. It reduces to recall exactly when n/N >= f, which is where most of the
    literature sits, so it is a correction in a corner rather than a wholesale revision.
    The quantity is not new: shortfall-from-maximum over shortfall-from-random is the
    "deficiency" of Baram, El-Yaniv & Luz (JMLR 5, 2004, Eq. 1), the Active Learning
    Challenge's globalscore = (ALC - Arand)/(Amax - Arand) (Guyon et al., JMLR W&CP 16,
    2011), and normalised enrichment in the virtual-screening lineage. What is carried
    here is the two-sided ceiling under a budget-coupled selection fraction.

    Where the counted hit total is recorded the ceiling is exact -- min(1, n/H) from the
    H the metric actually divided by. The continuum form min(1, (n/N)/f) is the fallback
    for older rows, and the two differ whenever a tied plateau sits on the threshold.
    """
    if not {"labels", "pool_size", "top_fraction", "top_recall"} <= set(frame.columns):
        return frame
    out = frame.copy()
    out["labelled_fraction"] = out.labels / out.pool_size
    if "hits_in_pool" in out.columns and out.hits_in_pool.notna().all():
        out["recall_ceiling"] = np.minimum(1.0, out.labels / out.hits_in_pool)
        out["ceiling_basis"] = "counted hits"
    else:
        bounds = [headroom(labelled, fraction)
                  for labelled, fraction in zip(out.labelled_fraction, out.top_fraction)]
        out["recall_ceiling"] = [b["recall_ceiling"] for b in bounds]
        out["ceiling_basis"] = "nominal top_fraction"
    # Against the analytic random rate n/N, not against a sampled random arm. A sampled
    # baseline carries its own protocol quirks -- a free positive in the warm start, a
    # pre-filtered pool -- straight into the normaliser.
    out["headroom_multiple"] = out.recall_ceiling / out.labelled_fraction
    out["attainment"] = out.top_recall / out.recall_ceiling
    return out


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
                         "pool_size": data.get("pool_size"),
                         "pool": data["pool"], "delay": data["delay"],
                         "file": str(path.relative_to(ROOT)), **row})
    if not rows:
        return
    df = with_bounds(pd.DataFrame(rows))
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


def summarize_display(directory):
    """Display-selection libraries, summarised per landscape and never pooled.

    These are the first macrocyclic substrates in the project, and one of them carries a
    selection-enrichment readout rather than a per-candidate score, so the enrichment
    columns are reported here beside recall. The endpoints differ from each other as well
    as from every other arm -- selection enrichment on one landscape, reported activity on
    the other -- and an enrichment ratio is not an affinity or a functional inhibition, so
    nothing here is pooled across landscapes and every figure carries its endpoint.
    """
    rows = []
    for path in (ROOT / "runs/display").glob("*.json"):
        data = json.loads(path.read_text())
        if data.get("smoke") or data.get("status") != "complete" or data.get("control"):
            continue
        for row in data["metrics"]:
            rows.append({"landscape": data["landscape"], "target": data["target"],
                         "modality": data["modality"], "endpoint": data["endpoint"],
                         "noncanonical_features": data.get("noncanonical_features"),
                         "pool_size": data["pool_size"], "policy": data["policy"],
                         "pool": data["pool"], "delay": data["delay"],
                         "file": str(path.relative_to(ROOT)), **row})
    if not rows:
        return
    df = with_bounds(pd.DataFrame(rows))
    df.to_csv(directory / "display_metrics.csv", index=False)
    summary = []
    for (landscape, labels), part in df[df.delay == 0].groupby(["landscape", "labels"]):
        wide = part.pivot_table(index="pool", columns="policy", values="top_recall")
        if "random" not in wide:
            continue
        ceiling = float(part.recall_ceiling.mean())
        baseline = float(wide["random"].mean())
        for policy in [p for p in ("greedy", "ucb", "diverse") if p in wide]:
            gain = wide[policy] - wide["random"]
            nonzero = gain[gain != 0]
            summary.append({
                "landscape": landscape, "labels": int(labels),
                "target": part.target.iloc[0], "endpoint": part.endpoint.iloc[0],
                "noncanonical_features": bool(part.noncanonical_features.iloc[0]),
                "pool_size": int(part.pool_size.iloc[0]),
                "top_fraction": part.top_fraction.iloc[0],
                "recall_ceiling": round(ceiling, 4),
                "policy": policy, "pools": len(gain),
                "random_recall": round(baseline, 4),
                "policy_recall": round(float(wide[policy].mean()), 4),
                "policy_attainment": round(float(wide[policy].mean()) / ceiling, 4)
                if ceiling else None,
                "random_attainment": round(baseline / ceiling, 4) if ceiling else None,
                "all_arms_identical": bool(all(
                    np.allclose(wide[p], wide["random"])
                    for p in ("greedy", "ucb", "diverse") if p in wide)),
                "mean_paired_gain": round(float(gain.mean()), 4),
                "pools_improved": int((gain > 0).sum()),
                "wilcoxon_p": round(float(wilcoxon(nonzero).pvalue), 4)
                if len(nonzero) >= 6 else None})
    if summary:
        pd.DataFrame(summary).to_csv(directory / "display_policy_gain.csv", index=False)
    for landscape, part in df.groupby("landscape"):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), layout="constrained")
        for policy, p in part.groupby("policy"):
            curve = p.groupby("labels")[["top_recall", "attainment",
                                         "returned_enrichment"]].mean()
            for ax, metric in zip(axes, curve.columns):
                ax.plot(curve.index, curve[metric], marker="o", label=policy)
                ax.set(xlabel="Total queried measurements", ylabel=metric.replace("_", " "))
        axes[2].axhline(1.0, color="0.6", lw=0.8, ls="--")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"{part.target.iloc[0]} ({part.modality.iloc[0]}) — endpoint: "
                     f"{part.endpoint.iloc[0]}", fontsize=10)
        fig.savefig(directory / f"display_{landscape}.png", dpi=180)
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
    df = with_bounds(pd.DataFrame(rows))
    df.to_csv(directory / "method_metrics.csv", index=False)
    # Every checkpoint, not only the terminal one. Filtering to the last checkpoint made
    # the ceiling correction inert by construction: at 128 labels the budget exceeds every
    # landscape's hit count, so the ceiling is exactly 1.0 in every row and the correction
    # could not move a number it was applied to. It bites at 16 and 32 labels or nowhere.
    summary = []
    for (landscape, labels), part in df[df.delay == 0].groupby(["landscape", "labels"]):
        wide = part.pivot_table(index="pool", columns="policy", values="top_recall")
        if "random" not in wide:
            continue
        ceiling = float(part.recall_ceiling.mean())
        labelled = float(part.labelled_fraction.mean())
        baseline = float(wide["random"].mean())
        arms = [p for p in ("greedy", "ucb", "diverse") if p in wide]
        # At a small budget the surrogate has no discriminative power and every arm
        # returns the same recall. Those rows carry no policy information, and saying so
        # in the table is better than leaving a reader to notice the values are identical.
        degenerate = all(np.allclose(wide[p], wide["random"]) for p in arms) if arms else False
        for policy in arms:
            gain = wide[policy] - wide["random"]
            nonzero = gain[gain != 0]
            summary.append({
                "landscape": landscape, "labels": int(labels),
                "protein": part.protein.iloc[0],
                "endpoint": part.endpoint.iloc[0], "pool_size": int(part.pool_size.iloc[0]),
                "top_fraction": part.top_fraction.iloc[0],
                "hits_in_pool": int(part.hits_in_pool.mean())
                if "hits_in_pool" in part.columns and part.hits_in_pool.notna().all() else None,
                "labelled_fraction": round(labelled, 4),
                "recall_ceiling": round(ceiling, 4),
                "headroom_multiple": round(ceiling / labelled, 2) if labelled else None,
                "policy": policy, "pools": len(gain),
                "random_recall": round(baseline, 4),
                "policy_recall": round(float(wide[policy].mean()), 4),
                "multiple_over_random": round(float(wide[policy].mean()) / baseline, 2)
                if baseline else None,
                # Attainment against the analytic ceiling. This is the quantity that is
                # comparable across landscapes; the raw multiple above is not.
                "policy_attainment": round(float(wide[policy].mean()) / ceiling, 4)
                if ceiling else None,
                "random_attainment": round(baseline / ceiling, 4) if ceiling else None,
                "ceiling_is_slack": ceiling >= 1.0,
                "all_arms_identical": degenerate,
                "mean_paired_gain": round(float(gain.mean()), 4),
                "paired_sd": round(float(gain.std()), 4) if len(gain) > 1 else None,
                "pools_improved": int((gain > 0).sum()),
                "wilcoxon_p": round(float(wilcoxon(nonzero).pvalue), 4)
                if len(nonzero) >= 6 else None})
    if summary:
        pd.DataFrame(summary).to_csv(directory / "method_policy_gain.csv", index=False)
        write_json(directory / "method_policy_gain_note.json", {
            "unit_of_analysis": "Pools are one seeded partition of a single assay table. A "
                                "paired test over them describes variation across sub-samples "
                                "of one measurement set, not replication across experiments, "
                                "and a sub-sampled metric is inconsistent with the full-pool "
                                "metric rather than merely noisier. The landscape, not the "
                                "pool, is the independent unit for a cross-landscape claim.",
            "attainment": "recall / min(1, (n/N)/f), against the analytic random rate n/N "
                          "rather than a sampled random arm. Reduces to recall when n/N >= f.",
            "prior_art": "The chance-and-ceiling normalisation is published: Baram, El-Yaniv & "
                         "Luz, JMLR 5 (2004) Eq. 1; Guyon et al., JMLR W&CP 16 (2011) "
                         "globalscore; normalised enrichment in the virtual-screening "
                         "lineage. Nothing here claims the normalisation as novel.",
            "where_the_correction_is_inert": "Wherever n/N >= f the ceiling is 1.0 and "
                                             "attainment equals recall exactly. At the terminal "
                                             "checkpoint of every configured landscape the "
                                             "budget exceeds the hit count, so the correction "
                                             "matters only at the early checkpoints."})


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
    for experiment in ("binding", "targets", "method", "display"):
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
    """Aggregate the decision episodes, keeping the control arm strictly apart.

    The zero-retrieval control is reported on its own, because what it measures is not
    policy performance but how much of the answer is recoverable without buying any
    evidence at all. That number is a floor: the evidence-using arm has to clear it
    before any of its performance can be credited to the evidence it paid for.
    """
    rows, probe, filtered = [], [], []
    for path in (ROOT / "runs/emergency").glob("*.json"):
        data = json.loads(path.read_text())
        if data.get("smoke"):
            continue
        if data.get("status") == "filtered":
            # Attrition has to be reported, not inferred from a missing file. An episode
            # that failed on a provider content filter is missing data with a known cause,
            # and the rate matters: it determines how much of the intended design survived.
            filtered.append({"file": str(path.relative_to(ROOT)), "packet": data["family_packet"],
                             "policy": data["policy"], "control": data.get("control"),
                             "replicate": data.get("replicate", 1)})
            continue
        if data.get("status") != "complete":
            continue
        record = {"file": str(path.relative_to(ROOT)), "packet": data["family_packet"],
                  "replicate": data.get("replicate", 1), **data["metrics"]}
        (probe if data.get("control") else rows).append(record)
    if rows:
        pd.DataFrame(rows).to_csv(directory / "emergency_metrics.csv", index=False)
    if not probe:
        return
    frame = pd.DataFrame(probe)
    frame.to_csv(directory / "memorisation_probe.csv", index=False)
    summary = []
    for (packet, policy), part in frame.groupby(["packet", "policy"]):
        decided = part[part.chosen_target.notna()]
        summary.append({
            "packet": packet, "policy": policy, "episodes": len(part),
            "evidence_retrieved": int(part.evidence_retrieved.sum()),
            # A policy that declines to recommend from an empty dossier is behaving
            # correctly, so refusals are counted rather than scored as failures.
            "declined_to_recommend": int(part.chosen_target.isna().sum()),
            "recommended": len(decided),
            "matched_reference": int(decided.chosen_matches_reference.sum()) if len(decided) else 0,
            "match_rate": round(float(decided.chosen_matches_reference.mean()), 4)
            if len(decided) else None})
    # The contrast the probe exists for: same readout, same policy, evidence bought or not.
    contrast = []
    if rows:
        using = pd.DataFrame(rows)
        for policy in sorted(set(frame.policy) & set(using.policy)):
            blind = frame[frame.policy == policy]
            seeing = using[using.policy == policy]
            blind_decided = blind[blind.chosen_target.notna()]
            seeing_decided = seeing[seeing.chosen_target.notna()]
            if not len(blind_decided) or not len(seeing_decided):
                continue
            blind_rate = float(blind_decided.chosen_matches_reference.mean())
            seeing_rate = float(seeing_decided.chosen_matches_reference.mean())
            contrast.append({
                "policy": policy,
                "zero_retrieval_episodes": len(blind_decided),
                "zero_retrieval_match_rate": round(blind_rate, 4),
                "zero_retrieval_evidence_bought": int(blind.evidence_retrieved.sum()),
                "evidence_using_episodes": len(seeing_decided),
                "evidence_using_match_rate": round(seeing_rate, 4),
                "evidence_using_evidence_bought": int(seeing.evidence_retrieved.sum()),
                "difference": round(seeing_rate - blind_rate, 4),
                # The conclusion to draw when the control is already at 1.0 is about the
                # READOUT, not about the policy. A measure that pretrained knowledge alone
                # saturates has no room left to register evidence use, so it cannot be
                # evidence that the retrieval mechanism works -- or that it does not.
                "readout_saturated_by_control": blind_rate >= 1.0,
                "verdict": "Control saturates the readout: target-selection agreement has no "
                           "power to detect evidence-driven decision quality here, whatever "
                           "the evidence-using arm scores."
                if blind_rate >= 1.0 else
                           "Control below ceiling: the margin over it is the part of "
                           "performance attributable to purchased evidence."})
    write_json(directory / "memorisation_probe.json", {
        "control": "zero_retrieval",
        "contrast_with_evidence_using_arm": contrast,
        "question": "With retrieval forbidden and no evidence pushed, does the policy still "
                    "name the target the full evidence base supports?",
        "by_cell": summary,
        "total_evidence_retrieved": int(frame.evidence_retrieved.sum()),
        "episodes_lost_to_content_filter": len([f for f in filtered if f["control"]]),
        "filtered_episodes": [f for f in filtered if f["control"]],
        "interpretation": "Any match here was not derived from evidence, because none was "
                          "bought. For a policy with pretrained knowledge of named targets "
                          "this measures hindsight, which the decision-clock boundary cannot "
                          "control. Read it as a floor on the evidence-using arm, not as skill."})


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
                    base.update({"pool": data["pool"], "delay": data["delay"],
                                 "pool_size": data.get("pool_size")})
                base.update({k: v for k, v in row.items() if k != "per_protein"})
                destination.append(base)
    if binding:
        df = with_bounds(pd.DataFrame(binding))
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
    summarize_display(directory)
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
              "provenance": provenance(),
              "interpretation": "Pilot results; no claim of non-canonical, antiviral, or full-system efficacy"}
    write_json(directory / "status.json", status)
    print(json.dumps(status))
