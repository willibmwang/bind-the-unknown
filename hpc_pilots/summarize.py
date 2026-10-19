import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common import ROOT, config, write_json


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
            if data.get("smoke") or data.get("status") != "complete":
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
