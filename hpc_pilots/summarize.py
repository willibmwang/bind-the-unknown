import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .common import ROOT, config, write_json


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
