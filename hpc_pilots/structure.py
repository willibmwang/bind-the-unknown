import json
import contextlib
import time

import numpy as np
import pandas as pd

from .common import ROOT, Agent, config, digest, provenance, write_json
from .data import fetch_atlas


def preprocess(index, smoke=False):
    import mdtraj as md
    c = config()["structure"]
    pdb, directory = fetch_atlas(index)
    top = directory / (pdb + ".pdb")
    out = ROOT / "data/processed/atlas" / ("sasa_smoke" if smoke else "sasa")
    out.mkdir(parents=True, exist_ok=True)
    arrays, costs, walltimes, counts = {}, {}, {}, {}
    # MDTraj's XTC reader requires ASCII filenames. Relative filenames also work
    # when the project lives under a Unicode directory on the user's Mac.
    with contextlib.chdir(directory):
        initial = md.load(pdb + ".pdb")
        static = md.shrake_rupley(initial, mode="residue", n_sphere_points=c["sphere_points"])[0]
        for rep in (1, 2, 3):
            values, cpu, wall = [], [], []
            for block in md.iterload(f"{pdb}_R{rep}.xtc", top=pdb + ".pdb",
                                    stride=c["stride"], chunk=c["block_frames"]):
                t, w = time.process_time(), time.perf_counter()
                values.append(md.shrake_rupley(block, mode="residue", n_sphere_points=c["sphere_points"]))
                cpu.append(time.process_time() - t)
                wall.append(time.perf_counter() - w)
                if smoke and sum(len(a) for a in values) >= 40:
                    break
            arrays[f"r{rep}"] = np.concatenate(values)
            arrays[f"r{rep}_cpu_seconds"] = np.array(cpu)
            arrays[f"r{rep}_wall_seconds"] = np.array(wall)
            costs[str(rep)], walltimes[str(rep)], counts[str(rep)] = sum(cpu), sum(wall), len(arrays[f"r{rep}"])
    if any(a.shape[1] != len(static) for k, a in arrays.items() if k in ("r1", "r2", "r3")):
        raise ValueError("Trajectory and starting structure residue counts disagree")
    np.savez_compressed(out / f"{pdb}.npz", static=static, **arrays)
    write_json(out / f"{pdb}.json", {"pdb": pdb, "smoke": smoke, "frames": counts,
        "residues": len(static), "units": "nm^2 per residue", "stride": c["stride"],
        "sphere_points": c["sphere_points"], "block_frames": c["block_frames"],
        "analysis_cpu_seconds": costs, "analysis_wall_seconds": walltimes,
        "archive_provenance": json.loads((directory / "provenance.json").read_text()),
        "array_sha256": digest(out / f"{pdb}.npz"),
        "config_sha256": digest(ROOT / "hpc_pilots/configs/today.json"),
        "reference": "R3 is a held-out simulation reference, not experimental truth"})
    print(json.dumps({"pdb": pdb, "frames": counts, "cpu_seconds": costs, "output": str(out)}))


def uncertainty(blocks):
    """Uncertainty heuristic on block means; no frame-independence claim."""
    means = np.array([np.mean(b, axis=0) for b in blocks])
    if len(means) < 2:
        return 0.0
    return float(np.sqrt(np.mean(np.var(means, axis=0, ddof=1))) / np.sqrt(len(means)))


def structure_replay(proteins, policy, additional_blocks=64, seed=0, agent=None):
    sc = config()["structure"]
    size = sc["block_frames"]
    public = {}
    for name, data in proteins.items():
        blocks, costs = [], []
        # Interleave chronological blocks from R1 and R2. R3 is never exposed to a policy.
        for start in range(0, min(len(data["r1"]), len(data["r2"])), size):
            for r in (1, 2):
                block = data[f"r{r}"][start:start + size]
                if len(block) == size:
                    blocks.append(block)
                    costs.append(float(data[f"r{r}_cpu_seconds"][start // size]))
        public[name] = {"available_blocks": blocks, "costs": costs, "observed": [],
                        "next": 0, "static": data["static"]}
    trace, metrics = [], []
    measured_cpu_cost = 0.0
    rng = np.random.default_rng(seed)
    started = time.perf_counter()

    def request(name):
        nonlocal measured_cpu_cost
        state = public[name]
        j = state["next"]
        state["observed"].append(state["available_blocks"][j])
        measured_cpu_cost += state["costs"][j]
        state["next"] += 1

    def measure(step):
        rows = []
        for name, state in public.items():
            estimate = (np.concatenate(state["observed"]).mean(axis=0)
                        if state["observed"] else state["static"])
            # Reference is used solely in the evaluator, never action-selection state.
            reference = np.mean(proteins[name]["r3"], axis=0)
            k = min(sc["top_residues"], len(reference))
            overlap = len(set(np.argsort(estimate)[-k:]) & set(np.argsort(reference)[-k:])) / k
            rows.append({"pdb": name, "rmse_nm2": float(np.sqrt(np.mean((estimate - reference) ** 2))),
                         "top_residue_overlap": overlap, "blocks": state["next"]})
        metrics.append({"additional_blocks": step, "total_blocks": sum(s["next"] for s in public.values()),
                        "processed_frames": sum(sum(len(b) for b in s["observed"]) for s in public.values()),
                        "estimated_acquisition_cpu_seconds": measured_cpu_cost,
                        "policy_wall_seconds": time.perf_counter() - started,
                        "mean_rmse_nm2": float(np.mean([r["rmse_nm2"] for r in rows])),
                        "mean_top_overlap": float(np.mean([r["top_residue_overlap"] for r in rows])),
                        "per_protein": rows})

    if policy != "static":
        for name in public:
            for _ in range(min(sc["initial_blocks_per_protein"], len(public[name]["available_blocks"]))):
                request(name)
    measure(0)
    stopped = False
    if policy != "static":
        for step in range(1, additional_blocks + 1):
            available = [n for n, s in public.items() if s["next"] < len(s["available_blocks"])]
            if not available:
                break
            if policy == "uniform":
                minimum = min(public[n]["next"] for n in available)
                name = str(rng.choice([n for n in available if public[n]["next"] == minimum]))
            elif policy == "uncertainty":
                # Only observed block costs are used to estimate future acquisition cost.
                scores = {n: uncertainty(public[n]["observed"]) /
                          max(float(np.mean(public[n]["costs"][:public[n]["next"]])), 1e-6)
                          for n in available}
                name = max(scores, key=scores.get)
            elif policy == "agent":
                if agent is None:
                    raise ValueError("Missing agent")
                state = {"remaining_block_budget": additional_blocks - step + 1,
                    "entries": [{"candidate": str(i), "observed_blocks": public[n]["next"],
                        "uncertainty_heuristic": uncertainty(public[n]["observed"]),
                        "observed_cpu_seconds_per_block": float(np.mean(public[n]["costs"][:public[n]["next"]]))}
                        for i, n in enumerate(available)],
                    "note": "Uncertainty is a block-mean heuristic, not a calibrated confidence interval."}
                decision = agent.choose("Select which entry needs more surface-exposure evidence, or stop.",
                                        state, [str(i) for i in range(len(available))] + ["stop"])
                if decision == "stop":
                    stopped = True
                    trace.append({"step": step, "action": "stop"})
                    break
                name = available[int(decision)]
            else:
                raise ValueError("Unknown structural policy")
            request(name)
            trace.append({"step": step, "action": "request_block", "pdb": name})
            measure(step)
    return {"metrics": metrics, "trace": trace, "stopped_early": stopped,
            "reference": "mean surface exposure in held-out R3 MD replicate",
            "cost_note": "Acquisition CPU cost is replayed from preprocessing block timings; it is not live MD cost."}


def run_structure(policy, smoke=False):
    directory = ROOT / "data/processed/atlas"
    selection = pd.read_csv(directory / "selection.csv")
    if smoke:
        selection = selection.loc[selection.PDB == "1dk8_A"]
    else:
        selection = selection.loc[selection.split == "evaluation"]
    if selection.empty:
        raise ValueError("No selected structural entries")
    subdirectory = directory / ("sasa_smoke" if smoke else "sasa")
    proteins = {}
    for pdb in selection.PDB:
        with np.load(subdirectory / f"{pdb}.npz") as data:
            proteins[pdb] = {k: data[k] for k in data.files}
    agent = Agent() if policy == "agent" else None
    result = structure_replay(proteins, policy, additional_blocks=2 if smoke else config()["structure"]["additional_blocks"],
                              seed=config()["seed"], agent=agent)
    policy = agent.label if agent else policy
    result.update({"experiment": "structure", "policy": policy, "smoke": smoke,
                   "status": "complete", "dataset_manifest_sha256": digest(directory / "manifest.json"),
                   "config_sha256": digest(ROOT / "hpc_pilots/configs/today.json"),
                   "provenance": provenance()})
    if agent:
        result["agent"] = agent.report()
    output = ROOT / "runs/structure" / f"{'smoke_' if smoke else ''}{policy}.json"
    write_json(output, result)
    print(json.dumps({"output": str(output), "final": result["metrics"][-1]}))
