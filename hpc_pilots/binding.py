import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .common import ROOT, Agent, config, digest, provenance, write_json
from .streams import stream, stream_int


def design_matrix(features, cardinalities=None):
    """One-hot the integer encoding.

    The default of 20 per column reproduces the canonical four-site GB1 encoding exactly.
    Pathogen landscapes are single-substitution scans encoded as (position, residue), so
    they pass explicit per-column cardinalities instead.
    """
    x = np.asarray(features)
    cards = list(cardinalities) if cardinalities else [20] * x.shape[1]
    return np.concatenate([np.eye(c, dtype=np.float32)[x[:, j]] for j, c in enumerate(cards)], axis=1)


def predictions(features, observed_indices, observed_values, seed, trees=64, cardinalities=None):
    model = RandomForestRegressor(n_estimators=trees, min_samples_leaf=2,
                                  max_features=1.0, n_jobs=1, random_state=seed)
    x = design_matrix(features, cardinalities)
    model.fit(x[observed_indices], observed_values)
    estimates = np.array([tree.predict(x) for tree in model.estimators_])
    return estimates.mean(axis=0), estimates.std(axis=0)


def distances(features, observed_indices):
    return np.min(np.mean(features[:, None, :] != features[observed_indices][None, :, :], axis=2), axis=1)


def pick_batch(features, observed_indices, observed_values, available, policy, size, seed, agent=None,
               decision_context=None, cardinalities=None, surrogate=None, round_number=0):
    """The policy interface accepts only revealed values, never the hidden outcome vector.

    `seed` is the episode's base seed and `round_number` a coordinate within it. They are
    kept apart rather than added together, because adding them made the acquisition stream
    of one pool identical to the initialisation stream of another (see `streams`).
    """
    rng = stream(seed, "acquire", round_number)
    if policy == "random":
        return rng.choice(available, size=size, replace=False), {"action": "random"}
    if policy == "diverse":
        chosen = []
        for _ in range(size):
            d = distances(features, np.r_[observed_indices, chosen].astype(int))
            allowed = np.setdiff1d(available, chosen)
            best = allowed[d[allowed] == d[allowed].max()]
            chosen.append(int(rng.choice(best)))
        return np.array(chosen), {"action": "diverse"}
    # Surrogate hyperparameters come from the caller's own config, defaulting to the GB1
    # block so recorded binding replays stay bit-identical. The pathogen and method configs
    # each carried `forest_trees` and `ucb_beta` that nothing read; the values coincided, so
    # no recorded result was wrong, but the keys described an intent the code did not honour.
    sc = surrogate or config()["binding"]
    # A stream of its own, so the surrogate's bootstrap is independent of the tie-break
    # that exists to stop its tied plateaus resolving in index order.
    mean, uncertainty = predictions(features, observed_indices, observed_values,
                                    stream_int(seed, "surrogate", round_number),
                                    sc["forest_trees"], cardinalities)
    d = distances(features, observed_indices)
    action = "explore" if policy == "ucb" else "greedy"
    if policy == "agent":
        if agent is None:
            raise ValueError("Missing agent")
        state = {"observed_labels": len(observed_indices), "batch_size": size,
                 "observed_score_summary": {"min": float(np.min(observed_values)),
                     "mean": float(np.mean(observed_values)), "max": float(np.max(observed_values))},
                 "remaining_candidates": len(available),
                 "candidate_summary": [{"candidate": int(i), "prediction": float(mean[i]),
                     "uncertainty_heuristic": float(uncertainty[i]), "diversity": float(d[i])}
                     for i in available[np.argsort(-mean[available], kind="stable")[:16]]],
                 "available_actions": {"greedy": "highest predicted score",
                     "explore": "predicted score plus uncertainty heuristic",
                     "diverse": "maximize distance from observed candidates"}}
        state["resource_state"] = decision_context or {}
        action = agent.choose("Choose the next acquisition strategy for a binding-assay batch.", state,
                              ["greedy", "explore", "diverse"])
    if action == "diverse":
        # Same coordinates as the diverse baseline would use, so an agent delegating to it
        # produces the identical batch and the two remain comparable.
        return pick_batch(features, observed_indices, observed_values, available, "diverse",
                          size, seed, round_number=round_number)
    acquisition = mean + (sc["ucb_beta"] * uncertainty if action == "explore" else 0)
    # Randomize ties reproducibly without perturbing score ordering.
    shuffled = rng.permutation(available)
    return shuffled[np.argsort(-acquisition[shuffled], kind="stable")[:size]], {"action": action}


def evaluate(hidden_outcomes, selected, fraction):
    threshold = np.quantile(hidden_outcomes, 1 - fraction)
    hits = hidden_outcomes >= threshold
    return {"labels": len(selected), "best_score": float(hidden_outcomes[selected].max()),
            "top_fraction": fraction, "top_hits": int(hits[selected].sum()),
            "top_recall": float(hits[selected].sum() / hits.sum())}


def enrichment(hidden_outcomes, selected, fraction):
    """Hit fraction among the recovered set over the hit fraction in the pool.

    This is the readout shape a display selection round actually produces: a ratio
    against the library baseline, not a per-candidate score. 1.0 means no better than
    the starting library. An enrichment ratio is not an affinity, a functional
    inhibition or an antiviral activity, and a single round cannot establish any of them.
    """
    if len(selected) == 0:
        return None
    threshold = np.quantile(hidden_outcomes, 1 - fraction)
    base = float((hidden_outcomes >= threshold).mean())
    if base <= 0:
        return None
    return float((hidden_outcomes[selected] >= threshold).mean() / base)


def replay(features, hidden_outcomes, policy, seed, maximum=128, initial=8, batch=8,
           delay=0, checkpoints=(8, 16, 32, 64, 128), agent=None, fraction=None,
           cardinalities=None, surrogate=None, control=None):
    """Replay one acquisition policy. `control="permuted"` is the falsification arm.

    Under the permuted control the outcome vector is shuffled before the episode begins
    and the episode is then graded against the shuffled vector, so the environment is
    self-consistent but carries no relationship between features and outcome. There is
    nothing left to learn, and every policy that works by learning one must fall to the
    random baseline. A policy that does not fall is scoring on something other than the
    evidence it paid for -- index order, a tie-breaking bias, pool construction -- and
    its headline number on the real landscape means correspondingly less. Running this
    arm costs one extra replay per configuration and is the cheapest check available
    that the measured gain is a gain.
    """
    if control == "permuted":
        hidden_outcomes = stream(seed, "permute").permutation(hidden_outcomes)
    elif control is not None:
        raise ValueError(f"Unknown control arm {control!r}")
    # Deliberately not a function of the policy: every policy must open from the identical
    # observations at a given seed, which is what removes initialisation variance from the
    # comparison and is what the shared-initialisation test pins.
    first = stream(seed, "initial").choice(len(features), size=initial, replace=False).tolist()
    queried, observed = first.copy(), first.copy()
    pending, trace, records = [], [], []
    round_number = 0
    started = time.perf_counter()
    bc = config()["binding"]
    fraction = bc["top_fraction"] if fraction is None else fraction

    def snapshot():
        metric = evaluate(hidden_outcomes, np.array(queried), fraction)
        returned = evaluate(hidden_outcomes, np.array(observed), fraction)
        metric.update({"round": round_number, "revealed_labels": len(observed),
                       "returned_best_score": returned["best_score"],
                       "returned_top_recall": returned["top_recall"],
                       "pending_labels": len(queried) - len(observed),
                       # Enrichment-ratio view of the same acquisitions, for comparability with
                       # a display selection readout. Retrospective and realtime kept separate.
                       "selected_enrichment": enrichment(hidden_outcomes, np.array(queried), fraction),
                       "returned_enrichment": enrichment(hidden_outcomes, np.array(observed), fraction),
                       "policy_wall_seconds": time.perf_counter() - started})
        records.append(metric)

    snapshot()
    while len(queried) < maximum:
        round_number += 1
        ready = [(due, ids) for due, ids in pending if due <= round_number]
        for _, ids in ready:
            observed.extend(ids)
        pending = [(due, ids) for due, ids in pending if due > round_number]
        available = np.setdiff1d(np.arange(len(features)), queried)
        count = min(batch, maximum - len(queried), len(available))
        ids, decision = pick_batch(features, np.array(observed), hidden_outcomes[observed], available,
                                   policy, count, seed, agent,
                                   {"remaining_label_budget": maximum - len(queried),
                                    "pending_labels": len(queried) - len(observed),
                                    "feedback_delay_rounds": delay, "decision_round": round_number},
                                   cardinalities, surrogate, round_number)
        ids = ids.tolist()
        if len(set(ids)) != count or set(ids) & set(queried):
            raise ValueError("Repeated or invalid acquisition")
        trace.append({"round": round_number, "revealed_before_action": observed.copy(),
                      "selected": ids, "decision": decision})
        queried.extend(ids)
        if delay == 0:
            observed.extend(ids)
        else:
            pending.append((round_number + delay, ids))
        if len(queried) in checkpoints or len(queried) == maximum:
            snapshot()
    # Selection quality is retrospective; delayed results are not available at submission time.
    final_return_round = max([round_number] + [due for due, _ in pending])
    return {"metrics": records, "trace": trace, "initial_indices": first,
            "final_return_round": final_return_round, "control": control,
            "control_note": "Outcomes were permuted before the episode and graded against the "
                            "permutation; every learning policy should fall to random here."
                            if control == "permuted" else "Real outcomes.",
            "delay_units": "synthetic decision rounds, not measured laboratory time"}


def run_binding(pool, policy, delay=0, smoke=False, control=None):
    c, path = config(), ROOT / "data/processed/binding"
    split = np.load(path / "splits.npz")["evaluation"][pool]
    features = np.load(path / "features.npy")[split]
    hidden = pd.read_csv(path / "outcomes.csv", usecols=["DMS_score"]).DMS_score.to_numpy()[split]
    maximum = 16 if smoke else c["binding"]["maximum_labels"]
    agent = Agent() if policy == "agent" else None
    result = replay(features, hidden, policy, c["seed"] + pool, maximum=maximum,
                    initial=c["binding"]["initial_labels"], delay=delay, agent=agent,
                    surrogate=c["binding"], control=control)
    policy = agent.label if agent else policy
    result.update({"experiment": "binding", "pool": pool, "policy": policy, "delay": delay,
                   "smoke": smoke, "status": "complete",
                   # Recorded so the attainable-recall ceiling can be computed at every
                   # checkpoint rather than reconstructed from the config afterwards.
                   "pool_size": len(features),
                   "top_fraction": c["binding"]["top_fraction"],
                   "dataset_manifest_sha256": digest(path / "manifest.json"),
                   "config_sha256": digest(ROOT / "hpc_pilots/configs/today.json"),
                   "provenance": provenance()})
    if agent:
        result["agent"] = agent.report()
    label = "smoke_" if smoke else ""
    # A control arm writes to its own file so it can never overwrite or pool with the real one.
    output = (ROOT / "runs/binding" /
              f"{label}{policy}_pool{pool:02d}_delay{delay}{f'_{control}' if control else ''}.json")
    write_json(output, result)
    print(json.dumps({"output": str(output), "final": result["metrics"][-1]}))


def target_config():
    return json.loads((ROOT / "hpc_pilots/configs/targets.json").read_text())


def run_method(landscape, pool, policy, delay=0, smoke=False, control=None):
    """Replay the acquisition policies against a method-development landscape.

    Identical machinery and the same fixed surrogate as every other replay, so a change of
    substrate is not confounded by a change of model. These landscapes carry measured
    binding endpoints, which the pathogen landscapes do not; that is the point of them.
    """
    mc = json.loads((ROOT / "hpc_pilots/configs/method.json").read_text())
    if landscape not in mc["landscapes"]:
        raise ValueError(f"Unknown method landscape {landscape!r}")
    spec = mc["landscapes"][landscape]
    path = ROOT / "data/processed/method" / landscape
    manifest = json.loads((path / "manifest.json").read_text())
    split = np.load(path / "splits.npz")["evaluation"][pool]
    features = np.load(path / "features.npy")[split]
    hidden = pd.read_csv(path / "outcomes.csv", usecols=["DMS_score"]).DMS_score.to_numpy()[split]
    agent = Agent() if policy == "agent" else None
    result = replay(features, hidden, policy, mc["seed"] + pool,
                    maximum=16 if smoke else mc["maximum_labels"],
                    initial=mc["initial_labels"], batch=mc["batch_size"], delay=delay,
                    checkpoints=tuple(mc["checkpoints"]), agent=agent,
                    fraction=spec["top_fraction"], cardinalities=manifest["cardinalities"],
                    surrogate=mc, control=control)
    policy = agent.label if agent else policy
    result.update({"experiment": "method", "landscape": landscape, "pool": pool,
                   "policy": policy, "delay": delay, "smoke": smoke, "status": "complete",
                   "protein": spec["protein"], "interface": spec["interface"],
                   "endpoint": spec["endpoint"], "top_fraction": spec["top_fraction"],
                   "pool_size": manifest["pool_geometry"]["pool_size"],
                   "role": "method development substrate; not a pathogen target proposal",
                   "endpoint_warning": "Do not pool metrics across landscapes with different "
                                       "assay endpoints or different pool geometries; the "
                                       "multiple over random is bounded by the labelled "
                                       "fraction of the pool.",
                   "dataset_manifest_sha256": digest(path / "manifest.json"),
                   "method_config_sha256": digest(ROOT / "hpc_pilots/configs/method.json"),
                   "provenance": provenance()})
    if agent:
        result["agent"] = agent.report()
    label = "smoke_" if smoke else ""
    output = (ROOT / "runs/method" / f"{label}{landscape}_{policy}_pool{pool:02d}"
              f"_delay{delay}{f'_{control}' if control else ''}.json")
    write_json(output, result)
    print(json.dumps({"output": str(output), "final": result["metrics"][-1]}))


def run_target(landscape, pool, policy, delay=0, smoke=False, control=None):
    """Replay the acquisition policies against a pathogen landscape.

    The surrogate hyperparameters are deliberately the same fixed forest the GB1 pilot
    used, so a substrate change is not confounded by a model change. Each landscape
    carries its own assay endpoint and results from different endpoints are never pooled.
    """
    tc = target_config()
    if landscape not in tc["landscapes"]:
        raise ValueError(f"Unknown landscape {landscape!r}")
    spec = tc["landscapes"][landscape]
    path = ROOT / "data/processed/targets" / landscape
    manifest = json.loads((path / "manifest.json").read_text())
    split = np.load(path / "splits.npz")["evaluation"][pool]
    features = np.load(path / "features.npy")[split]
    hidden = pd.read_csv(path / "outcomes.csv", usecols=["DMS_score"]).DMS_score.to_numpy()[split]
    agent = Agent() if policy == "agent" else None
    result = replay(features, hidden, policy, tc["seed"] + pool,
                    maximum=16 if smoke else tc["maximum_labels"],
                    initial=tc["initial_labels"], batch=tc["batch_size"], delay=delay,
                    checkpoints=tuple(tc["checkpoints"]), agent=agent,
                    fraction=tc["top_fraction"],
                    cardinalities=manifest["cardinalities"], surrogate=tc, control=control)
    policy = agent.label if agent else policy
    result.update({"experiment": "target", "landscape": landscape, "pool": pool, "policy": policy,
                   "delay": delay, "smoke": smoke, "status": "complete",
                   "target": spec["target"], "pathogen": spec["pathogen"], "family": spec["family"],
                   "endpoint": spec["endpoint"], "top_fraction": tc["top_fraction"],
                   "pool_size": manifest["pool_geometry"]["pool_size"],
                   "endpoint_warning": "Do not pool metrics across landscapes with different "
                                       "assay endpoints, and do not compare these top-fraction "
                                       "figures with the GB1 1% figures.",
                   "dataset_manifest_sha256": digest(path / "manifest.json"),
                   "targets_config_sha256": digest(ROOT / "hpc_pilots/configs/targets.json"),
                   "config_sha256": digest(ROOT / "hpc_pilots/configs/today.json"),
                   "provenance": provenance()})
    if agent:
        result["agent"] = agent.report()
    label = "smoke_" if smoke else ""
    output = (ROOT / "runs/targets" / f"{label}{landscape}_{policy}_pool{pool:02d}"
              f"_delay{delay}{f'_{control}' if control else ''}.json")
    write_json(output, result)
    print(json.dumps({"output": str(output), "final": result["metrics"][-1]}))
