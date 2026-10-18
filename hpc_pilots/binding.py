import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .common import ROOT, Agent, config, digest, write_json


def predictions(features, observed_indices, observed_values, seed, trees=64):
    model = RandomForestRegressor(n_estimators=trees, min_samples_leaf=2,
                                  max_features=1.0, n_jobs=1, random_state=seed)
    x = np.eye(20, dtype=np.float32)[features].reshape(len(features), -1)
    model.fit(x[observed_indices], observed_values)
    estimates = np.array([tree.predict(x) for tree in model.estimators_])
    return estimates.mean(axis=0), estimates.std(axis=0)


def distances(features, observed_indices):
    return np.min(np.mean(features[:, None, :] != features[observed_indices][None, :, :], axis=2), axis=1)


def pick_batch(features, observed_indices, observed_values, available, policy, size, seed, agent=None,
               decision_context=None):
    """The policy interface accepts only revealed values, never the hidden outcome vector."""
    rng = np.random.default_rng(seed)
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
    bc = config()["binding"]
    mean, uncertainty = predictions(features, observed_indices, observed_values, seed, bc["forest_trees"])
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
        return pick_batch(features, observed_indices, observed_values, available, "diverse", size, seed)
    acquisition = mean + (bc["ucb_beta"] * uncertainty if action == "explore" else 0)
    # Randomize ties reproducibly without perturbing score ordering.
    shuffled = rng.permutation(available)
    return shuffled[np.argsort(-acquisition[shuffled], kind="stable")[:size]], {"action": action}


def evaluate(hidden_outcomes, selected, fraction):
    threshold = np.quantile(hidden_outcomes, 1 - fraction)
    hits = hidden_outcomes >= threshold
    return {"labels": len(selected), "best_score": float(hidden_outcomes[selected].max()),
            "top_fraction": fraction, "top_hits": int(hits[selected].sum()),
            "top_recall": float(hits[selected].sum() / hits.sum())}


def replay(features, hidden_outcomes, policy, seed, maximum=128, initial=8, batch=8,
           delay=0, checkpoints=(8, 16, 32, 64, 128), agent=None):
    rng = np.random.default_rng(seed)
    first = rng.choice(len(features), size=initial, replace=False).tolist()
    queried, observed = first.copy(), first.copy()
    pending, trace, records = [], [], []
    round_number = 0
    started = time.perf_counter()
    bc = config()["binding"]

    def snapshot():
        metric = evaluate(hidden_outcomes, np.array(queried), bc["top_fraction"])
        returned = evaluate(hidden_outcomes, np.array(observed), bc["top_fraction"])
        metric.update({"round": round_number, "revealed_labels": len(observed),
                       "returned_best_score": returned["best_score"],
                       "returned_top_recall": returned["top_recall"],
                       "pending_labels": len(queried) - len(observed),
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
                                   policy, count, seed + round_number, agent,
                                   {"remaining_label_budget": maximum - len(queried),
                                    "pending_labels": len(queried) - len(observed),
                                    "feedback_delay_rounds": delay, "decision_round": round_number})
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
            "final_return_round": final_return_round,
            "delay_units": "synthetic decision rounds, not measured laboratory time"}


def run_binding(pool, policy, delay=0, smoke=False):
    c, path = config(), ROOT / "data/processed/binding"
    split = np.load(path / "splits.npz")["evaluation"][pool]
    features = np.load(path / "features.npy")[split]
    hidden = pd.read_csv(path / "outcomes.csv", usecols=["DMS_score"]).DMS_score.to_numpy()[split]
    maximum = 16 if smoke else c["binding"]["maximum_labels"]
    agent = Agent() if policy == "agent" else None
    result = replay(features, hidden, policy, c["seed"] + pool, maximum=maximum,
                    delay=delay, agent=agent)
    policy = agent.label if agent else policy
    result.update({"experiment": "binding", "pool": pool, "policy": policy, "delay": delay,
                   "smoke": smoke, "status": "complete",
                   "dataset_manifest_sha256": digest(path / "manifest.json"),
                   "config_sha256": digest(ROOT / "hpc_pilots/configs/today.json")})
    if agent:
        result["agent"] = agent.report()
    label = "smoke_" if smoke else ""
    output = ROOT / "runs/binding" / f"{label}{policy}_pool{pool:02d}_delay{delay}.json"
    write_json(output, result)
    print(json.dumps({"output": str(output), "final": result["metrics"][-1]}))
