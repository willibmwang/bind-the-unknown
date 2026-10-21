import json
import time

import numpy as np
import pandas as pd
from .arms import arm
from .common import ROOT, Agent, config, digest, provenance, write_json
from .streams import stream, stream_int
from . import surrogates


def design_matrix(features, cardinalities=None):
    """One-hot the integer encoding.

    The default of 20 per column reproduces the canonical four-site GB1 encoding exactly.
    Pathogen landscapes are single-substitution scans encoded as (position, residue), so
    they pass explicit per-column cardinalities instead.
    """
    x = np.asarray(features)
    cards = list(cardinalities) if cardinalities else [20] * x.shape[1]
    return np.concatenate([np.eye(c, dtype=np.float32)[x[:, j]] for j, c in enumerate(cards)], axis=1)


def predictions(features, observed_indices, observed_values, seed, cardinalities=None,
                block=None, name=None):
    """Fit the configured surrogate on the revealed labels and predict over the whole pool.

    A dispatcher rather than a model. Which surrogate runs is a registry lookup keyed by the
    caller's config, so trying a different one is a config change rather than an edit to the
    function every policy calls. `name=None` selects the forest, which is what every config
    currently on disk resolves to, so recorded replays reproduce bit-for-bit.

    The design matrix is built only for a surrogate that wants one. A model conditioning on
    rows of raw feature values is handed the integer encoding instead, which is the whole
    reason the distinction is recorded: the one-hot is where an unseen category loses its
    column, and a model that never had a per-column coefficient cannot lose one.
    """
    # The module rather than the function, because `pick_batch` already has a parameter named
    # `surrogate` holding the config block. Importing the lookup under that name would shadow
    # it there, and a shadowed name is how the answer key reached the wrong variable earlier
    # in this same file's history.
    model = surrogates.surrogate(name)
    design = design_matrix(features, cardinalities) if model.takes_design_matrix else None
    return model.fit_predict(design, np.asarray(features), observed_indices, observed_values,
                             seed, block)


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
                                    cardinalities, sc, sc.get("surrogate"))
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
    """Recall against the hit set, with the denominator it was computed from.

    `hits_in_pool` is recorded rather than left to be reconstructed as `fraction * len`.
    The two differ: the cut is a quantile compared with `>=`, so a tied plateau sitting on
    the threshold puts every tied candidate in the hit set and H exceeds f*N. Two pools of
    the same size can therefore divide by different denominators, which makes a recall
    figure unreadable without the H it used -- and H is what the attainable ceiling
    min(1, n/H) is computed from, so it has to travel with the metric.
    """
    threshold = np.quantile(hidden_outcomes, 1 - fraction)
    hits = hidden_outcomes >= threshold
    return {"labels": len(selected), "best_score": float(hidden_outcomes[selected].max()),
            "top_fraction": fraction, "top_hits": int(hits[selected].sum()),
            "top_recall": float(hits[selected].sum() / hits.sum()),
            "hits_in_pool": int(hits.sum()), "pool_candidates": int(len(hidden_outcomes)),
            "threshold": float(threshold),
            "ties_at_threshold": int((hidden_outcomes == threshold).sum()),
            "hit_rule": "numpy.quantile(1 - top_fraction), linear interpolation, compared >="}


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
    # Which surrogate produced the predictions, and what its spread meant. Both travel with
    # the result because `ucb_beta` multiplies a posterior for one model and a dispersion
    # heuristic for another, and a recall figure cannot be read against another arm's without
    # knowing which. Recorded even for policies that never consult a surrogate, so a `random`
    # row says so rather than leaving the field absent and ambiguous.
    model = surrogates.surrogate((surrogate or {}).get("surrogate"))
    return {"metrics": records, "trace": trace, "initial_indices": first,
            "final_return_round": final_return_round, "control": control,
            "surrogate": model.name,
            "surrogate_spread_is_posterior": model.spread_is_posterior,
            "surrogate_spread_note": model.spread_note,
            "surrogate_consulted": policy in ("greedy", "ucb", "agent"),
            "control_note": "Outcomes were permuted before the episode and graded against the "
                            "permutation; every learning policy should fall to random here."
                            if control == "permuted" else "Real outcomes.",
            "delay_units": "synthetic decision rounds, not measured laboratory time"}


def run_arm(name, landscape=None, pool=0, policy="greedy", delay=0, smoke=False, control=None):
    """Replay one acquisition policy against any arm.

    The four arms previously had a `run_*` apiece, differing in the config they read, the
    directory they load from, the column holding the outcome and the fields they stamp.
    Those differences are now data, in `arms.ARMS`, and this is the single code path, so
    the arms are identical by construction rather than by four separate edits staying in
    step -- which they had already stopped doing.

    Every replay parameter is passed explicitly even where `replay`'s default matches the
    configured value. The defaults coincided with `today.json` by history rather than by
    design, so an arm that quietly relied on them would silently diverge the moment either
    changed.
    """
    a = arm(name)
    cfg = json.loads((ROOT / a.config_path()).read_text())
    block = cfg[a.section] if a.section else cfg
    spec = cfg["landscapes"][landscape] if a.per_landscape else block
    if a.per_landscape and landscape not in cfg["landscapes"]:
        raise ValueError(f"Unknown {name} landscape {landscape!r}")

    path = ROOT / a.staged(landscape)
    manifest = json.loads((path / "manifest.json").read_text())
    geometry = manifest.get("pool_geometry", {})
    # A landscape staged without pools says so in one of two keys. `pool_geometry` writes
    # `disjoint` when it built pools and `pooled: False` when it could not, and this tested
    # only the first -- so the one landscape that is genuinely unreplayable fell straight
    # through to indexing an empty split and died with a bare IndexError, while the geometry
    # was carrying a `reason` string written to explain exactly that.
    unpooled = geometry.get("pooled") is False or not geometry.get("disjoint", True)
    if a.per_landscape and unpooled:
        raise ValueError(f"{landscape} was staged without pools and cannot be replayed; "
                         f"{geometry.get('reason', '')}")
    splits = np.load(path / "splits.npz")["evaluation"]
    # Checked rather than indexed. A geometry key nobody anticipated would otherwise reach
    # the same IndexError by a different route, and a missing pool is a staging fact worth
    # naming rather than an array error to be traced back.
    if pool >= len(splits):
        raise ValueError(f"{landscape} was staged with {len(splits)} evaluation pools; "
                         f"pool {pool} does not exist. "
                         f"{geometry.get('reason', '')}".strip())
    split = splits[pool]
    features = np.load(path / "features.npy")[split]
    hidden = pd.read_csv(path / "outcomes.csv",
                         usecols=[a.outcome_column])[a.outcome_column].to_numpy()[split]

    agent = Agent() if policy == "agent" else None
    fraction = spec["top_fraction"] if a.fraction_from_spec else block["top_fraction"]
    result = replay(features, hidden, policy, cfg["seed"] + pool,
                    maximum=16 if smoke else block["maximum_labels"],
                    initial=block["initial_labels"], batch=block["batch_size"], delay=delay,
                    checkpoints=tuple(block["checkpoints"]), agent=agent, fraction=fraction,
                    cardinalities=manifest.get("cardinalities"), surrogate=block,
                    control=control)
    policy = agent.label if agent else policy

    result.update({"experiment": name, "pool": pool, "policy": policy, "delay": delay,
                   "smoke": smoke, "status": "complete", "top_fraction": fraction,
                   "pool_size": geometry.get("pool_size", len(features)),
                   "dataset_manifest_sha256": digest(path / "manifest.json"),
                   # The manifest's own hash moves whenever the data is restaged, because it
                   # carries observation timestamps, so it cannot verify that a rerun used the
                   # same data. The content hash can, and both are kept: one identifies the
                   # file, the other the data.
                   "dataset_content_sha256": manifest.get("content_sha256"),
                   "provenance": provenance()})
    if a.per_landscape:
        result["landscape"] = landscape
    if a.describe:
        result.update(a.describe(spec, manifest))
    if a.warning:
        result["endpoint_warning"] = a.warning
    for key, filename in a.extra_hashes:
        result[key] = digest(ROOT / "hpc_pilots/configs" / filename)
    if agent:
        result["agent"] = agent.report()

    output = ROOT / a.runs / a.filename(landscape, policy, pool, delay, smoke, control)
    write_json(output, result)
    print(json.dumps({"output": str(output), "final": result["metrics"][-1]}))
    return result


def target_config():
    return json.loads((ROOT / "hpc_pilots/configs/targets.json").read_text())

# The four arms kept a `run_*` apiece until the registry in `arms` made their differences
# data. These remain as the named entry points the CLI, the sweep and the tests already
# use; each is now one call, and they were verified to reproduce their predecessors
# field-for-field before the bodies were deleted.
def run_binding(pool, policy, delay=0, smoke=False, control=None):
    """Replay against the staged GB1 pilot, the one arm with no landscape to name."""
    return run_arm("binding", None, pool, policy, delay, smoke, control)


def run_target(landscape, pool, policy, delay=0, smoke=False, control=None):
    """Replay against a pathogen landscape."""
    return run_arm("target", landscape, pool, policy, delay, smoke, control)


def run_method(landscape, pool, policy, delay=0, smoke=False, control=None):
    """Replay against a method-development landscape carrying a measured binding endpoint."""
    return run_arm("method", landscape, pool, policy, delay, smoke, control)


def run_display(landscape, pool, policy, delay=0, smoke=False, control=None):
    """Replay against a display-selection library."""
    return run_arm("display", landscape, pool, policy, delay, smoke, control)
