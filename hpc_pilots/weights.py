"""What the library model learned, as a coefficient per position and residue.

The acquisition result is reported as a gain over random and a Spearman correlation. Neither
says what the model learned, and a reader cannot get there from the artifacts: `binding.replay`
refits a surrogate from scratch at every round and discards it on return, so fifteen fitted
models per episode leave no trace beyond the batch they picked.

This module keeps them. At every checkpoint of one acquisition episode it fits both arms on the
labels revealed so far and records what each one's parameters are:

* the **ridge** arm's signed coefficient per one-hot column, which is the thing the word
  "weights" ordinarily means and which says *which residue at which position helps*;
* the **forest** arm's split importance per column, which is what the default surrogate has
  instead and which says only *which columns it used*, with no direction.

Kept apart and labelled, because calling an importance a weight would be the error this
repository is careful about everywhere else: an unsigned quantity reported as a signed one.

**Why the four-site library is the right place to do this.** The `binding` arm is Wu 2016's
four-site GB1 library: four randomised positions over the twenty canonical residues, so the
design matrix is exactly 4 x 20 = 80 columns and a coefficient vector *is* a table a reader can
look at. That is also the shape the project's own argument rests on -- a randomised region of a
few positions over an alphabet is what an mRNA display library is -- so this is the learned
object that corresponds to a macrocycle library, not a stand-in for one.

**Three readings of "without training", all of them already defined by the pilot.** A model
fitted on labels permuted before the episode (`control="permuted"`) has nothing to learn and its
coefficients should stay near zero; a `random` acquisition policy consults no surrogate at all
while still revealing labels, so it separates *what the model learned* from *what the policy
chose*; and the first checkpoint, eight labels, is as close to untrained as a fitted model gets.
All three are recorded beside the trained arm rather than described.

Nothing here changes the default surrogate, any recorded replay or any published figure. The
forest is still what `binding.replay` fits unless a config names otherwise, and this module only
reads an episode it runs itself.
"""
import numpy as np

from .binding import design_matrix, pick_batch
from .common import ROOT, config, content_digest, digest, provenance, write_json
from .data import AA
from .streams import stream, stream_int

# The alphabet the integer encoding indexes, in `data.AA` order, so a column index can be named
# rather than left as a number. Read from `data` rather than retyped: the encoding is built there
# and a second copy here would be a second claim about which residue column 7 is.
ALPHABET = AA


def load_pool(pool):
    """One evaluation pool of the four-site library: integer features, design matrix, outcomes.

    Mirrors `calibration.load_pool` but returns the integer features as well, because the
    acquisition policies take those and only the surrogates take the design matrix.
    """
    import pandas as pd

    path = ROOT / "data/processed/binding"
    split = np.load(path / "splits.npz")["evaluation"][pool]
    features = np.load(path / "features.npy")[split]
    outcomes = pd.read_csv(path / "outcomes.csv",
                           usecols=["DMS_score"]).DMS_score.to_numpy()[split]
    return features, design_matrix(features), outcomes


def positions():
    """The genome-order residue numbers of the randomised sites, from the dataset manifest.

    Named from the manifest so the figure's row labels are the assay's own positions and not
    `0..3`. A hard-coded list here would silently keep the old numbers if the dataset were
    restaged over a different construct.
    """
    import json

    manifest = json.loads((ROOT / "data/processed/binding/manifest.json").read_text())
    return [int(p) for p in manifest["positions"]]


# ---------------------------------------------------------------------------
# The two parameter vectors
# ---------------------------------------------------------------------------
def ridge_coefficients(design, outcomes, observed, alphas=None):
    """Signed coefficient per one-hot column, and the penalty leave-one-out chose.

    The penalty is selected on the revealed labels alone. A fixed one would suit a single label
    count, and this function is called at eight labels and again at a hundred and twenty-eight.
    """
    from sklearn.linear_model import RidgeCV

    observed = np.asarray(observed)
    model = RidgeCV(alphas=np.logspace(-3, 4, 29) if alphas is None else alphas)
    model.fit(design[observed], outcomes[observed])
    return (np.asarray(model.coef_, dtype=float), float(model.alpha_),
            float(model.intercept_), model.predict(design))


def forest_importances(design, outcomes, observed, seed, trees):
    """Split importance per one-hot column, from the surrogate the pilot actually uses.

    **An importance is not a weight and this function's output must never be called one.** It
    is non-negative by construction, so it cannot say whether a residue helps or hurts, and it
    is computed as the total variance reduction attributed to splits on that column. It is
    recorded so the default surrogate is not left with nothing to show, and so a reader can see
    whether the two arms at least agree about which columns matter.
    """
    from sklearn.ensemble import RandomForestRegressor

    observed = np.asarray(observed)
    model = RandomForestRegressor(n_estimators=trees, min_samples_leaf=2, max_features=1.0,
                                  n_jobs=1, random_state=seed)
    model.fit(design[observed], outcomes[observed])
    estimates = np.array([tree.predict(design) for tree in model.estimators_])
    return np.asarray(model.feature_importances_, dtype=float), estimates.mean(axis=0)


def as_map(vector, sites, alphabet=ALPHABET):
    """An 80-long column vector as rows of residue values, one row per randomised position.

    The reshape is the whole point of the module's output, so it is one function with one
    assertion rather than an index expression repeated at every call site. `design_matrix`
    concatenates the positions in order, each a full `len(alphabet)` block, so column
    `p * 20 + r` is residue `r` at position `p` -- and if that ever stops being true the length
    check here fails rather than the figure silently transposing.
    """
    width = len(alphabet)
    if len(vector) != len(sites) * width:
        raise ValueError(f"A {len(vector)}-column vector does not reshape to {len(sites)} "
                         f"positions of {width} residues; the design matrix and the site list "
                         f"disagree about the encoding.")
    return [{"position": site,
             "residues": {alphabet[r]: round(float(vector[index * width + r]), 6)
                          for r in range(width)}}
            for index, site in enumerate(sites)]


def extremes(vector, sites, count=8, alphabet=ALPHABET):
    """The columns furthest from zero, named, so the map has a legible top line."""
    width = len(alphabet)
    named = [(float(vector[i * width + r]), sites[i], alphabet[r])
             for i in range(len(sites)) for r in range(width)]
    named.sort(key=lambda item: item[0])
    worst = [{"position": p, "residue": a, "weight": round(v, 6)} for v, p, a in named[:count]]
    best = [{"position": p, "residue": a, "weight": round(v, 6)}
            for v, p, a in reversed(named[-count:])]
    return {"most_favourable": best, "most_unfavourable": worst}


# ---------------------------------------------------------------------------
# One episode, watched
# ---------------------------------------------------------------------------
def skill(predictions, outcomes, observed, fraction):
    """How good the predictions are on candidates the model has not been shown.

    Measured on the held-out complement only. Scoring on revealed labels would report a fit
    rather than a prediction, and at eight labels a ridge regression over eighty columns fits
    them exactly.
    """
    from scipy.stats import spearmanr

    held_out = np.setdiff1d(np.arange(len(outcomes)), observed)
    if len(held_out) < 3:
        return {"n_eval": int(len(held_out)), "spearman_pred_truth": None,
                "top_fraction_precision": None}
    threshold = np.quantile(outcomes, 1 - fraction)
    hits = outcomes >= threshold
    count = max(1, int(hits.sum()))
    top = held_out[np.argsort(-predictions[held_out], kind="stable")[:count]]
    correlation = spearmanr(predictions[held_out], outcomes[held_out]).statistic
    return {"n_eval": int(len(held_out)),
            "spearman_pred_truth": None if np.isnan(correlation) else round(float(correlation), 4),
            "top_fraction_precision": round(float(hits[top].mean()), 4)}


def episode(pool=0, policy="greedy", control=None, maximum=None, initial=None, batch=None,
            checkpoints=None, seed=None, trees=None):
    """Run one acquisition episode and record both parameter vectors at every checkpoint.

    The acquisition path is produced by `binding.pick_batch`, the function the pilot's own
    replay calls, rather than by a simplified greedy written here. A re-implementation would
    drift from the policy whose result this is meant to explain -- `calibration.trajectory`
    carries exactly that caveat about its own simplified regime -- and the tie-break matters at
    low label counts, where the forest produces large exactly-tied plateaus.

    The two arms are fitted on the same revealed labels at every checkpoint, so the ridge
    coefficients and the forest importances describe one episode rather than two. The episode's
    *path* is chosen by whichever surrogate the policy consults, which is the forest under the
    default config; the ridge arm is therefore a reading of the labels the forest's choices
    revealed, and the record says so.
    """
    c = config()
    bc = c["binding"]
    maximum = bc["maximum_labels"] if maximum is None else maximum
    initial = bc["initial_labels"] if initial is None else initial
    batch = bc["batch_size"] if batch is None else batch
    checkpoints = tuple(bc["checkpoints"]) if checkpoints is None else tuple(checkpoints)
    trees = bc["forest_trees"] if trees is None else trees
    seed = (c["seed"] + pool) if seed is None else seed
    fraction = bc["top_fraction"]
    sites = positions()

    features, design, outcomes = load_pool(pool)
    if control == "permuted":
        # The same arm `binding.replay` offers, for the same reason: the environment stays
        # self-consistent and carries no relationship between features and outcome, so a model
        # that still produces structured coefficients is reading something other than the
        # evidence it paid for.
        outcomes = stream(seed, "permute").permutation(outcomes)
    elif control is not None:
        raise ValueError(f"Unknown control arm {control!r}")

    observed = stream(seed, "initial").choice(len(features), size=initial,
                                              replace=False).tolist()
    rows, round_number = [], 0

    def snapshot():
        coefficients, alpha, intercept, ridge_predictions = ridge_coefficients(
            design, outcomes, observed)
        importances, forest_predictions = forest_importances(
            design, outcomes, observed,
            stream_int(seed, "weights-forest", len(observed)), trees)
        rows.append({
            "labels": len(observed), "round": round_number,
            "ridge": {
                "coefficients": as_map(coefficients, sites),
                "extremes": extremes(coefficients, sites),
                "alpha_chosen_by_leave_one_out": round(alpha, 6),
                "intercept": round(intercept, 6),
                "l2_norm": round(float(np.linalg.norm(coefficients)), 6),
                "largest_absolute": round(float(np.abs(coefficients).max()), 6),
                "skill": skill(ridge_predictions, outcomes, observed, fraction),
            },
            "forest": {
                "importances": as_map(importances, sites),
                "extremes": extremes(importances, sites),
                "l2_norm": round(float(np.linalg.norm(importances)), 6),
                "largest": round(float(importances.max()), 6),
                "skill": skill(forest_predictions, outcomes, observed, fraction),
                "note": "Split importances, not weights. Non-negative by construction, so they "
                        "say which columns the forest used and never whether a residue helps.",
            },
        })

    if initial in checkpoints:
        snapshot()
    while len(observed) < maximum:
        round_number += 1
        available = np.setdiff1d(np.arange(len(features)), observed)
        count = min(batch, maximum - len(observed), len(available))
        if count <= 0:
            break
        picked, _ = pick_batch(features, np.array(observed), outcomes[observed], available,
                               policy, count, seed, None, None, None, bc, round_number)
        observed.extend(int(i) for i in picked)
        if len(observed) in checkpoints or len(observed) == maximum:
            snapshot()

    return {
        "pool": pool, "policy": policy, "control": control,
        "positions": sites, "alphabet": ALPHABET,
        "columns": len(sites) * len(ALPHABET),
        "checkpoints": list(checkpoints), "labels": maximum,
        "initial_labels": initial, "batch_size": batch, "forest_trees": trees, "seed": seed,
        "top_fraction": fraction,
        "surrogate_the_path_consulted": (bc.get("surrogate") or "forest")
                                        if policy in ("greedy", "ucb", "agent") else None,
        "surrogate_consulted": policy in ("greedy", "ucb", "agent"),
        "steps": rows,
        "movement": movement(rows),
        "control_note": ("Outcomes were permuted before the episode, so there is nothing to "
                         "learn and a coefficient far from zero is reading something other "
                         "than the evidence." if control == "permuted" else "Real outcomes."),
    }


def movement(rows):
    """How far each parameter vector travelled between checkpoints, per arm.

    The one-number version of "the weights stop moving". Reported per arm because the two are
    on different scales and neither is comparable with the other: a coefficient is in the
    outcome's units and an importance is a fraction of total variance reduction.
    """
    out = {}
    for arm, key in (("ridge", "coefficients"), ("forest", "importances")):
        steps = []
        for before, after in zip(rows, rows[1:]):
            first = _flat(before[arm][key])
            second = _flat(after[arm][key])
            steps.append({"to_labels": after["labels"],
                          "distance": round(float(np.linalg.norm(second - first)), 6)})
        out[arm] = {
            "steps": steps,
            "largest_step": round(max((s["distance"] for s in steps), default=0.0), 6),
            "final_step": round(steps[-1]["distance"], 6) if steps else None,
            "settling": ("the step between the last two checkpoints is smaller than the "
                         "largest one" if steps and steps[-1]["distance"]
                         < max(s["distance"] for s in steps) else
                         "the vector was still moving as fast at the end as anywhere"),
        }
    return out


def _flat(rows):
    """A coefficient map back to one vector, in the map's own order."""
    return np.array([value for row in rows for value in row["residues"].values()], dtype=float)


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
def run_weights(pools=(0,), policies=("greedy", "random"), controls=True, record=True):
    """Run the arms, compare them, and write `runs/weights/binding.json`."""
    arms = {}
    for pool in pools:
        for policy in policies:
            arms[f"{policy}_pool{pool:02d}"] = episode(pool=pool, policy=policy)
        if controls:
            arms[f"greedy_pool{pool:02d}_permuted"] = episode(pool=pool, policy="greedy",
                                                              control="permuted")

    trained = [a for name, a in arms.items() if a["policy"] == "greedy" and not a["control"]]
    shuffled = [a for name, a in arms.items() if a["control"] == "permuted"]
    payload = {
        "experiment": "weights",
        "status": "complete",
        "question": "What does the library model learn as labels arrive, and what does the "
                    "same procedure produce when there is nothing to learn?",
        "answer": _verdict(trained, shuffled),
        "dataset": config()["binding"]["dataset"],
        "library": "Wu 2016 four-site GB1 library: four randomised positions over the twenty "
                   "canonical residues, which is the same shape as a randomised macrocycle "
                   "region and is why this is the learned object that corresponds to one",
        "positions": positions(),
        "alphabet": ALPHABET,
        "arms": arms,
        "what_each_arm_is": {
            "greedy": "the pilot's own acquisition policy, which consults a surrogate",
            "random": "acquisition without a surrogate, so the labels arrive unselected. It "
                      "separates what the model learned from what the policy chose.",
            "permuted": "outcomes shuffled before the episode. Nothing to learn, so the "
                        "coefficients are what this procedure produces from noise.",
        },
        "the_two_arms_are_not_the_same_quantity":
            "`ridge.coefficients` is signed and in the outcome's units: a positive value says "
            "that residue at that position raises the predicted score. `forest.importances` is "
            "non-negative and unitless: it says the forest split on that column and never "
            "which way. Only the first is a weight.",
        "default_surrogate_unchanged":
            "The forest is still what `binding.replay` fits. The ridge arm is registered in "
            "`surrogates.SURROGATES` so it can be selected and inspected; no config on disk "
            "selects it and no recorded replay is affected.",
        "limitations": [
            "One canonical-protein binding assay. No macrocycle, no non-canonical residue and "
            "no cyclisation appears anywhere in this landscape.",
            "A coefficient is fitted to a one-hot column, so a residue that never appears at a "
            "position among the revealed labels gets no coefficient and is reported as zero. "
            "Zero here means 'not seen', not 'no effect', and at eight labels most of the "
            "eighty columns are unseen.",
            "The episode's path is chosen by the policy's own surrogate, which is the forest. "
            "The ridge coefficients therefore describe the labels the forest's choices "
            "revealed, and a different policy reveals different labels.",
            "Ridge coefficients over a one-hot encoding are not identifiable without the "
            "penalty: the twenty columns of one position sum to one for every row, so the "
            "penalty is what picks a single solution out of a family of equivalent ones. "
            "`alpha_chosen_by_leave_one_out` is recorded per checkpoint because the solution "
            "it selects depends on it.",
            "Pool-to-pool variation is variation across sub-landscapes of one assay, not "
            "biological replication.",
        ],
        "config_sha256": digest(ROOT / "hpc_pilots/configs/today.json"),
        "dataset_manifest_sha256": digest(ROOT / "data/processed/binding/manifest.json"),
        "provenance": provenance(),
    }
    payload["content_sha256"] = content_digest(payload)
    if record:
        write_json(ROOT / "runs/weights/binding.json", payload)
    return payload


def _verdict(trained, shuffled):
    """One sentence contrasting the trained arm's learning curve with the control's."""
    if not trained:
        return "No trained arm was run."
    first = trained[0]["steps"][0]
    last = trained[0]["steps"][-1]
    start = first["ridge"]["skill"]["spearman_pred_truth"]
    end = last["ridge"]["skill"]["spearman_pred_truth"]
    norm_start = first["ridge"]["l2_norm"]
    norm_end = last["ridge"]["l2_norm"]
    control = (shuffled[0]["steps"][-1]["ridge"]["skill"]["spearman_pred_truth"]
               if shuffled else None)
    return (
        f"Training moves the weights and the movement buys prediction. Over "
        f"{first['labels']} to {last['labels']} labels the ridge arm's coefficient vector grows "
        f"from an L2 norm of {norm_start} to {norm_end}, and its correlation with the hidden "
        f"scores on candidates it has not been shown rises from {start} to {end}. On outcomes "
        f"permuted before the episode the same procedure ends at {control}, which is what this "
        f"fit produces when there is nothing to learn."
    )


def report(path=None):
    """Print the final coefficient map as a table, for a terminal rather than a page."""
    import json
    from pathlib import Path

    payload = json.loads(Path(path or ROOT / "runs/weights/binding.json").read_text())
    print(payload["answer"])
    for name, arm in payload["arms"].items():
        last = arm["steps"][-1]
        print()
        print(f"{name}  ({last['labels']} labels)")
        print("  " + "".join(f"{a:>7}" for a in payload["alphabet"]))
        for row in last["ridge"]["coefficients"]:
            print(f"{row['position']:<4}" + "".join(f"{v:>7.2f}"
                                                    for v in row["residues"].values()))
        print(f"  held-out Spearman {last['ridge']['skill']['spearman_pred_truth']}  "
              f"top-{payload['arms'][name]['top_fraction']:.0%} precision "
              f"{last['ridge']['skill']['top_fraction_precision']}")
    return payload
