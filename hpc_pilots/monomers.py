"""Monomer encoding, and the one thing a vocabulary provably cannot do.

Every surrogate in this project encodes a library member as a tuple of token indices and
one-hots them. That is a vocabulary: the model learns a coefficient per *named* building
block and has none for a block it never saw. `chemistry.encode` is explicit about it and
raises `KeyError` on an unregistered block rather than guessing.

For canonical residues the limitation is invisible, because a deep mutational scan contains
all twenty. For the molecule class this project designs it is the whole problem. A macrocycle
library is a combinatorial draw over non-canonical monomers, the measured fraction of that
space is vanishing, and the monomers that matter next are the ones not yet tried. A model
that can only score blocks it has already seen measured cannot rank a proposal containing a
new one -- which is what designing a library means.

The claim this module tests is sharper than "descriptors help", and part of it is not an
empirical claim at all.

**The structural half.** Hold out a monomer from training and the vocabulary model's one-hot
column for it has zero variance over the training rows, so no tree splits on it and no
coefficient is fitted to it. The model's prediction for two members differing *only* in which
held-out monomer they carry is therefore identical, exactly, by construction. Its ability to
discriminate among unseen monomers is zero and cannot be improved by more data of the same
kind. `paired_discrimination` measures it and the expected answer is a tied-prediction rate
of 1.0.

**The empirical half.** A model whose per-position features are the monomer's measured
structure -- molecular weight, logP, polar surface area, donors, acceptors, rotatable bonds,
rings, aromaticity, heavy atoms, sp3 fraction -- places an unseen monomer in the same space as
the seen ones and can interpolate. Whether that interpolation actually ranks held-out members
better is a question about this landscape, and it can come back null.

**The control that decides whether chemistry is doing the work.** A descriptor model could
beat a vocabulary model for reasons unrelated to chemistry: sixty-six continuous columns
against a sparse indicator basis is a different inductive bias, not a better representation.
So the descriptors are also run *shuffled* -- the same ten numbers per position, permuted
across monomers, which destroys the correspondence between a block's identity and its
structure while leaving dimensionality, scale and model class untouched. An advantage that
survives shuffling is not chemistry. This is the same move as the permuted-outcome control in
the acquisition arm, applied to the features instead of the labels, and it is the control a
reviewer asks for first.

Both encoders are additionally run under two model classes, because a forest and a ridge have
different biases over sparse indicators and dense continuous columns, and attributing an
encoder effect to the encoder requires that it not reverse when the model changes.

This is the same question the decision arm asks, on a different substrate. There, a score
carried by a target's *name* collapses when the target is renamed. Here, a score carried by a
monomer's *identity* collapses when the monomer is new. In both the measurement is what
survives removing the identity, and in both the part that survives is the part that transfers.
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV

from .arms import arm
from .binding import design_matrix, evaluate
from .common import ROOT, digest, provenance, write_json
from .data import DESCRIPTORS
from .streams import stream, stream_int

ENCODERS = ("vocabulary", "descriptor", "both", "descriptor_shuffled")
MODELS = ("forest", "ridge")

# Positions whose alphabet is large enough to hold monomers back from. A position carrying one
# symbol has nothing to hold out, and holding out from a position with four would remove a
# quarter of the library per monomer, so the split would be a statement about sample size
# rather than about representation.
MINIMUM_ALPHABET = 10


def _landscape(name="thrombin_cyclic"):
    spec = json.loads((ROOT / arm("display").config_path()).read_text())
    if name not in spec["landscapes"]:
        raise ValueError(f"Unknown display landscape {name!r}")
    return spec, spec["landscapes"][name]


def load(name="thrombin_cyclic"):
    """Features, outcomes, per-position descriptors and the monomer vocabularies.

    Raises when the landscape carries no descriptor matrix, rather than falling back to the
    token encoding. A transfer experiment silently run with both arms on the same features
    would report no difference and the null would look like a result about chemistry.
    """
    path = ROOT / arm("display").staged(name)
    manifest = json.loads((path / "manifest.json").read_text())
    if not (path / "descriptors.npy").exists():
        raise FileNotFoundError(f"{name} has no descriptor matrix; the transfer comparison "
                                f"needs one and must not fall back to token indices")
    features = np.load(path / "features.npy").astype(int)
    outcomes = pd.read_csv(path / "outcomes.csv",
                           usecols=[arm("display").outcome_column]).to_numpy().ravel()
    return (features, outcomes, np.load(path / "descriptors.npy"),
            manifest["vocabularies"], manifest)


def absence_flags(vocabularies, name="thrombin_cyclic"):
    """Which symbols mean "nothing at this position", read from the monomer table.

    Derived from the table rather than from an all-zero descriptor row. The staged manifest
    warns that a zero row is indistinguishable from a real monomer whose descriptors all
    happen to be zero, so inferring absence from the numbers would encode exactly the
    ambiguity the warning names. An explicit indicator lets the model represent "absent"
    separately from "measured as zero", and the shuffled control permutes the descriptors
    while leaving the indicator in place, so an advantage cannot come from the flag alone.
    """
    _, spec = _landscape(name)
    table = pd.read_csv(ROOT / spec["monomer_table"]).set_index("monomer")
    flags = np.zeros((len(vocabularies), max(len(v) for v in vocabularies)), dtype=np.float32)
    for position, vocabulary in enumerate(vocabularies):
        for index, symbol in enumerate(vocabulary):
            flags[position, index] = float(bool(table.loc[symbol, "is_absence"]))
    return flags


def holdout(features, vocabularies, count, seed, minimum=MINIMUM_ALPHABET):
    """Split the library by monomer rather than by member.

    A member-wise split leaks: every monomer appears in thousands of members, so a random
    80/20 split leaves each one well represented in training and measures interpolation
    within a known alphabet. The question here is the other one -- what happens when the
    monomer itself is new -- so the held-out monomers are removed from training wherever they
    occur, including from positions other than the one they were drawn for.

    Returns the training and test row indices and the monomers held back. Test rows are the
    members carrying at least one held-out monomer, which is the set a model would have to
    rank to evaluate a library built from a new block.
    """
    eligible = [p for p, v in enumerate(vocabularies) if len(v) >= minimum]
    if not eligible:
        raise ValueError(f"No position carries at least {minimum} symbols to hold out from")
    # Drawn from the union of the eligible positions' alphabets, because a monomer appearing
    # at two positions must be unseen at both or it is not held out at all.
    shared = sorted(set.union(*(set(vocabularies[p]) for p in eligible)))
    if count >= len(shared):
        raise ValueError(f"Holding out {count} of {len(shared)} monomers leaves nothing to "
                         f"train on")
    rng = stream(seed, "holdout", count)
    held = sorted(shared[int(i)] for i in rng.choice(len(shared), size=count, replace=False))

    carries = np.zeros(len(features), dtype=bool)
    for position in eligible:
        indices = {i for i, s in enumerate(vocabularies[position]) if s in held}
        if indices:
            carries |= np.isin(features[:, position], sorted(indices))
    return np.flatnonzero(~carries), np.flatnonzero(carries), held, eligible


def _standardise(block, train_rows):
    """Centre and scale on the training rows only, leaving constant columns at zero.

    Scaling on the whole matrix would let the test members' chemistry set the mean, which is
    a leak that flatters the descriptor arm specifically -- the arm whose advantage is the
    claim. A column constant across training carries no information and is set to zero rather
    than divided by its spread, and the count of such columns is reported so a representation
    that is mostly constant cannot be mistaken for a rich one.

    Constancy is tested on the column's extent rather than on its standard deviation, and
    that distinction is the whole of a real defect. Three columns on the thrombin library --
    the initiator position's sp3 fraction and the fixed position's weight and polar surface
    area -- hold a *single distinct value* over every training row, so their deviation is
    mathematically zero. numpy accumulates the mean in float32, which does not return that
    value exactly, and the residual depends on the reduction's accumulation order: the same
    constant column reduces to 1.2e-07 read as a contiguous slice and 3.2e-06 read
    column-wise out of the 66-wide matrix. A threshold on the deviation therefore catches the
    column or misses it according to how it happened to be reduced, which is why loosening
    the tolerance was not the fix. Dividing by that residual amplified float32 representation
    noise into three columns with mean +/-1 and spread 2e-5: constant features that look
    informative, whose sign would flip arbitrarily with the split, handed to the arm whose
    advantage is the claim under test.

    The extent is a max less a min. Nothing accumulates, so for identical values it is exactly
    zero under any reduction.
    """
    observed = block[train_rows]
    mean, spread = observed.mean(axis=0), observed.std(axis=0)
    extent = observed.max(axis=0) - observed.min(axis=0)
    constant = extent <= 1e-6 * np.maximum(np.abs(mean), 1.0)
    scaled = np.zeros_like(block)
    scaled[:, ~constant] = (block[:, ~constant] - mean[~constant]) / spread[~constant]
    return scaled, int(constant.sum())


def encode(features, descriptors, flags, vocabularies, train_rows, encoder, seed):
    """Build one design matrix. Returns the matrix and what it is made of.

    The shuffled arm permutes descriptor *rows* within each position, so monomer X7 receives
    some other registered monomer's measured structure. Dimensionality, column scales, the
    absence indicator and the model class are all unchanged; the only thing destroyed is the
    correspondence between which block sits at a position and what that block is made of.
    """
    if encoder not in ENCODERS:
        raise ValueError(f"Unknown encoder {encoder!r}; known encoders are {list(ENCODERS)}")
    cardinalities = [len(v) for v in vocabularies]
    if encoder == "vocabulary":
        return design_matrix(features, cardinalities), {"kind": "one-hot token index",
                                                        "constant_columns": 0}

    chemistry = descriptors.copy()
    if encoder == "descriptor_shuffled":
        for position, vocabulary in enumerate(vocabularies):
            order = stream(seed, "shuffle-descriptors", position).permutation(len(vocabulary))
            chemistry[position, :len(vocabulary)] = chemistry[position, order]

    columns = []
    for position in range(features.shape[1]):
        taken = features[:, position]
        rows = chemistry[position][taken]
        if np.isnan(rows).any():
            raise ValueError(f"Position {position} has a missing descriptor for an observed "
                             f"monomer; a half-populated chemistry matrix is worse than none")
        columns.append(np.column_stack([rows, flags[position][taken]]))
    block = np.concatenate(columns, axis=1).astype(np.float32)
    scaled, constant = _standardise(block, train_rows)
    detail = {"kind": f"{len(DESCRIPTORS)} RDKit descriptors plus an absence indicator per "
                      f"position", "constant_columns": constant}
    if encoder == "both":
        scaled = np.concatenate([design_matrix(features, cardinalities), scaled], axis=1)
        detail["kind"] = "one-hot token index concatenated with " + detail["kind"]
    return scaled, detail


def fit(matrix, outcomes, train_rows, model, seed, trees=64):
    if model == "forest":
        estimator = RandomForestRegressor(n_estimators=trees, min_samples_leaf=2,
                                          max_features=1.0, n_jobs=1, random_state=seed)
    elif model == "ridge":
        # The penalty is selected over a log-spaced grid on the training rows. A fixed penalty
        # would favour whichever encoder's column count happened to suit it.
        estimator = RidgeCV(alphas=np.logspace(-3, 4, 29))
    else:
        raise ValueError(f"Unknown model {model!r}; known models are {list(MODELS)}")
    estimator.fit(matrix[train_rows], outcomes[train_rows])
    return estimator.predict(matrix)


def paired_discrimination(features, outcomes, predictions, vocabularies, held, positions):
    """Can the model tell two members apart when they differ only in an unseen monomer?

    Members are grouped by every position except one; within a group they are identical but
    for which monomer sits at the remaining position. Pairs where both members carry a
    held-out monomer are the ones that matter, because that is where a vocabulary model has
    no coefficient for either and must predict them identically.

    `tied_rate` is the measurement and it is not a statistic about this landscape: for a
    one-hot encoding it is 1.0 by construction, and no quantity of training data of the same
    kind moves it. `concordance` is reported beside it over the untied pairs, so a model that
    discriminates but discriminates wrongly is distinguishable from one that cannot
    discriminate at all.
    """
    held_indices = {p: {i for i, s in enumerate(vocabularies[p]) if s in held}
                    for p in positions}
    tied = compared = concordant = 0
    for position in positions:
        others = [p for p in range(features.shape[1]) if p != position]
        groups = {}
        for row in range(len(features)):
            if features[row, position] in held_indices[position]:
                groups.setdefault(tuple(features[row, others]), []).append(row)
        for members in groups.values():
            for first, second in zip(members, members[1:]):
                if outcomes[first] == outcomes[second]:
                    continue
                gap = predictions[first] - predictions[second]
                if gap == 0:
                    tied += 1
                    continue
                compared += 1
                concordant += int((gap > 0) == (outcomes[first] > outcomes[second]))
    total = tied + compared
    return {"pairs": total, "tied_predictions": tied,
            "tied_rate": None if not total else round(tied / total, 4),
            "compared": compared,
            "concordance": None if not compared else round(concordant / compared, 4)}


def transfer(landscape="thrombin_cyclic", held_count=6, seed=20261021, budget=128,
             control=None):
    """Fit every encoder and model on a monomer-wise split and score the held-out members.

    `control="permuted"` shuffles the outcomes before fitting and grades against the shuffle,
    as the acquisition arm does. Every encoder must fall to chance there; one that does not is
    scoring on the split's construction rather than on the chemistry, and its headline number
    means correspondingly less.
    """
    if control not in (None, "permuted"):
        raise ValueError(f"Unknown control arm {control!r}")
    features, outcomes, descriptors, vocabularies, manifest = load(landscape)
    if control == "permuted":
        outcomes = stream(seed, "permute-transfer").permutation(outcomes)
    flags = absence_flags(vocabularies, landscape)
    train_rows, test_rows, held, positions = holdout(features, vocabularies, held_count, seed)
    _, spec = _landscape(landscape)
    fraction = spec["top_fraction"]

    rows = []
    for encoder in ENCODERS:
        matrix, detail = encode(features, descriptors, flags, vocabularies, train_rows,
                                encoder, seed)
        for model in MODELS:
            predicted = fit(matrix, outcomes, train_rows,
                            model, stream_int(seed, "surrogate", encoder, model))
            held_truth, held_predicted = outcomes[test_rows], predicted[test_rows]
            order = test_rows[np.argsort(-held_predicted, kind="stable")[:budget]]
            rows.append({
                "encoder": encoder, "model": model, "columns": int(matrix.shape[1]),
                "representation": detail["kind"],
                "constant_columns": detail["constant_columns"],
                "held_out_spearman": round(float(spearmanr(held_predicted, held_truth).statistic), 4),
                # Scored on the training rows too, so an encoder failing to transfer can be
                # distinguished from one failing to fit at all.
                "seen_spearman": round(float(spearmanr(predicted[train_rows],
                                                       outcomes[train_rows]).statistic), 4),
                "selection": evaluate(held_truth,
                                      np.searchsorted(test_rows, order), fraction),
                "paired": paired_discrimination(features, outcomes, predicted, vocabularies,
                                                held, positions)})

    out = {"experiment": "monomer-transfer", "landscape": landscape, "control": control,
           "status": "complete", "seed": seed, "budget": budget, "top_fraction": fraction,
           "held_out_monomers": held, "held_out_count": len(held),
           "holdout_positions": positions,
           "train_members": int(len(train_rows)), "test_members": int(len(test_rows)),
           "library_members": int(len(features)),
           "descriptors": list(DESCRIPTORS), "results": rows,
           "dataset_content_sha256": manifest.get("content_sha256"),
           "display_config_sha256": digest(ROOT / "hpc_pilots/configs/display.json"),
           "provenance": provenance(),
           "endpoint_warning": arm("display").warning,
           "interpretation": "A monomer-wise split measures whether a surrogate can rank a "
                             "library member built from a building block it has never seen "
                             "measured. The tied-prediction rate is structural for a "
                             "one-hot encoding, not an empirical finding. No binding, "
                             "affinity, inhibition or antiviral property is predicted for "
                             "any member.",
           "limitations": [
               "Descriptors are per-monomer. They describe the building blocks and capture "
               "nothing about the assembled macrocycle, its conformation or the ring.",
               "The held-out monomers are drawn from one library against one target, so the "
               "comparison is within this landscape and is not a benchmark-breadth claim.",
               "Members are not independent replicates; they are draws from one combinatorial "
               "library measured in one campaign."]}
    suffix = f"_{control}" if control else ""
    output = ROOT / "runs/monomers" / f"{landscape}_held{len(held):02d}{suffix}.json"
    write_json(output, out)
    print(json.dumps({"output": str(output), "held_out_monomers": held,
                      "train_members": out["train_members"],
                      "test_members": out["test_members"],
                      "results": [{k: r[k] for k in ("encoder", "model", "held_out_spearman",
                                                     "seen_spearman")}
                                  | {"tied_rate": r["paired"]["tied_rate"],
                                     "top_recall": r["selection"]["top_recall"]}
                                  for r in rows]}, indent=2))
    return out


def run_monomers(landscape="thrombin_cyclic", counts=(2, 4, 6, 9, 12), seed=20261021,
                 controls=True, smoke=False):
    """Sweep the held-out monomer count, which is the axis the claim is about.

    One holdout size cannot distinguish a representation that transfers from one that
    happened to suit a split. The quantity to read is how each encoder degrades as more of the
    alphabet becomes unseen: a vocabulary model's structural ceiling does not move with it,
    because its tied-prediction rate is 1.0 at every size.
    """
    sizes = (2, 4) if smoke else counts
    runs = [transfer(landscape, count, seed) for count in sizes]
    if controls:
        runs += [transfer(landscape, max(sizes), seed, control="permuted")]
    summary = [{"held_out_count": r["held_out_count"], "control": r["control"],
                "test_members": r["test_members"],
                "by_encoder": {f"{x['encoder']}/{x['model']}":
                               {"held_out_spearman": x["held_out_spearman"],
                                "tied_rate": x["paired"]["tied_rate"],
                                "top_recall": x["selection"]["top_recall"]}
                               for x in r["results"]}}
               for r in runs]
    output = ROOT / "runs/monomers" / f"{landscape}_sweep.json"
    write_json(output, {"experiment": "monomer-transfer-sweep", "landscape": landscape,
                        "seed": seed, "sizes": list(sizes), "summary": summary,
                        "status": "complete", "provenance": provenance()})
    print(json.dumps({"output": str(output), "summary": summary}, indent=2))
    return summary
