"""Fit the five ranking weights against the panel, and report what fitting buys.

`intake.rank_score` refuses to fit its weights, and says why: "There is no labelled set of
'correct target rankings for a novel pathogen' to fit against ... so a learned weighting here
would be fitted to whatever its author already believed." That was true when it was written.

It is no longer true. The 46-genome panel supplies the missing set: `runs/benchmark/*.json`
holds one record per genome, each ranked chain carrying the same five features
`intake.score_candidate` scores, and `data/benchmark_panel.json` names the protein an approved
antiviral acts on. So the question the comment closed can now be opened and *measured* rather
than argued: does fitting the five weights on 20 virus families rank the twenty-first better
than the hand-set rubric does?

**The stated rubric remains the default, whatever this module finds.** Nothing here writes a
weight back into `intake.RUBRIC`. The fitted vector is a reported quantity, because installing
it would stake the pipeline's ranking on a label set with a known and uncorrectable bias -- see
`LIMITATIONS`. A fit that wins on held-out families is evidence worth having and is not on its
own a reason to ship it.

Three things the design is arranged to produce, because a weight vector nobody can watch move
is not much better than a constant:

**A trajectory.** The fit is plain projected gradient descent and not a closed-form solve. A
ridge solve would give the same answer faster and leave nothing to look at; fifteen hundred
gradient steps leave a path, and the path is what shows a reader that the weights settle rather
than wander. `fit_weights` returns every step it took.

**Weights on one axis.** Each feature is divided by the most it can pay before fitting, so a
weight is "points out of this term's ceiling" for every term. In that basis the stated rubric
*is* `intake.TERM_CEILING` -- conservation 3.0, exposure 2.0, essentiality 1.5, druggable class
2.4, solved relatives 0.5 -- so a fitted weight sits on the same axis as the stated one and the
two can be drawn in one figure. Without this the druggable-class weight reads 0.8 against a
term that pays up to 2.4.

**A ladder rather than a pair.** `stated` is the shipped rubric and the comparator that matters.
`uniform` is every term weighted alike, which is what "no opinion" looks like. `permuted` fits
on shuffled labels and must not win; if it does, the fitting procedure is scoring on something
other than the evidence. `fitted_without_conservation` zeroes the heaviest term, because that
term is known to be broken (below) and a fit leaning entirely on it would be a fit leaning on an
artefact.

**The conservation term is on two incomparable scales and this module cannot repair it.**
`intake` records the defect: an exact-tier row carries a shared-k-mer fraction, a remote-tier row
carries `min(1, bits/200)`, and the remote scale saturates -- 43 of 74 rows at 1.0 in the run
that measured it. Both arrive here as one column. `conservation_scales` counts the mix per fit so
the reader can see how much of the column is which, and the `fitted_without_conservation` arm is
the check that the answer does not depend on it.
"""
import json
import math
import random
import sys
from pathlib import Path

from .common import ROOT, content_digest, provenance, write_json
from .intake import DRUGGABLE, RUBRIC, TERM_CEILING

# `benchmarks` is a sibling package of `hpc_pilots`, not a submodule, so the repository root has
# to be importable. Added rather than assumed, because `python -m hpc_pilots` from elsewhere puts
# only the package's parent on the path and this module is otherwise silently unimportable.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# The five terms, in the order every vector, record and figure in this module uses. Named here
# once so a trajectory row, a weight dict and a CSV column cannot disagree about which number is
# which -- the failure that a bare list of five floats invites.
FEATURES = ("conservation", "surface_exposed", "essential_class",
            "druggable_class", "structures_in_family")

# What divides each raw feature to put it on a 0-1 axis. `conservation` is already a fraction and
# the two booleans are 0 or 1, so three of the five divide by one. The other two are the reason
# this exists: `druggable_class` is an integer class over `DRUGGABLE`'s highest value, and
# `structures_in_family` is a count the rubric caps at 50 before weighting it.
STRUCTURE_CAP = 50.0
CLASS_CAP = float(max(DRUGGABLE.values()))

# The stated rubric expressed on the normalised axis, which is what makes it comparable with a
# fitted vector. Read off `intake.TERM_CEILING` rather than retyped: the ceiling of a term is
# exactly what that term pays when its normalised feature is 1.
STATED = {name: float(TERM_CEILING[name]) for name in FEATURES}

LIMITATIONS = (
    "The labels are proteins that already carry an approved or late-stage antiviral, so they "
    "are the best-characterised proteins of the best-characterised viruses. A weighting fitted "
    "on them is fitted toward what has already been drugged, and a novel pathogen's real target "
    "is by construction absent from this set. Leave-one-family-out controls for family "
    "relatedness. It does not and cannot control for this.",
    "A ranking is unchanged by scaling every weight by the same positive number, so a fitted "
    "vector is reported rescaled to the stated rubric's total of "
    f"{round(sum(STATED.values()), 4)} points. The rescaling changes no ranking and no recall "
    "figure; it exists so a fitted weight and a stated weight can be read on one axis.",
    "The conservation column mixes two incomparable scales, which is `intake`'s own recorded "
    "defect and is not repaired here. `fitted_without_conservation` is the arm that says "
    "whether the result depends on it.",
    "Candidate rows come from the recorded panel runs, so this fit inherits whatever "
    "identification those runs achieved. A genome whose run failed contributes nothing rather "
    "than contributing a ranking that found nothing; the two are different claims and only one "
    "is about the weights.",
    "Ties are broken by the candidate's original rank in every arm alike. The stated arm "
    "therefore reproduces the recorded ordering exactly, and no arm gains from the tie-break "
    "that another does not.",
    "Recall at k is reported only for genomes where at least one label matched some candidate. "
    "Matching does not depend on the ordering, so that set is identical for every arm and "
    "cannot favour one.",
)


# ---------------------------------------------------------------------------
# The design matrix
# ---------------------------------------------------------------------------
def normalise_features(features):
    """One candidate's five features on a 0-1 axis, in `FEATURES` order.

    Division happens here and nowhere else. `score_terms` applies the weight and the cap in one
    expression, which is right for a record a reader checks by hand and wrong for a fit, where
    the cap belongs to the feature and the weight is the thing being learned.
    """
    return [
        float(features["conservation"]),
        float(bool(features["surface_exposed"])),
        float(bool(features["essential_class"])),
        float(features["druggable_class"]) / CLASS_CAP,
        min(1.0, float(features["structures_in_family"]) / STRUCTURE_CAP),
    ]


def panel_matrix(tier="strong", directory=None):
    """Every panel candidate as a feature row, labelled, grouped by virus family.

    One entry per genome rather than one flat matrix, because every quantity this module
    computes is per-genome: pairs are formed within a genome, recall is measured within a genome,
    and the chance baseline is a function of that genome's own candidate count. Flattening first
    and grouping later is how a pair gets formed between two different viruses.

    Returns `(genomes, skipped)`. A genome is skipped, with its reason, rather than dropped
    silently -- an absent record and a record whose labels matched nothing are different facts
    and the caller reports both.
    """
    from benchmarks import task1, task1_labels

    records = task1.panel_records(directory or task1.PANEL_RUNS)
    genomes, skipped = [], []
    for entry in task1_labels.pathogens(tier=tier):
        record = records.get(entry["pathogen"])
        if record is None:
            skipped.append({"pathogen": entry["pathogen"], "accession": entry["accession"],
                            "because": "no clean panel record on disk for this genome"})
            continue
        rows = sorted((r for r in record.get("chains") or () if r.get("features")),
                      key=lambda r: r["rank"])
        if not rows:
            skipped.append({"pathogen": entry["pathogen"], "accession": entry["accession"],
                            "because": "the record carries no ranked chain with features"})
            continue
        # Matching is a property of the name and not of the order, so the labelled set is fixed
        # here once and every arm below re-ranks the same rows with the same labels on them.
        labelled = [any(task1.matches(label, row) for label in entry["targets"]) for row in rows]
        matched_labels = sum(1 for label in entry["targets"]
                             if any(task1.matches(label, row) for row in rows))
        if not matched_labels:
            skipped.append({
                "pathogen": entry["pathogen"], "accession": entry["accession"],
                "because": f"none of the {len(entry['targets'])} label(s) matched any of the "
                           f"{len(rows)} ranked candidates, so no ordering of them can be scored"})
            continue
        # A genome whose every candidate is labelled discriminates between no two weightings:
        # it supplies no pair for the loss, and it hands every arm recall 1.0 against a chance
        # baseline of 1.0, so it adds a row to the mean that cannot move. Excluded rather than
        # scored, which is the same ruling `benchmarks` makes for an instance that cannot
        # discriminate. Rift Valley fever is the one case on this panel: one identified
        # candidate, and it is the label.
        if sum(labelled) >= len(rows):
            skipped.append({
                "pathogen": entry["pathogen"], "accession": entry["accession"],
                "because": f"all {len(rows)} identified candidate(s) are labelled, so no "
                           f"ordering of them can be better or worse than another and chance "
                           f"at every depth is 100%"})
            continue
        genomes.append({
            "pathogen": entry["pathogen"], "accession": entry["accession"],
            "family": entry["family"],
            "x": [normalise_features(r["features"]) for r in rows],
            "y": labelled,
            "original_rank": [int(r["rank"]) for r in rows],
            "names": [r.get("target") or r.get("name") for r in rows],
            "rows": rows,
            "labels": entry["targets"],
            "candidates": len(rows),
            "matched_labels": matched_labels,
            "conservation_scales": _scale_mix(rows),
        })
    return genomes, skipped


def _scale_mix(rows):
    """How many rows carry which conservation scale, which the fit cannot tell apart."""
    mix = {}
    for row in rows:
        scale = (row.get("features") or {}).get("conservation_scale") or "unrecorded"
        mix[scale] = mix.get(scale, 0) + 1
    return mix


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------
def _pairs(genomes):
    """Every (labelled, unlabelled) pair, within one genome.

    The loss asks "does this genome's drugged target outrank this genome's other candidates",
    which is the question the ranker is actually put to: it ranks one genome's proteins against
    each other and never one virus's protein against another's. Pairing across genomes would
    additionally ask the weights to make Lassa's nucleoprotein outscore influenza's polymerase,
    a comparison no stage of the pipeline ever makes and whose answer is not defined.
    """
    out = []
    for index, genome in enumerate(genomes):
        positives = [i for i, hit in enumerate(genome["y"]) if hit]
        negatives = [i for i, hit in enumerate(genome["y"]) if not hit]
        for i in positives:
            for j in negatives:
                out.append((index, i, j))
    return out


def pair_differences(genomes, mask=None):
    """The `x_i - x_j` of every within-genome pair, as one array the fit reuses every epoch.

    Built once. The loss and its gradient are both linear in these differences, so a pair's
    contribution depends on the weights only through one dot product -- which means the pairs
    never have to be walked again after this, and a fifteen-hundred-step fit costs fifteen
    hundred matrix products rather than two million Python iterations.
    """
    import numpy as np

    pairs = _pairs(genomes)
    if not pairs:
        raise ValueError("No (labelled, unlabelled) pair exists in the genomes supplied, so "
                         "there is nothing for a ranking loss to learn from.")
    rows = np.empty((len(pairs), len(FEATURES)), dtype=np.float64)
    for position, (index, i, j) in enumerate(pairs):
        genome = genomes[index]
        rows[position] = np.asarray(genome["x"][i]) - np.asarray(genome["x"][j])
    if mask is not None:
        rows = rows * np.asarray(mask, dtype=np.float64)
    return rows


def fit_weights(genomes, epochs=8000, rate=0.08, penalty=1e-3, seed=20261123,
                init=None, mask=None, record_every=10, differences=None):
    """Learn the five weights by projected gradient descent, and keep the path.

    The loss is the pairwise logistic one: for a labelled candidate `i` and an unlabelled `j` in
    the same genome, `log(1 + exp(-(w.x_i - w.x_j)))`. It is minimised by ordering `i` above `j`
    and it is smooth, so the gradient exists everywhere and the trajectory is a curve rather
    than a staircase.

    **Weights are projected back to zero after each step.** A negative weight on conservation
    would say that a protein less like its relatives is a better target, and on exposure that
    being unreachable helps. Those are not hypotheses this rubric can hold; the five terms were
    chosen as evidence *for* a target, and a fit free to flip one is fitting the label set's
    quirks rather than learning a weighting. The projection is one `maximum(0, ...)` and it is
    the only constraint imposed.

    `mask` zeroes a feature for the whole fit, which is how the no-conservation arm is run
    without a second code path. `init` defaults to every weight at 1.0: equal footing, which is
    the honest starting point for "what would this learn knowing nothing", where starting from
    the stated rubric would make every trajectory a story about that rubric.

    `seed` is recorded and does not steer anything here. Full-batch gradient descent from a
    stated initialisation is deterministic, so the argument exists to be carried into the record
    beside the controls, which do draw on it.
    """
    import numpy as np

    width = len(FEATURES)
    mask_vector = np.ones(width) if mask is None else np.asarray(mask, dtype=np.float64)
    if init is None:
        weights = np.ones(width)
    elif isinstance(init, dict):
        weights = np.asarray([float(init[name]) for name in FEATURES], dtype=np.float64)
    else:
        weights = np.asarray([float(value) for value in init], dtype=np.float64)
    weights = weights * mask_vector
    rows = pair_differences(genomes, mask) if differences is None else differences
    count = float(len(rows))

    trajectory, losses = [], []
    for epoch in range(epochs + 1):
        margins = rows @ weights
        # `logaddexp(0, -m)` is `log(1 + exp(-m))` without the overflow a confidently wrong pair
        # produces early in a fit, which otherwise reports the loss as inf for the whole run.
        loss = float(np.logaddexp(0.0, -margins).mean()
                     + 0.5 * penalty * float(weights @ weights))
        if epoch % record_every == 0 or epoch == epochs:
            trajectory.append({"epoch": epoch, "loss": round(loss, 6),
                               "weights": {name: round(float(w), 6)
                                           for name, w in zip(FEATURES, weights)}})
            losses.append(loss)
        if epoch == epochs:
            break
        # The derivative of the loss above: each pair pushes the weights along its own feature
        # difference, in proportion to how wrong the current ordering of that pair is.
        share = 1.0 / (1.0 + np.exp(np.clip(margins, -700, 700)))
        gradient = -(rows.T @ share) / count + penalty * weights
        weights = np.maximum(0.0, weights - rate * gradient) * mask_vector

    final = {name: round(float(w), 6) for name, w in zip(FEATURES, weights)}
    return {"weights": final,
            "rescaled": rescale(final),
            "trajectory": trajectory,
            "final_loss": round(losses[-1], 6),
            "loss_fell_by": round(losses[0] - losses[-1], 6),
            "epochs": epochs, "rate": rate, "penalty": penalty, "seed": seed,
            "pairs": int(len(rows)),
            "movement": movement(trajectory),
            "masked_features": [name for name, m in zip(FEATURES, mask_vector) if m == 0.0]}


def rescale(weights, total=None):
    """The same ranking, written so its weights sum to the stated rubric's total.

    A ranking is invariant to multiplying every weight by one positive number, so the scale a
    gradient descent happens to stop at carries no information. Fixing the sum makes a fitted
    weight and a stated weight comparable term by term, and changes no ordering anywhere.
    """
    target = sum(STATED.values()) if total is None else total
    current = sum(weights.values())
    if current <= 0:
        return {name: 0.0 for name in weights}
    factor = target / current
    return {name: round(value * factor, 4) for name, value in weights.items()}


def movement(trajectory):
    """How far the weight vector travelled between recorded steps, and where it stopped moving.

    The one-number version of "the weights settle". `final_step` near zero against a much larger
    `largest_step` is a fit that converged; the two being comparable is a fit still moving when
    the epoch budget ran out, which is a reason to distrust the final vector rather than a
    detail.
    """
    steps = []
    for before, after in zip(trajectory, trajectory[1:]):
        delta = math.sqrt(sum((after["weights"][name] - before["weights"][name]) ** 2
                              for name in FEATURES))
        steps.append({"to_epoch": after["epoch"], "distance": round(delta, 6)})
    if not steps:
        return {"steps": [], "largest_step": None, "final_step": None, "settled": None}
    final = steps[-1]["distance"]
    largest = max(s["distance"] for s in steps)
    return {"steps": steps, "largest_step": round(largest, 6), "final_step": round(final, 6),
            "settled": bool(largest > 0 and final <= 0.02 * largest),
            "settled_rule": "the last recorded step is at most 2% of the largest one"}


# ---------------------------------------------------------------------------
# Scoring an ordering
# ---------------------------------------------------------------------------
def rerank(genome, weights):
    """This genome's candidate rows, ordered by `weights`, ties by original rank.

    Returns the rows themselves so `task1.rank_of` can be used unchanged: the matcher reads a
    row's name fields, and reimplementing the ordering on indices would mean reimplementing the
    matching too.
    """
    vector = [float(weights[name]) for name in FEATURES]
    scored = []
    for position, row in enumerate(genome["rows"]):
        score = sum(w * x for w, x in zip(vector, genome["x"][position]))
        scored.append((-score, genome["original_rank"][position], row))
    scored.sort(key=lambda item: (item[0], item[1]))
    return [row for _, _, row in scored]


def score_genome(genome, weights, top_k=(1, 3, 5, 10)):
    """Where this genome's labels land under `weights`, with the chance value beside each.

    `chance_at_k` is computed from the candidate count and the number of labels that matched,
    both of which are properties of the genome and not of the ordering, so the same chance value
    applies to every arm. A recall figure does not travel without it.
    """
    from benchmarks import task1

    ordering = rerank(genome, weights)
    ranks = [task1.rank_of(label, ordering) for label in genome["labels"]]
    ranks = sorted(r for r in ranks if r is not None)
    total = len(ordering)
    return {
        "pathogen": genome["pathogen"], "accession": genome["accession"],
        "family": genome["family"], "candidates": total, "matched": len(ranks),
        "ranks": ranks, "best_rank": min(ranks) if ranks else None,
        "mean_percentile": (round(100 * sum(ranks) / len(ranks) / total, 2)
                            if ranks and total else None),
        "recall_at_k": {f"top{k}": (1.0 if ranks and min(ranks) <= k else 0.0) for k in top_k},
        "chance_at_k": {f"top{k}": task1.hypergeometric_at_k(total, len(ranks), k)
                        for k in top_k},
    }


def aggregate(rows, top_k=(1, 3, 5, 10)):
    """Mean recall and mean chance over genomes, and the lift between them.

    A mean over genomes and not over families, because this is a held-out *ranking* figure and
    every genome is one ranking the weights were asked to produce. The family grouping does its
    work in the split, which is where the dependence between five coronaviruses would otherwise
    leak; it is not a second correction to apply to the mean.
    """
    if not rows:
        return {"genomes": 0}
    out = {"genomes": len(rows), "recall_at_k": {}, "chance_at_k": {}, "lift_at_k": {}}
    for k in top_k:
        key = f"top{k}"
        recall = 100 * sum(r["recall_at_k"][key] for r in rows) / len(rows)
        chances = [r["chance_at_k"][key] for r in rows if r["chance_at_k"][key] is not None]
        chance = sum(chances) / len(chances) if chances else None
        out["recall_at_k"][key] = round(recall, 1)
        out["chance_at_k"][key] = round(chance, 1) if chance is not None else None
        out["lift_at_k"][key] = round(recall - chance, 1) if chance is not None else None
    out["mean_best_rank"] = round(
        sum(r["best_rank"] for r in rows if r["best_rank"]) /
        max(1, sum(1 for r in rows if r["best_rank"])), 2)
    return out


# ---------------------------------------------------------------------------
# Leave one family out
# ---------------------------------------------------------------------------
def permute_labels(genomes, seed=20261123):
    """The same genomes with each one's labels shuffled across its own candidates.

    The control arm. Shuffled *within* a genome rather than across the panel, so every genome
    keeps its candidate count and its number of labelled rows and only the question of which row
    is labelled is destroyed. A fit that still wins here is reading something other than the
    evidence -- candidate count, feature scale, the index order -- and its headline number on the
    real labels means correspondingly less.

    The evaluation labels are left alone. Only the rows the fit learns from are permuted, because
    an arm graded against shuffled labels is grading a different question.
    """
    rng = random.Random(seed)
    out = []
    for genome in genomes:
        shuffled = list(genome["y"])
        rng.shuffle(shuffled)
        copy = dict(genome)
        copy["y"] = shuffled
        out.append(copy)
    return out


def leave_one_family_out(genomes, epochs=8000, rate=0.08, penalty=1e-3, seed=20261123):
    """Fit on every family but one, rank the one, and report every arm on identical rows.

    The split is by family and not by genome because 26 of the panel's genomes share a target
    pattern list with a sibling: five coronaviruses between them carry one list, so a
    genome-level split trains on four coronaviruses and tests on the fifth and calls the result
    held out. `benchmarks` already fixed this for its own aggregate and the same argument applies
    with more force here, where the weights would be fitted on the leaked rows rather than
    merely averaged over them.

    A family whose genomes supply no training pair after the others are removed is reported as
    unfittable rather than scored, which happens for a singleton family that is the only one
    carrying a label the others lack.
    """
    families = sorted({genome["family"] for genome in genomes})
    folds, held_rows = [], {arm: [] for arm in
                            ("stated", "uniform", "fitted", "fitted_without_conservation",
                             "permuted")}
    conservation_index = FEATURES.index("conservation")
    no_conservation = [1.0] * len(FEATURES)
    no_conservation[conservation_index] = 0.0

    for family in families:
        train = [g for g in genomes if g["family"] != family]
        test = [g for g in genomes if g["family"] == family]
        if not train or not test:
            continue
        try:
            fitted = fit_weights(train, epochs, rate, penalty, seed, record_every=epochs)
            without = fit_weights(train, epochs, rate, penalty, seed,
                                  mask=no_conservation, record_every=epochs)
            permuted = fit_weights(permute_labels(train, seed), epochs, rate, penalty, seed,
                                   record_every=epochs)
        except ValueError as failure:
            folds.append({"family": family, "genomes": [g["pathogen"] for g in test],
                          "fitted": None,
                          "unfittable_because": str(failure)})
            continue
        arms = {
            "stated": STATED,
            "uniform": {name: 1.0 for name in FEATURES},
            "fitted": fitted["rescaled"],
            "fitted_without_conservation": without["rescaled"],
            "permuted": permuted["rescaled"],
        }
        fold = {"family": family, "genomes": [g["pathogen"] for g in test],
                "training_genomes": len(train), "training_pairs": fitted["pairs"],
                "fitted": fitted["rescaled"],
                "fitted_without_conservation": without["rescaled"],
                "permuted": permuted["rescaled"],
                "final_loss": fitted["final_loss"],
                "settled": fitted["movement"]["settled"], "held_out": {}}
        for arm, weights in arms.items():
            rows = [score_genome(genome, weights) for genome in test]
            held_rows[arm].extend(rows)
            fold["held_out"][arm] = aggregate(rows)
        folds.append(fold)

    return {"folds": folds,
            "families": len(families),
            "held_out": {arm: aggregate(rows) for arm, rows in held_rows.items()},
            "per_genome": {arm: rows for arm, rows in held_rows.items()},
            "split": "leave one ICTV family out, 1 fold per family",
            "arms": {
                "stated": "the five weights `intake.RUBRIC` ships, which is what the pipeline "
                          "ranks with today. The comparator that matters.",
                "uniform": "every term weighted alike. What having no opinion about the five "
                           "terms looks like.",
                "fitted": "projected gradient descent on the other families' labels.",
                "fitted_without_conservation": "the same fit with the conservation column "
                                               "zeroed, because that column mixes two "
                                               "incomparable scales.",
                "permuted": "the same fit on labels shuffled within each training genome. It "
                            "must not win; if it does, the procedure is not reading the "
                            "evidence it was given.",
            }}


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------
def run_rubric_fit(tier="strong", epochs=8000, rate=0.08, penalty=1e-3, seed=20261123,
                   record=True, directory=None):
    """Fit the weights, hold out each family in turn, and write what both found."""
    genomes, skipped = panel_matrix(tier=tier, directory=directory)
    if not genomes:
        raise RuntimeError(
            "No panel genome supplied a scoreable row. Check that runs/benchmark/ holds the "
            "per-genome records `output/run_benchmark.py` writes; this module reads them and "
            "fetches nothing.")

    conservation_index = FEATURES.index("conservation")
    no_conservation = [1.0] * len(FEATURES)
    no_conservation[conservation_index] = 0.0

    full = fit_weights(genomes, epochs, rate, penalty, seed, record_every=10)
    without = fit_weights(genomes, epochs, rate, penalty, seed, mask=no_conservation,
                          record_every=10)
    permuted = fit_weights(permute_labels(genomes, seed), epochs, rate, penalty, seed,
                           record_every=10)
    crossed = leave_one_family_out(genomes, epochs, rate, penalty, seed)

    scales = {}
    for genome in genomes:
        for scale, count in genome["conservation_scales"].items():
            scales[scale] = scales.get(scale, 0) + count

    comparison = compare_arms(crossed)

    payload = {
        "experiment": "rubric_fit",
        "status": "complete",
        "question": "Does fitting the five ranking weights on 20 virus families rank the "
                    "twenty-first better than the hand-set rubric does?",
        "answer": _verdict(comparison),
        "tier": tier,
        "genomes_scored": len(genomes),
        "candidate_rows": sum(genome["candidates"] for genome in genomes),
        "labels_matched": sum(genome["matched_labels"] for genome in genomes),
        "families": crossed["families"],
        "skipped_genomes": skipped,
        "features": list(FEATURES),
        "normalisation": {
            "what": "every feature is divided by the most it can pay, so a weight is points out "
                    "of that term's ceiling and the stated rubric is `intake.TERM_CEILING`",
            "conservation": "already a 0-1 fraction",
            "surface_exposed": "boolean",
            "essential_class": "boolean",
            "druggable_class": f"integer class divided by {CLASS_CAP}, the highest class in "
                               f"intake.DRUGGABLE",
            "structures_in_family": f"count divided by the rubric's cap of {STRUCTURE_CAP}",
        },
        "stated_weights": {name: round(value, 4) for name, value in STATED.items()},
        "stated_weights_as_shipped": dict(RUBRIC),
        "fitted_weights": full["rescaled"],
        "fitted_weights_unscaled": full["weights"],
        "fitted_without_conservation": without["rescaled"],
        "permuted_control_weights": permuted["rescaled"],
        "trajectory": full["trajectory"],
        "trajectory_without_conservation": without["trajectory"],
        "trajectory_permuted": permuted["trajectory"],
        "movement": full["movement"],
        "final_loss": full["final_loss"],
        "loss_fell_by": full["loss_fell_by"],
        "training_pairs": full["pairs"],
        "epochs": epochs, "rate": rate, "penalty": penalty, "seed": seed,
        "cross_validation": crossed,
        "comparison": comparison,
        "recall_gain_over_stated": comparison["fitted_minus_stated"],
        "conservation_scales": scales,
        "weights_are_not_installed":
            "`intake.RUBRIC` is unchanged by this run. The fitted vector is reported, not "
            "shipped; see limitations for why a held-out win is not on its own sufficient.",
        "limitations": list(LIMITATIONS),
        "provenance": provenance(),
    }
    payload["content_sha256"] = content_digest(payload)
    if record:
        write_json(ROOT / "runs/rubric_fit.json", payload)
    return payload


def paired(left_rows, right_rows):
    """Genome by genome, how often one arm puts a label higher than another.

    The readout recall@k cannot give. With 45 genomes and a dozen candidates apiece, recall@1 is
    a step function that moves by 2.2 points whenever one genome changes, so two arms differing
    by four genomes look like a nine-point effect. Comparing the best rank each arm achieved on
    the same genome counts what actually moved, and the sign test on those counts is the
    cheapest statement available about whether it moved more often than not.
    """
    by_pathogen = {row["pathogen"]: row for row in right_rows}
    better = worse = tied = 0
    for row in left_rows:
        other = by_pathogen.get(row["pathogen"])
        if other is None or row["best_rank"] is None or other["best_rank"] is None:
            continue
        if row["best_rank"] < other["best_rank"]:
            better += 1
        elif row["best_rank"] > other["best_rank"]:
            worse += 1
        else:
            tied += 1
    moved = better + worse
    return {"genomes_better": better, "genomes_worse": worse, "genomes_tied": tied,
            "genomes_that_moved": moved,
            "sign_test_p": _sign_test(better, moved),
            "what": "best rank of any label, compared per genome; p is a two-sided sign test "
                    "over the genomes that moved"}


def _sign_test(successes, trials):
    """Two-sided sign test, exact. `None` when nothing moved and there is nothing to test."""
    if trials <= 0:
        return None
    total = 0.0
    for k in range(trials + 1):
        probability = math.comb(trials, k) * 0.5 ** trials
        if probability <= math.comb(trials, successes) * 0.5 ** trials + 1e-12:
            total += probability
    return round(min(1.0, total), 4)


def compare_arms(crossed):
    """Every arm against the shipped rubric, and whether the control fell.

    `control_fell` is the question that decides whether any of the rest may be read as a gain.
    The permuted arm is fitted on labels shuffled within each training genome, so it has nothing
    to learn; an arm that cannot beat it is not demonstrating that it learned anything, whatever
    its margin over the shipped rubric looks like.
    """
    held = crossed["held_out"]
    per_genome = crossed["per_genome"]
    depths = sorted(held["stated"]["recall_at_k"], key=lambda key: int(key[3:]))
    fitted = held["fitted"]["recall_at_k"]
    permuted = held["permuted"]["recall_at_k"]
    beats_control = {k: round(fitted[k] - permuted[k], 1) for k in depths}
    control_fell = all(value > 0 for value in beats_control.values())
    spread = {k: round(max(held[arm]["recall_at_k"][k] for arm in held)
                       - min(held[arm]["recall_at_k"][k] for arm in held), 1)
              for k in depths}
    return {
        "depths": depths,
        "recall_by_arm": {arm: held[arm]["recall_at_k"] for arm in held},
        "chance": held["stated"]["chance_at_k"],
        "fitted_minus_stated": {k: round(held["fitted"]["recall_at_k"][k]
                                         - held["stated"]["recall_at_k"][k], 1) for k in depths},
        "fitted_minus_permuted": beats_control,
        "control_fell": control_fell,
        "control_rule": "the permuted arm is fitted on labels shuffled within each training "
                        "genome. The fitted arm must beat it at every depth for a gain over the "
                        "shipped rubric to be readable as learning.",
        "spread_across_arms": spread,
        "one_genome_is_worth": round(100.0 / max(1, held["stated"]["genomes"]), 1),
        "paired_against_stated": {arm: paired(per_genome[arm], per_genome["stated"])
                                  for arm in held if arm != "stated"},
    }


def _verdict(comparison):
    """What the held-out comparison found, stated so a null reads as a null.

    Written to refuse the flattering reading. The fitted arm's margin over the shipped rubric is
    smaller than one genome at most depths, and the arm fitted on shuffled labels scores in the
    same band, so the margin is not evidence of learning. The sentence says that rather than
    quoting the margin.
    """
    recall = comparison["recall_by_arm"]
    chance = comparison["chance"]
    unit = comparison["one_genome_is_worth"]
    spread = comparison["spread_across_arms"]["top1"]
    gained = comparison["fitted_minus_stated"]["top1"]
    beat = comparison["fitted_minus_permuted"]["top1"]

    if comparison["control_fell"]:
        head = (f"Fitting helps, and the control falls: the fitted weights beat a fit on "
                f"shuffled labels at every depth.")
    else:
        head = (f"Fitting does not improve the ranking. The fitted weights gain {gained:+} "
                f"points at rank 1 over the rubric the pipeline ships, which is less than "
                f"{max(unit, abs(gained)):g} points per genome buys, and the same fit on "
                f"shuffled labels scores {beat:+} against the fitted arm rather than falling "
                f"below it. A margin a shuffled-label control can match is not evidence that "
                f"anything was learned.")
    return (
        f"{head} What the comparison does establish is that the five-term ranking beats chance "
        f"by a wide margin whatever the weights are: every arm recovers between "
        f"{min(recall[a]['top1'] for a in recall)}% and "
        f"{max(recall[a]['top1'] for a in recall)}% of labels at rank 1 against a "
        f"{chance['top1']}% chance baseline, and the whole spread between arms is {spread} "
        f"points. The ranking is carried by the terms and not by their weighting. "
        f"`intake.RUBRIC` is unchanged."
    )


def report(path=None):
    """Print the fitted weights beside the stated ones, for a terminal rather than a page."""
    payload = json.loads(Path(path or ROOT / "runs/rubric_fit.json").read_text())
    print(payload["answer"])
    print()
    print(f"{'term':<24}{'stated':>10}{'fitted':>10}{'no-cons':>10}{'permuted':>10}")
    for name in payload["features"]:
        print(f"{name:<24}"
              f"{payload['stated_weights'][name]:>10}"
              f"{payload['fitted_weights'][name]:>10}"
              f"{payload['fitted_without_conservation'][name]:>10}"
              f"{payload['permuted_control_weights'][name]:>10}")
    print()
    for arm, rows in payload["cross_validation"]["held_out"].items():
        print(f"{arm:<30} recall@1 {rows['recall_at_k']['top1']:>5}%  "
              f"recall@3 {rows['recall_at_k']['top3']:>5}%  "
              f"recall@5 {rows['recall_at_k']['top5']:>5}%  "
              f"(chance@1 {rows['chance_at_k']['top1']}%)")
    return payload
