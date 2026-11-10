"""The whole onset chain on one genome accession, with every stage's input and output kept.

**Why this module exists.** Each stage of the onset arm was already runnable and already
recorded, but each was recorded *separately* and against a different pathogen: the ranking in
`runs/intake_validation.json`, the surface in `runs/surface_markers.json` over a hand-picked
six, the fold in `runs/folded_structure/`, the library in the chemistry registry. A reader
asking the obvious question -- put one genome in, what does each stage hand the next? -- had
to assemble the answer from four files that do not share a pathogen, and could not see where
one stage's output is in fact *not* the next stage's input. This runs the chain once, on one
accession, and writes down what entered and what left at every step.

**The second reason is a dropped measurement.** The surface stage had no caller anywhere in
the package: `intake.enrich_with_surface` existed and nothing invoked it, so the only surfaces
ever measured were the six named by hand in the gate record. SARS-CoV-2's RNA-directed RNA
polymerase ranks fifth of twenty-three candidates on the genome the project leads with, and
had no surface measurement at all -- not refused, not reported as unmeasurable, simply never
asked about. A stage that silently covers a subset of what the stage before it produced is
indistinguishable, from the outside, from a stage that covers all of it.

So the surface stage here is exhaustive over its input by construction. Every ranked candidate
gets a row, and a row carries either a measurement or the reason there is none, which is a
finding about the target (no deposited structure covering the chain) or about the machine (no
SASA backend). `candidates_in` equals `rows_out` in the record, and a test pins that.

**What this is not.** It is not a new pipeline. Every stage here calls the same function the
registered tool dispatches, so a number in this record is the number that tool returns and not
a second implementation of it. The chain is the composition, written down.
"""
import re
import time

from .common import ROOT, provenance, write_json


# Stage 5 is the one place the chain does not actually connect, and it is recorded as absent
# rather than drawn as present. `chemistry.library_spec` takes an initiator, a randomised-region
# length range and a set of elongator classes. It does not take a patch area, a residue range,
# an accession or anything else measured upstream, so the library this arm defines is the same
# library whatever the surface stage found. The project's prose says the library is defined
# "over that surface"; today that describes the intent and not a data path. Naming the gap in
# the record is the only honest way to show a flowchart of this arm, because the arrow a reader
# would otherwise assume is the arrow that does not exist.
LIBRARY_READS_FROM_SURFACE = ()


def _decomposed_score(features, rubric):
    """The rank score split into its five terms, so a number can be read rather than trusted.

    `intake.rank_score` returns one float. Every term in it is a stated weight times a feature
    already in the row, so the decomposition is recoverable exactly -- and the point of
    recovering it is that the two highest-scoring SARS-CoV-2 candidates differ by 0.0016 and
    arrive there by opposite routes: the spike earns 2.0 for being on the outside of the virion
    and 1.386 for conservation, the main protease earns 0.0 and 2.504. A single score hides
    that. The sum is asserted against the recorded score rather than assumed.
    """
    terms = {
        "conservation": round(rubric["conservation"] * features["conservation"], 4),
        "surface_exposed": round(rubric["surface_exposed"]
                                 * float(features["surface_exposed"]), 4),
        "essential_class": round(rubric["essential_class"]
                                 * float(features["essential_class"]), 4),
        "structures_in_family": round(rubric["structures_in_family_capped"]
                                      * min(1.0, features["structures_in_family"] / 50.0), 4),
        "druggable_class": round(rubric["druggable_class"]
                                 * features["druggable_class"], 4),
    }
    return {"terms": terms, "total": round(sum(terms.values()), 4)}


def surface_rows(ranked, replicates=5, folds=()):
    """One surface row per ranked candidate, measured or refused with the reason.

    Exhaustive over `ranked` on purpose, and the `top=12` cap that `enrich_with_surface`
    applies is not repeated here. That cap is a cost control on the stage that costs seconds
    per structure, and on a viral genome of twenty-three candidates it drops nothing a reader
    would accept being dropped -- it dropped the polymerase for a year by dropping nothing
    visible.

    A row is never absent. `measured` is False with `unmeasured_because` naming which of the
    three situations obtained, because "this protein has no surface a macrocycle could lie
    across" and "no deposited structure covers this chain" and "this machine has no freesasa"
    are three different statements and only the first is about the target.

    `folds` is every fold the stage before this one made, and carrying them here is what makes
    that stage a stage this one reads. Where a chain has coordinates, the same patch arithmetic
    is run on them and reported as `predicted_surface`, under its own keys and never merged
    into `measured`: for a chain no deposition covers it is the only surface that will ever
    exist, and for a chain a deposition does cover it is a second number from a different kind
    of evidence. Where both exist the ratio between them is recorded, because that is the one
    place this pipeline can say how far a predicted surface sits from a measured one instead of
    hoping they agree.
    """
    from .geometry import surface_for_accession
    from .intake import amenable, chain_locations, exposed_from_locations

    # Keyed on the target name, which is what both stages key a row by. A fold that returned an
    # absence rather than coordinates is dropped here, so a chain with no usable prediction is
    # indistinguishable from a chain that was never folded -- both simply have no
    # `predicted_surface`, and the fold row already says which happened and why.
    coordinates = {f["target_name"]: f for f in (folds or ())
                   if f and f.get("pdb") and f.get("target_name")}

    rows = []
    for row in ranked:
        name = row.get("name")
        accession = row.get("nearest_homolog")
        span = row.get("chain_range")
        interval = {"start": span[0], "end": span[1]} if span else None
        entry = {"target": name, "rank": row.get("rank"),
                 "identified": bool(row.get("identified")),
                 "measured_on_accession": accession,
                 "chain_range": list(span) if span else None,
                 "measured": False, "unmeasured_because": None}
        if not entry["identified"] or not accession:
            entry["unmeasured_because"] = (
                "no reviewed homolog accession, so there is no deposited structure to resolve. "
                "A sequence-only candidate reaches the folding stage and not this one.")
            rows.append(entry)
            continue
        started = time.time()
        try:
            markers = surface_for_accession(accession, interval=interval,
                                            replicates=replicates)
        except ImportError as missing:
            entry["unmeasured_because"] = (f"no SASA backend on this machine: {missing}. "
                                           "A property of the environment, not of the target.")
            rows.append(entry)
            continue
        entry["seconds"] = round(time.time() - started, 2)
        if not markers or markers.get("surface_score") is None:
            entry["unmeasured_because"] = ((markers or {}).get("error")
                                           or "no deposited chain covers this residue range")
            entry["structures_offered"] = (markers or {}).get("structures_offered") or []
            entry["chains_mapped_to_accession"] = (markers or {}).get(
                "chains_mapped_to_accession")
            rows.append(entry)
            continue
        entry.update({
            "measured": True,
            "patch_median_a2": markers["hydrophobic_patch_area"],
            "patch_range_a2": [markers["patch_area_min"], markers["patch_area_max"]],
            "patch_spread_a2": markers["patch_area_spread"],
            "structures_measured": markers["structures_measured"],
            "structures_used": markers["structures_used"],
            "surface_score": markers["surface_score"],
            "patch_fraction_of_surface": markers["patch_fraction_of_surface"],
            "structures_clipped_to_domain": markers.get("structures_clipped_to_domain"),
            "residues_in_range": markers.get("residues_in_range"),
            "exposed_fraction_of_residues": markers["exposed_fraction"],
            "sequons_near_patch": markers["sequons_near_patch"],
            "available_from": markers.get("available_from"),
        })
        # Localisation scoped to this chain, never inherited from the polyprotein. The gate
        # reads it, and reading it off the entry is what once made a cytosolic protease
        # report as surface-exposed through its membrane-bound siblings.
        described = (row.get("features", {}).get("chain_description") or name or "")
        scoped = chain_locations(accession, described)
        verdict = exposed_from_locations(scoped["locations"])
        entry["locations"] = scoped["locations"][:4]
        entry["location_scope"] = scoped["scope"]
        entry["exposed"] = verdict["exposed"]
        entry["exposed_fraction_of_locations"] = verdict["fraction"]
        features = dict(row.get("features", {}))
        features.update({"hydrophobic_patch_area": markers["hydrophobic_patch_area"],
                         "patch_area_min": markers["patch_area_min"],
                         "patch_area_max": markers["patch_area_max"],
                         "patch_fraction_of_surface": markers["patch_fraction_of_surface"],
                         "surface_exposed": verdict["exposed"]})
        entry["gate"] = amenable(features)
        rows.append(entry)

    # After every deposited attempt, not during them. A row reaches this loop by one of four
    # routes -- measured, no accession, no backend, no covering chain -- and the predicted
    # surface has something different to say on each of them. Attaching it here is what lets
    # it say which: `only_route_to_a_surface` is read off the finished row rather than guessed
    # at from where in the loop the row happened to be.
    for entry in rows:
        folded = coordinates.get(entry["target"])
        if folded:
            entry["predicted_surface"] = _predicted_surface(folded["pdb"], folded, entry)
    return rows


def _predicted_surface(coordinates, folded, entry):
    """The patch on predicted coordinates, with its standing attached to it.

    Everything here is flagged twice over -- `is_deposited_measurement` false, `in_the_ranking`
    false -- because the number is arithmetic on a model's assertion and the two ways it could
    be misused are being counted as a solved structure and being fed to the rubric. The rubric
    takes five weights fixed before the run and no surface term is among them, so the second
    cannot happen by accident; the first is a reader's error and the flags exist for them.

    `ratio_to_deposited_median` is the honest part. On a chain that has both, this is the only
    number in the pipeline that says how far a predicted surface sits from a measured one. It
    is reported without an explanation attached: a monomer prediction has no partner to bury
    an interface against, and prediction error is not separable from that here, so stating a
    cause would be inventing one.
    """
    from .geometry import markers_from_coordinates

    started = time.time()
    try:
        markers = markers_from_coordinates(
            coordinates, label=f"{folded.get('model') or 'predicted'} · {entry['target']}")
    except ImportError as missing:
        return {"measured": False,
                "not_measured_because": f"no SASA backend on this machine: {missing}. "
                                        "A property of the environment, not of the target.",
                "is_deposited_measurement": False, "in_the_ranking": False,
                "only_route_to_a_surface": not entry["measured"]}
    if markers.get("error"):
        return {"measured": False, "not_measured_because": markers["error"],
                "is_deposited_measurement": False, "in_the_ranking": False,
                "only_route_to_a_surface": not entry["measured"]}

    deposited = entry.get("patch_median_a2")
    patch = markers["hydrophobic_patch_area"]
    return {
        "measured": True,
        "measured_on": "coordinates from the fold stage, not a deposition",
        "model": folded.get("model"),
        "mean_plddt": folded.get("mean_plddt"),
        "confident_fraction": folded.get("confident_fraction"),
        "residues": markers["residues"],
        "patch_a2": patch,
        "patch_fraction_of_surface": markers["patch_fraction_of_surface"],
        "exposed_fraction_of_residues": markers["exposed_fraction"],
        "sequons_near_patch": markers["sequons_near_patch"],
        "seconds": round(time.time() - started, 2),
        # True exactly when no deposited chain covered this target, which is the case the
        # reordering exists for: the novel target whose only coordinates are predicted ones.
        "only_route_to_a_surface": not entry["measured"],
        "deposited_patch_median_a2": deposited,
        "ratio_to_deposited_median": (round(patch / deposited, 2)
                                      if deposited else None),
        "is_deposited_measurement": False,
        "in_the_ranking": False,
        "caveat": "A patch computed on predicted coordinates. It inherits the prediction's "
                  "error, a monomer's surface is not necessarily exposed in an oligomer, and "
                  "over a low-confidence stretch it describes a fold the model did not commit "
                  "to. Reported beside a deposited measurement and never pooled with one.",
    }


def _refusal_sentence(measured):
    """Whether localisation alone accounts for every refusal, checked rather than asserted."""
    refused = [r for r in measured if r["gate"] == "no"]
    if not refused:
        return "No row is refused outright."
    on_location = [r for r in refused if not r["exposed"]]
    would_pass = [r["target"] for r in refused if r["gate_on_area_alone"] == "yes"]
    if len(on_location) == len(refused):
        text = (f"Localisation, which the gate reads before area, decides all "
                f"{len(refused)} refusals.")
    else:
        text = (f"Localisation decides {len(on_location)} of {len(refused)} refusals; the "
                f"rest fail on area.")
    if would_pass:
        text += (f" {len(would_pass)} of them would pass on patch area alone: "
                 + ", ".join(sorted(would_pass)) + ".")
    return text


def _rdrp_sentence(measured):
    """What the target that was missing turns out to measure, stated against the others."""
    row = next((r for r in measured if r["target"] == "RdRp (nsp12)"), None)
    if row is None:
        return "RdRp was not measured in this run."
    wider = sorted((r for r in measured if r["patch_spread"] > row["patch_spread"]),
                   key=lambda r: -r["patch_spread"])
    rank = (f"the widest spread of the {len(measured)} measured" if not wider else
            f"the {len(wider) + 1}{'st' if len(wider) == 0 else 'nd' if len(wider) == 1 else 'rd' if len(wider) == 2 else 'th'}"
            f" widest spread of the {len(measured)} measured, behind "
            + ", ".join(r["target"] for r in wider))
    return (f"RdRp measures a patch of {row['patch_median']} square angstroms over "
            f"{row['structures_measured']} structures of the same domain, with a spread of "
            f"{row['patch_spread']} -- {rank}, and wider than the "
            f"{600.0:.0f} square-angstrom footprint the gate thresholds on. Its gate verdict "
            f"is {row['gate']}, decided on localisation: the polymerase is inside the cell. "
            f"That is a usable answer and it was not available before, which is the whole "
            f"cost of the omission.")


def _ceiling_effect(measured):
    """Whether the ceiling changed any verdict in this run, counted rather than claimed.

    A rule that closes a hole nothing currently falls into is still worth having and is not
    worth overstating, and the difference between those two sentences is a count.
    """
    moved = [r["target"] for r in measured if r["gate"] != r["gate_without_ceiling"]]
    if not moved:
        refused_first = sum(1 for r in measured if not r["exposed"])
        return (f"None of the {len(measured)} measured rows changes verdict. The oversized "
                f"patches in this set belong to proteins the gate refuses on localisation "
                f"before it reaches area at all -- {refused_first} of {len(measured)} are "
                f"annotated inside the cell -- so the ceiling closes a hole that nothing in "
                f"this config currently falls into. It is not decorative: the chain record's "
                f"nsp4 reports 7,558 square angstroms over a quarter of its own surface, and "
                f"that row is only refused today because nsp4 is a membrane protein.")
    return (f"{len(moved)} of {len(measured)} rows change verdict: " + ", ".join(sorted(moved)))


def gate_table(replicates=5, record=True):
    """The macrocycle-amenability gate over every target `configs/structures.json` declares.

    **Why it is generated from the config and not from a list.** The gate record this replaces
    carried six targets, chosen by hand to span the modality contrast. Six was defensible as a
    contrast set and indefensible as coverage: SARS-CoV-2's RNA-directed RNA polymerase was not
    among them, and nothing in the record said so. A reader could not tell a target the gate
    had refused from a target the gate had never been pointed at, which is the one distinction
    a gate exists to make.

    So the input is now the config, every entry in it gets a row, and `targets_declared` equals
    `rows` in the output. Adding a target to the config adds it here, and a target absent from
    the gate is now absent from the config, where its absence is visible.

    The measurement is unchanged: the largest contiguous exposed hydrophobic patch, median over
    up to `replicates` deposited structures of the same domain, with the range beside it, and
    localisation read per mature chain rather than off the polyprotein entry.
    """
    from .geometry import surface_for_accession
    from .intake import (PATCH_FRACTION_CEILING, amenable, chain_locations,
                         exposed_from_locations)
    from .structures_evidence import chains, config, entry, ranges

    spec = config()
    rows, entries = [], {}
    for target, declared in spec["targets"].items():
        accession = declared["accession"]
        if accession not in entries:
            entries[accession] = entry(accession)
        row = {"target": target, "accession": accession,
               "clinical_modality": declared.get("clinical_modality"),
               "modality_agent": declared.get("modality_agent"),
               "measured": False, "unmeasured_because": None}
        try:
            intervals = ranges(declared, entries[accession])
        except ValueError as bad:
            row["unmeasured_because"] = f"the declared range does not resolve: {bad}"
            rows.append(row)
            continue
        # The first interval, and the record says so. An obligate two-chain complex has two,
        # and a patch measured on one of them is not the interface's patch -- which is a
        # narrower claim than the row would otherwise make.
        interval = {"start": intervals[0]["start"], "end": intervals[0]["end"]}
        row["interval"] = interval
        row["interval_is_one_of"] = len(intervals)
        markers = surface_for_accession(accession, interval=interval, replicates=replicates)
        if not markers or markers.get("surface_score") is None:
            # Two different absences, told apart rather than merged. "PDBe maps no chain at
            # all to this accession" is a fact about which entry the config points at;
            # "chains exist and none covers the range" is a fact about the domain. Influenza
            # neuraminidase is the first: the protein has hundreds of deposited structures
            # and none of them maps to P03470, the A/Puerto Rico/8/1934 entry this config
            # names, so the target's own caveat about subtypes becomes a measurement.
            mapped = len(chains(accession))
            row["chains_mapped_to_accession"] = mapped
            row["unmeasured_because"] = (
                (f"PDBe maps no deposited chain to {accession} at all, so the range cannot "
                 f"be covered by anything. The protein may be well solved under another "
                 f"accession; this entry is not it." if mapped == 0 else
                 (markers or {}).get("error") or
                 f"{mapped} chains map to {accession} and none covers "
                 f"{interval['start']}-{interval['end']}"))
            rows.append(row)
            continue
        described = next((want.get("description") for want in declared["ranges"]
                          if want.get("description")), target)
        scoped = chain_locations(accession, described)
        verdict = exposed_from_locations(scoped["locations"])
        features = {"hydrophobic_patch_area": markers["hydrophobic_patch_area"],
                    "patch_area_min": markers["patch_area_min"],
                    "patch_area_max": markers["patch_area_max"],
                    "patch_fraction_of_surface": markers["patch_fraction_of_surface"],
                    "surface_exposed": verdict["exposed"]}
        row.update({
            "measured": True,
            "location_scope": scoped["scope"], "locations": scoped["locations"][:4],
            "exposed": verdict["exposed"], "exposed_fraction": verdict["fraction"],
            "patch_median": markers["hydrophobic_patch_area"],
            "patch_range": [markers["patch_area_min"], markers["patch_area_max"]],
            "patch_spread": markers["patch_area_spread"],
            "structures_measured": markers["structures_measured"],
            "structures_used": markers["structures_used"],
            "structures_clipped_to_domain": markers.get("structures_clipped_to_domain"),
            "patch_fraction_of_surface": markers["patch_fraction_of_surface"],
            "residues_in_range": markers.get("residues_in_range"),
            "available_from": markers.get("available_from"),
            "gate": amenable(features),
            "gate_on_area_alone": amenable({**features, "surface_exposed": True}),
            # The verdict at three ceilings, for the reason the contact cutoff is reported at
            # three: a threshold that produces the answer is worth being suspicious of, and
            # what matters is whether the verdict moves when it does.
            "gate_at_ceiling": {str(c): amenable(features, max_patch_fraction=c)
                                for c in (0.10, 0.15, 0.20)},
            # What the row would have read with no ceiling at all, so the record can say
            # whether adding one changed any verdict instead of implying that it did.
            "gate_without_ceiling": amenable(features, max_patch_fraction=1.01)})
        rows.append(row)

    measured = [r for r in rows if r["measured"]]
    widest = max(measured, key=lambda r: r["patch_spread"]) if measured else None
    payload = {
        "experiment": "amenability-gate-v3", "status": "complete",
        "question": "Over every target the structures config declares -- not a hand-picked "
                    "subset -- does the macrocycle-amenability gate separate anything?",
        "answer": "It separates clear refusals from everything else, and the reason it cannot "
                  "do more is now measured on more targets rather than fewer. The patch moves "
                  + (f"by up to {widest['patch_spread']:.0f} square angstroms across "
                     f"structures of one domain ({widest['target']}), against a footprint of "
                     f"{600.0:.0f}" if widest else "") +
                  ", so most rows are borderline and that is the honest report. "
                  + _refusal_sentence(measured),
        "fix_3_nothing_dropped": {
            "problem": "v2 measured six targets named in a script. SARS-CoV-2's RdRp was not "
                       "one of them while ranking fifth of the pipeline's own candidates for "
                       "the genome this project leads with, and the record did not say it was "
                       "absent. An unasked target and a refused target looked the same.",
            "fix": "The gate is generated from `configs/structures.json`, every declared "
                   "target gets a row, and an unmeasurable one carries the reason. "
                   "`targets_declared` equals `rows`.",
            "finding": _rdrp_sentence(measured),
        },
        "footprint_a2": 600.0,
        "patch_fraction_ceiling": PATCH_FRACTION_CEILING,
        "fix_4_the_gate_had_no_ceiling": {
            "problem": "The band had a floor at half a footprint and nothing above, so the "
                       "larger the patch the more confidently the gate read `yes`. The "
                       "largest patches in a viral genome are percolated surfaces and "
                       "membrane proteins measured out of their bilayer, and neither is a "
                       "site a macrocycle can be put on.",
            "fix": "A patch covering more than `patch_fraction_ceiling` of the domain's own "
                   "accessible surface is reported borderline rather than yes. The quantity "
                   "is a fraction and not an area because the failure is percolation, which "
                   "scales with the chain.",
            "sensitivity": "Every row carries `gate_at_ceiling` at 0.10, 0.15 and 0.20, and "
                           "`gate_without_ceiling` beside it.",
            "effect_here": _ceiling_effect(measured),
        },
        "fix_5_the_patch_search_is_clipped_to_the_domain": {
            "problem": "The residue interval selected which structure to measure and then "
                       "did not restrict the measurement, so the quantity was the largest "
                       "patch anywhere on the deposited chain. For a construct larger than "
                       "the target that is not the target's patch.",
            "fix": "The patch search and the surface it is reported as a fraction of are "
                   "confined to the domain. Accessibility is still computed on the whole "
                   "deposited file, partners included, because a domain cut out of its chain "
                   "exposes a face that was never solvent-facing.",
            "numbering": "The clip needs the chain's author numbering. PDBe's mapping "
                         "endpoint publishes it for some entries and null for others, so "
                         "where it is absent the offset is inferred from the structure's own "
                         "residues against the reference sequence and accepted only above "
                         "0.9 agreement. `structures_clipped_to_domain` says how many of "
                         "each row's replicates the clip actually reached.",
        },
        "targets_declared": len(spec["targets"]),
        "rows": len(rows),
        "nothing_dropped": len(rows) == len(spec["targets"]),
        "measured": len(measured),
        "unmeasured": len(rows) - len(measured),
        "targets": rows,
        "limitations": [
            "Twelve targets, declared in a config to span the modality contrast rather than "
            "sampled from anything.",
            "The borderline band is wide once measurement spread is propagated, so the gate "
            "separates clear negatives from everything else rather than sorting the middle.",
            "No validation against measured macrocycle binding exists, because no such "
            "labelled set exists. Clinical modality is a proxy and a loose one, which is why "
            "each label names its agent.",
            "A target whose declared ranges are two -- the nsp10/nsp16 interface -- is "
            "measured on the first, so its row is about one chain and not the interface."],
        "provenance": provenance()}
    if record:
        write_json(ROOT / "runs/amenability_gate.json", payload)
    return payload


def _homolog_chain(accession, span, cache):
    """The homolog's own residues over the annotated range, which the range cuts correctly."""
    if not accession or not span:
        return None
    if accession not in cache:
        try:
            cache[accession] = _entry_sequence(accession)
        except Exception:                                 # noqa: BLE001
            cache[accession] = None
    sequence = cache[accession]
    return sequence[span[0] - 1:span[1]] if sequence else None


def _entry_sequence(accession):
    from .intake import _entry

    return _entry(accession)["sequence"]["value"]


def locate_chain(orf, chain, aligner=None):
    """Where the homolog's mature chain sits in this genome's own translation.

    **The question a slice cannot answer.** Mature-chain boundaries are annotated on the
    homolog, and under a holdout the homolog is a different species, so the pathogen's own
    numbering is not the relative's. SARS-CoV's main protease begins at 3241 of its
    polyprotein and SARS-CoV-2's at 3264; slicing one genome's translation at the other's
    coordinates yields 306 residues, the correct length, starting 23 residues into the wrong
    window. A length check accepts that. Comparing the slice against the relative's residues
    rejects it, which is better and still leaves the chain unfolded.

    Aligning finds it. The relative's chain is a sequence, the pathogen's ORF translation is a
    sequence, and a local alignment reports where in the second the first lies -- in the
    pathogen's own numbering, across a species boundary, with insertions and deletions
    absorbed rather than shifting everything after them.

    Returns the span, the sequence, how much of the relative's chain the alignment accounted
    for, and the identity over the aligned columns. Coverage is the acceptance criterion and
    identity is reported but not thresholded: identity is low across genera by construction --
    SARS-CoV-2's spike is about 46 per cent identical to bat coronavirus HKU3's, which is a
    real relative and a correct assignment -- so thresholding it would refuse exactly the
    distant relatives the holdout exists to force the pipeline through.
    """
    if not orf or not chain:
        return None
    from .intake import _aligner

    aligner = aligner or _aligner()
    try:
        alignment = aligner.align(orf, chain)[0]
    except Exception:                                     # noqa: BLE001
        return None
    blocks = alignment.aligned
    if len(blocks) < 2 or not len(blocks[0]):
        return None
    start, end = int(blocks[0][0][0]), int(blocks[0][-1][1])
    matched = sum(int(b - a) for a, b in blocks[0])
    identical = 0
    for (ts, te), (qs, qe) in zip(blocks[0], blocks[1]):
        identical += sum(1 for x, y in zip(orf[int(ts):int(te)], chain[int(qs):int(qe)])
                         if x == y)
    return {"span": [start + 1, end], "sequence": orf[start:end],
            "covered": round(matched / len(chain), 4),
            "identity": round(identical / matched, 4) if matched else 0.0}


# How much of the relative's mature chain a local alignment has to account for before the
# window it points at is taken as that chain in this genome. Half is a low bar for a real
# correspondence and a high one for a chance local hit on a BLOSUM62 alignment of a few
# hundred residues, and the record carries `covered` on every row so a reader can see where
# each one fell rather than only which side of the bar it was.
MINIMUM_CHAIN_COVERAGE = 0.5


def fold_stage(ranked, ceiling=400, pool=(), limit=12, budget=240.0):
    """Every ranked chain whose own sequence the chain can locate, folded.

    Returns `(primary, rows, folds)`. `primary` is the first fold in pick order and is the
    one the record and the page report in full; `folds` is every fold this stage made, which
    the surface stage reads. `limit` caps the number of folding requests one run will spend
    and a chain it drops is named in the record rather than quietly left out.

    **Why this is not simply "fold the top candidate".** The folding endpoint takes an
    amino-acid sequence, and the only pathogen sequence this chain holds is the six-frame
    translation of the genome. The mature-chain boundaries, meanwhile, are annotated on the
    *homolog's* entry, so cutting a chain out means finding where the relative's chain sits in
    this genome's translation.

    Slicing at the relative's coordinates was the first attempt and it fails two ways. It
    fails loudly when the numbering does not reach -- `chain_range` is numbered on polyprotein
    1ab while the ORF that matched is a shorter piece of it, so the polymerase's range selects
    a handful of residues or none. And it fails quietly when the numbering does reach and is
    wrong, which is the dangerous one: SARS-CoV's main protease begins at 3241 and
    SARS-CoV-2's at 3264, so one genome's translation sliced at the other's coordinates gives
    306 residues, the correct length, starting 23 residues into the wrong window.

    `locate_chain` aligns instead, so the span comes back in the pathogen's own numbering,
    across a species boundary, with indels absorbed.

    **And it searches every called ORF, not only the one that matched.** A coronavirus
    polyprotein is translated as two overlapping ORFs through a ribosomal frameshift, so the
    chains past the shift are genuinely not in the ORF whose match named them -- nsp12 to
    nsp16 are in ORF1b while the hit that identified them is ORF1a. Searching only the parent
    is what left the polymerase, the helicase and the two methyltransferases unfoldable; they
    are in the genome, in a frame the row does not point at. Each chain is aligned against
    every ranked ORF and the best-covered hit wins, with the ORF it was found in recorded.
    """
    from .folded_structure import fold
    from .intake import _aligner

    aligner, rows, pick, parents = None, [], None, {}
    for row in ranked:
        # Unidentified candidates are folded too, and used to be skipped here. The folding
        # endpoint takes an amino-acid sequence and nothing else -- no accession, no relative,
        # no annotation -- so it is the one stage of this chain that works perfectly well on a
        # protein nothing recognises, and it was the stage refusing to look at one. `intake`
        # carries the six-frame translation on every candidate for exactly this reason, so the
        # sequence was already on the row; the `continue` meant the only lane open to a
        # genuinely novel target was closed by the stage that needed least from it.
        #
        # They cannot reach the pick ahead of an identified chain: the pick runs in two passes
        # below, identified first. So every recorded run folds what it folded before.
        identified = bool(row.get("identified"))
        span, orf = row.get("chain_range"), (row.get("orf") or {}).get("protein") or ""
        declared = row.get("length_aa")
        entry = {"target": row.get("name") or "unidentified ORF",
                 "rank": row.get("rank"),
                 "identified": identified,
                 "identification_route": row.get("identification_route"),
                 "chain_range_on_the_relative": list(span) if span else None,
                 "declared_length_aa": declared,
                 "parent_orf_length_aa": len(orf) or None,
                 "sequence_recovered": False, "not_recovered_because": None}
        if not span:
            # A whole ORF rather than a chain of one, so the translation *is* the sequence.
            if orf:
                entry.update({"sequence_recovered": True, "recovered_length_aa": len(orf),
                              "recovered_sequence": orf,
                              "recovered_by": "the ORF translation is the candidate"})
            else:
                entry["not_recovered_because"] = "no translated ORF on the row"
            rows.append(entry)
            continue

        reference = _homolog_chain(row.get("nearest_homolog"), span, parents)
        if not reference:
            entry["not_recovered_because"] = (
                "the relative's own residues over this range could not be read, so there is "
                "nothing to look for in this genome's translation.")
            rows.append(entry)
            continue
        entry["reference_chain_length_aa"] = len(reference)
        aligner = aligner or _aligner()

        # The parent ORF first so a chain found there keeps the simplest provenance, then
        # every other ranked ORF, because a frameshifted product is in the genome and in
        # another frame.
        searched, best = [], None
        for label, candidate in [("the ORF that matched", orf)] + list(pool):
            if not candidate or candidate in searched:
                continue
            searched.append(candidate)
            located = locate_chain(candidate, reference, aligner)
            if located and (best is None or located["covered"] > best[1]["covered"]):
                best = (label, located, candidate)
        entry["orfs_searched"] = len(searched)
        if best is None:
            entry["not_recovered_because"] = (
                f"no local alignment of the relative's {len(reference)}-residue chain against "
                f"any of the {len(searched)} ranked ORFs returned a span.")
            rows.append(entry)
            continue

        label, located, source = best
        entry.update({"found_in": label,
                      "span_in_this_genome": located["span"],
                      "recovered_length_aa": len(located["sequence"]),
                      "chain_covered_by_alignment": located["covered"],
                      "identity_over_aligned_columns": located["identity"]})
        if located["covered"] >= MINIMUM_CHAIN_COVERAGE:
            entry.update({"sequence_recovered": True,
                          "recovered_sequence": located["sequence"],
                          "recovered_by": "local alignment of the relative's mature chain "
                                          f"against {label}, covering "
                                          f"{located['covered']:.0%} of it at "
                                          f"{located['identity']:.0%} identity"})
        else:
            entry["not_recovered_because"] = (
                f"the best alignment over {len(searched)} ranked ORFs accounted for only "
                f"{located['covered']:.0%} of the relative's {len(reference)}-residue chain, "
                f"under the {MINIMUM_CHAIN_COVERAGE:.0%} this stage requires, so the window "
                f"it points at is not that chain in this genome.")
        rows.append(entry)

    # Two passes, identified first. An unidentified ORF is foldable and must be reachable,
    # but it must never displace a named chain: the committed main-protease fold is what the
    # repository reproduces byte for byte, and a one-pass loop would hand the pick to whatever
    # happened to sort first. Explicit, so the ordering is a decision and not an accident of
    # how `ranked` was sorted.
    foldable = []
    for want_identified in (True, False):
        for entry in rows:
            if entry["identified"] is want_identified and entry["sequence_recovered"] \
                    and 0 < entry["recovered_length_aa"] <= ceiling:
                foldable.append((entry["target"], entry["recovered_sequence"], entry))
    if foldable:
        pick = foldable[0][:2]
    # Carried only to choose and fold; a row holding a whole protein sequence would multiply
    # the record's size by the number of candidates for no reader's benefit. Captured above
    # first, because every chain inside the ceiling is folded now and not only the pick.
    for entry in rows:
        entry.pop("recovered_sequence", None)

    if pick is None:
        return {"absence": "no ranked candidate yielded a sequence within the residue "
                           f"ceiling of {ceiling}"}, rows, []

    # Every chain inside the ceiling, not just the first one, and the reason is the stage after
    # this one. The surface stage can measure a prediction, so a chain's coordinates are the
    # difference between a surface and no surface -- and the chains that need that most are
    # exactly the ones this stage used to skip. Folding only the pick meant the lane fired on
    # whichever chain happened to rank highest inside the ceiling, which on every genome
    # recorded here was a well-studied enzyme that already had five deposited structures: a
    # comparison, never a rescue. On the SARS-CoV-2 holdout, 5 of the 11 chains no deposition
    # covers are inside the ceiling, and all 5 were being left without coordinates.
    #
    # `structures_in_family` is on the row and is NOT used to choose. It counts structures
    # across the homolog family while the surface stage resolves one accession, and the two
    # disagree in the direction that matters: the membrane protein carries 4 in the family and
    # nothing deposited against the accession this run landed on, so a proxy would have
    # dropped exactly the chain it was meant to find. This stage cannot know which chains the
    # next one will fail on, so it stops guessing and folds all of them.
    #
    # The first fold is the pick, in the order above, so the committed main-protease fold stays
    # the fold this repository reproduces byte for byte.
    # Two bounds, because one folding request is not one bounded cost. `REQUEST_TIMEOUT` is 45
    # seconds and the lane retries once, so a single chain can spend ninety seconds against a
    # flaky host -- twelve of those is nineteen minutes, against the twenty-five minute cap on
    # the workflow that serves a visitor's run. A count alone would have made the slow case a
    # timeout instead of a record. The wall budget is checked before each fold after the first:
    # the pick always runs, because it is the fold this repository reproduces.
    folds, dropped_limit, dropped_budget = [], 0, 0
    folding_started = time.time()
    for index, (target, sequence, entry) in enumerate(foldable):
        if limit is not None and index >= limit:
            dropped_limit = len(foldable) - index
            break
        if index and budget is not None and time.time() - folding_started >= budget:
            dropped_budget = len(foldable) - index
            break
        try:
            payload = fold(sequence, target_name=target or "", record=False)
        except Exception as failure:                      # noqa: BLE001
            payload = {"absence": f"{type(failure).__name__}: {str(failure)[:160]}",
                       "target_name": target}
        # Stated on the row as well as in the payload, so a reader of `foldable` can see which
        # chains have coordinates without holding the fold payloads beside it.
        entry["folded"] = bool(payload.get("folded"))
        entry["mean_plddt"] = payload.get("mean_plddt")
        if payload.get("absence"):
            entry["fold_absence"] = (payload["absence"].get("reason")
                                     if isinstance(payload["absence"], dict)
                                     else payload["absence"])
        folds.append(payload)
    # Never silently. A bound that drops a chain is a chain with no coordinates, which is
    # the defect this whole change exists to remove, so each is named in the record.
    if dropped_limit:
        folds[0].setdefault("chains_not_folded_by_limit", dropped_limit)
    if dropped_budget:
        folds[0].setdefault("chains_not_folded_by_budget", dropped_budget)
        folds[0].setdefault("folding_budget_seconds", budget)
    return folds[0], rows, folds


# ------------------------------------------------------------------ evidence on the top few
# The five tools the chain itself never calls, and why. Each is reachable from the four facts
# stage 1 already produces -- a name, a reviewed accession, a residue range and a sequence --
# so capability was never the obstacle. The obstacle is the declared cost. `chief` compares a
# tool's declared worst case against the wall budget and refuses it BEFORE dispatch, without
# making a request: families.entries declares 300s and bioactivity.prior_binders declares
# 600s, against a chain whose whole run is eighty-six seconds. Measured live, the five cost
# roughly 210s for one target, so running them over all twenty ranked chains is about seventy
# minutes and cannot fit the twenty-five minute cap on the workflow that serves a visitor.
#
# So they run here, over the top few targets only, after the ranking is already final.
# NOTHING THEY RETURN CHANGES A RANK. The rubric is five stated weights fixed before the run
# and none of these is among them; this stage annotates the shortlist and may not reorder it.
# A tool that refuses still gets a row, for the same reason a chain that cannot be measured
# still gets one: a step may refuse, it may not silently skip.
EVIDENCE_TOOLS = (
    ("literature.search",         "Europe PMC"),
    ("similarity.homologs",       "EBI HMMER"),
    ("families.entries",          "InterPro"),
    ("structures.predicted",      "AlphaFold DB, 3D-Beacons"),
    ("bioactivity.prior_binders", "ChEMBL, IEDB, MaveDB"),
)

# Large enough to admit the 600s worst case bioactivity.prior_binders declares. This is the
# budget the gate checks a DECLARATION against, not a promise about elapsed time.
EVIDENCE_BUDGET_SECONDS = 700.0

# Keys that are long, repeated per row, or an echo of the request rather than a result.
_EVIDENCE_NOISE = frozenset((
    "experiment", "status", "interpretation", "provenance", "query", "sequence",
    "accession", "subject", "target_name", "requested_ranges", "uniprot_entry_id"))


def _evidence_summary(value):
    """A few headline scalars from a tool return, never the payload.

    Each tool returns a different shape, so this is deliberately generic: scalars are kept as
    they are and anything list-shaped is reduced to how many of it there were. That is enough
    to say what came back without this record becoming a copy of five external databases.
    """
    if not isinstance(value, dict):
        return {}
    out = {}
    for key, item in value.items():
        if key in _EVIDENCE_NOISE or key.startswith("_"):
            continue
        if isinstance(item, bool) or isinstance(item, (int, float)):
            out[key] = item
        elif isinstance(item, str) and len(item) <= 60:
            out[key] = item
        elif isinstance(item, (list, tuple)):
            out[f"{key}_count"] = len(item)
        if len(out) >= 10:
            break
    return out


def _evidence_arguments(tool_name, row, sequence):
    """What each tool is asked, built only from facts the chain already holds."""
    accession = row.get("nearest_homolog")
    span = row.get("chain_range")
    if tool_name == "literature.search":
        return {"subject": row.get("target")}
    if tool_name == "similarity.homologs":
        return {"query": sequence} if sequence else None
    if tool_name == "families.entries":
        return {"accession": accession} if accession else None
    if tool_name == "structures.predicted":
        if not accession:
            return None
        args = {"accession": accession}
        if span:
            # A tuple of "start-end" STRINGS. The tool rejects a pair outright -- "this tool
            # declares one form on purpose; a pair that happens to unpack is a" coincidence
            # it refuses to rely on -- and passing (start, end) silently cost six AlphaFold
            # lookups the first time this ran.
            args["residue_ranges"] = (f"{int(span[0])}-{int(span[1])}",)
        return args
    if tool_name == "bioactivity.prior_binders":
        return {"accession": accession, "target_name": row.get("target")} if accession else None
    return None


def evidence_rows(scored, top=3, budget=EVIDENCE_BUDGET_SECONDS):
    """Call all five off-chain tools on the highest-ranked targets, and record every call.

    Returns one row per target, each carrying one entry per tool: the status the gate or the
    endpoint produced, what it cost, and a short summary of what came back. A refusal is a
    recorded outcome, not an omission.
    """
    from . import chief
    from .tools import tool as tool_spec

    rows, cache = [], {}
    for row in sorted(scored, key=lambda r: r.get("rank") or 10 ** 6)[:top]:
        sequence = None
        if row.get("nearest_homolog") and row.get("chain_range"):
            try:
                sequence = _homolog_chain(row["nearest_homolog"], row["chain_range"], cache)
            except Exception:
                sequence = None
        calls = []
        for tool_name, endpoint in EVIDENCE_TOOLS:
            arguments = _evidence_arguments(tool_name, row, sequence)
            if arguments is None:
                calls.append({"tool": tool_name, "endpoint": endpoint, "status": "not_asked",
                              "note": "the chain holds no argument this tool requires",
                              "seconds": None, "network_calls": None, "returned": {}})
                continue
            started = time.time()
            try:
                with chief.counting_requests() as tally:
                    spec = tool_spec(tool_name)
                    chief.schema_conformance(spec, arguments)
                    envelope = chief.call(spec, arguments, context=chief.context(
                        None, now=0, wall_budget_seconds=budget))
                calls.append({
                    "tool": tool_name, "endpoint": endpoint,
                    "status": envelope.get("status"),
                    "note": envelope.get("note") or None,
                    "seconds": round(time.time() - started, 2),
                    "network_calls": tally["made"],
                    "returned": _evidence_summary(envelope.get("value"))})
            except Exception as broke:
                # An endpoint failure is already an envelope; this is the unanticipated one,
                # and it is written down rather than raised so one bad lane cannot lose the
                # other four.
                calls.append({"tool": tool_name, "endpoint": endpoint, "status": "error",
                              "note": f"{type(broke).__name__}: {broke}"[:200],
                              "seconds": round(time.time() - started, 2),
                              "network_calls": None, "returned": {}})
        rows.append({"rank": row.get("rank"), "target": row.get("target"),
                     "nearest_homolog": row.get("nearest_homolog"),
                     "chain_range": row.get("chain_range"),
                     "sequence_available": bool(sequence), "calls": calls})
    return rows


def evidence_stage(scored, top=3, budget=EVIDENCE_BUDGET_SECONDS):
    """`evidence_rows`, plus the stage entry that goes in the record beside stages 1 to 5."""
    started = time.time()
    rows = evidence_rows(scored, top=top, budget=budget)
    calls = [c for r in rows for c in r["calls"]]
    ok = [c for c in calls if c["status"] == "ok"]
    stage = {
        "stage": 6, "name": "Evidence on the highest-ranked targets",
        "tool": " + ".join(name for name, _ in EVIDENCE_TOOLS),
        "endpoints": ["Europe PMC", "EBI HMMER", "InterPro", "AlphaFold DB", "3D-Beacons",
                      "ChEMBL", "IEDB", "MaveDB"],
        "input": {"targets_in": len(rows), "ranked_available": len(scored),
                  "wall_budget_seconds": budget},
        "output": {"targets_covered": len(rows), "calls_made": len(calls),
                   "ok": len(ok), "not_ok": len(calls) - len(ok),
                   "network_calls": sum(c["network_calls"] or 0 for c in calls),
                   "changes_any_rank": False,
                   "every_target_has_every_tool":
                       all(len(r["calls"]) == len(EVIDENCE_TOOLS) for r in rows)},
        "seconds": round(time.time() - started, 2),
        "hands_to_next": "nothing. This annotates the shortlist and never reorders it"}
    return stage, rows


def _case(accession, exclude_organism, top, replicates, fold, label, evidence_top=3):
    """Every stage of the onset arm, once, against one reference."""
    from . import chemistry
    from .intake import decompose, rank_score, run_intake

    stages, clock = [], time.time()

    started = time.time()
    intake = run_intake(accession, exclude_organism=exclude_organism, top=top, record=False)
    stages.append({
        "stage": 1, "name": "Genome to candidate proteins", "tool": "intake.rank_targets",
        "endpoints": ["NCBI efetch", "UniProt"],
        "input": {"genome_accession": accession,
                  "reference_excluded_organism": exclude_organism},
        "output": {"genome_length_nt": intake["genome_length_nt"],
                   "segments": intake["segments"],
                   "orfs_found": intake["orfs_found"],
                   "orfs_ranked": intake["orfs_ranked"],
                   "reference_entries": intake["reference_entries"],
                   "reference_entries_removed": intake["reference_entries_removed"],
                   "identified": sum(1 for r in intake["targets"] if r.get("identified")),
                   "unidentified": intake["unidentified"]},
        "seconds": round(time.time() - started, 2),
        "hands_to_next": "one reviewed accession and one residue range per candidate"})

    started = time.time()
    ranked = decompose(intake["targets"])
    rubric = intake["rubric"]
    scored = []
    for position, row in enumerate(ranked, 1):
        if not row.get("identified"):
            continue
        row["rank"] = position
        split = _decomposed_score(row["features"], rubric)
        # The decomposition has to reproduce the score the ranker computed, or it is a second
        # rubric dressed as an explanation of the first.
        recomputed = rank_score(row["features"])
        scored.append({
            "rank": position, "target": row.get("name"),
            "parent": row.get("parent"), "tier": row.get("tier"),
            "nearest_homolog": row.get("nearest_homolog"),
            "chain_range": row.get("chain_range"),
            "conservation": row["features"]["conservation"],
            "surface_exposed": row["features"]["surface_exposed"],
            "essential_class": row["features"]["essential_class"],
            "druggable_class": row["features"]["druggable_class"],
            "structures_in_family": row["features"]["structures_in_family"],
            "score_terms": split["terms"],
            "score": recomputed,
            "decomposition_reproduces_score": abs(split["total"] - recomputed) < 5e-4})
    stages.append({
        "stage": 2, "name": "Polyproteins to mature chains, and the score for each",
        "tool": "intake.decompose + intake.rank_score",
        "endpoints": ["UniProt"],
        "input": {"candidates_in": len(intake["targets"])},
        "output": {"chains_out": len(scored), "rubric": rubric,
                   "all_decompositions_reproduce_the_score":
                       all(r["decomposition_reproduces_score"] for r in scored)},
        "seconds": round(time.time() - started, 2),
        "hands_to_next": "every chain, with the accession and range its surface is measured on"})

    # Folding runs BEFORE the surface stage, and the order is the pipeline's answer to the
    # case it exists for. A deposited structure is what the surface stage would rather
    # measure, and for a genuinely novel pathogen there is none: PDBe maps nothing to the
    # chain, so the only coordinates that will ever exist for it are the ones a model emits
    # here. Measuring first and folding afterwards made the novel case unanswerable by
    # construction -- the stage that could have produced a surface ran after the stage that
    # needed one, and `hands_to_next` said in as many words that nothing downstream read it.
    #
    # The two are still different quantities and the record keeps them apart: a deposition is
    # an experiment someone performed, a prediction is a model's assertion, and the ratio
    # between the patches they yield is measured on this run rather than assumed (see
    # `surface_rows`). Neither is in the ranking.
    # Counted from the input list and not from the output, so `candidates_in` is a fact about
    # what these stages were handed rather than a restatement of what they returned. A stage
    # that reports its own output as its input can never be caught covering less than it was
    # given.
    handed = [r for r in ranked if r.get("identified")]

    folded, fold_rows, folds = None, None, []
    if fold:
        started = time.time()
        pool = [(f"ORF {i} of the ranked set, {len(t['orf']['protein'])} aa", t["orf"]["protein"])
                for i, t in enumerate(intake["targets"], 1)
                if (t.get("orf") or {}).get("protein")]
        folded, fold_rows, folds = fold_stage(ranked, pool=pool)
        stages.append({
            "stage": 3, "name": "Every chain inside the ceiling to coordinates, the one lane a novel target has",
            "tool": "structures.folded", "endpoints": ["ESM Atlas"],
            # This stage is handed every candidate, not only the identified ones, because
            # folding needs a sequence and nothing else. Both counts are stated so the
            # denominator is visible: `candidates_in` is what it was given, and
            # `identified_in` is the subset the earlier stages could name.
            "input": {"candidates_in": len(ranked),
                      "identified_in": len(handed)},
            "output": {"foldable": sum(1 for r in fold_rows
                                       if r["sequence_recovered"] and r["identified"]),
                       "not_foldable": sum(1 for r in fold_rows
                                           if not r["sequence_recovered"]
                                           and r["identified"]),
                       # The novel lane, reported separately rather than summed in: an
                       # unidentified ORF has no name to fold "for", and counting it beside a
                       # located mature chain would make the two look like the same result.
                       "unidentified_foldable": sum(1 for r in fold_rows
                                                    if r["sequence_recovered"]
                                                    and not r["identified"]),
                       # How many chains actually have coordinates, which is the number
                       # the surface stage can act on. `folded_target` names the pick
                       # because the page and the committed artifact report that one in
                       # full; the others are named in `also_folded`.
                       "chains_folded": sum(1 for f in folds if f.get("folded")),
                       "folds_attempted": len(folds),
                       "folds_refused": sum(1 for f in folds if not f.get("folded")),
                       "also_folded": [f.get("target_name") for f in folds[1:]
                                       if f.get("folded")],
                       "chains_not_folded_by_limit":
                           (folded or {}).get("chains_not_folded_by_limit", 0),
                       "chains_not_folded_by_budget":
                           (folded or {}).get("chains_not_folded_by_budget", 0),
                       "folded_target": (folded or {}).get("target_name"),
                       "residues": (folded or {}).get("sequence_length"),
                       "mean_plddt": (folded or {}).get("mean_plddt"),
                       "at_or_above_70": (folded or {}).get("confident_fraction"),
                       "sequence_sha256": (folded or {}).get("sequence_sha256"),
                       "error": (folded or {}).get("absence")},
            "seconds": round(time.time() - started, 2),
            "hands_to_next": "coordinates for every chain inside the ceiling. The surface "
                             "stage measures them with the same arithmetic it uses on a "
                             "deposition, and reports the two apart"})

    started = time.time()
    rows = surface_rows(handed, replicates=replicates, folds=folds)
    measured = [r for r in rows if r["measured"]]
    # Attempted and succeeded are different counts, and only the second may be reported as a
    # measurement. A predicted surface can fail for the same reason a deposited one can -- no
    # SASA backend, nothing readable in the coordinates -- and counting the attempt would put
    # a refusal in the column that says how many chains got an answer.
    attempted = [r for r in rows if r.get("predicted_surface")]
    predicted = [r for r in attempted if r["predicted_surface"]["measured"]]
    stages.append({
        "stage": 4, "name": "Every chain to a measured surface, or a stated reason",
        "tool": "geometry.surface_markers",
        "endpoints": ["PDBe", "RCSB", "UniProt"],
        # Two inputs now, and the second is the arrow this stage used not to have. The
        # coordinate count is stated even when it is zero, so a run where the fold produced
        # nothing is distinguishable from a run where the surface stage ignored what it was
        # handed.
        "input": {"candidates_in": len(handed),
                  "predicted_coordinates_in": sum(1 for f in folds if f.get("pdb"))},
        "output": {"rows_out": len(rows), "measured": len(measured),
                   "unmeasured": len(rows) - len(measured),
                   # The check Marta's review asked for, as an assertion in the record
                   # rather than as a claim in prose: the surface stage emits exactly as
                   # many rows as the stage before it emitted chains.
                   "nothing_dropped": len(rows) == len(scored),
                   # Counted apart from `measured`, which stays a count of depositions. A
                   # prediction measured is not a structure solved, and a single total would
                   # make the two indistinguishable at exactly the point a reader is deciding
                   # how much the number is worth.
                   "predicted_coordinates_offered": len(attempted),
                   "measured_on_predicted_coordinates": len(predicted),
                   "predicted_was_the_only_route": sum(
                       1 for r in predicted
                       if r["predicted_surface"]["only_route_to_a_surface"]),
                   "gate_yes": sum(1 for r in measured if r.get("gate") == "yes"),
                   "gate_borderline": sum(1 for r in measured
                                          if r.get("gate") == "borderline"),
                   "gate_no": sum(1 for r in measured if r.get("gate") == "no")},
        "seconds": round(time.time() - started, 2),
        "hands_to_next": "a patch median and range per chain; nothing downstream reads it yet"})

    started = time.time()
    spec = chemistry.library_spec()
    stages.append({
        "stage": 5, "name": "A library definition over the randomised region",
        "tool": "chemistry.library_spec", "endpoints": ["local tables"],
        "input": {"reads_from_upstream": list(LIBRARY_READS_FROM_SURFACE)},
        "output": {"initiator": spec["initiator"], "closure_residue": spec["closure_residue"],
                   "random_region": spec["random_region"],
                   "alphabet_size": len(spec["elongator_alphabet"]),
                   "theoretical_diversity": spec["theoretical_diversity"]},
        "seconds": round(time.time() - started, 2),
        "hands_to_next": "the position count the go/no-go test in arm 2 reads"})

    evidence = None
    if evidence_top:
        stage, evidence = evidence_stage(scored, top=evidence_top)
        stages.append(stage)

    return {
        "case": label,
        "genome_accession": accession,
        "reference_excluded_organism": exclude_organism,
        "stages": stages,
        "scores": scored,
        "surfaces": rows,
        "library": spec,
        "folded": {k: folded[k] for k in ("target_name", "sequence_length",
                                          "sequence_sha256", "mean_plddt",
                                          "confident_fraction", "pdb_sha256", "absence")
                   if folded and k in folded} if folded else None,
        # The other chains folded this run, in the same shape and under a key of their own.
        # `folded` stays the pick alone because the page, the committed artifact and every
        # record already on disk report that one, and widening it would have made this change
        # unreadable against them. The digest is carried per chain so a reader can tell two
        # predictions apart; the coordinates themselves are not, for the reason the primary's
        # are not -- 193KB per chain would be most of the record.
        "folded_also": [{k: f[k] for k in ("target_name", "sequence_length",
                                           "sequence_sha256", "mean_plddt",
                                           "confident_fraction", "pdb_sha256", "absence")
                         if k in f} for f in folds[1:]] or None,
        "foldable": fold_rows if fold else None,
        "evidence": evidence,
        "seconds": round(time.time() - clock, 2),
        "coverage": {
            # The one table a reader should be able to check without reading any prose: how
            # many rows each stage emitted against how many it was handed.
            "chains_ranked": len(scored),
            "surface_rows": len(rows),
            "surface_measured": len(measured),
            "surface_refused": len(rows) - len(measured),
            "surface_covers_every_chain": len(rows) == len(scored),
            # Kept out of `surface_measured` deliberately. These are patches computed on the
            # fold stage's coordinates, and the count that matters most is the second one:
            # how many chains got a surface they could not have had from any deposition.
            "surface_on_predicted_coordinates": len(predicted),
            "surface_predicted_was_the_only_route": sum(
                1 for r in predicted if r["predicted_surface"]["only_route_to_a_surface"]),
            # The number this change is for: chains with some surface, deposited or predicted,
            # against chains with none. A chain over the folding ceiling and absent from PDBe
            # is still in the second group, and stating the total is what keeps that visible.
            "chains_with_any_surface": sum(
                1 for r in rows
                if r["measured"] or (r.get("predicted_surface") or {}).get("measured")),
            # `fold_rows` counts the identified chains, so it stays comparable with
            # `chains_ranked` and with every record already on disk. The unidentified ORFs the
            # fold stage now also emits a row for are counted beside it rather than folded
            # into it: adding them to this total would report the no-dropped-rows invariant as
            # broken when what actually happened is that the stage covers more than it did.
            "fold_rows": (sum(1 for r in fold_rows if r["identified"])
                          if fold else None),
            "fold_rows_unidentified": (sum(1 for r in fold_rows
                                           if not r["identified"]) if fold else None),
            "fold_sequence_recovered": (sum(1 for r in fold_rows
                                            if r["sequence_recovered"]
                                            and r["identified"]) if fold else None),
            "fold_sequence_recovered_unidentified": (
                sum(1 for r in fold_rows
                    if r["sequence_recovered"] and not r["identified"]) if fold else None),
            "fold_covers_every_chain": (
                sum(1 for r in fold_rows if r["identified"]) == len(scored)
                if fold else None),
            "evidence_targets": len(evidence) if evidence else 0,
            "evidence_calls": sum(len(r["calls"]) for r in evidence) if evidence else 0},
    }


HOLDOUT = "severe acute respiratory syndrome coronavirus 2"

# The genomes this chain has been run end to end on, with the organism each one is identified
# without. Accessions and holdout names are the ones `runs/target_ranks.csv` already records,
# so the chain and the ranking experiment are speaking about the same runs.
#
# Influenza arrives as eight nucleotide records rather than one, and is passed as eight
# accessions: `fetch_genome` returns the segments separately rather than concatenated,
# because joining them would create reading frames across a junction that exists in no
# molecule and those frames would be called as ORFs.
#
# Three families rather than three strains, because the question these rows answer is whether
# a coverage number is a property of the pipeline or of how much structural work a pathogen's
# relatives have had, and three isolates of one family could not tell those apart.
GENOMES = (
    {"pathogen": "SARS-CoV-2", "accession": "NC_045512.2", "holdout": HOLDOUT},
    {"pathogen": "Nipah virus", "accession": "NC_002728.1", "holdout": "Nipah virus"},
    {"pathogen": "Influenza A PR8",
     "accession": ("NC_002016.1,NC_002017.1,NC_002018.1,NC_002019.1,NC_002020.1,"
                   "NC_002021.1,NC_002022.1,NC_002023.1"),
     "holdout": "Influenza A virus (strain A/Puerto Rico/8/1934 H1N1)"},
)


def _predicted_inventory(payload, stem=None):
    """Write the predicted-model inventory for the case the landing page renders.

    The evidence stage records `structures.predicted` as scalars, because
    `_evidence_summary` reduces any list to a count so a chain record does not become a copy
    of five external databases. One distinction does not survive that: the tool reports the
    host that served a model apart from the tool that computed it, and for these accessions
    the two disagree -- every model the AlphaFold host serves here was computed by ColabFold
    v1.5.2. A reader of `predicted_model_count 50` beside the label "AlphaFold DB" concludes
    the opposite.

    So the inventory is written beside the chain record rather than inside it. It runs here,
    as part of the run, because it reads the accessions back out of the case: regenerating
    it by hand against a record that has since been overwritten is how the page comes to
    cite an inventory of accessions the run no longer reaches.

    A failure is reported and swallowed. This is an annotation on a number already recorded,
    and a chain run that completed five stages must not be thrown away because an external
    structure aggregator timed out afterwards.
    """
    from . import predicted_inventory

    case = next((c for c in payload["cases"] if c.get("reference_excluded_organism")),
                payload["cases"][0])
    name = f"predicted_models_{stem}.json" if stem else "predicted_models.json"
    try:
        return predicted_inventory.run(case=case, path=ROOT / "runs" / name)
    except Exception as broke:
        print(f"predicted-model inventory not written ({type(broke).__name__}: {broke}); "
              f"the chain record itself is complete")
        return None


def run(accession="NC_045512.2", top=12, replicates=5, fold=True, record=True,
        cases=((None, "identified through its own curated entries"),
               (HOLDOUT, "the pathogen deleted from the reference"))):
    """The whole chain, run once per reference, so the comparison is in the record.

    **Why two cases and not one.** Which reference the genome is identified against changes
    what every later stage receives, and the two settings fail in opposite directions.

    Identified through its own entries, conservation is 1.0 for every candidate -- it is
    identity to the nearest curated relative and the nearest relative is the protein itself
    -- so the term carrying the most weight separates nothing and the top of the ranking
    ties. The later stages, though, work: the chain ranges are the pathogen's own, so the
    surface stage resolves structures for nearly every chain and the folding stage cuts the
    main protease out of the genome's own translation exactly, reproducing the fold already
    committed at `runs/folded_structure/`.

    With the organism deleted, the ranking separates properly and is the configuration every
    validation run in this repository uses. But the accession and the residue ranges now
    belong to a relative, so the surface stage loses the chains whose relative has no
    deposited structure over that range, and the folding stage loses every chain whose
    boundaries do not transfer -- which is most of a coronavirus polyprotein, because the
    mature-chain coordinates differ between species by tens of residues.

    Neither case alone is the system. Recording one and calling it the chain is what makes a
    reader believe a coverage number that only holds under the setting that produced it.
    """
    out = []
    for exclude, label in cases:
        out.append(_case(accession, exclude, top, replicates, fold, label))
    payload = {
        "experiment": "onset-chain-on-one-genome", "status": "complete",
        "question": "Put one pathogen genome in. What does each stage of the onset arm hand "
                    "the next, does any stage cover less than the stage before it, and what "
                    "changes when the reference changes?",
        "answer": "Five stages run, and the record carries the input and output of each, "
                  "under both references. The surface stage is the one that used to cover "
                  "less than its input: it had no caller, so the only surfaces ever measured "
                  "were six named by hand, and SARS-CoV-2's RNA-directed RNA polymerase -- "
                  "fifth of the ranked candidates -- was never asked about. It is exhaustive "
                  "here, one row per chain, measured or refused with the reason. Two breaks "
                  "remain and are recorded rather than drawn over: the library definition "
                  "reads nothing the surface stage measured, and under a holdout the folding "
                  "stage cannot cut a mature chain out of the genome because the only "
                  "boundaries it has are the relative's.",
        "genome_accession": accession,
        "cases": out,
        "seconds_total": round(sum(c["seconds"] for c in out), 2),
        "limitations": [
            "Stage 5 reads nothing from stages 1 to 4. The library definition is the same "
            "whatever the surface stage found, and the arrow a flowchart would draw there "
            "does not exist yet.",
            "The gate in stage 3 is advisory: the patch spread across structures of one "
            "domain is comparable to the footprint it is thresholded against, so most rows "
            "that are not refused on localisation read borderline rather than yes or no.",
            "The gate has a lower bound on patch area and no upper one, so a surface far "
            "larger than one footprint passes the area test although it is a region rather "
            "than a site. Nothing here turns on it, because every oversized patch in this "
            "record is refused on localisation first.",
            "Stage 4 refuses a chain whose annotated range does not transfer rather than "
            "aligning it. An alignment would recover the window, and `intake` already "
            "carries the aligner; until it is wired, a holdout costs this stage most of its "
            "input.",
            "One genome. The chain has been run on four pathogens and this record is one of "
            "them, chosen because it is the genome the project leads with."],
        "interpretation": "A record of what each stage produced. No binding, affinity, "
                          "inhibition or antiviral property is predicted for any protein or "
                          "any library member.",
        "provenance": provenance()}
    if record:
        write_json(ROOT / "runs/onset_chain.json", payload)
        _predicted_inventory(payload)
    return payload


def one(accession, pathogen=None, holdout=None, top=12, replicates=5,
        fold=True, record=False, evidence_top=3):
    """The whole chain on one genome the caller names. The entry point for a new pathogen.

    **Everything above this is a worked example; this is the thing itself.** The records in
    `runs/` cover three pathogens because those are the three this project had already
    validated a ranking on, and a reader who wants to know whether the chain works on a fourth
    should not have to edit a module to find out. `accession` is any NCBI nucleotide accession,
    or several separated by commas for a segmented genome, and nothing else is required.

    `holdout` is the organism deleted from the reference before identifying, and the default
    of None is the honest one for a genuinely new pathogen: there is nothing of it curated to
    delete. Pass it when the pathogen *is* already in UniProt and you want the chain to work
    through a relative instead, which is what makes a result on a known pathogen mean anything.
    The name has to match UniProt's own spelling of the organism, which the ranking stage
    reports back as `reference_entries_removed` -- zero there means the holdout matched nothing
    and the run is self-identified whatever it was asked for.

    The reference is reviewed viral proteins, because this is antiviral work. A genome over
    `intake.VIRAL_GENOME_LIMIT_NT` is refused rather than ranked: nothing would identify
    against that reference and the result would be a shortlist of whichever reading frames
    happened to be longest.

    Writes nothing by default. The committed records are the three validated pathogens, and a
    exploratory run should not quietly overwrite them; pass `record=True` to write
    `runs/onset_chain_<accession>.json` when a run is worth keeping.
    """
    label = pathogen or accession
    case = _case(accession, holdout, top, replicates, fold,
                 f"{label}, " + (f"{holdout} held out" if holdout
                                 else "nothing held out of the reference"),
                 evidence_top=evidence_top)
    case["pathogen"] = label
    if record:
        stem = re.sub(r"[^A-Za-z0-9]+", "_", accession.split(",")[0]).strip("_")
        write_json(ROOT / f"runs/onset_chain_{stem}.json", case)
        # `one` records a single case rather than the two `run` compares, so it is handed
        # that case directly instead of being left to pick one out of a `cases` list.
        _predicted_inventory({"cases": [case]}, stem=stem)
    return case


def sweep(genomes=GENOMES, top=12, replicates=5, fold=True, record=True):
    """The chain on every genome in `genomes`, each with its own organism held out.

    One genome is an anecdote about a pipeline. The coverage numbers this chain exists to
    report -- how many ranked chains keep a measurable surface, how many can supply their own
    sequence to fold -- are properties of how well studied a pathogen is as much as of the
    code, and a single row cannot show that. Three viral families can.

    Held out on every row, because that is the configuration the ranking is validated under
    and the one a novel pathogen is actually in. The self-identified contrast stays in
    `runs/onset_chain.json`, where it is a statement about the rubric rather than about
    coverage.
    """
    rows = []
    for genome in genomes:
        started = time.time()
        try:
            case = _case(genome["accession"], genome["holdout"], top, replicates, fold,
                         f"{genome['pathogen']}, {genome['holdout']} held out")
            case["pathogen"] = genome["pathogen"]
            rows.append(case)
        except Exception as failure:                      # noqa: BLE001
            rows.append({"pathogen": genome["pathogen"],
                         "genome_accession": genome["accession"],
                         "reference_excluded_organism": genome["holdout"],
                         "error": f"{type(failure).__name__}: {str(failure)[:300]}",
                         "seconds": round(time.time() - started, 2)})

    done = [r for r in rows if "coverage" in r]
    payload = {
        "experiment": "onset-chain-across-genomes", "status": "complete",
        "question": "Run the same five stages on three pathogens from three viral families. "
                    "How much of each stage's input does the next stage actually cover, and "
                    "is the answer a property of the code or of the pathogen?",
        "answer": "Of the pathogen, mostly. Every genome's surface stage emits one row per "
                  "ranked chain by construction, so coverage is never silently partial; what "
                  "varies is how many of those rows carry a measurement, and that tracks how "
                  "much structural work the family has had rather than anything in the "
                  "pipeline. The folding stage is the one the code limits, and it limits it "
                  "the same way everywhere.",
        "genomes": len(rows),
        "coverage": [{"pathogen": r.get("pathogen"),
                      "accession": r.get("genome_accession"),
                      **({"error": r["error"]} if "error" in r else r["coverage"]),
                      "seconds": r.get("seconds")} for r in rows],
        "surface_covers_every_chain_everywhere":
            all(r["coverage"]["surface_covers_every_chain"] for r in done) if done else None,
        "cases": rows,
        "limitations": [
            "Three genomes, all viral, and all three are pathogens this project already had "
            "records for, so they are not a sample of anything.",
            "Every row holds its own organism out, so the surfaces are the nearest curated "
            "relative's and not the pathogen's.",
            "Viral genomes only. This is antiviral work: the reference is reviewed viral "
            "proteins and the gene calling is six-frame, and `run_intake` refuses a genome "
            "too large to be viral rather than ranking it.",
            "A stage that fails for one genome is recorded as an error row rather than "
            "dropping the genome, for the same reason the surface stage refuses in writing."],
        "interpretation": "A record of what each stage produced on each genome. No binding, "
                          "affinity, inhibition or antiviral property is predicted for any "
                          "protein or any library member.",
        "provenance": provenance()}
    if record:
        write_json(ROOT / "runs/onset_chain_genomes.json", payload)
    return payload
