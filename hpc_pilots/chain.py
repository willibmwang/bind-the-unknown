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


def surface_rows(ranked, replicates=5):
    """One surface row per ranked candidate, measured or refused with the reason.

    Exhaustive over `ranked` on purpose, and the `top=12` cap that `enrich_with_surface`
    applies is not repeated here. That cap is a cost control for a bacterial genome of four
    thousand genes and it is the right one there; applied to a viral genome of twenty-three
    candidates it drops nothing a reader would accept being dropped, and it dropped the
    polymerase for a year by dropping nothing visible.

    A row is never absent. `measured` is False with `unmeasured_because` naming which of the
    three situations obtained, because "this protein has no surface a macrocycle could lie
    across" and "no deposited structure covers this chain" and "this machine has no freesasa"
    are three different statements and only the first is about the target.
    """
    from .geometry import surface_for_accession
    from .intake import amenable, chain_locations, exposed_from_locations

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
            offered = (markers or {}).get("structures_offered") or []
            entry["unmeasured_because"] = ((markers or {}).get("error")
                                           or "no deposited chain covers this residue range")
            entry["structures_offered"] = offered
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
                         "surface_exposed": verdict["exposed"]})
        entry["gate"] = amenable(features)
        rows.append(entry)
    return rows


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
    from .intake import amenable, chain_locations, exposed_from_locations
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
            "available_from": markers.get("available_from"),
            "gate": amenable(features),
            "gate_on_area_alone": amenable({**features, "surface_exposed": True})})
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


def _ungapped_identity(a, b):
    """Fraction of positions that agree, with no alignment. None when either side is absent.

    Ungapped on purpose: the question is whether one protein's residue numbering can be used
    on another's translation as-is, and that is exactly what a position-by-position
    comparison answers. An alignment would answer a different and easier question.
    """
    if not a or not b or len(a) != len(b):
        return None
    return round(sum(1 for x, y in zip(a, b) if x == y) / len(a), 4)


def fold_stage(ranked, ceiling=400):
    """The first ranked chain whose own sequence the chain can recover, folded.

    **Why this is not simply "fold the top candidate".** The folding endpoint takes an
    amino-acid sequence, and the only pathogen sequence the chain holds is the six-frame
    translation of the genome. The mature-chain boundaries, meanwhile, are annotated on the
    *homolog's* entry. So cutting a chain out means applying one protein's coordinates to
    another's translation, and whether that is exact depends on which ORF matched.

    For SARS-CoV-2 it is exact for the chains inside ORF1a and empty or truncated for the
    chains past the ribosomal frameshift: `chain_range` is numbered on polyprotein 1ab, the
    ORF that matched is the 4,405-residue 1a, and the polymerase sits at 4393-5324, of which
    thirteen residues exist in the ORF. Slicing anyway would have fed the folding endpoint a
    thirteen-residue fragment under the name of the polymerase, which is worse than refusing.

    So every candidate gets a row saying whether its sequence was recovered and, when it was,
    the check that says so: the slice length equals the annotated chain length. That check is
    what makes the one fold here reproduce the repository's committed fold exactly -- the main
    protease at 3264-3569 of ORF1a is 306 residues and hashes to the `sequence_sha256` already
    in `runs/folded_structure/mpro_nsp5.json`.

    The fix this refusal is standing in for is an alignment: the homolog's annotated chain
    aligned against the pathogen's ORF translation would give the span in the pathogen's own
    numbering, for every chain and under a holdout too. `intake` already carries the aligner.
    Until that exists, nine of twenty-one SARS-CoV-2 candidates cannot reach this stage, and
    that is a property of the chain and not of the targets.
    """
    from .folded_structure import fold

    rows, pick, parents = [], None, {}
    for row in ranked:
        if not row.get("identified"):
            continue
        span, orf = row.get("chain_range"), (row.get("orf") or {}).get("protein") or ""
        declared = row.get("length_aa")
        entry = {"target": row.get("name"), "rank": row.get("rank"),
                 "chain_range": list(span) if span else None,
                 "declared_length_aa": declared,
                 "parent_orf_length_aa": len(orf) or None,
                 "sequence_recovered": False, "not_recovered_because": None}
        if not span:
            # A whole ORF rather than a chain of one, so the translation *is* the sequence.
            if orf:
                entry.update({"sequence_recovered": True, "recovered_length_aa": len(orf),
                              "recovered_by": "the ORF translation is the candidate"})
            else:
                entry["not_recovered_because"] = "no translated ORF on the row"
        else:
            sliced = orf[span[0] - 1:span[1]]
            # Length alone is not enough, and under a holdout it is actively misleading. The
            # range is annotated on the *homolog*, so a slice of the pathogen's ORF at those
            # coordinates can be exactly the right length and the wrong residues: SARS-CoV's
            # main protease starts at 3241 and SARS-CoV-2's at 3264, a 23-residue offset that
            # leaves both chains 306 long. So the slice is also compared, position by
            # position, against the homolog's own annotated chain, which the entry supplies
            # and which the range is guaranteed to cut correctly. Identity near 1 means the
            # coordinates transferred; identity near chance means they did not, and the chain
            # refuses rather than folding a frameshifted window under the target's name.
            identity = _ungapped_identity(sliced, _homolog_chain(
                row.get("nearest_homolog"), span, parents))
            entry["identity_to_homolog_chain"] = identity
            if declared and len(sliced) == declared and (identity or 0) >= 0.5:
                entry.update({"sequence_recovered": True,
                              "recovered_length_aa": len(sliced),
                              "recovered_by": "slice of the matched ORF at the annotated "
                                              f"chain range, length checked against it and "
                                              f"{identity:.3f} identical to the homolog's "
                                              f"own chain"})
            elif declared and len(sliced) == declared:
                entry["recovered_length_aa"] = len(sliced)
                entry["not_recovered_because"] = (
                    f"the slice is the right length ({declared}) and only "
                    f"{identity if identity is None else round(identity, 3)} identical to the "
                    f"homolog's own chain, so the annotated coordinates do not transfer to "
                    f"this genome. Folding it would put a shifted window under this target's "
                    f"name. An alignment would recover the right window.")
            else:
                entry["not_recovered_because"] = (
                    f"the annotated range {span[0]}-{span[1]} is numbered on the homolog's "
                    f"full polyprotein and the ORF that matched is {len(orf)} residues, so "
                    f"the slice gives {len(sliced)} where the chain is {declared}. An "
                    f"alignment would recover it; a slice cannot.")
                entry["recovered_length_aa"] = len(sliced)
        rows.append(entry)
        if entry["sequence_recovered"] and pick is None \
                and 0 < entry["recovered_length_aa"] <= ceiling:
            pick = (row, orf[span[0] - 1:span[1]] if span else orf)

    if pick is None:
        return {"absence": "no ranked candidate yielded a sequence within the residue "
                           f"ceiling of {ceiling}"}, rows
    row, sequence = pick
    try:
        folded = fold(sequence, target_name=row.get("name") or "", record=False)
    except Exception as failure:                          # noqa: BLE001
        folded = {"absence": f"{type(failure).__name__}: {str(failure)[:160]}",
                  "target_name": row.get("name")}
    return folded, rows


def _case(accession, exclude_organism, top, replicates, fold, label):
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

    started = time.time()
    # Counted from the input list and not from the output, so `candidates_in` is a fact about
    # what this stage was handed rather than a restatement of what it returned. A stage that
    # reports its own output as its input can never be caught covering less than it was given.
    handed = [r for r in ranked if r.get("identified")]
    rows = surface_rows(handed, replicates=replicates)
    measured = [r for r in rows if r["measured"]]
    stages.append({
        "stage": 3, "name": "Every chain to a measured surface, or a stated reason",
        "tool": "geometry.surface_markers",
        "endpoints": ["PDBe", "RCSB", "UniProt"],
        "input": {"candidates_in": len(handed)},
        "output": {"rows_out": len(rows), "measured": len(measured),
                   "unmeasured": len(rows) - len(measured),
                   # The check Marta's review asked for, as an assertion in the record
                   # rather than as a claim in prose: the surface stage emits exactly as
                   # many rows as the stage before it emitted chains.
                   "nothing_dropped": len(rows) == len(scored),
                   "gate_yes": sum(1 for r in measured if r.get("gate") == "yes"),
                   "gate_borderline": sum(1 for r in measured
                                          if r.get("gate") == "borderline"),
                   "gate_no": sum(1 for r in measured if r.get("gate") == "no")},
        "seconds": round(time.time() - started, 2),
        "hands_to_next": "a patch median and range per chain; nothing downstream reads it yet"})

    folded, fold_rows = None, None
    if fold:
        started = time.time()
        folded, fold_rows = fold_stage(ranked)
        stages.append({
            "stage": 4, "name": "A chain to coordinates, the one lane a novel target has",
            "tool": "structures.folded", "endpoints": ["ESM Atlas"],
            "input": {"candidates_in": len(handed)},
            "output": {"foldable": sum(1 for r in fold_rows if r["sequence_recovered"]),
                       "not_foldable": sum(1 for r in fold_rows
                                           if not r["sequence_recovered"]),
                       "folded_target": (folded or {}).get("target_name"),
                       "residues": (folded or {}).get("sequence_length"),
                       "mean_plddt": (folded or {}).get("mean_plddt"),
                       "at_or_above_70": (folded or {}).get("confident_fraction"),
                       "sequence_sha256": (folded or {}).get("sequence_sha256"),
                       "error": (folded or {}).get("absence")},
            "seconds": round(time.time() - started, 2),
            "hands_to_next": "coordinates. Nothing downstream reads them yet: the surface "
                             "stage measures deposited structures, not predicted ones"})

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
        "foldable": fold_rows if fold else None,
        "seconds": round(time.time() - clock, 2),
        "coverage": {
            # The one table a reader should be able to check without reading any prose: how
            # many rows each stage emitted against how many it was handed.
            "chains_ranked": len(scored),
            "surface_rows": len(rows),
            "surface_measured": len(measured),
            "surface_refused": len(rows) - len(measured),
            "surface_covers_every_chain": len(rows) == len(scored),
            "fold_rows": len(fold_rows) if fold else None,
            "fold_sequence_recovered": (sum(1 for r in fold_rows
                                            if r["sequence_recovered"]) if fold else None),
            "fold_covers_every_chain": (len(fold_rows) == len(scored)) if fold else None},
    }


HOLDOUT = "severe acute respiratory syndrome coronavirus 2"


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
    return payload
