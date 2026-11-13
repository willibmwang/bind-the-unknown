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

# What stage 5 would have to be handed for that arrow to exist, named so the gap is a
# specification and not a complaint. Every one of these is a quantity stage 4 already
# measures and already records per chain, so the missing piece is not a measurement.
LIBRARY_WOULD_READ = ("patch_median_a2", "patch_fraction_of_surface", "sequons_near_patch")

# And the other end of the same absence. No LATER STAGE reads what stage 4 measures: stage 5
# defines the same library whatever the surface stage found, so this stays an empty tuple and
# a flowchart built from this record draws the dead end instead of the arrow a reader would
# assume. A test asserts the emptiness rather than trusting prose to stay true.
#
# What this comment used to say beyond that is now wrong, and the correction matters more than
# the tuple. Stage 4 was described here as "a measurement with no consumer", which was fair
# when the stage reported one number per chain -- the largest hydrophobic patch anywhere on
# it -- because nothing downstream read it and no reader could act on it either. It now
# reports the patch AT the residues the curators marked as doing the chemistry, in this
# genome's own numbering, with the positions listed: the project lead's third ask, which is
# answered by a reader and not by a stage. On 6LU7 chain A, measured 2026-11-12, those two
# questions give different answers -- the main protease's catalytic dyad has 93 of the chain's
# 306 residues within macrocycle reach while the chain's best patch meets that neighbourhood
# at one residue, `site_patch_fraction_of_the_site` 0.037 -- so the unanchored number was not
# a weaker version of the useful one, it was a different one. "No consumer" therefore becomes
# the narrower and true claim: no stage of this pipeline consumes it.
#
# Connecting it to stage 5 is still not a wiring job. `library_spec` takes an initiator, a
# randomised-region length range and a set of elongator classes, and all three are properties
# of the display platform rather than of a target; turning a measured patch area into a
# randomised-region length is a quantitative claim this project has not established and cannot
# cite. An arrow drawn on an invented conversion would be worse than no arrow, because it would
# read as a result. So the absence stays declared until there is a stated rule to put behind it.
SURFACE_READ_BY = ()


def _decomposed_score(features, rubric):
    """The rank score split into its five terms, so a number can be read rather than trusted.

    `intake.rank_score` returns one float. Every term in it is a stated weight times a feature
    already in the row, so the decomposition is recoverable exactly -- and the point of
    recovering it is that the two highest-scoring SARS-CoV-2 candidates differ by 0.0016 and
    arrive there by opposite routes: the spike earns 2.0 for being on the outside of the virion
    and 1.386 for conservation, the main protease earns 0.0 and 2.504. A single score hides
    that. The sum is asserted against the recorded score rather than assumed.

    The arithmetic itself lives in `intake.score_terms`, beside the `rank_score` it is a
    reading of. It was duplicated here, which made the assertion weaker than it looks: two
    copies of one sum agree with each other until someone edits one, and the copy a reader
    checks is the published one.
    """
    from .intake import score_terms

    return score_terms(features, rubric)


def _anchors_from(sites, key, span):
    """The anchor residue numbers one structure's own numbering wants.

    `geometry` refuses to convert an anchor, and says why at length: once inside, a converted
    number and a given number are indistinguishable, so a silent conversion would make every
    site figure unauditable. The conversion therefore belongs to the caller, and this is the
    caller doing it in one place with the arithmetic written down.

    `key` names which of the two numberings each site is read in -- `positions_on_the_relative`
    for a deposited structure of the homolog, `positions` for coordinates predicted from this
    genome -- and `span` is the window the structure's residue 1 corresponds to, inclusive on
    both ends. A position outside the window is dropped rather than wrapped, which is what
    keeps a region's anchors to the residues the region actually contains.

    **The measured hazard this exists to answer.** UniProt numbers the main protease's
    catalytic cysteine 3,408 on P0DTD1's replicase polyprotein, where nsp5 begins at 3,264,
    and every deposited structure of that protease calls the same residue 145. Asking
    `surface_for_accession` for the nsp5 interval with anchors 3304 and 3408 on 2026-11-12
    returned both of them in `anchors_missing` and a neighbourhood of nothing, where 41 and
    145 returned 94 residues. So an entry position is turned into a chain-relative one here --
    3408 - 3264 + 1 = 145, 3304 - 3264 + 1 = 41 -- and the result is recorded beside the
    numbering it is in.

    That conversion is right for a deposition of a mature chain and it is not guaranteed: a
    construct deposited in the entry's own numbering reports every anchor missing instead.
    Nothing here guesses between the two. `surface_for_accession` promotes the site block from
    the first structure that found the anchors, an anchor it could not find is named in
    `anchors_missing`, and the row says which numbering was asked for -- which is the
    difference between a measurement that can be checked and a confident zero.
    """
    first = int(span[0]) if span and span[0] is not None else 1
    last = int(span[1]) if span and len(span) > 1 and span[1] is not None else None
    wanted = set()
    for site in sites or ():
        for position in site.get(key) or ():
            if position is None:
                continue
            position = int(position)
            if position < first or (last is not None and position > last):
                continue
            wanted.add(position - first + 1)
    return sorted(wanted)


def _annotated_residues(sites, key, span, accession):
    """Which residue the annotation expects at each anchor, keyed by the anchor's own number.

    **The defect this exists to catch, with the number that exposed it.** `_anchors_from`
    converts an entry position into a structure's numbering as `position - chain_start + 1`,
    and `geometry` then reports `anchors_found` and `anchors_missing` BY NUMBER ALONE. A
    number that exists is reported as found whatever residue sits on it, so a deposition
    numbered on some other convention does not fail loudly -- it returns a full set of anchors
    and a neighbourhood measured somewhere else. Measured live on 2026-11-12 over the four
    chains of this ask, both cases of `runs/onset_chain.json`: on the held-out case nsp2's
    twelve anchors were all found on 7fac:A and all twelve carry the wrong residue -- 20 is
    Glu where P0C6X7 annotates Cys, 54 is Gly where it annotates His, and so on for the whole
    zinc-binding cluster. 7fac is titled "crystal structure of C-TERMINUS of the
    non-structural protein 2", its author numbering runs 4 to 527, and its first residue is
    position 115 of the mature chain: a construct numbered from its own beginning, so every
    anchor landed 111 residues from the residue it names. The same chain's anchors agree 12 of
    12 on 7msx:A in the other case, which is the point -- the conversion is not wrong, the
    structure it is offered to decides whether it lands, and nothing was checking.

    So the expectation is read off the homolog's own sequence, which is the only place the
    annotated residue can come from: `intake._entry(accession)["sequence"]["value"]` is the
    sequence the curator numbered the feature on. Returns `({anchor: (letter, entry_position)},
    refused_because)` -- the anchor number as the structure will be asked for it, the one-letter
    residue the annotation expects there, and the entry position it came from so a disagreement
    can be reported in both numberings.

    `key` selects the route exactly as `_anchors_from` does, and the two must be called with
    the same one or the expectations will be keyed on anchors nobody asked for.
    `positions_on_the_relative` and `positions` are index for index on every site
    `functional_sites` returns -- that is the property its docstring states and the reason a
    gap stays None -- so the entry position behind a mapped position is recoverable by zipping
    the two, and this is the only place that pairing is needed.

    `geometry` already carries `numbering_offset` for precisely this hazard on the patch clip,
    inferring the shift from the residues themselves rather than assuming a convention. The
    anchors never went through anything like it, and the remedy here is deliberately the weaker
    one: this reports where the anchors landed and does not move them. Shifting them would be a
    second guess about the numbering on top of the first, and the measured case above shows
    what a confident guess buys -- twelve residues of a different part of the protein,
    published as the catalytic site.
    """
    if not accession:
        return {}, ("no reviewed homolog accession, so there is no curated sequence to say "
                    "which residue the annotation expects at an anchor.")
    try:
        sequence = _entry_sequence(accession)
    except Exception as failure:                          # noqa: BLE001
        return {}, (f"the relative's sequence could not be read, so the anchors could not be "
                    f"checked against it: {type(failure).__name__}: {str(failure)[:160]}")
    first = int(span[0]) if span and span[0] is not None else 1
    last = int(span[1]) if span and len(span) > 1 and span[1] is not None else None
    expected = {}
    for site in sites or ():
        relatives = list(site.get("positions_on_the_relative") or ())
        for entry_position, position in zip(relatives, list(site.get(key) or ())):
            if position is None or entry_position is None:
                continue
            position, entry_position = int(position), int(entry_position)
            if position < first or (last is not None and position > last):
                continue
            if not 0 < entry_position <= len(sequence):
                continue
            # `setdefault` rather than assignment because two features routinely name one
            # residue -- an `Active site` and a `Binding site` at the same position -- and both
            # read the expectation off the same sequence, so the second is the same letter.
            expected.setdefault(position - first + 1,
                                (sequence[entry_position - 1], entry_position))
    return expected, None


def _letters_from_coordinates(text):
    """Author residue number -> one-letter residue, read off a PDB text's alpha carbons.

    The filter matches `geometry._residues`' own -- `ATOM` lines whose atom name is `CA` --
    because the residue this answers for has to be the residue the patch arithmetic measured.
    Alternate locations beyond the first are dropped, which `geometry` does by taking whatever
    freesasa hands back and this does by keeping the blank or `A` altloc; a residue modelled in
    two conformations is one residue with one identity either way.

    `geometry._letter` is called rather than copied, so a residue named `C` here is the residue
    the patch arithmetic calls hydrophobic or does not. The table is nineteen lines long and a
    second copy of it would agree with the first until someone added selenomethionine to one.

    No accessibility is computed and none is needed: this reads a name, and running freesasa
    again over a file it has already measured would double the cost of the most expensive step
    in the stage to answer a question about three columns of text.

    A line too short to hold the columns is skipped rather than indexed. `folded_structure`'s
    own parser skips an ATOM line whose columns will not parse and still returns the text with
    `folded` true, as `markers_from_coordinates` records, so a damaged line arrives here inside
    an accepted prediction -- and an IndexError raised while building a row would take down a
    stage that runs after four others have already paid for their answers.
    """
    from .geometry import _letter, _resnum

    letters = {}
    for line in (text or "").splitlines():
        if len(line) < 27 or not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA" or line[16] not in (" ", "A"):
            continue
        letters.setdefault(_resnum(line[22:27]), _letter(line[17:20]))
    return letters


def _letters_from_deposition(measured_on):
    """The same reading, for the deposited structure the site block was promoted from.

    `measured_on` is `geometry`'s `site_measured_on`, `"<pdb id>:<chain>"`, and it names the
    one structure of the replicates whose residues the site figures were actually measured on
    -- not the first one measured, which is a different structure whenever the first did not
    find the anchors. Checking the residues of any other deposition would answer for a file
    the numbers did not come from.

    The chain is honoured, and it is not optional. `geometry._within_reach` documents the
    measured case: on 6M0J ACE2 carries His417 and the spike's receptor-binding domain carries
    Lys417, so a number read over the whole deposition matches two different residues. The
    agreement reported here would then depend on which polymer happened to be read first.

    Returns `(letters, refused_because)`. The file is the one the measurement already
    downloaded, so `fetch_structure` is a cache hit and no request is made; a miss is recorded
    as a refusal like every other, because a residue check that cannot be run is not a residue
    check that passed.
    """
    from .geometry import _letter, _resnum, fetch_structure

    pdb_id, _, chain_id = (measured_on or "").partition(":")
    if not pdb_id:
        return {}, ("the measurement does not name the structure its site figures were taken "
                    "on, so there is no file to read the residues from.")
    try:
        text = fetch_structure(pdb_id).read_text()
    except Exception as failure:                          # noqa: BLE001
        return {}, (f"{pdb_id} could not be read on this machine, so the anchors could not be "
                    f"checked against it: {type(failure).__name__}: {str(failure)[:160]}")
    letters = {}
    for line in text.splitlines():
        # The same length guard `_letters_from_coordinates` carries, and for the same reason:
        # a truncated line in a deposited file must be skipped and not indexed into.
        if len(line) < 27 or not line.startswith("ATOM"):
            continue
        if line[12:16].strip() != "CA" or line[16] not in (" ", "A"):
            continue
        if chain_id and line[21] != chain_id:
            continue
        letters.setdefault(_resnum(line[22:27]), _letter(line[17:20]))
    if not letters:
        return {}, (f"no alpha carbon of chain {chain_id or '?'} was read from {pdb_id}, so "
                    f"the anchors could not be checked against it.")
    return letters, None


def _as_sentence(entry):
    """Why a chain inside the ceiling was not folded whole, as prose rather than as a code.

    Two different things land in these fields and only one of them is a sentence.
    `not_folded_because` is written here and reads as English. `fold_absence` comes from
    `folded_structure.fold` and is a machine reason -- `endpoint_unavailable` is a real value
    of it -- so splicing it straight into a sentence produced "that fold did not happen:
    endpoint_unavailable So its 1 annotated region ...", a code mid-clause with no stop after
    it. A reason that cannot be read is not a reason given.

    So a value that does not end in a full stop is treated as a code: underscores become
    spaces, it is quoted as the endpoint's own word rather than paraphrased into something it
    did not say, and the stop is added. Quoted rather than translated because this module does
    not own that vocabulary and a friendly rewording would be this file inventing a cause.
    """
    reason = (entry.get("not_folded_because") or entry.get("fold_absence") or "").strip()
    if not reason:
        return "no reason was recorded, which is itself a defect if it is ever read."
    if reason.endswith((".", "!", "?")):
        return reason
    return f"the folding endpoint gave “{reason.replace('_', ' ')}”."


def _reach_a():
    """How far from a marked residue the site arithmetic reaches, read and never typed.

    `geometry` derives it from the footprint the scoring already assumes -- a circular patch of
    600 square angstroms has a radius of sqrt(600 / pi) -- so the distance a macrocycle spans
    and the area it buries cannot drift apart. Reading it here rather than writing 13.8 keeps
    that property one key further out: the stage dictionary and every `site_surface` block on
    every row then quote the same number, and raising the footprint moves all of them at once.
    """
    from .geometry import MACROCYCLE_REACH_A

    return MACROCYCLE_REACH_A


def _site_block(markers, anchors, numbering, absent=None, expected=None, observed=None,
                not_checked_because=None):
    """The patch at the functional site, read off a measurement that was anchored on it.

    This is the second half of the project lead's third ask -- the residues within a distance
    a macrocyclic peptide can span, and the stage-4 patch arithmetic run on that location
    specifically -- lifted out of the geometry's return and onto the row under one key.

    `site_shares_the_patch` is the one to read first, and the informative answer is frequently
    False. Measured on 6LU7 chain A on 2026-11-12, the main protease's catalytic dyad has 93 of
    the chain's 306 residues within macrocycle reach, 41 of them hydrophobic, while the chain's
    largest patch -- 415.2 square angstroms over Leu50, Phe134, Ala191, Ala193 and Ala194 --
    meets that neighbourhood at exactly one residue, so `site_patch_fraction_of_the_site` is
    0.037. On 6M0J's receptor-binding domain, anchored at the three ACE2-contacting residues,
    the patch and the site do not meet at all. That is the finding and not a failure: the
    surface the amenability score was earned on is somewhere other than the chemistry, and a
    reader aiming a macrocycle at an active site has to be told so rather than handed the
    chain's best number.

    The four patch figures keep the names `geometry` gives them rather than being reworded for
    this row. They are the same quantity measured by the same arithmetic, and two names for one
    number is the same defect as two copies of one number -- they agree until someone edits one,
    and the page would then be reading whichever name it happened to be wired to. The reach
    counts are flattened out of the geometry's `site` block and renamed, because `residues`
    beside `residues_in_range` on the same row would be unreadable.

    `absent` is the reason there were no anchors to pass, when there were none. A block is
    returned either way, because a row that simply lacked the key would be indistinguishable
    from a site measured and found empty.

    **`measured` and `measured_at_the_annotated_residues` are two different questions and both
    have to be read.** The first says anchors were found and a neighbourhood was measured
    around them. The second says those anchors are the residues the annotation names, which
    `geometry` cannot know: it matches an anchor by number and reports it found whatever
    residue is on it. `expected` is `_annotated_residues`' map and `observed` is the structure's
    own residues, and comparing the two is the whole of the check. Measured live on 2026-11-12
    across both cases of `runs/onset_chain.json`: nsp5 4 of 4 anchors on 6w79:A (positions 1,
    41, 145 and 306 carrying Ser, His, Cys and Gln -- the catalytic dyad), nsp12 13 of 13 on
    6nur:A, nsp13 21 of 21 on 6jyt:A, and nsp2 **0 of 12** on 7fac:A, a C-terminal construct
    whose numbering puts every anchor 111 residues from the residue it names.

    The figures are kept rather than withheld when the residues disagree, and
    `not_the_annotated_residues_because` says so in one sentence. The argument for keeping them
    is the argument the whole file makes about refusals: a number that is reported with what is
    wrong with it can be checked, and a number that was silently dropped cannot. The argument
    for the sentence is the other half of it -- a site patch measured at the wrong residues is
    worse than no site patch, because only the second is visible as an absence.
    """
    site = markers.get("site") or {} if markers else {}
    found = [int(a) for a in (site.get("anchors_found") or ())]
    missing = [int(a) for a in (site.get("anchors_missing") or ())]
    block = {
        "measured": bool(found),
        "anchors_asked": list(anchors or ()),
        "anchors_numbering": numbering,
        "anchors_found": found,
        "anchors_missing": missing,
        "measured_on": (markers or {}).get("site_measured_on") or (markers or {}).get("pdb_id"),
        "reach_a": site.get("reach_a"),
        "positions_within_reach": site.get("positions"),
        "residues_within_reach": site.get("residues"),
        "hydrophobic_within_reach": site.get("hydrophobic"),
        "exposed_within_reach": site.get("exposed"),
        "site_patch_area_a2": (markers or {}).get("site_patch_area_a2"),
        "site_patch_positions": (markers or {}).get("site_patch_positions"),
        "site_patch_fraction_of_the_site": (markers or {}).get(
            "site_patch_fraction_of_the_site"),
        "site_shares_the_patch": (markers or {}).get("site_shares_the_patch"),
        "not_measured_because": None,
    }
    block.update(_residue_agreement(found, expected, observed, not_checked_because))
    if found:
        return block
    if absent:
        block["not_measured_because"] = absent
    elif not anchors:
        block["not_measured_because"] = (
            "no annotated site position could be put in this structure's numbering, so there "
            "was nothing to anchor the measurement on.")
    else:
        block["not_measured_because"] = (
            f"none of the {len(anchors)} anchor residues asked for in {numbering} exists in "
            f"the structure that was measured. A catalytic residue can be disordered in a "
            f"deposition, and author numbering is frequently not the numbering an annotation "
            f"arrives in; the figures are withheld rather than reported as an empty "
            f"neighbourhood.")
    return block


def _residue_agreement(found, expected, observed, not_checked_because=None):
    """Did each anchor land on the residue the annotation names? The keys that answer it.

    One function for both routes, because the question is the same one and two copies of a
    comparison agree until someone edits one. Returns the seven keys `_site_block` publishes:

        anchors_agreeing_with_the_annotation, anchors_disagreeing_with_the_annotation,
        anchors_not_checked_against_the_annotation, anchors_at_the_wrong_residue,
        measured_at_the_annotated_residues, not_the_annotated_residues_because,
        residues_not_checked_because

    `anchors_at_the_wrong_residue` carries both residues and both numberings per anchor, which
    is what makes a disagreement diagnosable instead of merely visible: `{"anchor": 20,
    "on_the_relative": 200, "expected_residue": "C", "residue_found": "E"}` says that the
    structure's residue 20 is Glu where P0C6X7's position 200 is the annotated Cys, and three
    of those in a row with a constant gap between the numbers is a shifted construct rather
    than three surprises.

    **What a disagreement means differs by route and the sentence says both, because this
    function cannot tell them apart.** On a deposition of the relative the residue should be
    the annotated one, so a disagreement is the numbering landing somewhere else -- a
    construct numbered from its own first residue, a chain deposited in the parent polyprotein's
    numbering -- or a deliberately mutated construct, of which an inactivated active site is the
    commonest kind in this corpus. On predicted coordinates the residues are this genome's and
    the annotation is the relative's, so a single disagreement can be an ordinary substitution
    across the species gap. Those causes are distinguishable by eye from the per-anchor rows and
    not by a count, so no count here claims to have distinguished them. What the counts do
    settle is the thing that was unanswerable before: whether the figures beside them describe
    the annotated site at all. Measured on the held-out run's own record, `runs/site_patches.json`
    of 2026-11-12, the fold side agrees on 106 of 106 anchors across 23 folded units -- so a
    disagreement there is a finding and not the ordinary case.

    `measured_at_the_annotated_residues` is None rather than False when nothing could be
    checked, because "the anchors are not the annotated residues" and "nobody could say" are
    different statements and a False would assert the first.
    """
    agreeing, wrong, checked = 0, [], 0
    for anchor in found or ():
        want = (expected or {}).get(anchor)
        has = (observed or {}).get(anchor)
        # `X` is `geometry._letter`'s answer for a residue outside the twenty it names -- a
        # selenomethionine or a modified residue -- and comparing a letter that means "not one
        # of these" against an annotated residue would report a disagreement about the table
        # rather than about the numbering.
        if want is None or has is None or has == "X":
            continue
        checked += 1
        if has == want[0]:
            agreeing += 1
        else:
            wrong.append({"anchor": anchor, "on_the_relative": want[1],
                          "expected_residue": want[0], "residue_found": has})
    block = {
        "anchors_agreeing_with_the_annotation": agreeing,
        "anchors_disagreeing_with_the_annotation": len(wrong),
        "anchors_not_checked_against_the_annotation": len(found or ()) - checked,
        "anchors_at_the_wrong_residue": wrong or None,
        "measured_at_the_annotated_residues": (None if not checked else not wrong),
        "not_the_annotated_residues_because": None,
        # Never an unexplained shortfall. A found anchor that was not compared is a refusal
        # like any other in this file, so where the caller passed no reason the generic one is
        # written out rather than left as a count with nothing behind it.
        "residues_not_checked_because": (
            (not_checked_because
             or "the residue at one or more of the anchors found could not be compared: "
                "either the relative's entry has no residue at the annotated position, or the "
                "structure names a residue there that is not one of the twenty "
                "`geometry._letter` knows, and comparing an annotated residue against `X` "
                "would report a disagreement about that table rather than about the numbering.")
            if checked < len(found or ()) else None),
    }
    if not wrong:
        return block
    named = "; ".join(f"{row['anchor']} carries {row['residue_found']} where the relative's "
                      f"entry has {row['expected_residue']} at {row['on_the_relative']}"
                      for row in wrong[:3])
    if len(wrong) > 3:
        named += f"; and {len(wrong) - 3} more"
    block["not_the_annotated_residues_because"] = (
        f"{len(wrong)} of the {checked} anchors this measurement was taken at do not carry the "
        f"residue the annotation names there -- {named}. An annotated position is offered to a "
        f"structure as `position - chain_start + 1`, which is right for a deposition of the "
        f"mature chain and wrong for a construct numbered from its own first residue or for a "
        f"chain deposited in the parent polyprotein's numbering; a deliberately mutated "
        f"construct and a substitution across the species gap read the same way from here. The "
        f"figures in this block were measured at the residues named above and are reported "
        f"rather than withheld so that they can be seen to be the wrong ones, but they are not "
        f"the annotated site.")
    return block


def surface_rows(ranked, replicates=5, folds=(), fold_rows=()):
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

    `fold_rows` is the stage-before's row per chain, and it is here for two reasons now. The
    first is the chains that end with no surface by either route: such a row could already say
    why no deposition was measured, and what it could not say is why there were no coordinates
    either, because that reason belongs to the stage before this one. The second is the
    functional positions. Stage 3 holds the alignment an annotated residue has to be mapped
    through, so it maps them and writes them on its row; this stage reads them off, and where
    no fold row is handed over it reads the annotation itself and reports the positions in the
    relative's numbering alone.

    **The patch is measured at those positions as well as wherever the chain's best one lies.**
    Both routes take the site anchors -- `site_surface` on the deposited measurement,
    `predicted_surface.site_surface` on the prediction -- and the two questions have different
    answers on a construct carrying several domains, which is why both are reported. The
    unanchored number says whether the protein has a surface a macrocycle could lie across; the
    anchored one says what a macrocycle lying across the chemistry would actually touch.

    **A region's predicted surface is reported under its chain's row and never as a row of its
    own.** `domain_surfaces` carries one entry per folded region, because a region is a closer
    look at a target and not a new target: a second row would double-count the chain in every
    coverage number this stage publishes, and `rows_out == candidates_in` is the invariant the
    whole stage rests on.
    """
    from .geometry import surface_for_accession
    from .intake import amenable, chain_locations, exposed_from_locations

    # Keyed on the target name, which is what both stages key a row by. A fold that returned an
    # absence rather than coordinates is dropped here, so a chain with no usable prediction is
    # indistinguishable from a chain that was never folded -- both simply have no
    # `predicted_surface`, and the fold row already says which happened and why.
    #
    # A region's fold is keyed on the chain it was cut from instead, off the `unit` block the
    # fold stage puts on the payload. A payload with no `unit` is read as a whole chain, which
    # is what every fold made before the domain route existed is.
    coordinates, regions = {}, {}
    for folded in (folds or ()):
        if not (folded and folded.get("pdb") and folded.get("target_name")):
            continue
        unit = folded.get("unit") or {}
        if unit.get("kind") == "domain":
            regions.setdefault(unit.get("chain"), []).append(folded)
        else:
            coordinates[folded["target_name"]] = folded
    attempts = {r.get("target"): r for r in (fold_rows or ())}

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
        # Ask 3's positions, on every row and before any branch, so a chain this stage cannot
        # measure at all still publishes the residues a binder would be aimed at. Preferred
        # from the fold row because that stage holds the alignment; recomputed here when this
        # stage was called without one, in which case `positions` is None throughout and the
        # relative's numbering is all there is.
        attempt = attempts.get(name) or {}
        if attempt.get("functional_sites") is not None:
            sites = attempt["functional_sites"]
            sites_absent = attempt.get("functional_sites_absent_because")
        else:
            sites, sites_absent = functional_sites(accession, span)
        entry["functional_sites"] = sites
        entry["functional_sites_absent_because"] = sites_absent
        entry["site_positions_in_this_genome"] = sorted(
            {p for site in sites for p in (site.get("positions") or ()) if p is not None})
        entry["site_surface"] = None
        if not entry["identified"] or not accession:
            entry["unmeasured_because"] = (
                "no reviewed homolog accession, so there is no deposited structure to resolve. "
                "A sequence-only candidate reaches the folding stage and not this one.")
            rows.append(entry)
            continue
        # The anchors the deposited route wants, in the numbering a deposition of a mature
        # chain uses. One call and not two: passing anchors adds the site block and changes no
        # other number in the return, so asking twice would double the SASA work -- the
        # expensive step of this stage -- for a second copy of the patch median. The
        # conversion and the hazard in it are documented on `_anchors_from`.
        site_anchors = _anchors_from(sites, "positions_on_the_relative", span)
        started = time.time()
        try:
            markers = surface_for_accession(accession, interval=interval,
                                            replicates=replicates,
                                            anchors=site_anchors or None)
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
            # The patch anywhere on the chain is the number above; these are the residues it
            # is made of. They were being counted and discarded, and which five residues a
            # 415-square-angstrom patch is made of is the whole of what a binder designer can
            # act on. `patch_positions_measured_on` names the deposition they are numbered in,
            # because the area above is a median over several and a median of position lists
            # is not a position list.
            "patch_positions": markers.get("patch_positions"),
            "patch_residue_names": markers.get("patch_residue_names"),
            "patch_positions_measured_on": markers.get("patch_positions_measured_on"),
        })
        # Where the anchors landed, checked against the residue the annotation names there.
        # Only when some anchor was found: with none found there is nothing to check and the
        # block already says why it measured nothing, and reading a structure's residues to
        # compare against an empty list would record a refusal about a question nobody asked.
        wanted, wanted_absent, observed, observed_absent = {}, None, {}, None
        if (markers.get("site") or {}).get("anchors_found"):
            wanted, wanted_absent = _annotated_residues(
                sites, "positions_on_the_relative", span, accession)
            observed, observed_absent = _letters_from_deposition(
                markers.get("site_measured_on"))
        entry["site_surface"] = _site_block(
            markers, site_anchors,
            "author numbering of the deposited chain, taken as 1-based within the mature "
            "chain the relative's entry delimits",
            absent=sites_absent, expected=wanted, observed=observed,
            not_checked_because=wanted_absent or observed_absent)
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
        fold = attempts.get(entry["target"]) or {}
        sites = entry["functional_sites"]
        folded = coordinates.get(entry["target"])
        # The anchors are checked against the annotated residue on this route too, and the
        # expectation is built per UNIT rather than per row: an anchor number is relative to
        # the window that was folded, so a region's anchor 12 and the whole chain's anchor 12
        # are different residues. The check is not the same question it is on a deposition --
        # these residues are this genome's and the annotation is the relative's, so a single
        # disagreement can be a substitution across the species gap -- which is why the block
        # publishes both residues per anchor instead of only a verdict.
        accession_for_sites = entry.get("measured_on_accession")
        if folded:
            # The prediction is the located window numbered from 1, so a position in this
            # genome's numbering becomes `position - span[0] + 1` in the coordinates. This is
            # the one lane in which the site figures are reachable for a genuinely novel
            # pathogen: nothing is deposited, so either the patch at the marked residues is
            # measured on a prediction or it is not measured at all.
            span_folded = fold.get("span_in_this_genome")
            wanted, wanted_absent = _annotated_residues(sites, "positions", span_folded,
                                                        accession_for_sites)
            entry["predicted_surface"] = _predicted_surface(
                folded["pdb"], folded, entry,
                anchors=_anchors_from(sites, "positions", span_folded),
                absent=entry["functional_sites_absent_because"],
                expected=wanted, not_checked_because=wanted_absent)
        # One entry per folded region, under the chain's own row. A region's anchors are the
        # site positions inside its span and nothing else: a residue outside the window it was
        # cut at is not in these coordinates, and anchoring on it would report a
        # neighbourhood around whatever residue happened to carry that number.
        for region in regions.get(entry["target"]) or ():
            unit = region.get("unit") or {}
            span_folded = unit.get("span_in_this_genome")
            wanted, wanted_absent = _annotated_residues(sites, "positions", span_folded,
                                                        accession_for_sites)
            measured = _predicted_surface(
                region["pdb"], region, entry,
                anchors=_anchors_from(sites, "positions", span_folded),
                absent=entry["functional_sites_absent_because"], unit=unit,
                expected=wanted, not_checked_because=wanted_absent)
            entry.setdefault("domain_surfaces", []).append(measured)

    # And for a chain that got neither, both halves of the answer on one row. A reader asking
    # why a chain ends with no surface at all had to join two stages by hand to find out, and
    # the join is not symmetric: one reason is about what has been solved, the other about
    # what this pipeline was willing to fold.
    for entry in rows:
        if entry["measured"] or (entry.get("predicted_surface") or {}).get("measured"):
            continue
        fold = attempts.get(entry["target"]) or {}
        refused = (entry.get("predicted_surface") or {}).get("not_measured_because")
        units = fold.get("fold_units") or ()
        measured_regions = [s for s in (entry.get("domain_surfaces") or ())
                            if s.get("measured")]
        entry["no_surface_because"] = {
            "deposited": entry["unmeasured_because"],
            # In order of how far the chain got: a prediction that was measured and failed,
            # then one that was never attempted and why, then a sequence never located.
            "predicted": (refused
                          or fold.get("not_folded_because")
                          or fold.get("fold_absence")
                          or fold.get("not_recovered_because")
                          or "no coordinates were offered to this stage for this chain"),
            # The one of those a reader should be able to count, and the one the ceiling
            # change narrows. It used to be true whenever the chain's own length put it over
            # the ceiling; the ceiling is now applied per fold unit, so it is true only when NO
            # unit of the chain was inside it -- neither the whole chain nor any annotated
            # functional region of it. A chain whose region was folded is a chain this
            # pipeline's own number did not strand, and saying otherwise would keep claiming a
            # cost that has been paid. Where the row carries no units at all, which is every
            # record written before the domain route existed, the old reading stands.
            "the_folding_ceiling_closed_the_only_route": (
                not any(unit.get("inside_the_ceiling") for unit in units) if units
                else fold.get("inside_the_ceiling") is False),
            # And the route that did reach it, named so this dictionary cannot be read as
            # "nothing was measured for this chain" when a region of it was. The row is still
            # counted as having no surface of its own, because the measurement is of a region
            # and `measured` is a statement about the chain.
            "but_a_region_of_it_was_measured":
                [s["unit"]["label"] for s in measured_regions] or None,
        }
    return rows


def _predicted_surface(coordinates, folded, entry, anchors=None, absent=None, unit=None,
                       expected=None, not_checked_because=None):
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

    `anchors` are the marked residues in these coordinates' own numbering -- the prediction is
    numbered from one over the sequence that was folded -- and the measurement at them comes
    back under `site_surface`. `absent` is the reason there were none, carried through so the
    block says why rather than simply reporting nothing.

    `expected` is `_annotated_residues`' map for the window that was folded, and the residues
    these coordinates actually carry are read straight out of the text, which is already in
    hand. The deposited route needs the same check and needs a file to run it on; here the file
    is the argument. What the check costs is one pass over the coordinate lines and no SASA
    work at all, and what it buys is that `anchors_found` stops being a statement about numbers
    alone -- a span mapped one residue short, or a frame the window disagrees with, now shows
    up as the wrong residue instead of as a confident neighbourhood. The held-out run's fold
    side agrees on 106 of 106 anchors across 23 units, measured 2026-11-12, so the check is
    quiet on this genome by being satisfied rather than by being absent.

    `unit` makes this the measurement of one REGION of the chain rather than of the chain, and
    the only thing it changes besides the identity keys is the comparison: the deposited median
    on the row is the whole chain's patch, so a ratio against a region's patch would divide two
    quantities measured over different surfaces. It is withheld with its reason instead of
    being published as a number that looks like agreement or disagreement.
    """
    from .geometry import markers_from_coordinates

    label = (unit or {}).get("label") or entry["target"]
    started = time.time()
    try:
        markers = markers_from_coordinates(
            coordinates, label=f"{folded.get('model') or 'predicted'} · {label}",
            anchors=anchors or None)
    except ImportError as missing:
        return {"measured": False,
                "not_measured_because": f"no SASA backend on this machine: {missing}. "
                                        "A property of the environment, not of the target.",
                "is_deposited_measurement": False, "in_the_ranking": False,
                "unit": unit or None,
                "only_route_to_a_surface": not entry["measured"]}
    except Exception as failure:                          # noqa: BLE001
        # Belt and braces behind the guard inside `markers_from_coordinates`. This stage runs
        # after four stages have already paid for their answers, and it now measures up to
        # twelve predictions rather than one, so the cost of letting any single one raise is
        # the whole record. A refusal here is a row that says why, like every other refusal.
        return {"measured": False,
                "not_measured_because": f"{type(failure).__name__}: {str(failure)[:160]}",
                "is_deposited_measurement": False, "in_the_ranking": False,
                "unit": unit or None,
                "only_route_to_a_surface": not entry["measured"]}
    if markers.get("error"):
        return {"measured": False, "not_measured_because": markers["error"],
                "is_deposited_measurement": False, "in_the_ranking": False,
                "unit": unit or None,
                "only_route_to_a_surface": not entry["measured"]}

    deposited = entry.get("patch_median_a2")
    patch = markers["hydrophobic_patch_area"]
    return {
        "measured": True,
        "measured_on": "coordinates from the fold stage, not a deposition",
        # What was folded, present only on a region's measurement. A reader meeting two
        # surfaces under one chain has to be able to tell which residues each is of, and the
        # span is in this genome's own numbering while `entry_span_on_the_relative` is the
        # provenance -- the curator annotated the second and the alignment produced the first.
        "unit": unit or None,
        "model": folded.get("model"),
        "mean_plddt": folded.get("mean_plddt"),
        "confident_fraction": folded.get("confident_fraction"),
        "residues": markers["residues"],
        "patch_a2": patch,
        "patch_fraction_of_surface": markers["patch_fraction_of_surface"],
        "patch_positions": markers.get("patch_positions"),
        "patch_residue_names": markers.get("patch_residue_names"),
        "exposed_fraction_of_residues": markers["exposed_fraction"],
        "sequons_near_patch": markers["sequons_near_patch"],
        "site_surface": _site_block(
            markers, anchors,
            "the predicted coordinates' own numbering, 1-based over the "
            + ("region" if unit else "chain") + " that was folded",
            absent=absent, expected=expected,
            observed=_letters_from_coordinates(coordinates),
            not_checked_because=not_checked_because),
        "seconds": round(time.time() - started, 2),
        # True exactly when no deposited chain covered this target, which is the case the
        # reordering exists for: the novel target whose only coordinates are predicted ones.
        "only_route_to_a_surface": not entry["measured"],
        "deposited_patch_median_a2": deposited,
        "ratio_to_deposited_median": (round(patch / deposited, 2)
                                      if deposited and not unit else None),
        "not_compared_because": (
            "the deposited median on this row is the whole chain's patch and this is a patch "
            "on one region of it, so the two are measured over different surfaces."
            if deposited and unit else None),
        "is_deposited_measurement": False,
        "in_the_ranking": False,
        "caveat": "A patch computed on predicted coordinates. It inherits the prediction's "
                  "error, a monomer's surface is not necessarily exposed in an oligomer, and "
                  "over a low-confidence stretch it describes a fold the model did not commit "
                  "to. Reported beside a deposited measurement and never pooled with one."
                  + (" A region folded on its own has also lost whatever its neighbours in the "
                     "chain contributed to its shape." if unit else ""),
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

    **And it returns `blocks`, which is what makes a position transferable and not only a
    span.** The alignment's aligned blocks are kept as plain nested lists of ints,
    `[[[t0, t1], ...], [[q0, q1], ...]]`, the first list indexing this genome's ORF
    translation and the second the relative's chain, both 0-based and half-open -- the shape
    `domains.map_position` documents and consumes. They were being computed, read for the two
    scalars above and thrown away, and throwing them away is what left every annotation on the
    relative's entry stranded there: the span alone carries the chain's two endpoints, so a
    domain or an active-site residue inside the chain could only be carried over by assuming
    the two numberings agree between the endpoints. They do not. Under the holdout, spike S1
    runs 14-654 on the relative and 24-685 in this genome -- +10 at the start and +31 at the
    end -- and the 21-residue disagreement between those two offsets is exactly the indel
    content the blocks describe and a constant cannot.

    They are `int()`-ed on the way out because Biopython hands back numpy integers, which
    `json.dump` refuses: a record is written at the end of every run, so a numpy int here is
    not a type nicety but the difference between a run that records and a run that raises.
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
            "identity": round(identical / matched, 4) if matched else 0.0,
            "blocks": [[[int(a), int(b)] for a, b in blocks[0]],
                       [[int(a), int(b)] for a, b in blocks[1]]]}


# How much of the relative's mature chain a local alignment has to account for before the
# window it points at is taken as that chain in this genome. Half is a low bar for a real
# correspondence and a high one for a chance local hit on a BLOSUM62 alignment of a few
# hundred residues, and the record carries `covered` on every row so a reader can see where
# each one fell rather than only which side of the bar it was.
MINIMUM_CHAIN_COVERAGE = 0.5

# The residue length above which this stage does not send a SEQUENCE to the folding endpoint.
# Named rather than left as a default argument because the record has to carry it and the site
# has to read it: nine of the twenty chains on the SARS-CoV-2 holdout are longer than this,
# including the three spike chains and the polymerase, and a page that drew the line at a
# hand-typed 400 would keep drawing it there after the ceiling moved.
#
# It used to be described here as the number deciding which chains the surface stage can
# rescue. It is not that any more, and the correction is the point of this change: the ceiling
# is applied per FOLD UNIT, so a chain over it is no longer a chain with no coordinates -- its
# leading functional regions are folded instead, and all nine of those chains carry at least
# one annotated region at or under 400 residues (measured against UniProt on 2026-11-12). What
# the ceiling still decides is which units reach the endpoint, which is a statement about
# lengths rather than about chains.
FOLD_CEILING = 400

# Annotated functional regions folded per chain whose whole sequence is over the ceiling, in
# the relevance order `domain_inventory` establishes. STATED as a budget, not fitted: nothing
# was optimised against it. A cap is needed at all because nsp3 offers eighteen foldable
# regions on the relative's entry -- nineteen once they are mapped into this genome, because
# the `Disordered` span at 972-1003 is 32 residues there and 56 here, over the minimum this
# side of the species gap -- and folding every region of every over-ceiling chain would spend
# one run on one chain: 75 folds across the shortlist is not a page, it is a database.
#
# Three rather than one, and the measurement on the held-out run is why. The relevance order
# leads on contained annotated site residues, which is decisive for an enzyme and close to
# silent on a glycoprotein, so with a cap of one the leading region is sometimes the one worth
# looking at and sometimes not. Measured over the thirty-seven units folded on that run:
# nsp3 led with `CoV Nsp3 Y` at 35.1 mean confidence while its `Peptidase C16` -- the protease
# the chain is named for -- folded at 76.4 and would not have been folded at all; nsp13 led
# with `CV ZBD` at 66.1 while its `(+)RNA virus helicase C-terminal` folded at 87.7, the best
# structure of the thirty-seven; and nsp12 led with its `RdRp catalytic` at 73.7 and also
# reached `RdRp Palm C-ter` at 74.3. Three reaches the region worth reading on every chain in
# that run; it does not promise to on another.
DOMAIN_UNITS_PER_CHAIN = 3

# Folding requests one run will spend, over every unit of every chain. Twelve when a run folded
# whole chains and nothing else, and twelve is the wrong number now: the held-out SARS-CoV-2
# run offers eleven whole chains inside the ceiling and up to three regions for each of the
# nine chains over it, so a cap of twelve would fold every chain and exactly one region, and
# the rescue this stage exists for would be dropped by a bound set before the rescue existed.
#
# It is not the bound that keeps a run inside the twenty-five minute cap on the workflow that
# serves a visitor, and it was never really that either. `budget` is: it is checked before
# every fold after the first, so the worst case is one fold at the 90 seconds a retried
# `REQUEST_TIMEOUT` allows, plus the 240-second budget, plus the fold in flight when the
# budget expires -- about seven minutes a case however high this number is. Raising this one
# moves which of the two bounds decides, from a count that predates the domain route to the
# wall clock that measures what the run actually costs. Every drop either of them causes is
# named on the unit it dropped and counted by kind in the record.
FOLD_REQUESTS_PER_RUN = 40

# Seconds one run may spend inside the folding endpoint. Named rather than left in the
# signature for the reason the ceiling is named: the stage dictionary states it, and a record
# quoting a literal while the code holds the default is a record that goes on quoting it after
# the default moves. The number itself is unchanged and the argument for it is in the fold loop
# below -- a retried `REQUEST_TIMEOUT` is 90 seconds, so this is the bound that keeps a run
# inside the twenty-five minute cap on the workflow that serves a visitor, and the seconds it
# spends on calls the endpoint refuses are counted apart from the seconds it spends folding.
FOLD_BUDGET_SECONDS = 240.0


def _in_relevance_order(regions, accession, chain_range, ceiling):
    """One chain's fold units, reordered from longest-first to most-relevant-first.

    `domains.fold_units` hands the regions back longest first, which is the order a caller
    spending a bounded number of folds would spend them in if length were the question. It is
    not, and the measurement that says so is `domain_inventory`'s: sorted by length, spike S1
    leads with `BetaCoV S1-NTD` at 290 residues and no annotated site residue while the
    `Receptor-binding domain (RBD)` sits behind it at 223; spike S2' leads with
    `Extracellular`, a membrane-topology span whose whole content is which side of a bilayer
    the chain is on; and the polymerase leads with `NiRAN` at 255 residues and two site
    residues while `RdRp catalytic` waits at 163 with six. Length put the best-known antiviral
    target in the set second and a topology annotation first.

    So the order is read from `domain_inventory`, which is where that correction lives and
    where it is recorded per region -- `site_residues_inside` as data, `relevance_terms_matched`
    as an authored term list validated against no labelled corpus, and
    `describes_position_not_function` sorting the topology spans last. Reached through its
    `_chain_row` rather than reimplemented here, because a second copy of an ordering rule
    agrees with the first exactly until someone edits one, and the copy a reader checks is the
    published one. The three signals are carried onto each unit so a page showing a region
    first can say why it is first.

    Sorting is stable and a region the inventory does not account for keeps its place at the
    end of the list, so a chain the inventory refuses -- no accession, an entry that could not
    be read -- is returned in the length order it arrived in rather than scrambled.
    """
    if not regions or not accession:
        return list(regions)
    from . import domain_inventory

    try:
        rows = domain_inventory._chain_row(
            {"nearest_homolog": accession,
             "chain_range": list(chain_range) if chain_range else None}, {}, ceiling)
        ranking = {(row["type"], tuple(row["entry_span_on_the_relative"])): (index, row)
                   for index, row in enumerate(rows.get("regions") or ())}
    except Exception:                                     # noqa: BLE001
        # The order is a display decision and this function may not be the reason a run
        # fails. The length order `fold_units` produced stands, and every unit still carries
        # its own length, span and refusal.
        return list(regions)

    ordered = list(regions)
    for unit in ordered:
        found = ranking.get((unit["type"], tuple(unit["entry_span"] or ())))
        row = found[1] if found else {}
        unit["site_residues_inside"] = row.get("site_residues_inside")
        unit["relevance_terms_matched"] = row.get("relevance_terms_matched")
        unit["describes_position_not_function"] = row.get("describes_position_not_function")
    ordered.sort(key=lambda unit: ranking.get(
        (unit["type"], tuple(unit["entry_span"] or ())), (len(ranking), None))[0])
    for place, unit in enumerate(ordered, 1):
        unit["relevance_place"] = place
    return ordered


def _not_folded(entry, unit, sentence):
    """One bound's refusal, written on the unit it refused and on the row only if it was one.

    A row's `not_folded_because` is the whole chain's answer and three tests read it as such,
    including the one that pins the ceiling's sentence on an over-ceiling chain. A region
    dropped by a bound must not overwrite that: the chain was not declined by the cap, one of
    its regions was, and a reader told otherwise would conclude the ceiling had moved. So the
    sentence goes on the unit always and on the row only for the whole-chain unit.
    """
    unit["refused_because"] = sentence
    if unit["kind"] == "whole_chain":
        entry["not_folded_because"] = sentence


def _set_aside_sentence(regions, inside):
    """Why a chain folded whole offers none of its regions as a unit of its own.

    Carried as a count and one sentence rather than as 53 rows across the run repeating that
    sentence. A region of a chain this stage folded whole can never have coordinates of its own
    here, so each of its rows would be a name, a span and the same refusal -- against a record
    this stage already quadrupled. What the annotation offers in full, foldable or not, is
    `domain_inventory`'s record and the question that one exists to answer.

    **It agrees with its own numbers at every count, which it did not.** The sentence took its
    plural from `len(regions)` and then hard-coded "them", "are" and "none", so at exactly one
    region it read "This chain's 1 annotated region, 1 of them inside the ceiling, are already
    covered by the whole-chain fold, so none is offered as a unit of its own." Four of the
    twenty chains on the held-out run carry exactly one annotated site residue and the same
    arithmetic produces single-region chains, so the singular is an ordinary case and not an
    edge -- and a sentence that cannot count to one is the kind a reader stops believing the
    numbers beside. The zero case is worded differently again, because "0 annotated regions are
    already covered by the whole-chain fold" asserts a coverage of nothing: a chain whose
    relative's entry annotates no functional region inside it has nothing to set aside, and
    saying so is a different fact about the annotation.
    """
    if not regions:
        return ("this stage folds a region only where the whole chain is over the ceiling, and "
                "the relative's entry annotates no functional region inside this chain in any "
                "case, so there is none to set aside. An absence in the annotation, not in the "
                "protein.")
    if len(regions) == 1:
        return ("this stage folds a region only where the whole chain is over the ceiling. "
                "This chain's single annotated region, "
                + ("inside the ceiling, is" if inside else "over the ceiling, is")
                + " already covered by the whole-chain fold, so it is not offered as a unit of "
                  "its own.")
    return (f"this stage folds a region only where the whole chain is over the ceiling. This "
            f"chain's {len(regions)} annotated regions, {len(inside)} of them inside the "
            f"ceiling, are already covered by the whole-chain fold, so none is offered as a "
            f"unit of its own.")


def _still_waiting(tail):
    """The queued units a bound really dropped, which is not every unit behind it.

    A contingency unit whose chain was folded whole is behind the bound in the order and was
    never going to be folded: its residues are inside coordinates this stage already holds, so
    counting it as dropped would publish a cost nobody paid. On the held-out SARS-CoV-2 run the
    contingency tier is 53 units long and every one of them is skipped, so a tail read without
    this filter would report the fold cap refusing fifty-odd regions on a run that folded
    everything it meant to.

    Everything else in the tail is a genuine drop, including a contingency unit whose chain's
    own fold did NOT land -- that one was about to be folded instead of the chain, and a bound
    taking it is the reason the chain ends with no coordinates.
    """
    kept = []
    for item in tail:
        _target, _sequence, entry, _unit, contingent = item
        if contingent and entry.get("folded"):
            continue
        kept.append(item)
    return kept


def _enqueue(queue, entry, unit, source, contingent=False):
    """One region appended to the fold order, or refused because the frame disagrees with it.

    Lifted out of the fold order so that the two tiers which offer a region -- the regions of a
    chain the ceiling refuses, and the regions held in reserve for a chain whose own fold does
    not happen -- cut their residues the same way. They did not have to come from one place
    until the second tier existed, and a second copy of this slice would be a second chance to
    get the frame wrong: a region's mapped span is in the numbering of the ORF the chain was
    located in, which is what `domains.map_span` returns and what the record reports, so it is
    cut from that reading frame and never from the chain window.

    `contingent` travels with the queued unit and is read by the fold loop, which skips a
    contingent unit whose chain was folded whole. It is the fifth member of the queue tuple
    rather than a key on the unit, because the unit is published in the record and this is a
    fact about one run's fold order and not about the region.

    A length disagreement is a refusal with its arithmetic written out rather than a silent
    skip, which is this file's rule everywhere: the span, the residues the frame actually
    yields, the frame's own length and the length the mapping reported, so a reader can see
    which of the two is wrong.
    """
    sequence = (source or "")[unit["span"][0] - 1:unit["span"][1]]
    if len(sequence) != unit["length"]:
        unit["refused_because"] = (
            f"the mapped span {unit['span'][0]}-{unit['span'][1]} selects "
            f"{len(sequence)} residues of the {len(source or '')}-residue reading "
            f"frame it was located in, against the {unit['length']} the mapping "
            f"reports, so the window and the frame disagree and nothing is sent.")
        return False
    queue.append((f"{entry['target']} · {unit['label']}", sequence, entry, unit, contingent))
    return True


def functional_sites(accession, chain_range, located=None):
    """The residues that do the chemistry, in the relative's numbering and in this genome's.

    The project lead's third ask begins with a list of amino-acid positions, and this is that
    list: every `Active site`, `Binding site` and `Site` residue the curators marked inside
    this chain. Measured 2026-11-12, P0C6X7 carries 73 Binding site, 22 Active site and 14
    Site features, so the annotation is dense rather than occasional, and the polymerase's
    `RdRp catalytic` domain contains six of them.

    Returns `(sites, refused_because)`. Each site is

        {"type", "description", "ligand",
         "positions_on_the_relative": [int, ...],   # the entry's own numbering
         "positions": [int | None, ...],            # this genome's, index for index
         "positions_unmapped": int}

    **A position that cannot be mapped is None and stays None.** `positions` is index for
    index with `positions_on_the_relative` so the two can be read together, and a gap in the
    alignment -- an insertion in this genome, or a residue of the relative this genome
    deleted -- leaves None in its place. Substituting the nearest mapped neighbour would hand
    back a number that looks like an answer and names a residue the curator did not annotate,
    and the error does not stay small: under the holdout spike S1 maps from [14, 654] to
    [24, 685], +10 at the start and +31 at the end, and the 21-residue disagreement between
    those offsets is exactly the indel content a neighbour substitution papers over. An active
    site reported at the wrong residue is worse than one reported as unmappable, because only
    the second can be seen to be wrong.

    With no `located` -- a row whose candidate is its own ORF translation, which this stage
    never aligns -- every `positions` entry is None and the entry numbering is still returned.
    That is deliberate and not a degraded answer: the deposited route anchors on the relative's
    own numbering and needs no alignment at all, so the sites are usable for the structure that
    exists while being honestly unmapped against the one that was predicted.

    A chain with no accession has no curated entry to read and comes back as an empty list with
    the reason, like every other refusal in this file.
    """
    if not accession:
        return [], ("no reviewed homolog accession, so no curated annotation exists to read. "
                    "A sequence-only candidate has no site positions to carry over.")
    from . import domains

    within = tuple(chain_range) if chain_range else None
    try:
        annotated = domains.annotated_sites(accession, within=within)
    except Exception as failure:                          # noqa: BLE001
        return [], (f"the relative's entry could not be read: {type(failure).__name__}: "
                    f"{str(failure)[:160]}")
    # Entry position `p` is index `p - chain_start` in the chain sequence that was aligned,
    # because `fold_stage` hands `locate_chain` the slice beginning at `chain_range[0]`. A
    # chain with no annotated range is a protein that was never cut out of a polyprotein, so
    # its entry numbering is its own and the first residue is 1.
    chain_start = int(chain_range[0]) if chain_range else 1
    sites = []
    for site in annotated:
        positions = list(site.get("positions") or ())
        mapped = [domains.map_position(located, position - chain_start)
                  for position in positions]
        sites.append({"type": site["type"], "description": site["description"],
                      "ligand": site.get("ligand"),
                      "positions_on_the_relative": positions,
                      "positions": mapped,
                      "positions_unmapped": sum(1 for p in mapped if p is None)})
    if not sites:
        return [], (f"{accession} carries no Active site, Binding site or Site annotation "
                    f"inside this chain's range, so there is no marked residue to aim at. An "
                    f"absence in the annotation, not in the protein.")
    return sites, None


def unit_counts(fold_rows):
    """What stage 3 offered and what it folded, counted off its own rows.

    Derived rather than tallied during the run, for the reason the over-ceiling names are
    derived in `_case`: the stage dictionary, the coverage table and the page all state these
    numbers, and three tallies kept in three places agree until one of them is edited. Every
    count here is a count of rows or units the record itself carries, so a reader can
    recompute any of them from the record and get the same answer.

    `chains_with_no_coordinates` counts every row the stage emitted, identified or not, and
    that is the number this whole change exists to move: on the held-out SARS-CoV-2 run it was
    nine chains refused on the length of their whole sequence, and each of those nine carries
    an annotated region inside the ceiling.

    **Offered and recorded are counted separately, and they differ on purpose.** A chain folded
    whole carries no region rows -- its regions are inside the structure it already has -- so
    `domain_units_offered` is summed from each row's own count of what the annotation gave it
    and `domain_units_recorded` counts the rows actually carried. Reporting only the second
    would say the annotation offered nothing for eleven chains, which is false; reporting only
    the first would imply there are rows to read that are not there.
    """
    rows = list(fold_rows or ())
    units = [unit for row in rows for unit in (row.get("fold_units") or ())]
    regions = [unit for unit in units if unit["kind"] == "domain"]
    whole = [unit for unit in units if unit["kind"] == "whole_chain"]
    offered = sum(row.get("domain_units_offered") or 0 for row in rows)
    inside = sum(row.get("domain_units_inside_the_ceiling") or 0 for row in rows)
    with_coordinates = [row for row in rows
                        if any(unit.get("folded") for unit in (row.get("fold_units") or ()))]
    return {
        "units_offered": len(whole) + offered,
        "units_inside_the_ceiling": sum(1 for unit in whole
                                        if unit["inside_the_ceiling"]) + inside,
        "units_recorded": len(units),
        "units_folded": sum(1 for unit in units if unit.get("folded")),
        "domain_units_offered": offered,
        "domain_units_inside_the_ceiling": inside,
        "domain_units_recorded": len(regions),
        "domain_units_folded": sum(1 for unit in regions if unit.get("folded")),
        "chains_offered_a_domain_route": sum(1 for row in rows
                                             if row.get("domain_units_inside_the_ceiling")),
        "chains_rescued_by_the_domain_route": sum(1 for row in rows if row.get("domain_route")),
        "chains_with_coordinates": len(with_coordinates),
        "chains_with_no_coordinates": len(rows) - len(with_coordinates),
        "domain_units_per_chain": DOMAIN_UNITS_PER_CHAIN,
    }


def fold_stage(ranked, ceiling=FOLD_CEILING, pool=(), limit=FOLD_REQUESTS_PER_RUN,
               budget=FOLD_BUDGET_SECONDS):
    """Every ranked chain whose own sequence the chain can locate, folded.

    Returns `(primary, rows, folds)`. `primary` is the first fold in pick order and is the
    one the record and the page report in full; `folds` is every fold this stage made, which
    the surface stage reads. `limit` caps the number of folding requests one run will spend
    and a unit it drops is named in the record rather than quietly left out.

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

    **The ceiling is applied per fold unit, which is what a chain over it now gets instead of
    nothing.** A located chain is decomposed into units by `domains.fold_units`: the whole
    chain first, then its annotated functional regions carried into this genome's numbering
    through the alignment blocks. Each unit meets the ceiling on its own length. Measured
    against UniProt on 2026-11-12 over the twenty-one ranked chains of the held-out
    SARS-CoV-2 run, nine chains are longer than 400 residues -- the polymerase at 823, nsp3 at
    1,945, the helicase at 601, the three spike chains, the nucleoprotein, nsp2 and nsp4 --
    and ALL NINE carry at least one annotated region at or under it. The old behaviour threw
    the whole chain away on the strength of its length, so the stage that could give a novel
    target its only coordinates refused the nine targets a binder designer would pick first.

    Regions are folded only where the whole chain could not be, and `DOMAIN_UNITS_PER_CHAIN`
    bounds how many. A chain inside the ceiling is folded whole and its regions are left
    alone: their residues are already in that structure, and spending a fold on them again
    would buy a second view of coordinates this stage already has.

    **"Could not be" means the fold did not happen, and not merely that the chain is long.**
    The regions of a chain inside the ceiling are HELD rather than dropped, and the hold is
    released by the outcome of that chain's own fold. The reason is a defect reproduced on
    2026-11-12: with two 300-residue chains inside a 400-residue ceiling, the second carrying a
    200-residue Domain, and `budget=0.1`, the wall budget stopped the second chain's fold --
    and because its regions had been set aside the moment the ceiling said "inside", the row
    ended with `units_folded` 0, no coordinates, and its one 200-residue Domain neither folded
    nor mentioned. The same hole opens under `limit`, and under a folding endpoint that refuses
    a call, which on 11 Nov 2026 was five calls in seven. A chain whose whole-chain fold did
    not happen must still be able to reach its regions, so the held units go onto the row and
    into the fold order when it does not, with `offered_its_regions_because` saying why.

    **Whole chains are ordered ahead of domains across the whole stage, not per chain.** The
    fold order is every whole-chain unit in the existing identified-then-unidentified order,
    then every domain unit of a chain over the ceiling, then the held regions as a contingency.
    Interleaving the first two per chain would let nsp3's three regions, which sit at rank 6,
    spend the limit and the wall budget that the whole chains at ranks 7 to 20 need -- so a
    bound firing would stop being a statement about cost and become a statement about where in
    the ranking a long chain happened to fall. Ordered this way, a domain fold can only ever
    consume what is left after every whole chain this run can fold has been folded, and the
    contingency tier can only ever spend what is left after both.

    **What `primary` is when no whole chain is foldable: a REGION, flagged as one on the
    payload.** The old docstring said `primary` is the first whole chain in the pick order, and
    that is true whenever the whole-chain half of the order is non-empty -- the invariant the
    committed main-protease fold rests on, and it is unchanged. That half can be empty: with a
    single 600-residue chain carrying one 200-residue Domain, the first fold is the Domain, and
    `primary["unit"]["kind"]` is `"domain"`. The choice taken here is to report it rather than
    to return an absence, because withholding coordinates this stage actually produced is the
    very defect the ceiling change exists to remove -- so the payload carries
    `primary_is_a_region_not_a_whole_chain` on every run, true or false, and
    `primary_is_a_region_because` naming the region, its type, its chain and its span whenever
    it is true. A consumer reads the flag; it never has to infer a whole chain from a silence.
    It cannot arise on this genome, where nsp5 is always inside the ceiling.
    """
    from . import domains
    from .folded_structure import fold
    from .intake import _aligner

    # `holdings` carries, beside each row, the three things the unit pass needs and the record
    # does not keep: the ranked row the annotation is read off, the alignment that maps a span
    # into this genome, and the ORF translation a domain's residues are cut from. Held here
    # rather than written onto the row for the reason `recovered_sequence` is popped below --
    # a record carrying twenty ORF translations would be mostly ORF translations.
    # `contingency` holds the regions of a chain this stage means to fold WHOLE, until the end
    # of the stage says whether that fold happened. See the hold below for the measured defect
    # that put them here rather than on the row.
    aligner, rows, pick, parents, holdings, contingency = None, [], None, {}, [], []
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
                              "recovered_by": "the ORF translation is the candidate"})
                entry["recovered_sequence"] = orf
                # No alignment was run on this branch, so the holding carries no blocks. The
                # unit pass will offer the whole ORF and no region: an annotation on the
                # relative's entry cannot be carried into a numbering nothing has aligned, and
                # assuming the two agree is the error `domains` exists to remove. The cost is
                # real and recorded rather than hidden -- the nucleoprotein is one of the nine
                # over-ceiling chains and arrives here with no `chain_range`, so it gets no
                # domain route even though its entry carries foldable regions.
                holdings.append((entry, row, None, orf))
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
            holdings.append((entry, row, located, source))
        else:
            entry["not_recovered_because"] = (
                f"the best alignment over {len(searched)} ranked ORFs accounted for only "
                f"{located['covered']:.0%} of the relative's {len(reference)}-residue chain, "
                f"under the {MINIMUM_CHAIN_COVERAGE:.0%} this stage requires, so the window "
                f"it points at is not that chain in this genome.")
        rows.append(entry)

    # The ceiling, applied to every FOLD UNIT of every located chain before anything is
    # folded, and written onto the row and onto each unit it decides.
    #
    # It used to be applied to the chain and to nothing else, which made it the largest of
    # three silent drops in this stage: the fold cap and the wall budget each published a
    # count, while the chains declined on length alone were visible only as the gap between
    # `foldable` and `chains_folded` -- nine of twenty on the SARS-CoV-2 holdout, left for a
    # reader to find by subtracting. The count was then published, and that is as far as a
    # count can take it. The drop itself is what this pass removes. Those nine chains are the
    # polymerase, the papain-like protease, the helicase and the whole receptor-binding
    # machinery, every one of them carries an annotated functional region at or under the
    # ceiling (measured against UniProt on 2026-11-12), and the region is the part a binder
    # would be designed against in any case. So the length of the whole chain stops deciding
    # whether the chain gets coordinates and decides only which of its units reach the
    # endpoint. What is left of the old drop is a chain with no unit inside the ceiling at
    # all, which is a much narrower claim and the one `the_folding_ceiling_closed_the_only_
    # route` now makes in stage 4.
    for entry, row, located, source in holdings:
        if not entry["sequence_recovered"]:
            continue
        accession, chain_range = row.get("nearest_homolog"), row.get("chain_range")
        units = domains.fold_units(row, located, accession, ceiling)
        whole, regions = units[0], _in_relevance_order(units[1:], accession, chain_range,
                                                       ceiling)
        # Unchanged in meaning and in wording: this is the whole chain's own verdict, which
        # the record, the site and three tests read as the chain's length against the ceiling.
        # A unit's verdict lives on the unit.
        entry["inside_the_ceiling"] = bool(whole["inside_the_ceiling"])
        inside = [unit for unit in regions if unit["inside_the_ceiling"]]
        # Counted before anything is set aside, so the count is what the annotation offered
        # this chain rather than what this run chose to carry rows for.
        entry["domain_units_offered"] = len(regions)
        entry["domain_units_inside_the_ceiling"] = len(inside)
        entry["domain_units_set_aside_because"] = None
        # The per-chain budget, applied to both branches below and to neither one's exclusion.
        # It used to live inside the over-ceiling branch, where it could not describe a region
        # held back for a chain folded whole -- and such a region is published as soon as that
        # whole-chain fold does not happen, so it needs its reason written on it too.
        for rank, unit in enumerate(inside):
            if rank >= DOMAIN_UNITS_PER_CHAIN:
                # Refused by the per-chain budget and not by the ceiling, so
                # `inside_the_ceiling` stays true on it: the unit is foldable and this run
                # declined to spend a request on it, which are different statements and
                # the stage after this one reads the first one.
                unit["refused_because"] = (
                    f"number {rank + 1} of this chain's {len(inside)} regions inside the "
                    f"ceiling, in relevance order, and past the first "
                    f"{DOMAIN_UNITS_PER_CHAIN} this run folds. nsp3 alone offers nineteen "
                    f"and the whole order would spend a run on one chain.")
        if entry["inside_the_ceiling"]:
            # HELD, not discarded, and the difference is a defect this fixes. The set-aside
            # sentence below is true only once the whole chain has actually been folded, and
            # the fold loop is 150 lines further down: deciding here keyed the regions off
            # `inside_the_ceiling`, which says the chain is SHORT ENOUGH to fold and not that
            # it was folded. Reproduced on 2026-11-12 with two 300-residue chains inside a
            # 400-residue ceiling, the second carrying a 200-residue Domain, under
            # `budget=0.1`: the wall budget stopped the second chain's own fold, its regions
            # had already been dropped, and the row ended with `units_folded` 0,
            # `fold_units` holding the whole chain alone, and `domain_units_offered` 1 --
            # a chain with no coordinates whose 200-residue domain this run could not reach
            # and did not mention. The same thing happens under `limit`, and it happens on
            # every call the folding endpoint refuses, which on 11 Nov 2026 was five of
            # seven.
            #
            # So the regions wait in `contingency` and the row carries the whole chain alone
            # for now. If the whole-chain fold lands, they are set aside exactly as before and
            # the record keeps the shape it had: one sentence and a count rather than 53 rows
            # repeating that sentence, because a region of a chain folded whole is inside the
            # coordinates this stage already holds. If it does not land, they go onto the row
            # and into the fold order, which is the only way a chain whose own fold failed can
            # still reach the part of it a binder would be designed against.
            contingency.append((entry, regions, source))
            regions = []
        else:
            length = entry.get("recovered_length_aa") or 0
            entry["not_folded_because"] = (
                f"{length} residues, over the {ceiling}-residue ceiling this stage sends to "
                f"the folding endpoint. "
                + (f"{len(inside)} annotated functional region"
                   f"{'' if len(inside) == 1 else 's'} of it "
                   f"{'is' if len(inside) == 1 else 'are'} inside the ceiling, and up to "
                   f"{DOMAIN_UNITS_PER_CHAIN} of them are folded instead."
                   if inside else
                   "No coordinates are predicted for it, so a deposited structure covering "
                   "the chain is the only surface it can have."))
        entry["fold_units"] = [whole] + regions
        # Ask 3's positions, mapped here because this is the stage holding the alignment they
        # have to be mapped through. Stage 4 reads them off the row and anchors the patch
        # arithmetic on them; stage 3 itself does nothing with them.
        sites, absent = functional_sites(accession, chain_range, located)
        entry["functional_sites"] = sites
        entry["functional_sites_absent_because"] = absent

    # The fold order: every whole chain first, in the identified-then-unidentified order the
    # pick has always used, and only then every domain unit. Two passes within each half, for
    # the reason the pick was given two passes in the first place -- an unidentified ORF is
    # foldable and must be reachable, and it must never displace a named chain, because the
    # committed main-protease fold is what this repository reproduces byte for byte and a
    # one-pass loop would hand the pick to whatever happened to sort first.
    #
    # Whole chains ahead of domains is the same decision taken one level up, and it is a
    # decision about bounds rather than about relevance. A domain fold must never consume the
    # limit or the wall budget that a later chain's whole-chain fold needs: nsp3 sits at rank 6
    # and offers three regions, so interleaving per chain would spend three requests there
    # before the whole chains at ranks 7 to 20 were folded at all, and a bound firing would
    # then report where a long chain fell in the ranking rather than what the run cost.
    # Ordered this way, a domain can only ever take what is left once every whole chain this
    # run can fold has been folded.
    queue = []
    for want_identified in (True, False):
        for entry, _row, _located, _source in holdings:
            if entry["identified"] is want_identified and entry.get("inside_the_ceiling"):
                queue.append((entry["target"], entry["recovered_sequence"], entry,
                              entry["fold_units"][0], False))
    whole_chain_units = len(queue)
    for want_identified in (True, False):
        for entry, _row, _located, source in holdings:
            if (entry["identified"] is not want_identified
                    or entry.get("inside_the_ceiling") is not False):
                continue
            for unit in entry.get("fold_units") or ():
                if (unit["kind"] != "domain" or not unit["inside_the_ceiling"]
                        or unit["refused_because"]):
                    continue
                _enqueue(queue, entry, unit, source)
    # Counted here, between the second tier and the third, because this is the number the
    # record has always published as `domain_units_in_the_order` -- the regions of the chains
    # the ceiling refuses, which is the work the domain route exists to do. The contingency
    # tier below is counted separately rather than added to it.
    over_ceiling_units = len(queue) - whole_chain_units
    # And last, the regions held back from the chains this stage folds WHOLE, as a contingency
    # and not as work. Every one of them is skipped the moment its chain's own fold lands --
    # which on every genome recorded here is every one of them -- and each is folded only if
    # that fold did not happen, so this tier costs nothing on a healthy run and is the whole
    # answer on a run where the endpoint refused a chain. It is last because it must never
    # consume the limit or the wall budget that a whole chain or an over-ceiling chain's region
    # needs: a contingency can only ever spend what is left after the work.
    #
    # In the order the chains were held in, which is the ranked order, and with no
    # identified-then-unidentified pass of its own. That pass exists to protect the pick, and
    # a contingency unit cannot reach it: it exists only because some chain is inside the
    # ceiling, and such a chain is already ahead of it in the order.
    for entry, held, source in contingency:
        for unit in held:
            if unit["inside_the_ceiling"] and not unit["refused_because"]:
                _enqueue(queue, entry, unit, source, contingent=True)
    if queue:
        pick = queue[0][:2]
    # Carried only to choose and fold; a row holding a whole protein sequence would multiply
    # the record's size by the number of candidates for no reader's benefit. Captured above
    # first, because every chain inside the ceiling is folded now and not only the pick. A
    # domain's residues are never written onto the row at all -- they live in the queue, for
    # the same reason and with no exception needed.
    for entry in rows:
        entry.pop("recovered_sequence", None)

    if pick is None:
        return {"absence": "no unit of any ranked candidate was inside the residue ceiling "
                           f"of {ceiling}: neither a whole chain nor any annotated functional "
                           f"region of one"}, rows, []

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
    # seconds and the lane retries once, so a single unit can spend ninety seconds against a
    # flaky host, and a count of requests alone would have made the slow case a timeout instead
    # of a record. The wall budget is the bound that keeps a run inside the twenty-five minute
    # cap on the workflow that serves a visitor: it is checked before each fold after the first,
    # so the worst case is one fold at ninety seconds, plus the budget, plus the fold in flight
    # when the budget expires -- about seven minutes whatever `limit` is. `limit` is the second
    # bound rather than the first, and `FOLD_REQUESTS_PER_RUN` records why it is no longer 12.
    # The pick always runs, because it is the fold this repository reproduces.
    # Seconds the budget spent on calls that came back with nothing, counted apart from the
    # seconds it spent folding. The two are indistinguishable to the budget and must not be
    # indistinguishable in the record: when the endpoint is down, every chain past the first
    # few is dropped with `chains_not_folded_by_budget`, which reads as "the cap is too
    # tight" when what happened is "the host refused". Observed on 11 Nov 2026, when ESM
    # Atlas returned 504 to five of seven calls and the refusals alone exhausted the budget.
    #
    # Both bounds count their drops BY KIND, and the two counts are not interchangeable.
    # `chains_not_folded_by_limit` has meant whole chains since it was first recorded, and the
    # record's own arithmetic -- chains inside the ceiling equals folds attempted plus the two
    # bounds' drops -- is asserted in the suite against it. A domain unit dropped by the same
    # bound is counted beside it under its own name, so the invariant stays an invariant and
    # the domain route's cost is still visible.
    folds, dropped_limit, dropped_budget = [], 0, 0
    dropped_limit_units, dropped_budget_units = 0, 0
    refused_calls, refused_seconds = 0, 0.0
    attempted = 0
    folding_started = time.time()
    for index, (target, sequence, entry, unit, contingent) in enumerate(queue):
        # A contingency whose chain was folded whole is not work this run declined, it is work
        # this run does not need: the region's residues are inside the coordinates the chain
        # already has. Skipped before either bound is consulted, and deliberately not counted
        # as a drop by either -- counting it would turn a healthy run into one reporting fifty
        # units refused by the cap. The set-aside sentence is written after the loop, where it
        # is true of the units it names.
        if contingent and entry.get("folded"):
            continue
        # The bounds count ATTEMPTS and no longer queue positions, which is what both constants
        # have always said they are -- `limit` is "folding requests one run will spend". The two
        # were the same number while every queued unit was folded; a skipped contingency makes
        # them differ, and reading the position would spend the cap on units nothing was sent
        # for.
        if limit is not None and attempted >= limit:
            late = _still_waiting(queue[index:])
            dropped_limit = sum(1 for _, _, _, late_unit, _ in late
                                if late_unit["kind"] == "whole_chain")
            dropped_limit_units = len(late) - dropped_limit
            # Named on the row and on the unit, not only counted in the payload, for the same
            # reason the ceiling is: the stage after this one reports why a chain has no
            # surface, and it can only report a reason that was written down somewhere.
            for _, _, late_entry, late_unit, _late_contingent in late:
                _not_folded(late_entry, late_unit,
                            f"the cap of {limit} folds on one run was spent on the units "
                            f"ahead of this one in the fold order, which folds every whole "
                            f"chain before any region.")
            break
        if attempted and budget is not None and time.time() - folding_started >= budget:
            late = _still_waiting(queue[index:])
            dropped_budget = sum(1 for _, _, _, late_unit, _ in late
                                 if late_unit["kind"] == "whole_chain")
            dropped_budget_units = len(late) - dropped_budget
            for _, _, late_entry, late_unit, _late_contingent in late:
                _not_folded(late_entry, late_unit,
                            f"the {budget:.0f}-second folding budget for one run was spent "
                            f"before the fold order reached this unit"
                            + (f", {refused_seconds:.0f} s of it on calls the folding "
                               f"endpoint refused" if refused_calls else "")
                            + ".")
            break
        attempted += 1
        call_started = time.time()
        try:
            payload = fold(sequence, target_name=target or "", record=False)
        except Exception as failure:                      # noqa: BLE001
            payload = {"absence": f"{type(failure).__name__}: {str(failure)[:160]}",
                       "target_name": target}
        if not payload.get("folded"):
            refused_calls += 1
            refused_seconds += time.time() - call_started
        # What this fold was of, carried on the payload so the stage after this one can tell a
        # chain's coordinates from a region's without re-deriving it from the name. A region's
        # `target_name` is `"<chain> · <region>"`, which is a key and not a target: the surface
        # stage reports it under the chain's own row, because a region is a closer look at a
        # target and not a new one.
        payload["unit"] = {"kind": unit["kind"], "chain": entry["target"],
                           "label": unit["label"], "type": unit["type"],
                           "span_in_this_genome": unit["span"],
                           "entry_span_on_the_relative": unit["entry_span"],
                           "length_aa": unit["length"]}
        # Stated on the row as well as in the payload, so a reader of `foldable` can see which
        # chains have coordinates without holding the fold payloads beside it. A row's own
        # `folded` stays the whole chain's answer; a region's is on the region.
        unit["folded"] = bool(payload.get("folded"))
        unit["target_name"] = target
        unit["mean_plddt"] = payload.get("mean_plddt")
        unit["confident_fraction"] = payload.get("confident_fraction")
        unit["pdb_sha256"] = payload.get("pdb_sha256")
        if unit["kind"] == "whole_chain":
            entry["folded"] = bool(payload.get("folded"))
            entry["mean_plddt"] = payload.get("mean_plddt")
        if payload.get("absence"):
            absence = (payload["absence"].get("reason")
                       if isinstance(payload["absence"], dict) else payload["absence"])
            unit["fold_absence"] = absence
            if unit["kind"] == "whole_chain":
                entry["fold_absence"] = absence
        folds.append(payload)
    # The held-back regions, resolved now that the whole chain's own fold has an outcome. This
    # is the decision that used to be taken 150 lines above this point off `inside_the_ceiling`,
    # which says a chain is short enough to fold and not that it was folded.
    for entry, held, _source in contingency:
        inside = [unit for unit in held if unit["inside_the_ceiling"]]
        if entry.get("folded"):
            entry["domain_units_set_aside_because"] = _set_aside_sentence(held, inside)
            continue
        # The chain's own fold did not happen -- a bound, or a call the endpoint refused -- so
        # the regions stop being redundant and go onto the row. Each carries its own outcome:
        # folded where the order reached it, and otherwise the sentence the bound wrote on it.
        # `domain_units_set_aside_because` stays None, because nothing was set aside.
        entry["fold_units"] = entry["fold_units"][:1] + held
        # Nothing to say when the annotation offered nothing. The sentence below asserts that
        # regions were released, and at zero regions it asserted the release of nothing --
        # "its 0 annotated regions inside the ceiling are offered after all" -- which is the
        # same defect `_set_aside_sentence` was just given a zero form for. Every chain inside
        # the ceiling reaches this branch, including the unidentified ORF of the held-out run,
        # which has no curated entry and therefore no regions at all.
        if not inside:
            entry["offered_its_regions_because"] = None
        else:
            entry["offered_its_regions_because"] = (
                f"this chain is inside the {ceiling}-residue ceiling and would have been "
                f"folded whole, and that fold did not happen: {_as_sentence(entry)} So "
                + (f"its single annotated region inside the ceiling is offered after all."
                   if len(inside) == 1 else
                   f"its {len(inside)} annotated regions inside the ceiling are offered "
                   f"after all.")
                + " A chain whose own fold failed must still be able to reach the part of it "
                  "a binder would be designed against, and these regions were being dropped "
                  "before the fold order was ever consulted.")
    # The row's verdict on the domain route, after the folding is done because it is a
    # statement about outcomes rather than about offers: true when the only coordinates this
    # chain has are a region's. It is the count the page leads with -- a chain the ceiling
    # used to leave with nothing and this stage leaves with a structure.
    for entry in rows:
        units = entry.get("fold_units") or ()
        entry["units_folded"] = sum(1 for unit in units if unit.get("folded"))
        entry["domain_route"] = bool(
            units and not (units[0].get("folded"))
            and any(unit.get("folded") for unit in units[1:]))
    # Never silently. A bound that drops a chain is a chain with no coordinates, which is
    # the defect this whole change exists to remove, so each is named in the record.
    if not folds:
        # Reachable only with a bound that forbids the first fold, such as `limit=0`. Every
        # caller here folds at least the pick, but returning `folds[0]` on an empty list would
        # raise an IndexError in place of the absence this stage is built to report.
        return {"absence": "no fold was attempted: the bounds on this run forbade the first "
                           "one"}, rows, []
    # What `primary` IS, stated on the payload rather than left to the shape of the queue. The
    # order puts every whole chain inside the ceiling ahead of every region, so the first fold
    # is a whole chain whenever one is foldable at all -- which is the invariant the committed
    # main-protease fold depends on and is asserted both in the suite and against the recorded
    # run. But that half of the order can be empty, and then the first fold is a region: with
    # one 600-residue chain carrying one 200-residue Domain, `primary["unit"]["kind"]` is
    # `domain`, reproduced 2026-11-12. The docstring claimed a whole chain unconditionally.
    #
    # It stays a region rather than becoming an absence. Refusing to report coordinates this
    # stage holds is the defect the whole change exists to remove -- a chain over the ceiling
    # getting nothing -- so the honest answer is the region plus a flag saying unmistakably
    # that it is one. The flag is set on every run, false included, so a consumer reads it
    # rather than inferring a whole chain from its absence.
    primary_unit = folds[0].get("unit") or {}
    folds[0].setdefault("primary_is_a_region_not_a_whole_chain",
                        primary_unit.get("kind") == "domain")
    if primary_unit.get("kind") == "domain":
        folds[0].setdefault("primary_is_a_region_because", (
            f"no whole chain of this run was inside the {ceiling}-residue ceiling, so the "
            f"first fold in the order is the leading annotated region of the leading chain "
            f"over it. This payload is {primary_unit.get('label')}, a "
            f"{primary_unit.get('type')} of {primary_unit.get('chain')} at "
            f"{primary_unit.get('span_in_this_genome')} in this genome's numbering, and every "
            f"figure in it describes those {primary_unit.get('length_aa')} residues and not "
            f"the chain. The pick order itself is unchanged: a region can lead only when no "
            f"whole chain was foldable, never by displacing one."))
    if dropped_limit:
        folds[0].setdefault("chains_not_folded_by_limit", dropped_limit)
    if dropped_budget:
        folds[0].setdefault("chains_not_folded_by_budget", dropped_budget)
        folds[0].setdefault("folding_budget_seconds", budget)
    if dropped_limit_units:
        folds[0].setdefault("domain_units_not_folded_by_limit", dropped_limit_units)
    if dropped_budget_units:
        folds[0].setdefault("domain_units_not_folded_by_budget", dropped_budget_units)
        folds[0].setdefault("folding_budget_seconds", budget)
    # How the order was spent, so a reader can see that the domain route took what was left
    # rather than competing for it: whole chains first, then the regions of the chains over the
    # ceiling, then the contingency tier.
    #
    # The contingency is counted apart from the other regions and not added into them. It is
    # work this run offers and expects not to do -- 17 units on the held-out run, none of them
    # folded -- so summing the two would inflate `domain_units_in_the_order` from 23 to 40 and
    # a reader comparing it against `domain_units_folded` would see a 40 per cent shortfall
    # that is really the tier doing exactly what it is for.
    folds[0].setdefault("whole_chain_units_in_the_order", whole_chain_units)
    folds[0].setdefault("domain_units_in_the_order", over_ceiling_units)
    folds[0].setdefault("contingency_units_in_the_order",
                        len(queue) - whole_chain_units - over_ceiling_units)
    folds[0].setdefault("contingency_units_folded",
                        sum(1 for _, _, _, unit, contingent in queue
                            if contingent and unit.get("folded")))
    # Recorded whenever any call came back empty, not only when a bound fired. A run where
    # the endpoint refused half its calls and still folded everything it was asked to is a
    # run worth being able to see afterwards.
    if refused_calls:
        folds[0].setdefault("folds_refused_by_the_endpoint", refused_calls)
        folds[0].setdefault("seconds_spent_on_refused_folds", round(refused_seconds, 1))
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
    from .intake import (GATE_FLOOR, GATE_TERMS, candidate_table, decompose, gate_rows,
                         rank_score, run_intake)

    stages, clock = [], time.time()

    started = time.time()
    intake = run_intake(accession, exclude_organism=exclude_organism, top=top, record=False)
    intake_seconds = round(time.time() - started, 2)
    # The rows behind stage 1's counts, kept rather than discarded. This stage used to publish
    # five integers, so a reader could learn that nine of twelve ORFs were identified and had
    # no way to ask which nine, what any of them scored, or what the other three were. The
    # table is the same rows `candidates` flattens for `intake.candidate_record`, so the page
    # renders one shape whichever record it is reading.
    candidates = candidate_table(intake)

    started = time.time()
    ranked = decompose(intake["targets"])
    rubric = intake["rubric"]
    # Position in the ranking BEFORE the gate, written while the whole list is still here. A
    # kept row's rank is therefore its place among everything the rubric scored, and a gap in
    # the numbering on the page is a row the gate removed rather than a numbering fault.
    # Renumbering the survivors 1..n would have hidden the removal in the one column a reader
    # checks first.
    for position, row in enumerate(ranked, 1):
        if row.get("identified"):
            row["rank"] = position
    # The gate. Everything below this line -- the shortlist, the folding stage, the surface
    # stage, the evidence sweep and both scored tables on the page -- sees the kept rows and
    # not the dropped ones, which is the whole point of putting it here rather than hiding the
    # rows at the end: a candidate with no exposure, no essentiality, no drugged class and no
    # solved relatives should not be costing an ESM Atlas call.
    #
    # Stage 2 is gated first and stage 1 second, in that order and not the reverse, because an
    # ORF is scored on its own annotation while the chains cut out of it are scored on theirs.
    # A polyprotein can pay nothing on all four terms and still carry a chain that pays the
    # full exposure term, so gating stage 1 on its own row would drop that chain before stage 2
    # ever scored it. The survivors' parents are spared for exactly that reason.
    ranked, gated_out = gate_rows(ranked, rubric)
    candidates, candidates_gated_out = gate_rows(
        candidates, rubric, spare={row.get("parent") for row in ranked},
        spare_because="a mature chain cut out of it cleared the gate")
    scored = []
    for row in ranked:
        if not row.get("identified"):
            continue
        position = row["rank"]
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
            # Beside the number, because this is where the term is published as points. The
            # two tiers report different quantities into it and this record carries `tier`
            # already; without the scale a reader cannot tell a saturated remote bit score
            # from a measured k-mer fraction.
            "conservation_scale": row["features"].get("conservation_scale"),
            "conservation_at_ceiling": row["features"].get("conservation_at_ceiling"),
            "surface_exposed": row["features"]["surface_exposed"],
            "essential_class": row["features"]["essential_class"],
            "druggable_class": row["features"]["druggable_class"],
            "structures_in_family": row["features"]["structures_in_family"],
            "score_terms": split["terms"],
            "score": recomputed,
            "decomposition_reproduces_score": abs(split["total"] - recomputed) < 5e-4})
    # The gate, in the one block both stages publish it in. Named rather than counted alone:
    # "six chains dropped" and "Protein C, prM, NS2A, NS2A-alpha, NS4A, NS4B dropped, none of
    # them paying more than 0.0 on any of the four" are different statements to a reader
    # deciding whether the floor is set right, and only the second can be argued with.
    def gate_block(dropped, kept):
        return {"terms": list(GATE_TERMS), "floor": GATE_FLOOR,
                "rule": (f"a row is kept when at least one of {len(GATE_TERMS)} evidence "
                         f"terms pays {GATE_FLOOR} points or more. Conservation is not among "
                         f"them: it says how well this genome was recognised, not whether "
                         f"what was recognised is worth aiming at"),
                "kept": len(kept), "gated_out": len(dropped), "dropped": dropped}

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
                   "unidentified": intake["unidentified"],
                   # `identified` counts what the stage recognised and is unchanged by the
                   # gate, which runs after it. The table beside it is the kept rows, so the
                   # two numbers differ by this one and a reader is told why rather than left
                   # to find a table shorter than its own headline count.
                   "gate": gate_block(candidates_gated_out, candidates)},
        "seconds": intake_seconds,
        "hands_to_next": "one reviewed accession and one residue range per candidate"})
    stages.append({
        "stage": 2, "name": "Polyproteins to mature chains, and the score for each",
        "tool": "intake.decompose + intake.rank_score",
        "endpoints": ["UniProt"],
        "input": {"candidates_in": len(intake["targets"])},
        "output": {"chains_out": len(scored), "rubric": rubric,
                   # Counted over every row the gate was offered, identified or not, which is
                   # the denominator `gated_out` is a fraction of. `chains_out` counts the
                   # identified survivors and is therefore not that denominator.
                   "chains_scored_before_the_gate": len(ranked) + len(gated_out),
                   "gate": gate_block(gated_out, ranked),
                   "all_decompositions_reproduce_the_score":
                       all(r["decomposition_reproduces_score"] for r in scored)},
        "seconds": round(time.time() - started, 2),
        "hands_to_next": (
            f"every chain that cleared the gate, with the accession and range its surface is "
            f"measured on. {len(gated_out)} did not clear it and go no further: nothing below "
            f"this line folds them, measures them or looks for literature on them")})

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
    #
    # What stage 3 hands stage 4 is wider than coordinates now, and the second half of it is
    # what makes stage 4 answer the question it was asked. Stage 3 holds the only alignment
    # between the relative's entry and this genome, so it is the only stage that can carry a
    # curated position across the species gap -- `locate_chain` keeps the alignment blocks and
    # `functional_sites` walks them. It therefore hands over the marked residues in this
    # genome's own numbering as well as the structures, and stage 4 anchors the patch
    # arithmetic on them instead of reporting the largest patch wherever it happens to lie.
    # It also hands over coordinates for REGIONS of the chains too long to fold whole, which
    # stage 4 measures and reports under the chain's own row.
    # Counted from the input list and not from the output, so `candidates_in` is a fact about
    # what these stages were handed rather than a restatement of what they returned. A stage
    # that reports its own output as its input can never be caught covering less than it was
    # given.
    #
    # `ranked` is the gated list, so the exhaustiveness the surface stage rests on is
    # exhaustiveness over what the gate kept, and the rows the gate removed are named in stage
    # 2's record rather than being absent from both ends of the pipeline.
    handed = [r for r in ranked if r.get("identified")]

    folded, fold_rows, folds, over_ceiling = None, None, [], []
    if fold:
        started = time.time()
        # Every called ORF, gated or not, and deliberately so. The pool is a SEARCH CORPUS and
        # not a fold list: `fold_stage` aligns each kept chain against it to find where that
        # chain's residues actually sit in this genome, and a coronavirus chain past the
        # ribosomal frameshift lives in an ORF that is not the one that named it. Gating the
        # pool would hide a survivor's own sequence from the stage that has to fold it.
        # Nothing is folded for being in here; only `ranked` decides that.
        pool = [(f"ORF {i} of the ranked set, {len(t['orf']['protein'])} aa", t["orf"]["protein"])
                for i, t in enumerate(intake["targets"], 1)
                if (t.get("orf") or {}).get("protein")]
        folded, fold_rows, folds = fold_stage(ranked, pool=pool)
        # The chains whose WHOLE SEQUENCE this stage declined on length, named once and read
        # three times below: as a count, as a list, and in what the stage says it hands on.
        # Named here rather than recomputed at each of those because the three must agree.
        # It is no longer the same set as the chains that end with no coordinates --
        # `units` carries that, and the difference between the two is what the domain route
        # bought.
        over_ceiling = [r["target"] for r in fold_rows
                        if r["identified"] and r.get("inside_the_ceiling") is False]
        # Counted off the rows rather than tallied during the run, so the stage dictionary,
        # the coverage table and the page all read one derivation.
        units = unit_counts(fold_rows)
        whole_folds = [f for f in folds
                       if (f.get("unit") or {}).get("kind", "whole_chain") == "whole_chain"]
        region_folds = [f for f in folds if (f.get("unit") or {}).get("kind") == "domain"]
        stages.append({
            "stage": 3, "name": "Every chain to coordinates, whole where it fits and by its "
                                "functional regions where it does not",
            "tool": "structures.folded", "endpoints": ["ESM Atlas"],
            # This stage is handed every candidate, not only the identified ones, because
            # folding needs a sequence and nothing else. Both counts are stated so the
            # denominator is visible: `candidates_in` is what it was given, and
            # `identified_in` is the subset the earlier stages could name.
            # The ceiling is an input like the candidate list is, and it was missing. It
            # decides which UNITS reach the endpoint, so a record without it cannot be read
            # for why a chain was folded by its regions rather than whole -- and the site drew
            # the line from a literal of its own.
            "input": {"candidates_in": len(ranked),
                      "identified_in": len(handed),
                      # What this stage was NOT handed, which is the number the gate exists
                      # to move. `candidates_in` counts what reached the endpoint's queue;
                      # without this beside it a reader comparing stage 2's chain count with
                      # stage 3's input would read the difference as a stage losing rows.
                      "gated_out_before_this_stage": len(gated_out),
                      "residue_ceiling": FOLD_CEILING,
                      # The per-chain budget on the domain route, in the record because the
                      # page states it and because it is the reason an offered region can go
                      # unfolded with nothing wrong with it.
                      "domain_units_per_chain": DOMAIN_UNITS_PER_CHAIN,
                      # And the two bounds on the run as a whole, for the same reason the
                      # ceiling is here: a reader asking why a unit went unfolded is told
                      # which bound fired, and cannot check it against a number the record
                      # does not carry.
                      "fold_requests_per_run": FOLD_REQUESTS_PER_RUN,
                      "folding_budget_seconds": FOLD_BUDGET_SECONDS},
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
                       # How many chains were folded WHOLE, which is what these four keys
                       # have counted since they were written and what the suite asserts the
                       # arithmetic of: chains inside the ceiling equals folds attempted plus
                       # the two bounds' drops. The domain route is counted under its own
                       # names below rather than added in here, because a region folded and a
                       # chain folded are different results and one total would hide which
                       # happened. `folded_target` names the pick because the page and the
                       # committed artifact report that one in full; the others are named in
                       # `also_folded`.
                       "chains_folded": sum(1 for f in whole_folds if f.get("folded")),
                       "folds_attempted": len(whole_folds),
                       "folds_refused": sum(1 for f in whole_folds if not f.get("folded")),
                       "also_folded": [f.get("target_name") for f in whole_folds[1:]
                                       if f.get("folded")],
                       # The domain route, in the counts a page can read directly. `offered`
                       # is what the annotation gave this run after the per-chain cap,
                       # `folded` is what reached the endpoint, and `chains_rescued_by_the_
                       # domain_route` is the headline: chains that would have ended the run
                       # with no coordinates at all and now have a structure.
                       "units_offered": units["units_offered"],
                       "units_inside_the_ceiling": units["units_inside_the_ceiling"],
                       "units_recorded": units["units_recorded"],
                       "units_folded": units["units_folded"],
                       "domain_units_offered": units["domain_units_offered"],
                       "domain_units_inside_the_ceiling":
                           units["domain_units_inside_the_ceiling"],
                       "domain_units_recorded": units["domain_units_recorded"],
                       "domain_units_folded": units["domain_units_folded"],
                       "domain_units_attempted": len(region_folds),
                       "domains_folded": [f.get("target_name") for f in region_folds
                                          if f.get("folded")],
                       "chains_offered_a_domain_route":
                           units["chains_offered_a_domain_route"],
                       "chains_rescued_by_the_domain_route":
                           units["chains_rescued_by_the_domain_route"],
                       "chains_with_coordinates": units["chains_with_coordinates"],
                       # What is left of the old drop, and the number this change exists to
                       # move. A chain is here only when no unit of it -- not the chain, not
                       # any annotated region of it -- was inside the ceiling or reached the
                       # endpoint.
                       "chains_with_no_coordinates": units["chains_with_no_coordinates"],
                       "domain_units_not_folded_by_limit":
                           (folded or {}).get("domain_units_not_folded_by_limit", 0),
                       "domain_units_not_folded_by_budget":
                           (folded or {}).get("domain_units_not_folded_by_budget", 0),
                       "chains_not_folded_by_limit":
                           (folded or {}).get("chains_not_folded_by_limit", 0),
                       "chains_not_folded_by_budget":
                           (folded or {}).get("chains_not_folded_by_budget", 0),
                       # What the budget was spent on, so a reader can tell a cap that is too
                       # tight from a host that was down. Both produce the same drop count.
                       "seconds_spent_on_refused_folds":
                           (folded or {}).get("seconds_spent_on_refused_folds", 0),
                       # The chains whose own sequence is too long to send, named as well as
                       # counted because "nine chains" and "nsp3, nsp12, the spike" are
                       # different statements to a reader deciding whether the ceiling costs
                       # them anything. It is no longer the count of chains that end with
                       # nothing: `chains_rescued_by_the_domain_route` is how many of these
                       # got coordinates anyway, and `chains_with_no_coordinates` is what the
                       # ceiling still strands.
                       "chains_over_the_ceiling": len(over_ceiling),
                       "over_the_ceiling": over_ceiling,
                       # Stage 3 and stage 4 measure; neither selects. The rubric is five
                       # weights fixed before the run and no fold or surface term is among
                       # them, so the ranking is final at the end of stage 2 and nothing
                       # here can move a row. Recorded as a false rather than left to prose,
                       # the way the evidence stage already records it, because a reader
                       # meeting a reordered pipeline will otherwise read the swap as a
                       # change to what gets picked.
                       "changes_any_rank": False,
                       "folded_target": (folded or {}).get("target_name"),
                       "residues": (folded or {}).get("sequence_length"),
                       "mean_plddt": (folded or {}).get("mean_plddt"),
                       "at_or_above_70": (folded or {}).get("confident_fraction"),
                       "sequence_sha256": (folded or {}).get("sequence_sha256"),
                       "error": (folded or {}).get("absence")},
            "seconds": round(time.time() - started, 2),
            "hands_to_next": (
                f"coordinates for {units['chains_with_coordinates']} of the "
                f"{len(fold_rows)} chains it reported on -- whole where the chain fits the "
                f"{FOLD_CEILING}-residue ceiling, and by up to {DOMAIN_UNITS_PER_CHAIN} of its "
                f"annotated functional regions where it does not, which rescued "
                f"{units['chains_rescued_by_the_domain_route']} of the {len(over_ceiling)} "
                f"chains over it. Plus the functional site positions, carried through the "
                f"alignment into this genome's numbering, which the surface stage anchors the "
                f"patch arithmetic on. {units['chains_with_no_coordinates']} chains end with "
                f"no coordinates by either route. The surface stage measures what it is given "
                f"with the same arithmetic it uses on a deposition and reports the two apart; "
                f"a region's surface goes under its chain's row and never as a row of its "
                f"own. It reorders nothing: the ranking was final at the end of stage 2")})

    started = time.time()
    rows = surface_rows(handed, replicates=replicates, folds=folds,
                        fold_rows=fold_rows or ())
    measured = [r for r in rows if r["measured"]]
    # The chains that end with nothing, and which of them this pipeline's own ceiling is
    # responsible for. Counted here because this is the first stage that can see both routes
    # fail on the same chain.
    no_surface = [r for r in rows
                  if not r["measured"]
                  and not (r.get("predicted_surface") or {}).get("measured")]
    closed_by_ceiling = sum(
        1 for r in no_surface
        if r["no_surface_because"]["the_folding_ceiling_closed_the_only_route"])
    # Attempted and succeeded are different counts, and only the second may be reported as a
    # measurement. A predicted surface can fail for the same reason a deposited one can -- no
    # SASA backend, nothing readable in the coordinates -- and counting the attempt would put
    # a refusal in the column that says how many chains got an answer.
    attempted = [r for r in rows if r.get("predicted_surface")]
    predicted = [r for r in attempted if r["predicted_surface"]["measured"]]
    # The region route, counted apart from both of the above. A region's surface is reported
    # under its chain's row, so it adds no rows and moves none of the counts those three keys
    # carry; it is a second measurement of part of a chain and is counted as one.
    region_surfaces = [s for r in rows for s in (r.get("domain_surfaces") or ())]
    regions_measured = [s for s in region_surfaces if s["measured"]]
    with_a_region = [r for r in rows
                     if any(s["measured"] for s in (r.get("domain_surfaces") or ()))]
    # Where ask 3 actually landed. `site_patch_*` counts the rows whose patch arithmetic was
    # anchored on the marked residues and found them, which is strictly fewer than the rows
    # carrying positions: an anchor has to exist in the structure being measured, and author
    # numbering frequently is not the numbering an annotation arrives in.
    with_sites = [r for r in rows if r.get("functional_sites")]
    site_on_deposition = [r for r in rows if (r.get("site_surface") or {}).get("measured")]
    site_on_prediction = [
        r for r in rows
        if ((r.get("predicted_surface") or {}).get("site_surface") or {}).get("measured")
        or any((s.get("site_surface") or {}).get("measured")
               for s in (r.get("domain_surfaces") or ()))]
    # And of those, the ones measured at residues the annotation does not name, which the two
    # counts above cannot distinguish and must not be read as if they did. `anchors_found` is
    # matched by NUMBER, so a deposition on another numbering convention reports a full set of
    # anchors and a neighbourhood somewhere else: measured live on 2026-11-12, nsp2's twelve
    # anchors are all found on 7fac:A -- a C-terminal construct numbered from its own first
    # residue -- and all twelve land 111 residues from the residues they name, while the same
    # anchors agree 12 of 12 on 7msx:A in the other case. One of these counts being non-zero
    # is a finding about the numbering and not about the protein, and the row's own
    # `not_the_annotated_residues_because` names the anchors.
    site_at_wrong_residues = [
        r for r in rows
        if (r.get("site_surface") or {}).get("measured_at_the_annotated_residues") is False]
    site_at_wrong_residues_predicted = [
        r for r in rows
        if ((r.get("predicted_surface") or {}).get("site_surface")
            or {}).get("measured_at_the_annotated_residues") is False
        or any((s.get("site_surface") or {}).get("measured_at_the_annotated_residues") is False
               for s in (r.get("domain_surfaces") or ()))]
    stages.append({
        "stage": 4, "name": "Every chain to a measured surface, and the patch at the residues "
                            "that do the chemistry",
        "tool": "geometry.surface_markers",
        "endpoints": ["PDBe", "RCSB", "UniProt"],
        # Three inputs now. The second is the arrow this stage used not to have, stated even
        # when it is zero so a run where the fold produced nothing is distinguishable from a
        # run where the surface stage ignored what it was handed. The third is the one ask 3
        # needs: the marked residues, already carried into this genome's numbering by the stage
        # that holds the alignment.
        "input": {"candidates_in": len(handed),
                  "predicted_coordinates_in": sum(1 for f in folds if f.get("pdb")),
                  "predicted_region_coordinates_in": sum(
                      1 for f in folds
                      if f.get("pdb") and (f.get("unit") or {}).get("kind") == "domain"),
                  "chains_with_functional_site_positions_in": sum(
                      1 for r in (fold_rows or ()) if r.get("functional_sites")),
                  "macrocycle_reach_a": _reach_a()},
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
                   # The complement, and the share of it the ceiling decided. A chain here
                   # has no deposition covering it and no whole-chain coordinates either.
                   # `no_surface_over_the_folding_ceiling` is now the narrow claim it should
                   # always have been: not "the chain was too long" but "no unit of it was
                   # inside the ceiling", so a chain whose region was folded is no longer
                   # counted as one the ceiling stranded.
                   "no_surface_by_either_route": len(no_surface),
                   "no_surface_over_the_folding_ceiling": closed_by_ceiling,
                   # The region route, which adds no rows and is counted on its own. A chain
                   # in `chains_with_a_region_surface` has a measured patch on part of itself
                   # and no measurement of the whole, which is why it can appear in
                   # `no_surface_by_either_route` as well: `measured` is a statement about the
                   # chain and this is a statement about a region of it.
                   "region_surfaces_offered": len(region_surfaces),
                   "region_surfaces_measured": len(regions_measured),
                   "chains_with_a_region_surface": len(with_a_region),
                   "chains_with_no_surface_and_no_region_surface": sum(
                       1 for r in no_surface
                       if not any(s["measured"] for s in (r.get("domain_surfaces") or ()))),
                   # Ask 3, as counts. The first is how many chains carry the marked residues
                   # at all; the other two are how many of them the patch could actually be
                   # measured at, by each route. The gap between them is the numbering
                   # problem, not a property of the proteins.
                   "chains_with_functional_sites": len(with_sites),
                   "functional_site_positions": sum(
                       len(s.get("positions_on_the_relative") or ())
                       for r in rows for s in (r.get("functional_sites") or ())),
                   "functional_site_positions_mapped_into_this_genome": sum(
                       len(r.get("site_positions_in_this_genome") or []) for r in rows),
                   "site_patch_measured_on_a_deposition": len(site_on_deposition),
                   "site_patch_measured_on_a_prediction": len(site_on_prediction),
                   # The subset of each that was anchored on residues the annotation does not
                   # name there, so neither count above can be read as that many measurements
                   # of a functional site. A site patch at the wrong residues is worse than no
                   # site patch, because only the second is visible as an absence.
                   "site_patch_on_a_deposition_not_at_the_annotated_residues":
                       len(site_at_wrong_residues),
                   "site_patch_on_a_prediction_not_at_the_annotated_residues":
                       len(site_at_wrong_residues_predicted),
                   "gate_yes": sum(1 for r in measured if r.get("gate") == "yes"),
                   "gate_borderline": sum(1 for r in measured
                                          if r.get("gate") == "borderline"),
                   "gate_no": sum(1 for r in measured if r.get("gate") == "no"),
                   # The same false stage 3 records, for the same reason. The gate on these
                   # rows is advisory and is read by nothing that ranks.
                   "changes_any_rank": False,
                   # And the asymmetry the reorder created, as a field rather than as a
                   # sentence: stage 3 gained a consumer, no later stage reads this one.
                   "read_by_later_stages": list(SURFACE_READ_BY)},
        "seconds": round(time.time() - started, 2),
        "hands_to_next": (
            f"a patch median and range per chain, the residues the patch is made of, and the "
            f"patch measured at the marked functional residues on "
            f"{len(site_on_deposition)} chains through a deposition and "
            f"{len(site_on_prediction)} through a prediction. No LATER STAGE reads any of it: "
            f"`SURFACE_READ_BY` is empty and so is `LIBRARY_READS_FROM_SURFACE`, and stage 5 "
            f"defines the same library whatever this stage found. The site figures are the "
            f"stage's product for a reader designing a binder rather than an input to "
            f"anything downstream, which is a narrower claim than the one this key used to "
            f"make and the honest one")})

    started = time.time()
    spec = chemistry.library_spec()
    stages.append({
        "stage": 5, "name": "A library definition over the randomised region",
        "tool": "chemistry.library_spec", "endpoints": ["local tables"],
        # Three fields where there used to be one empty list, because an empty list on its own
        # reads as an oversight. The gap is a specification: these are the quantities stage 4
        # measures, and this is what stands between them and this stage.
        "input": {"reads_from_upstream": list(LIBRARY_READS_FROM_SURFACE),
                  "would_read_if_connected": list(LIBRARY_WOULD_READ),
                  "not_connected_because":
                      "the library definition takes an initiator, a randomised-region "
                      "length range and "
                      "a set of elongator classes, all three properties of the display "
                      "platform rather than of a target. No stated rule turns a measured "
                      "patch area into any of them, and inventing one would publish a "
                      "conversion as a result."},
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
        # Stage 1's rows and stage 2's rows, and they are different lists on purpose.
        # `candidates` is whole ORFs, one row per thing the gene caller found; `scores` is
        # what stage 2 cut those into, so a polyprotein is one candidate and fourteen chains.
        # Collapsing them to one table would have to pick which of the two the page means by
        # "the ranking", and the honest answer is that the shortlist comes from the second and
        # the coverage claim comes from the first.
        "candidates": candidates,
        # The two stage-1 thresholds, carried because they are arguments to that stage rather
        # than outputs of it, and the page quotes both. A record that carries its own
        # candidate rows but not the cutoffs they were produced under sends the page back to
        # a second record to find them -- and the page reads the rows from here as soon as
        # they are present, so it would find the rows and lose the thresholds.
        "minimum_orf_aa": intake.get("minimum_orf_aa"),
        "minimum_hit_score": intake.get("minimum_hit_score"),
        "scores": scored,
        "surfaces": rows,
        "library": spec,
        "folded": {k: folded[k] for k in ("target_name", "sequence_length",
                                          "sequence_sha256", "mean_plddt",
                                          "confident_fraction", "pdb_sha256", "absence")
                   if folded and k in folded} if folded else None,
        # The other things folded this run, in the same shape and under a key of their own.
        # `folded` stays the pick alone because the page, the committed artifact and every
        # record already on disk report that one, and widening it would have made this change
        # unreadable against them. The digest is carried per fold so a reader can tell two
        # predictions apart; the coordinates themselves are not, for the reason the primary's
        # are not -- 193KB per chain would be most of the record. `unit` is carried because
        # this list now holds regions as well as chains, and a region's `target_name` is
        # `"<chain> · <region>"`: without the block saying which chain and which residues,
        # a reader would have to parse the name to find out.
        "folded_also": [{k: f[k] for k in ("target_name", "unit", "sequence_length",
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
            # Stated as well as implied. The complement of the line above is the number a
            # reader is actually asking for -- how many chains this pipeline cannot say
            # anything about -- and leaving it to be subtracted is how the ceiling's cost
            # stayed invisible for as long as it did.
            "chains_with_no_surface": len(no_surface),
            "chains_with_no_surface_over_the_ceiling": closed_by_ceiling,
            "chains_over_the_folding_ceiling": (len(over_ceiling) if fold else None),
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
            # The domain route in the one table a reader checks first, because the rest of
            # this table reads as though a chain over the ceiling still ends with nothing.
            # `chains_with_no_coordinates` is the number that replaced it: every row the fold
            # stage emitted where no unit at all reached the endpoint.
            "fold_units_offered": (units["units_offered"] if fold else None),
            "fold_units_folded": (units["units_folded"] if fold else None),
            "domain_units_offered": (units["domain_units_offered"] if fold else None),
            "domain_units_folded": (units["domain_units_folded"] if fold else None),
            "chains_rescued_by_the_domain_route": (
                units["chains_rescued_by_the_domain_route"] if fold else None),
            "chains_with_no_coordinates": (
                units["chains_with_no_coordinates"] if fold else None),
            # And the same for stage 4: a surface on part of a chain is not a surface on the
            # chain, so it is counted beside `chains_with_any_surface` and never inside it.
            "chains_with_a_region_surface": len(with_a_region),
            "region_surfaces_measured": len(regions_measured),
            "chains_with_functional_sites": len(with_sites),
            "site_patch_measured_on_a_deposition": len(site_on_deposition),
            "site_patch_measured_on_a_prediction": len(site_on_prediction),
            # Carried into the coverage table as well as the stage dictionary, because this is
            # the table a reader checks first and a count of site patches with no count of the
            # ones taken at the wrong residues is the half of the answer that flatters.
            "site_patch_on_a_deposition_not_at_the_annotated_residues":
                len(site_at_wrong_residues),
            "site_patch_on_a_prediction_not_at_the_annotated_residues":
                len(site_at_wrong_residues_predicted),
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

    Identified through its own entries, conservation is 1.0 for every candidate -- it is the
    fraction of the query's 5-mers found in the nearest curated relative, and the nearest
    relative is the protein itself -- so the term carrying the most weight separates nothing
    and the top of the ranking ties. The later stages, though, work: the chain ranges are the
    pathogen's own, so the surface stage resolves structures for nearly every chain and the
    folding stage cuts the main protease out of the genome's own translation exactly,
    reproducing the fold already committed at `runs/folded_structure/`.

    That description used to read "identity to the nearest curated relative", which was the
    repo's only written definition of its heaviest term and was wrong for both tiers: the
    exact tier computes a shared-5-mer fraction and the remote tier a bit score capped at 200,
    and neither is an identity. The rows now carry `conservation_scale` so which one produced
    a given number is readable, and the term is at its ceiling here whichever scale applies,
    which is what `conservation_at_ceiling` records.

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
        "answer": "Six stages run, and the record carries the input and output of each, "
                  "under both references. The surface stage is the one that used to cover "
                  "less than its input: it had no caller, so the only surfaces ever measured "
                  "were six named by hand, and SARS-CoV-2's RNA-directed RNA polymerase -- "
                  "fifth of the ranked candidates -- was never asked about. It is exhaustive "
                  "here, one row per chain, measured or refused with the reason. Folding now "
                  "runs ahead of it and the residue ceiling is applied per fold unit rather "
                  "than per chain, so a chain too long to fold whole is folded by its "
                  "annotated functional regions instead and a chain no deposition covers can "
                  "still be measured on predicted coordinates; the two are reported apart and "
                  "neither is in the ranking, which is final at the end of stage 2. Stage 4 "
                  "also reports the patch at the residues the curators marked as doing the "
                  "chemistry, in this genome's own numbering, which is a different quantity "
                  "from the largest patch on the chain and frequently somewhere else. One "
                  "break remains and is recorded rather than drawn over: the library "
                  "definition reads nothing the surface stage measured, so no later stage "
                  "consumes stage 4's output.",
        "genome_accession": accession,
        "cases": out,
        "seconds_total": round(sum(c["seconds"] for c in out), 2),
        "limitations": [
            "The library stage reads nothing the four stages before it produced. The library "
            "definition is the same whatever the surface stage found, and the arrow a "
            "flowchart would draw there does not exist yet. The reorder sharpened this "
            "rather than fixing it: folding gained a consumer in the surface stage, and no "
            "later stage reads the surface stage, so stage 4 is the one stage here whose "
            "output nothing downstream consumes. `SURFACE_READ_BY` and "
            "`LIBRARY_READS_FROM_SURFACE` are both empty in the record for that reason. What "
            "stage 4 now produces for a reader rather than for a stage -- the marked residue "
            "positions, the residues within macrocycle reach of them and the patch measured "
            "there -- is the project lead's third ask and is not a substitute for that arrow.",
            "Stage 2 now selects as well as scores. A chain is carried forward only where at "
            "least one of four evidence terms -- surface exposure, essential class, drugged "
            "class, structures in the family -- pays half a point or more, so a chain that "
            "pays nothing on all four is never folded, never measured and never searched for "
            "literature. Conservation is deliberately not one of the four: it says the genome "
            "was recognised, not that what was recognised is worth aiming at, and it is "
            "inherited unchanged by every chain cut from one polyprotein. The consequence is "
            "the one to hold against this record: a highly conserved protein whose relatives "
            "carry no localisation, no class keyword and no deposited structure leaves the "
            "pipeline at stage 2 on the strength of four annotations that are themselves a "
            "measure of how well studied it is. Every dropped row is named in the stage "
            "block with the four numbers it failed on.",
            "Stages 3 and 4 measure and do not select. The rubric is five weights fixed "
            "before the run and no fold or surface term is among them, so the ranking is "
            "final at the end of stage 2 and the swap changed what can be measured, not "
            "what gets picked. Both stages record `changes_any_rank` false.",
            "The residue ceiling now decides which fold UNITS reach the endpoint rather than "
            "which chains do. Nine of the twenty chains on the SARS-CoV-2 holdout are longer "
            "than it, including the three spike chains and the polymerase, and every one of "
            "them carries an annotated functional region at or under it, so the region is "
            "folded where the chain cannot be. The count, the names, the per-chain reason and "
            "the per-region outcome are in the record. Two things that follow and are not "
            "softened: a region folded on its own is a model of that region in isolation and "
            "has lost whatever its neighbours in the chain contributed to its shape, and a "
            "chain whose row carries no annotated range is never aligned, so it gets no "
            "region route even where its entry offers one -- the nucleoprotein is that case.",
            "The gate in the surface stage is advisory: the patch spread across structures of "
            "one domain is comparable to the footprint it is thresholded against, so most "
            "rows that are not refused on localisation read borderline rather than yes or "
            "no.",
            "The gate has a lower bound on patch area and no upper one, so a surface far "
            "larger than one footprint passes the area test although it is a region rather "
            "than a site. Nothing here turns on it, because every oversized patch in this "
            "record is refused on localisation first.",
            "The folding stage locates a chain by aligning the relative's own residues "
            "against every ranked ORF, not by slicing this genome at the relative's "
            "coordinates. That is what makes the stage survive a holdout, and it is also a "
            "sequence-level correspondence and nothing more: a located window at 50 per cent "
            "coverage is this stage's claim that the chain is there, not a verified "
            "boundary.",
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
            "dropping the genome, for the same reason the surface stage refuses in writing.",
            "Coverage is measured over the chains that cleared stage 2's evidence gate, not "
            "over every chain the cut produced. The gate drops a chain paying under half a "
            "point on all four of surface exposure, essential class, drugged class and "
            "structures in the family, and how many that is differs by genome -- so the "
            "denominator these percentages are taken over is itself a result of the run and "
            "is stated per genome in each case's stage-2 block."],
        "interpretation": "A record of what each stage produced on each genome. No binding, "
                          "affinity, inhibition or antiviral property is predicted for any "
                          "protein or any library member.",
        "provenance": provenance()}
    if record:
        write_json(ROOT / "runs/onset_chain_genomes.json", payload)
    return payload
