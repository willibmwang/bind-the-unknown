"""The hydrophobic patch at the functional site, not at the chain's best address.

Stage 4 reports the largest hydrophobic patch anywhere on a structure. That is the right
number for the question it was built for -- is this protein amenable to a macrocycle at all --
and it is the wrong number for the question the project lead actually asked third: *within a
domain, which amino-acid positions are functionally important, how many residues can a peptide
sitting on them reach, and what does the step-4 hydrophobic arithmetic say about that location
specifically*. The two answers routinely disagree, and this record is where the disagreement
becomes visible.

Measured here on 2026-11-12, over the 37 units `runs/domain_folds.json` folded:

    the main protease's largest patch is 2,012.6 square angstroms, and the part of it at the
    catalytic dyad is 185.6 -- so **9%** of the surface stage 4 scored this chain on is at
    the chemistry and 91% of it is somewhere else;

    nsp12's `RdRp catalytic` domain is the opposite case and the reason this is reported as a
    number rather than as a yes or no: 999.8 of its 1,839.7 square angstroms are at the site,
    which is 54%;

    nsp13's `(+)RNA virus helicase C-terminal` domain carries a 432.7 square-angstrom patch
    and **none of it** is at the one annotated residue inside the domain, which
    `site_shares_the_patch` reports as false rather than reporting the 432.7 and letting a
    reader assume it is where the chemistry is.

**What transfers from a prediction to a deposition, and what does not.** The main protease is
the one chain here that can be checked against a solved structure, and it was, live on
2026-11-12. Anchoring on the dyad alone, `6LU7` chain A gives 93 residues within reach, 41 of
them hydrophobic and 33 exposed; the prediction gives 89, 41 and 30. The patch residue at the
site is Leu50 on the deposition and Leu49 with Leu50 on the prediction. So the reach
neighbourhood and the positions inside it transfer within a few percent, and a reader can act
on them. The patch *area* does not transfer at all: 415.2 square angstroms over five residues
on the deposition against 2,012.6 over nineteen on the prediction, a factor of almost five,
with 18 separate patches found on the prediction at all. A predicted monomer at mean pLDDT 62.6
is loosely packed, so hydrophobic surface that is buried in the solved fold reads as exposed
here and neighbouring components merge. The rule that follows is the one
`geometry.markers_from_coordinates` already states for a different reason: a patch area measured
on a prediction is comparable with another prediction and with nothing else.

**Three numbering systems, and the mapping between them is where the errors hide.** The
annotation lives on the relative's UniProt entry. The pathogen's residues are numbered along an
open reading frame of its own genome. The folded coordinates are numbered 1..N, because ESMFold
numbers the sequence it was handed from one and knows nothing about where that sequence came
from. So an annotated position travels

    entry position  ->  minus the mature chain's start, giving an index into the chain
                        sequence that was aligned
                    ->  `domains.map_position` through the alignment blocks, giving a
                        1-based position in this genome's reading frame
                    ->  minus the folded unit's own span start, plus one.

Every step is necessary and none of them is a constant. The held-out spike S1 chain runs 14-654
on the relative and 24-685 here: +10 at its start and +31 at its end, so a single offset is
wrong by 21 residues at the far end. nsp12 sits at 4370-5301 on the relative and 1-823 here,
because the polymerase is translated from a different reading frame past the ribosomal
frameshift -- there is no offset at all to get that one wrong by.

**The middle numbering is frame-local, and a row that did not say which frame was not a
numbering at all.** `in_this_genome` and `span_in_this_genome` are 1-based along *one open
reading frame* of this genome, not along the genome -- and the frame is whichever of the twelve
the chain aligned best to, which is why nsp12 reads 1-823 rather than somewhere past residue
four thousand. The folds record keeps that choice per chain as `found_in_orf`; this record did
not propagate it, so every span and every `positions_in_this_genome` list here was a number
with no stated coordinate system. The consequence is not hypothetical and was counted over this
run on 2026-11-12: **50 pairs** of the 37 rows carry numerically overlapping spans in
*different* frames -- spike S1's [24, 312] against the nucleoprotein's [41, 186] is the kind of
pair, two windows with no residue in common that a join or a sort on the span alone would place
on top of each other. So every row now carries a `reading_frame` block: the pool index, and
beside it the strand, the frame offset and the nucleotide start that identify the frame without
reference to a pool somebody else assembled. `found_in_orf_on_the_folds_record` and
`agrees_with_the_folds_record` travel with it, because this module locates each chain a second
time and a disagreement between the two locations would mean these rows describe different
residues from the coordinates they were measured on. Over this run the two agree on all 20
chains; the key is there so that a run where they do not is loud rather than silent.

The check that the chain is assembled correctly is the main protease, whose catalytic dyad is
famous and whose fold is the one this repository reproduces byte for byte. Verified live on
2026-11-12: entry positions 3281 and 3385 on P0C6X7, mature-chain 41 and 145, this genome's
reading frame 3304 and 3408, folded positions **41 and 145**, and the residues standing at
those two positions in `r02_3c_like_proteinase_nsp5__3c_like_proteinase_nsp5.pdb` are **HIS and
CYS**, read out of the coordinate file rather than assumed. His41 carries pLDDT 77 and Cys145
carries 85, so the dyad is in the confident part of a fold whose mean is 62.6.

That one chain is an anecdote, so the check was then run over everything. Each of the 106
anchors that reach a folded unit is compared against the residue the relative's entry carries at
the annotated position, and all 106 agree. The reason that is a check and not a tautology is
what happens when the mapping is moved: shifting every anchor by one residue in either direction
drops the agreement to 9 of 106, and by two to 5 or 6 of 106. An off-by-one anywhere in the
three-step chain would therefore have been loud rather than silent, and `agrees_with_the_relative`
on every anchor is where a later change would show it.

**The same check as a row-level verdict, under the name the other lane already uses.** Per
anchor is the right grain for a reader checking the mapping and the wrong grain for a page,
which has to decide whether to print a row's figures as a measurement at the annotated site or
as a measurement at some other residues. `chain._residue_agreement` already answers exactly
that for the deposited lane and publishes it as `measured_at_the_annotated_residues`, so this
record calls that function rather than reducing `agrees_with_the_relative` itself: two copies
of one comparison agree until somebody edits one, and the page then reads whichever name it
happens to be wired to. The six companion keys come with it, of which
`anchors_at_the_wrong_residue` is the one that makes a disagreement diagnosable -- it carries
both residues and both numberings per anchor, so three wrong anchors with a constant gap
between the numbers read as a shifted window rather than as three surprises.

On this record the verdict is True on all 23 measured units, which is what 106 of 106 agreeing
looks like from the row level, and the key therefore changes no figure on this page today. It
is not idle: the deposited lane has a measured case where it fires, nsp2 on `7fac:A` at 0 of 12
anchors, a C-terminal construct whose numbering puts every anchor 111 residues from the residue
it names. A page that printed a site patch without reading this key would publish that as a
real measurement of nsp2's active site, and nothing else on the row would say otherwise.

**Not every annotated residue is a place to put a binder, and this is the correction the record
makes rather than a nicety.** UniProt's `Site` features on a coronavirus polyprotein are
overwhelmingly proteolytic processing boundaries -- measured over this run, 28 of the 29 plain
`Site` residues name a cleavage -- and a cleavage site sits at the *end* of a mature chain by
construction. Eight of the 23 measured units carry nothing else, so measuring their site means
measuring the dangling terminus of a protein that was cut out of a longer one. In four of them
the only anchor is the unit's own last residue: `Nsp4C` at position 99 of 99, nsp2's C-terminal
domain at 129 of 129, nsp13's helicase C-terminal domain at 163 of 163 and nsp6 at 287 of 287.
Those four return site patches of 621.5, nothing, nothing and 284.3 square angstroms, which is
a fact about chain ends and not about druggability. So every anchor is labelled, a second
measurement over the chemistry residues alone travels beside the first wherever the two differ,
and `chemistry_only` is the one a binder designer should read.

**`how_to_read` is quoted and not paraphrased, so a wrong line in it is a wrong line on the
page.** That is the argument for shipping a glossary inside the record, and it is also the
argument for checking every sentence of one against the code that computes the field. Three of
its nine lines were wrong on 12 Nov 2026 and each was wrong in the same direction -- describing
a simpler arithmetic than the one that runs:

    `reach.hydrophobic` and `reach.exposed` said "the residues that are both are
    site_patch_positions". They are not. On the main protease the neighbourhood holds **11**
    residues that are both hydrophobic and exposed -- 3, 49, 50, 59, 82, 129, 141, 165, 300,
    303, 305, covering 1,118.5 square angstroms -- and `site_patch_positions` holds **5**, 49,
    50, 300, 303 and 305 covering 586.7. Connectivity is the whole of the difference: the patch
    is the largest connected component on the unit and the other 6 residues belong to other
    components of the 18 this fold carries. A reader quoting the old line would have named the
    wrong residues to a chemist.

    `site_patch_area_a2` described "the hydrophobic patch arithmetic at this location", which
    reads as a search at the site. It is an intersection, not a search, and the consequence is
    one-sided: a smaller patch sitting squarely on the chemistry is invisible. nsp13's helicase
    C-terminal domain and nsp2's C-terminal domain both report nothing at the site, while each
    neighbourhood holds 3 residues that are both hydrophobic and exposed -- 236.4 and 244.7
    square angstroms of surface in components that are not the largest one.

    `chemistry_only` said "Active site and Binding site residues alone", naming a feature-type
    rule. The partition is on the description, which is the measured decision `PROCESSING_TERMS`
    records below, and the difference is visible in this very run: Protein 3's plain `Site` at
    entry 133 of Q0Q474, `Involved in polymerization`, is kept as chemistry and is that chain's
    only annotated residue.

**What the honest absences are, named rather than left as empty rows.** Three chains of the
held-out run carry no annotated site residue whatsoever -- the membrane protein (Q0Q472),
Protein 7a (Q3I5J0) and ORF9b (P59636) -- so there is no functionally important location to
measure on them and the record says exactly that. The spike is the harder case and the more
interesting one: its S1 chain carries a single annotated residue across 641, a cleavage site at
entry 654, and that position falls in an alignment gap, so it has no counterpart in this genome
at all; S2 and S2' carry three cleavage sites between them, all of which map and none of which
lies inside any of the three domains that were folded. The whole spike therefore has no
measurable functional site on any folded unit, and that sentence is the finding. It is also the
reason this record exists separately from the surface markers: a page reading only the patch
numbers would show the spike's domains with a patch area each and no indication that nothing is
known about where on them the chemistry happens.

**Read the confidence of the residues a reader cares about, not of the chain's ends.** The
glossary says to take `anchor_confidence` first and that no figure below it means anything
without it, and the row's block is computed over every anchor -- processing sites included. A
cleavage boundary sits at a mature chain's terminus by construction and a predicted terminus is
disordered, so the full-set figure condemns measurements for a reason they do not depend on. The
main protease is the case: mean 52.8 with 2 of 4 anchors under the folding tool's own threshold,
and the two under it are folded positions 1 and 306 at pLDDT 29 and 20 -- the cleavage
boundaries. Over the dyad alone it is mean 81.0, min 77.0, none under the threshold.
`chemistry_only.anchor_confidence` therefore travels beside the row's own, and over this run the
two headlines differ by a lot: 70 of 106 anchors under the threshold across 20 of the 23
measured units, against 54 of 89 across 10 of the 15 units that carry a chemistry residue at
all. It is not a whitewash -- nsp3's Y domain still has 8 of 8 under it and nsp10 8 of 8, both
predicted at a mean near pLDDT 30 -- which is why both figures are reported and neither replaces
the other.

**Every number here is measured on a prediction and every row now says so.**
`on_predicted_coordinates` and `is_deposited_measurement` are read off
`geometry.markers_from_coordinates`, whose docstring requires that the standing travel with the
numbers and never be pooled with a deposited measurement. Neither key reached this record until
12 Nov 2026: every one of the 37 rows describes an ESMFold prediction -- the 23 that carry a
measurement took it on one and the 14 that carry a refusal refused one -- and none of them
carried the label, so a page joining these rows against the deposited surface markers had no
field with which to tell the two apart. The arithmetic was right and the label was missing,
which is the worse of the two to have.

    python -m hpc_pilots.site_patches

Writes `runs/site_patches.json`.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from . import domains, geometry
from .common import ROOT, content_digest, provenance, write_json

RECORD = ROOT / "runs" / "site_patches.json"
FOLDS = ROOT / "runs" / "domain_folds.json"
STORE = ROOT / "runs" / "domain_folds"

# A plain `Site` whose description names one of these is a point where a protease cuts the
# polyprotein, not a residue that does chemistry a binder could sit against. The partition is
# made on the DESCRIPTION and not on the feature type, and the measurement is why: over the 20
# chains of the held-out run on 2026-11-12, 29 residues carry a plain `Site` and 28 of them name
# a cleavage, so a type-level rule -- keep `Active site` and `Binding site`, discard `Site` --
# would be right 28 times and wrong once. The once is Protein 3's `Involved in polymerization`
# at entry 133 of Q0Q474, which is that chain's only annotated residue; a type rule would have
# left a genuinely functional position out of the one measurement it appears in.
#
# This is an AUTHORED term list, validated against no labelled corpus, in the sense
# `domain_inventory.RELEVANCE_TERMS` and `literature.AXIS_TERMS` are. It is recorded per anchor
# with the term that fired, it partitions a measurement into two that are both reported, and it
# never drops a residue from the record.
PROCESSING_TERMS = ("cleavage", "cleaved", "cleaves")

# Below this pLDDT a residue's position is the model's guess rather than its commitment, so a
# site measured on such residues is a measurement of noise. Stated here only as the fallback:
# the number actually used is read per unit off the fold store's own `confident_plddt`, which is
# what the folding tool applied when it wrote the coordinates. Reading it rather than restating
# it is this repository's standing rule, and `docs/index.html` hand-typing the folding ceiling
# in three places is the defect that rule exists because of.
CONFIDENT_PLDDT = 70.0


def _panels():
    """The notebook's coordinate reader, imported the way `build_site` imports it.

    `notebooks/panels.py` is not a package module and imports matplotlib at module scope, so
    the two environment variables have to be set before the import rather than after it --
    matplotlib picks its backend and its cache directory once. This is the same four-line
    preamble `build_site.build_domain_fold_images` runs, and it is here rather than copied into
    the one caller so that a record and a figure read the same B-factor column through the same
    parser. Two parsers would be two chances for a confidence value on the page to disagree
    with the confidence value in the record.
    """
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-site")
    os.environ.setdefault("MPLBACKEND", "Agg")
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    notebooks = str(ROOT / "notebooks")
    if notebooks not in sys.path:
        sys.path.insert(0, notebooks)
    import panels                                                       # noqa: E402

    return panels


def _backends():
    """Both measurement backends, checked once, or the reason neither measurement can be made.

    Two imports stand behind every row: `notebooks/panels.py` for the confidence column, which
    needs matplotlib at module scope, and freesasa for the accessibility.
    `geometry.markers_from_coordinates` deliberately re-raises an `ImportError` rather than
    recording it -- its docstring says why, that a missing SASA backend is a fact about this
    machine and not about the target, and must not be written into a row as though it were one.

    The same argument says where it should be recorded instead: once, at the top, as a refused
    record. Checking it per unit would write the identical sentence 37 times, which buries the
    one thing a reader needs to know, and letting the import fire inside the loop would abort a
    run after it had already paid for twenty alignments.
    """
    try:
        _panels()
    except Exception as failure:                                        # noqa: BLE001
        return (f"the confidence column cannot be read on this machine: importing "
                f"notebooks/panels.py raised {type(failure).__name__}: {str(failure)[:160]}. "
                f"A site patch reported without the confidence of the residues it was measured "
                f"on is a number a reader cannot judge, so no rows are written.")
    try:
        import freesasa                                                 # noqa: F401
    except ImportError as failure:
        return (f"no solvent-accessibility backend on this machine: {failure}. This is a fact "
                f"about the machine and not about any target, which is why it is recorded here "
                f"once rather than on every row.")
    return None


def _fold_residues(pdb_text):
    """Per-residue confidence and one-letter identity, by folded position.

    Returns `({number: plddt}, {number: letter}, scale_note)`.

    The confidence comes through `panels.ca_trace` and `panels.plddt_on_0_100` because the scale
    is detected and not assumed: this endpoint writes pLDDT into the B-factor column on a 0-to-1
    scale while the bands the page draws run 0 to 100, and reading the column as it stands would
    report the main protease's catalytic histidine at 0.77 of 100 rather than 77 of 100 -- a
    catastrophic-looking prediction that is really a unit error. `plddt_on_0_100` returns the
    sentence describing which branch it took, and that sentence is carried into the record so a
    reader is not left to trust the normalisation.

    The residue identities need a second pass over the same lines, because `ca_trace` returns
    numbers, coordinates and B-factors and not names. The filter below is deliberately written
    to match `ca_trace`'s exactly -- `ATOM`, alpha carbon, alternate location blank or `A` -- so
    the two dictionaries cannot come to describe different sets of atoms. A copy of a filter is
    a liability; a copy of a filter with the reason for the copy written beside it is a smaller
    one, and the alternative is widening a notebook function that the committed figures already
    call. The three-letter-to-one-letter table is `geometry._letter`'s rather than a second copy,
    so a residue named `H` here is the same residue the patch arithmetic calls hydrophobic or
    does not.
    """
    panels = _panels()
    resnum, _xyz, bfac = panels.ca_trace(pdb_text)
    plddt, note = panels.plddt_on_0_100(bfac)
    confidence = {int(n): round(float(v), 1) for n, v in zip(resnum, plddt)}
    letters = {}
    for line in pdb_text.splitlines():
        if line.startswith("ATOM") and line[12:16].strip() == "CA" and line[16] in (" ", "A"):
            letters[int(line[22:26])] = geometry._letter(line[17:20])
    return confidence, letters, note


def _reading_frame(located, records, declared):
    """Which open reading frame of this genome the row's middle numbering runs along.

    `in_this_genome` is 1-based along one ORF and not along the genome, so a span here is
    meaningless without the frame it is a span in -- and until this block existed the record
    carried the span and not the frame. Measured over the 37 rows of this run on 2026-11-12,
    50 pairs of them have numerically overlapping spans in different frames: spike S1's
    [24, 312] and the nucleoprotein's [41, 186] share no residue and would sort, join or plot
    as though they did.

    **The pool index is not the identifier, which is why it is not the only key here.**
    `orf_index` is a position in the twelve longest ORFs of this genome, so it identifies a
    frame only to a reader who can rebuild the same pool in the same order -- a different
    `ORF_POOL`, a curated genome, or a segmented one called in another order renumbers it
    silently. `strand`, `frame` and `nt_start` come off `intake.orfs` and are the frame itself:
    the strand the ORF is read on, which of the three codon offsets it starts at, and where its
    first base sits. `reading_frame` is those three in one string for a page or a filename.

    `declared` is the folds record's own `found_in_orf` for this chain. This module aligns the
    chain a second time -- `run` re-locates every chain rather than trusting a span -- so the
    two can differ, and a difference means these rows are numbered along one frame while the
    coordinates they were measured on were cut out of another. That is a silent wrong answer
    rather than a missing one, so the comparison is a key and not an assumption. None where the
    folds record carries no frame for the chain, because "they disagree" and "there was nothing
    to compare" are different statements.
    """
    index = (located or {}).get("orf_index")
    orf = records[index] if records and index is not None and 0 <= index < len(records) else {}
    strand, frame, start = orf.get("strand"), orf.get("frame"), orf.get("nt_start")
    return {
        "orf_index": index,
        "strand": strand,
        "frame": frame,
        "nt_start": start,
        "orf_length_aa": orf.get("length_aa"),
        "reading_frame": (f"{strand}{frame} at nt {start + 1}"
                          if None not in (strand, frame, start) else None),
        "found_in_orf_on_the_folds_record": declared,
        "agrees_with_the_folds_record": (None if declared is None or index is None
                                         else int(declared) == int(index)),
    }


def _agreement(found=(), expected=None, observed=None):
    """`measured_at_the_annotated_residues` and its six companions, from the one function.

    `chain._residue_agreement` is the comparison, imported here rather than reimplemented:
    this record already answers the same question per anchor as `agrees_with_the_relative`, and
    a page needs it per row to decide whether a figure may be printed as a measurement of the
    annotated site. Two copies of the comparison would agree until one was edited.

    Imported inside the call because `chain` is the largest module in the package and this one
    is read by a page build that has no other reason to load it -- the same late-import rule
    `domains._features` follows for `intake`. Called with no arguments from `_row_skeleton`, so
    an unmeasured row carries the seven keys at their neutral values and the key set holds.
    """
    from .chain import _residue_agreement

    return _residue_agreement(list(found), expected, observed)


def _processing_term(description):
    """Which of `PROCESSING_TERMS` this description names, or None."""
    text = (description or "").lower()
    return next((term for term in PROCESSING_TERMS if term in text), None)


def _count(number, noun):
    """`1 annotated site residue` or `3 annotated site residues`.

    Here because the refusal sentences below are read by a human off the page, not by a parser,
    and `all 1 annotated site residue(s) fall` is the kind of sentence that makes a reader
    distrust the number beside it. Four of this run's twenty chains carry exactly one annotated
    residue -- both spike chains that have any, Protein 3 and nsp1 -- so the singular is a
    quarter of the cases and not an edge.
    """
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _anchor_rows(sites, located, chain_start, unit_span, entry_sequence):
    """Every annotated site residue of one chain, carried into one folded unit's numbering.

    Returns `(inside, outside, unmapped)`. A residue appears in exactly one of the three, so no
    annotated position is lost between them and the three counts add up to what the chain
    carries -- which is the property that lets the record state an absence as an absence rather
    than as a row that happens to be short.

    `inside` rows carry the position in all three numbering systems, the residue the relative
    has there and the residue this genome has, and whether those two agree. The disagreement is
    not an error and must not be read as one: the annotation was curated on a different species,
    so a substitution at an annotated position is biology the alignment found, and a page that
    hid it would be claiming the relative's residue for this pathogen. The main protease's dyad
    agrees on both residues; where a row says otherwise, the site is conserved in position and
    not in identity, which is a different and weaker claim.

    `outside` is a residue of this chain that falls beyond this unit's span -- ordinary for a
    domain, which is a window onto a chain and not the whole of it. `unmapped` is a residue
    whose position has no counterpart in this genome, which `domains.map_position` answers with
    None and this function passes on as None rather than filling in a neighbour. The spike S1
    chain's single annotated residue is one of these, and substituting the residue next door
    would have turned "nothing is known about where the chemistry is on this chain" into a
    confident measurement at the wrong address.
    """
    low, high = unit_span
    inside, outside, unmapped = [], [], []
    for site in sites:
        term = _processing_term(site.get("description"))
        for position in site.get("positions") or ():
            row = {
                "feature_type": site.get("type"),
                "description": site.get("description") or "",
                "ligand": site.get("ligand"),
                "is_a_processing_site": bool(term),
                "processing_term_matched": term,
                "on_the_relative": position,
                "residue_on_the_relative": (entry_sequence[position - 1]
                                            if entry_sequence
                                            and 0 < position <= len(entry_sequence) else None),
            }
            in_genome = domains.map_position(located, position - chain_start)
            if in_genome is None:
                row["in_this_genome"] = None
                row["in_the_fold"] = None
                unmapped.append(row)
                continue
            row["in_this_genome"] = in_genome
            if not low <= in_genome <= high:
                row["in_the_fold"] = None
                outside.append(row)
                continue
            row["in_the_fold"] = in_genome - low + 1
            inside.append(row)
    return inside, outside, unmapped


def _predicted_standing(markers=None):
    """`on_predicted_coordinates` and `is_deposited_measurement`, carried onto every row.

    `geometry.markers_from_coordinates` attaches both keys to everything it returns, including
    its own error returns, and its docstring is explicit about why: a patch computed on a
    prediction inherits every error of the prediction, it "is reported beside a deposited
    measurement, never pooled with one", and the standing has to travel with the numbers rather
    than be available somewhere else. Neither key reached this record. The arithmetic was right
    and the label was missing, which is the worse of the two defects to have -- a page joining
    these rows against the deposited surface markers would have found no field saying which was
    which, while every one of the 37 rows here describes an ESMFold prediction -- 23 of them a
    measurement taken on one and 14 a refusal to take one.

    Read off the markers wherever a measurement was attempted, so this record and the geometry
    module cannot come to disagree about one measurement's standing. Stated only where the
    arithmetic was never reached at all -- a unit with no coordinates on this machine, a chain
    that never reached coordinates -- and stating it there asserts nothing: every unit of this
    record comes from `runs/domain_folds`, every file in that store is a prediction, and no
    deposited coordinate enters this module by any route. A deposited row is a row this module
    cannot produce, which is why the standing is a constant here and a measurement in
    `geometry`.
    """
    if markers is not None:
        return {"on_predicted_coordinates": markers.get("on_predicted_coordinates"),
                "is_deposited_measurement": markers.get("is_deposited_measurement")}
    return {"on_predicted_coordinates": True, "is_deposited_measurement": False}


def _reach_and_patch(pdb_text, anchors, slug):
    """The stage-4 arithmetic, asked at the anchors rather than at the chain's best address.

    One call to `geometry.markers_from_coordinates`, which runs the accessibility and the patch
    search over the whole unit and then intersects the result with the neighbourhood around the
    anchors. The intersection and not a fresh search inside the neighbourhood is the point, and
    the argument is `geometry._markers`' own: accessibility is a property of a residue in its
    surroundings, so re-growing patches inside a 13.8-angstrom ball would report a surface
    assembled from faces the whole structure keeps buried.

    Returns `(reach, site_patch, unit_patch, standing, error)` with the geometry module's own
    key names kept verbatim inside each block. Renaming them here would mean a page quoting
    `site_patch_fraction_of_the_site` could not be grepped back to the function that computed
    it, and this record exists to be checkable. `standing` comes back on the error path too,
    because `markers_from_coordinates` attaches it there as well: coordinates that could not be
    read are still predicted coordinates, and a row saying so must not lose the label along
    with the number.
    """
    markers = geometry.markers_from_coordinates(pdb_text, label=slug, anchors=anchors)
    standing = _predicted_standing(markers)
    if markers.get("error"):
        return None, None, None, standing, markers["error"]
    reach = markers.get("site")
    whole = markers.get("hydrophobic_patch_area") or 0.0
    at_site = markers.get("site_patch_area_a2") or 0.0
    site_patch = {
        "site_patch_area_a2": markers.get("site_patch_area_a2"),
        "site_patch_positions": markers.get("site_patch_positions"),
        "site_patch_fraction_of_the_site": markers.get("site_patch_fraction_of_the_site"),
        # Frequently false, and that is the reading rather than a failure: it says the surface
        # stage 4 scored this unit on is somewhere other than the chemistry.
        "site_shares_the_patch": markers.get("site_shares_the_patch"),
        # Derived here and not in `geometry`, because it is the comparison this record exists
        # to make and not a property of one measurement: how much of the patch stage 4 scored
        # the unit on is actually at the chemistry. `site_shares_the_patch` answers the same
        # question as a yes or no and is true on 21 of the 23 measured units, which is too
        # coarse to act on -- the main protease's catalytic dyad holds 9% of its chain's best
        # patch and nsp12's RdRp catalytic site holds 54%, and both are "true".
        "site_share_of_the_largest_patch": round(at_site / whole, 3) if whole else 0.0,
    }
    unit_patch = {
        "hydrophobic_patch_area": markers.get("hydrophobic_patch_area"),
        "hydrophobic_patch_residues": markers.get("hydrophobic_patch_residues"),
        "patch_positions": markers.get("patch_positions"),
        "patch_residue_names": markers.get("patch_residue_names"),
        "patches_found": markers.get("patches_found"),
        "exposed_fraction": markers.get("exposed_fraction"),
        "total_sasa": markers.get("total_sasa"),
        "residues": markers.get("residues"),
    }
    return reach, site_patch, unit_patch, standing, None


def _confidence_block(anchors, confidence, threshold, note):
    """How much the model believed in the residues the site was measured on.

    This is the number that decides whether the rest of the row means anything. A patch measured
    around residues the model placed at pLDDT 30 is a patch around coordinates the model never
    committed to, and the arithmetic cannot tell the difference -- freesasa will return an area
    for any geometry. So the confidence travels with the figures instead of being available
    somewhere else, and `below_the_confident_threshold` is the count a reader should check
    before quoting the area.

    The threshold is the folding tool's own `confident_plddt` as it stood when the coordinates
    were written, read off the fold store by the caller; `CONFIDENT_PLDDT` is only the fallback
    for a store that does not carry it.
    """
    values = [a["plddt"] for a in anchors if a.get("plddt") is not None]
    if not values:
        return {"anchors_with_a_confidence": 0, "mean_plddt": None, "min_plddt": None,
                "max_plddt": None, "confident_plddt": threshold,
                "below_the_confident_threshold": 0, "plddt_scale": note}
    below = [v for v in values if v < threshold]
    return {
        "anchors_with_a_confidence": len(values),
        "mean_plddt": round(sum(values) / len(values), 1),
        "min_plddt": min(values),
        "max_plddt": max(values),
        "confident_plddt": threshold,
        "below_the_confident_threshold": len(below),
        "plddt_scale": note,
    }


def _row_skeleton(chain, label, reason=None, site_residues=None, frame=None):
    """Every key a row of this record carries, with nothing measured and nothing assumed.

    Both row builders start here, and the defect that made that necessary is countable. The
    chain-level fallback row -- written for a chain with `not_folded_because`, or with no
    reviewed accession, or whose entry could not be read -- carried **8 keys against the 27**
    every unit row carried, missing 19 of them including `anchors`, `positions_in_the_fold`,
    `reach`, `site_patch`, `unit_patch`, `anchor_confidence` and `chemistry_only`. Reproduced
    on 12 Nov 2026 by forcing `not_folded_because` onto the first chain of the recorded run and
    re-running: 35 rows, 34 of them 27 keys wide and one of them 8. A consumer joining this
    record's rows on any of those keys therefore found a value on every row but one, which
    surfaces as a crash in a page template rather than as a number a reader could question.

    The nulls are explicit, and they are nulls rather than the chain's own figures on purpose.
    A chain-level row does have a real `chain_range_on_the_relative`, and it would fit
    `entry_span_on_the_relative` neatly, but that key means *this unit's* span on every other
    row -- so filling it with a chain range would make a chain that was never folded look like
    a measured unit of itself. An explicit null says the unit does not exist; a plausible
    number says it was measured and came out this way. `site_residues` is the one exception and
    only because it is chain-level on every row already: it is passed where the annotation was
    actually read and left null where the function returned before reading it, so the key says
    "not known here" rather than "none".

    The unmeasured defaults are the ones an unmeasured unit row already carried, verbatim --
    empty lists for the three anchor lists and the three position lists, False for `measured`
    and `anchors_are_processing_sites_only`, None for the four measurement blocks -- so parity
    holds by construction instead of by two literals being kept in step by hand.
    """
    return {
        "rank": chain.get("rank"),
        "target": chain.get("target"),
        "nearest_homolog": chain.get("nearest_homolog"),
        "slug": None,
        "kind": None,
        "label": label,
        "region_type": None,
        "description": None,
        "entry_span_on_the_relative": None,
        "span_in_this_genome": None,
        # Which frame that span is a span IN, without which it is a pair of integers. Passed
        # where the chain was located and None on a row that never reached an alignment, so the
        # key says "not known here" rather than naming a frame nothing was read from.
        "reading_frame": frame,
        "fold_length_aa": None,
        "mean_plddt": None,
        "site_residues_in_the_chain": site_residues,
        "anchors": [], "anchors_outside_this_unit": [], "anchors_unmapped": [],
        "positions_on_the_relative": [], "positions_in_this_genome": [],
        "positions_in_the_fold": [],
        "measured": False, "not_measured_because": reason,
        "reach": None, "site_patch": None, "unit_patch": None,
        "anchor_confidence": None,
        "anchors_are_processing_sites_only": False,
        "chemistry_only": None,
        # The row-level verdict on whether the anchors are the residues the annotation names,
        # through the one function that answers it for both lanes. Neutral here -- None rather
        # than False, because an unmeasured row has nothing to assert either way.
        **_agreement(),
        # Last, so a reader scanning a row meets the figures before the standing that governs
        # how far they travel -- and present on every row, including this skeleton's unmeasured
        # ones, because the key set is the whole point of the function.
        **_predicted_standing(),
    }


def _unit_row(chain, unit, located, chain_start, sites, entry_sequence, frame=None):
    """One folded unit, measured at its functional site or recorded as having none."""
    label = unit.get("label")
    row = _row_skeleton(chain, label, frame=frame,
                        site_residues=sum(len(s.get("positions") or ()) for s in sites))
    row.update({
        "slug": unit.get("slug"),
        "kind": unit.get("kind"),
        "region_type": unit.get("region_type"),
        "description": unit.get("description"),
        "entry_span_on_the_relative": unit.get("entry_span_on_the_relative"),
        "span_in_this_genome": unit.get("span_in_this_genome"),
        "fold_length_aa": unit.get("length_aa"),
        "mean_plddt": unit.get("mean_plddt"),
    })
    if not unit.get("folded") or not unit.get("slug"):
        row["not_measured_because"] = (
            f"this unit has no coordinates: {unit.get('refused_because') or 'it was not folded'}")
        return row
    pdb = STORE / f"{unit['slug']}.pdb"
    store = STORE / f"{unit['slug']}.json"
    if not pdb.exists():
        row["not_measured_because"] = (
            f"the record says this unit was folded but {pdb.name} is not on this machine. The "
            f"coordinates under runs/ are gitignored, so a checkout without a local fold run "
            f"reaches this branch and it is a fact about the machine, not about the target.")
        return row
    span = unit.get("span_in_this_genome") or []
    if len(span) != 2:
        row["not_measured_because"] = (
            "this unit carries no span in this genome, so an annotated position cannot be "
            "carried into the coordinates' own numbering.")
        return row
    pdb_text = pdb.read_text()
    confidence, letters, scale = _fold_residues(pdb_text)
    threshold = CONFIDENT_PLDDT
    if store.exists():
        threshold = json.loads(store.read_text()).get("confident_plddt") or CONFIDENT_PLDDT
    inside, outside, unmapped = _anchor_rows(sites, located, chain_start, span, entry_sequence)
    for anchor in inside:
        position = anchor["in_the_fold"]
        anchor["residue_in_the_fold"] = letters.get(position)
        anchor["plddt"] = confidence.get(position)
        anchor["agrees_with_the_relative"] = (
            bool(anchor.get("residue_on_the_relative"))
            and anchor["residue_in_the_fold"] == anchor["residue_on_the_relative"])
    row["anchors"] = inside
    row["anchors_outside_this_unit"] = outside
    row["anchors_unmapped"] = unmapped
    # The same per-anchor comparison the loop above wrote, reduced to the row-level verdict a
    # consumer has to gate on. Keyed on the folded position because that is the numbering the
    # anchors were handed to the arithmetic in; `_residue_agreement` wants the relative's
    # position alongside the residue so a disagreement names both numberings.
    row.update(_agreement(
        [a["in_the_fold"] for a in inside],
        {a["in_the_fold"]: (a["residue_on_the_relative"], a["on_the_relative"])
         for a in inside if a.get("residue_on_the_relative")},
        {a["in_the_fold"]: a.get("residue_in_the_fold") for a in inside}))
    row["positions_on_the_relative"] = [a["on_the_relative"] for a in inside]
    row["positions_in_this_genome"] = [a["in_this_genome"] for a in inside]
    # Ask 3's "list of AA positions", in the numbering the coordinates actually use. This is the
    # list a reader opens a structure viewer with, and it is the only one of the three that can
    # be typed into one.
    row["positions_in_the_fold"] = [a["in_the_fold"] for a in inside]

    if not inside:
        row["not_measured_because"] = _absence(sites, outside, unmapped, row["kind"], label)
        return row

    anchors = row["positions_in_the_fold"]
    reach, site_patch, unit_patch, standing, error = _reach_and_patch(
        pdb_text, anchors, unit["slug"])
    # Read off the markers rather than restated, and on the error path as well as the measured
    # one: `markers_from_coordinates` labels its own failures too, and a row that lost the
    # label while keeping the refusal would be a row a page could not place.
    row.update(standing)
    if error:
        row["not_measured_because"] = f"the coordinates could not be measured: {error}"
        return row
    row.update({"measured": True, "reach": reach, "site_patch": site_patch,
                "unit_patch": unit_patch,
                "anchor_confidence": _confidence_block(inside, confidence, threshold, scale)})
    row["anchors_are_processing_sites_only"] = all(a["is_a_processing_site"] for a in inside)
    row["chemistry_only"] = _chemistry_only(pdb_text, inside, unit["slug"], threshold, scale)
    return row


def _absence(sites, outside, unmapped, kind, label):
    """Why one unit has no functional site to measure at, in the terms that distinguish them.

    Four different absences, and collapsing them into "no site" would throw away the only part
    of this record a reader can act on. A chain with no annotation at all is a gap in what has
    been curated; a site that falls outside a domain is a reason to fold a different domain; a
    site with no counterpart in this genome is an indel the alignment found and a reason to
    trust nothing about that position; and a unit whose own span never mapped is a failure of
    the pipeline rather than of the biology.
    """
    carried = sum(len(s.get("positions") or ()) for s in sites)
    if not carried:
        return ("the relative's entry carries no Active site, Binding site or Site annotation "
                "for this chain, so there is no functionally important location to measure at. "
                "This is an absence in what has been curated, not a measurement: a residue "
                "that does chemistry here may exist and be unannotated.")
    noun = _count(carried, "annotated site residue")
    if unmapped and not outside:
        each = "it falls" if carried == 1 else "every one of them falls"
        return (f"none of the {noun} this chain carries has a counterpart in this genome -- "
                f"{each} in an alignment gap. The positions are recorded under "
                f"`anchors_unmapped` on the relative's numbering; substituting the nearest "
                f"mapped neighbour would name a residue the curator did not annotate.")
    where = "this domain" if kind == "domain" else "this chain's folded window"
    verb = "lies" if carried == 1 else "lie"
    gaps = (f" {len(unmapped)} of them "
            f"{'falls' if len(unmapped) == 1 else 'fall'} in an alignment gap with no "
            f"counterpart in this genome." if unmapped else "")
    return (f"the {noun} this chain carries {verb} outside {where} ({label}).{gaps} A domain "
            f"is a window onto a chain and the chemistry is not obliged to fall inside the "
            f"window that was folded.")


def _chemistry_only(pdb_text, inside, slug, threshold, scale):
    """The same measurement over the residues that do chemistry, processing sites dropped.

    Reported beside the full-set figures and never instead of them, because the two answer
    different questions and only one of them is ask 3's. A cleavage site is a point where a
    protease cuts the polyprotein; it sits at a mature chain's terminus by construction, and a
    neighbourhood drawn around it is a neighbourhood of a dangling end. The main protease is the
    clean illustration: four annotated residues, of which positions 1 and 306 are the cleavage
    boundaries of the chain itself and only 41 and 145 are the catalytic dyad, so the full set
    measures a neighbourhood spanning both termini and the dyad at once.

    **`anchor_confidence` is here because the row's own figure can condemn a measurement for
    the wrong reason.** `how_to_read` tells a reader to take the confidence first and that no
    figure below it means anything without it, and until this was added the only confidence in
    the row was computed over every anchor including the processing sites. The main protease
    again, measured 12 Nov 2026: over all four anchors the confidence is mean 52.8, min 20.0,
    2 of 4 below the folding tool's own threshold -- and the two below it are the cleavage
    boundaries at folded positions 1 and 306, carrying pLDDT 29 and 20 because a predicted
    chain terminus is disordered and that is a fact about chain ends rather than about the
    chemistry. Over the dyad alone it is mean 81.0, min 77.0, none below the threshold. So the
    reader was being told the measurement was untrustworthy on the strength of two residues the
    measurement they care about does not use.

    It is not a whitewash and the run says so. Over all 23 measured units the full set puts 70
    of 106 anchors below the threshold across 20 of them; the chemistry residues alone put 54
    of 89 below it across 10 of the 15 units that carry one at all. Four units have a strict
    chemistry subset with a confidence of its own, and the verdict flips in two: nsp5 from 2
    anchors below to 0, nsp13's zinc-binding domain from 1 of 13 to 0 of 12. In the other two
    it does not move -- nsp3's Y domain goes from 9 of 9 below to 8 of 8 and nsp10 from 10 of
    10 to 8 of 8, both still predicted at a mean near pLDDT 30 -- which is why both figures are
    reported rather than one replacing the other.

    `same_as_the_full_set` is true whenever the subset is the whole set, and then no second
    arithmetic is run and `reach`, `site_patch` and `anchor_confidence` are None -- the row's
    own figures are the answer and duplicating them would create two numbers that a later edit
    could separate. That is 11 of the 15 units with a chemistry residue, so the None is the
    common case and not an edge.
    """
    chemistry = [a for a in inside if not a["is_a_processing_site"]]
    anchors = [a["in_the_fold"] for a in chemistry]
    block = {
        "anchors_in_the_fold": anchors,
        "anchors_on_the_relative": [a["on_the_relative"] for a in chemistry],
        "feature_types": sorted({a["feature_type"] for a in chemistry}),
        "processing_sites_dropped": len(inside) - len(chemistry),
        "same_as_the_full_set": len(chemistry) == len(inside),
        "measured": False, "not_measured_because": None,
        "reach": None, "site_patch": None, "anchor_confidence": None,
    }
    if not anchors:
        block["not_measured_because"] = (
            f"every one of this unit's {len(inside)} annotated residue(s) is a proteolytic "
            f"processing site, so the figures above measure a neighbourhood around a chain "
            f"terminus rather than around a binding location.")
        return block
    if block["same_as_the_full_set"]:
        block["measured"] = True
        return block
    reach, site_patch, _unit, _standing, error = _reach_and_patch(pdb_text, anchors, slug)
    if error:
        block["not_measured_because"] = f"the coordinates could not be measured: {error}"
        return block
    block.update({"measured": True, "reach": reach, "site_patch": site_patch,
                  "anchor_confidence": _confidence_block(chemistry, None, threshold, scale)})
    return block


def _shown(out):
    """The written path as a reader should see it: tree-relative inside the repository, absolute
    outside it.

    `out.relative_to(ROOT)` raises `ValueError` for a path that is not under the repository, and
    both completion prints below call it on a `--path` the caller chose. The ordering is what
    made this a defect rather than a cosmetic bug: `write_json` has already run by then, so the
    record on disk is complete and correct, and the process dies in the sentence announcing it.
    A successful run exits 1 with a traceback. Reproduced 12 Nov 2026 by writing this record
    into a scratch directory, which is exactly where a verification run writes it:
    `ValueError: '.../sp_out_probe.json' is not in the subpath of '.../bind-the-unknown'`, with
    the full 37-row record sitting on disk beside the traceback.

    `literature_stage.run` carries the same guard inline, having hit the same thing on the same
    day. This is a function rather than two more copies because both records print a path, and
    the refused one is the likelier of the two to be written outside the tree -- it is the
    branch a machine with no SASA backend takes, which is the machine someone probes with a
    scratch `--path` in the first place.
    """
    try:
        return out.relative_to(ROOT)
    except ValueError:
        return out


def _refused(payload_in, reason, record, path):
    """A record saying nothing could be measured, and why, rather than no record at all.

    The same shape as a complete one, so a page reading this record does not need a second code
    path for the machine that has no SASA backend: `rows` is empty, `status` is `refused` and
    `answer` is the reason. A page that crashed on a missing key here would be a page whose
    behaviour depends on which machine built it, which is the failure this repository records
    absences to avoid.
    """
    payload = {
        "experiment": "site_patches",
        "status": "refused",
        "question": "At the residues the annotation says do the chemistry, what can a "
                    "macrocycle reach, and is there a hydrophobic patch there?",
        "answer": reason,
        "genome_accession": payload_in.get("genome_accession"),
        "case": payload_in.get("case"),
        "reference_excluded_organism": payload_in.get("reference_excluded_organism"),
        "reach_a": geometry.MACROCYCLE_REACH_A,
        "macrocycle_footprint_a2": geometry.MACROCYCLE_FOOTPRINT_A2,
        "folds_record_sha256": payload_in.get("content_sha256"),
        # The standing of a measurement that was never made, stated once here because this
        # record has no rows to carry it on. A page that reads the standing per row would learn
        # nothing at all from a refused record, and "no measurement" must not be mistaken for
        # "a deposited measurement".
        **_predicted_standing(),
        "summary": {"chains": 0, "units": 0, "units_measured": 0},
        "rows": [],
        "limitations": [reason],
        "provenance": provenance(),
    }
    payload["content_sha256"] = content_digest(payload)
    if record:
        out = Path(path) if path else RECORD
        write_json(out, payload)
        print(f"wrote {_shown(out)}  (refused: {reason[:90]})")
    return payload


def run(folds=None, record=True, path=None):
    """Measure every folded unit at its functional site, and record every absence with why."""
    from .intake import _aligner, _entry
    from . import domain_folds

    payload_in = json.loads(Path(folds or FOLDS).read_text())
    missing = _backends()
    if missing:
        return _refused(payload_in, missing, record, path)
    # The records and not only the translations, because every row's middle numbering runs
    # along one of these frames and `_reading_frame` names it from the strand, the frame offset
    # and the nucleotide start rather than from a position in this list.
    records = domain_folds._orf_records(payload_in.get("genome_accession"))
    pool = [orf["protein"] for orf in records]
    aligner, cache = _aligner(), {}

    rows, failed_chains = [], []
    for chain in payload_in.get("chains") or ():
        accession = chain.get("nearest_homolog")
        span = chain.get("chain_range_on_the_relative")
        # Every chain-level refusal below goes out through `_row_skeleton`, so it carries the
        # same key set a unit row carries. These three branches used to share a seven-key
        # `base` dict, which made the fallback row 8 keys wide against 27 and left a consumer
        # joining on `reach` or `chemistry_only` with nothing on that one row.
        if chain.get("not_folded_because") or not accession:
            # A chain that never reached coordinates still gets a row. The alternative is a
            # table whose length silently depends on how the fold run went, which is the defect
            # `domain_folds` records its own refusals to avoid.
            rows.append(_row_skeleton(
                chain, chain.get("target"),
                chain.get("not_folded_because")
                or "no reviewed homolog accession, so no curated annotation exists to read."))
            failed_chains.append(chain.get("target"))
            continue
        try:
            reference = domain_folds._reference(accession, span, cache)
            located = domain_folds._locate(reference, pool, aligner)
            sites = domains.annotated_sites(accession, within=tuple(span) if span else None)
            entry_sequence = _entry(accession)["sequence"]["value"]
        except Exception as failure:                                    # noqa: BLE001
            # `site_residues_in_the_chain` stays null here and only here: the annotation is read
            # inside this `try`, so on this branch it is genuinely not known rather than zero.
            rows.append(_row_skeleton(
                chain, chain.get("target"),
                f"the relative's entry or this genome's reading frames could not be read: "
                f"{type(failure).__name__}: {str(failure)[:160]}"))
            failed_chains.append(chain.get("target"))
            continue
        if not located:
            rows.append(_row_skeleton(
                chain, chain.get("target"),
                f"the relative's {len(reference)}-residue chain aligned to none of the "
                f"{len(pool)} reading frames searched, so no annotated position can be carried "
                f"into this genome's numbering.",
                site_residues=sum(len(s.get("positions") or ()) for s in sites)))
            failed_chains.append(chain.get("target"))
            continue
        # The same convention `domains.map_span` documents: entry position p is index
        # p - chain_start in the sequence that was aligned, and a chain with no annotated range
        # was aligned as its whole entry, whose index 0 is entry position 1.
        chain_start = span[0] if span else 1
        frame = _reading_frame(located, records, chain.get("found_in_orf"))
        for unit in chain.get("units") or ():
            rows.append(_unit_row(chain, unit, located, chain_start, sites, entry_sequence,
                                  frame=frame))

    measured = [r for r in rows if r.get("measured")]
    siteless = [r for r in rows if not r.get("measured")]
    at_the_site = [r for r in measured if (r["site_patch"] or {}).get("site_shares_the_patch")]
    processing = [r for r in measured if r.get("anchors_are_processing_sites_only")]
    chemistry = [r for r in measured
                 if (r.get("chemistry_only") or {}).get("measured")]
    differs = [r for r in chemistry
               if not (r.get("chemistry_only") or {}).get("same_as_the_full_set")]
    disagreements = [a for r in measured for a in r["anchors"]
                     if a.get("residue_on_the_relative")
                     and not a.get("agrees_with_the_relative")]
    low = [r for r in measured
           if (r["anchor_confidence"] or {}).get("below_the_confident_threshold")]
    chains = {r["target"] for r in rows}
    chains_without = {r["target"] for r in rows
                      if not r.get("site_residues_in_the_chain")
                      and r.get("kind") is not None}
    # Counted over DISTINCT chain-and-entry-position pairs and not summed over units, because a
    # chain with three folded domains reports every one of its annotated residues as outside
    # two of them. Summing the per-unit lists gave 154 residues outside their unit and 6 with no
    # counterpart in this genome, against 55 and 3 distinct ones -- the first pair of numbers
    # counts rows and the second counts residues, and only the second is a fact about the
    # annotation. A site inside any folded unit of its chain is not in either count.
    anywhere, nowhere, ungapped = set(), set(), set()
    for row in rows:
        key = row.get("rank")
        for anchor in row.get("anchors") or ():
            anywhere.add((key, anchor["on_the_relative"]))
        for anchor in row.get("anchors_outside_this_unit") or ():
            nowhere.add((key, anchor["on_the_relative"]))
        for anchor in row.get("anchors_unmapped") or ():
            ungapped.add((key, anchor["on_the_relative"]))
    nowhere -= anywhere
    ungapped -= anywhere
    anchors_total = sum(len(r["anchors"]) for r in measured)
    anchors_low = sum((r["anchor_confidence"] or {}).get("below_the_confident_threshold") or 0
                      for r in measured)
    # The same confidence counted over the chemistry residues alone. The headline figures above
    # are computed over every anchor including the proteolytic processing sites, and a predicted
    # chain terminus is disordered by construction, so they overstate the problem: 70 of 106
    # anchors below the threshold across 20 of the 23 measured units, against 54 of 89 across 10
    # of the 15 units that carry a chemistry residue at all. Both are reported, because the
    # second is not an improvement on the first -- it is the answer to a different question, and
    # `how_to_read` now names which question each one settles.
    #
    # Where the chemistry subset IS the full set the row's own block is already the chemistry
    # block, which is the convention `_chemistry_only` documents and the reason the fallback
    # here is the row rather than a zero.
    chemistry_anchors = chemistry_low = chemistry_units_low = 0
    for unit_row in chemistry:
        block = (unit_row["chemistry_only"].get("anchor_confidence")
                 or unit_row["anchor_confidence"] or {})
        chemistry_anchors += block.get("anchors_with_a_confidence") or 0
        below = block.get("below_the_confident_threshold") or 0
        chemistry_low += below
        chemistry_units_low += 1 if below else 0

    # What the reading-frame block buys, counted rather than asserted. `overlapping` is the
    # hazard it exists for: two rows whose spans overlap as integers while being numbered along
    # different frames, which is two windows with no residue in common that a join or a sort on
    # the span alone places on top of each other. Counted over unordered pairs of rows that
    # carry both a span and a frame.
    frames = sorted({(r.get("reading_frame") or {}).get("orf_index") for r in rows
                     if (r.get("reading_frame") or {}).get("orf_index") is not None})
    disagreeing = [r for r in rows
                   if (r.get("reading_frame") or {}).get("agrees_with_the_folds_record") is False]
    spanned = [r for r in rows
               if len(r.get("span_in_this_genome") or ()) == 2
               and (r.get("reading_frame") or {}).get("orf_index") is not None]
    overlapping = 0
    for i, first in enumerate(spanned):
        for second in spanned[i + 1:]:
            if (first["reading_frame"]["orf_index"]
                    == second["reading_frame"]["orf_index"]):
                continue
            # Named away from `low`, which is this function's list of units carrying a
            # low-confidence anchor. Shadowing it here left the headline calling `len` on an
            # integer -- a crash in the sentence announcing a record that was already correct
            # on disk, which is the same shape of defect `_shown` exists for.
            first_low, first_high = first["span_in_this_genome"]
            second_low, second_high = second["span_in_this_genome"]
            if first_low <= second_high and second_low <= first_high:
                overlapping += 1
    # The row-level verdict, counted both ways. A measured unit whose anchors are not the
    # annotated residues is a unit whose figures describe some other location, and the count of
    # them is the number a page has to act on; `not_checked` is the third state and is not a
    # failure of the check -- it is the relative's entry having no residue at an annotated
    # position, or the coordinates naming one `geometry._letter` does not.
    at_annotated = [r for r in measured
                    if r.get("measured_at_the_annotated_residues") is True]
    not_annotated = [r for r in measured
                     if r.get("measured_at_the_annotated_residues") is False]

    payload = {
        "experiment": "site_patches",
        "status": "complete",
        "question": "At the residues the annotation says do the chemistry -- rather than "
                    "wherever on a structure the largest patch happens to lie -- how many "
                    "residues can a macrocycle reach, and is there a hydrophobic patch there?",
        "answer": (f"{len(measured)} of the {len(rows)} units carry at least one annotated "
                   f"site residue inside the coordinates that were folded, and the patch at "
                   f"that location is measured for each. The largest patch of the unit is at "
                   f"the site in {len(at_the_site)} of them. {len(processing)} units have "
                   f"nothing but proteolytic processing sites in them, so their figures "
                   f"describe a chain terminus; {len(rows) - len(measured)} units have no "
                   f"measurable site at all and each says why. Read the confidence first: "
                   f"{anchors_low} of the {anchors_total} anchors sit below the folding tool's "
                   f"own confident threshold, so for {len(low)} of the {len(measured)} "
                   f"measured units at least one residue the site was measured on is a "
                   f"position the model did not commit to. Over the chemistry residues alone "
                   f"-- processing sites dropped, which are chain termini and disordered by "
                   f"construction -- it is {chemistry_low} of {chemistry_anchors} anchors "
                   f"across {chemistry_units_low} of the {len(chemistry)} units that carry "
                   f"one, and that is the figure to read beside `chemistry_only`. "
                   f"{len(at_annotated)} of the {len(measured)} measured units are measured at "
                   f"the residues the annotation names and "
                   f"{len(not_annotated)} are not, which is the gate a page must pass a figure "
                   f"through before printing it as this site's. The rows are numbered along "
                   f"{len(frames)} reading frames of this genome, named per row, because "
                   f"{overlapping} pairs of them carry spans that overlap as integers while "
                   f"sharing no residue."),
        "genome_accession": payload_in.get("genome_accession"),
        "case": payload_in.get("case"),
        "reference_excluded_organism": payload_in.get("reference_excluded_organism"),
        "reach_a": geometry.MACROCYCLE_REACH_A,
        "macrocycle_footprint_a2": geometry.MACROCYCLE_FOOTPRINT_A2,
        # The record-wide standing, stated once as well as per row. Every unit here comes from
        # `runs/domain_folds`, every file in that store is an ESMFold prediction, and nothing in
        # this record is a deposited measurement -- so a consumer that reads the record's header
        # and a consumer that reads a row arrive at the same answer instead of one of them
        # arriving at none.
        **_predicted_standing(),
        "folds_record_sha256": payload_in.get("content_sha256"),
        "summary": {
            "chains": len(chains),
            "units": len(rows),
            "units_measured": len(measured),
            "units_with_no_measurable_site": len(siteless),
            "chains_with_no_annotated_site_residue": len(chains_without),
            "anchors_in_the_folds": anchors_total,
            "site_residues_inside_some_folded_unit": len(anywhere),
            "site_residues_inside_no_folded_unit": len(nowhere),
            "site_residues_with_no_counterpart_in_this_genome": len(ungapped),
            "units_whose_largest_patch_is_at_the_site": len(at_the_site),
            "units_whose_anchors_are_processing_sites_only": len(processing),
            "units_with_at_least_one_chemistry_residue": len(chemistry),
            "units_whose_chemistry_subset_differs_from_the_full_set": len(differs),
            "units_with_an_anchor_below_the_confident_threshold": len(low),
            "anchors_below_the_confident_threshold": anchors_low,
            "chemistry_anchors_with_a_confidence": chemistry_anchors,
            "chemistry_anchors_below_the_confident_threshold": chemistry_low,
            "units_with_a_chemistry_anchor_below_the_confident_threshold": chemistry_units_low,
            "anchor_residues_differing_from_the_relative": len(disagreements),
            "units_measured_at_the_annotated_residues": len(at_annotated),
            "units_not_measured_at_the_annotated_residues": len(not_annotated),
            "reading_frames_the_rows_are_numbered_along": len(frames),
            "units_whose_frame_differs_from_the_folds_record": len(disagreeing),
            "unit_pairs_overlapping_in_different_reading_frames": overlapping,
            "chains_not_reached": len(failed_chains),
        },
        "rows": rows,
        "numbering": {
            "on_the_relative": "the homolog's UniProt entry, which is where the annotation "
                               "lives and the only numbering a curator ever wrote.",
            "in_this_genome": "1-based along the open reading frame of this genome the chain "
                              "was located in, through the alignment blocks and never through "
                              "an offset: the held-out spike S1 chain is +10 at its start and "
                              "+31 at its end, and nsp12 sits in a different reading frame "
                              "entirely. Frame-local, so the row's `reading_frame` is part of "
                              "the number and not a note beside it.",
            # Counted off this run's own rows rather than typed. A glossary written to be
            # quoted is worse than none when one of its lines is wrong, and three lines of
            # this one were -- so the figure in it is the figure the summary reports.
            "which_reading_frame": (f"`reading_frame` on every row: the pool index, and "
                                    f"beside it the strand, the codon offset and the "
                                    f"nucleotide start, which identify the frame without "
                                    f"rebuilding this run's pool. Two spans in different "
                                    f"frames can overlap as integers and share no residue -- "
                                    f"over this run {overlapping} pairs of rows do -- so a "
                                    f"join, a sort or a plot on `span_in_this_genome` alone "
                                    f"is wrong by construction. "
                                    f"`agrees_with_the_folds_record` checks this module's own "
                                    f"alignment against the frame the coordinates were cut "
                                    f"from."),
            "in_the_fold": "1..N of the folded coordinates, because ESMFold numbers the "
                           "sequence it was given from one. This is the numbering to type into "
                           "a structure viewer, and it is the unit's span start subtracted "
                           "from the position in this genome, plus one.",
            "verified_on": "the main protease, whose catalytic dyad maps to folded positions "
                           "41 and 145, where the coordinate file carries HIS and CYS. Then "
                           "over everything: all 106 anchors carry the residue the relative's "
                           "entry has at the annotated position, where shifting every anchor "
                           "by one residue drops that to 9 of 106 and by two to 5 or 6. The "
                           "chain of three numberings is checked and not assumed.",
        },
        # What each number answers, in the order a reader should take them. Here rather than on
        # the page because the page should quote this record and not paraphrase it: two of these
        # keys are fractions with different denominators, and a caption that mixed them up would
        # be wrong in a way nothing would catch.
        #
        # Which is exactly what happened to this glossary, so every line of it was then checked
        # against the code that computes the field. Three were wrong. `reach.hydrophobic` and
        # `reach.exposed` said "the residues that are both are site_patch_positions": on r02
        # nsp5 the neighbourhood holds 11 residues that are both, [3, 49, 50, 59, 82, 129, 141,
        # 165, 300, 303, 305] covering 1,118.5 square angstroms, while `site_patch_positions`
        # holds 5, [49, 50, 300, 303, 305] covering 586.7, because connectivity is the
        # difference and the glossary had dropped it. `chemistry_only` named a feature-type rule
        # the code does not apply. `site_patch_area_a2` described a search at the site when the
        # arithmetic is the unit's largest patch clipped to the site. A glossary written to be
        # quoted rather than paraphrased is worse than no glossary at all when one of its lines
        # is wrong, because the wrong line travels verbatim onto the page.
        "how_to_read": {
            "before_anything_else": (f"measured_at_the_annotated_residues. True means the "
                                     f"anchors this row's figures were measured at carry the "
                                     f"residues the annotation names there; False means they "
                                     f"do not, and then the figures describe some other "
                                     f"location and must not be published as this site's. "
                                     f"None means nobody could say -- the relative's entry "
                                     f"has no residue at an annotated position, or the "
                                     f"coordinates name one outside the twenty "
                                     f"`geometry._letter` knows. "
                                     f"anchors_at_the_wrong_residue carries both residues and "
                                     f"both numberings per anchor, so a constant gap between "
                                     f"the numbers reads as a shifted window rather than as "
                                     f"several surprises. True on {len(at_annotated)} of the "
                                     f"{len(measured)} measured units here; the deposited lane "
                                     f"has a measured case where it is 0 of 12."),
            "first": "anchor_confidence.mean_plddt and below_the_confident_threshold. A site "
                     "measured on residues the model placed at pLDDT 30 is a measurement of "
                     "noise, and no figure below it means anything without this one. Where "
                     "chemistry_only.anchor_confidence is present it is the one that governs "
                     "chemistry_only's figures: the row's block is computed over every anchor "
                     "including the proteolytic processing sites, which are chain termini and "
                     "disordered by construction, so it can condemn a measurement for a reason "
                     "the measurement does not depend on.",
            "positions_in_the_fold": "ask 3's list of amino-acid positions, in the numbering "
                                     "the coordinates use. This is what goes into a viewer. "
                                     "Every annotated site residue of the chain that falls "
                                     "inside this unit, processing sites included -- for the "
                                     "main protease that is [1, 41, 145, 306], of which only "
                                     "41 and 145 are the catalytic dyad and 1 and 306 are the "
                                     "cleavage boundaries of the chain itself. "
                                     "chemistry_only.anchors_in_the_fold is the shorter list.",
            "reach.residues": (f"ask 3's count -- how many residues a macrocycle sitting on "
                               f"those positions could reach, at "
                               f"{geometry.MACROCYCLE_REACH_A:.1f} angstroms alpha carbon to "
                               f"alpha carbon, the anchors themselves included. A floor, never "
                               f"a ceiling."),
            "reach.hydrophobic_and_reach.exposed": "two counts over the same neighbourhood, "
                                                   "not a conjunction, and neither is "
                                                   "site_patch_positions. The residues that "
                                                   "are both hydrophobic and exposed are the "
                                                   "population a patch is drawn FROM; the "
                                                   "patch itself is the largest connected "
                                                   "component of them on the whole unit, so "
                                                   "site_patch_positions is that component "
                                                   "intersected with this neighbourhood and is "
                                                   "routinely much shorter. On r02 nsp5, 11 "
                                                   "residues here are both and 5 are in the "
                                                   "patch; connectivity is the difference.",
            "site_patch_positions": (f"the residues of the unit's largest hydrophobic patch "
                                     f"that lie inside this neighbourhood -- an intersection "
                                     f"and not a search. Deliberately not a fresh patch search "
                                     f"around the anchors: accessibility is a property of a "
                                     f"residue in its surroundings, so re-growing components "
                                     f"inside a {geometry.MACROCYCLE_REACH_A:.1f}-angstrom ball "
                                     f"would report a surface assembled from faces the whole "
                                     f"fold keeps buried. The cost is that a smaller patch "
                                     f"sitting at the site is invisible here, so an empty list "
                                     f"means the unit's largest patch is elsewhere, never that "
                                     f"the site has no hydrophobic surface: nsp13's helicase "
                                     f"C-terminal domain and nsp2's C-terminal domain both "
                                     f"report nothing while each neighbourhood holds 3 residues "
                                     f"that are both hydrophobic and exposed, 236.4 and 244.7 "
                                     f"square angstroms of them."),
            "site_patch_area_a2": "ask 3's step 4 -- that intersection's area, so the patch "
                                  "stage 4 scored the unit on, measured only where it reaches "
                                  "this location. Comparable with another prediction and with "
                                  "nothing else.",
            "site_patch_fraction_of_the_site": "of everything a peptide at this site can "
                                               "touch, how much is patch. Denominator: the "
                                               "neighbourhood's own surface.",
            "site_share_of_the_largest_patch": "of the patch stage 4 scored this unit on, how "
                                               "much is at the chemistry. Denominator: the "
                                               "unit's largest patch. This is the number that "
                                               "says whether the surface score was earned "
                                               "where a binder would go.",
            "chemistry_only": "the reach and the site patch over the residues that do "
                              "chemistry, with its own anchor_confidence; no unit_patch, which "
                              "is a property of the whole unit and does not change. The subset "
                              "is NOT a feature-type rule: a residue is dropped when its own "
                              "description names a cleavage, so a plain Site can be kept -- "
                              "Protein 3's `Involved in polymerization` is, and it is that "
                              "chain's only annotated residue. same_as_the_full_set true means "
                              "the subset is the whole set, and then reach, site_patch and "
                              "anchor_confidence are null on purpose: the row's own figures "
                              "are the answer and a copy could drift from them. Where it "
                              "differs from the row, this is the one to read.",
        },
        "limitations": [
            "Every structure measured here is a prediction, and every row says so in its own "
            "`on_predicted_coordinates` and `is_deposited_measurement` -- read off "
            "`geometry.markers_from_coordinates` where a measurement was made rather than "
            "restated, so no row of this record can be pooled with a deposited one by a "
            "consumer that checks. ESMFold returns coordinates for any sequence it accepts, so "
            "a patch area at a site is an arithmetic on a model and not an observation of a "
            "protein. `anchor_confidence` is the only guide to whether the model committed to "
            "the residues the site was measured on, and it is the model's own opinion rather "
            "than an independent check.",
            "The reach is alpha carbon to alpha carbon. `geometry._residues` keeps one atom "
            "per residue, so no side-chain coordinate exists to measure from: an extended "
            "lysine's terminal nitrogen sits about 6.4 angstroms out from its own alpha "
            "carbon, and two long side chains can therefore touch across a gap this counts as "
            "out of range. The neighbourhood is a floor on what a macrocycle at the site can "
            "contact and never a ceiling.",
            "`reach.hydrophobic` and `reach.exposed` are two counts over the same "
            "neighbourhood and not a conjunction. The residues that are both -- which is the "
            "population a patch is drawn from -- are reported as `site_patch_positions`, "
            "restricted to the largest connected component, because that is the set the "
            "stage-4 arithmetic actually selects. The restriction is most of the difference "
            "rather than a detail: on the main protease 11 residues of the neighbourhood are "
            "both hydrophobic and exposed, covering 1,118.5 square angstroms, and 5 of them "
            "covering 586.7 are in the largest patch -- the other 6 belong to other components "
            "of the 18 that unit carries. So a smaller patch sitting at the site is invisible "
            "to `site_patch_area_a2`, and a zero there says the unit's largest patch is "
            "elsewhere and never that the site has no hydrophobic surface.",
            "The reach itself is 13.8 angstroms, derived in `geometry` from the 600 "
            "square-angstrom footprint the scoring already assumes, and it inherits that "
            "footprint's roughness: 'roughly 400 to 800'. It is good to a couple of angstroms "
            "and not to a decimal place, so a residue count near the boundary should be read "
            "as approximate.",
            "`PROCESSING_TERMS` is an authored term list, validated against no labelled "
            "corpus. It partitions a measurement into two that are both reported and it never "
            "drops a residue; the term that fired is recorded on every anchor so its effect is "
            "visible. Measured over this run, 28 of 29 plain `Site` residues name a cleavage.",
            "A residue absent from this record is not a residue that does nothing. The "
            "annotation is bounded by what has been curated, and a novel pathogen's nearest "
            "relative is the least studied case by construction -- which is the case this "
            "pipeline exists for. Three chains here carry no annotated site residue at all.",
            "A patch area measured on a prediction is comparable with another prediction and "
            "with nothing else. Checked on the one chain here that has a solved structure: "
            "anchored on the main protease's catalytic dyad, 6LU7 chain A gives 93 residues "
            "within reach against this prediction's 89, with 41 hydrophobic in both -- but a "
            "largest patch of 415.2 square angstroms over five residues against 2,012.6 over "
            "nineteen. The reach counts and the positions transfer; the areas do not.",
            "An area off a low-confidence fold is inflated, and by how much is measurable "
            "here. The 18 units whose anchors average below pLDDT 70 have a mean exposed "
            "fraction of 0.793 and a mean largest patch of 3,608.5 square angstroms; the 5 at "
            "or above it have 0.611 and 1,592.4. A loosely predicted domain presents "
            "hydrophobic surface that a tightly packed one buries. What this does NOT touch is "
            "`site_share_of_the_largest_patch`, which is a ratio of two areas measured on the "
            "same coordinates: its Spearman correlation with the anchors' mean confidence over "
            "the 23 measured units is +0.21, which is no relationship worth reading. So a high "
            "share is not an artefact of a bad fold, and a large absolute area may well be.",
            f"`span_in_this_genome` and `positions_in_this_genome` are numbered along one open "
            f"reading frame of this genome and not along the genome, so they are meaningless "
            f"without the row's `reading_frame`. Two rows in different frames can carry spans "
            f"that overlap as integers and share no residue: {overlapping} pairs of the rows "
            f"here do, spike S1's [24, 312] against the nucleoprotein's [41, 186] among them. "
            f"Nothing may join, sort or plot on the span alone, and `orf_index` is a position "
            f"in this run's {len(records)}-ORF pool rather than a frame identifier -- the "
            f"strand, the codon offset and the nucleotide start beside it are the frame.",
            "An anchor whose residue differs from the relative's is a substitution the "
            "alignment found, not a mapping error. The site is then conserved in position and "
            "not in identity, which is a weaker claim than the row's figures alone suggest.",
            "A domain folded alone has lost whatever its neighbours contributed to its shape, "
            "and the neighbourhood is drawn only from residues inside the unit that was "
            "folded. A residue of the adjacent domain eleven angstroms from an active site is "
            "in contact with a macrocycle bound there and is not counted here, because these "
            "coordinates do not contain it.",
        ],
        "provenance": provenance(),
    }
    payload["content_sha256"] = content_digest(payload)
    if record:
        out = Path(path) if path else RECORD
        write_json(out, payload)
        print(f"wrote {_shown(out)}  "
              f"({len(measured)} of {len(rows)} units measured at their functional site, "
              f"{len(at_the_site)} with the unit's largest patch there, "
              f"{len(processing)} on processing sites only, "
              f"{len(rows) - len(measured)} with no measurable site)")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--folds", default=None,
                    help="the domain_folds record to measure (default runs/domain_folds.json)")
    ap.add_argument("--path", default=None, help="where to write (default runs/site_patches.json)")
    args = ap.parse_args(argv)
    run(folds=args.folds, path=args.path)


if __name__ == "__main__":
    main()
