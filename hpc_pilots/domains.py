"""Functionally important spans of a target, read off a homolog and mapped into its numbering.

**The defect this replaces.** Stage 3 recovered a mature chain's own residues and then asked a
single question of them: are there four hundred or fewer? A chain above that ceiling was sent
nowhere. It got no coordinates, so the surface stage could not rescue it either, and its row
said so -- `the_folding_ceiling_closed_the_only_route`. Measured against UniProt on
2026-11-12 over the twenty-one ranked chains of `runs/onset_chain.json` case 0, nine chains
fall above it: nsp12 at 823 residues, nsp3 at 1,945, nsp13 at 601, the three spike chains at
672, 588 and 458, the nucleoprotein at 419, nsp2 at 638 and nsp4 at 500. That is the
polymerase, the papain-like protease, the helicase and the whole receptor-binding machinery --
nine of twenty-one, and not the easy nine.

The ceiling was never refusing hard targets. It was refusing *long* ones, and the part of a
long one a binder is designed against is short. All nine carry at least one annotated
functional region at or under 400 residues, and the annotation was already sitting on a
payload the pipeline had fetched and read for something else:

* Spike S1 -> `Region 319-541 "Receptor-binding domain (RBD)"`, 223 aa, named the RBD by
  UniProt itself, and `Region 437-508 "Receptor-binding motif; binding to human ACE2"`, 72 aa.
* nsp12 -> `Domain 5004-5166 "RdRp catalytic"`, 163 aa.
* nsp13 -> `Domain 5581-5762 "(+)RNA virus helicase ATP-binding"`, 182 aa.
* nsp3 -> `Domain 1025-1194 "Macro 1"`, 170 aa, and `Domain 1634-1898 "Peptidase C16"`, 265 aa.

So the order of the two steps is reversed. The domains are found first and the ceiling is
applied per unit afterwards, rather than the ceiling being applied to a whole chain and
everything inside it discarded with it.

**The annotation is read, not inferred.** Every span here comes off `intake._entry`, the same
cached UniProt payload `intake.mature_chains` already reads to cut a polyprotein into chains.
That matters twice. It costs no extra request -- the entry is in the process cache by the time
stage 3 runs, because stage 2 fetched it to score the row. And it keeps the claim modest: this
module asserts nothing about where a pocket is. It reports what a curator wrote down, with the
feature type and the description attached, so a reader can see that "Receptor-binding domain
(RBD)" is UniProt's sentence and not this repository's inference.

**Numbering is mapped through the alignment, never by an offset.** The spans are annotated on
the *homolog's* entry, and under a holdout the homolog is a different species. Measured on
`runs/onset_chain.json` case 1, where SARS-CoV-2 is deleted from the reference before the run:
nsp5 moves from the relative's [3241, 3546] to the pathogen's own [3264, 3569], a clean +23 --
and spike S1 moves from [14, 654] to [24, 685], which is +10 at the start and +31 at the end.
A single offset is wrong by 21 residues at the far end of that one chain, because there are
indels between the two species. nsp12 is worse still: [4370, 5301] on the relative against
[1, 823] in this genome, because it was found in a different ORF past the ribosomal
frameshift. So `map_position` walks the alignment blocks `chain.locate_chain` already computed,
and a position that falls in a gap returns None rather than a neighbour.

**What is not a functional unit.** Secondary structure is excluded, and the numbers say why:
P0C6X7 carries 191 Beta strand, 171 Helix and 44 Turn features against 36 Domain and 21
Region. A six-residue beta strand is not a fold unit, 191 of them are not 191 binding
opportunities, and admitting them would bury the four domains that matter under two hundred
spans that mean "there is a strand here". `FUNCTIONAL_REGIONS` is therefore a whitelist rather
than a blocklist: a feature type absent from it is excluded by construction, which is why
`Natural variant` (69 on P0DTD1), `Mutagenesis`, `Glycosylation`, `Disulfide bond`,
`Compositional bias`, `Cross-link`, `Lipidation` and `Signal` never appear without being named
anywhere. `Chain` is excluded for a different reason, recorded at the constant.

**One name per region, from one function.** `label_for` is the only fallback in the repository
for a region whose curator description is empty, and it is public for that reason. Three
fallbacks existed and two disagreed: this module appended the span and said
`Coiled coil 918-962` while `domain_inventory` and `domain_folds` dropped back to the bare
feature type and said `Coiled coil`, so one region of Spike protein S2 had two names and any
join keyed on the label lost it -- silently, because every other region on that chain has a
description and joined fine. The span is in the fallback and not merely the type because S2's
range carries two undescribed `Coiled coil` features, 918-962 and 1144-1172, and a name that
cannot tell them apart is not a name. Every region `annotated_regions` returns carries its
`label`, so a consumer need not derive one at all.
"""


# Feature types that mark a functionally important span, each with the clause saying what a
# binder gets from it. A whitelist: anything not keyed here is excluded, and the module
# docstring names the exclusions that a reader would otherwise go looking for.
#
# DELIBERATELY EXCLUDED, and the reasons are not interchangeable:
#
# * "Beta strand", "Helix", "Turn" -- secondary structure, not a functional unit. P0C6X7
#   carries 191 beta strands, measured 2026-11-12; a six-residue strand is not a fold unit and
#   folding one would return coordinates that mean nothing.
# * "Natural variant", "Mutagenesis" -- observations about single residues, often thousands of
#   them, and a variant is a fact about a population rather than a span to bind.
# * "Glycosylation", "Disulfide bond", "Lipidation", "Cross-link" -- modifications of residues
#   inside a span, not spans. `geometry` already counts sequons near a patch, which is the
#   question glycosylation answers for a binder.
# * "Compositional bias" -- a statement about amino-acid composition, usually a low-complexity
#   stretch, which is the opposite of a bindable fold.
# * "Signal" -- cleaved before the mature protein exists, so it is not present to be bound.
# * "Chain" -- that is `intake.mature_chains`' job, and re-reading it here would hand back the
#   chain as a domain of itself. The whole chain already arrives as a fold unit of its own in
#   `fold_units`, which is where that one belongs.
FUNCTIONAL_REGIONS = {
    "Domain": "a folding unit the curators delimit, so it is the piece most likely to hold "
              "its shape when it is cut out of the chain",
    "Region": "a curated span with a stated function -- UniProt's own "
              "\"Receptor-binding domain (RBD)\" on the spike is a Region and not a Domain, "
              "so excluding this type would lose the single best-known target in the set",
    "Motif": "a short functional signature; usually too short to fold, kept because its "
             "residues are a site a macrocycle can be designed at",
    "Zinc finger": "a metal-coordinating unit whose geometry is itself the binding feature",
    "DNA binding": "the span that holds nucleic acid, which a binder competes for directly",
    "Coiled coil": "an exposed helical interface -- the shallow, extended shape macrocycles "
                   "reach and small molecules do not",
    "Repeat": "one copy of a repeated unit, so the same surface is presented more than once "
              "and a binder against it has more than one chance to engage",
    "Topological domain": "which side of a membrane a span sits on, which decides whether a "
                          "binder can reach it without crossing one",
    "Transmembrane": "a membrane-embedded span, carried so it can be recognised as out of "
                     "reach rather than folded as though it were soluble",
}

# Point or short-span annotations naming the residues that do the chemistry. These are the
# "list of AA positions" the project lead asked for: not a span to fold, a set of residues to
# aim at. Kept apart from `FUNCTIONAL_REGIONS` because the two answer different questions and
# a two-residue cleavage site must never be offered as something to fold.
FUNCTIONAL_SITES = ("Active site", "Binding site", "Site")

# Below this many residues a span is a motif rather than a fold unit. STATED, not fitted:
# nothing was optimised against it and no labelled set of foldable spans exists to optimise
# against. The spans either side of it on P0DTC2, measured 2026-11-12, are what it is stated
# from -- the `KxHxx, ER retrieval signal (COPI)` motif is 5 residues and the second coiled
# coil is 28, while `Heptad repeat 1` is 51 and the RBD is 223. Folding a five-residue peptide
# returns coordinates that describe nothing, which is the same failure as the ceiling it works
# against, pointed the other way: that one refused spans too long to model, this one refuses
# spans too short to mean anything. A span under it is NOT discarded -- it keeps its positions
# and stays available as a site, because a five-residue motif is exactly the kind of place a
# macrocycle is designed at. It is refused only as a thing to send to the folding endpoint.
MINIMUM_DOMAIN_AA = 40

# How much of the whole chain a domain has to cover before it is reported as coincident with
# it. Not a new number: `intake.mature_chains` already states 0.95 for the mirror-image
# judgement, where a Chain feature spanning that much of its entry is taken to be the whole
# protein rather than a cleavage product of it. The case this exists for is nsp5, whose
# `Domain 3264-3569 "Peptidase C30"` is the mature chain to the residue -- a target that is
# already one domain must reproduce the whole-chain answer through the domain route, and the
# caller needs to see that it would be folding the same residues twice.
COINCIDENT_FRACTION = 0.95


def annotated_regions(accession, within=None):
    """Functionally important spans of one UniProt entry, clipped to one mature chain.

    `within` is an inclusive `(start, end)` on the ENTRY's numbering -- a mature chain's
    `chain_range`, exactly as `intake.mature_chains` reports it and the ranked row carries it.
    A region straddling the boundary is clipped to it and marked rather than dropped: the
    extracellular topological domain of the spike runs 14-1213 and the S1 chain ends at 685,
    and the 672 residues they share are a real span even though neither endpoint is.

    Returns, sorted by descending length and then by ascending start so the order is
    reproducible across runs:

        [{"type", "description", "start", "end", "length", "clipped", "entry_span", "label"}]

    `start`, `end` and `length` describe the clipped span; `entry_span` is what the curator
    annotated, before clipping, so the provenance survives. Both are on the entry's numbering
    -- nothing here is in the pathogen's numbering yet, which is `map_span`'s job.

    `label` is `label_for(region)`, carried on the region rather than left to each consumer to
    derive. It is here because deriving it separately is what went wrong: three callers wrote
    three fallbacks for a region with an empty description and two of them disagreed, giving
    `Coiled coil 918-962` here and a bare `Coiled coil` in the fold store for one feature of
    Spike protein S2. A consumer that reads this key cannot reintroduce that, and one that
    needs the function instead should call `label_for` rather than write a fourth fallback.

    Reads the same `intake._entry(accession)` payload `intake.mature_chains` already reads, so
    the annotation is read rather than inferred and costs no extra request.
    """
    regions = []
    for feature in _features(accession):
        if feature.get("type") not in FUNCTIONAL_REGIONS:
            continue
        bounds = _bounds(feature)
        if bounds is None:
            continue
        start, end = bounds
        clipped = False
        if within:
            low, high = int(within[0]), int(within[1])
            if end < low or start > high:
                continue
            clipped = start < low or end > high
            start, end = max(start, low), min(end, high)
        region = {"type": feature["type"], "description": _description(feature),
                  "start": start, "end": end, "length": end - start + 1,
                  "clipped": clipped, "entry_span": [bounds[0], bounds[1]]}
        region["label"] = label_for(region)
        regions.append(region)
    regions.sort(key=lambda region: (-region["length"], region["start"], region["type"]))
    return regions


def annotated_sites(accession, within=None):
    """Point annotations -- the residues that do the chemistry.

    A domain says where to look; these say what to touch. The polymerase's `Domain 5004-5166
    "RdRp catalytic"` is 163 residues, and the fourteen Binding site and Active site residues
    inside it are the ones a designed macrocycle has to sit against. Measured 2026-11-12,
    P0C6X7 carries 73 Binding site, 22 Active site and 14 Site features, so this is a dense
    annotation and not an occasional one.

    Returns `[{"type", "description", "ligand", "positions"}]` sorted by first position, where
    `positions` is every residue the feature covers on the ENTRY's numbering -- a list rather
    than a span because most of these features are one residue wide and a reader wanting a
    span can take the ends. `ligand` is the ligand name for a `Binding site` that has one and
    None otherwise; UniProt puts it at `feature["ligand"]["name"]`, verified live against
    P0C6X7 on 2026-11-12, where all 73 Binding site features carry it and the names are
    `Zn(2+)`, `ATP`, `Mg(2+)`, `Mn(2+)`, `S-adenosyl-L-methionine` and `ADP-D-ribose`. No
    other feature type on that entry carries the key at all, so reading it unconditionally is
    safe and reading it only for `Binding site` would be a guess about the schema.

    `within` clips to a mature chain's range the same way `annotated_regions` does. A site is
    a point, so clipping it either keeps it or drops it; there is no partial site to mark.
    """
    sites = []
    for feature in _features(accession):
        if feature.get("type") not in FUNCTIONAL_SITES:
            continue
        bounds = _bounds(feature)
        if bounds is None:
            continue
        start, end = bounds
        if within:
            low, high = int(within[0]), int(within[1])
            if end < low or start > high:
                continue
            start, end = max(start, low), min(end, high)
        sites.append({"type": feature["type"], "description": _description(feature),
                      "ligand": (feature.get("ligand") or {}).get("name") or None,
                      "positions": list(range(start, end + 1))})
    sites.sort(key=lambda site: (site["positions"][0], site["type"], site["description"]))
    return sites


def map_position(located, position):
    """One position on the homolog's mature chain -> the pathogen's own numbering.

    `located` is what `chain.locate_chain` returns, which carries `blocks`: the local
    alignment's aligned blocks as plain nested lists of ints,
    `[[[t0, t1], ...], [[q0, q1], ...]]`, where the first list indexes the pathogen's ORF
    translation and the second the homolog's chain, both 0-based and half-open. `position` is
    a 0-based index into the homolog's chain sequence. Returns a 1-based position in the
    pathogen's ORF, or None.

    **Why None, and why it must not be filled in.** A position between two blocks is a
    position with no counterpart: either the pathogen has an insertion there, or the homolog
    has a residue the pathogen deleted. Substituting the nearest mapped neighbour would return
    a number that looks like an answer and names a residue that is not the annotated one, and
    the measurement behind this whole module is that such errors do not stay small. Under the
    holdout, spike S1 maps from the relative's [14, 654] to the pathogen's [24, 685]: +10 at
    the start and +31 at the end. The 21-residue disagreement between those two offsets is
    precisely the indel content a neighbour substitution would paper over. A site reported at
    the wrong residue is worse than a site reported as unmappable, because only the second one
    can be seen to be wrong.
    """
    blocks = (located or {}).get("blocks")
    if not blocks or len(blocks) < 2:
        return None
    position = int(position)
    for target, query in zip(blocks[0], blocks[1]):
        query_start, query_end = int(query[0]), int(query[1])
        if query_start <= position < query_end:
            return int(target[0]) + (position - query_start) + 1
    return None


def map_span(located, chain_start, start, end):
    """A span on the homolog's ENTRY -> the pathogen's own numbering.

    `chain_start` is the mature chain's start on the entry, so entry position `p` is index
    `p - chain_start` in the chain sequence that was aligned -- `chain.locate_chain` is handed
    `sequence[span[0] - 1:span[1]]`, whose index 0 is entry position `span[0]`.

    Both ends are mapped through `map_position`. Where an end falls in a gap the scan moves
    inward until a position maps, so the span returned is the largest window inside the
    annotation that this genome actually has residues for, and `ends_adjusted` says that
    happened while `unmapped_ends` says for how many of the two. Walking inward rather than
    outward is the conservative direction: it can only ever return a span the annotation
    covers, where walking outward would claim residues the curator did not annotate.

    Returns, or None when no position in the span maps at all -- a domain the alignment never
    reached, which happens when the chain was located in a different ORF from the one the
    domain sits in:

        {"start", "end", "length",            # this genome's numbering
         "entry_span", "entry_length",        # the homolog's, for the window actually mapped
         "ends_adjusted", "unmapped_ends"}

    `length` is the width of the mapped window and can exceed the annotated width, because an
    insertion in the pathogen puts residues inside the span that the homolog does not have.
    That is the honest number: it is the count of residues a fold of this span would contain.

    **`entry_span` is the window this call mapped, not the window it was asked about, and it
    was missing.** The two differ whenever an end falls outside the alignment, and a consumer
    publishing the annotated span beside the mapped one then prints two numbers that describe
    different sets of residues. Measured on nsp12's `Domain "NiRAN"` on 2026-11-12: asked for
    the relative's [4376, 4630], 255 annotated residues, it returns [1, 152] in this genome,
    152 residues -- the chain was located in a reading frame the first 103 of them fall
    outside. The caller had no way to say which 152 of the 255 it got, so the unit it built
    carried `entry_span` [4376, 4630] against `span` [1, 152] and a reader comparing the two
    widths saw a 103-residue discrepancy with nothing to attribute it to. `entry_span` here is
    [4479, 4630] and `entry_length` is 152, which agrees with `length` and closes the question.

    `entry_length` can still differ from `length` in the other direction, through an indel
    inside the window rather than at its ends, and that difference is the real one: it is the
    count of residues this genome gained or lost within the annotated region.
    """
    start, end = int(start), int(end)
    if end < start:
        return None
    first = first_at = None
    for position in range(start, end + 1):
        first = map_position(located, position - chain_start)
        if first is not None:
            first_at = position
            break
    last = last_at = None
    for position in range(end, start - 1, -1):
        last = map_position(located, position - chain_start)
        if last is not None:
            last_at = position
            break
    if first is None or last is None:
        return None
    unmapped = int(first_at != start) + int(last_at != end)
    return {"start": first, "end": last, "length": last - first + 1,
            "entry_span": [first_at, last_at], "entry_length": last_at - first_at + 1,
            "ends_adjusted": bool(unmapped), "unmapped_ends": unmapped}


def label_for(region):
    """The one name for one annotated region, and the only fallback any caller may use.

    Public, and public is the point. The fallback existed three times over and the three did
    not agree: this module appended the span and gave `Coiled coil 918-962`, while
    `domain_inventory` and `domain_folds` fell back to the bare feature type and gave
    `Coiled coil`. Verified on Spike protein S2 (Q3LZX1, chain range [655, 1242]) on
    2026-11-12, where the stored folds under `runs/domain_folds/` carry the bare form in their
    slugs -- `r01_spike_protein_s2__coiled_coil` and `r03_...` -- and this module's own fold
    units carry the span form for the same feature. Two names for one region is one name too
    many: a page joining a fold record to a unit on the label finds nothing for that region,
    and the failure is silent because every other region on the chain has a description and
    joins fine.

    **Why the span is in the fallback rather than the bare type.** A label has two jobs: it
    names the region in the dropdown, and it is what a per-region join is keyed on. The bare
    type fails both the moment a chain carries two features of one type with no description,
    which is not hypothetical -- S2's range holds `Coiled coil 918-962` and
    `Coiled coil 1144-1172`, measured 2026-11-12, so the bare form puts two identically named
    entries in the dropdown and gives a join two rows it cannot tell apart. Only one of the two
    is over `MINIMUM_DOMAIN_AA` and reaches the fold store today, which is why the collision
    has not yet produced a wrong structure under a right name; it is a property of the
    annotation and not of this genome, and anything that cuts one chain out of another will
    produce more of them. `Coiled coil 918-962` is a worse name than
    `Receptor-binding domain (RBD)` and it is still a name that means one region, which is the
    requirement.

    **Shapes accepted.** One function is only one name if every caller can actually call it, so
    this reads the three dict shapes the annotation travels in rather than requiring a caller
    to reshape its row: `annotated_regions`' own region, `domain_inventory`'s row (which keys
    the span `entry_span_on_the_relative` because the numbering is the relative's), and a
    `domain_folds` unit (which keys the type `region_type`). The span is read from the entry
    side in every case, because a label keyed on this genome's numbering would change when the
    alignment changed and stop matching a record written before it.
    """
    description = (region.get("description") or "").strip()
    if description:
        return description
    kind = region.get("type") or region.get("region_type") or "Region"
    span = region.get("entry_span") or region.get("entry_span_on_the_relative")
    if not span:
        # The clipped span, which is what a region carries when nothing recorded the
        # unclipped one. Named rather than silently skipped: a label built from a clipped
        # span still identifies one region, and a label with no span at all would be the bare
        # type this function exists to replace.
        span = [region.get("start"), region.get("end")]
    span = tuple(span)
    if len(span) != 2 or None in span:
        # No span on either side, which leaves only the bare type. Recorded here as the one
        # case where the bare type is the honest answer -- there is nothing to disambiguate
        # with -- rather than left as an unremarked fall-through.
        return kind
    return f"{kind} {span[0]}-{span[1]}"


# The private name this function was first written under. Nothing outside this module imported
# it -- grepped across the package on 2026-11-12, the only two call sites were here -- so the
# alias exists for a session holding the old name while the parallel edits land, and the next
# reader to find no caller for it should delete it.
_label = label_for


def fold_units(row, located, accession, ceiling):
    """Every unit of one chain that stage 3 could fold, longest-first within each kind.

    Returns

        [{"kind": "whole_chain" | "domain", "label", "type", "description",
          "span": [start, end],              # the pathogen's own numbering
          "entry_span": [start, end],        # the homolog's numbering, as annotated
          "entry_span_mapped": [start, end], # the homolog's numbering, as actually mapped
          "entry_length_mapped",
          "length", "inside_the_ceiling", "refused_because",
          "clipped", "coincides_with_the_whole_chain", "mapping",
          "mapping_reproduces_the_span"}]

    `clipped`, `coincides_with_the_whole_chain` and `mapping` are additions to the shape the
    API contract states, each because a number on the page would otherwise be unexplainable:
    `clipped` says a region was cut at the chain boundary and is therefore partial,
    `coincides_with_the_whole_chain` says folding this unit would re-fold residues the
    whole-chain unit already covers, and `mapping` is the `map_span` block, which is where
    `ends_adjusted` and `unmapped_ends` live.

    **`entry_span` and `span` are not always the same residues, and now say so.**
    `entry_span` is what the curator annotated and `span` is what this genome has for it, and
    where an end of the annotation falls outside the alignment the second is a part of the
    first. `entry_span_mapped` is the window `map_span` actually carried over, so the two spans
    a page prints side by side describe one set of residues: on nsp12's `Domain "NiRAN"`,
    measured 2026-11-12, `entry_span` is [4376, 4630] at 255 annotated residues while `span` is
    [1, 152] at 152, and `entry_span_mapped` is [4479, 4630] at 152 -- the 103 residues the
    alignment never reached are now visible as the difference between two stated windows rather
    than as an unexplained gap between two widths. Both are kept: `entry_span` is the
    provenance, and dropping it would lose what was annotated.

    **The whole chain is always the first unit.** The pick order in `chain.fold_stage` takes
    the first foldable unit, so putting the whole chain first means the committed
    main-protease fold stays the fold this repository reproduces byte for byte, and a domain
    can never displace a whole chain. Domains follow, longest first, which is the order a
    caller spending a bounded number of folds should spend them in. The cap on how many of
    them one chain may have belongs to the caller -- `chain.DOMAIN_UNITS_PER_CHAIN` -- because
    it is a budget decision and this function does not know the budget.

    **A chain with no accession or no `located` yields only the whole-chain unit.** Both are
    required and neither is optional. With no accession there is no entry to read spans off.
    With no `located` there are no alignment blocks, so a span annotated on the homolog could
    only be carried over by assuming the two numberings agree -- which is the exact error this
    module exists to remove, measured at +10 residues at one end of spike S1 and +31 at the
    other. It also keeps the fold-stage test that fixes the pick order passing, whose rows
    carry an ORF translation and no homolog at all.

    The cost of that honesty is recorded rather than hidden: the nucleoprotein is one of the
    nine over-ceiling chains and its row carries no `chain_range`, so stage 3 never aligns it
    and it reaches here with `located` None. It gets a whole-chain unit at 419 residues, over
    the ceiling, and no domain route -- even though P0DTC9 carries foldable regions, because
    nothing here knows where they land in this genome.

    Every refusal is named on the unit it refused and nothing is dropped silently, so a unit
    with `inside_the_ceiling` False carries the sentence saying why and a unit whose span
    could not be mapped carries `span` None beside it. A caller must gate on
    `inside_the_ceiling` before reading `span`.
    """
    units = [_whole_chain_unit(row, located, ceiling)]
    chain_range = row.get("chain_range")
    if not (accession and located and chain_range):
        return units

    whole = units[0]
    chain_start = int(chain_range[0])
    for region in annotated_regions(accession, within=chain_range):
        mapping = map_span(located, chain_start, region["start"], region["end"])
        unit = {"kind": "domain",
                "label": label_for(region),
                "type": region["type"],
                "description": region["description"],
                "span": None,
                "entry_span": [region["start"], region["end"]],
                "entry_span_mapped": None,
                "entry_length_mapped": None,
                "length": None,
                "inside_the_ceiling": False,
                "refused_because": None,
                "clipped": region["clipped"],
                "coincides_with_the_whole_chain": False,
                "mapping": mapping,
                # Carried so both kinds of unit have the same keys, and None rather than True
                # because on a domain there is nothing to check: `span` below is read straight
                # out of `mapping`, so a comparison of the two could not fail and a True would
                # claim a check that never ran. The whole-chain unit is where the question has
                # content, and `_whole_chain_unit` makes the argument.
                "mapping_reproduces_the_span": None}
        if mapping is None:
            # Not a failure of the annotation and not a failure of this genome: the chain was
            # located in one ORF and the alignment simply never reached these residues. nsp12
            # is the case -- its chain is in ORF1b past the ribosomal frameshift while the hit
            # that named it is in ORF1a -- so the reason has to name the alignment rather than
            # the domain, or a reader concludes the domain does not exist here.
            unit["refused_because"] = (
                f"no position of the relative's {region['type'].lower()} at "
                f"{region['start']}-{region['end']} falls in a block of the alignment that "
                f"located this chain, so it has no counterpart in this genome's numbering to "
                f"fold.")
            units.append(unit)
            continue
        unit["span"] = [mapping["start"], mapping["end"]]
        unit["length"] = mapping["length"]
        unit["entry_span_mapped"] = list(mapping["entry_span"])
        unit["entry_length_mapped"] = mapping["entry_length"]
        unit["coincides_with_the_whole_chain"] = _coincides(unit["span"], whole["span"])
        if unit["length"] < MINIMUM_DOMAIN_AA:
            unit["refused_because"] = (
                f"{unit['length']} residues, under the {MINIMUM_DOMAIN_AA}-residue minimum "
                f"this stage requires of a fold unit. It stays available as a site, which "
                f"carries its positions without claiming a fold.")
        elif unit["length"] > ceiling:
            unit["refused_because"] = (
                f"{unit['length']} residues, over the {ceiling}-residue ceiling this stage "
                f"sends to the folding endpoint.")
        else:
            unit["inside_the_ceiling"] = True
        units.append(unit)
    return units


# --------------------------------------------------------------------------------- internals

def _features(accession):
    """The feature list off the cached entry, imported late the way `chain` imports `intake`.

    Late because `intake` is the largest module in the package and this one is a leaf that
    `chain` reaches for inside a stage. `intake._entry` memoises per process, so stage 3
    reading an entry stage 2 already scored costs nothing.
    """
    from .intake import _entry

    return _entry(accession).get("features") or []


def _bounds(feature):
    """A feature's inclusive 1-based span, or None when a position is not a number.

    `intake.mature_chains` guards `if None in (start, end)` and the guard is kept here for the
    same reason, now with the measurement behind it. Across 200 reviewed viral entries
    carrying a Domain feature, fetched 2026-11-12, 10,643 of 10,668 feature positions are
    `EXACT`; the remaining 25 are 13 `UNSURE`, 6 `OUTSIDE` and 6 `UNKNOWN`. Only `UNKNOWN`
    carries `value: None` -- C1JCT2's `Chain "Helicase"` ends at an unknown position -- and
    that is the one case refused here, because a span with no second endpoint cannot be
    clipped, mapped or folded. `UNSURE` and `OUTSIDE` carry a real integer, which is the
    curator's placement of a boundary they are hedging rather than an absence of one, so they
    are taken at their stated value; clipping to the mature chain then bounds whatever an
    `OUTSIDE` marker was warning about.
    """
    location = feature.get("location") or {}
    start = (location.get("start") or {}).get("value")
    end = (location.get("end") or {}).get("value")
    if None in (start, end):
        return None
    return int(start), int(end)


def _description(feature):
    """The curator's description, trimmed of the punctuation UniProt leaves dangling.

    `Integrin-binding motif;` arrives with its semicolon on P0DTC2 because the field is a
    fragment of a longer annotation, and it reads as a truncation on a page. The text itself
    is never rewritten: `Receptor-binding motif; binding to human ACE2` keeps its semicolon,
    because that one separates two clauses the curator wrote.
    """
    return (feature.get("description") or "").strip().rstrip(";,").strip()


def _whole_chain_unit(row, located, ceiling):
    """The whole chain or the whole ORF as a fold unit, first in the list and never displaced.

    Its length is `len(located["sequence"])` -- the residues this genome actually has over the
    located span -- which is the same quantity `chain.fold_stage` already writes to
    `recovered_length_aa` and already applies the ceiling to. Deliberately the same quantity
    and not the span's width or the row's declared length, so that moving the ceiling decision
    into this module cannot change which chains the existing records say were folded.

    **The chain's own alignment can leave part of the chain unmapped, and this unit claimed it
    never did.** `mapping` was hard-coded None here, so a whole chain published its full
    annotated `entry_span` beside a shorter `span` with no flag of any kind -- while a region
    annotated over the identical range, arriving through `map_span` in `fold_units`, carried
    `ends_adjusted` and `unmapped_ends` for the same shortfall. Measured on nsp12 on
    2026-11-12: `entry_span` [4370, 5301] is 932 residues of the relative's chain against
    `span` [1, 823], 823 residues in this genome, and the alignment covers 88.3% of the chain.
    The 109-residue difference is real -- the chain was located in a reading frame past the
    ribosomal frameshift, and the alignment never reaches the first 109 residues of the
    relative's chain -- and the unit said nothing about it, while the
    `Topological domain "Cytoplasmic"` clipped to exactly [4370, 5301] reported
    `ends_adjusted` True on the same row. One of those two was lying by omission.

    So the chain's own range is mapped through `map_span` like any other span, and the block is
    carried. `span` and `length` are **not** taken from it: they stay `located`'s, for the
    reason above, and the mapping is a statement about them rather than a replacement for them.
    The two agree by construction whenever `chain_start` really is index 0 of the sequence that
    was aligned -- the assumption `map_span` documents -- so `mapping_reproduces_the_span`
    checks that rather than assuming it, and a False there is the one signal available that the
    range handed in does not index the alignment it was handed with. Verified True on nsp12,
    whose mapped window [4479, 5301] is 823 residues and whose `span` is [1, 823].
    """
    orf = (row.get("orf") or {}).get("protein") or ""
    chain_range = row.get("chain_range")
    mapping = None
    if located and chain_range:
        mapping = map_span(located, int(chain_range[0]),
                           int(chain_range[0]), int(chain_range[1]))
    if located:
        span = [int(located["span"][0]), int(located["span"][1])]
        length = len(located.get("sequence") or "") or (span[1] - span[0] + 1)
    elif orf:
        # No mature-chain boundaries were read, so the ORF translation is the candidate -- the
        # same branch `chain.fold_stage` takes when a row carries no `chain_range`.
        span, length = [1, len(orf)], len(orf)
    else:
        span, length = None, int(row.get("length_aa") or 0)
    unit = {"kind": "whole_chain",
            "label": row.get("name") or "unidentified ORF",
            # "Chain" is the UniProt feature type this unit corresponds to, and the one type
            # `FUNCTIONAL_REGIONS` excludes: the chain arrives here rather than through the
            # region reader, which is the whole reason for excluding it there.
            "type": "Chain" if chain_range else "ORF",
            "description": ("the whole mature chain, as the relative's entry delimits it"
                            if chain_range else
                            "the whole called ORF; no mature-chain boundaries were read"),
            "span": span,
            "entry_span": [int(chain_range[0]), int(chain_range[1])] if chain_range else None,
            "entry_span_mapped": list(mapping["entry_span"]) if mapping else None,
            "entry_length_mapped": mapping["entry_length"] if mapping else None,
            "length": length,
            "inside_the_ceiling": 0 < length <= ceiling,
            "refused_because": None,
            "clipped": False,
            "coincides_with_the_whole_chain": True,
            "mapping": mapping,
            # The cross-check the docstring above argues for, and None where there was no
            # mapping to check: an ORF with no mature-chain range has no entry-side window, so
            # a boolean there would be an answer to a question nobody asked.
            "mapping_reproduces_the_span": (None if not (mapping and span) else
                                            [mapping["start"], mapping["end"]] == span)}
    if not unit["inside_the_ceiling"]:
        # Worded to match the sentence `chain.fold_stage` has written onto an over-ceiling row
        # since the ceiling was first named, so a reader comparing an old record against a new
        # one sees the same reason rather than a reworded one.
        unit["refused_because"] = (
            f"{length} residues, over the {ceiling}-residue ceiling this stage sends to the "
            f"folding endpoint." if length else
            "no sequence was recovered for this candidate, so there is nothing to fold.")
    return unit


def _coincides(span, whole):
    """Whether a domain covers enough of the whole chain to be the same fold twice.

    The nsp5 case: `Domain "Peptidase C30"` is the mature chain to the residue, so a caller
    that folded both would spend two requests on one set of coordinates. Reported rather than
    refused, because the consistency check the project lead's ask implies -- a chain that is
    already one domain must give the same answer through the domain route -- needs the unit to
    be there to check.
    """
    if not span or not whole:
        return False
    overlap = min(span[1], whole[1]) - max(span[0], whole[0]) + 1
    width = whole[1] - whole[0] + 1
    return width > 0 and overlap >= width * COINCIDENT_FRACTION


def main(argv=None):
    """Print one entry's functional regions and sites, for checking the read by hand.

    The probe stage 3 cannot run for you: it says what spans an accession offers and which of
    them this module would send to the folding endpoint, without folding anything.

        python -m hpc_pilots.domains P0DTD1 --within 3264-3569
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Functionally important spans of one UniProt entry.")
    parser.add_argument("accession")
    parser.add_argument("--within", default=None,
                        help="a mature chain's range on the entry, as START-END")
    parser.add_argument("--ceiling", type=int, default=None,
                        help="residue ceiling for a fold unit; defaults to chain.FOLD_CEILING")
    args = parser.parse_args(argv)

    within = None
    if args.within:
        low, _, high = args.within.partition("-")
        within = (int(low), int(high))
    ceiling = args.ceiling
    if ceiling is None:
        # Read rather than typed. The ceiling is one number and a second copy of it here would
        # keep printing 400 after the first one moved.
        from .chain import FOLD_CEILING

        ceiling = FOLD_CEILING

    regions = annotated_regions(args.accession, within)
    sites = annotated_sites(args.accession, within)
    foldable = [r for r in regions if MINIMUM_DOMAIN_AA <= r["length"] <= ceiling]
    print(f"{args.accession}"
          + (f" within {within[0]}-{within[1]}" if within else " (whole entry)")
          + f": {len(regions)} functional region{'' if len(regions) == 1 else 's'}, "
            f"{len(foldable)} of them foldable at or under {ceiling} residues, "
            f"{len(sites)} site{'' if len(sites) == 1 else 's'}")
    for region in regions:
        if region["length"] < MINIMUM_DOMAIN_AA:
            verdict = f"under {MINIMUM_DOMAIN_AA}"
        elif region["length"] > ceiling:
            verdict = f"over {ceiling}"
        else:
            verdict = "foldable"
        print(f"  {region['type']:19s} {region['start']:5d}-{region['end']:<5d} "
              f"{region['length']:5d} aa  {verdict:11s} "
              f"{'clipped  ' if region['clipped'] else ''}{region['label']}")
    for site in sites:
        positions = ", ".join(str(p) for p in site["positions"])
        print(f"  {site['type']:19s} {positions:17s} "
              f"{site['ligand'] or ''}{'  ' if site['ligand'] else ''}"
              f"{site['description']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
