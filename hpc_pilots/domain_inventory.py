"""What the annotation offers for the chains the folding ceiling refuses whole.

Stage 3 sends an amino-acid sequence to a folding endpoint that refuses anything over 400
residues, and until now a chain over that length was simply abandoned: no coordinates, and
therefore no predicted surface for stage 4 either. On the held-out SARS-CoV-2 run that is
nine of twenty-one chains, and the landing page said of them that "nothing can be predicted
for these" and that "no coordinates exist for them at all, none solved and none predicted".

Both sentences were true of the pipeline and false of the biology, and this record is the
measurement that separates the two. Measured 2026-11-12 against UniProt, over the twenty-one
ranked chains of the held-out case:

    all nine chains over the ceiling carry at least one annotated functional region
    at or under it.

Not marginally, either. The spike's receptor-binding domain -- the single best-characterised
antiviral target in the set, which UniProt names `Receptor-binding domain (RBD)` in so many
words -- is 223 residues inside a 672-residue chain. The polymerase's `RdRp catalytic` domain
is 163 inside 823. nsp3 is hopeless at 1,945 residues and carries `Macro 1` at 170 and
`Peptidase C16` at 265. So the ceiling was never refusing hard targets; it was refusing *long*
ones, and throwing away the short, functionally specific piece of each that a binder would
actually be aimed at.

**Why this is a record and not just a branch in stage 3.** Stage 3 will fold these, and the
chain record will then carry what it folded. But the claim the landing page has to make is
about what the annotation *offers*, which is a larger set than what any one run had the budget
to fold -- nsp3 alone offers twenty-one foldable regions against a per-chain cap of three. A
page that read only the fold results could say "three domains of nsp3 were folded" and could
not say that the ceiling costs nothing in reachable targets, which is the actual finding. The
two records answer different questions and the page needs both.

**The ordering is the part that required a correction, and it is stated here rather than
assumed.** Sorting a chain's regions by length -- the obvious choice, and the first one made
-- does not surface the binding parts. Measured on the same run:

    Spike S1   1. BetaCoV S1-NTD                 290 aa    0 site residues
               2. Receptor-binding domain (RBD)  223 aa    0 site residues
    Spike S2'  1. Extracellular                  398 aa    1 site residue
               2. Heptad repeat 1                 51 aa    0 site residues
    nsp12      1. NiRAN                          255 aa    2 site residues
               3. RdRp catalytic                 163 aa    6 site residues

The receptor-binding domain comes second to an N-terminal domain with no stated binding
function; a membrane-topology span -- the answer to "which side of the membrane is this on" --
comes first for S2'; and for the polymerase the annotation already says which domain matters,
six contained active- and binding-site residues against NiRAN's two, and length ignores it.

So the order is composed from two signals of different standing, and the difference is
recorded on every row rather than blended into a score. `site_residues_inside` is **data**:
how many of the entry's own `Active site`, `Binding site` and `Site` positions fall within the
region. It is decisive for enzymes and silent on the spike, which carries one cleavage site
across the whole protein and names the RBD in a description instead. `relevance_terms_matched`
is therefore **an authored heuristic**, in the sense `literature.AXIS_TERMS` is one: a term
list nobody validated against a labelled corpus, carried with the term that fired so a reader
can see what it did. It orders a list and does nothing else -- it never refuses a region, never
changes a target's rank, and never decides what is folded.

Topological and transmembrane spans sort last whatever their length, because their own entries
in `domains.FUNCTIONAL_REGIONS` say what they are for: deciding whether a binder can reach a
span, and recognising one as out of reach. That is reach information, not a binding part. They
stay in the record and a reader can still pick them; they simply stop being the default.

**What this record does not carry, and why.** The regions are numbered on the *homolog's
entry*, which is where the annotation lives, and not in the pathogen's own numbering. Those
differ -- the held-out spike S1 chain runs 14-654 on the relative and 24-685 in this genome,
+10 at its start and +31 at its end -- and closing that gap needs the local alignment stage 3
computes, not an offset applied here. Applying one would be the exact error the alignment
exists to avoid, so this record states the relative's numbering, says so on every row, and
leaves the mapping to the stage that has the alignment.

    python -m hpc_pilots.domain_inventory

Writes `runs/domain_inventory.json`.
"""

import json
from pathlib import Path

from . import domains
from .common import ROOT, content_digest, provenance, write_json

RECORD = ROOT / "runs" / "domain_inventory.json"
LANDING = ROOT / "runs" / "onset_chain.json"

# The ceiling this record measures against. Imported rather than written, so a record built
# after the endpoint's limit moves is measured against the limit that actually applied. The
# same argument `build_site._ceiling` makes for the page.
from .chain import FOLD_CEILING                                          # noqa: E402

# Feature types that describe where a span sits relative to a membrane rather than what it
# does. They are kept as regions -- reach is a real question for a binder, and
# `domains.FUNCTIONAL_REGIONS` says as much -- and sorted last, because "Extracellular" is
# not a binding part and should not be any chain's default offer.
POSITIONAL_TYPES = ("Topological domain", "Transmembrane")

# An AUTHORED term list, validated against no labelled corpus, which promotes a region whose
# description names a binding or catalytic function. It exists because the data signal below
# it -- contained annotated site residues -- is silent exactly where it is needed most: the
# spike carries one annotated site across 1,273 residues and still names its receptor-binding
# domain in words. Matched case-insensitively on the description, recorded per row as
# `relevance_terms_matched` so what fired is visible, and used for ORDER ONLY. It refuses
# nothing and reranks no target.
#
# Deliberately not included: "domain", "region", "repeat", "helix", "chain", which appear in
# almost every description and would promote everything equally, which is the same as
# promoting nothing.
RELEVANCE_TERMS = (
    "receptor-binding", "receptor binding", "binding", "catalytic", "active",
    "substrate", "nucleotide", "atp", "gtp", "rna", "dna", "nucleic acid",
    "interface", "dimerization", "dimerisation", "oligomerization", "oligomerisation",
    "fusion", "cleavage", "protease", "peptidase", "polymerase", "helicase",
    "methyltransferase", "exonuclease", "endonuclease", "zinc", "metal",
)


def _relevance_terms(description, feature_type):
    """Which of `RELEVANCE_TERMS` the description names, longest match first.

    The feature type is searched too, so a `Zinc finger` with an empty description -- the
    spike's two coiled coils carry none at all -- is still recognised for what it is rather
    than falling to the bottom for having nothing written about it.
    """
    haystack = f"{description or ''} {feature_type or ''}".lower()
    hits = [term for term in RELEVANCE_TERMS if term in haystack]
    # A description naming "receptor-binding" also contains "binding"; keeping both would
    # score one annotation twice for saying one thing. The longest match wins and the
    # substrings of it are dropped.
    return sorted({term for term in hits
                   if not any(term != other and term in other for other in hits)},
                  key=lambda t: (-len(t), t))


def _site_positions(accession, within):
    """Every annotated site residue inside this chain, as a flat set of entry positions."""
    positions = set()
    for site in domains.annotated_sites(accession, within=within):
        positions.update(site.get("positions") or ())
    return positions


def _region_row(region, site_positions, ceiling):
    """One annotated region, with why it sorts where it does and whether it can be folded."""
    start, end = region["start"], region["end"]
    inside = sorted(p for p in site_positions if start <= p <= end)
    terms = _relevance_terms(region.get("description"), region.get("type"))
    positional = region["type"] in POSITIONAL_TYPES
    row = {
        "type": region["type"],
        "description": region.get("description") or "",
        # Named for what it is. These are the relative's coordinates, not this genome's, and
        # the two differ by more than a constant; the module docstring says why the mapping
        # is stage 3's and not this record's.
        "entry_span_on_the_relative": [start, end],
        "length_aa": region["length"],
        "clipped_to_the_chain": bool(region.get("clipped")),
        "site_residues_inside": len(inside),
        "site_positions_inside_on_the_relative": inside[:12],
        "relevance_terms_matched": terms,
        "describes_position_not_function": positional,
        "foldable": False,
        "refused_because": None,
    }
    if region["length"] < domains.MINIMUM_DOMAIN_AA:
        row["refused_because"] = (
            f"{region['length']} residues, under the {domains.MINIMUM_DOMAIN_AA}-residue "
            f"minimum for a fold unit. Its positions are still a site a macrocycle could be "
            f"designed at; it is refused only as something to send to the endpoint.")
    elif region["length"] > ceiling:
        row["refused_because"] = (
            f"{region['length']} residues, over the {ceiling}-residue ceiling the folding "
            f"endpoint applies, so this region is no more foldable than the whole chain.")
    else:
        row["foldable"] = True
    return row


def _order(rows):
    """Functional relevance first, length only as the tie-break.

    The four keys, in order, and each is a correction of what length alone did: a span that
    describes position rather than function sorts last; then the data signal, contained
    annotated site residues; then the authored term list; then length, which between two
    otherwise equivalent spans is still the right tie-break because the longer one carries
    more structural context.
    """
    return sorted(rows, key=lambda r: (r["describes_position_not_function"],
                                       -r["site_residues_inside"],
                                       -len(r["relevance_terms_matched"]),
                                       -r["length_aa"]))


def _chain_row(score, fold, ceiling):
    """One ranked chain: its length against the ceiling, and what the annotation offers."""
    accession = score.get("nearest_homolog")
    span = score.get("chain_range")
    length = fold.get("recovered_length_aa")
    row = {
        "rank": score.get("rank"),
        "target": score.get("target"),
        "nearest_homolog": accession,
        "chain_range_on_the_relative": list(span) if span else None,
        "span_in_this_genome": fold.get("span_in_this_genome"),
        "length_aa": length,
        "sequence_recovered": bool(fold.get("sequence_recovered")),
        "whole_chain_inside_the_ceiling": bool(length and length <= ceiling),
        "regions_offered": 0, "regions_foldable": 0,
        "functional_site_residues": 0,
        "regions": [], "best_region": None,
        "not_inventoried_because": None,
    }
    if not accession:
        # A candidate identified from its own sequence alone has no curated entry to read, so
        # there is no annotation to offer. Recorded as an absence with its reason, never as a
        # chain with zero regions -- those are different statements and only one is about the
        # annotation.
        row["not_inventoried_because"] = (
            "no reviewed homolog accession, so no curated annotation exists to read. A "
            "sequence-only candidate reaches the folding stage and not this inventory.")
        return row
    within = tuple(span) if span else None
    try:
        regions = domains.annotated_regions(accession, within=within)
        sites = _site_positions(accession, within)
    except Exception as failure:                                        # noqa: BLE001
        row["not_inventoried_because"] = (
            f"the relative's entry could not be read: {type(failure).__name__}: "
            f"{str(failure)[:160]}")
        return row
    rows = _order([_region_row(r, sites, ceiling) for r in regions])
    foldable = [r for r in rows if r["foldable"]]
    row.update({
        "regions_offered": len(rows),
        "regions_foldable": len(foldable),
        "functional_site_residues": len(sites),
        "regions": rows,
        # The one the page offers first. It is the most functionally specific FOLDABLE region,
        # which is not the same as the first row -- a 5-residue motif can out-rank everything
        # on relevance and still be refused as a fold unit.
        "best_region": foldable[0] if foldable else None,
    })
    return row


def run(case=None, path=None, record=True, landing=None):
    """The inventory over one case of the landing record, written as a record."""
    payload_in = json.loads(Path(landing or LANDING).read_text())
    if case is None:
        # The held-out case, for the reason `build_site.build_index` and
        # `predicted_inventory` both give: the landing page is the run with SARS-CoV-2
        # deleted from its own reference, and inventorying the un-held-out case would put the
        # held-out organism's own curated entries on a page whose premise is that they were
        # deleted before the run.
        case = next((c for c in payload_in["cases"] if c.get("reference_excluded_organism")),
                    payload_in["cases"][0])
    ceiling = FOLD_CEILING
    folds = {f.get("target"): f for f in (case.get("foldable") or ())}
    chains = [_chain_row(s, folds.get(s.get("target")) or {}, ceiling)
              for s in (case.get("scores") or ())]

    inventoried = [c for c in chains if not c["not_inventoried_because"]]
    over = [c for c in inventoried if not c["whole_chain_inside_the_ceiling"]]
    rescued = [c for c in over if c["regions_foldable"]]
    stranded = [c for c in over if not c["regions_foldable"]]
    siteless = [c for c in inventoried if not c["functional_site_residues"]]

    payload = {
        "experiment": "domain_inventory",
        "status": "complete",
        "question": "For the chains the folding ceiling refuses whole, does the curated "
                    "annotation offer a functionally important region short enough to fold?",
        "answer": (f"Yes, for {len(rescued)} of the {len(over)} chains over the "
                   f"{ceiling}-residue ceiling. "
                   + (f"{len(stranded)} of them have no foldable annotated region: "
                      + "; ".join(c["target"] for c in stranded) + "."
                      if stranded else
                      "Every chain the ceiling refuses whole has one.")),
        "genome_accession": case.get("genome_accession"),
        "case": case.get("case"),
        "reference_excluded_organism": case.get("reference_excluded_organism"),
        "residue_ceiling": ceiling,
        "minimum_fold_unit_aa": domains.MINIMUM_DOMAIN_AA,
        "summary": {
            "chains": len(chains),
            "chains_inventoried": len(inventoried),
            "chains_without_a_curated_entry": len(chains) - len(inventoried),
            "chains_over_the_ceiling": len(over),
            "chains_over_the_ceiling_with_a_foldable_region": len(rescued),
            "chains_over_the_ceiling_with_none": len(stranded),
            "chains_with_no_annotated_site_residues": len(siteless),
            "regions_offered": sum(c["regions_offered"] for c in inventoried),
            "regions_foldable": sum(c["regions_foldable"] for c in inventoried),
        },
        "chains": chains,
        "ordering": {
            "keys": ["describes_position_not_function ascending",
                     "site_residues_inside descending",
                     "relevance_terms_matched descending",
                     "length_aa descending"],
            "site_residues_inside": "data: counted from the entry's own Active site, "
                                    "Binding site and Site annotations.",
            "relevance_terms_matched": "an authored heuristic, validated against no labelled "
                                       "corpus, in the sense literature.AXIS_TERMS is one. "
                                       "It orders a list and nothing else.",
            "verified": False,
        },
        "limitations": [
            "Region coordinates are on the RELATIVE's entry, not in this genome's numbering. "
            "The two differ by more than a constant -- the held-out spike S1 chain runs "
            "14-654 on the relative and 24-685 here, +10 at its start and +31 at its end -- "
            "so the mapping needs stage 3's local alignment and is deliberately not "
            "approximated here.",
            "This record says a region is SHORT ENOUGH to fold. It does not say the fold "
            "succeeded, that the coordinates are good, or that the domain holds its shape "
            "cut out of its chain. A predicted structure is a prediction whatever its "
            "length.",
            "What the annotation offers is bounded by what has been curated. A region absent "
            "here may exist and be unannotated, and a novel pathogen's nearest relative is "
            "the least studied case by construction -- which is the case this pipeline "
            "exists for.",
            "`relevance_terms_matched` is an authored term list. It is recorded per row so "
            "its effect is visible, and it is used for display order only: no region is "
            "refused by it and no target's rank is touched.",
            "A region's foldability is judged against the ceiling alone. The per-chain cap "
            "stage 3 applies means a foldable region here may still go unfolded in any "
            "given run, which is why the chain record and this one are both needed.",
        ],
        "provenance": provenance(),
    }
    payload["content_sha256"] = content_digest(payload)
    if record:
        out = Path(path) if path else RECORD
        write_json(out, payload)
        print(f"wrote {out.relative_to(ROOT)}  "
              f"({len(over)} chains over the {ceiling}-residue ceiling, "
              f"{len(rescued)} with a foldable annotated region, "
              f"{payload['summary']['regions_foldable']} foldable regions in all)")
    return payload


if __name__ == "__main__":
    run()
