"""Coordinates for the chains the ceiling refuses whole, by folding their domains instead.

`domain_inventory` establishes that every chain over the 400-residue ceiling carries an
annotated functional region at or under it. That is a claim about lengths, and a page can
only show it as a table. This module takes the next step and actually folds them, so the
claim becomes a structure a reader can look at.

Two sets of coordinates come out of it, and they answer different questions:

* **The domain of an over-ceiling chain.** The chain itself cannot be folded -- the endpoint
  refuses anything longer than 400 residues -- so this is the only structure that will ever
  exist for it short of someone solving one. For nsp3 at 1,945 residues it is the difference
  between a row that says "too long" and a 369-residue domain with coordinates.
* **The whole chain, for the chains already inside the ceiling.** The recorded run folded
  these and kept one of them, the main protease, because that was the one the page drew. The
  other ten were folded and their coordinates discarded at the end of the run. Keeping them
  is what lets every row on the page open onto its own structure rather than only the one.

**Why this reconstructs the sequences rather than reading them.** `chain.fold_stage` computes
exactly the sequences this module needs and then deletes them: `recovered_sequence` is popped
off every row before the record is written, because a record carrying twenty whole protein
sequences would multiply in size for no reader's benefit. That is the right call for the
record and it means the sequences have to be derived again here. They are derived the same
way, from the two things the record does keep -- the genome accession and each chain's
annotated range on its homolog -- so nothing is guessed:

1. The genome is fetched and translated in all six frames, and the twelve longest open
   reading frames are kept. Twelve is not a choice made here: `intake` ranks the twelve
   longest, so the twelve longest are the ORFs the recorded run searched.
2. The homolog's own residues over the annotated range are read from its entry. For a chain
   with no annotated range -- a protein that was never part of a polyprotein -- the whole
   entry sequence is the reference, which makes one code path serve both.
3. That reference is aligned against every one of the twelve ORFs and the best-covered hit
   wins. Searching all twelve and not only the parent is load-bearing: a coronavirus
   polyprotein is translated as two overlapping ORFs through a ribosomal frameshift, so
   nsp12 to nsp16 sit in ORF1b while the hit that named them is in ORF1a.
4. The alignment's blocks map each annotated region from the homolog's numbering into this
   genome's, through `domains.map_span`, which walks inward past indels and returns nothing
   rather than a neighbour when a boundary has no counterpart.

**What is deliberately not re-derived.** The pick order, the ranking and the committed
main-protease fold are all untouched. This module writes under `runs/domain_folds/` and never
into `runs/folded_structure/`, so `mpro_nsp5.pdb` -- the fold this repository reproduces byte
for byte -- stays exactly as it is, and the main protease gets a second, separate fold here
like any other chain inside the ceiling.

    python -m hpc_pilots.domain_folds [--limit N] [--budget SECONDS]

Writes `runs/domain_folds.json` and one JSON and PDB per fold under `runs/domain_folds/`.
"""

import argparse
import json
import re
import time
from pathlib import Path

from . import domains
from .chain import FOLD_CEILING, _homolog_chain
from .common import ROOT, content_digest, provenance, write_json

RECORD = ROOT / "runs" / "domain_folds.json"
STORE = ROOT / "runs" / "domain_folds"
LANDING = ROOT / "runs" / "onset_chain.json"
INVENTORY = ROOT / "runs" / "domain_inventory.json"

# How many open reading frames the relative's chain is searched against. Not a number chosen
# here: `intake.run_intake` ranks the twelve longest, so twelve is the pool the recorded run
# had, and searching a different number would locate chains that run never located.
ORF_POOL = 12

# Domains folded per over-ceiling chain, in the inventory's relevance order. A cap is needed
# at all because nsp3 offers eighteen foldable regions and folding every one of them would
# spend a whole run on a single chain; 75 across the shortlist is not a page, it is a
# database.
#
# Three rather than one, and the spike is why. The inventory orders on contained annotated
# site residues first, which is decisive for an enzyme and silent on a glycoprotein: spike S1
# carries one marked site residue in its whole length, so neither signal fires and length
# breaks the tie. That hands the lead to `BetaCoV S1-NTD` at 281 residues, which folds at 26.9
# mean confidence -- the worst of the twenty -- while `BetaCoV S1-CTD`, the domain that is
# structurally the receptor-binding one, sits behind it at 176. Folding one per chain showed
# the reader the weakest structure on the page and hid the one worth looking at. Three reaches
# it on every chain in this run.
DOMAINS_PER_CHAIN = 3

# Seconds the whole run may spend inside the folding endpoint. Measured on the recorded run:
# a 306-residue chain takes about 13 seconds and a 400-residue one about 26, and the endpoint
# returned 504 to five of seven calls on 11 Nov 2026, so the budget has to cover refusals as
# well as work. Twenty folds at the slow end is roughly nine minutes; this allows for the bad
# afternoon as well as the good one.
BUDGET_SECONDS = 1500.0


def _slug(text):
    return re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_") or "unnamed"


def _unit_slug(rank, target, label):
    """A filename for one unit's coordinates, qualified by rank so it cannot collide.

    The rank is not decoration. Slugging the name alone collapsed `Spike protein S2'` and
    `Spike protein S2` onto one string -- the apostrophe is not alphanumeric and is stripped
    -- so ranks 1 and 3 wrote the same `spike_protein_s2__heptad_repeat_1.pdb`, the second
    fold overwrote the first, and the page would have drawn one chain's structure under both
    rows. Both chains genuinely offer a `Heptad repeat 1`, because S2' is a cleavage product
    of S2 and they share it, so the collision is a property of this data and not a freak
    name: anything that cuts one chain out of another will produce more of them.
    """
    return f"r{int(rank):02d}_{_slug(target)}__{_slug(label)}"


def _orf_records(accession):
    """The twelve longest open reading frames, with the frame metadata that identifies them.

    Through `genome_records` and not `fetch_genome`, because the latter returns a bare
    `(header, sequence)` pair for one accession and a list of them for a segmented genome.
    `genome_records` always returns the list, and ORFs are called per segment and pooled:
    translating a concatenation would create reading frames across a junction that exists in
    no molecule, which is the reason `fetch_genome` refuses to join them in the first place.

    `_orf_pool` below used to be the only door and it kept `protein` alone, so an index into
    the pool was the whole of what a downstream record could say about which reading frame a
    position was numbered along. `intake.orfs` returns `strand`, `frame` and `nt_start` beside
    the translation, and those three are the frame itself rather than a position in a list that
    a different pool would order differently. Kept here, discarded there, and `site_patches`
    takes this door because its rows publish positions numbered along one of these frames.
    """
    from .intake import genome_records, orfs

    found = []
    for _header, sequence in genome_records(accession):
        found.extend(orfs(sequence))
    found.sort(key=lambda o: -o["length_aa"])
    return found[:ORF_POOL]


def _orf_pool(accession):
    """The same twelve, as translations alone -- what `_locate` indexes and aligns against."""
    return [o["protein"] for o in _orf_records(accession)]


def _locate(reference, pool, aligner):
    """Where the homolog's chain lies in this genome, with the alignment blocks kept.

    `chain.locate_chain` answers the same question and discards the blocks, which are what
    `domains.map_span` needs to carry an annotated boundary across the species gap. Rather
    than widen that function from here -- it is on the recorded path and this module is not --
    the alignment is run again and the blocks are kept, in the plain nested-int form
    `map_position` documents.
    """
    best = None
    for index, orf in enumerate(pool):
        if not orf:
            continue
        try:
            alignment = aligner.align(orf, reference)[0]
        except Exception:                                             # noqa: BLE001
            continue
        blocks = alignment.aligned
        if len(blocks) < 2 or not len(blocks[0]):
            continue
        start, end = int(blocks[0][0][0]), int(blocks[0][-1][1])
        matched = sum(int(b - a) for a, b in blocks[0])
        identical = 0
        for (ts, te), (qs, qe) in zip(blocks[0], blocks[1]):
            identical += sum(1 for x, y in zip(orf[int(ts):int(te)], reference[int(qs):int(qe)])
                             if x == y)
        located = {
            "orf_index": index,
            "span": [start + 1, end],
            "sequence": orf[start:end],
            "covered": round(matched / len(reference), 4),
            "identity": round(identical / matched, 4) if matched else 0.0,
            "blocks": [[[int(a), int(b)] for a, b in blocks[0]],
                       [[int(a), int(b)] for a, b in blocks[1]]],
            "orf": orf,
        }
        if best is None or located["covered"] > best["covered"]:
            best = located
    return best


def _reference(accession, span, cache):
    """The homolog's residues the chain is annotated over, or its whole sequence.

    A chain with no annotated range is a protein that was never cut out of a polyprotein, so
    the entry sequence is the chain. One path for both, which is what keeps the membrane
    protein and the main protease in the same loop.
    """
    if span:
        return _homolog_chain(accession, span, cache)
    from .intake import _entry

    return _entry(accession)["sequence"]["value"]


def _region_label(region):
    """One name for one region, through `domains.label_for` and never built here.

    This module built the fallback itself and disagreed with the module that owns it: the
    bare feature type against `domains.label_for`'s type-plus-span. Measured on Spike protein
    S2 (Q3LZX1) on 2026-11-12, an undescribed `Coiled coil` was `Coiled coil` in the fold
    store and `Coiled coil 918-962` everywhere else, so a per-region join between the two
    records failed on exactly the regions a curator left undescribed. The inventory row names
    its span differently from an annotated region, so it is translated here rather than
    `label_for` being taught a second shape.
    """
    span = region.get("entry_span_on_the_relative") or []
    return domains.label_for({
        "type": region.get("type"),
        "description": region.get("description"),
        "start": span[0] if len(span) == 2 else None,
        "end": span[1] if len(span) == 2 else None})


def _units(chain, located, inventory_chain, ceiling):
    """What to fold for one chain: the whole chain if it fits, else its leading domains."""
    span = chain.get("chain_range_on_the_relative") or chain.get("chain_range")
    chain_start = span[0] if span else 1
    whole = located["sequence"]
    units = []
    if 0 < len(whole) <= ceiling:
        units.append({"kind": "whole_chain", "label": chain["target"],
                      "region_type": None, "description": None,
                      "entry_span_on_the_relative": list(span) if span else None,
                      "span_in_this_genome": located["span"],
                      "length_aa": len(whole), "sequence": whole,
                      "ends_adjusted": False})
        return units
    # Over the ceiling: the regions the inventory offers, in its order, foldable ones only.
    offered = [r for r in (inventory_chain.get("regions") or ()) if r.get("foldable")]
    for region in offered:
        if len(units) >= DOMAINS_PER_CHAIN:
            break
        entry_span = region.get("entry_span_on_the_relative") or []
        if len(entry_span) != 2:
            continue
        mapped = domains.map_span(located, chain_start, entry_span[0], entry_span[1])
        if not mapped:
            # The alignment never reached this region -- which happens when the chain was
            # located in a different ORF from the one the region sits in. Recorded rather
            # than skipped, because an absence with a reason is the house rule.
            units.append({"kind": "domain", "label": _region_label(region),
                          "region_type": region.get("type"),
                          "description": region.get("description"),
                          "entry_span_on_the_relative": entry_span,
                          "span_in_this_genome": None, "length_aa": None,
                          "sequence": None, "ends_adjusted": False,
                          "refused_because": "no position in this region mapped through the "
                                             "alignment, so this genome's residues for it "
                                             "could not be identified."})
            continue
        sequence = located["orf"][mapped["start"] - 1:mapped["end"]]
        unit = {"kind": "domain",
                "label": _region_label(region),
                "region_type": region.get("type"),
                "description": region.get("description"),
                "entry_span_on_the_relative": entry_span,
                "span_in_this_genome": [mapped["start"], mapped["end"]],
                "length_aa": len(sequence),
                "mapped_length_aa": mapped["length"],
                "ends_adjusted": mapped["ends_adjusted"],
                "unmapped_ends": mapped["unmapped_ends"],
                "site_residues_inside": region.get("site_residues_inside"),
                "sequence": sequence, "refused_because": None}
        if not 0 < len(sequence) <= ceiling:
            unit["refused_because"] = (
                f"the mapped window is {len(sequence)} residues, outside the 1-{ceiling} "
                f"range the endpoint accepts. An insertion in this genome can widen an "
                f"annotated span past the ceiling it fitted inside on the relative.")
            unit["sequence"] = None
        units.append(unit)
    return units


def run(limit=None, budget=BUDGET_SECONDS, record=True, reuse=True):
    """Fold a unit for every ranked chain, and record every refusal with its reason."""
    from .folded_structure import fold
    from .intake import _aligner

    landing = json.loads(LANDING.read_text())
    case = next((c for c in landing["cases"] if c.get("reference_excluded_organism")),
                landing["cases"][0])
    inventory = json.loads(INVENTORY.read_text()) if INVENTORY.exists() else {"chains": []}
    by_target = {c.get("target"): c for c in (inventory.get("chains") or ())}

    accession = case.get("genome_accession")
    pool = _orf_pool(accession)
    aligner, cache = _aligner(), {}
    STORE.mkdir(parents=True, exist_ok=True)

    rows, folded_count, refused_count = [], 0, 0
    started = time.time()
    for score in case.get("scores") or ():
        target = score.get("target")
        homolog = score.get("nearest_homolog")
        span = score.get("chain_range")
        row = {"rank": score.get("rank"), "target": target, "nearest_homolog": homolog,
               "chain_range_on_the_relative": list(span) if span else None,
               "units": [], "not_folded_because": None}
        if not homolog:
            row["not_folded_because"] = (
                "no reviewed homolog accession, so neither the chain's own residues nor any "
                "annotation could be read for it.")
            rows.append(row)
            continue
        try:
            reference = _reference(homolog, span, cache)
        except Exception as failure:                                  # noqa: BLE001
            row["not_folded_because"] = (
                f"the relative's residues could not be read: {type(failure).__name__}: "
                f"{str(failure)[:160]}")
            rows.append(row)
            continue
        if not reference:
            row["not_folded_because"] = "the relative's entry carried no sequence."
            rows.append(row)
            continue
        located = _locate(reference, pool, aligner)
        if not located or located["covered"] < 0.5:
            row["not_folded_because"] = (
                f"the best alignment over {len(pool)} reading frames accounted for only "
                f"{(located or {}).get('covered', 0):.0%} of the relative's "
                f"{len(reference)}-residue chain, under the 50% needed to call the window it "
                f"points at this chain in this genome.")
            rows.append(row)
            continue
        row.update({"found_in_orf": located["orf_index"],
                    "span_in_this_genome": located["span"],
                    "chain_covered_by_alignment": located["covered"],
                    "identity_over_aligned_columns": located["identity"],
                    "length_aa": len(located["sequence"])})
        units = _units({"target": target, "chain_range_on_the_relative": span},
                       located, by_target.get(target) or {}, FOLD_CEILING)
        for unit in units:
            out = {k: v for k, v in unit.items() if k != "sequence"}
            sequence = unit.get("sequence")
            if not sequence:
                out.setdefault("refused_because",
                               "no sequence was recovered for this unit.")
                out["folded"] = False
                refused_count += 1
                row["units"].append(out)
                continue
            slug = _unit_slug(score.get("rank") or 0, target, unit["label"])
            store = STORE / f"{slug}.json"
            pdb = STORE / f"{slug}.pdb"
            if reuse and store.exists() and pdb.exists():
                payload = json.loads(store.read_text())
                out.update({"folded": True, "reused": True, "slug": slug,
                            "mean_plddt": payload.get("mean_plddt"),
                            "confident_fraction": payload.get("confident_fraction"),
                            "pdb_sha256": payload.get("pdb_sha256")})
                folded_count += 1
                row["units"].append(out)
                continue
            if budget is not None and time.time() - started >= budget:
                out.update({"folded": False, "refused_because":
                            f"the {budget:.0f}-second folding budget for this run was spent "
                            f"before the order reached this unit."})
                refused_count += 1
                row["units"].append(out)
                continue
            if limit is not None and folded_count >= limit:
                out.update({"folded": False, "refused_because":
                            f"the cap of {limit} folds on this run was spent on the units "
                            f"ahead of this one."})
                refused_count += 1
                row["units"].append(out)
                continue
            try:
                payload = fold(sequence, target_name=unit["label"] or target, record=False)
            except Exception as failure:                              # noqa: BLE001
                payload = {"absence": f"{type(failure).__name__}: {str(failure)[:160]}"}
            if not payload.get("folded"):
                absence = payload.get("absence")
                out.update({"folded": False, "refused_because":
                            f"the folding endpoint returned no coordinates: "
                            f"{absence.get('reason') if isinstance(absence, dict) else absence}"})
                refused_count += 1
                row["units"].append(out)
                continue
            text = payload.pop("pdb", None)
            payload["unit"] = {k: v for k, v in out.items() if k != "folded"}
            write_json(store, payload)
            if text:
                pdb.write_text(text)
            out.update({"folded": True, "reused": False, "slug": slug,
                        "mean_plddt": payload.get("mean_plddt"),
                        "confident_fraction": payload.get("confident_fraction"),
                        "pdb_sha256": payload.get("pdb_sha256")})
            folded_count += 1
            row["units"].append(out)
        rows.append(row)

    whole = [u for r in rows for u in r["units"] if u.get("kind") == "whole_chain"]
    domain = [u for r in rows for u in r["units"] if u.get("kind") == "domain"]
    payload = {
        "experiment": "domain_folds",
        "status": "complete",
        "question": "Can the chains the folding ceiling refuses whole be given coordinates "
                    "by folding the functionally important region inside them instead?",
        "answer": (f"{len([u for u in domain if u.get('folded')])} of {len(domain)} domain "
                   f"units folded, and {len([u for u in whole if u.get('folded')])} of "
                   f"{len(whole)} whole chains inside the ceiling."),
        "genome_accession": accession,
        "case": case.get("case"),
        "reference_excluded_organism": case.get("reference_excluded_organism"),
        "residue_ceiling": FOLD_CEILING,
        "orf_pool": len(pool),
        "domains_per_chain": DOMAINS_PER_CHAIN,
        "summary": {
            "chains": len(rows),
            "units": len(whole) + len(domain),
            "folded": folded_count,
            "refused": refused_count,
            "whole_chain_units": len(whole),
            "domain_units": len(domain),
            "domain_units_folded": len([u for u in domain if u.get("folded")]),
        },
        "chains": rows,
        "limitations": [
            "Every structure here is a prediction. ESMFold returns coordinates for any "
            "sequence it accepts, and a confident-looking fold of a domain cut out of its "
            "chain is still a model of that domain in isolation.",
            "A domain folded alone has lost whatever its neighbours contributed to its "
            "shape. The per-residue confidence is the only guide to that here, and it is the "
            "model's own opinion rather than an independent check.",
            "The sequences are reconstructed from the genome and the relative's annotated "
            "ranges rather than read from the chain record, which deletes them. The "
            "reconstruction is the same alignment the chain stage runs, but it is a second "
            "computation of the same thing and could differ from the recorded run if the "
            "reference entry has been curated since.",
            "`runs/folded_structure/mpro_nsp5.pdb` is untouched. The main protease is folded "
            "again here, separately, so the committed artifact this repository reproduces "
            "byte for byte is not disturbed by this module.",
        ],
        "provenance": provenance(),
    }
    payload["content_sha256"] = content_digest(payload)
    if record:
        write_json(RECORD, payload)
        print(f"wrote {RECORD.relative_to(ROOT)}  "
              f"({folded_count} folded, {refused_count} refused, "
              f"{len(domain)} domain units, {len(whole)} whole chains)")
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--limit", type=int, default=None, help="cap on folds this run")
    ap.add_argument("--budget", type=float, default=BUDGET_SECONDS,
                    help="seconds this run may spend folding")
    ap.add_argument("--refold", action="store_true",
                    help="fold again even where coordinates are already stored")
    args = ap.parse_args(argv)
    run(limit=args.limit, budget=args.budget, reuse=not args.refold)


if __name__ == "__main__":
    main()
