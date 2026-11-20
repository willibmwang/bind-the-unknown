"""What folding a region alone costs, measured instead of conceded.

**The limitation this answers is one this project states about itself and has never
measured.** `modal_esmfold`'s own docstring: seven of the eight SARS-CoV-2 chains the public
host refuses whole "are rescued by folding their annotated regions instead, which is a real
result and also a weaker one: a region folded alone has lost whatever its neighbours in the
chain contributed to its shape, and the chain record says so in its own limitations." Across
the panel the domain route is what gives ten chains their only coordinates. Every claim that
rests on those coordinates rests on a caveat nobody has priced.

It was never measured because the pipeline is built not to produce the comparison. A region
held behind its chain is skipped when the chain's own fold succeeded -- correctly, its residues
are inside coordinates the chain already has -- so for a chain inside the ceiling there is a
whole-chain fold and no region fold, and for a chain over it there is a region fold and no
whole-chain fold. The two folds of one region never coexist. This module makes them coexist on
purpose, for chains where both are possible, and reports three differences:

1. **Shape.** Calpha RMSD between the region's own fold and the same residues carved out of the
   whole-chain fold, after Kabsch superposition. This is the caveat's literal claim.
2. **The patch.** The hydrophobic patch at the region, measured in context against measured
   alone. `geometry` already argues the mechanism: accessibility "is a property of a residue in
   its surroundings, so a domain cut out of its chain and measured alone exposes the face that
   was never solvent-facing", and `keep` restricts the patch search without restricting the
   accessibility. So the in-context number comes from the whole-chain fold with `keep` set to
   the region's range, and the alone number from the region's own fold with no `keep`. This is
   the difference that matters to this project, because the patch is what it ranks on.
3. **Confidence.** Mean pLDDT of the region alone against the mean over the same residues
   inside the whole chain, which says whether the model itself reports the fragment as worse.

**Measured on the relative's curated chain and not on the pathogen's ORF translation, and that
is deliberate.** The question is what folding a fragment alone does to a fold, and the
provenance of the sequence does not bear on it. Taking the chain from the reviewed entry makes
every case reproducible from an accession and a range, with no six-frame translation and no
held-out reference in the way. The regions are the same `domains.annotated_regions` the fold
units come from, under the same minimum length and the same lone-helix refusal, so the units
compared here are the units the pipeline would have folded.

    python3 -m hpc_pilots.region_fidelity --cases 12
    python3 -m hpc_pilots.region_fidelity --accession P0DTC2 --span 14 1273
"""
import argparse
import json
import pathlib
import time

from .common import ROOT, provenance, write_json

MINIMUM_REGION_AA = 40
"""Matches `domains.MINIMUM_DOMAIN_AA`. Imported lazily below rather than at module scope so
this file can be read without the rest of the package; asserted equal when it is."""


def _ca_coordinates(text, low=None, high=None):
    """Calpha coordinates by residue number, optionally restricted to an inclusive range.

    Keyed on `resSeq` and never on arrival order, which is the rule `folded_structure._parse_pdb`
    states and the reason an ambiguous residue dropped from the model does not shift everything
    after it. A residue with no Calpha is absent from the mapping rather than zero-filled.
    """
    out = {}
    for line in text.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        try:
            number = int(line[22:26])
            point = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
        except ValueError:                                 # pragma: no cover
            continue
        if low is not None and not (low <= number <= high):
            continue
        out[number] = point
    return out


def _plddt(text, low=None, high=None):
    """Mean B-factor over Calpha atoms in range, on the 0-100 scale.

    ESMFold writes pLDDT into the B-factor column on a 0-1 scale through this lane, which
    `folded_structure` multiplies by 100 and records as `plddt_scale`. The same multiplication
    is applied here, and only to values that look like a fraction, so a file already on the
    0-100 scale is not scaled twice into nonsense.
    """
    values = []
    for line in text.splitlines():
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        try:
            number = int(line[22:26])
            b = float(line[60:66])
        except ValueError:                                 # pragma: no cover
            continue
        if low is not None and not (low <= number <= high):
            continue
        values.append(b)
    if not values:
        return None
    mean = sum(values) / len(values)
    return round(mean * 100 if mean <= 1.0 else mean, 2)


def kabsch_rmsd(first, second):
    """Calpha RMSD after optimal superposition, over the residues both sets hold.

    Kabsch rather than a raw coordinate difference, because the two folds are in unrelated
    frames -- each model emits its own -- and an unsuperposed RMSD between them measures the
    arbitrary orientation rather than the shape. Returns `(rmsd, residues_compared)`, and
    `(None, n)` when fewer than three residues are shared, where a superposition is not defined.
    """
    import numpy

    shared = sorted(set(first) & set(second))
    if len(shared) < 3:
        return None, len(shared)
    a = numpy.array([first[i] for i in shared], dtype=float)
    b = numpy.array([second[i] for i in shared], dtype=float)
    a -= a.mean(axis=0)
    b -= b.mean(axis=0)
    covariance = a.T @ b
    u, _s, vt = numpy.linalg.svd(covariance)
    # The reflection guard. Without it an SVD can return the improper rotation that mirrors one
    # structure onto the other, which fits a left-handed helix onto a right-handed one and
    # reports an RMSD no real superposition could reach.
    sign = numpy.sign(numpy.linalg.det(vt.T @ u.T))
    correction = numpy.diag([1.0, 1.0, sign])
    rotation = vt.T @ correction @ u.T
    rotated = (rotation @ a.T).T
    rmsd = float(numpy.sqrt(((rotated - b) ** 2).sum() / len(shared)))
    return round(rmsd, 3), len(shared)


def _patch(text, keep=None):
    """The patch measurement on predicted coordinates, in context when `keep` is given."""
    from . import geometry

    try:
        markers = geometry.markers_from_coordinates(text, label="region fidelity", keep=keep)
    except ImportError as failure:
        # A missing SASA backend is a fact about this machine and not about the structure,
        # which is the distinction `markers_from_coordinates` raises on rather than records.
        raise RuntimeError(f"the surface backend is unavailable: {failure}") from failure
    if markers.get("error"):
        return {"error": markers["error"]}
    # `hydrophobic_patch_area` and not `patch_median_a2`. The latter is a median the chain
    # takes over several structures of one domain and does not exist on a single measurement,
    # so reading it here returned None for every row of the first run of this module and made
    # the patch comparison silently empty while the RMSD column was full.
    return {"hydrophobic_patch_area_a2": markers.get("hydrophobic_patch_area"),
            "patch_fraction_of_surface": markers.get("patch_fraction_of_surface"),
            "patch_residues": markers.get("hydrophobic_patch_residues"),
            "exposed_fraction": markers.get("exposed_fraction"),
            "total_sasa_a2": markers.get("total_sasa"),
            "residues_in_range": markers.get("residues_in_range"),
            "macrocycle_surface_score": geometry.macrocycle_surface_score(markers)}


def cases_from_records(limit=None, records=None):
    """Chains the comparison is possible on, taken from the committed panel records.

    Possible means: the chain was identified through a reviewed relative, so there is a curated
    sequence and a curated annotation; the whole chain is inside the folding ceiling, so it can
    be folded whole; and the relative annotates at least one region that the pipeline would
    itself have folded. A chain over the ceiling is excluded for the reason the comparison
    exists -- it has no whole-chain fold to compare against, which is exactly why its region
    folds carry the caveat.
    """
    from . import domains
    from .chain import FOLD_CEILING

    assert MINIMUM_REGION_AA == domains.MINIMUM_DOMAIN_AA, (
        "the region minimum here must track domains.MINIMUM_DOMAIN_AA")

    paths = sorted(records or (ROOT / "runs").glob("onset_chain_NC_*.json"))
    seen, out = set(), []
    for path in paths:
        case = json.loads(path.read_text())
        # The two halves of a chain live in different sections and are joined on the target
        # name, which is the key both use: `scores` carries the reviewed relative it was
        # identified through, `foldable` carries the range on that relative and the length the
        # ceiling is applied to. Joining on position instead would attribute one chain's
        # relative to another the first time the two lists are emitted in a different order.
        homologs = {r.get("target"): r.get("nearest_homolog")
                    for r in (case.get("scores") or [])}
        for row in (case.get("foldable") or []):
            homolog = homologs.get(row.get("target"))
            length = row.get("declared_length_aa")
            if not (homolog and row.get("identified")):
                continue
            if not length or length > FOLD_CEILING:
                continue
            # A null range means the chain IS the whole reviewed entry rather than a cleavage
            # product of it, which `intake.mature_chains` reports by offering no sub-chain at
            # all. The comparison is the same either way, so the span is filled in from the
            # entry's own length rather than the case being dropped -- and dropping them would
            # bias the sample towards polyproteins, which are exactly the entries whose
            # regions are best annotated.
            span = row.get("chain_range_on_the_relative") or row.get("chain_range")
            if not span:
                span = [1, length]
            key = (homolog, tuple(span))
            if key in seen:
                continue
            seen.add(key)
            out.append({"pathogen": case.get("pathogen"),
                        "genome_accession": case.get("genome_accession"),
                        "chain": row.get("target"), "homolog": homolog,
                        "chain_span_on_the_relative": list(span),
                        "declared_length_aa": length})
            if limit and len(out) >= limit:
                return out
    return out


def compare(case, fold=None):
    """One chain: fold it whole, fold each of its regions alone, and report the differences."""
    from . import domains
    from .chain import _homolog_chain
    from .folded_structure import fold as default_fold

    fold = fold or default_fold
    start, end = case["chain_span_on_the_relative"]
    sequence = _homolog_chain(case["homolog"], [start, end], {})
    if not sequence:
        return {**case, "absence": "the relative's chain sequence could not be fetched"}

    regions = []
    for region in domains.annotated_regions(case["homolog"], within=[start, end]):
        low, high = region["start"], region["end"]
        length = high - low + 1
        if length < MINIMUM_REGION_AA or length >= len(sequence):
            continue
        if domains.is_a_lone_helix(region):
            # Refused for folding by `domains.fold_units`, so comparing a fold of it would
            # price a unit the pipeline does not produce. The reason is in that module: a
            # single helix has no monomeric structure, and a confident prediction of one is
            # confidence that the residues are helical rather than confidence in a shape.
            continue
        regions.append({"label": domains.label_for(region), "type": region.get("type"),
                        "entry_span": [low, high], "length_aa": length,
                        # Where the region sits inside the chain's own numbering, which is what
                        # the whole-chain fold is numbered in: ESMFold numbers from 1.
                        "span_in_the_chain": [low - start + 1, high - start + 1]})
    if not regions:
        return {**case, "absence": "the relative annotates no region this stage would fold"}

    whole = fold(sequence, target_name=f"{case['chain']} whole", record=False)
    if not whole.get("folded"):
        return {**case, "absence": f"the whole chain did not fold: {whole.get('absence')}"}
    whole_pdb = whole["pdb"]

    rows = []
    for region in regions:
        low, high = region["span_in_the_chain"]
        fragment = sequence[low - 1:high]
        alone = fold(fragment, target_name=f"{case['chain']} {region['label']}", record=False)
        row = {**region, "folded_alone": bool(alone.get("folded"))}
        if not alone.get("folded"):
            row["absence"] = alone.get("absence")
            rows.append(row)
            continue
        alone_pdb = alone["pdb"]
        # The region's own fold is numbered from 1, so it is shifted into the chain's numbering
        # before the two are intersected. Comparing them unshifted would superpose the region's
        # residue 1 onto the chain's residue 1, which is a different residue entirely.
        in_chain = _ca_coordinates(whole_pdb, low, high)
        standalone = {number + low - 1: point
                      for number, point in _ca_coordinates(alone_pdb).items()}
        rmsd, compared = kabsch_rmsd(in_chain, standalone)
        row.update({
            "ca_rmsd_a": rmsd,
            "residues_compared": compared,
            "mean_plddt_alone": _plddt(alone_pdb),
            "mean_plddt_in_the_chain": _plddt(whole_pdb, low, high),
            "patch_alone": _patch(alone_pdb),
            "patch_in_the_chain": _patch(whole_pdb, keep=(low, high)),
        })
        rows.append(row)
    return {**case, "whole_chain": {"length_aa": len(sequence),
                                    "mean_plddt": whole.get("mean_plddt"),
                                    "sequence_sha256": whole.get("sequence_sha256")},
            "regions": rows}


def run(cases=None, limit=8, record=True, path=None):
    """The comparison over several chains, with the aggregate the caveat is really about."""
    started = time.time()
    cases = cases if cases is not None else cases_from_records(limit=limit)
    results = [compare(case) for case in cases]
    paired = [r for case in results for r in (case.get("regions") or [])
              if r.get("ca_rmsd_a") is not None]

    def spread(values):
        values = sorted(v for v in values if v is not None)
        if not values:
            return None
        middle = len(values) // 2
        return {"n": len(values), "min": values[0], "max": values[-1],
                "median": round(values[middle] if len(values) % 2
                                else (values[middle - 1] + values[middle]) / 2, 3)}

    payload = {
        "what": "Calpha RMSD, patch area and confidence for an annotated region folded alone "
                "against the same residues folded inside their chain.",
        "why": "The domain route gives ten chains of the panel their only coordinates, and the "
               "records carry a stated caveat that a region folded alone has lost what its "
               "neighbours contributed. This prices that caveat instead of repeating it.",
        "cases": results,
        "regions_compared": len(paired),
        "ca_rmsd_a": spread([r["ca_rmsd_a"] for r in paired]),
        "mean_plddt_alone": spread([r["mean_plddt_alone"] for r in paired]),
        "mean_plddt_in_the_chain": spread([r["mean_plddt_in_the_chain"] for r in paired]),
        "patch_area_a2_alone": spread([(r["patch_alone"] or {}).get("hydrophobic_patch_area_a2")
                                       for r in paired]),
        "patch_area_a2_in_the_chain": spread(
            [(r["patch_in_the_chain"] or {}).get("hydrophobic_patch_area_a2")
             for r in paired]),
        "macrocycle_score_alone": spread(
            [(r["patch_alone"] or {}).get("macrocycle_surface_score") for r in paired]),
        "macrocycle_score_in_the_chain": spread(
            [(r["patch_in_the_chain"] or {}).get("macrocycle_surface_score") for r in paired]),
        "seconds": round(time.time() - started, 2),
        "provenance": provenance(),
    }
    if record:
        write_json(pathlib.Path(path) if path else ROOT / "runs/region_fidelity.json", payload)
    return payload


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", type=int, default=8,
                    help="Chains to compare, taken from the committed panel records")
    ap.add_argument("--accession", help="One reviewed accession to compare instead")
    ap.add_argument("--span", type=int, nargs=2, metavar=("START", "END"),
                    help="The chain's range on that accession")
    ap.add_argument("--no-record", action="store_true")
    args = ap.parse_args(argv)

    if args.accession:
        if not args.span:
            ap.error("--accession needs --span START END")
        cases = [{"pathogen": None, "genome_accession": None,
                  "chain": args.accession, "homolog": args.accession,
                  "chain_span_on_the_relative": list(args.span),
                  "declared_length_aa": args.span[1] - args.span[0] + 1}]
    else:
        cases = None

    payload = run(cases=cases, limit=args.cases, record=not args.no_record)
    print(f"\n{payload['regions_compared']} region(s) compared in {payload['seconds']:.0f}s")
    for key in ("ca_rmsd_a", "mean_plddt_alone", "mean_plddt_in_the_chain",
                "patch_area_a2_alone", "patch_area_a2_in_the_chain",
                "macrocycle_score_alone", "macrocycle_score_in_the_chain"):
        print(f"  {key:<32}{payload[key]}")
    for case in payload["cases"]:
        if case.get("absence"):
            print(f"\n  {case['chain']}: {case['absence']}")
            continue
        print(f"\n  {case['pathogen']} / {case['chain']} "
              f"({case['whole_chain']['length_aa']} aa, "
              f"pLDDT {case['whole_chain']['mean_plddt']})")
        for r in case["regions"]:
            if r.get("ca_rmsd_a") is None:
                print(f"    {r['label'][:40]:<42} not compared: "
                      f"{str(r.get('absence'))[:60]}")
                continue
            alone = (r['patch_alone'] or {}).get('hydrophobic_patch_area_a2')
            inside = (r['patch_in_the_chain'] or {}).get('hydrophobic_patch_area_a2')
            print(f"    {r['label'][:40]:<42}{r['length_aa']:>5} aa  "
                  f"RMSD {r['ca_rmsd_a']:>6} A   "
                  f"pLDDT {r['mean_plddt_alone']:>6} alone / "
                  f"{r['mean_plddt_in_the_chain']:>6} in chain   "
                  f"patch {alone} / {inside}")
    return payload


if __name__ == "__main__":
    main()
