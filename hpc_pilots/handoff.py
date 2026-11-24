"""The pipeline's output as a handoff: one row per target, one row per place to bind it.

Everything this project produces about a target is in its chain record, and none of it is in a
form another platform can read. A chemist asking "which protein, which residues, and what does
it do" has to open a 37-field surface row, know that `patch_positions` is numbered within the
mature chain while `functional_sites[].positions` is numbered along a reading frame of the
genome, and know which of the two numbering systems the structure viewer wants. That is a
reasonable thing to ask of the record and an unreasonable thing to ask of a consumer.

So this module emits two files per genome and one across all of them:

* **`runs/handoff/<stem>.json`** -- one object per target, carrying the identity, the score and
  its five terms, what the protein does, what it touches, the regions, every proposed site with
  its residues in every numbering the record supports, the measured area with its spread, and
  the caveats that apply to that row rather than to the project.
* **`runs/handoff/targets.csv`** -- one row per **proposed site**, which is the shape an intake
  form wants. A target with three kinds of site gets three rows; a target with none gets one row
  saying so, because a target that reached no site is a result and must not vanish from the
  table.

**Four site kinds, kept apart because they answer different questions.**

| kind | what it is | what it is for |
| --- | --- | --- |
| `hydrophobic_patch` | the largest unbroken exposed greasy area on a deposited structure | the surface a macrocycle lies across |
| `hydrophobic_patch_predicted` | the same measurement on predicted coordinates | the only route where nothing has been solved |
| `functional_site` | residues a curator marked as active site, binding site or cleavage site | where the chemistry happens |
| `site_patch` | the part of the patch that reaches the marked residues | a macrocycle that would also block the chemistry |

**Three numberings, named on every row.** `in_the_chain` is 1-based within the mature chain and
is the one to type into a structure viewer. `in_the_reading_frame` is 1-based along one open
reading frame of this genome and is meaningless without the frame, which travels in the same
row. `nt` is the genome coordinate, emitted only where `annotate` confirmed the reading frame by
reading its residues back, and carrying which end of the span that confirmation covered.

**Nothing here is a measurement on the pathogen.** Every function, family and partner is curated
annotation transferred from the nearest reviewed relative, and every structure behind an area is
either a deposition of a relative's domain or an ESMFold prediction. The columns say which, per
row, because a handoff that loses that distinction is worse than no handoff.
"""
import csv
import json
from pathlib import Path

from .common import ROOT, content_digest, provenance, write_json

SITE_KINDS = ("hydrophobic_patch", "hydrophobic_patch_predicted", "functional_site",
              "site_patch")

# One row per proposed site. Ordered so the identity comes first, the place to bind comes
# second, and the evidence and caveats come last -- which is the order a reader needs them in
# and not the order the record happens to hold them.
COLUMNS = (
    "accession", "pathogen", "family", "rank", "target", "parent", "uniprot", "tier",
    "score", "gate",
    "site_kind", "residue_count", "residues_in_the_chain", "residues_in_the_reading_frame",
    "residues_nt", "reading_frame", "nt_confirmed_at",
    "chain_length_aa", "chain_span_in_the_reading_frame", "chain_span_nt",
    "area_a2", "area_spread_a2", "area_fraction_of_surface", "measured_on",
    "on_predicted_coordinates", "mean_plddt",
    "function", "interacts_with_host", "interacts_with_viral", "families", "keywords",
    "already_drugged", "site_types", "caveat",
)


def _chain_local(positions, span):
    """Reading-frame positions as positions within the mature chain, 1-based.

    The conversion is one subtraction and it is the difference between a number a structure
    viewer accepts and a number it silently mislocates, so it is a function rather than an
    expression repeated at four call sites. Returns None where the span is unknown, because
    guessing the offset is how a residue list ends up pointing at the wrong protein.
    """
    if not positions or not span or len(span) != 2:
        return None
    return [int(p) - int(span[0]) + 1 for p in positions]


def _frame_local(positions, span):
    """Positions within the mature chain as positions along the reading frame, 1-based."""
    if not positions or not span or len(span) != 2:
        return None
    return [int(p) + int(span[0]) - 1 for p in positions]


def _nt(positions, frame, segment_length):
    """Each position's codon as a genome range, or None if the frame was not confirmed."""
    from .annotate import residue_to_nt

    if not positions or not frame or not frame.get("verified"):
        return None
    out = [residue_to_nt(p, frame, segment_length) for p in positions]
    return out if all(out) else None


def _sites(surface, span, frame, segment_length):
    """Every proposed site on one chain, in the four kinds, each with its own numbering.

    A kind that was not measured produces no row rather than a row of nulls, and the reason it
    was not measured travels on the target object. The exception is a chain with no site of any
    kind, which `rows_for_case` gives one row saying so -- a chain can be refused, it cannot
    disappear.
    """
    out = []
    patch = surface.get("patch_positions")
    if patch:
        out.append({
            "site_kind": "hydrophobic_patch",
            "residues_in_the_chain": list(patch),
            "residue_names": surface.get("patch_residue_names"),
            "area_a2": surface.get("patch_median_a2"),
            "area_spread_a2": surface.get("patch_spread_a2"),
            "area_range_a2": surface.get("patch_range_a2"),
            "area_fraction_of_surface": surface.get("patch_fraction_of_surface"),
            "measured_on": surface.get("patch_positions_measured_on"),
            "structures_measured": surface.get("structures_measured"),
            "on_predicted_coordinates": False,
            "what_it_is": "the largest unbroken exposed water-repelling area on a deposited "
                          "structure of this domain, which is the surface a macrocyclic "
                          "peptide has to lie across",
            "caveat": "The area is a median over structures of the same domain and the spread "
                      "is how far it moves between them. One macrocycle footprint is about 600 "
                      "square angstroms, so a spread of that size means the area is not a "
                      "property of the domain alone.",
        })
    predicted = surface.get("predicted_surface") or {}
    if predicted.get("measured") and predicted.get("patch_positions"):
        out.append({
            "site_kind": "hydrophobic_patch_predicted",
            "residues_in_the_chain": list(predicted["patch_positions"]),
            "residue_names": predicted.get("patch_residue_names"),
            "area_a2": predicted.get("patch_a2"),
            "area_fraction_of_surface": predicted.get("patch_fraction_of_surface"),
            "measured_on": predicted.get("measured_on") or predicted.get("model"),
            "mean_plddt": predicted.get("mean_plddt"),
            "confident_fraction": predicted.get("confident_fraction"),
            "on_predicted_coordinates": True,
            "only_route_to_a_surface": predicted.get("only_route_to_a_surface"),
            "ratio_to_deposited_median": predicted.get("ratio_to_deposited_median"),
            "what_it_is": "the same measurement on coordinates a predictor emitted, which is "
                          "the only route to a surface for a protein nothing has solved",
            "caveat": "An area measured on a prediction is comparable with another prediction "
                      "and not with a deposition; on the one chain carrying both, the "
                      "prediction read several times larger. The residue positions transfer "
                      "far better than the area does.",
        })
    sites = surface.get("functional_sites") or ()
    frame_positions = surface.get("site_positions_in_this_genome")
    if frame_positions:
        out.append({
            "site_kind": "functional_site",
            "residues_in_the_reading_frame": list(frame_positions),
            "residues_in_the_chain": _chain_local(frame_positions, span),
            "site_types": sorted({s.get("type") for s in sites if s.get("type")}),
            "site_descriptions": [s.get("description") for s in sites if s.get("description")],
            "ligands": sorted({s.get("ligand") for s in sites if s.get("ligand")}),
            # Not a measurement on coordinates at all, so neither True nor False is right and
            # a blank would be read as False beside the rows that are.
            "on_predicted_coordinates": "not a structural measurement",
            "what_it_is": "residues a curator marked on the nearest relative as an active "
                          "site, a binding site or a cleavage site, carried into this genome's "
                          "numbering through the alignment",
            "caveat": "A marked residue is an annotation transferred from a relative and was "
                      "not measured on this pathogen. A cleavage site is where this protein is "
                      "cut, which is not the same as somewhere to bind it.",
        })
    site_surface = surface.get("site_surface") or {}
    if site_surface.get("measured") and site_surface.get("site_patch_positions"):
        out.append({
            "site_kind": "site_patch",
            "residues_in_the_chain": list(site_surface["site_patch_positions"]),
            "area_a2": site_surface.get("site_patch_area_a2"),
            "area_fraction_of_surface": site_surface.get("site_patch_fraction_of_the_site"),
            "measured_on": site_surface.get("measured_on"),
            "shares_the_largest_patch": site_surface.get("site_shares_the_patch"),
            "at_the_annotated_residues": site_surface.get("measured_at_the_annotated_residues"),
            # This one is read off the deposited lane -- `predicted_surface` carries its own
            # `site_surface` for the other -- so it is a deposited measurement and says so.
            # Left as None it printed as a blank cell, which on a column whose other values
            # are True and False reads as False rather than as "not recorded".
            "on_predicted_coordinates": False,
            "what_it_is": "the part of the greasy patch that lies within reach of the marked "
                          "residues, which is where a macrocycle would both bind the surface "
                          "and block the chemistry",
            "caveat": "Read `at_the_annotated_residues` before this area. Where it is false "
                      "the measurement was taken at residues that are not the ones the "
                      "curator marked, and the area is about a different place.",
        })

    for site in out:
        chain_positions = site.get("residues_in_the_chain")
        if chain_positions and not site.get("residues_in_the_reading_frame"):
            site["residues_in_the_reading_frame"] = _frame_local(chain_positions, span)
        site["residue_count"] = len(chain_positions or
                                    site.get("residues_in_the_reading_frame") or ())
        site["residues_nt"] = _nt(site.get("residues_in_the_reading_frame"), frame,
                                  segment_length)
        site["numbering"] = {
            "residues_in_the_chain": "1-based within the mature chain. This is the numbering "
                                     "to type into a structure viewer.",
            "residues_in_the_reading_frame": "1-based along one open reading frame of this "
                                             "genome. Meaningless without `reading_frame`, "
                                             "which is on the target this site belongs to.",
            "residues_nt": "genome coordinate of each residue's codon, 1-based inclusive, low "
                           "coordinate first on either strand. Absent where the reading frame "
                           "was not confirmed.",
        }
    return out


def rows_for_case(case, annotation, panel=None):
    """Every target of one genome as a handoff object, sites and all."""
    surfaces = {row.get("target"): row for row in case.get("surfaces") or ()}
    foldables = {row.get("target"): row for row in case.get("foldable") or ()}
    annotated = {row.get("target"): row for row in (annotation or {}).get("chains") or ()}
    drugged = _drugged(case, panel)

    out = []
    for score in sorted(case.get("scores") or (), key=lambda r: r.get("rank", 0)):
        target = score.get("target")
        surface = surfaces.get(target) or {}
        foldable = foldables.get(target) or {}
        extra = annotated.get(target) or {}
        frame = extra.get("reading_frame") or {}
        span = foldable.get("span_in_this_genome")
        length = extra.get("segment_length_nt")
        sites = _sites(surface, span, frame, length)
        annotated_entry = extra.get("annotation") or {}
        about = annotated_entry.get("about_this_chain") or {}
        entry_about = annotated_entry.get("about_the_whole_entry") or {}
        interactions = annotated_entry.get("interactions") or {}

        out.append({
            "accession": case.get("genome_accession"),
            "pathogen": case.get("pathogen") or (annotation or {}).get("pathogen"),
            "rank": score.get("rank"),
            "target": target,
            "parent": score.get("parent"),
            "uniprot": score.get("nearest_homolog"),
            "tier": score.get("tier"),
            "score": score.get("score"),
            "score_terms": score.get("score_terms"),
            "gate": surface.get("gate"),
            "gate_means": "whether this chain sits where a binder can reach it and is worth "
                          "the stages below. `no` is a verdict and not a failure.",
            "chain_length_aa": foldable.get("declared_length_aa")
                               or surface.get("residues_in_range"),
            "chain_span_on_the_relative": score.get("chain_range"),
            "chain_span_in_the_reading_frame": span,
            "chain_span_nt": extra.get("chain_span_nt"),
            "nt_confirmed_at": extra.get("chain_span_nt_confirmed_at"),
            "reading_frame": frame.get("reading_frame"),
            "reading_frame_unresolved_because": frame.get("because"),
            "function": (about.get("FUNCTION") or [None])[0],
            "function_is_about": ("this mature chain" if about.get("FUNCTION")
                                  else "the whole entry" if entry_about.get("FUNCTION")
                                  else None),
            "function_of_the_whole_entry": (entry_about.get("FUNCTION") or [None])[0]
                                           if not about.get("FUNCTION") else None,
            "assembles_into": (about.get("SUBUNIT") or [None])[0],
            "catalyses": about.get("CATALYTIC ACTIVITY"),
            "needs_bound": about.get("COFACTOR"),
            "interacts_with_host": interactions.get("with_a_host_protein") or [],
            "interacts_with_this_virus": interactions.get("with_a_protein_of_this_virus") or [],
            "host_species": annotated_entry.get("declared_hosts") or [],
            "families": annotated_entry.get("signatures") or {},
            "go_terms": annotated_entry.get("go_terms") or {},
            "keywords": annotated_entry.get("keywords") or [],
            "complexes": annotated_entry.get("complexes") or [],
            "structures_deposited_on_the_relative":
                annotated_entry.get("structures_deposited"),
            "regions": [
                {"label": unit.get("label"), "type": unit.get("type"),
                 "description": unit.get("description"),
                 "span_in_the_reading_frame": unit.get("span"),
                 "span_in_the_chain": _chain_local(unit.get("span"), span)
                                      if unit.get("span") else None,
                 "span_on_the_relative": unit.get("entry_span"),
                 "length_aa": unit.get("length"),
                 "folded": unit.get("folded"),
                 "mean_plddt": unit.get("mean_plddt"),
                 "confident_fraction": unit.get("confident_fraction"),
                 "refused_because": unit.get("refused_because")}
                for unit in foldable.get("fold_units") or ()],
            "folded": foldable.get("folded"),
            "mean_plddt": foldable.get("mean_plddt"),
            "already_drugged": drugged.get(target),
            "sites": sites,
            "no_site_because": None if sites else (
                surface.get("unmeasured_because") or surface.get("no_surface_because")
                or "no surface was measured and no functional site was annotated for this "
                   "chain"),
            "not_annotated_because": extra.get("not_annotated_because"),
        })
    return out


def _drugged(case, panel=None):
    """Which targets of this genome an approved antiviral already acts on.

    Read from the benchmark panel's own label patterns through the matcher `benchmarks` already
    uses, so the substring traps it defends against are defended here too. The answer makes the
    table readable as "this one is already drugged, this one is not", which is the first thing
    anyone asks of a target list.
    """
    try:
        from benchmarks import task1, task1_labels
    except Exception:                                                   # noqa: BLE001
        return {}
    accession = case.get("genome_accession")
    entries = [e for e in task1_labels.pathogens(tier="all", panel=panel)
               if e.get("accession") == accession]
    if not entries:
        return {}
    out = {}
    for row in case.get("scores") or ():
        hits = [label["name"] for entry in entries for label in entry["targets"]
                if task1.matches(label, row)]
        out[row.get("target")] = hits or None
    return out


def _csv_rows(targets):
    """One flat row per proposed site, and one per target that reached none."""
    rows = []
    for target in targets:
        base = {
            "accession": target["accession"], "pathogen": target["pathogen"],
            "family": target.get("family"), "rank": target["rank"],
            "target": target["target"], "parent": target.get("parent"),
            "uniprot": target.get("uniprot"), "tier": target.get("tier"),
            "score": target.get("score"), "gate": target.get("gate"),
            "chain_length_aa": target.get("chain_length_aa"),
            "chain_span_in_the_reading_frame": _join(
                target.get("chain_span_in_the_reading_frame"), "-"),
            "chain_span_nt": _join(target.get("chain_span_nt"), "-"),
            "reading_frame": target.get("reading_frame"),
            "nt_confirmed_at": _join(target.get("nt_confirmed_at"), "; "),
            "function": _one_line(target.get("function")
                                  or target.get("function_of_the_whole_entry")),
            "interacts_with_host": _join(
                [p.get("partner_gene") or p.get("partner_accession")
                 for p in target.get("interacts_with_host") or ()], " "),
            "interacts_with_viral": len(target.get("interacts_with_this_virus") or ()),
            "families": _join([f"{name}:{item['id']}"
                               for name, items in (target.get("families") or {}).items()
                               for item in items[:2]], " "),
            "keywords": _join((target.get("keywords") or [])[:6], "; "),
            "already_drugged": _join(target.get("already_drugged"), "; "),
            "mean_plddt": target.get("mean_plddt"),
        }
        if not target["sites"]:
            rows.append({**base, "site_kind": "none",
                         "residue_count": 0, "caveat": target.get("no_site_because")})
            continue
        for site in target["sites"]:
            rows.append({
                **base,
                "site_kind": site["site_kind"],
                "residue_count": site.get("residue_count"),
                "residues_in_the_chain": _join(site.get("residues_in_the_chain"), " "),
                "residues_in_the_reading_frame": _join(
                    site.get("residues_in_the_reading_frame"), " "),
                "residues_nt": _join([f"{a}-{b}" for a, b in site.get("residues_nt") or ()],
                                     " "),
                "area_a2": site.get("area_a2"),
                "area_spread_a2": site.get("area_spread_a2"),
                "area_fraction_of_surface": site.get("area_fraction_of_surface"),
                "measured_on": site.get("measured_on"),
                "on_predicted_coordinates": site.get("on_predicted_coordinates"),
                "mean_plddt": site.get("mean_plddt", base["mean_plddt"]),
                "site_types": _join(site.get("site_types"), "; "),
                "caveat": _one_line(site.get("caveat")),
            })
    return rows


def _join(values, separator):
    if not values:
        return ""
    return separator.join(str(v) for v in values if v is not None)


def _one_line(text, limit=400):
    if not text:
        return ""
    collapsed = " ".join(str(text).split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit - 1] + "…"


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------
def _records(accession=None):
    """Chain records to export: one accession, or one per genome with a record on disk.

    **One per genome, which is not the same as one per file.** SARS-CoV-2 has two records: the
    per-genome one the panel wrote and `runs/onset_chain.json`, which is the landing page's.
    Both carry accession NC_045512.2, and exporting both put that genome into the CSV twice --
    49 rows where the record has 23. The per-genome file wins, which is the one
    `build_site.build_results` publishes a page for, and the landing record is used only for a
    genome no per-genome file covers.
    """
    if accession:
        stem = accession.replace(".", "_")
        direct = ROOT / f"runs/onset_chain_{stem}.json"
        return [direct if direct.exists() else ROOT / "runs/onset_chain.json"]
    found = [p for p in sorted((ROOT / "runs").glob("onset_chain_*.json"))
             if p.name != "onset_chain_genomes.json"]
    return found + [ROOT / "runs/onset_chain.json"]


def run_handoff(accession=None, record=True, annotate_missing=True):
    """Export every target of every genome, as one JSON per genome and one CSV over all."""
    from . import annotate as annotate_module

    families = _family_map()
    everything, per_genome = [], {}
    for path in _records(accession):
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        case = annotate_module._case(payload)
        genome = case.get("genome_accession") or payload.get("genome_accession")
        if not case.get("scores"):
            continue
        stem = (genome or path.stem).replace(".", "_")
        if stem in per_genome:
            continue
        side = ROOT / f"runs/annotation/{stem}.json"
        annotation = None
        if side.exists():
            annotation = json.loads(side.read_text())
        elif annotate_missing:
            try:
                annotation = annotate_module.run_annotate(path=path)
            except Exception as failure:                                # noqa: BLE001
                annotation = {"chains": [], "failed_because":
                              f"{type(failure).__name__}: {str(failure)[:200]}"}
        targets = rows_for_case(case, annotation)
        for target in targets:
            target["family"] = families.get(genome)
        per_genome[stem] = {
            "experiment": "handoff",
            "status": "complete",
            "what": "Every target this pipeline proposed for one genome, with the residues to "
                    "bind and what the protein does, in a form another platform can read.",
            "genome_accession": genome,
            "pathogen": targets[0]["pathogen"] if targets else None,
            "family": families.get(genome),
            "reference_excluded_organism": case.get("reference_excluded_organism"),
            "source_record": str(path.relative_to(ROOT)),
            "annotation_record": str(side.relative_to(ROOT)) if side.exists() else None,
            "site_kinds": list(SITE_KINDS),
            "targets": targets,
            "counts": {
                "targets": len(targets),
                "with_a_site": sum(1 for t in targets if t["sites"]),
                "with_a_measured_patch": sum(
                    1 for t in targets
                    if any(s["site_kind"] == "hydrophobic_patch" for s in t["sites"])),
                "with_a_functional_site": sum(
                    1 for t in targets
                    if any(s["site_kind"] == "functional_site" for s in t["sites"])),
                "with_nucleotide_coordinates": sum(
                    1 for t in targets if t.get("chain_span_nt")),
                "with_a_function": sum(
                    1 for t in targets
                    if t.get("function") or t.get("function_of_the_whole_entry")),
                "already_drugged": sum(1 for t in targets if t.get("already_drugged")),
            },
            "limitations": LIMITATIONS,
            "provenance": provenance(),
        }
        per_genome[stem]["content_sha256"] = content_digest(per_genome[stem])
        everything.extend(targets)

    rows = _csv_rows(everything)
    if record:
        for stem, payload in per_genome.items():
            write_json(ROOT / f"runs/handoff/{stem}.json", payload)
        write_csv(ROOT / "runs/handoff/targets.csv", rows)
        write_json(ROOT / "runs/handoff/index.json", {
            "experiment": "handoff-index",
            "what": "What the handoff covers, across every genome exported.",
            "genomes": len(per_genome),
            "targets": len(everything),
            "site_rows": len(rows),
            "rows_by_site_kind": {kind: sum(1 for r in rows if r["site_kind"] == kind)
                                  for kind in SITE_KINDS + ("none",)},
            "targets_already_drugged": sum(1 for t in everything if t.get("already_drugged")),
            # **Counted as the file carries it, not as the chain-level field alone.** An entry
            # that describes only itself -- 166 of the relatives here are single-chain entries
            # with no per-molecule annotation at all, and a few more name the molecule without
            # giving it its own FUNCTION -- still says what the protein does, and the CSV's
            # `function` column already falls back to it. Counting only `function` reported 146
            # where the file holds 327, which understated the export by more than half.
            "targets_with_a_function": sum(
                1 for t in everything
                if t.get("function") or t.get("function_of_the_whole_entry")),
            "targets_with_a_function_for_this_chain": sum(
                1 for t in everything if t.get("function")),
            "targets_with_only_the_entrys_function": sum(
                1 for t in everything
                if not t.get("function") and t.get("function_of_the_whole_entry")),
            "what_the_two_function_counts_mean":
                "A polyprotein entry carries one FUNCTION paragraph per mature chain, and that "
                "is the better answer where it exists. A single-chain entry carries one for "
                "itself, which describes the same protein. Both are in the file and the row "
                "says which it got; only the first is attributable to one product of a "
                "polyprotein.",
            "targets_with_a_host_interaction": sum(
                1 for t in everything if t.get("interacts_with_host")),
            "targets_with_nucleotide_coordinates": sum(
                1 for t in everything if t.get("chain_span_nt")),
            # **The denominator that makes the coverage readable.** A chain the fold stage
            # never located in this genome's translation has no span, so no reading frame and
            # no coordinate are possible for it -- that is a limit of identification, not of
            # this conversion. Against all 396 targets the coverage looks poor; against the
            # ones that have a span at all it is most of them.
            "targets_located_in_a_reading_frame": sum(
                1 for t in everything if t.get("chain_span_in_the_reading_frame")),
            "targets_with_no_span_to_convert": sum(
                1 for t in everything if not t.get("chain_span_in_the_reading_frame")),
            "columns": list(COLUMNS),
            "limitations": LIMITATIONS,
            "provenance": provenance(),
        })
    return {"genomes": per_genome, "targets": everything, "rows": rows}


def _family_map():
    """Virus family per accession, from the panel's own ICTV assignment where it has one."""
    path = ROOT / "data/benchmark_families.json"
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text()).get("families") or {})


def write_csv(path, rows):
    """One row per proposed site, with every declared column present on every row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})
    return path


LIMITATIONS = [
    "No row here is a measurement on the pathogen. Function, families, partners and marked "
    "residues are curated annotation on the nearest reviewed relative, transferred across a "
    "species gap. A deposited area is measured on a relative's domain; a predicted area is "
    "measured on coordinates a model emitted.",
    "An area measured on predicted coordinates is comparable with another prediction and not "
    "with a deposition. On the one chain carrying both, the prediction read several times "
    "larger. Residue positions transfer far better than areas do.",
    "A nucleotide range appears only where the reading frame was confirmed by reading its "
    "residues back against the relative's chain. `nt_confirmed_at` says which end of the span "
    "that confirmation covered: a chain spanning a ribosomal frameshift can be exact at one end "
    "and short at the other, and that row says so rather than being dropped or trusted whole.",
    "`residues_in_the_chain` is 1-based within the mature chain and is the numbering for a "
    "structure viewer. `residues_in_the_reading_frame` is 1-based along one reading frame of "
    "this genome and is meaningless without the frame in the same row.",
    "A cleavage site is where this protein is cut. It is a marked residue and it is not "
    "somewhere to bind.",
    "The ranking is the pipeline's five stated weights. Fitting those weights against the "
    "proteins approved antivirals already act on does not improve them, which is recorded in "
    "runs/rubric_fit.json; the ranking is carried by the terms rather than their weighting.",
    "`already_drugged` marks a target an approved or late-stage antiviral acts on, which is the "
    "label set this pipeline is scored against. It is not a claim that the target is easy, and "
    "an unmarked target is not a claim that it is novel.",
]
