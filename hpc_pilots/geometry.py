"""Surface markers that score a target the way a macrocycle sees it.

Written after the keyword rubric was found to score zero on six of the project's own eleven
targets, including Mpro and the neuraminidase, because their names carry no enzyme word. The
first conclusion drawn from that was that non-enzymatic targets fall outside the project's
scope. The opposite is true, and the theory of change says so directly: small molecules
"typically require deep, well-defined binding pockets and therefore leave many targets
(especially protein-protein interfaces) effectively undruggable", and macrocycles exist to
reach exactly those. Seven of the eleven targets are interfaces or non-enzymatic surfaces. A
rubric biased against them is biased against the molecule class the project is built on.

**Why this does not use a druggability score.** The obvious fix is fpocket or SiteMap, and it
would reintroduce the same bias through geometry instead of vocabulary. Those scores are
calibrated on small-molecule sites: deep, enclosed, buried, hydrophobic. A macrocycle binds
the opposite -- a large flat patch, a shallow groove, an interface with no cavity at all. On
pocket depth the SARS-CoV-2 main protease outscores the spike receptor-binding domain, which
is correct for a small molecule and backwards for this project. So the markers here are
interface-like rather than pocket-like, and the distinction is the whole point of the module.

**What is measured, and why each one.**

* `hydrophobic_patch_area` -- the largest contiguous run of exposed hydrophobic surface. This
  is the single most macrocycle-relevant number available from a structure. A macrocyclic
  peptide buries roughly 400 to 800 square angstroms on binding, far more than a small
  molecule, and it needs a contiguous patch to do it against. Computed as connected
  components over exposed hydrophobic residues whose alpha carbons are within 8 angstroms.
* `exposed_fraction` -- how much of the protein is surface at all. A small domain presenting
  most of itself is a better display reagent than a buried subunit of a complex.
* `patch_fraction_of_surface` -- the patch as a share of total surface, which separates a
  genuinely large contiguous site from a large protein with many small ones.
* `sequons_near_patch` -- N-linked glycosylation sequons within reach of the patch. This is the
  honest counterweight on viral surface proteins: the haemagglutinin stem and the spike are
  conserved and exposed and also heavily shielded, and a patch covered in glycan is not the
  binding opportunity its area suggests.

Every marker is computed from one structure and needs no literature, which is the property the
holdout work established matters: a pocket can be measured on a protein nobody has screened,
while an antibody count or a citation count cannot.
"""
import json
import pathlib
import urllib.request
import warnings

from .common import ROOT

RCSB_FILE = "https://files.rcsb.org/download/"

# Subcellular locations that put a protein where a macrocycle can reach it without crossing a
# membrane. Kept here so `intake` and the gate read one list.
EXPOSED_LOCATIONS = ("virion membrane", "host cell membrane", "virion surface", "secreted",
                     "cell surface", "host extracellular")

# Kyte-Doolittle positive residues, the ones that form binding hot spots. Tryptophan and
# tyrosine are included despite middling hydropathy because aromatic side chains dominate
# peptide and macrocycle interfaces in practice.
HYDROPHOBIC = set("AVILMFCWY")

# Relative solvent accessibility above which a residue counts as surface. Twenty-five per cent
# is the conventional cut and is used here unchanged rather than tuned, since tuning it on the
# targets whose ranking is the claim would be fitting the instrument to the answer.
EXPOSED_RSA = 0.25

# Alpha carbons closer than this are treated as adjacent when growing a patch. Ten angstroms
# is the distance commonly used to define residues belonging to one protein-protein interface,
# and it is chosen on that ground rather than on its effect here.
#
# The effect was measured anyway, because a cutoff that produces the claim is worth being
# suspicious of. Over six structures -- two enzyme pockets and four interfaces -- the gap in
# mean score between the two classes is +0.43 at eight angstroms, +0.84 at ten and +0.34 at
# twelve. Ten is the best separation as well as the conventional value, which is the kind of
# coincidence that should be stated rather than relied on. What matters is that the gap is
# positive at all three: the direction of the finding does not depend on the cutoff, only its
# size does. Eight fragments the surface into components too small to be a binding site -- the
# spike trimer's largest patch is seven residues there -- and twelve begins merging sites that
# are distinct.
CONTACT_A = 10.0

# How far an N-linked glycan reaches from its attachment point. A complex N-glycan extends
# roughly 15 to 20 angstroms, so a sequon within this distance of a patch can shield it even
# though the asparagine is not part of it.
GLYCAN_REACH_A = 15.0


def fetch_structure(pdb_id, directory=None):
    """One PDB entry, cached. Returns the local path."""
    directory = directory or ROOT / "data/raw/structures"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{pdb_id.lower()}.pdb"
    if not path.exists():
        urllib.request.urlretrieve(f"{RCSB_FILE}{pdb_id.lower()}.pdb", path)
    return path


def _residues(path, chain=None):
    """Per-residue relative accessibility, sequence and alpha-carbon coordinates.

    The structure is copied to an ASCII path before freesasa sees it. freesasa is a C library
    that takes a filename, and this repository lives under a directory containing a typographic
    apostrophe -- "William's MacBook Pro" with U+2019 -- which it cannot encode. The failure is
    a UnicodeEncodeError from inside the extension, nowhere near anything structural, and it
    would reappear on any machine whose checkout path is not plain ASCII.
    """
    import shutil
    import tempfile

    import freesasa

    with tempfile.TemporaryDirectory() as scratch:
        safe = pathlib.Path(scratch) / "structure.pdb"
        shutil.copyfile(path, safe)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = freesasa.Structure(str(safe))
            result = freesasa.calc(structure)
            areas = result.residueAreas()

    coords = {}
    for line in open(path):
        if not line.startswith("ATOM") or line[12:16].strip() != "CA":
            continue
        key = (line[21], line[22:27].strip())
        coords[key] = (float(line[30:38]), float(line[38:46]), float(line[46:54]))

    out = []
    for chain_id, residues in areas.items():
        if chain and chain_id != chain:
            continue
        for number, area in residues.items():
            position = coords.get((chain_id, number))
            if position is None:
                continue
            relative = area.relativeTotal
            if relative is None:
                continue
            out.append({"chain": chain_id, "number": number,
                        "residue": area.residueType, "rsa": relative,
                        "sasa": area.total, "xyz": position})
    return out


def _resnum(number):
    """The numeric part of an author residue number, insertion code dropped.

    PDB author numbering is a string: `52`, `52A`, `-3`. The clip below and the sequon
    ordering both need it as an integer, and both were parsing it inline.
    """
    digits = "".join(c for c in str(number) if c.isdigit() or c == "-")
    return int(digits) if digits not in ("", "-") else 0


def _letter(three):
    table = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
             "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
             "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
             "TYR": "Y", "VAL": "V"}
    return table.get(three.strip().upper(), "X")


def numbering_offset(residues, sequence, seed=12):
    """What this chain's author numbering has to be shifted by to read as UniProt positions.

    PDBe's mapping endpoint publishes an author residue number for some entries and `null`
    for others -- 9fc2 maps the spike at UniProt 332 to 534 and gives no author number for
    either end -- so a clip depending on it alone stops clipping on roughly half of what it is
    pointed at, which is the failure this change exists to end. Assuming a convention instead
    is worse: viral proteins are often deposited in their parent's numbering, mature chains of
    a polyprotein are usually renumbered from 1, and picking either one blindly selects the
    wrong residues on everything that uses the other.

    So the offset is read off the structure that was downloaded anyway. A window of residues
    with consecutive author numbers is a substring of the reference sequence wherever the
    chain really sits, and finding it gives the shift directly; the shift is then scored
    against every observed residue, so a window that matched by chance scores badly and is
    discarded. Returns `(offset, agreement)` with `unp_position = author_number + offset`, or
    `(None, 0.0)` when nothing lines up.

    Three windows rather than one, because a single window can fall across a modified residue
    or an engineered mutation, and because a short repeat can place it in the wrong copy.
    """
    if not residues or not sequence:
        return None, 0.0
    ordered = sorted(residues, key=lambda r: _resnum(r["number"]))
    numbers = [_resnum(r["number"]) for r in ordered]
    letters = [_letter(r["residue"]) for r in ordered]

    # Runs of consecutive author numbers, since only those are substrings of the reference.
    runs, start = [], 0
    for i in range(1, len(numbers) + 1):
        if i == len(numbers) or numbers[i] != numbers[i - 1] + 1:
            if i - start >= seed:
                runs.append((start, i))
            start = i
    if not runs:
        return None, 0.0

    candidates = set()
    for low, high in runs:
        for fraction in (0.25, 0.5, 0.75):
            at = low + int((high - low - seed) * fraction)
            window = "".join(letters[at:at + seed])
            if "X" in window:
                continue
            found = sequence.find(window)
            if found >= 0:
                candidates.add((found + 1) - numbers[at])

    best, score = None, 0.0
    for offset in candidates:
        hits = total = 0
        for number, letter in zip(numbers, letters):
            position = number + offset
            if not 1 <= position <= len(sequence):
                continue
            total += 1
            hits += letter == sequence[position - 1]
        agreement = hits / total if total else 0.0
        if agreement > score:
            best, score = offset, round(agreement, 4)
    return best, score


# How much of a chain's observed sequence has to land where the inferred offset says before
# the clip is taken from it. High on purpose: not clipping leaves the status quo, which is
# merely imprecise, while clipping at a wrong offset selects the wrong residues entirely.
NUMBERING_AGREEMENT = 0.9


def surface_markers(pdb_id, chain=None, keep=None, verify=None):
    """Interface-shaped markers for one structure, optionally for one domain of it.

    `chain` restricts the calculation to one polymer, which matters for a complex: the spike
    receptor-binding domain deposited with ACE2 has its best patch buried by the partner, and
    measuring the pair would report the interface as inaccessible when it is the target.

    `keep` is an inclusive `(low, high)` range of author residue numbers, and restricts the
    patch search to a domain within the chain. **It does not restrict the accessibility
    calculation**, and the difference is the whole point of the argument. Solvent accessibility
    is a property of a residue in its surroundings, so a domain cut out of its chain and
    measured alone exposes the face that was never solvent-facing. The SASA is therefore
    computed on the deposited file exactly as before, and only the search for a patch -- and
    the surface the patch is reported as a fraction of -- is confined to the domain.

    Without it the quantity is the largest patch anywhere on the deposited chain, which for a
    construct much larger than the target is not the target's patch at all: SARS-CoV-2's nsp3
    is a 1,945-residue chain carrying the 265-residue papain-like protease, and the largest
    patch on the construct belongs to whichever of its domains happens to carry one.
    """
    residues = _residues(fetch_structure(pdb_id), chain)
    if not residues:
        return {"pdb_id": pdb_id, "chain": chain, "residues": 0, "error": "no residues read"}
    return _markers(residues, pdb_id, chain, keep, verify)


def markers_from_coordinates(text, label="predicted coordinates"):
    """The same measurement, on coordinates a model emitted rather than a deposited file.

    This is the only route to a surface for a target nothing has solved, and it is the lane a
    genuinely novel pathogen arrives on: no deposition covers the chain, so either the surface
    is measured on a prediction or it is not measured at all. The arithmetic is identical --
    the same freesasa accessibility, the same contact-linked hydrophobic patch, the same
    footprint comparison -- because the quantity is a property of coordinates and does not ask
    where they came from.

    What differs is its standing, and the return says so rather than leaving a reader to infer
    it. `on_predicted_coordinates` and `is_deposited_measurement` travel with the numbers, and
    the caller must keep them out of any total that counts depositions: a patch computed on a
    prediction inherits every error of the prediction, and over a low-confidence stretch it
    describes a fold the model never committed to. It is reported beside a deposited
    measurement, never pooled with one, and it is not in the ranking -- the rubric's five
    weights were fixed before the run and no surface term is among them.

    `keep` is not offered. A deposited file is a construct that may carry much more than the
    target, which is what `keep` exists to confine the patch search to; these coordinates are
    the folded chain itself, numbered from one, so there is nothing to clip.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        path = pathlib.Path(scratch) / "predicted.pdb"
        path.write_text(text)
        residues = _residues(path)
        if not residues:
            return {"pdb_id": label, "chain": None, "residues": 0,
                    "error": "no residues read from the predicted coordinates",
                    "on_predicted_coordinates": True, "is_deposited_measurement": False}
        markers = _markers(residues, label, None, None, None)
    markers["on_predicted_coordinates"] = True
    markers["is_deposited_measurement"] = False
    return markers


def _markers(residues, pdb_id, chain=None, keep=None, verify=None):
    """The measurement itself, over residues already read from a file.

    Separated from `surface_markers` so a prediction and a deposition go through one body of
    arithmetic instead of two. The `chain` and `keep` semantics are documented on
    `surface_markers`, which is the registered tool; this function is the shared half.
    """
    # No author numbering from the mapping endpoint, but a reference sequence to locate the
    # chain in: infer the shift from the residues themselves and clip at it.
    agreement, offset = None, None
    if keep is None and verify and verify.get("sequence") and verify.get("interval"):
        offset, agreement = numbering_offset(residues, verify["sequence"])
        if offset is not None and agreement >= NUMBERING_AGREEMENT:
            low, high = verify["interval"]
            keep = (low - offset, high - offset)

    # The patch search runs over the domain; the accessibility behind it was computed in the
    # whole chain's company, partners included.
    within = residues
    if keep:
        low, high = keep
        within = [r for r in residues if low <= _resnum(r["number"]) <= high]
        if not within:
            return {"pdb_id": pdb_id, "chain": chain, "residues": len(residues),
                    "residues_in_range": 0, "keep": [low, high],
                    "error": f"no residue of this chain is numbered inside {low}-{high}"}

    exposed = [r for r in within if r["rsa"] >= EXPOSED_RSA]
    total_sasa = sum(r["sasa"] for r in within)
    hydrophobic = [r for r in exposed if _letter(r["residue"]) in HYDROPHOBIC]

    # Connected components over exposed hydrophobic residues. A patch is what a macrocycle can
    # lie across, so adjacency is spatial and not sequential -- a binding surface is routinely
    # assembled from residues far apart in sequence.
    index = {id(r): i for i, r in enumerate(hydrophobic)}
    parent = list(range(len(hydrophobic)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, first in enumerate(hydrophobic):
        for j in range(i + 1, len(hydrophobic)):
            second = hydrophobic[j]
            dx = first["xyz"][0] - second["xyz"][0]
            dy = first["xyz"][1] - second["xyz"][1]
            dz = first["xyz"][2] - second["xyz"][2]
            if dx * dx + dy * dy + dz * dz <= CONTACT_A * CONTACT_A:
                a, b = find(i), find(j)
                if a != b:
                    parent[a] = b

    groups = {}
    for i, residue in enumerate(hydrophobic):
        groups.setdefault(find(i), []).append(residue)
    patches = sorted(groups.values(), key=lambda g: -sum(r["sasa"] for r in g))
    best = patches[0] if patches else []
    patch_area = sum(r["sasa"] for r in best)

    # Sequons are read on the patch, not the whole protein: shielding matters where the
    # binding would happen. Sequence order within the chain is reconstructed from residue
    # numbers, which is adequate for the N-X-S/T motif.
    # Scanned over the whole chain even when the patch search was confined to a domain: a
    # glycan anchored on the neighbouring domain still reaches across and shields the site.
    # Only membership in the patch is restricted, never what can shadow it.
    ordered = sorted(residues, key=lambda r: (r["chain"], _resnum(r["number"])))
    sequence = "".join(_letter(r["residue"]) for r in ordered)
    # Counted NEAR the patch, not in it, and the distinction is the difference between a live
    # term and a dead one. The patch is hydrophobic by construction and asparagine is not
    # hydrophobic, so a sequon can never be a patch member and the first version of this
    # returned zero for every structure -- including the spike trimer, which is one of the most
    # heavily glycosylated proteins known. A glycan does not have to sit inside a binding site
    # to block it; it has to be close enough to cover it.
    sequons = 0
    for i in range(len(ordered) - 2):
        if not (_letter(ordered[i]["residue"]) == "N"
                and _letter(ordered[i + 1]["residue"]) != "P"
                and _letter(ordered[i + 2]["residue"]) in "ST"):
            continue
        site = ordered[i]["xyz"]
        for residue in best:
            dx = site[0] - residue["xyz"][0]
            dy = site[1] - residue["xyz"][1]
            dz = site[2] - residue["xyz"][2]
            if dx * dx + dy * dy + dz * dz <= GLYCAN_REACH_A * GLYCAN_REACH_A:
                sequons += 1
                break

    return {
        "pdb_id": pdb_id, "chain": chain,
        "residues": len(residues), "residues_in_range": len(within),
        "keep": list(keep) if keep else None,
        "numbering_agreement": agreement, "numbering_offset": offset,
        "exposed_residues": len(exposed),
        "exposed_fraction": round(len(exposed) / len(within), 3),
        "total_sasa": round(total_sasa, 1),
        "hydrophobic_patch_area": round(patch_area, 1),
        "hydrophobic_patch_residues": len(best),
        "patch_fraction_of_surface": round(patch_area / total_sasa, 3) if total_sasa else 0.0,
        "patches_found": len(patches),
        "sequons_near_patch": sequons,
        "sequence_length": len(sequence),
    }


# A macrocyclic peptide buries roughly this much on binding. A patch far below it cannot host
# one; a patch far above it is a large surface with no single site, which the fraction catches.
MACROCYCLE_FOOTPRINT_A2 = 600.0


def macrocycle_surface_score(markers):
    """One number from the markers, weighted for a macrocycle rather than a small molecule.

    Deliberately rewards a large contiguous hydrophobic patch and penalises glycan shielding
    on it, which is the opposite emphasis to a small-molecule druggability score. The weights
    are stated and not fitted; there is no labelled set of macrocycle-amenable targets to fit
    against, and inventing one from the targets under test would be circular.
    """
    if markers.get("error"):
        return None
    patch = markers["hydrophobic_patch_area"]
    # Saturating rather than linear: twice the footprint is not twice as good, it is one site
    # a macrocycle can bind either way.
    size = min(1.0, patch / MACROCYCLE_FOOTPRINT_A2)
    contiguity = min(1.0, markers["patch_fraction_of_surface"] / 0.15)
    exposure = markers["exposed_fraction"]
    shielding = min(1.0, markers["sequons_near_patch"] / 4.0)
    return round(3.0 * size + 1.5 * contiguity + 1.0 * exposure - 1.5 * shielding, 4)


# ---------------------------------------------------------------------------
# Attaching surface markers to a ranked candidate.

def representative_structure(accession, minimum_residues=50, interval=None):
    """The best-resolution deposited chains for one UniProt accession, or None.

    Picks on resolution rather than on coverage, because the markers are computed on whatever
    is present and a sharper structure gives better side-chain placement, which is what the
    accessibility calculation depends on. Chains shorter than `minimum_residues` are skipped:
    a peptide fragment co-crystallised with something else has a surface, but not the target's.

    **Rows are grouped per chain before anything is decided.** SIFTS returns one row per
    contiguous mapped segment, so a chain that maps to the accession in two pieces arrives as
    two rows. Judged separately, each piece can fail the coverage threshold while the chain
    passes it, and the same chain can be offered twice as if it were two structures.

    Each pick carries `keep`, the author-numbering range answering to `interval`, read from
    the entry's own UniProt-to-author mapping. The offset is constant within one mapped
    segment, which is what makes the conversion exact rather than an assumption about the
    construct -- but it has to come from the mapping endpoint, because the `start` field
    beside `unp_start` in `best_structures` is the residue's index within the entity and not
    its author number. A chain whose entry gives no author number is measured unclipped and
    says so, rather than being clipped in a numbering it does not use.

    Returned as a list rather than a single pick, because the best-resolution entry is often
    the newest and the newest are frequently mmCIF-only: RCSB stopped emitting legacy PDB
    files for recent depositions, so the 1.16-angstrom main-protease structure this chose
    first answers 404. Picking on resolution selects for exactly the entries most likely to be
    missing the format, which is why one pick is not enough.
    """
    from .structures_evidence import chains, residue_mapping

    by_chain = {}
    for row in chains(accession):
        if row.get("resolution") is None:
            continue
        if row.get("unp_start") is None or row.get("unp_end") is None:
            continue
        by_chain.setdefault((row["pdb_id"], row["chain_id"]), []).append(row)

    ranked = []
    for (pdb, chain), rows in by_chain.items():
        mapped = sum((r["unp_end"] or 0) - (r["unp_start"] or 0) for r in rows)
        if mapped < minimum_residues:
            continue
        if interval:
            # Restricted to chains actually covering the domain. Without this every mature
            # chain of a polyprotein resolves to the same parent accession and therefore to
            # the same structure: the main protease, the polymerase, the helicase and the
            # papain-like protease of SARS-CoV-2 all received one surface score of 2.85 and
            # one patch of 381 square angstroms, which is the replicase's and none of theirs.
            want = interval["end"] - interval["start"] + 1
            overlap = sum(max(0, min(r["unp_end"], interval["end"])
                              - max(r["unp_start"], interval["start"]) + 1) for r in rows)
            if overlap < 0.5 * want:
                continue
        ranked.append((min(r["resolution"] for r in rows), pdb, chain))
    if not ranked:
        return None
    ranked.sort()

    # The author numbering is resolved here and not in the loop above, because it costs one
    # request per entry and the loop above runs over every chain mapped to the accession --
    # 7,227 of them for the SARS-CoV-2 replicase. Asking PDBe for all of those to use eight is
    # how this earned a connection reset; the coverage filter needs only the UniProt numbering
    # that `chains` already returned, and the author numbering is needed only where a
    # measurement will actually be taken.
    picks = []
    for resolution, pdb, chain in ranked[:8]:
        keep = None
        if interval:
            try:
                segments = residue_mapping(pdb, accession).get(chain, ())
            except Exception:                             # noqa: BLE001
                segments = ()
            spans = []
            for unp_low, unp_high, auth_low, _ in segments:
                low = max(unp_low, interval["start"])
                high = min(unp_high, interval["end"])
                if low > high:
                    continue
                offset = auth_low - unp_low
                spans.append((low + offset, high + offset))
            # One range across the mapped segments. A chain whose segments are far apart in
            # author numbering keeps the residues between them in the search, which is the
            # conservative direction: it can only make the patch larger, never smaller, so a
            # clipped measurement never overstates how confined the search was.
            #
            # No span at all means the entry published no author numbering for this chain, or
            # the lookup failed. The chain is then measured whole and says so, through `keep`
            # coming back null on the markers, rather than being dropped or clipped in a
            # numbering it does not use.
            keep = (min(a for a, _ in spans), max(b for _, b in spans)) if spans else None
        picks.append({"pdb_id": pdb, "chain": chain, "resolution": resolution, "keep": keep})
    return picks


def surface_for_accession(accession, interval=None, replicates=5, cache={}):
    """Surface markers for a UniProt accession via its best deposited structure.

    Cached in process. A genome's candidates frequently share a homolog -- every mature chain
    of a polyprotein resolves to the same entry -- and the SASA calculation is the expensive
    step in the ranking.
    """
    # `replicates` belongs in the key and was missing from it. The median is taken over up to
    # `replicates` structures, so it is an input to the answer and not a performance knob: with
    # it absent from the key, a call with `replicates=1` and a later call with `replicates=5`
    # returned the same cached number, and the second caller's argument was silently ignored.
    # Registering this function as a tool is what made that reachable -- `replicates` is now a
    # declared parameter a policy can set, so a key that drops it would let the schema promise
    # a control the code does not offer, which is the defect `pick_batch` already had once with
    # `forest_trees` and `ucb_beta`.
    key = (accession, tuple(interval.values()) if interval else None, replicates)
    if key in cache:
        return cache[key]
    try:
        picks = representative_structure(accession, interval=interval)
        if not picks:
            # Why there was nothing to measure, not merely that there was nothing. The two
            # cases read identically from outside and are opposite findings: an accession
            # PDBe maps no chain to at all is a statement about which entry the pipeline
            # landed on, and an accession with chains none of which reach the domain is a
            # statement about what has been solved of that protein. Influenza is the first
            # case across the board -- its reviewed entries are split so finely by strain
            # that the nearest curated relative of a held-out protein is usually an entry
            # with no structure of its own, while influenza proteins are among the best
            # solved in virology.
            from .structures_evidence import chains

            try:
                mapped = len(chains(accession))
            except Exception:                             # noqa: BLE001
                mapped = None
            cache[key] = {
                "error": (f"PDBe maps no deposited chain to {accession} at all. The protein "
                          f"may be well solved under another accession; this entry is not it."
                          if mapped == 0 else
                          f"{mapped} chains map to {accession} and none covers "
                          f"{interval['start']}-{interval['end']}" if interval and mapped
                          else f"no usable deposited chain maps to {accession}"),
                "chains_mapped_to_accession": mapped,
                "surface_score": None}
            return cache[key]
        # Fetched once per accession and only when a domain was asked for: the fallback test
        # for a chain whose entry published no author numbering needs the reference sequence
        # those numbers would index into.
        verify = None
        if interval and any(pick.get("keep") is None for pick in picks):
            try:
                from .intake import _entry

                verify = {"sequence": _entry(accession)["sequence"]["value"],
                          "interval": (interval["start"], interval["end"])}
            except Exception:                             # noqa: BLE001
                verify = None
        # Measured over several structures of the same domain, not one. A single structure is
        # one sample of a conformational ensemble and the quantity being thresholded moves
        # with it: SARS-CoV-2's main protease measures 343 square angstroms through one
        # homolog's entry and 825 through another, a factor of 2.4. Reporting one number from
        # one crystal as though it were the protein is what made the amenability gate
        # unusable, and a median over a handful with the spread beside it is what makes it
        # readable -- a target whose patch is 600 give or take 40 is a different claim from
        # one that is 600 give or take 300.
        samples, reasons = [], []
        for pick in picks:
            try:
                measured = surface_markers(pick["pdb_id"], pick["chain"],
                                           keep=pick.get("keep"), verify=verify)
            except ImportError:
                # Never swallowed into a per-structure skip. A missing SASA backend fails
                # identically on every structure, so swallowing it turns one environment
                # problem into "no structure could be parsed" and sends the reader to the
                # network. Raised so the caller's `requires` check is what reports it.
                raise
            except Exception as failure:                  # noqa: BLE001
                reasons.append(f"{pick['pdb_id']}:{pick['chain']}: "
                               f"{type(failure).__name__}: {str(failure)[:120]}")
                continue
            if measured.get("error"):
                reasons.append(f"{pick['pdb_id']}:{pick['chain']}: {measured['error']}")
                continue
            measured["resolution"] = pick["resolution"]
            measured["surface_score"] = macrocycle_surface_score(measured)
            samples.append(measured)
            if len(samples) >= replicates:
                break
        if not samples:
            # Why nothing was measured, not just that nothing was. The bare string blamed the
            # network or the file in every case, and for a long time the actual cause was an
            # absent `freesasa` -- the C library every marker here is computed with, missing
            # from the scientific environment, reported as a download failure. A target with
            # no usable deposited structure and an environment with no SASA backend are
            # opposite situations: the first is a finding about the target and the ordinary
            # case for a novel pathogen, the second is a machine that cannot answer the
            # question at all. `tools.py` declares `freesasa` under `requires` so the caller
            # returns an explicit charged absence before dispatch, and this is the message for
            # a direct caller that bypasses it.
            cache[key] = {"error": f"no candidate structure could be measured: "
                                   f"{reasons[0] if reasons else 'no structure parsed'}",
                          "reasons": sorted(set(reasons)),
                          "structures_offered": [f"{p['pdb_id']}:{p['chain']}" for p in picks],
                          "surface_score": None}
            return cache[key]

        def median(values):
            ordered = sorted(values)
            middle = len(ordered) // 2
            if len(ordered) % 2:
                return ordered[middle]
            return (ordered[middle - 1] + ordered[middle]) / 2

        areas = [m["hydrophobic_patch_area"] for m in samples]
        scores = [m["surface_score"] for m in samples if m["surface_score"] is not None]
        markers = dict(samples[0])
        # Every quantity a consumer reads has to come from the same place as the area, which
        # is the median over the replicates. Copying the first sample's dict and overriding
        # only the areas left the patch fraction and the exposed fraction belonging to
        # whichever structure happened to sort first: the influenza HA stem reported a median
        # patch of 0 beside a patch fraction of 0.281, two numbers about different structures
        # printed as if they described one, and the gate reads both.
        for field in ("patch_fraction_of_surface", "exposed_fraction"):
            markers[field] = round(median([m[field] for m in samples]), 3)
        for field in ("residues", "residues_in_range", "hydrophobic_patch_residues"):
            values = [m[field] for m in samples if m.get(field) is not None]
            markers[field] = int(median(values)) if values else None
        # Singular facts about one structure, and misleading on an aggregate. The per-structure
        # values stay reachable through `structures_used`.
        for field in ("keep", "numbering_offset", "numbering_agreement", "resolution"):
            markers.pop(field, None)
        markers["hydrophobic_patch_area"] = round(median(areas), 1)
        markers["patch_area_min"] = round(min(areas), 1)
        markers["patch_area_max"] = round(max(areas), 1)
        markers["patch_area_spread"] = round(max(areas) - min(areas), 1)
        markers["structures_measured"] = len(samples)
        markers["structures_used"] = [f"{m['pdb_id']}:{m['chain']}" for m in samples]
        # How many of the replicates the patch search was actually confined to the domain on.
        # A median taken over a mix of clipped and whole-chain measurements is a mix of two
        # quantities, and a reader cannot see that from the median.
        markers["structures_clipped_to_domain"] = sum(1 for m in samples if m.get("keep"))
        markers["surface_score"] = round(median(scores), 4) if scores else None
        # When this claim became knowable, which is the earliest deposition among the
        # structures it was actually measured on. Without it the markers are a 2026
        # calculation with no date, and `Dossier.observe` compares `available_from` against
        # the decision clock -- so an undated claim enters a dossier at any clock and silently
        # opens the temporal boundary the historical replays depend on. A patch is not
        # evidence from the day it was computed; it is evidence from the day the coordinates
        # that show it were deposited.
        #
        # Earliest rather than latest, matching `structures_evidence`'s own rule: the first
        # structure that shows the surface is the one that made it knowable. Failure to
        # resolve a date leaves it None and the caller marks the claim uncitable, which keeps
        # a measured surface visible and unusable rather than dropping it.
        try:
            from .structures_evidence import dates

            stamps = dates([m["pdb_id"] for m in samples])
            markers["deposition_dates"] = {pdb: stamps[pdb] for pdb in sorted(stamps)}
            markers["available_from"] = min(stamps.values()) if stamps else None
        except Exception as failure:                      # noqa: BLE001
            markers["deposition_dates"] = {}
            markers["available_from"] = None
            markers["dating_error"] = f"{type(failure).__name__}: {str(failure)[:160]}"
        cache[key] = markers
    except ImportError:
        # Not cached and not converted. An absent backend is a property of the machine, and
        # caching it would make the first call's environment the answer for the whole process.
        raise
    except Exception as error:                            # noqa: BLE001
        # Reported as absent rather than raised. A target with no usable structure is the
        # ordinary case for a novel pathogen, and the ranker has to keep working without one.
        cache[key] = {"error": f"{type(error).__name__}: {error}", "surface_score": None}
    return cache[key]
