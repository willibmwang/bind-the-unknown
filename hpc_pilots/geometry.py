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


def _letter(three):
    table = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
             "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
             "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
             "TYR": "Y", "VAL": "V"}
    return table.get(three.strip().upper(), "X")


def surface_markers(pdb_id, chain=None):
    """Interface-shaped markers for one structure.

    `chain` restricts the calculation to one polymer, which matters for a complex: the spike
    receptor-binding domain deposited with ACE2 has its best patch buried by the partner, and
    measuring the pair would report the interface as inaccessible when it is the target.
    """
    residues = _residues(fetch_structure(pdb_id), chain)
    if not residues:
        return {"pdb_id": pdb_id, "chain": chain, "residues": 0, "error": "no residues read"}

    exposed = [r for r in residues if r["rsa"] >= EXPOSED_RSA]
    total_sasa = sum(r["sasa"] for r in residues)
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
    ordered = sorted(residues, key=lambda r: (r["chain"], int("".join(
        c for c in r["number"] if c.isdigit() or c == "-") or 0)))
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
        "residues": len(residues), "exposed_residues": len(exposed),
        "exposed_fraction": round(len(exposed) / len(residues), 3),
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
    """The best-resolution deposited chain for one UniProt accession, or None.

    Picks on resolution rather than on coverage, because the markers are computed on whatever
    is present and a sharper structure gives better side-chain placement, which is what the
    accessibility calculation depends on. Chains shorter than `minimum_residues` are skipped:
    a peptide fragment co-crystallised with something else has a surface, but not the target's.
    """
    from .structures_evidence import chains

    from .structures_evidence import covers

    ranked = []
    for row in chains(accession):
        resolution = row.get("resolution")
        if resolution is None:
            continue
        length = (row.get("unp_end") or 0) - (row.get("unp_start") or 0)
        if length < minimum_residues:
            continue
        # Restricted to structures actually covering the domain, when one is named. Without
        # this every mature chain of a polyprotein resolves to the same parent accession and
        # therefore to the same structure: the main protease, the polymerase, the helicase and
        # the papain-like protease of SARS-CoV-2 all received one surface score of 2.85 and one
        # patch of 381 square angstroms, which is the replicase's and none of theirs.
        if interval and not covers(row, interval, 0.5):
            continue
        ranked.append((resolution, row["pdb_id"], row["chain_id"]))
    if not ranked:
        return None
    ranked.sort()

    # Returned as a list rather than a single pick, because the best-resolution entry is
    # often the newest and the newest are frequently mmCIF-only: RCSB stopped emitting legacy
    # PDB files for recent depositions, so the 1.16-angstrom main-protease structure this
    # chose first answers 404. Picking on resolution selects for exactly the entries most
    # likely to be missing the format, which is why one pick is not enough.
    return [{"pdb_id": pdb, "chain": chain, "resolution": resolution}
            for resolution, pdb, chain in ranked[:8]]


def surface_for_accession(accession, interval=None, replicates=5, cache={}):
    """Surface markers for a UniProt accession via its best deposited structure.

    Cached in process. A genome's candidates frequently share a homolog -- every mature chain
    of a polyprotein resolves to the same entry -- and the SASA calculation is the expensive
    step in the ranking.
    """
    key = (accession, tuple(interval.values()) if interval else None)
    if key in cache:
        return cache[key]
    try:
        picks = representative_structure(accession, interval=interval)
        if not picks:
            cache[key] = None
            return None
        # Measured over several structures of the same domain, not one. A single structure is
        # one sample of a conformational ensemble and the quantity being thresholded moves
        # with it: SARS-CoV-2's main protease measures 343 square angstroms through one
        # homolog's entry and 825 through another, a factor of 2.4. Reporting one number from
        # one crystal as though it were the protein is what made the amenability gate
        # unusable, and a median over a handful with the spread beside it is what makes it
        # readable -- a target whose patch is 600 give or take 40 is a different claim from
        # one that is 600 give or take 300.
        samples = []
        for pick in picks:
            try:
                measured = surface_markers(pick["pdb_id"], pick["chain"])
            except Exception:                             # noqa: BLE001
                continue
            if measured.get("error"):
                continue
            measured["resolution"] = pick["resolution"]
            measured["surface_score"] = macrocycle_surface_score(measured)
            samples.append(measured)
            if len(samples) >= replicates:
                break
        if not samples:
            cache[key] = {"error": "no candidate structure could be downloaded or parsed",
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
        markers["hydrophobic_patch_area"] = round(median(areas), 1)
        markers["patch_area_min"] = round(min(areas), 1)
        markers["patch_area_max"] = round(max(areas), 1)
        markers["patch_area_spread"] = round(max(areas) - min(areas), 1)
        markers["structures_measured"] = len(samples)
        markers["structures_used"] = [f"{m['pdb_id']}:{m['chain']}" for m in samples]
        markers["surface_score"] = round(median(scores), 4) if scores else None
        cache[key] = markers
    except Exception as error:                            # noqa: BLE001
        # Reported as absent rather than raised. A target with no usable structure is the
        # ordinary case for a novel pathogen, and the ranker has to keep working without one.
        cache[key] = {"error": f"{type(error).__name__}: {error}", "surface_score": None}
    return cache[key]
