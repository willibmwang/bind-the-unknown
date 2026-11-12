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
import math
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


def surface_markers(pdb_id, chain=None, keep=None, verify=None, anchors=None):
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

    `anchors` is a list of author residue numbers -- an active site, a ligand-binding site, a
    catalytic dyad -- and asks for the patch arithmetic *at that location* instead of wherever
    the chain's best patch lies. It answers a question `keep` cannot: `keep` narrows the search
    to a domain, and a domain is still hundreds of residues of surface, while a macrocycle
    covers one 13.8-angstrom disc of it. Measured on 6LU7 chain A on 2026-11-12, the main
    protease's catalytic dyad has 93 of the chain's 306 residues within reach, so naming the
    anchors discards two thirds of the surface the unanchored call reports on. The
    neighbourhood, its hydrophobic count and the share of the site the patch covers come back
    under `site` and the `site_*` keys.

    **The anchors must be the deposited chain's author numbering.** Nothing here converts them,
    and that is deliberate rather than unfinished: once inside, a converted number and a given
    number are indistinguishable, so a silent conversion would make every site figure
    unauditable. The hazard is real and was measured, not imagined -- UniProt numbers the main
    protease's catalytic cysteine 3,408 on P0DTD1's replicase polyprotein, where nsp5 begins
    at 3,264, and every Mpro deposition calls the same residue 145. Asking
    `surface_for_accession` for the nsp5 interval with anchors 3304 and 3408 on 2026-11-12
    returned both of them in `anchors_missing` and a neighbourhood of nothing, where 41 and 145
    returned 94 residues. An anchor that does not exist on the chain is reported there rather
    than dropped or snapped to a neighbour, which is what makes that mistake visible instead of
    quietly producing a confident zero.

    That is also why `site_patch_measured` travels with the site figures. A site patch of 0.0
    is the right answer often enough to be the common case -- the main protease's catalytic
    dyad is a His/Cys pair and a His/Cys pair is not a hydrophobic patch -- so the figures are
    None rather than zero when no anchor was found, and `site_patch_absent_because` says so.
    `_markers` carries the measurement behind that.
    """
    residues = _residues(fetch_structure(pdb_id), chain)
    if not residues:
        return {"pdb_id": pdb_id, "chain": chain, "residues": 0, "error": "no residues read"}
    return _markers(residues, pdb_id, chain, keep, verify, anchors)


def markers_from_coordinates(text, label="predicted coordinates", anchors=None):
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

    `anchors` is offered, and the same fact about the numbering is why. A clip is meaningless
    here because the coordinates are already only the target; an anchor is not, because the
    chain is numbered from one and that is exactly the numbering a functional site mapped into
    the pathogen's own protein arrives in. This is also the only lane in which the site figures
    are reachable for a genuinely novel pathogen: no deposition covers the chain, so either the
    patch at the active site is measured on a prediction or it is not measured at all. It
    inherits every caveat above -- `on_predicted_coordinates` travels with it, and a site
    measured over a low-confidence stretch describes a fold the model never committed to. It
    also carries `site_patch_measured`, and on this lane that key is doing more work than on a
    deposition: a predicted chain is numbered from one, so an anchor mapped into the pathogen's
    own numbering should land, and a `False` here says the mapping is wrong rather than that
    the site has no patch.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as scratch:
        path = pathlib.Path(scratch) / "predicted.pdb"
        path.write_text(text)
        # freesasa raises a bare `Exception` for a file it cannot read, and the fold stage will
        # hand over coordinates that reach that state: `_parse_pdb` skips an ATOM line whose
        # columns will not parse and still returns the text with `folded` true, so a damaged
        # coordinate column arrives here inside an accepted prediction. Unguarded, that killed
        # the whole run -- one bad line in one chain of twelve aborting a fifteen-minute chain
        # after every other stage had already paid for its answers. A prediction that cannot be
        # read is a refusal like any other: recorded on the row, and the run continues.
        try:
            residues = _residues(path)
        except ImportError:
            # The one failure that is about this machine and not about the coordinates. Raised
            # on, because the caller distinguishes it: a missing SASA backend is not a fact
            # about the target and must not be recorded as one.
            raise
        except Exception as failure:                       # noqa: BLE001
            return {"pdb_id": label, "chain": None, "residues": 0,
                    "error": f"the predicted coordinates could not be read: "
                             f"{type(failure).__name__}: {str(failure)[:160]}",
                    "on_predicted_coordinates": True, "is_deposited_measurement": False}
        if not residues:
            return {"pdb_id": label, "chain": None, "residues": 0,
                    "error": "no residues read from the predicted coordinates",
                    "on_predicted_coordinates": True, "is_deposited_measurement": False}
        markers = _markers(residues, label, None, None, None, anchors)
    markers["on_predicted_coordinates"] = True
    markers["is_deposited_measurement"] = False
    return markers


def _markers(residues, pdb_id, chain=None, keep=None, verify=None, anchors=None):
    """The measurement itself, over residues already read from a file.

    Separated from `surface_markers` so a prediction and a deposition go through one body of
    arithmetic instead of two. The `chain` and `keep` semantics are documented on
    `surface_markers`, which is the registered tool; this function is the shared half.

    `anchors` is last in the signature because it has to be: `surface_markers` and
    `markers_from_coordinates` both call this positionally, so a parameter inserted anywhere
    earlier would hand `keep` a list of residue numbers and be read as a clip. It is a list of
    author residue numbers -- the residues doing the chemistry -- and when it is given the
    return carries a `site` block measuring the neighbourhood around them, plus the patch
    arithmetic at that location. The reach is `MACROCYCLE_REACH_A` and is not an argument here,
    because every caller of this function is asking about the same molecule; `residues_within`
    is the public door for a caller that wants the neighbourhood at another distance.

    Two things are now returned unconditionally that the arithmetic was already computing and
    throwing away. `patch_positions` and `patch_residue_names` are the largest patch's own
    residues: the component is the same one `hydrophobic_patch_residues` counts, read out
    instead of counted and discarded, so they cost no distance calculation and no second pass.
    A reader of 6LU7 chain A was previously told that the main protease carries a patch of 5
    residues covering 415.2 square angstroms and never which 5 -- measured 2026-11-12, they
    are Leu50, Phe134, Ala191, Ala193 and Ala194 -- and which 5 is the whole of what a binder
    designer can act on.

    **The site figures distinguish an empty site from an unmeasured one, and did not.** The
    site block is the patch intersected with the neighbourhood, so it comes back empty for two
    reasons that are opposite findings. Measured on the main protease with anchors 41 and 145
    on 2026-11-12, `site_patch_area_a2` is 0.0 on 7K3T:A, on 6YB7:A and on 6W79:A while their
    chain patches are 637.0, 651.0 and 343.4 square angstroms -- and all three zeros are the
    right answer, because the catalytic site is a His/Cys dyad and a histidine-cysteine pair
    is not a hydrophobic patch. An anchor handed over in the wrong numbering produces the same
    0.0 and means nothing about the protein: asking for the same site with the UniProt numbers
    3304 and 3408 puts both in `anchors_missing` and intersects the patch with an empty
    neighbourhood. `site_patch_measured` is therefore returned beside the figures and the
    figures are None rather than zero when it is False, with `site_patch_absent_because`
    carrying the sentence. A zero that a reader cannot tell from a failure is worse than no
    number, because only the second one can be seen to be missing.
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

    # The patch's own residue numbers, which this arithmetic already had in hand. `best` is
    # the largest component and only its length and its area were being reported; naming its
    # residues is pure gain from work already done, not a new measurement.
    #
    # One entry per patch residue rather than per distinct number, so `len(patch_positions)`
    # is `hydrophobic_patch_residues` by construction and the two can never disagree. Across
    # several chains the same author number can therefore appear twice; that is two residues
    # on two chains and not a duplicate, and `chain` is how a caller confines it.
    in_order = sorted(best, key=lambda r: (_resnum(r["number"]), r["chain"]))
    patch_positions = [_resnum(r["number"]) for r in in_order]
    patch_residue_names = [_letter(r["residue"]) for r in in_order]

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

    # The same patch, asked about at a location the caller names rather than wherever on the
    # chain the largest one happens to lie. The neighbourhood is drawn from every residue read
    # and not from the `keep`-clipped domain, which is the rule the sequon scan above already
    # follows: anything that can reach the site is looked for over the whole file, and only
    # membership of the patch is confined. A residue of the neighbouring domain eleven
    # angstroms from the catalytic cysteine is in contact with a macrocycle bound there whether
    # or not the clip kept it.
    # Named `site_summary` rather than `site`, which the sequon loop above is already using
    # for one sequon's coordinates. The two never overlap in time, and relying on that is how
    # a later edit that moves either block produces a wrong number instead of an error.
    site_summary = None
    if anchors:
        site_summary, near, _seats = _within_reach(residues, anchors, MACROCYCLE_REACH_A)
        # The existing patch intersected with that neighbourhood, and explicitly **not** a
        # fresh patch search over the neighbourhood alone. The argument is the one `keep`'s
        # docstring makes about accessibility: a region measured by itself exposes faces that
        # were never solvent-facing, so re-growing components inside a 13.8-angstrom ball
        # would report a patch assembled from residues the whole structure keeps buried.
        # Identity is the intersection key because `best` and `near` hold the same dicts
        # `_residues` returned, so no numbering question arises at all here.
        reachable = {id(r) for r in near}
        shared = [r for r in in_order if id(r) in reachable]
        # Floats rather than whatever `sum` returns, because `sum` of an empty list is the
        # integer 0 and `round(0, 1)` is the integer 0. The site figures are frequently empty
        # on exactly the targets this exists for -- the main protease's catalytic dyad is one
        # -- so the common case was the one emitting an int where every neighbouring key is a
        # float, and a reader comparing 0 against 0.0 across records sees a change that is not
        # one.
        site_patch_area = float(sum(r["sasa"] for r in shared))
        site_sasa = float(sum(r["sasa"] for r in near))
        # Whether the site was measured at all, which is NOT the same question as whether the
        # patch reaches it. See the comment on `site_patch_measured` below: this one boolean is
        # what separates a biological zero from an absent measurement.
        site_measured = bool(site_summary["anchors_found"])
        site_absent = None if site_measured else (
            f"none of the {len(site_summary['anchors_missing'])} anchors asked for exists on "
            f"this chain's author numbering -- "
            f"{', '.join(str(a) for a in site_summary['anchors_missing'])} are all in "
            f"`anchors_missing` -- so there is no neighbourhood for the patch to intersect. "
            f"The site figures are absent rather than zero: a zero here would read as a "
            f"finding about the site when it is a fact about the numbering.")

    markers = {
        "pdb_id": pdb_id, "chain": chain,
        "residues": len(residues), "residues_in_range": len(within),
        "keep": list(keep) if keep else None,
        "numbering_agreement": agreement, "numbering_offset": offset,
        "exposed_residues": len(exposed),
        "exposed_fraction": round(len(exposed) / len(within), 3),
        "total_sasa": round(total_sasa, 1),
        "hydrophobic_patch_area": round(patch_area, 1),
        "hydrophobic_patch_residues": len(best),
        "patch_positions": patch_positions,
        "patch_residue_names": patch_residue_names,
        "patch_fraction_of_surface": round(patch_area / total_sasa, 3) if total_sasa else 0.0,
        "patches_found": len(patches),
        "sequons_near_patch": sequons,
        "sequence_length": len(sequence),
    }
    if site_summary is not None:
        markers.update({
            "site": site_summary,
            # **A measured zero and an unmeasured zero are opposite findings, and until this
            # key existed they were the same four characters.** The site figures below are an
            # intersection, so they come out empty for two unrelated reasons. The first is
            # biology and is the common case on the headline target: the main protease's
            # catalytic site is a histidine-cysteine dyad, a His/Cys pair is not hydrophobic,
            # and a hydrophobic patch therefore has no business being there -- measured
            # 2026-11-12 with anchors 41 and 145, the site patch is 0.0 square angstroms on
            # 7K3T:A beside a chain patch of 637.0, on 6YB7:A beside 651.0 and on 6W79:A
            # beside 343.4. All three are correct answers about the protein. The second is a
            # failed lookup: an anchor given in the wrong numbering, or a residue disordered
            # out of the deposition, leaves `anchors_found` empty, and the same intersection
            # of a patch with nothing is then 0.0 as well while saying nothing whatever about
            # the site. Reported as a number, the two are indistinguishable, and the wrong one
            # reads as "the active site has no patch" when the truth is "the active site was
            # never located".
            #
            # So the figures are present only when `anchors_found` is non-empty, and None
            # otherwise with `site_patch_absent_because` carrying the sentence. A consumer
            # reading 0.0 is then reading a measurement, and a consumer reading None is being
            # told the measurement did not happen.
            "site_patch_measured": site_measured,
            "site_patch_absent_because": site_absent,
            "site_patch_area_a2": round(site_patch_area, 1) if site_measured else None,
            "site_patch_positions": ([_resnum(r["number"]) for r in shared]
                                     if site_measured else None),
            # Against the neighbourhood's own surface, matching `patch_fraction_of_surface`
            # one key above: how much of what a peptide at this site can touch is patch. The
            # count fraction was the other candidate and is the worse one, because a patch is
            # bound by area and four large aromatics are a site where eight small residues
            # are not.
            "site_patch_fraction_of_the_site": (
                (round(site_patch_area / site_sasa, 3) if site_sasa else 0.0)
                if site_measured else None),
            # Whether the chain's best patch is at the site at all. The informative answer is
            # frequently False, and that is the finding rather than a failure: it says the
            # surface the markers scored is somewhere other than the chemistry, which is
            # exactly what a reader pointing a macrocycle at an active site needs told. None,
            # not False, when the site was never located, for the reason above.
            "site_shares_the_patch": bool(shared) if site_measured else None,
        })
    return markers


# A macrocyclic peptide buries roughly this much on binding. A patch far below it cannot host
# one; a patch far above it is a large surface with no single site, which the fraction catches.
MACROCYCLE_FOOTPRINT_A2 = 600.0

# How far from one residue a macrocycle bound there can still reach. Derived from the footprint
# above and not stated beside it: a circular patch of 600 square angstroms has a radius of
# sqrt(600 / pi) = 13.8 angstroms, so the span of the molecule is not a second free parameter
# -- it is the radius the footprint the scoring already assumes implies. Typing 13.8 here would
# leave two numbers describing one molecule, free to disagree the moment either moves: raise
# the footprint to 800 and a hand-typed reach would still be answering for 600.
#
# It inherits the footprint's roughness, which is "roughly 400 to 800" in the module docstring
# above, so the reach is good to a couple of angstroms and not to a decimal place. That is
# enough for the question it is asked -- which residues a peptide at this site could contact --
# and it is why no narrower figure is quoted anywhere it is reported.
MACROCYCLE_REACH_A = math.sqrt(MACROCYCLE_FOOTPRINT_A2 / math.pi)


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


def _within_reach(residues, anchors, reach):
    """One neighbourhood, computed once, returned in the three shapes its callers need.

    Returns `(summary, neighbourhood, anchor_records)`. `residues_within` wants the summary
    alone; `_markers` wants the records too, because it already holds a patch and the site
    figures are that patch intersected with this neighbourhood. Split this way the distance
    pass runs once per measurement rather than once per consumer, and the two callers report
    one neighbourhood instead of two that could drift apart -- the same reason `_markers`
    itself was split out of `surface_markers`.

    **The distance is alpha-carbon to alpha-carbon, said plainly rather than dressed up.**
    `_residues` keeps one atom per residue -- it skips every line whose atom name is not `CA`
    -- so no side-chain coordinate exists here to measure from, and calling this atom-to-atom
    would claim a precision the data does not carry. The error is one-sided and worth stating
    where it is reported: an extended lysine's terminal nitrogen sits about 6.4 angstroms out
    from its own alpha carbon, so two long side chains can touch across a gap this counts as
    out of range. The neighbourhood is therefore a floor on what a macrocycle at the site can
    contact and never a ceiling. Reading every atom would widen it, and would also move the
    patch arithmetic every other number in this module is computed from, which is not a change
    to make inside a feature about positions.

    Insertion codes are dropped on both sides through `_resnum`, so an anchor given as 52
    matches a residue deposited as 52A. That is the normalisation the `keep` clip already uses,
    and the alternative -- matching the author number as a string -- misses the residue the
    caller meant on any entry carrying an insertion, silently.

    **An anchor is a number and not a chain-and-number pair, which matters on a complex.**
    Every residue carrying the number is taken, so an unrestricted read of two polymers matches
    it on both. Measured on 6M0J on 2026-11-12: ACE2 carries His417, Thr453 and Ala501 and the
    spike receptor-binding domain carries Lys417, Tyr453 and Asn501, so asking for the
    receptor-binding motif over the whole deposition took all six and returned one
    "neighbourhood" of 268 residues that is really two sites, 182 residues around ACE2's own
    numbers and 86 around the motif. The remedy is the caller's and it already exists -- pass
    `chain` to `surface_markers`, which is what `surface_for_accession` does on every pick --
    and `anchor_chains` in the return names the polymers so a merged measurement is visible
    rather than inferred from a count that merely looks large.
    """
    wanted = sorted({_resnum(number) for number in (anchors or ())})
    asked = set(wanted)
    seats = [r for r in residues if _resnum(r["number"]) in asked]
    present = {_resnum(r["number"]) for r in seats}
    found = sorted(present)
    missing = [number for number in wanted if number not in present]

    # An anchor is at zero distance from itself, so the anchors are in the neighbourhood by
    # construction rather than by being appended afterwards.
    near = []
    for residue in residues:
        for seat in seats:
            dx = residue["xyz"][0] - seat["xyz"][0]
            dy = residue["xyz"][1] - seat["xyz"][1]
            dz = residue["xyz"][2] - seat["xyz"][2]
            if dx * dx + dy * dy + dz * dz <= reach * reach:
                near.append(residue)
                break

    summary = {
        "anchors_found": found,
        "anchors_missing": missing,
        "positions": sorted(_resnum(r["number"]) for r in near),
        "residues": len(near),
        "hydrophobic": sum(1 for r in near if _letter(r["residue"]) in HYDROPHOBIC),
        "exposed": sum(1 for r in near if r["rsa"] >= EXPOSED_RSA),
        "reach_a": reach,
        # Which polymers the anchors were matched on, so a merged neighbourhood is visible in
        # the record instead of being inferred from a count that looks high. More than one
        # chain here means the anchor numbers exist on more than one chain and the figures
        # describe both sites at once; see the docstring above for the measured case.
        "anchor_chains": sorted({r["chain"] for r in seats}),
    }
    return summary, near, seats


def residues_within(residues, anchors, reach=MACROCYCLE_REACH_A):
    """Residues whose alpha carbon lies within `reach` of any anchor residue's alpha carbon.

    This is the question a binder designer asks of a functional site and the pipeline could
    not answer: not "is this protein amenable" but "if a macrocycle sits on these residues,
    what else does it touch". The surface markers above report the largest patch *anywhere* on
    the chain, which on a construct carrying several domains is frequently nowhere near the
    chemistry -- SARS-CoV-2's nsp3 is 1,945 residues and its best patch belongs to whichever
    of its domains happens to carry one. Naming the anchors moves the question to the site.

    `anchors` is a list of author residue numbers, as deposited; `residues` is what `_residues`
    read, so the neighbourhood spans every chain in that read unless the caller restricted it.
    A partner chain's residues inside the reach are real contacts and are counted, which is
    what makes this readable on an interface: the ACE2 residues a peptide at the spike's
    receptor-binding motif would meet are the reason to bind there. The cost of the same
    freedom is that an anchor number existing on two chains matches on both -- `_within_reach`
    documents the measured case and `anchor_chains` reports it.

    Returns
      {"anchors_found": [...], "anchors_missing": [...],
       "positions": [...],            # author numbers, sorted, anchors included
       "residues": int, "hydrophobic": int, "exposed": int,
       "reach_a": reach, "anchor_chains": [...]}

    `positions` carries one entry per residue rather than per distinct number, so `residues`
    is `len(positions)` and the two cannot disagree; across several chains the same author
    number can appear twice, which is two residues and not a duplicate.

    `hydrophobic` is the count a macrocycle cares about -- the stage-4 patch arithmetic asked
    at this location -- and `exposed` says how much of the neighbourhood is surface at all,
    since a site whose neighbours are buried offers less to lie against than the count
    suggests.

    An anchor absent from the structure is reported in `anchors_missing` rather than dropped.
    That case is ordinary and not an error: a catalytic residue can be disordered in the
    deposition, and author numbering frequently is not the numbering the annotation arrived
    in, so an empty `anchors_found` beside a populated `anchors_missing` is the pipeline saying
    which of the two happened instead of returning a confident neighbourhood of nothing.

    What the numbers look like, measured on 6LU7 chain A on 2026-11-12, with the main
    protease's catalytic dyad -- His41 and Cys145 -- as the anchors: 93 of the chain's 306
    residues are within reach, 41 of them hydrophobic and 33 of them exposed. The chain's
    largest patch, 415.2 square angstroms over five residues, meets that neighbourhood at
    exactly one of them, Leu50, so `_markers` reports `site_shares_the_patch` true and
    `site_patch_fraction_of_the_site` 0.037. Those two keys read together are the point: the
    patch the surface score was earned on is almost entirely somewhere other than the
    chemistry, and a reader aiming a macrocycle at the active site has to be told that rather
    than handed the chain's best number.

    The distance is alpha carbon to alpha carbon, not atom to atom; `_within_reach` carries
    the full statement of what that costs and why the coordinates allow nothing better.
    """
    return _within_reach(residues, anchors, reach)[0]


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


def surface_for_accession(accession, interval=None, replicates=5, cache={}, anchors=None):
    """Surface markers for a UniProt accession via its best deposited structure.

    Cached in process. A genome's candidates frequently share a homolog -- every mature chain
    of a polyprotein resolves to the same entry -- and the SASA calculation is the expensive
    step in the ranking.

    `anchors` is passed to each structure unchanged and carries the numbering warning on
    `surface_markers` with it, which bites hardest here: this function chooses the structures
    itself, so a caller cannot know whose author numbering it is handing numbers in. Only
    structures that actually found the anchors contribute, because a chain deposited in a
    different numbering reports every anchor missing and its empty site block would otherwise
    read as "nothing is near the site".

    **The site figures are aggregated over the replicates, exactly as the patch is.** They
    were not, and that was the defect: `site_patch_area_a2`, `site_patch_positions`,
    `site_patch_fraction_of_the_site` and `site_shares_the_patch` were lifted whole out of the
    first structure that found the anchors while `hydrophobic_patch_area` beside them was a
    median with `patch_area_min` and `patch_area_max` recorded. The spread those keys exist for
    is the reason -- five structures of the main protease span 625.1 to 824.6 square angstroms,
    199.5 of spread, and five of the spike receptor-binding domain span 496 -- and the site
    figures are the same patch intersected with a neighbourhood, so they inherit the same
    ensemble error. Measured 2026-11-12 with anchors 41 and 145, the dyad's site patch is 0.0
    on 7K3T:A, 7GEF:A, 6YB7:A and 6W79:A and 121.0 on 6LU7:A: one structure in five disagrees
    with the other four, and promoting one of them answered a question about the ensemble with
    a sample of size one.

    So the area and the fraction are medians over the structures that measured the site, with
    `site_patch_area_min`, `site_patch_area_max` and `site_patch_area_spread` beside them, and
    `site_shares_the_patch` is the majority -- which for a sample of booleans is the same
    median rule. `site_structures_measured`, `site_structures_offered`,
    `site_structures_sharing_the_patch` and `site_structures_used` say what the aggregate was
    taken over. `site` and `site_patch_positions` stay one structure's, named by
    `site_patch_positions_measured_on`, because they are author residue numbers and a median
    over position lists is not a position list -- the same reservation
    `patch_positions_measured_on` already carries.

    When no structure found the anchors every site figure is None, not zero, and
    `site_patch_absent_because` says so. A site patch of 0.0 is a real and frequent answer --
    a His/Cys catalytic dyad is not a hydrophobic patch -- and it has to be readable apart from
    a site that was never located.
    """
    # `replicates` belongs in the key and was missing from it. The median is taken over up to
    # `replicates` structures, so it is an input to the answer and not a performance knob: with
    # it absent from the key, a call with `replicates=1` and a later call with `replicates=5`
    # returned the same cached number, and the second caller's argument was silently ignored.
    # Registering this function as a tool is what made that reachable -- `replicates` is now a
    # declared parameter a policy can set, so a key that drops it would let the schema promise
    # a control the code does not offer, which is the defect `pick_batch` already had once with
    # `forest_trees` and `ucb_beta`.
    #
    # `anchors` joins the key for the same reason and before it can repeat the fault: one call
    # asking for the patch at the catalytic dyad and a later call asking for the chain's best
    # patch are different questions about the same accession, and sharing one cached answer
    # would hand the second caller the first caller's site block. Normalised through `_resnum`
    # and sorted so the key matches on what the measurement matches on -- two callers naming
    # the same residues in a different order are asking one question.
    key = (accession, tuple(interval.values()) if interval else None, replicates,
           tuple(sorted({_resnum(a) for a in anchors})) if anchors else None)
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
                                           keep=pick.get("keep"), verify=verify,
                                           anchors=anchors)
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
        # Which structure the residue numbers belong to. `patch_positions` is the first
        # sample's and `hydrophobic_patch_residues` just above is the median over all of them,
        # so the list and the count will not always be the same length -- deliberately, because
        # author numbering differs between depositions and a median over position lists is not
        # a position list. Named rather than dropped: the positions are the one output a binder
        # designer acts on, and the fix for an ambiguous provenance is to state it.
        markers["patch_positions_measured_on"] = (f"{samples[0]['pdb_id']}:"
                                                  f"{samples[0]['chain']}")
        if anchors:
            # **Aggregated over the replicates, not promoted from one of them.** Until this
            # block was rewritten the three site figures were copied out of whichever
            # structure found the anchors first, while every quantity around them -- the patch
            # area, its fraction of the surface, the exposed fraction, the surface score -- was
            # a median over up to `replicates` structures with a recorded minimum and maximum.
            # That asymmetry is not a tidiness complaint: the reason the median and the spread
            # exist is that one crystal is one sample of an ensemble and the quantity moves
            # with it, measured on 2026-11-12 at 199.5 square angstroms of spread across five
            # structures of the main protease -- 625.1 to 824.6 -- and at 496 across five of
            # the spike receptor-binding domain. The site figures are the same kind of
            # quantity, computed by intersecting the same patch with a neighbourhood, so a
            # single structure's value carries the same ensemble error with none of the warning
            # labels. The site patch of the main protease's dyad is 0.0 on 7K3T:A, 7GEF:A,
            # 6YB7:A and 6W79:A and 121.0 on 6LU7:A, which is exactly the spread the promotion
            # was hiding.
            #
            # Only the structures that actually found the anchors are aggregated. A deposition
            # numbered in another convention reports every anchor missing, and averaging its
            # empty intersection in with real measurements would pull the median towards zero
            # for a reason that has nothing to do with the protein.
            at_site = [m for m in samples if m.get("site_patch_measured")]
            markers["site_structures_measured"] = len(at_site)
            markers["site_structures_offered"] = len(samples)
            if at_site:
                # The neighbourhood block and the patch positions stay one structure's and are
                # named, for the reason `patch_positions_measured_on` gives one key above:
                # these are author residue numbers, author numbering differs between
                # depositions, and a median over position lists is not a position list.
                chosen = at_site[0]
                markers["site"] = chosen["site"]
                markers["site_patch_positions"] = chosen["site_patch_positions"]
                markers["site_patch_positions_measured_on"] = (f"{chosen['pdb_id']}:"
                                                               f"{chosen['chain']}")
                site_areas = [m["site_patch_area_a2"] for m in at_site]
                markers["site_patch_measured"] = True
                markers["site_patch_absent_because"] = None
                markers["site_patch_area_a2"] = round(median(site_areas), 1)
                markers["site_patch_area_min"] = round(min(site_areas), 1)
                markers["site_patch_area_max"] = round(max(site_areas), 1)
                markers["site_patch_area_spread"] = round(max(site_areas) - min(site_areas), 1)
                markers["site_patch_fraction_of_the_site"] = round(
                    median([m["site_patch_fraction_of_the_site"] for m in at_site]), 3)
                # A majority of the structures measured, which for a sample of zeros and ones
                # IS the median -- so the boolean is aggregated by the same rule as every
                # number beside it rather than by a second rule invented for booleans. A tie
                # resolves to False, the conservative direction: half the depositions showing
                # the chain's best patch somewhere else is not a basis for telling a reader
                # the patch is at the chemistry. The raw vote stays readable in
                # `site_structures_sharing_the_patch` next to `site_structures_measured`.
                sharing = sum(1 for m in at_site if m["site_shares_the_patch"])
                markers["site_structures_sharing_the_patch"] = sharing
                markers["site_shares_the_patch"] = sharing * 2 > len(at_site)
                markers["site_structures_used"] = [f"{m['pdb_id']}:{m['chain']}"
                                                   for m in at_site]
                markers["site_measured_on"] = f"{chosen['pdb_id']}:{chosen['chain']}"
            else:
                # None of them found the anchors, which is the honest version of that answer
                # and not a measurement of an empty site -- see `_markers`. The first sample's
                # block stands so `anchors_missing` and the reach are still readable, every
                # figure is None rather than zero, and `site_patch_absent_because` says which
                # of the two happened.
                stood = samples[0]
                looked_in = ", ".join(f"{m['pdb_id']}:{m['chain']}" for m in samples)
                markers["site"] = stood.get("site")
                markers["site_patch_measured"] = False
                markers["site_patch_absent_because"] = (
                    f"{stood.get('site_patch_absent_because') or 'the anchors were not found'}"
                    f" This held on all {len(samples)} structures measured ({looked_in}), so "
                    f"the site figures are absent for every one of them.")
                # Written out rather than left to `.get` returning None, so the record carries
                # the same keys whether or not the site was found. A consumer that has to tell
                # an absent key from a null one is reading the shape of the dict instead of
                # what it says, and the page's per-field joins do exactly that.
                for field in ("site_patch_area_a2", "site_patch_positions",
                              "site_patch_fraction_of_the_site", "site_shares_the_patch",
                              "site_patch_area_min", "site_patch_area_max",
                              "site_patch_area_spread", "site_patch_positions_measured_on",
                              "site_structures_sharing_the_patch"):
                    markers[field] = None
                markers["site_structures_used"] = []
                markers["site_measured_on"] = f"{stood['pdb_id']}:{stood['chain']}"
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
