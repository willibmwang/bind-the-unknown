"""Prior binders, assay formats and public mutational landscapes, per protein accession.

Three axes in `dossier.AXES` have no backend anywhere in this package. `precedent` asks
whether a macrocycle or peptide campaign has been run against a target; `assay_availability`
asks whether a functional orthogonal assay exists beyond selection enrichment;
`public_labels` asks whether a public DMS or binding landscape exists for method development.
Today all three are answered by hand-authored strings in `targets.py` citing the project's own
unreviewed write-up. This module answers them from three public keyless endpoints, and -- more
importantly -- states for each what the return cannot establish.

**Three lanes, one accession.**

*ChEMBL* (`/target`, `/assay`, `/activity`, `/molecule`, `/document`) is the bioactivity
record. The accession resolves to one or more `target_chembl_id`s by component accession,
which is a join on an identifier rather than a title match, so the same objection
`structures_evidence` raises against searching structure titles does not apply. From there the
assay table carries `assay_type`, `bao_label` and `confidence_score`, and that triple -- not
the activity count -- is the `assay_availability` signal: a `confidence_score` of 9 on a
"single protein format" biochemical assay is a different object from the same score on an
"organism-based format" phenotypic screen, and only the first is a candidate orthogonal
readout for a selection campaign.

*IEDB* (`query-api.iedb.org`, a PostgREST surface) is the mapped-epitope record. It is queried
by `UNIPROT:<accession>` against `parent_source_antigen_iri`, which is again an identifier
join. Two things come out of it. Its `assay_names` column enumerates the assay formats that
have actually been run on this antigen -- 129 distinct ones for SARS-CoV-2 spike, including
ELISA, surface-plasmon and crystallographic formats -- which is a far better
`assay_availability` signal than any count. And its B-cell epitope count bears on
`accessibility`, with the caveat below.

*MaveDB* (`api.mavedb.org`) is the multiplexed-assay-of-variant-effect record, and the only
keyless public index of deep mutational scanning datasets. Its
`/statistics/target/gene/uniprot-identifier` endpoint returns the complete UniProt-keyed map
in one 7 kB call, so presence and absence are both exact.

**Four things this module refuses to let a reader conclude.**

*A bioactivity row is not a campaign.* ChEMBL records measurements extracted from
publications. One `IC50` against a target is a measurement; a macrocycle campaign is a
selection, a hit list, a resynthesis and an orthogonal confirmation. The payload therefore
carries `peptide_or_macrocycle_campaign_established` pinned to False with the reason, and the
counts are named `..._bioactivity_rows`, never `..._campaigns`.

*An assay in a database is not an assay we could run.* A `bao_label` tells you the format a
published assay used. It does not say the construct is obtainable, the reagent is available,
the protocol is published, or the readout transfers to this pathogen's orthologue. So
`orthogonal_assay_candidates` is the field name, and
`runnable_assay_established` is pinned to False.

*ChEMBL cannot tell a peptide from a small molecule, and carries no macrocycle flag.* This was
measured, not assumed: `molecule_type` admits exactly nine values and "Peptide" is not among
them. Cyclosporine, a macrocyclic undecapeptide, is typed "Protein"; sirolimus, a macrolide
macrocycle, is typed "Small molecule". The one curated peptide signal is `helm_notation`,
which is non-null for sequence-defined entities (cyclosporine has one) and is reported as
`sequence_defined_peptides`. Beyond that the module computes its own classification from the
canonical SMILES -- largest smallest-ring through each ring bond, and amide-bond count over
the parsed molecular graph -- and every such field is prefixed `heuristic_` and carries
`unverified: True`. RDKit is not installed in this environment and a ring perception written
here is not a substitute for one.

*A UniProt-keyed MaveDB absence is not an absence of DMS data.* Measured on 2026-10-27:
MaveDB holds six SARS-CoV-2 receptor-binding-domain score sets (binding, escape and
expression for Omicron BA.1 and BA.2) and not one of them carries a UniProt identifier, so the
authoritative accession index reports zero for P0DTC2 while the data exists. The accession
lane is therefore reported as exact-but-sparse, and a second name-matched lane runs when a
target name is supplied, with every hit labelled `accession_confirmed` true or false and the
whole lane labelled `unverified`. Reporting only the accession lane would have produced a
confident false zero on the project's own worked example, which is the costliest error
available on this axis.

**Transport.** Every request goes through `urllib.request.urlopen` with a descriptive
User-Agent and a pause between calls. Not for politeness alone: the harness counts network
calls by patching `urlopen`, so a tool reaching the network by any other transport would
certify a network cost of zero. A lane whose host fails degrades to a recorded absence
carrying the exception string and the payload status becomes `partial`; it does not raise, and
it does not take the other two lanes with it. A malformed accession raises, because that is a
contract violation and not a fact about the world.
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque

from .common import ROOT, content_digest, provenance, write_json

CHEMBL = "https://www.ebi.ac.uk/chembl/api/data"
IEDB = "https://query-api.iedb.org"
MAVEDB = "https://api.mavedb.org/api/v1"

# Sent on every request. The structural tool sends the default Python urllib agent to
# EMBL-EBI thirty-six times per call with no contact string; an endpoint that cannot
# identify a misbehaving client cannot do anything about it except block the range.
AGENT = "hpc-pilots/1.0 (research prototype; contact via repository)"

# Between successive requests. Three different hosts, none of which publishes a rate limit
# for anonymous use, so the number is a courtesy and not a measured ceiling.
PAUSE = 0.34

# Per-request socket timeout. The only ceiling on this tool's wall time, so the declared
# `worst_case_seconds` in the registry is an arithmetic consequence of it and of the call
# caps below, not a guess: at most 17 requests (1 target + 3 x 2 assay/activity + 2 molecule
# chunks + 2 document chunks + 4 IEDB + 2 MaveDB), hence 17 * 30 + 16 * PAUSE ~= 516 s.
TIMEOUT = 30

# Caps, so the call count is bounded and the arithmetic above holds. Each is a prefix of a
# record that may be far larger, and the payload reports the backend's own total beside what
# was read so a reader can see the truncation rather than infer it.
MAX_CHEMBL_TARGETS = 3
MAX_ASSAY_ROWS = 200
MAX_ACTIVITY_ROWS = 500
MOLECULE_CHUNK = 25
MAX_DOCUMENTS = 100
DOCUMENT_CHUNK = 50

# The official UniProt accession pattern. Checked before any request, because a nucleotide
# accession is the one fact an episode starts with and the plausible wrong argument here:
# `NC_045512.2` would sail through all three endpoints and return three empty results that
# read exactly like a real absence.
ACCESSION = re.compile(
    r"\A(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})"
    r"(?:-[0-9]+)?\Z")

# ChEMBL's `molecule_type` vocabulary, recovered by probing each candidate value against
# `/molecule` on 2026-10-27 and recording the returned `total_count`. Kept here because the
# absence of "Peptide" is the finding, and a reader who assumes the obvious value exists
# writes a filter that returns zero rather than an error -- ChEMBL applies an unknown *value*
# and silently ignores an unknown *parameter*, which are two different traps.
CHEMBL_MOLECULE_TYPES = ("Small molecule", "Unknown", "Protein", "Antibody",
                         "Oligonucleotide", "Gene", "Enzyme", "Cell", "Oligosaccharide")

# A ring of twelve or more atoms is the conventional macrocycle threshold.
MACROCYCLE_RING = 12
# Four amide bonds is a tetrapeptide backbone's worth. Caffeine has three cyclic amides and
# is deliberately below the line; this is a screen and not a determination.
PEPTIDE_AMIDES = 4

# BAO formats in which a measurement is made on the purified target rather than on a cell or
# an organism. Only these are orthogonal-assay candidates for a selection campaign: a
# cell-based antiviral readout cannot confirm that a macrocycle binds the protein it was
# selected against.
BIOCHEMICAL_FORMATS = ("single protein format", "protein format", "protein complex format",
                       "assay format", "nucleic acid format", "cell-free format",
                       "biochemical format")

# Substrings in an IEDB `assay_names` entry that indicate a direct binding measurement on the
# antigen, as opposed to a cellular or functional immunology readout.
BINDING_ASSAY_HINTS = ("binding", "affinity", "dissociation", "surface plasmon",
                       "biolayer", "calorimetry", "kd", "ka ")
STRUCTURAL_ASSAY_HINTS = ("3d structure", "x-ray", "crystallograph", "electron microscopy",
                          "nmr", "small-angle")

_SMILES_TOKEN = re.compile(
    r"(\[[^\]]*\])|(Cl|Br|[BCNOPSFIbcnops])|(%\d\d|\d)|(\()|(\))|([=#$:/\\~-])|(\.)")
_SMILES_ELEMENT = re.compile(r"\A\[(?:\d+)?([A-Za-z][a-z]?)")


class _Trace:
    """Every URL this call hit, in order, with its status and size.

    A `sources` list assembled from module constants is a claim about which endpoints the
    code contains. This is a record of which ones the call reached, which is the only version
    a provenance field can be checked against.
    """

    def __init__(self):
        self.calls = []
        self._last = 0.0

    def get(self, url, headers=None, timeout=TIMEOUT):
        return self._send(urllib.request.Request(
            url, headers={"User-Agent": AGENT, **(headers or {})}), url, "GET", timeout)

    def post_json(self, url, body, timeout=TIMEOUT):
        request = urllib.request.Request(
            url, data=json.dumps(body).encode(), method="POST",
            headers={"User-Agent": AGENT, "Content-Type": "application/json",
                     "Accept": "application/json"})
        return self._send(request, url, "POST", timeout)

    def _send(self, request, url, method, timeout):
        wait = PAUSE - (time.monotonic() - self._last)
        if self.calls and wait > 0:
            time.sleep(wait)
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            raw = handle.read()
            status, headers = handle.status, dict(handle.headers)
        self._last = time.monotonic()
        self.calls.append({"method": method, "url": url, "status": status,
                           "bytes": len(raw)})
        return status, headers, raw

    def json(self, url, headers=None, timeout=TIMEOUT):
        _, _, raw = self.get(url, headers=headers, timeout=timeout)
        return json.loads(raw.decode())

    @property
    def count(self):
        return len(self.calls)

    @property
    def urls(self):
        return [c["url"] for c in self.calls]


def _query(base, **params):
    pairs = [(k, v) for k, v in params.items() if v is not None]
    return base + "?" + urllib.parse.urlencode(pairs, safe=":,{}.*")


def _absence(reason, **extra):
    """A zero that says it is a zero. Never a plausible value and never a silent empty dict."""
    out = {"status": "absent", "found": False, "reason": reason}
    out.update(extra)
    return out


def _unavailable(exc):
    """A backend that did not answer, charged and named rather than read as an absence."""
    return {"status": "unavailable", "found": None,
            "error": f"{type(exc).__name__}: {exc}",
            "reason": "The backend did not answer. This is not evidence that the record is "
                      "empty, and must not be read as one."}


# --------------------------------------------------------------------------- SMILES graph

def _parse_smiles(smiles):
    """Atoms, bonds and bond orders from a SMILES string.

    Enough of a parser to perceive rings and amides, and nothing more. No aromaticity model,
    no valence check, no stereochemistry: a bracket atom contributes its element symbol and
    its contents are otherwise discarded.
    """
    atoms, bonds, order = [], set(), {}
    stack, previous, pending, warnings = [], None, {}, []
    next_bond = None
    for match in _SMILES_TOKEN.finditer(smiles):
        bracket, organic, ring, opener, closer, bond, dot = match.groups()
        if bracket or organic:
            element = _SMILES_ELEMENT.match(bracket) if bracket else None
            atoms.append(element.group(1) if element else (bracket or organic))
            index = len(atoms) - 1
            if previous is not None and next_bond != ".":
                key = (min(previous, index), max(previous, index))
                bonds.add(key)
                order[key] = next_bond or "-"
            previous, next_bond = index, None
        elif opener:
            stack.append(previous)
        elif closer:
            previous = stack.pop() if stack else None
        elif ring:
            if previous is None:
                continue
            if ring in pending:
                left, right = pending.pop(ring), previous
                if left != right:
                    key = (min(left, right), max(left, right))
                    bonds.add(key)
                    order.setdefault(key, next_bond or "-")
            else:
                pending[ring] = previous
            next_bond = None
        elif bond:
            next_bond = bond
        elif dot:
            next_bond, previous = ".", None
    if pending:
        warnings.append("unclosed ring-bond numbers: " + ",".join(sorted(pending)))
    return atoms, bonds, order, warnings


def _adjacency(count, bonds):
    adjacency = [[] for _ in range(count)]
    for left, right in bonds:
        adjacency[left].append(right)
        adjacency[right].append(left)
    return adjacency


def _ring_bonds(count, bonds):
    """Bonds not in a spanning forest, i.e. one bond per independent cycle."""
    adjacency = _adjacency(count, bonds)
    seen, tree = set(), set()
    for start in range(count):
        if start in seen:
            continue
        seen.add(start)
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for neighbour in adjacency[node]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    tree.add((min(node, neighbour), max(node, neighbour)))
                    queue.append(neighbour)
    return sorted(bonds - tree)


def _smallest_ring_through(count, bonds, bond):
    """Size of the smallest cycle containing `bond`, by BFS with that bond removed.

    The naive alternative -- the distance between the two atoms of a ring-closure digit in the
    SMILES string -- was tried first and reported cholesterol as a seventeen-membered ring,
    because a fused polycyclic closes across a path whose shortcuts are other ring bonds.
    Removing one bond and shortest-pathing the rest gives 6 for cholesterol and 33 for
    cyclosporine, which is the distinction this field exists to make.
    """
    adjacency = _adjacency(count, bonds - {bond})
    source, sink = bond
    distance = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if neighbour not in distance:
                distance[neighbour] = distance[node] + 1
                if neighbour == sink:
                    return distance[neighbour] + 1
                queue.append(neighbour)
    return 0


def smiles_shape(smiles):
    """Ring sizes and amide-bond count for a SMILES string. A heuristic, labelled as one.

    Returned fields carry `unverified` because nothing here was checked against a cheminformatics
    toolkit at run time. What was checked, once, by hand: benzene 6, naphthalene 6, ibuprofen 6,
    cholesterol 6, caffeine 5, a cyclic octapeptide 12, an erythromycin macrolactone 14,
    cyclosporine 33 with eleven amide bonds, and a linear tripeptide no rings with two.
    """
    if not smiles:
        return {"largest_ring": None, "amide_bonds": None, "heavy_atoms": None,
                "unverified": True, "warnings": ["no canonical SMILES in the record"]}
    atoms, bonds, order, warnings = _parse_smiles(smiles)
    count = len(atoms)
    rings = [_smallest_ring_through(count, bonds, bond)
             for bond in _ring_bonds(count, bonds)]
    rings = [size for size in rings if size]
    carbonyl = set()
    for (left, right), kind in order.items():
        if kind == "=" and {atoms[left].upper(), atoms[right].upper()} == {"C", "O"}:
            carbonyl.add(left if atoms[left].upper() == "C" else right)
    amides = 0
    for left, right in bonds:
        if {atoms[left].upper(), atoms[right].upper()} != {"C", "N"}:
            continue
        if order.get((left, right), "-") != "-":
            continue
        carbon = left if atoms[left].upper() == "C" else right
        if carbon in carbonyl:
            amides += 1
    return {"largest_ring": max(rings) if rings else 0, "ring_sizes": sorted(rings),
            "amide_bonds": amides, "heavy_atoms": count,
            "unverified": True, "warnings": warnings}


# --------------------------------------------------------------------------- ChEMBL lane

def _chembl_lane(trace, accession, max_molecules):
    found = trace.json(_query(
        f"{CHEMBL}/target.json", target_components__accession=accession, limit=20,
        only="target_chembl_id,pref_name,organism,target_type"))
    targets = [{"target_chembl_id": t.get("target_chembl_id"),
                "pref_name": t.get("pref_name"), "organism": t.get("organism"),
                "target_type": t.get("target_type")} for t in found.get("targets", [])]
    if not targets:
        return _absence(
            f"No ChEMBL target has a component with UniProt accession {accession}. ChEMBL "
            f"holds no bioactivity record keyed to this protein.",
            targets=[], activity_rows_total=0, assays_total=0,
            peptide_or_macrocycle_campaign_established=False,
            runnable_assay_established=False)

    # Pinned to the single-protein targets first: a PROTEIN FAMILY or SELECTIVITY GROUP row
    # aggregates activities measured against relatives, and counting those as precedent
    # against this protein is the transfer-of-annotation error this project names elsewhere.
    ordered = sorted(targets, key=lambda t: t.get("target_type") != "SINGLE PROTEIN")
    chosen = ordered[:MAX_CHEMBL_TARGETS]

    assay_rows, activity_rows = [], []
    assays_total = activities_total = 0
    for target in chosen:
        identifier = target["target_chembl_id"]
        assays = trace.json(_query(
            f"{CHEMBL}/assay.json", target_chembl_id=identifier, limit=MAX_ASSAY_ROWS,
            only="assay_chembl_id,assay_type,description,confidence_score,bao_label,"
                 "assay_organism,assay_category"))
        assays_total += assays.get("page_meta", {}).get("total_count", 0)
        for row in assays.get("assays", []):
            row["_target_chembl_id"] = identifier
            assay_rows.append(row)
        acts = trace.json(_query(
            f"{CHEMBL}/activity.json", target_chembl_id=identifier, limit=MAX_ACTIVITY_ROWS,
            only="molecule_chembl_id,assay_chembl_id,assay_type,standard_type,"
                 "standard_value,standard_units,standard_relation,pchembl_value,"
                 "document_chembl_id,target_organism"))
        activities_total += acts.get("page_meta", {}).get("total_count", 0)
        activity_rows.extend(acts.get("activities", []))

    by_type, by_format, by_confidence = {}, {}, {}
    candidates = []
    for row in assay_rows:
        by_type[row.get("assay_type") or "unknown"] = \
            by_type.get(row.get("assay_type") or "unknown", 0) + 1
        label = row.get("bao_label") or "unlabelled"
        by_format[label] = by_format.get(label, 0) + 1
        score = row.get("confidence_score")
        by_confidence[str(score)] = by_confidence.get(str(score), 0) + 1
        if (score or 0) >= 8 and label.lower() in BIOCHEMICAL_FORMATS:
            candidates.append({"assay_chembl_id": row.get("assay_chembl_id"),
                               "assay_type": row.get("assay_type"),
                               "bao_label": label, "confidence_score": score,
                               "assay_organism": row.get("assay_organism"),
                               "description": (row.get("description") or "")[:300]})

    standard_types, molecules, documents = {}, [], {}
    for row in activity_rows:
        kind = row.get("standard_type") or "unlabelled"
        standard_types[kind] = standard_types.get(kind, 0) + 1
        if row.get("molecule_chembl_id") and row["molecule_chembl_id"] not in molecules:
            molecules.append(row["molecule_chembl_id"])
        if row.get("document_chembl_id"):
            documents[row["document_chembl_id"]] = True

    classified = _chembl_molecules(trace, molecules[:max_molecules])
    papers = _chembl_documents(trace, sorted(documents)[:MAX_DOCUMENTS])

    return {
        "status": "complete", "found": True,
        "targets": targets, "targets_queried": [t["target_chembl_id"] for t in chosen],
        "assays_total": assays_total, "assays_read": len(assay_rows),
        "assay_types": by_type, "assay_formats": by_format,
        "assay_confidence_scores": by_confidence,
        "assay_type_key": {"B": "binding", "F": "functional", "A": "ADME",
                           "T": "toxicity", "P": "physicochemical", "U": "unclassified"},
        "orthogonal_assay_candidates": candidates[:25],
        "orthogonal_assay_candidate_count": len(candidates),
        "activity_rows_total": activities_total, "activity_rows_read": len(activity_rows),
        "activity_standard_types": standard_types,
        "distinct_molecules_in_rows_read": len(molecules),
        **classified,
        "documents": papers,
        # The two guards. Both pinned, both with their reason in the payload rather than in
        # a docstring a consumer will not read.
        "peptide_or_macrocycle_campaign_established": False,
        "campaign_guard": "ChEMBL rows are measurements extracted from publications. A row "
                          "is not a campaign: a campaign is a selection, a hit list, a "
                          "resynthesis and an orthogonal confirmation, none of which ChEMBL "
                          "records. Read these counts as attention, not as precedent that a "
                          "macrocycle campaign succeeded.",
        "runnable_assay_established": False,
        "assay_guard": "A bao_label records the format a published assay used. It does not "
                       "say the construct is obtainable, the reagent is available, the "
                       "protocol is published, or that the readout transfers to this "
                       "pathogen's orthologue.",
    }


def _chembl_molecules(trace, identifiers):
    """Curated molecule fields plus this module's own SMILES classification, kept apart."""
    if not identifiers:
        return {"molecules_classified": 0, "curated_molecule_types": {},
                "sequence_defined_peptides": 0, "approved_or_clinical": 0,
                "natural_products": 0,
                "heuristic_macrocycle_candidates": 0, "heuristic_peptide_like": 0,
                "heuristic_macrocyclic_peptide_candidates": 0,
                "molecule_rows": [],
                "molecule_classification_absent": "No molecule had a bioactivity row to "
                                                  "classify."}
    rows, by_type = [], {}
    peptides = approved = natural = macro = peptide_like = both = 0
    for start in range(0, len(identifiers), MOLECULE_CHUNK):
        chunk = identifiers[start:start + MOLECULE_CHUNK]
        page = trace.json(_query(
            f"{CHEMBL}/molecule.json", molecule_chembl_id__in=",".join(chunk), limit=MOLECULE_CHUNK,
            only="molecule_chembl_id,pref_name,molecule_type,structure_type,helm_notation,"
                 "natural_product,max_phase,molecule_properties,molecule_structures"))
        for molecule in page.get("molecules", []):
            # "Unknown" is one of ChEMBL's nine curated values; a molecule whose record omits
            # the field entirely is a different thing and gets a different key, because the
            # two read identically once lower-cased and a reader would pool them.
            kind = molecule.get("molecule_type") or "not-in-record"
            by_type[kind] = by_type.get(kind, 0) + 1
            properties = molecule.get("molecule_properties") or {}
            structures = molecule.get("molecule_structures") or {}
            shape = smiles_shape(structures.get("canonical_smiles"))
            is_macro = (shape["largest_ring"] or 0) >= MACROCYCLE_RING
            is_peptide_like = (shape["amide_bonds"] or 0) >= PEPTIDE_AMIDES
            has_helm = bool(molecule.get("helm_notation"))
            phase = molecule.get("max_phase")
            try:
                phase_value = float(phase) if phase is not None else None
            except (TypeError, ValueError):
                phase_value = None
            peptides += int(has_helm)
            approved += int((phase_value or 0) >= 1)
            natural += int(bool(molecule.get("natural_product")))
            macro += int(is_macro)
            peptide_like += int(is_peptide_like)
            both += int(is_macro and is_peptide_like)
            rows.append({
                "molecule_chembl_id": molecule.get("molecule_chembl_id"),
                "pref_name": molecule.get("pref_name"),
                "curated_molecule_type": molecule.get("molecule_type"),
                "curated_structure_type": molecule.get("structure_type"),
                "curated_sequence_defined_peptide": has_helm,
                "curated_natural_product": bool(molecule.get("natural_product")),
                "curated_max_phase": phase,
                "full_mwt": properties.get("full_mwt"),
                "num_ro5_violations": properties.get("num_ro5_violations"),
                "heuristic_largest_ring": shape["largest_ring"],
                "heuristic_amide_bonds": shape["amide_bonds"],
                "heuristic_macrocycle_candidate": is_macro,
                "heuristic_peptide_like": is_peptide_like,
                "heuristic_unverified": True,
                "heuristic_warnings": shape["warnings"],
            })
    return {
        "molecules_classified": len(rows),
        "curated_molecule_types": by_type,
        "curated_molecule_type_vocabulary": list(CHEMBL_MOLECULE_TYPES),
        "sequence_defined_peptides": peptides,
        "approved_or_clinical": approved,
        "natural_products": natural,
        "heuristic_macrocycle_candidates": macro,
        "heuristic_peptide_like": peptide_like,
        "heuristic_macrocyclic_peptide_candidates": both,
        "heuristic_method": f"Largest smallest-ring over the parsed SMILES graph, macrocycle "
                            f"at >= {MACROCYCLE_RING} atoms; amide bonds counted as C-N "
                            f"single bonds whose carbon carries a double-bonded oxygen, "
                            f"peptide-like at >= {PEPTIDE_AMIDES}. Computed here, not by a "
                            f"cheminformatics toolkit, and unverified.",
        "curated_type_cannot_identify_peptides": "ChEMBL molecule_type has nine values and "
                                                 "'Peptide' is not one of them. Cyclosporine "
                                                 "is typed 'Protein' and sirolimus 'Small "
                                                 "molecule'. Only helm_notation is a curated "
                                                 "peptide signal, and it is sparse.",
        "molecule_rows": rows,
        "molecule_sample_is_a_prefix_not_a_sample":
            "These molecules are the first distinct molecule_chembl_ids encountered in "
            "ChEMBL's own activity ordering, truncated at max_molecules. That ordering is "
            "not random and is not sorted by anything a reader would choose, so a zero in "
            "any heuristic_* count is an absence in the sampled prefix and not an absence "
            "in the record. Measured 2026-10-27: SARS-CoV-2 Mpro (P0DTD1) has 19332 "
            "activity rows and the first fifty distinct molecules contain no peptide-like "
            "compound at all, although macrocyclic peptide inhibitors of Mpro are "
            "published. Raise max_molecules or read activity_rows_total before concluding "
            "that no peptide precedent exists.",
    }


def _chembl_documents(trace, identifiers):
    """Years and DOIs for the publications the bioactivity rows came from."""
    if not identifiers:
        return _absence("No bioactivity row carried a document identifier.",
                        count=0, years=[], earliest_year=None, dois=[])
    years, dois, titles = {}, [], []
    for start in range(0, len(identifiers), DOCUMENT_CHUNK):
        chunk = identifiers[start:start + DOCUMENT_CHUNK]
        page = trace.json(_query(
            f"{CHEMBL}/document.json", document_chembl_id__in=",".join(chunk), limit=DOCUMENT_CHUNK,
            only="document_chembl_id,year,journal,title,doi"))
        for document in page.get("documents", []):
            year = document.get("year")
            if year:
                years[str(year)] = years.get(str(year), 0) + 1
            if document.get("doi"):
                dois.append(document["doi"])
            if document.get("title"):
                titles.append(document["title"][:160])
    return {"status": "complete", "found": bool(years or dois),
            "count": len(identifiers), "years": years,
            "earliest_year": min(years) if years else None,
            "latest_year": max(years) if years else None,
            "dois": sorted(set(dois))[:50], "titles": titles[:20]}


# --------------------------------------------------------------------------- IEDB lane

def _iedb_count(trace, filters):
    """An exact row count from PostgREST, in one call that transfers one row.

    `Prefer: count=exact` with `Range: 0-0` returns 206 and a `Content-Range` of
    `0-0/<total>`. The alternative -- paging the rows and measuring the list -- would have
    cost tens of calls and megabytes for an antigen like spike, which has 9257 epitopes.
    """
    url = _query(f"{IEDB}/epitope_search", select="structure_id", **filters)
    _, headers, _ = trace.get(url, headers={"Prefer": "count=exact", "Range": "0-0",
                                            "Range-Unit": "items"})
    span = headers.get("Content-Range", "")
    if "/" not in span:
        return None
    tail = span.rsplit("/", 1)[1]
    return int(tail) if tail.isdigit() else None


def _iedb_lane(trace, accession):
    iri = f"UNIPROT:{accession.split('-')[0]}"
    antigen = trace.json(_query(
        f"{IEDB}/antigen_search", parent_source_antigen_iri=f"eq.{iri}",
        select="parent_source_antigen_iri,parent_source_antigen_names,"
               "parent_source_antigen_source_org_name,structure_types,mhc_classes,"
               "assay_names,reference_dates,pdb_ids,epitope_structures_defined,"
               "antibody_isotypes,qualitative_measures"))
    if not antigen:
        return _absence(
            f"IEDB holds no antigen record whose parent source antigen is {iri}. No mapped "
            f"epitope, and therefore no immune-assay format, is on record for this protein.",
            iri=iri, epitopes_total=0, surface_exposure_established=False)

    row = antigen[0]
    assay_names = sorted(row.get("assay_names") or [])
    lowered = [name.lower() for name in assay_names]
    binding = [name for name, low in zip(assay_names, lowered)
               if any(hint in low for hint in BINDING_ASSAY_HINTS)]
    structural = [name for name, low in zip(assay_names, lowered)
                  if any(hint in low for hint in STRUCTURAL_ASSAY_HINTS)]
    contains = {"parent_source_antigen_iris": "cs.{%s}" % iri}
    total = _iedb_count(trace, contains)
    bcell = _iedb_count(trace, {**contains, "bcell_iris": "not.is.null"})
    tcell = _iedb_count(trace, {**contains, "tcell_iris": "not.is.null"})
    pdb = row.get("pdb_ids") or []
    years = sorted(str(y) for y in (row.get("reference_dates") or []) if y)

    return {
        "status": "complete", "found": True, "iri": iri,
        "antigen_names": row.get("parent_source_antigen_names") or [],
        "source_organism": row.get("parent_source_antigen_source_org_name"),
        "epitopes_total": total,
        "epitopes_with_bcell_assay": bcell,
        "epitopes_with_tcell_assay": tcell,
        "structure_types": sorted(row.get("structure_types") or []),
        "epitope_structures_defined": sorted(row.get("epitope_structures_defined") or []),
        "mhc_classes": sorted(row.get("mhc_classes") or []),
        "antibody_isotypes": sorted(row.get("antibody_isotypes") or []),
        "qualitative_measures": sorted(row.get("qualitative_measures") or []),
        "assay_formats_distinct": len(assay_names),
        "assay_formats": assay_names,
        "binding_assay_formats": binding,
        "structural_assay_formats": structural,
        "reference_years": years,
        "earliest_reference_year": years[0] if years else None,
        "pdb_entry_count": len(pdb),
        "pdb_entries_sample": sorted(pdb)[:20],
        # The guard this lane needs. A B-cell epitope count is the most tempting number in
        # the payload and the easiest to over-read.
        "surface_exposure_established": False,
        "accessibility_guard": "A linear peptide bound by antibodies in an ELISA was "
                               "presented as a synthetic peptide, not necessarily as a "
                               "solvent-exposed surface of the folded protein in the "
                               "relevant conformation. Discontinuous epitopes and the "
                               "structural assay formats are the entries that bear on "
                               "accessibility; a linear-peptide count on its own does not.",
        "tcell_guard": "T-cell and MHC-binding entries record proteasomal processing and "
                       "presentation of fragments. They say nothing about whether the "
                       "epitope is exposed on the intact protein, and must not be pooled "
                       "with the B-cell count.",
    }


# --------------------------------------------------------------------------- MaveDB lane

def _mavedb_lane(trace, accession, target_name):
    index = trace.json(f"{MAVEDB}/statistics/target/gene/uniprot-identifier")
    bare = accession.split("-")[0]
    exact = int(index.get(bare, 0) or 0)

    named, name_reason = [], None
    if target_name:
        _, _, raw = trace.post_json(f"{MAVEDB}/score-sets/search", {"text": target_name})
        answer = json.loads(raw.decode())
        for score_set in (answer.get("scoreSets") or []):
            identifiers = []
            for gene in (score_set.get("targetGenes") or []):
                for external in (gene.get("externalIdentifiers") or []):
                    inner = external.get("identifier") or {}
                    identifiers.append(f"{inner.get('dbName')}:{inner.get('identifier')}")
            named.append({
                "urn": score_set.get("urn"), "title": score_set.get("title"),
                "num_variants": score_set.get("numVariants"),
                "published_date": score_set.get("publishedDate"),
                "target_genes": [g.get("name") for g in (score_set.get("targetGenes") or [])],
                "external_identifiers": identifiers,
                "accession_confirmed": any(
                    i.lower() == f"uniprot:{bare}".lower() for i in identifiers),
            })
    else:
        name_reason = ("No target_name was supplied, so only the exact UniProt index was "
                       "consulted. MaveDB free-text search does not index UniProt "
                       "accessions, so the accession alone cannot recover score-set URNs.")

    confirmed = [row for row in named if row["accession_confirmed"]]
    payload = {
        "status": "complete",
        "found": bool(exact or confirmed or named),
        "uniprot_index_size": len(index),
        "accession_in_uniprot_index": bare in index,
        "accession_score_set_count": exact,
        "accession_confirmed_score_sets": confirmed,
        "name_matched_score_sets": named,
        "name_match_query": target_name,
        "name_match_unverified": True,
        "name_match_reason": name_reason,
        "dms_landscape_established": bool(exact or confirmed),
        "dms_guard": "A score set is a public label set for method development. It is not "
                     "evidence that the target is tractable, essential or accessible, and a "
                     "variant-effect map of a relative is not a measurement on this "
                     "pathogen's protein.",
        "accession_index_is_sparse": "MaveDB's UniProt index is exact but incomplete. "
                                     "Measured 2026-10-27: six SARS-CoV-2 receptor-binding-"
                                     "domain score sets (binding, escape and expression for "
                                     "Omicron BA.1 and BA.2) carry no UniProt identifier, so "
                                     "the index reports zero for P0DTC2 while the data "
                                     "exists. A zero here is a zero in the index, not in "
                                     "MaveDB.",
    }
    if not payload["found"]:
        payload["status"] = "absent"
        payload["reason"] = (f"No MaveDB score set is keyed to UniProt {bare}, and no name "
                             f"match was available. Read with accession_index_is_sparse.")
    return payload


# --------------------------------------------------------------------------- entry point

def binders_and_assays(accession, target_name=None, max_molecules=50, record=True):
    """Prior bioactivities, assay formats and public mutational landscapes for one protein.

    `accession` is a UniProt accession -- the `protein_accession` fact, which `intake` provides
    and the onset state does not hold. `target_name` is used only for the MaveDB name-matched
    lane and every hit it produces is labelled unverified. `max_molecules` caps how many
    distinct ChEMBL molecules are fetched and classified. `record=True` writes the artifact;
    a call inside an episode passes False, because the episode writes the record.

    Raises ValueError on an accession that is not a UniProt accession. A lane whose host fails
    returns a recorded absence naming the exception and the payload status becomes `partial`.
    """
    if not isinstance(accession, str) or not ACCESSION.match(accession.strip().upper()):
        raise ValueError(
            f"{accession!r} is not a UniProt accession. This tool is keyed on "
            f"protein_accession; a nucleotide accession such as NC_045512.2 would return "
            f"three empty lanes indistinguishable from a real absence, so it is refused "
            f"rather than queried.")
    accession = accession.strip().upper()
    if not isinstance(max_molecules, int) or max_molecules < 1:
        raise ValueError(f"max_molecules must be a positive integer, not {max_molecules!r}")
    # Clamped rather than raised. `tools.Parameter.maximum` is advisory in this project -- a
    # caller that exceeds it is making a well-typed wrong guess the caller charges and
    # records -- but `worst_case_seconds` is arithmetic over the call count, so an unclamped
    # value would silently falsify the one number the wall gate reads. The clamp is therefore
    # reported in the payload rather than applied quietly.
    ceiling = MOLECULE_CHUNK * 2
    applied = min(max_molecules, ceiling)

    trace = _Trace()
    lanes, failed = {}, []
    for name, call in (("chembl", lambda: _chembl_lane(trace, accession, applied)),
                       ("iedb", lambda: _iedb_lane(trace, accession)),
                       ("mavedb", lambda: _mavedb_lane(trace, accession, target_name))):
        try:
            lanes[name] = call()
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError,
                json.JSONDecodeError) as exc:
            lanes[name] = _unavailable(exc)
            failed.append(name)

    years = []
    documents = (lanes["chembl"].get("documents") or {}) if "chembl" in lanes else {}
    if isinstance(documents, dict) and documents.get("earliest_year"):
        years.append(str(documents["earliest_year"]))
    if lanes.get("iedb", {}).get("earliest_reference_year"):
        years.append(lanes["iedb"]["earliest_reference_year"])
    for row in lanes.get("mavedb", {}).get("name_matched_score_sets") or []:
        if row.get("accession_confirmed") and row.get("published_date"):
            years.append(str(row["published_date"])[:4])

    if failed and len(failed) == 3:
        status = "unavailable"
    elif failed:
        status = "partial"
    elif not any(lanes[name].get("found") for name in lanes):
        status = "absent"
    else:
        status = "complete"

    payload = {
        "experiment": "binders-and-assays",
        "status": status,
        "accession": accession,
        "target_name": target_name,
        "max_molecules_requested": max_molecules,
        "max_molecules_applied": applied,
        "max_molecules_clamped": applied != max_molecules,
        "lanes_unavailable": failed,
        "chembl": lanes.get("chembl"),
        "iedb": lanes.get("iedb"),
        "mavedb": lanes.get("mavedb"),
        # Year granularity, not a date: ChEMBL documents and IEDB references carry a year and
        # nothing finer, so a day would be invented. The dossier reads this as
        # `available_from` and tests only whether it is None, so coarseness costs nothing
        # there and a fabricated day would cost something here.
        "earliest_evidence_year": min(years) if years else None,
        "network_calls": trace.count,
        "calls": trace.calls,
        "sources": [
            {"name": "ChEMBL web services", "url": f"{CHEMBL}/target.json",
             "licence": "CC BY-SA 3.0; EMBL-EBI terms of use"},
            {"name": "IEDB query API", "url": f"{IEDB}/epitope_search",
             "licence": "IEDB data free for non-commercial use; attribution required"},
            {"name": "MaveDB API", "url": f"{MAVEDB}/score-sets/search",
             "licence": "per score set, usually CC0 1.0 or CC BY 4.0"},
        ],
        "urls_hit": trace.urls,
        "axis_bearing": {
            "precedent": "Bears on it. ChEMBL bioactivity rows and their molecule "
                         "classification say whether anyone has measured a peptide-like or "
                         "macrocyclic compound against this protein. They do not say a "
                         "campaign was run or that it worked.",
            "assay_availability": "Bears on it. ChEMBL assay bao_label plus confidence_score "
                                  "and IEDB's enumerated assay formats say which readouts "
                                  "have been published on this target. Neither says a "
                                  "readout is runnable here.",
            "public_labels": "Bears on it. MaveDB's UniProt index is exact for presence; its "
                             "absences are index absences, not data absences.",
            "accessibility": "Bears on it weakly. IEDB discontinuous epitopes and structural "
                             "assay formats indicate surface presentation; linear-peptide "
                             "and T-cell entries do not.",
            "reagent_tractability": "Does not bear on it. Nothing here says a construct is "
                                    "expressible, soluble or immobilisable.",
            "essentiality": "Does not bear on it. A measured IC50 is not a loss-of-function "
                            "phenotype.",
            "conservation": "Does not bear on it. Every lane is keyed to one accession and "
                            "reports nothing about the family.",
        },
        "limitations": [
            "Keyed to one UniProt accession. A hit against a relative is not a measurement "
            "on this pathogen's protein, and no lane here transfers annotation across "
            "homologues.",
            "A ChEMBL activity row is a measurement extracted from a publication. It is not "
            "a campaign, not a validated assay, and not evidence that a binder was "
            "developable.",
            "ChEMBL molecule_type cannot distinguish a peptide from a small molecule: the "
            "vocabulary has no 'Peptide' value, cyclosporine is typed 'Protein' and "
            "sirolimus 'Small molecule'. The curated peptide signal is helm_notation and it "
            "is sparse.",
            "ChEMBL carries no macrocycle flag. Every heuristic_* field is computed in this "
            "module from canonical SMILES without a cheminformatics toolkit, is marked "
            "unverified, and should be treated as a screen that generates candidates for "
            "inspection rather than a classification.",
            "The macrocycle screen fires on any ring of twelve or more atoms, so a "
            "disulfide-bridged loop inside a large linear peptide counts as a macrocycle "
            "candidate. Both candidates returned for P0DTC2 are ~4.06 kDa peptides of "
            "this kind, not synthetic macrocycles. Read heuristic_largest_ring and "
            "full_mwt together, per row, before treating a candidate as precedent.",
            "ChEMBL silently ignores an unrecognised query parameter and returns the "
            "unfiltered count, so any filter not verified against a changed total_count is "
            "worthless. Only accession, target and molecule-identifier filters used here "
            "were verified to change the count.",
            "Activity and assay rows are read to a cap (500 and 200 per ChEMBL target, at most "
            "three targets) and molecules are classified only to max_molecules. *_total is "
            "ChEMBL's own count and *_read is what was read. The molecule sample is the "
            "head of ChEMBL's activity ordering, not a random sample: measured on "
            "2026-10-27, Mpro (P0DTD1, 19332 activity rows) yielded no peptide-like "
            "molecule in the first fifty, though macrocyclic peptide inhibitors of Mpro "
            "are published. A zero in a heuristic_* count is a zero in the prefix.",
            "An IEDB epitope count is a research-attention proxy in the same sense as a "
            "structure count. An antigen with no entries may be unstudied rather than "
            "inaccessible.",
            "IEDB B-cell entries are measured on presented peptides, often synthetic, and do "
            "not establish exposure on the folded protein in the relevant conformation.",
            "MaveDB's UniProt index is exact but incomplete; SARS-CoV-2 RBD score sets carry "
            "no UniProt identifier. An accession-only zero is not an absence of DMS data.",
            "The MaveDB name-matched lane is a text match. Every hit is labelled "
            "accession_confirmed true or false and the lane is unverified.",
            "Dates are year granularity for ChEMBL and IEDB. earliest_evidence_year is the "
            "earliest year any lane reported, not a publication date for any single claim.",
            "Nothing here predicts binding, affinity, permeability or developability for any "
            "molecule, and nothing here was computed from a structure.",
        ],
        "interpretation": "Prior-art and assay-format evidence for one protein accession, "
                          "from ChEMBL, IEDB and MaveDB. Counts record what has been "
                          "measured and published, which is attention; the guard fields "
                          "(peptide_or_macrocycle_campaign_established, "
                          "runnable_assay_established, surface_exposure_established, "
                          "dms_landscape_established) are the only claims about the world, "
                          "and three of the four are pinned False by construction.",
        "provenance": provenance(),
    }
    payload["content_sha256"] = content_digest(payload)
    if record:
        write_json(ROOT / f"data/processed/bioactivity/{accession}.json", payload)
    return payload
