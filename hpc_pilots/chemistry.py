"""Macrocyclic building-block registry and library representation.

Block *classes* are from the genetic-code-reprogramming literature (Katoh & Suga
2022, Annu Rev Biochem 91:221-243; Huang, Wiedmann & Suga 2019, Chem Rev
119:10360-10391). This module is a schema with literature-derived classes, **not
the SynX catalog**. Chemical identifiers, CCD codes, stereochemistry descriptors,
molecular weights, reprogrammed codon assignments and Rosetta parameters are
recorded as pending and must arrive from the platform export. Nothing here
asserts that a block is available on the SynX platform, nor predicts any
binding, enrichment, inhibition or antiviral property of a designed macrocycle.
"""
import json

from .common import ROOT, write_json

CANONICAL = "ACDEFGHIKLMNPQRSTVWY"

# RaPID-style chemistry: a chloroacetyl initiator is incorporated by a flexizyme-charged
# initiator tRNA, and its chloroacetyl group reacts with a downstream cysteine thiol to
# close a thioether macrocycle. The initiator and the closing Cys are therefore fixed
# positions; the region between them is the randomised library.
INITIATORS = ("ClAc-D-Tyr", "ClAc-L-Tyr")
CLOSURE = "C"

# (code, block_class, role, stereochemistry). Representative classes the ribosome is
# reported to accept, not an availability claim and not an exhaustive alphabet.
BLOCKS = (
    ("ClAc-D-Tyr", "chloroacetyl initiator", "initiator", "D"),
    ("ClAc-L-Tyr", "chloroacetyl initiator", "initiator", "L"),
    ("NMe-Gly", "N-methyl", "elongator", "achiral"),
    ("NMe-Ala", "N-methyl", "elongator", "L"),
    ("NMe-Phe", "N-methyl", "elongator", "L"),
    ("D-Ala", "D-alpha-amino", "elongator", "D"),
    ("D-Phe", "D-alpha-amino", "elongator", "D"),
    ("D-Ser", "D-alpha-amino", "elongator", "D"),
    ("beta-Ala", "beta-amino", "elongator", "achiral"),
)

PENDING_FIELDS = ("chemical_identifier", "ccd_code", "smiles", "stereochemistry_descriptor",
                  "molecular_weight", "reprogrammed_codon", "rosetta_params", "rotamer_library")


def registry():
    """One row per building block, with every platform-supplied field explicitly pending."""
    rows = [{"code": c, "block_class": "canonical", "role": "elongator", "stereochemistry": "L",
             "source": "canonical proteinogenic alphabet"} for c in CANONICAL]
    rows += [{"code": code, "block_class": klass, "role": role, "stereochemistry": stereo,
              "source": "Katoh & Suga 2022 block class; platform availability unverified"}
             for code, klass, role, stereo in BLOCKS]
    for row in rows:
        for field in PENDING_FIELDS:
            row.setdefault(field, None)
    return rows


def alphabet(noncanonical=True):
    """Ordered block codes. Canonical-only reproduces the 20-letter encoding of the GB1 pilot."""
    codes = list(CANONICAL)
    if noncanonical:
        codes += [code for code, *_ in BLOCKS]
    return codes


def library_spec(initiator="ClAc-D-Tyr", random_region=(4, 8), elongator_classes=
                 ("canonical", "N-methyl", "D-alpha-amino")):
    """A focused thioether-macrocycle library definition: fixed initiator, randomised region, closing Cys."""
    if initiator not in INITIATORS:
        raise ValueError(f"{initiator!r} is not a registered chloroacetyl initiator")
    low, high = random_region
    if not 1 <= low <= high:
        raise ValueError("Invalid randomised-region bounds")
    codes = [c for c in CANONICAL if c != CLOSURE]
    codes += [code for code, klass, role, _ in BLOCKS
              if role == "elongator" and klass in elongator_classes]
    if not codes:
        raise ValueError("Empty elongator alphabet")
    return {"initiator": initiator, "closure_residue": CLOSURE, "random_region": [low, high],
            "elongator_classes": list(elongator_classes), "elongator_alphabet": codes,
            "cyclisation": "spontaneous thioether between chloroacetyl initiator and downstream Cys",
            "theoretical_diversity": sum(len(codes) ** n for n in range(low, high + 1)),
            "endpoint_note": "A library definition only; no binding, enrichment, inhibition "
                             "or antiviral property is predicted for any member."}


def sample_library(spec, count, rng):
    """Enumerate candidate macrocycles as block-code sequences. A design proposal, not a prediction."""
    codes, low, high = spec["elongator_alphabet"], *spec["random_region"]
    members = []
    for _ in range(count):
        n = int(rng.integers(low, high + 1))
        middle = [codes[int(i)] for i in rng.integers(0, len(codes), size=n)]
        members.append([spec["initiator"], *middle, spec["closure_residue"]])
    return members


def encode(members, codes=None):
    """Integer-encode block sequences against the alphabet, right-padded with -1.

    Returns the matrix and the per-column cardinality so a surrogate can one-hot it
    without assuming the canonical 20-letter alphabet.
    """
    codes = codes or alphabet()
    index = {c: i for i, c in enumerate(codes)}
    width = max(len(m) for m in members)
    matrix = [[index[b] if b in index else _unknown(b) for b in m] + [-1] * (width - len(m))
              for m in members]
    return matrix, len(codes) + 1


def _unknown(block):
    raise KeyError(f"Block {block!r} is not in the registry; add it before encoding")


def build_registry():
    out = ROOT / "data/processed/chemistry"
    rows = registry()
    spec = library_spec()
    write_json(out / "registry.json", {
        "blocks": rows,
        "counts": {"canonical": len(CANONICAL), "noncanonical": len(BLOCKS)},
        "default_library": spec,
        "pending_fields": list(PENDING_FIELDS),
        "limitations": [
            "Block classes are literature-derived; this is not the SynX catalog.",
            "No chemical identifier, CCD code, SMILES, stereochemistry descriptor or "
            "molecular weight is asserted for any non-canonical block.",
            "Reprogrammed codon assignments and Rosetta parameters/rotamer libraries are "
            "pending the platform export; the physics-based non-canonical track is untouched.",
            "Theoretical diversity is a combinatorial count of a specification, not a "
            "synthesised library and not a claim about display coverage.",
            "Library enumeration predicts nothing. Enrichment, affinity, inhibition and "
            "antiviral activity are separate endpoints requiring separate assays."]})
    print(json.dumps({"blocks": len(rows), "noncanonical": len(BLOCKS),
                      "theoretical_diversity": spec["theoretical_diversity"],
                      "output": str(out / "registry.json")}))
