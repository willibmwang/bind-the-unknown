"""Prototype-family evidence packets and the synthetic evidence-update probes.

Provenance discipline. Most axis assessments below are the project's own Week 4
target-selection write-up, awaiting review with Askar and the SynX experimental lead.
They are recorded as `internal assessment` and are **not** independent literature
citations. Only the public-label axis is independently verified, against the
ProteinGym archive actually on disk. Claims the write-up flagged as asserted from
memory carry `verified=False` and must not be used as positive controls until a
citation is attached.

The update probes are synthetic. Per the research brief, synthetic updates test
evidence-following; they do not test the absence of historical knowledge already in
model weights.
"""
import json

from .common import ROOT, write_json
from .dossier import AXES, Dossier, evidence

INTERNAL = "W4 target-selection write-up; proposal pending Askar/SynX review"
ABSENT = ("No claim on this axis in the W4 target-selection write-up. Recorded as an explicit "
          "absence so that the axis is offered and returns nothing, rather than not being "
          "offered at all.")

# Whether a target with no claim on an axis still offers that axis, answering `unknown`.
#
# Off, the packets are as authored and coverage is uneven: seven axes on Mpro, four on Nipah's
# F glycoprotein. `retrievable_axes` carries that shape into the state, so a policy can rank
# targets by counting offers without reading a claim -- and `coverage_static`, which does
# exactly that and nothing else, outscores every policy that reads the state. The audit marks
# the twin readout `led_by_a_trivial_policy` and `discriminating_readouts` is empty.
#
# On, every target offers all seven axes and the count is constant across targets, so counting
# carries no information by construction. Nothing else moves: `assess` already answers
# `unknown` for an axis with no entries, and `unknown` never substantiates, so `readiness`,
# the ranking and the reference answer are untouched. What changes is the offered set, and
# therefore the chance floor the probe is normalised against -- more offers that settle
# nothing makes the probe harder, which is a real change and not a free repair.
#
# A module constant rather than a parameter because that is how this project varies an
# instrument: `improve` patches `twins.OBSERVED_FRACTION` the same way, and the guarded search
# can then evaluate the variant against the planted hacks rather than trusting it.
PAD_ABSENT_AXES = False
PG = "ProteinGym v1.3 archive verified present in data/raw/proteingym_v1.3.zip"

# target -> (pathogen, compartment, [(axis, strength, claim, source, verified, round)])
#
# Three positions, not four: `family` is a packet-level key and has no per-target slot. The
# comment said otherwise for as long as this table existed, which matters now that something
# other than a human may write one -- every consumer destructures a 3-tuple positionally
# (`invariance` additionally reads `entry[3]` for the source), so the arity and the order are
# load-bearing. Note also that the axis tuple is strength-then-claim while `evidence()` takes
# claim-then-strength; `packet` reorders them.
PACKETS = {
    "coronaviridae": {
        "family": "Coronaviridae",
        "pathogen": "SARS-CoV-2",
        "prototype_rationale": "WHO prototype-pathogen approach; betacoronavirus prototype.",
        "targets": {
            "Mpro (nsp5)": ("SARS-CoV-2", "cytosolic", [
                ("reagent_tractability", "strong", "Routine soluble E. coli expression; well-behaved and inexpensive reagent.", INTERNAL, True, 0),
                ("essentiality", "strong", "Polyprotein processing at eleven sites is obligatory for replication.", INTERNAL, True, 0),
                ("conservation", "strong", "Active site highly conserved across sarbecoviruses; workable across betacoronaviruses.", INTERNAL, True, 0),
                ("accessibility", "weak", "Cytosolic homodimeric cysteine protease; requires a membrane-permeability story.", INTERNAL, True, 0),
                ("assay_availability", "strong", "Standard FRET peptide-cleavage assay available for orthogonal functional follow-up.", INTERNAL, True, 0),
                ("public_labels", "strong", "R1AB_SARS2_Flynn_2022: 5,725 single substitutions, FRET and growth readout.", PG, True, 0),
                ("precedent", "moderate", "Published macrocyclic peptide campaigns including RaPID-derived work are asserted from memory and uncited.", INTERNAL, False, 0)]),
            "PLpro (nsp3)": ("SARS-CoV-2", "cytosolic", [
                ("reagent_tractability", "moderate", "Soluble and tractable; the zinc-binding domain requires attention in construct design.", INTERNAL, True, 0),
                ("essentiality", "strong", "Cleaves nsp1/2/3 boundaries; additionally strips ubiquitin and ISG15 from host proteins.", INTERNAL, True, 0),
                ("conservation", "moderate", "Conserved but less so than Mpro across genera.", INTERNAL, True, 0),
                ("accessibility", "weak", "Cytosolic; permeability story required.", INTERNAL, True, 0),
                ("public_labels", "unknown", "No public deep mutational scan identified.", INTERNAL, True, 0)]),
            "nsp10/nsp16 interface": ("SARS-CoV-2", "cytosolic", [
                ("reagent_tractability", "weak", "Two-protein obligate complex; a harder reagent than a single soluble enzyme.", INTERNAL, True, 0),
                ("essentiality", "strong", "nsp16 2'-O-methyltransferase caps viral RNA to evade innate sensing and is catalytically dependent on nsp10.", INTERNAL, True, 0),
                ("conservation", "strong", "Conserved and essential for immune evasion.", INTERNAL, True, 0),
                ("accessibility", "weak", "Cytosolic; permeability story required.", INTERNAL, True, 0),
                ("precedent", "unknown", "No macrocycle campaign identified against this interface.", INTERNAL, True, 0),
                ("public_labels", "unknown", "No public deep mutational scan identified.", INTERNAL, True, 0)]),
            "nsp13 helicase": ("SARS-CoV-2", "cytosolic", [
                ("reagent_tractability", "moderate", "Moderately tractable reagent; superfamily-1 helicase with a zinc-binding domain.", INTERNAL, True, 0),
                ("essentiality", "strong", "NTPase and 5'-to-3' unwinding activity essential for replication.", INTERNAL, True, 0),
                ("conservation", "strong", "Highly conserved across coronaviruses.", INTERNAL, True, 0),
                ("accessibility", "weak", "Cytosolic; permeability story required.", INTERNAL, True, 0),
                ("public_labels", "unknown", "No public deep mutational scan identified.", INTERNAL, True, 0)]),
            "Spike RBD": ("SARS-CoV-2", "extracellular", [
                ("reagent_tractability", "strong", "Well-established soluble receptor-binding-domain reagent.", INTERNAL, True, 0),
                ("essentiality", "strong", "ACE2 engagement required for entry.", INTERNAL, True, 0),
                ("conservation", "weak", "Immune-evasion surface under intense positive selection; wrong basis for a family-transfer claim.", INTERNAL, True, 0),
                ("accessibility", "strong", "Extracellular; no permeability story required.", INTERNAL, True, 0),
                ("public_labels", "strong", "SPIKE_SARS2_Starr_2020_binding (3,802) and _expression (3,798); among the cleanest binding landscapes available.", PG, True, 0)]),
            "Spike S2 stem-helix": ("SARS-CoV-2", "extracellular", [
                ("reagent_tractability", "weak", "Needs a stabilised construct or peptide presentation rather than whole prefusion spike.", INTERNAL, True, 0),
                ("essentiality", "strong", "Fusion machinery required for entry.", INTERNAL, True, 0),
                ("conservation", "strong", "Far better conserved across betacoronaviruses than the RBD.", INTERNAL, True, 0),
                ("accessibility", "strong", "Extracellular and targeted by known broadly neutralising antibodies, establishing accessibility.", INTERNAL, True, 0),
                ("public_labels", "unknown", "No public deep mutational scan identified.", INTERNAL, True, 0)]),
        }},
    "orthomyxoviridae": {
        "family": "Orthomyxoviridae",
        "pathogen": "Influenza A",
        "prototype_rationale": "Second family chosen for runnable labelled transfer tests (H1 to H3).",
        "targets": {
            "HA stem (HA2 helix A)": ("Influenza A", "extracellular", [
                ("reagent_tractability", "moderate", "Stem-directed constructs are established but require stabilisation.", INTERNAL, True, 0),
                ("essentiality", "strong", "Fusion machinery required for entry.", INTERNAL, True, 0),
                ("conservation", "strong", "Conserved epitope recognised by group-specific broadly neutralising antibodies.", INTERNAL, True, 0),
                ("accessibility", "strong", "Extracellular; no permeability story required.", INTERNAL, True, 0),
                ("public_labels", "strong", "A0A2Z5U3Z0_9INFA_Doud_2016 (H1N1, 10,715) and C6KNH7_9INFA_Lee_2018 (H3N2, 10,754) support a grouped H1-to-H3 shift test.", PG, True, 0),
                ("precedent", "moderate", "Peptide-binder precedent against the stem is asserted and uncited.", INTERNAL, False, 0)]),
            "NA catalytic site": ("Influenza A", "extracellular", [
                ("reagent_tractability", "strong", "Sialidase expressed as a soluble tetrameric head; validated drug target.", INTERNAL, True, 0),
                ("essentiality", "strong", "Sialidase activity required for virion release.", INTERNAL, True, 0),
                ("conservation", "strong", "Catalytic residues and framework highly conserved.", INTERNAL, True, 0),
                ("accessibility", "strong", "Extracellular; no permeability story required.", INTERNAL, True, 0),
                ("assay_availability", "strong", "Established sialidase activity assay available as an orthogonal endpoint.", INTERNAL, True, 0),
                ("public_labels", "moderate", "NRAM_I33A0_Jiang_2016 is present but small (298 mutants).", PG, True, 0)]),
            "PA-PB1 interface": ("Influenza A", "cytosolic", [
                ("reagent_tractability", "moderate", "PA C-terminal domain binds an N-terminal PB1 peptide; a well-characterised interface.", INTERNAL, True, 0),
                ("essentiality", "strong", "Polymerase assembly required for replication.", INTERNAL, True, 0),
                ("accessibility", "weak", "Intracellular; permeability story required.", INTERNAL, True, 0),
                ("public_labels", "moderate", "RDRP_I33A0_Li_2023 (PB1, 12,003) and A4D664_9INFA_Soh_2019 (PB2, 14,421) are on disk.", PG, True, 0),
                ("precedent", "moderate", "Published peptide inhibitors of this interface are asserted and uncited.", INTERNAL, False, 0)]),
        }},
    "henipavirus": {
        "family": "Henipavirus",
        "pathogen": "Nipah virus",
        "prototype_rationale": "Stretch robustness family: WHO R&D Blueprint priority, better narrative, no public labels.",
        "targets": {
            "G attachment glycoprotein": ("Nipah virus", "extracellular", [
                ("reagent_tractability", "moderate", "Obtainable as a soluble six-bladed beta-propeller head domain.", INTERNAL, True, 0),
                ("essentiality", "strong", "Ephrin-B2 and ephrin-B3 binding required for entry.", INTERNAL, True, 0),
                ("conservation", "moderate", "Conservation ladder available for transfer testing: NiV-B, NiV-M, Hendra, Cedar, Langya.", INTERNAL, True, 0),
                ("accessibility", "strong", "Genuinely extracellular; receptor-binding site well mapped.", INTERNAL, True, 0),
                ("public_labels", "unknown", "No public deep mutational scan; labelled transfer tests stay blocked on SynX data.", INTERNAL, True, 0)]),
            "F fusion glycoprotein": ("Nipah virus", "extracellular", [
                ("reagent_tractability", "weak", "Metastable prefusion conformation needs stabilisation to be a usable display reagent.", INTERNAL, True, 0),
                ("essentiality", "strong", "Class I fusion protein required for entry; activated by cathepsin L cleavage.", INTERNAL, True, 0),
                ("accessibility", "strong", "Extracellular.", INTERNAL, True, 0),
                ("public_labels", "unknown", "No public deep mutational scan identified.", INTERNAL, True, 0)]),
        }},
}

# Synthetic evidence-update probes, released at a decision round. `kind` is what the
# release *should* do to the recommendation, which is what the stability metric scores.
PROBES = {
    "coronaviridae": [
        ("relevant", "Mpro (nsp5)", "assay_availability", "strong",
         "Orthogonal FRET cleavage assay confirmed available within the round's turnaround.", 2),
        ("contradictory", "Mpro (nsp5)", "conservation", "weak",
         "Active-site substitutions reported in a divergent betacoronavirus clade, weakening the family-wide conservation claim.", 4),
        ("irrelevant", "nsp13 helicase", "precedent", "weak",
         "A review mentions helicase inhibitors of an unrelated virus family; bears on no axis of the leading target.", 3),
    ],
    "orthomyxoviridae": [
        ("relevant", "HA stem (HA2 helix A)", "reagent_tractability", "strong",
         "Stabilised stem construct confirmed obtainable and immobilisable for multiple selection rounds.", 2),
        ("contradictory", "NA catalytic site", "conservation", "weak",
         "Antigenic drift reported at positions adjacent to the catalytic framework, contesting the conservation assessment.", 4),
        ("irrelevant", "PA-PB1 interface", "precedent", "moderate",
         "A structural note on an unrelated polymerase; bears on no axis of the leading target.", 3),
    ],
    "henipavirus": [
        ("relevant", "G attachment glycoprotein", "assay_availability", "moderate",
         "Pseudotyped entry-inhibition assay identified as a feasible orthogonal endpoint.", 2),
        ("irrelevant", "F fusion glycoprotein", "precedent", "weak",
         "A general review of class I fusion proteins; bears on no axis of the leading target.", 3),
    ],
}


def packet(name):
    """Build the dossiers and the release timeline for one family.

    At most one base entry per `(target, axis)`, enforced rather than assumed. Every consumer
    builds the retrievable pool as `{(target, axis): entry}`, so a second entry on the same
    pair silently replaces the first: the dossier still observes both and `assess` still sees
    both, but only one of them is ever purchasable, and the count of offered claims quietly
    disagrees with the count of authored ones. All eleven hand-written targets happen to
    satisfy this, which is why it never surfaced; a generated packet has no such luck.

    Intra-axis contradiction belongs in `PROBES`, which is dated and so enters the pool on a
    release schedule rather than competing for the same opening key.
    """
    spec = PACKETS[name]
    dossiers, entries = [], []
    for target, (pathogen, compartment, axes) in spec["targets"].items():
        dossiers.append(Dossier(target, pathogen, spec["family"], compartment))
        seen = set()
        for axis, strength, claim, source, verified, round_ in axes:
            if axis in seen:
                raise ValueError(f"{name}/{target} carries two base entries on {axis!r}; the "
                                 f"retrievable pool is keyed on (target, axis) and the second "
                                 f"would replace the first unseen")
            seen.add(axis)
            entries.append((target, evidence(f"{target}:{axis}", axis, claim, strength,
                                             source, round_, verified)))
        if PAD_ABSENT_AXES:
            # Available from round 0 like the authored claims, and `verified=True`: the
            # absence is a real and checkable statement about the write-up, not an assertion
            # from memory about the biology. Strength `unknown` keeps it out of every
            # substantiation count, so the ranking is identical to the unpadded packet.
            for axis in AXES:
                if axis not in seen:
                    entries.append((target, evidence(f"{target}:{axis}", axis, ABSENT,
                                                     "unknown", INTERNAL, 0, True)))
    probes = []
    for kind, target, axis, strength, claim, round_ in PROBES.get(name, []):
        prior = f"{target}:{axis}"
        conflicts = prior if kind == "contradictory" else None
        probes.append({"kind": kind, "target": target,
                       "entry": evidence(f"probe:{kind}:{target}:{axis}", axis, claim, strength,
                                         "synthetic evidence-update probe", round_, True, conflicts)})
    return spec, dossiers, entries, probes


def build_packets():
    out = ROOT / "data/processed/targets"
    summary = {}
    for name in PACKETS:
        spec, dossiers, entries, probes = packet(name)
        by_target = {d.target: d for d in dossiers}
        for target, entry in entries:
            by_target[target].observe(entry, now=0)
        rows = [{"target": d.target, "compartment": d.compartment,
                 "axes": {a: d.assess(a) for a in AXES}, "readiness": d.readiness(),
                 "open_uncertainties": d.open_uncertainties()} for d in dossiers]
        uncited = [e["id"] for _, e in entries if not e["verified"]]
        write_json(out / f"{name}.json", {
            "family": spec["family"], "pathogen": spec["pathogen"],
            "prototype_rationale": spec["prototype_rationale"],
            "selection_rubric": AXES, "targets": rows,
            "evidence": [e for _, e in entries],
            "update_probes": probes,
            "claims_requiring_citation": uncited,
            "limitations": [
                "Axis assessments marked 'internal assessment' are the project's own write-up, "
                "not independent literature citations, and are pending expert review.",
                "Entries with verified=false are asserted from memory and must not be used as "
                "positive controls until a citation is attached.",
                "Update probes are synthetic. They test evidence-following, not the absence of "
                "pretrained hindsight knowledge.",
                "Readiness measures evidence coverage of the rubric. It is not binding, "
                "enrichment, inhibition, antiviral activity or therapeutic efficacy."]})
        summary[name] = {"targets": len(rows), "evidence": len(entries), "probes": len(probes),
                         "claims_requiring_citation": len(uncited)}
    print(json.dumps({"output": str(out), "families": summary}, indent=2))
