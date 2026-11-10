"""Which viral protein each antiviral actually binds, graded, as labels for a retrieval benchmark.

**This is an overlay, not a copy.** `data/benchmark_panel.json` is the raw curation and stays
read-only: 47 genomes and 157 target patterns, each with the `validated_by` prose that justifies
it. This module adds the two things the panel deliberately does not carry -- a graded `status` per
target, and an alias set precise enough to match without a substring -- and it names the rows that
have to be dropped. Keeping them apart is what stops a weak label being silently upgraded: the
panel says what was curated, this says what each row is worth.

A pattern in the panel with no entry here raises rather than defaulting. A default status would be
a judgement nobody made, applied to the row least likely to deserve it.

**Four tiers, and the gap between the third and fourth is the one that matters.**

`approved` and `approved (EUA)` are the strong tier: an approved drug binds this protein in this
virus. Everything else is reported separately and never pooled into a headline.

`class_effect` and `vaccine_antigen` are weak but real -- a homologous protein is drugged in a
relative, or the protein is what a licensed vaccine or immunoglobulin binds. Real evidence of the
wrong kind, and `status` handles them.

`investigational` and `research target` are weaker still, and graded for the same reason.

**Mislabels are not a tier, they are a drop.** A row that is simply wrong grades to whatever tier
it is assigned and keeps scoring, so grading cannot catch it. `DROPPED` carries each one with its
reason, which makes the correction auditable and stops a later re-key quietly reinstating it.

**The label is the protein, never the drug.** A drug acting on a host factor says nothing about
whether a viral protein was worth ranking, so bulevirtide (NTCP), maraviroc (CCR5) and the
interferons are absent by construction -- and so is `DrugVirus.info` as a source, since a large
share of broad-spectrum antivirals act on the host.
"""

from __future__ import annotations

import json
import os

PANEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "data", "benchmark_panel.json")

# How much a status says about "this protein is worth binding". Supplied for a graded metric and
# used by no default, so the headline stays a plain rank-sum over the strong tier.
STATUS_WEIGHT = {
    "approved": 1.0,
    "approved (EUA)": 0.9,
    "investigational (compassionate use)": 0.6,
    "investigational": 0.5,
    "vaccine_antigen": 0.4,
    "class_effect": 0.3,
    "research target": 0.25,
}

# The tier a status belongs to. `strong` is the only one a default run scores.
STRONG = ("approved", "approved (EUA)")
WEAK = ("investigational (compassionate use)", "investigational", "vaccine_antigen",
        "class_effect", "research target")


# ---------------------------------------------------------------------------
# Aliases: what a pattern must match, as whole words
# ---------------------------------------------------------------------------

# Matching is on a contiguous run of whole tokens, not a substring, and these are the runs each
# pattern is allowed to match. Two shapes in the pipeline's own chain names force this:
#
#   * an affix the pattern does not carry -- `3C-like proteinase nsp5`, `NCAP_SEOUS Nucleoprotein`,
#     `RNA-directed RNA polymerase nsp12` -- so strict equality misses nearly everything;
#   * a longer name that merely starts the same -- `NS5A` and `NS5B` against `NS5`.
#
# Token-run matching handles both, and the second is the case substring matching gets wrong: `ns5`
# is inside the string `ns5a` but is not one of its tokens, so HCV's NS5A and NS5B stay distinct
# from a bare NS5 while a `3C-like proteinase` still finds its `nsp5` chain.
#
# A pattern absent here matches on its own words alone, which is right for anything already
# specific. Entries exist only where the panel's wording and UniProt's differ.
ALIASES = {
    "Spike": ("spike", "spike protein", "spike glycoprotein"),
    "3C-like proteinase": ("3c like proteinase", "3c like protease", "3cl protease", "mpro"),
    "RNA-directed RNA polymerase": ("rna directed rna polymerase", "rna dependent rna polymerase",
                                    "rdrp"),
    "Reverse transcriptase": ("reverse transcriptase", "rt"),
    "Integrase": ("integrase",),
    "Protease": ("protease", "proteinase"),
    "Serine protease": ("serine protease", "serine proteinase"),
    "DNA polymerase": ("dna polymerase", "dna directed dna polymerase"),
    "Thymidine kinase": ("thymidine kinase",),
    "terminase": ("terminase", "terminase large subunit", "tripartite terminase subunit"),
    "Protein kinase UL97": ("protein kinase ul97", "ul97", "serine threonine protein kinase ul97"),
    "Envelope": ("envelope", "envelope protein", "envelope glycoprotein"),
    "Envelope glycoprotein": ("envelope glycoprotein", "glycoprotein", "spike glycoprotein"),
    "Glycoprotein": ("glycoprotein", "glycoprotein precursor", "envelope glycoprotein"),
    "Glycoprotein G": ("glycoprotein g", "attachment glycoprotein g", "g glycoprotein"),
    "Fusion glycoprotein": ("fusion glycoprotein", "fusion protein", "fusion glycoprotein f0"),
    "Nucleoprotein": ("nucleoprotein", "nucleocapsid protein"),
    "Phosphoprotein": ("phosphoprotein",),
    "Hemagglutinin": ("hemagglutinin", "haemagglutinin"),
    "Hemagglutinin-neuraminidase": ("hemagglutinin neuraminidase", "haemagglutinin neuraminidase"),
    "Major surface glycoprotein": ("major surface glycoprotein", "attachment glycoprotein"),
    "Capsid protein VP1": ("capsid protein vp1", "vp1"),
    "Protease 3C": ("protease 3c", "3c protease", "protease 3c like"),
    "Hexon": ("hexon", "hexon protein"),
    "p37": ("p37", "palmitoylated eev membrane glycoprotein", "f13", "vp37"),
    "Replication protein E1": ("replication protein e1", "e1"),
    "Protein E6": ("protein e6", "e6"),
    "Protein E7": ("protein e7", "e7"),
    "Major capsid protein L1": ("major capsid protein l1", "l1"),
    "Non-structural protein 1": ("non structural protein 1", "nonstructural protein 1", "ns1"),
    "nsP2": ("nsp2", "protease nsp2", "non structural protein 2"),
    "nsP4": ("nsp4", "non structural protein 4"),
    "E2": ("e2", "e2 envelope glycoprotein", "spike glycoprotein e2"),
    "NS3": ("ns3", "serine protease ns3", "protease ns3"),
    "NS5": ("ns5",),
    "NS5A": ("ns5a",),
    "NS5B": ("ns5b",),
    # Influenza. These two sets are not new: they are the ones `EXTRA` already carried for PR8,
    # whose source field names `runs/target_ranks.csv` -- so they were read off what the ranker
    # actually named, which is the standard the override table below insists on. Moving them
    # here is what lets PR8 come through the panel like every other pathogen.
    "Neuraminidase": ("neuraminidase", "sialidase"),
    "Polymerase acidic": ("polymerase acidic protein", "pa endonuclease",
                          "cap snatching endonuclease", "polymerase acidic"),
    # `Matrix protein 2` is deliberately absent and falls back to its own lowercased name.
    # `EXTRA` never carried an M2 row, so there is no record of what the ranker names it, and
    # inventing `m2 protein` or `proton channel` here is precisely the move the override table
    # warns against: it would convert a protein the candidate stage may genuinely never propose
    # into a label that matches something. If M2 reports unmatched, that is the honest result
    # and the fix is to read a record and add the name it really uses.
}


# Aliases for one pathogen only, where a shared pattern would otherwise need a name that is
# meaningless or wrong in the other viruses using it.
#
# **Every entry here was found by running the matcher against the records and reading what the
# ranker actually named, not by guessing at synonyms.** That matters, because the alternative
# failure mode is the one being guarded against: a missing alias and a protein the candidate
# stage genuinely never proposed are indistinguishable in the output, both arriving as an
# unmatched label. Four of the eight strong-tier misses on the first pass were this -- a name
# gap, not an absence -- and four were real. Widening an alias until a label matches is how the
# second kind gets quietly converted into the first, so each of these names a chain present in a
# committed record.
ALIAS_OVERRIDE = {
    # The ranker names HIV's envelope by its two mature products, which is what UniProt calls
    # them. `gp120` and `gp41` would be meaningless in the flaviviruses and alphaviruses that
    # share the bare `Envelope` pattern, so they are scoped here.
    ("HIV-1", "Envelope"): ("surface protein gp120", "transmembrane protein gp41",
                            "gp120", "gp41", "envelope glycoprotein gp160"),
    # HBV's reverse transcriptase is a domain of protein P, and the record names the whole
    # entry. This is also why the panel's bare `polymerase` pattern had to go: the same protein
    # was named twice, once ambiguously.
    ("Hepatitis B virus", "Reverse transcriptase"): ("protein p", "reverse transcriptase",
                                                     "dna polymerase"),
    # UniProt spells HCV's NS5A out in full.
    ("Hepatitis C virus", "NS5A"): ("ns5a", "non structural protein 5a"),
    # And names NS5B by its function, which is the chain the dropped duplicate pattern was
    # pointing at.
    ("Hepatitis C virus", "NS5B"): ("ns5b", "non structural protein 5b",
                                    "rna directed rna polymerase"),
}


def aliases_for(pathogen, pattern):
    """The alias set for one panel row, per-pathogen override first."""
    if (pathogen, pattern) in ALIAS_OVERRIDE:
        return ALIAS_OVERRIDE[(pathogen, pattern)]
    return ALIASES.get(pattern, (pattern.lower(),))


# ---------------------------------------------------------------------------
# Status, per (pathogen, pattern)
# ---------------------------------------------------------------------------

# Derived by reading each pathogen's `validated_by` prose against its own target list, which is
# the only place the panel records which drug answers for which protein. The derivation is
# recorded per row rather than applied as a rule, because the prose does not say which target it
# refers to and a rule would have to guess: "pocapavir targets VP1; 3C is the class target" grades
# two of a pathogen's three patterns differently in one sentence.
#
# Read the second element as the justification, not as a restatement. Where it disagrees with
# `validated_by` it says so -- several rows name an approved drug the panel's prose omitted.
OVERLAY = {
    # -- Coronaviridae. One pathogen carries the drugs; the other four are class effect, which
    # is exactly the non-independence the family grouping exists to handle.
    ("SARS-CoV-2", "3C-like proteinase"): ("approved", "nirmatrelvir binds Mpro"),
    ("SARS-CoV-2", "RNA-directed RNA polymerase"): ("approved", "remdesivir binds nsp12"),
    ("SARS-CoV-2", "Spike"): ("approved (EUA)",
                              "sotrovimab and tixagevimab-cilgavimab bind spike under EUA"),
    ("SARS-CoV", "3C-like proteinase"): ("class_effect", "panel: class effect with SARS-CoV-2"),
    ("SARS-CoV", "RNA-directed RNA polymerase"): ("class_effect", "panel: class effect"),
    ("SARS-CoV", "Spike"): ("class_effect", "panel: class effect"),
    ("MERS-CoV", "3C-like proteinase"): ("class_effect", "panel: class effect"),
    ("MERS-CoV", "RNA-directed RNA polymerase"): ("class_effect", "panel: class effect"),
    ("MERS-CoV", "Spike"): ("investigational", "panel: mAbs in trials, none approved"),
    ("HCoV-229E", "3C-like proteinase"): ("class_effect", "panel: class effect"),
    ("HCoV-229E", "RNA-directed RNA polymerase"): ("class_effect", "panel: class effect"),
    ("HCoV-229E", "Spike"): ("class_effect", "panel: class effect"),
    ("HCoV-OC43", "3C-like proteinase"): ("class_effect", "panel: class effect"),
    ("HCoV-OC43", "RNA-directed RNA polymerase"): ("class_effect", "panel: class effect"),
    ("HCoV-OC43", "Spike"): ("class_effect", "panel: class effect"),

    # -- Retroviridae.
    ("HIV-1", "Reverse transcriptase"): ("approved", "zidovudine and tenofovir bind RT"),
    ("HIV-1", "Protease"): ("approved", "ritonavir and darunavir bind the protease"),
    ("HIV-1", "Integrase"): ("approved", "raltegravir and dolutegravir bind integrase"),
    ("HIV-1", "Envelope"): ("approved",
                            "enfuvirtide binds gp41 and fostemsavir binds gp120; both approved "
                            "and both omitted by validated_by. Not maraviroc, which binds host "
                            "CCR5 and is correctly absent from this panel"),
    ("HIV-2", "Reverse transcriptase"): ("class_effect", "panel: class effect with HIV-1"),
    ("HIV-2", "Protease"): ("class_effect", "panel: class effect"),
    ("HIV-2", "Integrase"): ("class_effect", "panel: class effect"),

    # -- Hepadnaviridae.
    ("Hepatitis B virus", "Reverse transcriptase"): (
        "approved", "tenofovir and entecavir bind the RT domain of protein P. Bulevirtide binds "
                    "host NTCP and is correctly absent"),

    # -- Flaviviridae, Hepacivirus.
    ("Hepatitis C virus", "NS3"): ("approved", "glecaprevir and grazoprevir bind NS3/4A"),
    ("Hepatitis C virus", "NS5A"): ("approved", "ledipasvir and velpatasvir bind NS5A"),
    ("Hepatitis C virus", "NS5B"): ("approved", "sofosbuvir and dasabuvir bind NS5B"),

    # -- Herpesviridae. Thymidine kinase is the interesting row: acyclovir depends on it but does
    # not inhibit it, so it is an activating enzyme and not a target.
    ("Herpes simplex virus 1", "DNA polymerase"): ("approved",
                                                   "acyclovir and foscarnet bind UL30"),
    ("Herpes simplex virus 1", "Thymidine kinase"): (
        "research target", "UL23 phosphorylates acyclovir and is not inhibited by it. An "
                           "activating enzyme is not a binding target, so this is the weakest "
                           "tier rather than approved"),
    ("Herpes simplex virus 2", "DNA polymerase"): ("approved", "acyclovir binds the DNA pol"),
    ("Herpes simplex virus 2", "Thymidine kinase"): ("research target",
                                                     "activating enzyme, not inhibited"),
    ("Varicella-zoster virus", "DNA polymerase"): ("approved",
                                                   "acyclovir and brivudine bind ORF28"),
    ("Varicella-zoster virus", "Thymidine kinase"): ("research target",
                                                     "activating enzyme, not inhibited"),
    ("Human cytomegalovirus", "DNA polymerase"): ("approved",
                                                  "ganciclovir and foscarnet bind UL54"),
    ("Human cytomegalovirus", "terminase"): ("approved", "letermovir binds the UL56 subunit"),
    ("Human cytomegalovirus", "Protein kinase UL97"): (
        "approved", "maribavir binds UL97; approved and omitted by validated_by"),
    ("Epstein-Barr virus", "DNA polymerase"): ("class_effect", "panel: class effect"),
    ("Epstein-Barr virus", "Thymidine kinase"): ("class_effect", "panel: class effect"),

    # -- Flaviviridae, Orthoflavivirus. Seven genomes, no approved antiviral among them: every
    # row is a class target or a vaccine antigen, which is why pooling them with the HCV rows
    # would be the single most misleading thing this overlay could do.
    ("Zika virus", "NS3"): ("class_effect", "panel: flavivirus class target"),
    ("Zika virus", "NS5"): ("class_effect", "panel: flavivirus class target"),
    ("Zika virus", "Envelope"): ("vaccine_antigen", "panel: vaccines target E"),
    ("Dengue virus 2", "NS3"): ("class_effect", "panel: flavivirus class target"),
    ("Dengue virus 2", "NS5"): ("investigational", "panel: NS5 inhibitors in trials"),
    ("Dengue virus 2", "Envelope"): ("vaccine_antigen", "panel: dengue vaccine targets E"),
    ("West Nile virus", "NS3"): ("class_effect", "panel: flavivirus class target"),
    ("West Nile virus", "NS5"): ("class_effect", "panel: flavivirus class target"),
    ("West Nile virus", "Envelope"): ("class_effect", "panel: class target, no vaccine named"),
    ("Yellow fever virus", "NS3"): ("class_effect", "panel: flavivirus class target"),
    ("Yellow fever virus", "NS5"): ("class_effect", "panel: flavivirus class target"),
    ("Yellow fever virus", "Envelope"): ("vaccine_antigen", "panel: 17D vaccine targets E"),
    ("Japanese encephalitis virus", "NS3"): ("class_effect", "panel: flavivirus class target"),
    ("Japanese encephalitis virus", "NS5"): ("class_effect", "panel: flavivirus class target"),
    ("Japanese encephalitis virus", "Envelope"): ("vaccine_antigen", "panel: vaccine targets E"),
    ("Tick-borne encephalitis virus", "NS3"): ("class_effect", "panel: class target"),
    ("Tick-borne encephalitis virus", "NS5"): ("class_effect", "panel: class target"),
    ("Tick-borne encephalitis virus", "Envelope"): ("vaccine_antigen",
                                                    "panel: vaccine targets E"),
    ("Powassan virus", "NS3"): ("class_effect", "panel: flavivirus class target"),
    ("Powassan virus", "NS5"): ("class_effect", "panel: flavivirus class target"),
    ("Powassan virus", "Envelope"): ("class_effect", "panel: class target, no vaccine named"),

    # -- Paramyxoviridae.
    ("Nipah virus", "Fusion glycoprotein"): ("research target",
                                             "no clinical agent binds F; the mAb targets G"),
    ("Nipah virus", "Glycoprotein G"): ("investigational (compassionate use)",
                                        "panel: m102.4 mAb targets G"),
    ("Nipah virus", "RNA-directed RNA polymerase"): ("investigational",
                                                     "panel: remdesivir active, not approved "
                                                     "for Nipah"),
    ("Nipah virus", "Phosphoprotein"): ("research target", "no agent named for P"),
    ("Hendra virus", "Fusion glycoprotein"): ("research target", "no agent binds F"),
    ("Hendra virus", "Glycoprotein G"): ("investigational (compassionate use)",
                                         "panel: m102.4 mAb"),
    ("Hendra virus", "RNA-directed RNA polymerase"): ("class_effect",
                                                      "class effect with Nipah"),
    ("Measles virus", "RNA-directed RNA polymerase"): ("class_effect", "paramyxovirus L class"),
    ("Measles virus", "Fusion glycoprotein"): ("vaccine_antigen", "panel: vaccine targets F"),
    ("Measles virus", "Hemagglutinin"): ("vaccine_antigen", "panel: vaccine targets H"),
    ("Mumps virus", "RNA-directed RNA polymerase"): ("class_effect", "paramyxovirus L class"),
    ("Mumps virus", "Fusion glycoprotein"): ("vaccine_antigen", "panel: vaccine targets F"),
    ("Mumps virus", "Hemagglutinin-neuraminidase"): ("vaccine_antigen",
                                                     "panel: vaccine targets HN"),

    # -- Pneumoviridae.
    ("Human respiratory syncytial virus B", "Fusion glycoprotein"): (
        "approved", "nirsevimab and palivizumab are approved anti-F monoclonals"),
    ("Human respiratory syncytial virus B", "RNA-directed RNA polymerase"): (
        "investigational", "RSV polymerase inhibitors have reached trials; none approved"),
    ("Human respiratory syncytial virus B", "Major surface glycoprotein"): (
        "research target", "no clinical agent binds G"),
    ("Human metapneumovirus", "Fusion glycoprotein"): ("class_effect",
                                                       "panel: F is the class target"),
    ("Human metapneumovirus", "RNA-directed RNA polymerase"): ("class_effect",
                                                               "pneumovirus L class"),

    # -- Filoviridae.
    ("Ebola virus Zaire", "RNA-directed RNA polymerase"): (
        "investigational", "panel: remdesivir targets L. Approved for COVID-19 and not for "
                           "Ebola, so this must not be upgraded"),
    ("Ebola virus Zaire", "Envelope glycoprotein"): (
        "approved", "ansuvimab and the atoltivimab-maftivimab-odesivimab combination are "
                    "approved anti-GP monoclonals"),
    ("Ebola virus Zaire", "Nucleoprotein"): ("research target", "no agent binds NP"),
    ("Marburg virus", "RNA-directed RNA polymerase"): ("class_effect",
                                                       "panel: class effect with Ebola"),
    ("Marburg virus", "Envelope glycoprotein"): ("class_effect", "panel: class effect"),

    # -- The cross-family RdRp / Glycoprotein / Nucleoprotein group. Five unrelated viruses with
    # one pattern list between them, and three of the five justified by ribavirin, whose protein
    # target in these viruses is not established.
    ("Rabies virus", "RNA-directed RNA polymerase"): ("research target", "no agent binds L"),
    ("Rabies virus", "Glycoprotein"): ("vaccine_antigen",
                                       "panel: rabies immunoglobulin targets G"),
    ("Rabies virus", "Nucleoprotein"): ("research target", "no agent binds N"),
    ("Lassa virus", "Glycoprotein"): ("vaccine_antigen", "panel: LASV GPC is the vaccine target"),
    ("Lassa virus", "Nucleoprotein"): ("research target",
                                       "ribavirin has no established protein target in LASV"),
    ("Hantaan virus", "Nucleoprotein"): ("research target",
                                         "ribavirin has no established protein target"),
    ("Crimean-Congo hemorrhagic fever virus", "RNA-directed RNA polymerase"): (
        "investigational", "panel: favipiravir in trials"),
    ("Rift Valley fever virus", "RNA-directed RNA polymerase"): (
        "investigational", "panel: favipiravir active in models -- preclinical only"),
    ("Rift Valley fever virus", "Glycoprotein"): ("research target", "no agent named"),
    ("Rift Valley fever virus", "Nucleoprotein"): ("research target", "no agent named"),

    # -- Togaviridae.
    ("Chikungunya virus", "nsP2"): ("class_effect",
                                    "panel: nsP2 protease is the class target"),
    ("Chikungunya virus", "nsP4"): ("class_effect", "alphavirus polymerase class target"),
    ("Chikungunya virus", "Envelope"): ("vaccine_antigen", "panel: IXCHIQ vaccine targets E"),
    ("Venezuelan equine encephalitis virus", "nsP2"): ("class_effect",
                                                       "panel: alphavirus class target"),
    ("Venezuelan equine encephalitis virus", "nsP4"): ("class_effect", "panel: class target"),
    ("Venezuelan equine encephalitis virus", "Envelope"): ("class_effect",
                                                           "panel: class target"),
    ("Sindbis virus", "nsP2"): ("class_effect", "panel: alphavirus class target"),
    ("Sindbis virus", "nsP4"): ("class_effect", "panel: class target"),
    ("Sindbis virus", "Envelope"): ("class_effect", "panel: class target"),

    # -- Picornaviridae. Rhinovirus is the one row where two real binders both failed to be
    # approved, which is a different claim from a class effect and is graded as such.
    ("Poliovirus", "Protease 3C"): ("class_effect", "panel: 3C is the class target"),
    ("Poliovirus", "RNA-directed RNA polymerase"): ("class_effect", "picornavirus 3D class"),
    ("Poliovirus", "Capsid protein VP1"): ("investigational",
                                           "panel: pocapavir targets VP1; never approved"),
    ("Enterovirus A71", "Protease 3C"): ("class_effect", "panel: picornavirus class target"),
    ("Enterovirus A71", "RNA-directed RNA polymerase"): ("class_effect", "panel: class target"),
    ("Enterovirus A71", "Capsid protein VP1"): ("class_effect", "panel: class target"),
    ("Rhinovirus A", "Protease 3C"): ("investigational",
                                      "panel: rupintrivir targets 3C; never approved"),
    ("Rhinovirus A", "RNA-directed RNA polymerase"): ("class_effect", "picornavirus 3D class"),
    ("Rhinovirus A", "Capsid protein VP1"): ("investigational",
                                             "panel: pleconaril targets VP1; never approved"),

    # -- Caliciviridae.
    ("Norwalk virus", "Protease"): ("class_effect", "panel: calicivirus class target"),
    ("Norwalk virus", "RNA-directed RNA polymerase"): ("class_effect", "panel: class target"),
    ("Norwalk virus", "Capsid protein VP1"): ("class_effect", "panel: class target"),

    # -- Adenoviridae.
    ("Human adenovirus 2", "DNA polymerase"): ("approved",
                                               "panel: cidofovir targets the DNA polymerase"),
    ("Human adenovirus 2", "Protease"): ("research target", "no agent binds adenain"),
    ("Human adenovirus 2", "Hexon"): ("research target", "no agent binds hexon"),

    # -- Poxviridae.
    ("Monkeypox virus", "DNA polymerase"): ("approved", "panel: cidofovir targets DNA pol"),
    ("Monkeypox virus", "Thymidine kinase"): ("research target",
                                              "no agent binds the poxvirus TK"),
    ("Monkeypox virus", "p37"): ("approved", "panel: tecovirimat targets VP37/F13"),
    ("Vaccinia virus", "DNA polymerase"): ("approved", "panel: cidofovir"),
    ("Vaccinia virus", "Thymidine kinase"): ("research target", "no agent binds TK"),
    ("Vaccinia virus", "p37"): ("approved", "panel: tecovirimat"),

    # -- Papillomaviridae. No antiviral binds any of these; L1 is a vaccine antigen and E6/E7 are
    # oncoproteins studied as targets.
    ("Human papillomavirus 16", "Replication protein E1"): ("research target",
                                                            "no clinical agent binds E1"),
    ("Human papillomavirus 16", "Protein E6"): ("research target",
                                                "panel: E6/E7 are the oncoprotein targets"),
    ("Human papillomavirus 16", "Protein E7"): ("research target", "panel: oncoprotein target"),
    ("Human papillomavirus 16", "Major capsid protein L1"): ("vaccine_antigen",
                                                             "panel: Gardasil targets L1"),

    # -- Parvoviridae.
    ("Human parvovirus B19", "Non-structural protein 1"): ("class_effect",
                                                           "panel: NS1 helicase is the class "
                                                           "target"),
    ("Human parvovirus B19", "Capsid protein VP1"): ("research target", "no agent binds VP1"),

    # -- Orthomyxoviridae. These three rows are why `pathogens()` raised rather than ran: the
    # panel gained an `Influenza A PR8` entry while PR8 was already being supplied through
    # `EXTRA`, and nothing reconciled the two. Grading them here rather than dropping them is
    # the right half of the fix because the panel's own `validated_by` names a drug for each of
    # the three, which is exactly the evidence this table is derived from. The other half is in
    # `EXTRA` and in `pathogens()`, which previously appended PR8 a second time.
    ("Influenza A PR8", "Neuraminidase"): ("approved",
                                           "oseltamivir and zanamivir bind neuraminidase"),
    # Amantadine and rimantadine are approved and bind the M2 proton channel, which is what
    # this status asserts. They are no longer recommended against circulating strains, because
    # S31N is near-universal in H3N2 and H1N1pdm09 -- but PR8 is a 1934 laboratory strain
    # without it, so the drug does bind this protein in this virus. The caveat is recorded
    # because a reader checking the status against current treatment guidance will not find it.
    ("Influenza A PR8", "Matrix protein 2"): ("approved",
                                              "amantadine binds the M2 ion channel; resistance "
                                              "is near-universal in circulating strains but not "
                                              "in PR8"),
    ("Influenza A PR8", "Polymerase acidic"): ("approved",
                                               "baloxavir marboxil binds the PA endonuclease"),
}


# ---------------------------------------------------------------------------
# Dropped rows
# ---------------------------------------------------------------------------

# Three kinds, and the first is the one grading could never have caught.
#
#   `mislabel`  -- wrong, full stop. Would score at whatever tier it was given.
#   `duplicate` -- a second pattern for a protein another pattern already names, so one protein
#                  is counted twice. This is the same defect as chain-level counting, arriving
#                  through the label list instead of through `decompose`.
#   `parent`    -- the polyprotein containing targets already named individually, which would
#                  credit a hit three times over.
#
# Every drop is recorded rather than deleted, so a later re-key cannot silently reinstate it and
# a reader can disagree with a specific decision instead of with the whole set.
DROPPED = {
    ("Herpes simplex virus 1", "terminase"): (
        "mislabel", "letermovir binds the HCMV UL56 terminase and HSV-1 has no approved "
                    "terminase inhibitor. Pattern-matched across from this panel's own HCMV row"),
    ("Hepatitis B virus", "Large envelope"): (
        "mislabel", "a vaccine antigen in a row whose validated_by names tenofovir and "
                    "entecavir, neither of which binds it"),
    ("Hepatitis B virus", "surface antigen"): (
        "mislabel", "HBsAg is a vaccine antigen, not a target of the drugs this row cites"),
    ("Hepatitis B virus", "polymerase"): (
        "duplicate", "the same protein P that `Reverse transcriptase` already names, and a bare "
                     "token that would also match `RNA-directed RNA polymerase` in any virus"),
    ("Hepatitis C virus", "Serine protease"): (
        "duplicate", "NS3 is the serine protease; both patterns name one protein"),
    ("Hepatitis C virus", "RNA-directed RNA polymerase"): (
        "duplicate", "NS5B is the polymerase; both patterns name one protein"),
    ("HIV-1", "Gag-Pol"): (
        "parent", "the polyprotein containing RT, protease and integrase, all named separately. "
                  "Counting it as well credits one hit three times"),
    ("HIV-2", "Gag-Pol"): ("parent", "as HIV-1"),
    ("Zika virus", "Serine protease"): ("duplicate", "NS3 is the serine protease"),
    ("Zika virus", "RNA-directed RNA polymerase"): ("duplicate", "NS5 is the polymerase"),
    ("Dengue virus 2", "Serine protease"): ("duplicate", "NS3 is the serine protease"),
    ("Dengue virus 2", "RNA-directed RNA polymerase"): ("duplicate", "NS5 is the polymerase"),
    ("West Nile virus", "Serine protease"): ("duplicate", "NS3 is the serine protease"),
    ("Yellow fever virus", "Serine protease"): ("duplicate", "NS3 is the serine protease"),
    ("Chikungunya virus", "Protease nsP2"): ("duplicate", "nsP2 is the protease"),
    ("Chikungunya virus", "RNA-directed RNA polymerase"): ("duplicate", "nsP4 is the polymerase"),
    ("Chikungunya virus", "E2"): (
        "duplicate", "E2 is part of the envelope glycoprotein `Envelope` already names"),
    ("Venezuelan equine encephalitis virus", "E2"): ("duplicate", "part of Envelope"),
    ("Sindbis virus", "E2"): ("duplicate", "part of Envelope"),
    ("Poliovirus", "Genome polyprotein"): (
        "parent", "the polyprotein containing 3C, 3D and VP1, all named separately"),
    ("Monkeypox virus", "Palmitoylated EEV membrane"): ("duplicate", "the same protein as p37"),
    ("Vaccinia virus", "Palmitoylated EEV membrane"): ("duplicate", "the same protein as p37"),
    ("Lassa virus", "RNA-directed RNA polymerase"): (
        "absent_from_genome", "the L polymerase is encoded on the L segment and the panel's "
                              "accession NC_004296.1 is the S segment, so this protein is not "
                              "in the genome handed to the ranker"),
    ("Hantaan virus", "RNA-directed RNA polymerase"): (
        "absent_from_genome", "L segment product; NC_005218.1 is the S segment"),
    ("Hantaan virus", "Glycoprotein"): (
        "absent_from_genome", "M segment product; NC_005218.1 is the S segment"),
    ("Crimean-Congo hemorrhagic fever virus", "Glycoprotein"): (
        "absent_from_genome", "a different segment from NC_005300.2, whose identified chains are "
                              "all polymerase-complex products"),
    ("Crimean-Congo hemorrhagic fever virus", "Nucleoprotein"): (
        "absent_from_genome", "a different segment from NC_005300.2"),
}


# Instances that cannot be scored, with the reason, kept out of every aggregate. An instance with
# fewer candidates than labels cannot be informative and one with no candidates has no score
# defined at all, so counting either as a miss would make the ranker answer for the panel's
# accession choice.
UNSCOREABLE = {
    "Human parvovirus B19": "the ranker identified zero candidate chains from NC_000883.2, so "
                            "there is no ranking to score",
    "Hantaan virus": "NC_005218.1 is one segment of three and yields a single candidate chain "
                     "against a single surviving label, so rank 1 of 1 is guaranteed whatever "
                     "the ranker does",
}

# Pathogens whose panel accession is one segment of a multi-segment genome. Recorded because it
# explains the `absent_from_genome` drops above and because supplying every segment accession,
# comma-joined, turns these back into real instances -- `fetch_genome` already accepts that form
# and returns the segments separately rather than concatenating them, which is correct: joining
# would create reading frames across junctions that exist in no molecule.
SEGMENTED = {
    "Lassa virus": "bi-segmented; NC_004296.1 is S, the L polymerase is on L",
    "Hantaan virus": "tri-segmented; NC_005218.1 is S, with G on M and L on L",
    "Crimean-Congo hemorrhagic fever virus": "tri-segmented; NC_005300.2 carries the "
                                             "polymerase-complex products only",
    "Rift Valley fever virus": "tri-segmented; verify NC_014397.1 covers the labelled proteins",
}


# Viral families, for aggregating over independent instances rather than over genomes. 26 of the
# 47 genomes share a pattern list with at least one sibling, so a p-value combined across 47 as
# though they were independent is inflated. Combined within family first, then across families.
#
# The denominator was 46 until the panel gained its `Influenza A PR8` row. PR8's pattern list is
# unique, so the 26 is unchanged and only the total moved -- recounted rather than assumed, since
# a new genome sharing a list with a sibling would have moved both.
FAMILY = {
    "SARS-CoV-2": "Coronaviridae", "SARS-CoV": "Coronaviridae", "MERS-CoV": "Coronaviridae",
    "HCoV-229E": "Coronaviridae", "HCoV-OC43": "Coronaviridae",
    "HIV-1": "Retroviridae", "HIV-2": "Retroviridae",
    "Hepatitis B virus": "Hepadnaviridae",
    "Hepatitis C virus": "Flaviviridae/Hepacivirus",
    "Herpes simplex virus 1": "Herpesviridae", "Herpes simplex virus 2": "Herpesviridae",
    "Varicella-zoster virus": "Herpesviridae", "Human cytomegalovirus": "Herpesviridae",
    "Epstein-Barr virus": "Herpesviridae",
    "Zika virus": "Flaviviridae/Orthoflavivirus",
    "Dengue virus 2": "Flaviviridae/Orthoflavivirus",
    "West Nile virus": "Flaviviridae/Orthoflavivirus",
    "Yellow fever virus": "Flaviviridae/Orthoflavivirus",
    "Japanese encephalitis virus": "Flaviviridae/Orthoflavivirus",
    "Tick-borne encephalitis virus": "Flaviviridae/Orthoflavivirus",
    "Powassan virus": "Flaviviridae/Orthoflavivirus",
    "Nipah virus": "Paramyxoviridae", "Hendra virus": "Paramyxoviridae",
    "Measles virus": "Paramyxoviridae", "Mumps virus": "Paramyxoviridae",
    "Human respiratory syncytial virus B": "Pneumoviridae",
    "Human metapneumovirus": "Pneumoviridae",
    "Ebola virus Zaire": "Filoviridae", "Marburg virus": "Filoviridae",
    "Rabies virus": "Rhabdoviridae", "Lassa virus": "Arenaviridae",
    "Hantaan virus": "Hantaviridae",
    "Crimean-Congo hemorrhagic fever virus": "Nairoviridae",
    "Rift Valley fever virus": "Phenuiviridae",
    "Chikungunya virus": "Togaviridae",
    "Venezuelan equine encephalitis virus": "Togaviridae", "Sindbis virus": "Togaviridae",
    "Poliovirus": "Picornaviridae", "Enterovirus A71": "Picornaviridae",
    "Rhinovirus A": "Picornaviridae",
    "Norwalk virus": "Caliciviridae", "Human adenovirus 2": "Adenoviridae",
    "Monkeypox virus": "Poxviridae", "Vaccinia virus": "Poxviridae",
    "Human papillomavirus 16": "Papillomaviridae",
    "Human parvovirus B19": "Parvoviridae",
    "Influenza A PR8": "Orthomyxoviridae",
}


# Targets the panel does not carry, merged into the panel's own row for that pathogen.
#
# **This used to be a whole second PR8 and that was a double count.** The comment here read
# "one pathogen the panel does not carry", which stopped being true when the panel gained an
# `Influenza A PR8` entry of its own. Nothing reconciled them: `pathogens()` appended this
# block after walking the panel, so PR8 came back twice -- once ungraded from the panel, which
# is the raise, and once from here. Two of the three targets were the same proteins under
# different names (`Neuraminidase` / `Neuraminidase (NA)`, `Polymerase acidic` /
# `PA endonuclease`), so every rank PR8 contributed was counted twice, which is the same defect
# `DROPPED` records as `duplicate` arriving through a different door. It matters beyond a
# headline count because the families are combined across pathogens as independent instances,
# and one pathogen entered as two is not independent of itself.
#
# What is left is the single target the panel genuinely lacks. HA is kept rather than dropped
# because it is real evidence of the weak kind -- the panel's PR8 row names neither it nor any
# antibody programme -- and because losing it would quietly shrink the weak tier while this
# change was advertised as a de-duplication.
EXTRA = (
    {
        "pathogen": "Influenza A PR8",
        "accession": ("NC_002016.1,NC_002017.1,NC_002018.1,NC_002019.1,NC_002020.1,"
                      "NC_002021.1,NC_002022.1,NC_002023.1"),
        "holdout": "Influenza A virus (strain A/Puerto Rico/8/1934 H1N1)",
        "targets": (
            {"name": "Haemagglutinin (HA)", "aliases": ("hemagglutinin", "haemagglutinin"),
             "status": "investigational",
             "why": "broadly neutralising antibodies have reached trials; none approved",
             "source": "runs/target_ranks.csv"},
        ),
    },
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _panel():
    with open(PANEL_PATH) as handle:
        return json.load(handle)["panel"]


def pathogens(tier="strong", include_unscoreable=False, panel=None):
    """The label set, one row per pathogen, graded and with the dropped rows removed.

    `tier` is `"strong"` (approved only, the default and the only one a headline may quote),
    `"weak"` (everything else), or `"all"`. A pathogen left with no target in the requested tier
    disappears rather than appearing empty, because it contributes no ranks and would only add a
    row that cannot be scored.

    Raises on a panel pattern this module has not graded. That is deliberate: the panel is
    curated elsewhere and may gain a row, and a new row defaulting to some status would be a
    judgement nobody made applied to the row least likely to deserve it.
    """
    wanted = {"strong": set(STRONG), "weak": set(WEAK),
              "all": set(STRONG) | set(WEAK)}[tier]
    out = []
    for entry in (panel if panel is not None else _panel()):
        name = entry["pathogen"]
        if name in UNSCOREABLE and not include_unscoreable:
            continue
        targets = []
        for pattern in entry["targets"]:
            key = (name, pattern)
            if key in DROPPED:
                continue
            if key not in OVERLAY:
                raise ValueError(
                    f"panel row ({name!r}, {pattern!r}) has no status in task1_labels.OVERLAY "
                    f"and no entry in DROPPED. Grade it or drop it; it must not default.")
            status, why = OVERLAY[key]
            if status not in wanted:
                continue
            targets.append({
                "name": pattern,
                "aliases": aliases_for(name, pattern),
                "status": status, "why": why, "verified": True,
                "source": f"data/benchmark_panel.json, validated_by="
                          f"{entry.get('validated_by')!r}",
            })
        if targets:
            out.append({"pathogen": name, "family": FAMILY.get(name, "unassigned"),
                        "accession": entry["accession"], "holdout": entry["holdout"],
                        "targets": tuple(targets),
                        "segmented": SEGMENTED.get(name),
                        "unscoreable": UNSCOREABLE.get(name)})

    # Merged into the panel's row for the same pathogen, never appended beside it. Appending is
    # what produced two `Influenza A PR8` rows and counted its ranks twice; see the comment on
    # `EXTRA`. A collision raises instead of resolving itself, for the same reason an ungraded
    # pattern raises above: the two candidate resolutions -- keep the panel's label or keep this
    # one -- are a judgement about which curation is better, and this module's rule is that such
    # a judgement is written down by hand or not made at all.
    by_name = {row["pathogen"]: row for row in out}
    for entry in EXTRA:
        targets = tuple(t for t in entry["targets"] if t["status"] in wanted)
        if not targets:
            continue
        existing = by_name.get(entry["pathogen"])
        if existing is None:
            row = {**entry, "family": FAMILY.get(entry["pathogen"], "unassigned"),
                   "targets": targets, "segmented": None, "unscoreable": None}
            out.append(row)
            by_name[entry["pathogen"]] = row
            continue
        taken = {alias for target in existing["targets"] for alias in target["aliases"]}
        for target in targets:
            clash = sorted(set(target["aliases"]) & taken)
            if clash:
                raise ValueError(
                    f"EXTRA target {target['name']!r} for {entry['pathogen']!r} shares the "
                    f"alias(es) {clash} with a graded panel target, so merging it would count "
                    f"one protein twice. Drop it from EXTRA or narrow its aliases.")
        existing["targets"] = existing["targets"] + targets
    return tuple(out)


def counts(panel=None):
    """What the overlay says about the panel, which is the first thing to report."""
    rows = panel if panel is not None else _panel()
    graded = {}
    for entry in rows:
        for pattern in entry["targets"]:
            key = (entry["pathogen"], pattern)
            if key in DROPPED:
                graded.setdefault("dropped", []).append(DROPPED[key][0])
            elif key in OVERLAY:
                graded.setdefault(OVERLAY[key][0], []).append(key)
    by_status = {k: len(v) for k, v in sorted(graded.items()) if k != "dropped"}
    drops = {}
    for kind in graded.get("dropped", []):
        drops[kind] = drops.get(kind, 0) + 1
    strong = pathogens(tier="strong", panel=rows)
    return {
        "panel_genomes": len(rows),
        "panel_patterns": sum(len(e["targets"]) for e in rows),
        "graded_by_status": by_status,
        "dropped_by_kind": drops,
        "strong_labels": sum(len(e["targets"]) for e in strong),
        "strong_pathogens": len(strong),
        "families": len({e["family"] for e in strong}),
        "unscoreable": sorted(UNSCOREABLE),
        "segmented": sorted(SEGMENTED),
    }
