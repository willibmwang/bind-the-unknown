"""Which viral protein each antiviral actually binds, as labels for a retrieval benchmark.

**The label is the protein, never the drug.** A drug that works by binding a *host* factor says
nothing about whether a viral protein was worth ranking, so bulevirtide (NTCP), maraviroc (CCR5)
and the interferons are deliberately absent, and so is every broad-spectrum repurposing hit whose
mechanism runs through host biology. This is also why `DrugVirus.info` is not the source: it
catalogues compounds with broad antiviral activity, and a large share of those act on the host.

**Two tiers, and the distinction is load-bearing.** `verified=True` rows are the eleven already
recorded in `runs/target_ranks.csv`, carried over unchanged so the benchmark and the existing
experiment speak about the same labels. `verified=False` rows are drafted from standard antiviral
pharmacology and have *not* been checked against a primary source in this repository. They are
kept separate, excluded by default from any headline number, and `--include-draft` is the only
way to score them.

That split is not bureaucracy. A benchmark whose labels were written from recollection and
reported as curated would be a fabrication channel pointed at the result, which is the failure
this project names explicitly. Each draft row carries the check that would promote it.

**Status is graded, not binary.** An approved drug and a research-grade chemical probe are
different evidence that a protein is worth binding, and collapsing them would let a weak label
carry a strong claim. `STATUS_WEIGHT` is supplied for anyone who wants a graded metric; the
default metric uses the unweighted rank-sum, because weighting is a modelling choice and the
unweighted test is the one `intake.rank_enrichment` already implements and powers.
"""

from __future__ import annotations

# How much a status says about "this protein is worth binding". Supplied for a graded metric and
# used by no default, so the headline number stays a plain rank-sum.
STATUS_WEIGHT = {
    "approved": 1.0,
    "approved (EUA)": 0.9,
    "investigational (compassionate use)": 0.6,
    "investigational": 0.5,
    "research target": 0.25,
}

# A note on genome size, because it changes what the metric means rather than merely how long a
# run takes. Six-frame ORF calling on a 236 kb herpesvirus proposes far more candidates than on a
# 3.2 kb hepadnavirus, so placing three targets highly is a much harder test on the former. The
# denominator rule in `intake.rank_enrichment` -- candidates the pipeline identified and ranked --
# is applied identically at every size, which is what keeps the per-pathogen p-values comparable.
PATHOGENS = (
    # -- verified: carried over from runs/target_ranks.csv, unchanged ----------------------
    {
        "pathogen": "SARS-CoV-2",
        "family": "Coronaviridae",
        "accession": "NC_045512.2",
        "holdout": "severe acute respiratory syndrome coronavirus 2",
        "genome_nt": 29903,
        "targets": (
            {"name": "Spike glycoprotein",
             "aliases": ("spike", "surface glycoprotein", "s protein", "spike protein"),
             "drug": "sotrovimab / tixagevimab-cilgavimab", "status": "approved (EUA)",
             "verified": True, "source": "runs/target_ranks.csv"},
            {"name": "Mpro / 3CLpro (nsp5)",
             "aliases": ("3c-like proteinase", "3clpro", "mpro", "main protease", "nsp5",
                         "replicase polyprotein 1ab"),
             "drug": "nirmatrelvir (Paxlovid)", "status": "approved",
             "verified": True, "source": "runs/target_ranks.csv"},
            {"name": "RdRp (nsp12)",
             "aliases": ("rna-directed rna polymerase", "rdrp", "nsp12",
                         "replicase polyprotein 1ab"),
             "drug": "remdesivir", "status": "approved",
             "verified": True, "source": "runs/target_ranks.csv"},
            {"name": "PLpro (nsp3)",
             "aliases": ("papain-like protease", "plpro", "nsp3",
                         "replicase polyprotein 1ab"),
             "drug": "none approved", "status": "research target",
             "verified": True, "source": "runs/target_ranks.csv"},
            {"name": "nsp13 helicase",
             "aliases": ("helicase", "nsp13", "replicase polyprotein 1ab"),
             "drug": "none approved", "status": "research target",
             "verified": True, "source": "runs/target_ranks.csv"},
        ),
    },
    {
        "pathogen": "Nipah virus",
        "family": "Paramyxoviridae",
        "accession": "NC_002728.1",
        "holdout": "Nipah virus",
        "genome_nt": 18246,
        "targets": (
            {"name": "F fusion glycoprotein",
             "aliases": ("fusion glycoprotein f0", "fusion protein", "f protein"),
             "drug": "none approved", "status": "research target",
             "verified": True, "source": "runs/target_ranks.csv"},
            {"name": "G attachment glycoprotein",
             "aliases": ("glycoprotein g", "attachment glycoprotein", "g protein",
                         "hemagglutinin-neuraminidase"),
             "drug": "m102.4 monoclonal antibody",
             "status": "investigational (compassionate use)",
             "verified": True, "source": "runs/target_ranks.csv"},
            {"name": "L polymerase",
             "aliases": ("rna-directed rna polymerase l", "large structural protein",
                         "l protein", "polymerase"),
             "drug": "remdesivir / favipiravir", "status": "investigational",
             "verified": True, "source": "runs/target_ranks.csv"},
        ),
    },
    {
        "pathogen": "Influenza A PR8",
        "family": "Orthomyxoviridae",
        "accession": ("NC_002016.1,NC_002017.1,NC_002018.1,NC_002019.1,NC_002020.1,"
                      "NC_002021.1,NC_002022.1,NC_002023.1"),
        "holdout": "Influenza A virus (strain A/Puerto Rico/8/1934 H1N1)",
        "genome_nt": 13588,
        "targets": (
            {"name": "Neuraminidase (NA)",
             "aliases": ("neuraminidase", "na", "sialidase"),
             "drug": "oseltamivir / zanamivir", "status": "approved",
             "verified": True, "source": "runs/target_ranks.csv"},
            {"name": "Haemagglutinin (HA)",
             "aliases": ("hemagglutinin", "haemagglutinin", "ha"),
             "drug": "broadly neutralising antibodies", "status": "investigational",
             "verified": True, "source": "runs/target_ranks.csv"},
            {"name": "PA endonuclease",
             "aliases": ("polymerase acidic protein", "pa", "pa endonuclease",
                         "cap-snatching endonuclease"),
             "drug": "baloxavir marboxil", "status": "approved",
             "verified": True, "source": "runs/target_ranks.csv"},
        ),
    },

    # -- draft: NOT verified in this repository ---------------------------------------------
    # Every row below needs the same check: confirm the drug is approved or in trials for this
    # virus, and confirm the molecular target is the viral protein named and not a host factor.
    # ChEMBL mechanism-of-action records and DrugBank targets are the intended primary sources,
    # and ChEMBL is already reached by `bioactivity.prior_binders`, so the check is a lookup and
    # not a literature project.
    {
        "pathogen": "HIV-1",
        "family": "Retroviridae",
        "accession": "NC_001802.1",
        "holdout": "Human immunodeficiency virus type 1",
        "genome_nt": 9181,
        "targets": (
            {"name": "Reverse transcriptase",
             "aliases": ("reverse transcriptase", "rt", "gag-pol polyprotein", "pol polyprotein",
                         "p66 rt"),
             "drug": "tenofovir / efavirenz / dolutegravir-class partners",
             "status": "approved", "verified": False,
             "source": "draft; check ChEMBL mechanism records for NNRTI/NRTI targets"},
            {"name": "Protease",
             "aliases": ("protease", "retropepsin", "gag-pol polyprotein", "pol polyprotein"),
             "drug": "darunavir / ritonavir", "status": "approved", "verified": False,
             "source": "draft; check ChEMBL mechanism records for PI targets"},
            {"name": "Integrase",
             "aliases": ("integrase", "gag-pol polyprotein", "pol polyprotein"),
             "drug": "dolutegravir / raltegravir", "status": "approved", "verified": False,
             "source": "draft; check ChEMBL mechanism records for INSTI targets"},
            {"name": "gp41 transmembrane glycoprotein",
             "aliases": ("transmembrane protein gp41", "gp41", "envelope glycoprotein gp160",
                         "env polyprotein"),
             "drug": "enfuvirtide", "status": "approved", "verified": False,
             "source": "draft; confirm the binding site is gp41 HR1 and not a host receptor"},
            {"name": "gp120 surface glycoprotein",
             "aliases": ("surface protein gp120", "gp120", "envelope glycoprotein gp160",
                         "env polyprotein"),
             "drug": "fostemsavir", "status": "approved", "verified": False,
             "source": "draft; fostemsavir binds gp120 -- distinguish from maraviroc, which "
                       "binds host CCR5 and must stay excluded"},
        ),
    },
    {
        "pathogen": "Hepatitis C virus",
        "family": "Flaviviridae",
        "accession": "NC_004102.1",
        "holdout": "Hepatitis C virus",
        "genome_nt": 9646,
        "targets": (
            {"name": "NS3/4A protease",
             "aliases": ("ns3", "ns3 protease", "serine protease ns3", "ns4a",
                         "genome polyprotein"),
             "drug": "glecaprevir / grazoprevir", "status": "approved", "verified": False,
             "source": "draft; check ChEMBL mechanism records"},
            {"name": "NS5B RNA polymerase",
             "aliases": ("ns5b", "rna-directed rna polymerase", "genome polyprotein"),
             "drug": "sofosbuvir / dasabuvir", "status": "approved", "verified": False,
             "source": "draft; check ChEMBL mechanism records"},
            {"name": "NS5A",
             "aliases": ("ns5a", "non-structural protein 5a", "genome polyprotein"),
             "drug": "velpatasvir / ledipasvir", "status": "approved", "verified": False,
             "source": "draft; check ChEMBL mechanism records"},
        ),
    },
    {
        "pathogen": "Hepatitis B virus",
        "family": "Hepadnaviridae",
        "accession": "NC_003977.2",
        "holdout": "Hepatitis B virus",
        "genome_nt": 3215,
        "targets": (
            {"name": "Polymerase / reverse transcriptase",
             "aliases": ("protein p", "polymerase", "reverse transcriptase", "dna polymerase"),
             "drug": "tenofovir / entecavir", "status": "approved", "verified": False,
             "source": "draft; check ChEMBL mechanism records. Note bulevirtide targets host "
                       "NTCP and is correctly absent"},
            {"name": "Core protein",
             "aliases": ("core protein", "capsid protein", "hbcag", "external core antigen"),
             "drug": "capsid assembly modulators", "status": "investigational",
             "verified": False,
             "source": "draft; confirm at least one CAM reached trials for HBV"},
        ),
    },
    {
        "pathogen": "Human cytomegalovirus",
        "family": "Herpesviridae",
        "accession": "NC_006273.2",
        "holdout": "Human cytomegalovirus",
        "genome_nt": 235646,
        "targets": (
            {"name": "DNA polymerase UL54",
             "aliases": ("dna polymerase", "ul54", "dna polymerase catalytic subunit"),
             "drug": "ganciclovir / foscarnet", "status": "approved", "verified": False,
             "source": "draft; check ChEMBL mechanism records"},
            {"name": "Terminase subunit UL56",
             "aliases": ("ul56", "terminase", "tripartite terminase subunit"),
             "drug": "letermovir", "status": "approved", "verified": False,
             "source": "draft; confirm letermovir's target subunit is UL56"},
            {"name": "UL97 kinase",
             "aliases": ("ul97", "protein kinase ul97", "serine/threonine protein kinase"),
             "drug": "maribavir", "status": "approved", "verified": False,
             "source": "draft; check ChEMBL mechanism records"},
        ),
    },
    {
        "pathogen": "Herpes simplex virus 1",
        "family": "Herpesviridae",
        "accession": "NC_001806.2",
        "holdout": "Human herpesvirus 1",
        "genome_nt": 152222,
        "targets": (
            {"name": "DNA polymerase UL30",
             "aliases": ("dna polymerase", "ul30", "dna polymerase catalytic subunit"),
             "drug": "acyclovir / foscarnet", "status": "approved", "verified": False,
             "source": "draft; acyclovir inhibits the viral DNA polymerase after activation by "
                       "viral thymidine kinase -- both are viral, so both may be labels, but "
                       "the inhibited target is the polymerase"},
            {"name": "Thymidine kinase UL23",
             "aliases": ("thymidine kinase", "ul23"),
             "drug": "acyclovir (activating enzyme)", "status": "research target",
             "verified": False,
             "source": "draft; decide whether an activating enzyme counts as a target at all. "
                       "It is not inhibited, so the conservative choice is to drop this row"},
            {"name": "Helicase-primase complex",
             "aliases": ("helicase-primase", "ul5", "ul52", "dna helicase", "dna primase"),
             "drug": "pritelivir", "status": "investigational", "verified": False,
             "source": "draft; confirm trial stage"},
        ),
    },
    {
        "pathogen": "Respiratory syncytial virus",
        "family": "Pneumoviridae",
        "accession": "NC_001781.1",
        "holdout": "Human respiratory syncytial virus",
        "genome_nt": 15225,
        "targets": (
            {"name": "F fusion glycoprotein",
             "aliases": ("fusion glycoprotein f0", "fusion protein", "f protein"),
             "drug": "nirsevimab / palivizumab", "status": "approved", "verified": False,
             "source": "draft; both are anti-F monoclonals -- confirm the epitope is on F"},
            {"name": "L polymerase",
             "aliases": ("rna-directed rna polymerase l", "large polymerase", "l protein"),
             "drug": "ziresovir / RSV polymerase inhibitors", "status": "investigational",
             "verified": False,
             "source": "draft; confirm ziresovir's target -- some RSV inhibitors bind F, not L, "
                       "and this row is wrong if so"},
        ),
    },
    {
        "pathogen": "Zaire ebolavirus",
        "family": "Filoviridae",
        "accession": "NC_002549.1",
        "holdout": "Zaire ebolavirus",
        "genome_nt": 18959,
        "targets": (
            {"name": "GP glycoprotein",
             "aliases": ("envelope glycoprotein", "spike glycoprotein", "gp1,2",
                         "glycoprotein"),
             "drug": "ansuvimab / atoltivimab-maftivimab-odesivimab",
             "status": "approved", "verified": False,
             "source": "draft; both approved products are anti-GP monoclonals"},
            {"name": "L polymerase",
             "aliases": ("rna-directed rna polymerase l", "l protein", "polymerase"),
             "drug": "remdesivir", "status": "investigational", "verified": False,
             "source": "draft; remdesivir is approved for COVID-19, not for Ebola -- status "
                       "here is investigational and must not be upgraded"},
        ),
    },
    {
        "pathogen": "Monkeypox virus",
        "family": "Poxviridae",
        "accession": "NC_063383.1",
        "holdout": "Monkeypox virus",
        "genome_nt": 197209,
        "targets": (
            {"name": "F13 phospholipase",
             "aliases": ("f13", "palmitoylated egf-like protein", "phospholipase f13",
                         "envelope phospholipase"),
             "drug": "tecovirimat", "status": "approved", "verified": False,
             "source": "draft; tecovirimat targets the VP37/F13 envelope protein -- confirm "
                       "the ortholog name in the MPXV reference annotation"},
            {"name": "DNA polymerase E9",
             "aliases": ("dna polymerase", "e9", "dna-directed dna polymerase"),
             "drug": "brincidofovir / cidofovir", "status": "approved", "verified": False,
             "source": "draft; confirm approval is for smallpox rather than for MPXV, and "
                       "downgrade the status if so"},
        ),
    },
    {
        "pathogen": "Varicella-zoster virus",
        "family": "Herpesviridae",
        "accession": "NC_001348.1",
        "holdout": "Human herpesvirus 3",
        "genome_nt": 124884,
        "targets": (
            {"name": "DNA polymerase ORF28",
             "aliases": ("dna polymerase", "orf28", "dna polymerase catalytic subunit"),
             "drug": "acyclovir / brivudine", "status": "approved", "verified": False,
             "source": "draft; check ChEMBL mechanism records"},
        ),
    },
)


def pathogens(include_draft=False):
    """The label set, verified rows only unless asked for otherwise.

    A pathogen whose every target is draft disappears entirely rather than appearing with an
    empty target list, because a pathogen with no labels contributes no ranks and would only
    dilute the aggregate with a row that cannot be scored.
    """
    out = []
    for entry in PATHOGENS:
        kept = tuple(t for t in entry["targets"] if include_draft or t.get("verified"))
        if kept:
            out.append({**entry, "targets": kept})
    return tuple(out)


def counts():
    """How much of the label set is verified, which is the first thing to report."""
    verified = sum(1 for e in PATHOGENS for t in e["targets"] if t.get("verified"))
    draft = sum(1 for e in PATHOGENS for t in e["targets"] if not t.get("verified"))
    return {"pathogens_total": len(PATHOGENS),
            "pathogens_verified": len(pathogens(include_draft=False)),
            "targets_verified": verified, "targets_draft": draft,
            "statuses": sorted(STATUS_WEIGHT)}
