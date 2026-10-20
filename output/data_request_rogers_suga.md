# Draft data request — Rogers / Suga, PNAS 2018

**Status: draft, not sent.** Needs sign-off on how the project and the SynX collaboration are
described before it goes anywhere.

**Why this exists.** Rogers, Passioura & Suga, *Nonproteinogenic deep mutational scanning of
linear and cyclic peptides*, PNAS 2018;115(43):10959–64, is the only dataset in the project's
bibliography that combines macrocyclic peptides, non-canonical residues and a quantitative
binding endpoint. The per-variant values were never deposited — no data availability statement,
no accession, no supplementary table, and the ΔΔG matrices appear only as rasterised heatmaps
(verified 20 October 2026). Asking the authors is the only route to the numbers.

**To:** Joseph M. Rogers — Department of Drug Design and Pharmacology, University of Copenhagen
(he did the data analysis on the paper)
**cc:** Hiroaki Suga — Department of Chemistry, University of Tokyo (corresponding author,
`hsuga@chem.s.u-tokyo.ac.jp`)

Current addresses should be confirmed from the authors' present institutional pages before
sending; the affiliation above is the one given in the 2018 paper's author note.

**Subject:** Request for per-variant enrichment data — Rogers, Passioura & Suga, PNAS 2018

---

Dear Dr Rogers,

I am a researcher working on machine-learning methods for macrocyclic peptide discovery, in a
project with SynX Therapeutics on their mRNA display platform. I am writing to ask whether the
per-variant data behind your 2018 PNAS paper, *Nonproteinogenic deep mutational scanning of
linear and cyclic peptides*, might be available.

Specifically, I am after the processed ENRICH output — per-variant log2(enrichment) and, if they
exist as a table, the calibrated ΔΔG values — for the PUMA/MCL1 and CP2/KDM4A scans. Per-variant
standard deviations across the repeats would be valuable too if they were retained, since the
paper notes that measurement error varies systematically with input read depth.

The reason I am asking rather than digitising the figures is that yours appears to be the only
published dataset combining macrocyclic peptides, non-canonical residues and a quantitative
binding readout. Almost all public data for this kind of work is canonical amino acids on folded
protein domains, which is the wrong molecule class for what we are trying to predict. The CP2
scan in particular — a genuinely thioether-cyclised macrocycle scanned across a 40-residue
alphabet — has no equivalent I have been able to find. Recovering values from the heatmaps would
give estimates, and I would rather not treat estimates as measurements.

What I would like to do with it is test whether binding can be predicted from a small number of
labels for this molecule class, and specifically whether anything learned from canonical-residue
landscapes transfers to backbone-modified and D-amino-acid substitutions. I would of course cite
the paper as the source of the data, and I am happy to agree any conditions you would prefer on
use or redistribution — including not redistributing it at all, and sharing back whatever the
analysis shows before anything is written up.

If the processed tables are no longer to hand, I would also be glad to know whether the raw
sequencing reads were retained anywhere, or whether you would rather I did not pursue it.

Thank you for considering it, and for the paper — the barcoded-primer trick for deconvoluting
21 separate translation reactions is genuinely clever.

With best wishes,

William Wang
[affiliation, role]
[contact]

---

## Notes before sending

- Confirm both authors' current affiliations and addresses.
- Decide how much to say about the SynX collaboration — the draft keeps it to one clause, which
  is probably right for a cold request, but check there is no confidentiality issue in naming
  the company at all.
- The last paragraph gives them an easy decline. Keep it: a clean no is more useful than silence,
  and it closes the question for the week's planning.
- If there is no reply within about a week, the fallback options are in
  [PROJECT_OVERVIEW_AND_PLAN.md](../PROJECT_OVERVIEW_AND_PLAN.md) §10, item 6.
