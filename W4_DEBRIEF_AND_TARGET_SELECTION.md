# Dataset engineering, biological debrief, target selection, and co-scientist architecture

**Prepared 18 October 2026.** Feeds gates B07 (two prototype families, opens Oct 19),
D01 (single-round experimental question, Oct 22–23) and A06 (feasibility and novelty
review, Oct 23).

No claim of therapeutic efficacy, antiviral activity, binding affinity or full-system
performance is made anywhere below. Targets in §3 are proposals for review with Askar
and the SynX experimental lead, not selections.

---

## 1. How the dataset was engineered

Two decision-policy datasets were constructed from public sources. The engineering
value is almost entirely in the **leakage controls and the accounting**, not in the
substrates, so that is what this section documents.

### 1.1 Binding dataset

**Source.** ProteinGym v1.3, assay `SPG1_STRSG_Wu_2016` — 149,360 measured variants.
ProteinGym records `selection_assay: Binding (IgG)`, `selection_type: binding`, so the
readout is affinity selection rather than a growth or fitness proxy.

**Candidate representation.** Only the four variable residue identities are exposed.
Features are one-hot encoded per position and flattened — `np.eye(20)[features]`
reshaped to 80 dimensions per candidate. No sequence context, no structural features, no
evolutionary features. This is deliberate: it keeps the surrogate weak and interpretable
so that measured gains are attributable to the acquisition policy rather than to feature
engineering.

**Pool construction.** Twenty evaluation pools of 4,096 candidates plus one separate
4,096-candidate development pool. All pools are mutually disjoint and were selected
**without outcome filtering** — no pool was chosen or rejected on the basis of the scores
it contains. Pool-to-pool variability is therefore descriptive spread across
sub-landscapes; it is not biological replication across independent proteins, and must
never be reported as such.

**Hidden-outcome isolation.** The full outcome vector stays in the evaluator. The policy
interface accepts only revealed values, enforced at the function boundary in
`pick_batch` and covered by a test. Top-1% membership is never supplied to a policy.
Candidate identities in LLM prompts are opaque integer pool indices, so the model cannot
recognise a variant and recall its phenotype.

**Paired design.** The eight initial measurements are identical across all policies
within a pool. Every policy comparison inside a pool therefore starts from the same
evidence, which removes initialisation variance from the comparison.

**Acquisition loop.** Total budget 128 measurements, acquired in batches of 8, with
checkpoints recorded at 8 / 16 / 32 / 64 / 128. Repeated or previously-queried
acquisitions raise rather than being silently deduplicated.

**Surrogate.** A single fixed 64-tree random forest (`min_samples_leaf=2`,
`max_features=1.0`) is shared by greedy, UCB and the LLM-mediated policy, so no policy
gets a better model than another. Per-tree prediction dispersion is used as the
uncertainty heuristic. It is a heuristic, not a calibrated posterior — see §2.3.

**Tie handling.** Candidates are permuted with a seeded RNG before a stable argsort, so
ties at the acquisition threshold are broken reproducibly without perturbing the score
ordering. This matters more than it sounds: the GB1 landscape has large exactly-tied
plateaus at low budget, and unseeded tie-breaking silently injects a diversity bonus
into the greedy policy.

**Delayed feedback.** Acquisitions enter a pending queue keyed by `(round + delay, ids)`
and become visible only when the round arrives. Conditions are 0 and 3 **decision
rounds**. These are synthetic stress tests and are explicitly not estimates of assay
turnaround.

**Dual evaluation.** Every checkpoint records both the retrospective metrics
(`best_score`, `top_recall` over everything *selected*) and the realtime metrics
(`returned_best_score`, `returned_top_recall`, `pending_labels` over what has actually
*returned*). Keeping both is the single most useful design decision in the dataset — see
§2.2.

### 1.2 Structural dataset

**Source.** ATLAS, pinned metadata `2024_11_18_ATLAS_info.tsv`, three 100 ns MD
replicates per entry.

**Selection.** Human organism, 50–250 residues, nonredundant flag, one entry per ECOD
family, sorted by PDB identifier, split 4 development / 8 evaluation. Critically,
**trajectory-derived outcome columns were excluded from the selection criteria**, so
entries were not chosen for having convenient dynamics.

**Observable.** Solvent-accessible surface area via MDTraj `shrake_rupley`, nm² per
residue, 240 sphere points, trajectory stride 5, in acquisition blocks of 20 saved
frames.

**Evidence isolation.** Each protein's first two blocks expose one block from replicate
R1 and one from R2; later requests interleave chronological blocks from R1 and R2.
**R3 is held out entirely** and is used only inside the evaluator as a simulation
reference. Temporal correlation within a replicate is preserved rather than shuffled
away, so block dispersion is a heuristic and not a frame-independent confidence interval.

**Cost accounting.** Acquisition cost is replayed from *measured* per-block preprocessing
CPU time, recorded during preprocessing. It is not live MD cost and not request latency.
LLM latency and tokens are logged separately so compute cost and inference cost never
pool.

### 1.3 Provenance and reproducibility

- SHA-256 hashes for every downloaded input and every frozen split, in
  `data/processed/*/manifest.json`.
- Every result file embeds `dataset_manifest_sha256` and `config_sha256`, so a result
  can be tied to the exact config and data that produced it.
- Atomic JSON writes (temp file then replace) with `allow_nan=False`, so a NaN cannot
  silently enter a results file and a partial write cannot be mistaken for a result.
- Smoke outputs carry a `smoke` flag and are excluded from scientific summaries.
- Inference or analysis failure fails its task explicitly. A failed LLM call never
  degrades into a baseline policy result.

### 1.4 Agent interface

The LLM policy speaks to any OpenAI-compatible backend selected by
`PILOT_LLM_PROVIDER` (local vLLM, OpenRouter, Anthropic, OpenAI), with per-provider
request shaping. Actions are constrained by a strict JSON schema; an action outside the
allowed set, or a reply containing no JSON object, raises. Hosted runs get a
provider-specific policy label so they can never pool with or overwrite local-model runs,
and the completion accounting counts only the configured grid.

### 1.5 What this design does and does not model

Worth stating plainly, because it governs §2:

| The dataset models | The dataset does not model |
| --- | --- |
| Iterative small-batch assay follow-up | A single large display selection round |
| Dense, fully-enumerated candidate space (4 sites × 20 canonical AAs) | ~10¹² sparse macrocyclic library |
| Per-candidate quantitative scores | Enrichment ratios across selection rounds |
| 20 canonical amino acids | Non-canonical residues, cyclisation chemistry |
| Synthetic delay in decision rounds | Real reagent and assay turnaround |

---

## 2. Biological debrief, and whether the earlier experiments survive

### 2.1 What the substrates actually are

**Binding arm.** `SPG1_STRSG_Wu_2016` is the **B1 domain of streptococcal protein G
(GB1)** — residues 265–280 of a 448-residue parent, four co-varying sites, measured by
IgG Fc affinity selection. A bacterial immunoglobulin-binding domain with canonical
amino acids at four positions. Not a peptide, not cyclic, no non-canonical residues, no
pathogen relevance.

**Structural arm.** All twelve ATLAS entries are **human proteins**, selected for ECOD
family diversity with no therapeutic intent.

- *Evaluation (8):* TFIIF subunit 2, platelet basic protein (CXCL7), retinoic acid
  receptor γ, SKP2, non-secretory ribonuclease, IGF-1, UBR5, major prion protein.
- *Development (4):* galectin-3-binding protein, Axin-1, moesin, MEF2A.

No macrocyclic peptide, no non-canonical amino acid and no pathogen target appears
anywhere in the codebase or results.

### 2.2 The argument for keeping the binding arm

Your skepticism is largely right, but three things survive substrate-independently and
are worth defending:

**(a) It is a validated scoring harness, and the training plan requires one.** The
Week 8–10 harness work depends on baseline parity — a scorer cannot be validated against
nothing. The existing tests cover disjoint pools, unique acquisitions, delayed-outcome
visibility, policy isolation from hidden labels, structural reference isolation and
resource accounting. That is reusable infrastructure regardless of substrate, and
rebuilding it against a pathogen target is a config change plus pool-size rework, not a
rewrite.

**(b) The selected-versus-returned gap is a measurement-design finding, not a biology
finding — and it is real.** Scoring a policy on what it *selected* rather than what has
*returned* overstates it by roughly 1.4× at a three-round delay (top-1% recall 0.204 vs
0.144 for greedy; 0.213 vs 0.162 for UCB). That is a statement about how to construct a
decision metric under delayed feedback, and it transfers directly into the episode rubric
and the uplift loss definition. Any evaluation of an emergency scientist that scores
retrospective selection quality is measuring something the agent could not have known.

**(c) The calibration result de-risks the project premise, and this is the strongest
argument.** Acquisition policy delivers roughly 8× top-1% recall over random *while the
surrogate is only weakly informative* — Spearman ≈ 0.31 against truth at the 128-label
budget, with plain ridge regression matching a 64-tree forest. If decision-policy gains
had required a good predictor, the whole emergency premise would collapse, because at
emergency timescales there are no labels to fit one. The result says the gains come from
the policy. That is the claim worth carrying forward, and it is substrate-independent in
a way the recall numbers are not.

**What does not survive.** The quantitative numbers transfer not at all. GB1 is dense,
small and fully enumerated; display is sparse, vast and macrocyclic with an enrichment
readout. "128 labels from a 4,096 pool in batches of 8" models iterative plate-based or
biophysical follow-up, not a selection round. The 0-vs-3-round delay is unanchored to any
real turnaround. None of this should appear in a paper as a finding about peptide
discovery.

**Verdict: keep, re-point, and demote.** Treat the binding arm as harness validation plus
hypothesis (b) and (c), re-run against a pathogen target, and never present the GB1
recall figures as a scientific result.

### 2.3 The argument against keeping the structural arm

I cannot construct a good one.

The arm measures how fast mean per-residue SASA estimates converge to a held-out MD
replicate, under different block-allocation policies, on twelve human proteins chosen for
family diversity. The measured effect is real but small — uncertainty-weighted allocation
reached the same accuracy as uniform at about 33% less replayed CPU (68 s vs 101 s),
then plateaued and drifted slightly worse.

The problem is that **SASA agreement with a held-out MD replicate is not an
epitope-accessibility measure** in any sense that feeds target ranking. Nothing in the
theory of change consumes it. Surface exposure does not establish binding, inhibition or
antiviral activity, and the README already says so. Re-pointing it at viral
glycoproteins would not fix this, because the thing being measured — convergence of an
average to a sibling simulation — still would not answer an epitope question.

**Verdict: retire the arm as a scientific component.** Keep the cost-accounting and
evidence-isolation code, which is reusable. If structural evidence allocation is still
wanted as a decision domain, it needs rebuilding around a question a dossier actually
asks: *is this epitope solvent-exposed and conformationally stable enough to be worth a
display campaign*, scored against something other than another MD replicate.

### 2.4 Calibration detail, for the record

| budget | overconfidence ratio | 1σ coverage | Spearman(σ, abs err) |
| --- | --- | --- | --- |
| 8 | 63.5 | 0.56 | 0.21 |
| 32 | 30.2 (greedy) | 0.73 | 0.56 |
| 128 | 1.8 (greedy) / 3.1 (random) | 0.91 / 0.84 | 0.87 |

Forest-dispersion sigma is severely overconfident as a magnitude at low budget but ranks
error reliably past roughly 64 labels. Usable for ordering, not as an interval.

| budget | Spearman(pred, truth), forest | top-1% precision, forest | Spearman, ridge |
| --- | --- | --- | --- |
| 8 | 0.05 | 0.026 | 0.13 |
| 128 | 0.31 | 0.088 | 0.28 |
| 1024 | 0.51 | 0.235 | 0.46 |

**Open gap:** both calibration studies exist as result CSVs with no generating script
committed. They are not reproducible from source. Fix before A06.

---

## 3. Next targets

### 3.1 The constraint that should drive selection

RaPID-style mRNA display requires a **purified, soluble, immobilisable target** —
typically biotinylated and captured on streptavidin beads, surviving several iterative
selection rounds. The library (~10¹²–10¹³) is generated by flexizyme-charged initiator
incorporation with spontaneous thioether macrocyclisation, so the chemistry is largely
target-independent; **the reagent is the binding constraint, not the library**.

Separately, macrocyclic peptides are generally poor at crossing membranes — which is
precisely why the reading list carries PeptideCLM (Feller & Wilke 2025) and the
13-method permeability benchmark (PMC12392647).

This produces a tension that should be stated rather than finessed:

> The best-behaved viral reagents are **soluble intracellular enzymes** (proteases,
> helicases, methyltransferases), which carry a permeability liability. The
> **extracellular glycoproteins** need no permeability story but are harder reagents and
> sit under strong positive immune selection, which undermines conservation-based
> family claims.

Choosing a protease and *owning* the permeability problem is more defensible than
choosing a glycoprotein and quietly inheriting a conservation problem — because
permeability is predictable in silico with tools already in the bibliography, whereas
conservation of an immune-evasion surface cannot be engineered around.

### 3.2 Selection rubric

B07's acceptance condition requires documented selection criteria. Proposed rubric —
these seven axes are also the backbone of the target dossier in core deliverable 2:

| Axis | What it asks | Why it gates |
| --- | --- | --- |
| **Reagent tractability** | Can SynX obtain soluble, correctly-folded, immobilisable protein that survives multiple selection rounds? | Hard gate. An untractable reagent ends the campaign regardless of biology. |
| **Essentiality** | Does loss of function block replication? | A binder to a dispensable protein cannot become a therapeutic. |
| **Conservation** | Is the epitope conserved across the family and under what selective pressure? | Gates every family-wide and prototype-pathogen claim. |
| **Accessibility** | Intracellular or extracellular; is the epitope solvent-exposed in the relevant conformation? | Sets whether a permeability story is required. |
| **Assay availability** | Is there a functional orthogonal assay beyond enrichment? | Without one, a single round can only substantiate enrichment. |
| **Public label availability** | Is there a DMS or binding landscape for method development? | Determines whether transfer tests are runnable now or blocked on SynX. |
| **Precedent** | Prior macrocycle or peptide campaigns against this target? | Supplies positive controls and de-risks failure interpretation. |

### 3.3 Family 1 — Coronaviridae (prototype SARS-CoV-2)

#### Mpro / 3CLpro (nsp5) — primary recommendation

A cysteine protease, active only as a homodimer, with a Cys145–His41 catalytic dyad. It
cleaves the viral polyprotein at eleven sites with an unusual Gln↓(Ser/Ala/Gly)
specificity that has no close human protease counterpart, which helps selectivity.

| Axis | Assessment |
| --- | --- |
| Reagent tractability | **Strong.** Routine soluble *E. coli* expression, well-behaved, inexpensive. |
| Essentiality | **Strong.** Polyprotein processing is obligatory for replication. |
| Conservation | **Strong.** Active site highly conserved across sarbecoviruses; workable across betacoronaviruses. |
| Accessibility | **Weak.** Cytosolic. Requires a permeability story. |
| Assay availability | **Strong.** Standard FRET peptide-cleavage assay for functional follow-up. |
| Public labels | **Yes** — `R1AB_SARS2_Flynn_2022`, 5,725 mutants, FRET + growth readout, already downloaded. |
| Precedent | Published macrocyclic peptide campaigns exist, including RaPID-derived work. **Verify specific citations before treating any as a positive control.** |

The cytosolic liability is arguably an asset for this project. It forces a genuinely
multi-objective decision — affinity against predicted permeability — which is a harder
and more interesting problem for a co-scientist than single-score ranking, and it
activates an entire section of the reading list that would otherwise go unused.

#### PLpro (nsp3 papain-like protease) — strong secondary

Cleaves the nsp1/2/3 boundaries and additionally strips ubiquitin and ISG15 from host
proteins, so inhibition is simultaneously antiviral and immune-restorative. Soluble and
tractable; a zinc-binding domain requires attention in construct design. Conserved but
less so than Mpro across genera. No public DMS. Cytosolic.

#### nsp10/nsp16 interface — the interesting protein–protein option

nsp16 is a 2′-O-methyltransferase that caps viral RNA to evade innate immune sensing,
and it is **catalytically dependent on nsp10 as an obligate cofactor**. The nsp10–nsp16
interface is therefore a protein–protein interaction that macrocycles are
constitutionally better suited to than small molecules are. Conserved, essential for
immune evasion. Harder reagent (two-protein complex), no public DMS, cytosolic.

Worth flagging because "macrocycles beat small molecules at flat PPI interfaces" is the
strongest mechanistic argument for the modality, and a PPI target lets the project make
that argument rather than competing with small molecules on an enzyme active site.

#### nsp13 helicase

Superfamily-1 helicase with 5′→3′ polarity, NTPase activity, zinc-binding domain. Highly
conserved, essential. Reagent is moderately tractable. Cytosolic, no public DMS.

#### Spike RBD — best data, worst conservation argument

`SPIKE_SARS2_Starr_2020_binding` (3,802 variants) and `_expression` (3,798) are already
on disk and are among the cleanest binding landscapes in existence. Extracellular, so no
permeability story needed, and the reagent is well established.

But the RBD is an immune-evasion surface under intense positive selection. It is
therefore the **wrong choice for any conservation or family-transfer claim**, despite the
data quality. Use it for method development; do not build the family narrative on it.

#### Spike S2 stem-helix / fusion peptide — the conserved extracellular option

Far better conserved across betacoronaviruses than the RBD, and the target of known
broadly-neutralising antibodies, which establishes that the epitope is accessible and
druggable. Harder reagent — needs a stabilised construct or peptide presentation rather
than whole prefusion spike — and no public DMS. This is the right answer if the project
needs a conserved *extracellular* coronavirus target.

### 3.4 Family 2 — two options, and the obvious choice is not the better experiment

#### Henipavirus (Nipah) — better narrative

- **G attachment glycoprotein.** Tetrameric, with a six-bladed β-propeller head that
  binds ephrin-B2 and ephrin-B3. Genuinely extracellular, so no permeability story.
  Receptor-binding site is well mapped. Reagent is obtainable as a soluble head domain.
- **F fusion glycoprotein.** Class I trimeric fusion protein requiring cathepsin L
  cleavage for activation. The prefusion conformation is metastable and needs
  stabilisation to be a usable display reagent — a genuine reagent risk.

Strengths: WHO R&D Blueprint priority pathogen; a real conservation ladder for transfer
testing (NiV-Bangladesh → NiV-Malaysia → Hendra → Cedar → Langya); matches the
"extracellular peptides" framing from the Oct 5 discussion. Weakness: **no public DMS**,
so the labeled molecular-transfer tests stay blocked on SynX data.

#### Influenza A — better experiment

- **HA stem (HA2 helix A region).** The conserved epitope recognised by
  group-1-specific and group-2-specific broadly-neutralising antibodies. Extracellular,
  well conserved within a group, strong peptide-binder precedent.
- **NA catalytic site.** Sialidase active site, highly conserved framework and catalytic
  residues, extracellular, validated as a drug target.
- **PA–PB1 interface.** The PA C-terminal domain binds an N-terminal PB1 peptide — a
  well-characterised PPI with published peptide inhibitors. Intracellular.

Decisive advantage: **four influenza landscapes are already downloaded** — HA for H1N1
(Doud 2016, 10,715 mutants) and H3N2 (Lee 2018, 10,754), NA (Jiang 2016), and polymerase
subunits. The H1→H3 transfer test is runnable this week on data already on disk, which is
exactly the grouped-holdout shift test the brief specifies.

#### Recommendation

**Coronaviridae (Mpro) + Orthomyxoviridae (HA stem / NA).**

Both families have public labeled data, which unblocks the molecular transfer tests
immediately rather than making them dependent on a collaborator. Together they also span
the tension in §3.1 deliberately — one soluble intracellular enzyme with a permeability
problem, one conserved extracellular epitope without one — which makes the pair a
designed contrast rather than two arbitrary picks.

Retain **Nipah as the stretch robustness family**: better story, no labels.

### 3.5 Pathogen landscapes already on disk

| DMS id | Target | Assay | Mutants |
| --- | --- | --- | --- |
| `R1AB_SARS2_Flynn_2022` | SARS-CoV-2 Mpro | FRET, growth | 5,725 |
| `SPIKE_SARS2_Starr_2020_binding` | SARS-CoV-2 Spike RBD | ACE2 binding | 3,802 |
| `SPIKE_SARS2_Starr_2020_expression` | SARS-CoV-2 Spike RBD | ACE2 binding | 3,798 |
| `A0A2Z5U3Z0_9INFA_Doud_2016` | Influenza HA (H1N1) | Viral replication | 10,715 |
| `C6KNH7_9INFA_Lee_2018` | Influenza HA (H3N2) | Viral replication | 10,754 |
| `NRAM_I33A0_Jiang_2016` | Influenza neuraminidase | — | 298 |
| `A4D664_9INFA_Soh_2019` | Influenza PB2 | Viral replication (avian cells) | 14,421 |
| `RDRP_I33A0_Li_2023` | Influenza PB1 | Viral replication | 12,003 |
| `POLG_DEN26_Suphatrakul_2023` | Dengue NS5 | Viral replication | 16,897 |
| `A0A140D2T1_ZIKV_Sourisseau_2019` | Zika envelope | Viral replication | 9,576 |

Engineering note: Spike RBD has 3,802 variants and Mpro 5,725, both at or below the
current `pool_size: 4096`. The twenty-disjoint-pool design needs rework for these —
either fewer, smaller pools, or resampling with documented overlap. Do not silently
reuse candidates across "disjoint" pools.

### 3.6 Other families, for completeness

- **Flaviviridae NS2B–NS3 protease** (dengue, Zika). Soluble, standard fluorogenic
  assay, requires the NS2B cofactor peptide for activity. Among the most tractable viral
  protease reagents available, and dengue NS5 and Zika envelope landscapes are already on
  disk. A credible third family if influenza is rejected.
- **Filoviridae GP** (Ebola, Marburg). The NPC1-binding site is only exposed after
  cathepsin cleavage, which makes the conformational state of the reagent the central
  design question. High-value, harder.
- **Arenaviridae GPC** (Lassa). Prefusion trimer, notoriously difficult to stabilise.
  Not recommended as a first campaign.
- **Bunyavirales Gn/Gc** (Rift Valley fever, CCHF). Both WHO priorities; reagents less
  established.

### 3.7 The ambitious alternative worth 10 minutes at the gate

The reading list already contains PMC12464979 — macrocyclic binders, covalent modifiers
and degraders of a **structured RNA**, discovered by mRNA display.

The SARS-CoV-2 **frameshift stimulatory element** is a pseudoknot between ORF1a and
ORF1b that sets the −1 programmed ribosomal frameshift ratio. It is highly conserved
(structured RNA elements are constrained by base-pairing, not just by sequence) and
functionally essential, since the virus requires a specific ratio of ORF1a to ORF1b
products.

Targeting it would be the most novel element available to the project, has direct
precedent in the project's own bibliography, and sidesteps the conservation problem
entirely because RNA secondary structure is under far stronger constraint than a surface
glycoprotein epitope. Against it: non-standard reagent and assay, a less certain
functional readout, and no public landscape. Worth putting to Askar even if declined.

### 3.8 Recommendation for the single display round (D01, due Oct 23)

**Mpro.** Cheapest and best-behaved reagent, strongest conservation case, an orthogonal
functional assay available if the schedule allows, and interpretable failure. Report
enrichment as enrichment — one round does not establish affinity, functional inhibition
or antiviral activity.

---

## 4. Next steps and co-scientist architecture

### 4.1 The gap

The agent's current decision state is:

```
observed_score_summary, candidate_summary[{candidate: <int>, prediction,
uncertainty_heuristic, diversity}], remaining_label_budget, pending_labels,
feedback_delay_rounds, decision_round
```

with an action space of `{greedy, explore, diverse}`. Candidate identities are
deliberately opaque integers.

So what exists is a **bandit-strategy selector**. It reasons competently about budget
and uncertainty, but it contains no biology, and it is several abstraction layers from
the proposal's Part 1: pathogen intake from genomic sequence → evidence retrieval →
structure prediction → target and epitope ranking → library design.

Closing that gap means the decision state becomes the **target dossier schema** — the
seven axes in §3.2 — and the action space becomes what the brief already specifies:
retrieve missing evidence, request a structure calculation, revisit a target ranking,
invoke a design tool, recommend an experiment. The budgeted-acquisition machinery
already built becomes the resource accounting underneath that policy.

### 4.2 Architecture brainstorm: treating this as a data problem, not a prompting problem

The five capabilities you flagged map onto the existing Week 8–10 harness more cleanly
than I expected. Below, each one, what it means here, and honestly how much of it is
already planned versus genuinely new.

#### (a) Scale synthetic episodes with containerised verification

*Pattern: generate synthetic tasks at volume, verify each one automatically in a
container, keep only what passes.*

**The scientific analogue is strong.** Public DMS landscapes are
**auto-gradeable decision environments**: the answer is already known for every variant,
so a proposed acquisition, ranking or stop decision can be scored deterministically with
no human and no wet lab. Each of the ten landscapes in §3.5 generates an effectively
unbounded number of episodes by varying pool composition, budget, delay, initial
evidence, corrupted evidence and conflicting evidence. The containerised oracle holds the
hidden outcomes, exposes only a tool API, and returns a verified score — the same
isolation boundary already enforced in `pick_batch`, just packaged.

**Already planned:** the synthetic world generator with known latent mechanisms and an
independently validated oracle, plus the five stress conditions (clean, confounding,
mechanism shift, evidence corruption, longer horizon).

**Genuinely new:** generating episodes from the *molecular* landscapes rather than only
the transcriptomic worlds, which would let the applied SynX case and the forecasting
testbed share one episode format.

**Honest scope check:** the pattern scales to trillions of tokens in a pretraining
context. This project budgets ~12.5M training tokens total. The verification
architecture is the transferable part; the scale is not, and claiming otherwise in a
paper would be indefensible.

#### (b) Distil a flagship model into a fast policy

*Pattern: generate reasoning traces with an expensive model, train a cheap model on
them.*

**This is the missing piece that makes the rest work.** The current agent makes one call
per decision round at ~2.8 s. A co-scientist that *searches* over actions — evaluating
several candidate next-actions before committing — needs hundreds to thousands of
rollouts per episode. That is unaffordable at flagship prices and latency, and it is the
real reason the current design collapsed into a three-option strategy selector rather
than an action search.

The route: run a flagship model on verified episodes from (a) with full evidence
retrieval and extended rationale, keep only trajectories that the container scored well,
and use those as the training curriculum for the small trainable bases. The existing two
curricula — conclusion/evidence reasoning, and grounded tool use/planning — are exactly
the right two buckets, and distilled flagship traces are a far better source for them
than hand-written data.

**Status: not in the plan.** This is the most valuable addition and it slots into the
existing curriculum slots without changing the experiment matrix.

#### (c) Mid-training data mixtures to raise the RL ceiling

*Pattern: the mixture determines the headroom that later RL can exploit.*

The question for this project: what mixture of protein sequence data, structural data,
assay protocols, primary literature and decision traces does a base model need before
decision-quality RL has anything to climb? A model that has never seen an assay protocol
cannot learn to sequence experiments, however good the reward.

**Already planned, with the variable already in place:** the matrix has 2 curricula × 3
models × 2 seeds, and the uplift metric `U(BT, BI) = C(post-trained, BI) − C(base, BI)`
with conclusion, grounding and process losses reported separately. That *is* a mixture
ablation with a ceiling measure attached.

**The sharpening:** treat curriculum composition as the designed variable rather than
a fixed pair of labels, and report uplift as a function of mixture rather than only of
budget. Decomposed losses make this readable — a mixture that improves grounding while
flattening conclusion quality is a result, not noise.

#### (d) Long-context data recipes

*Pattern: deliberately engineer how long inputs are composed, not just how long they
are.*

This is a structural property of the problem, not an add-on. A target dossier accumulates
monotonically within an episode: genome, MSA, structure predictions, conservation
analysis, literature, prior assay rounds, and the rationale for every earlier decision.
By the final decision of an episode the evidence state is the largest object in the
system, and **what gets evicted from it is itself a decision the policy should make**.

Concrete recipe questions: does the dossier stay as raw evidence or as a running
summary; is prior rationale retained verbatim, summarised, or dropped; when evidence
conflicts, does the newer entry replace the older or sit alongside it with both
timestamps. The brief's requirement that the decision state record the evidence available
at that moment means conflicting evidence must coexist rather than being overwritten —
which has a direct and under-appreciated context cost.

**Partially planned:** the inference budget (4,096 generated tokens, 8 tool calls) with
a control arm at 8,192 and 16 calls, plus "longer horizon" as an explicit stress
condition, which separates post-training gains from extra inference. The eviction policy
itself is not yet specified.

#### (e) Robust, diverse evaluation of mid-training checkpoints

*Pattern: without a cheap, reliable, diverse eval, neither mixture ablation nor RL can
be steered.*

**This is already the best-specified part of the plan**, and it is worth recognising as
such: 51 model states (48 trained checkpoints at 16K / 64K / 256K / 1M tokens, plus 3
zero-shot bases) × 5 stress conditions × 10 paired cases × 2 rollout seeds = 5,100
episodes, plus 600 inference-control episodes. Cases are allocated across four
development world families, three unseen mechanism families and three audited real
studies, with holdouts grouped by entire model and curriculum groups rather than by
budget point or rollout seed.

**The addition:** the verified molecular episodes from (a) become additional eval
conditions, giving the checkpoint suite a domain where ground truth is exact rather than
expert-assessed. That is a useful complement to rubric-scored episodes, whose inter-rater
reliability is itself a project risk.

### 4.3 The reframe this suggests

Taken together, the five capabilities point at a sharper contribution than "an agent that
helps scientists decide."

> The bottleneck in scientific-decision agents is not the policy architecture. It is the
> absence of **verifiable decision environments** at the volume needed to train and
> evaluate one. Public DMS landscapes are auto-gradeable decision environments that
> already exist, in quantity, and are not being used that way.

The project is unusually well placed to argue this: it already has the harness, the
uplift metric, the checkpoint grid, and — as of §3.5 — ten landscapes on disk. The
pipeline that turns them into verified episodes is the missing component, and it is
smaller than the harness that would consume it.

This also rescues the Oct 12 work: the GB1 pilot is the first instance of exactly this
pattern — a verified, auto-graded decision environment built from a public landscape with
enforced label isolation. It was built as a biology pilot, which is why it looks weak.
As infrastructure it is the prototype for (a).

**Risk, stated plainly:** a mid-training data programme is not a ten-week deliverable.
The minimum viable version is the episode generator plus containerised oracle over two or
three landscapes, feeding the eval conditions in (e). Distillation (b) and mixture
ablation (c) are W8–W9 at the earliest and only if access and spend gates clear.

### 4.4 Ordered actions

1. **Oct 19 — B07.** Put Mpro + influenza to Askar with the §3.2 rubric as the
   documented selection criteria and §3.5 as the availability argument. Gates C06, D01
   and A06.
2. **Re-point the binding harness** at Mpro and the two HA landscapes. Mostly a config
   change plus pool-size rework for the smaller landscapes; the policy is
   dataset-agnostic. Converts a methods demo into a result about pathogen targets.
3. **Retire the structural arm** as a scientific component, per §2.3. Keep the
   cost-accounting and evidence-isolation code.
4. **Write the dossier schema before the policy that consumes it.** The §3.2 axes are
   deliverable 2. Building B06 first produces another bandit.
5. **Specify the episode format once**, so molecular and transcriptomic episodes share
   it. This is what makes (a) and (e) in §4.2 compose instead of forking.
6. **Restore the calibration scripts** so §2.4 reproduces from source.
7. **Pull Rogers, Passioura & Suga (2018)** nonproteinogenic DMS (PNAS 115(43),
   10959–10964) — a deep mutational scan of linear *and cyclic* peptides including
   non-canonical residues, the only labeled macrocycle dataset with ncAAs in the
   bibliography, and a far better substrate for low-label ranking work than GB1.
8. **Prototype the containerised oracle** over one landscape before committing to the
   full episode generator.

---

## 5. Open items

- Calibration results in §2.4 are not reproducible from source.
- No prototype family, target, assay or chemical alphabet selected. B07 is the blocker.
- The entire physics-based ncAA track remains untouched — Flex ddG, FakeRotLib,
  MakeRotLib, Dunbrack rotamer libraries, the RosettaCommons ncAA interface protocol.
  Nothing in the repository approaches it.
- Macrocyclic-peptide precedent against Mpro is asserted from memory in §3.3 and must be
  verified before it is used as a positive-control argument.
- A06 additionally depends on the lab route and data-return terms, live trainable model
  access and spend cap, and the novelty review — all external to this repository.
- ATLAS data is CC BY-NC 4.0; this work is scoped to noncommercial research. Commercial
  use requires separate terms from the dataset owner.
