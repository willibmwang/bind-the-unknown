# Week 4 debrief and target selection

**Prepared 18 October 2026 (W3D7).** Covers the state of the pilots entering Week 4
(Oct 19–23), the biological content of what has actually been built, a calibration
audit that changes how the Oct 12 baselines should be read, and recommended prototype
families and targets for gates B07 (opens Oct 19), D01 (Oct 22–23) and A06 (Oct 23).

Scope note: this document records the current evidence state and recommendations. It
makes no claim of therapeutic efficacy, antiviral activity, or full-system performance.
Target recommendations below are proposals for review with Askar and the SynX
experimental lead, not selections.

---

## 1. Summary of position

Three things are true at once, and the tension between them is the main thing to resolve
this week:

1. **The decision machinery works and is validated.** Budgeted acquisition, delayed
   feedback, and structural evidence allocation all produce clean, reproducible results
   across 160 binding episodes and three structural policies.
2. **None of it touches macrocyclic peptides, non-canonical amino acids, or pathogen
   targets.** The substrates are a bacterial IgG-binding domain and twelve human
   structural controls. No target, family, assay or chemical alphabet has been selected.
3. **Better substrates are already on disk.** `data/raw/proteingym_metadata.csv`
   contains SARS-CoV-2 Mpro and Spike RBD landscapes, plus four influenza landscapes,
   all downloaded on Oct 12 and unused.

The gating task, B07 — choose two prototype pathogen families with Askar — opens Oct 19
and feeds C06, D01 and the A06 feasibility review on Oct 23.

---

## 2. What has been built

### 2.1 Week 3 work (Oct 17)

**Multi-provider agent interface.** `hpc_pilots/common.py` previously assumed a local
vLLM server inside a Slurm allocation. The `Agent` class now speaks to any
OpenAI-compatible backend selected by `PILOT_LLM_PROVIDER`: local vLLM, OpenRouter,
Anthropic, or OpenAI.

- `load_env()` reads a gitignored `.env`; real environment variables take precedence.
- Per-provider quirk handling: `NO_TEMPERATURE` (current Claude models reject
  `temperature`), `MAX_TOKENS_KEY` (OpenAI expects `max_completion_tokens`).
- Actions are constrained by a strict JSON schema. An out-of-set action or a non-JSON
  reply raises; failed inference never silently substitutes a baseline policy.
- Hosted runs receive a provider-specific policy label
  (e.g. `agent-anthropic-claude-haiku-5-5`) so they cannot overwrite or pool with
  local-model runs. `summarize.py` counts only the configured grid toward completion.

The significance: **the agent arm no longer depends on a GPU allocation.** That was the
blocker. `runs/summary/status.json` still reads binding 160/200 and structure 3/4, and
the missing arm in both cases is the agent — the queued L40 job was
`PENDING|(Dependency)` at the Oct 12 checkpoint.

**Contract smoke test, off-cluster.** `runs/binding/smoke_agent-anthropic-claude-haiku-5-5_pool00_delay0.json`:
pool 0, 16 labels, Claude Haiku 5.5. One call, 2.83 s, 1,525 prompt / 165 completion
tokens. The agent selected `explore` with a rationale grounded in the supplied state —
it correctly noted that predictions were near-degenerate, uncertainty was high and
roughly uniform, and diversity values were coarse. The contract works end to end. That
is one episode, not a result.

### 2.2 Calibration audit (Oct 17) — reinterprets the Oct 12 baselines

Two studies, both of which should change how the pilot results are described.

**`runs/calibration/rf_calibration.csv`** (20 pools × 5 budgets × 2 regimes) audits the
random forest's tree-dispersion sigma — the uncertainty heuristic the whole pilot relies
on:

| budget | overconfidence ratio | 1σ coverage | Spearman(σ, abs err) |
| --- | --- | --- | --- |
| 8 | 63.5 | 0.56 | 0.21 |
| 32 | 30.2 (greedy) | 0.73 | 0.56 |
| 128 | 1.8 (greedy) / 3.1 (random) | 0.91 / 0.84 | 0.87 |

Sigma is severely overconfident as a *magnitude* at low budget, but *ranks* error
reliably once past roughly 64 labels. Usable for acquisition ordering; not usable as a
confidence interval. This is what `hpc_pilots/README.md` already asserted — now with
numbers behind it.

**`runs/calibration/surrogate_skill.csv`** (20 pools × 8 budgets × forest/ridge,
8→1024 labels) is the more consequential one:

| budget | Spearman(pred, truth), forest | top-1% precision, forest | Spearman, ridge |
| --- | --- | --- | --- |
| 8 | 0.05 | 0.026 | 0.13 |
| 128 | 0.31 | 0.088 | 0.28 |
| 1024 | 0.51 | 0.235 | 0.46 |

At the pilot's actual 128-label budget the surrogate reaches Spearman ≈ 0.31 and top-1%
precision ≈ 0.09, and plain ridge regression matches the forest throughout.

**The reframing this forces, and it is a better claim than the original:** acquisition
policy produces an ~8× improvement in top-1% recall over random *even when the
underlying predictor is only weakly informative*. The result is about the decision
policy, not about model quality.

**Reproducibility gap:** both calibration CSVs exist as results with no generating
script in `hpc_pilots/`. They are not reproducible from source. This should be fixed
before A06 on Oct 23.

### 2.3 Oct 12 baselines

160 binding episodes (20 disjoint pools × 4 policies × 2 feedback conditions), at the
128-label budget:

| delay | policy | best score | top-1% recall (selected) | recall (actually returned) |
| --- | --- | --- | --- | --- |
| 0 | greedy | 5.41 ± 1.07 | 0.277 ± 0.130 | 0.277 |
| 0 | ucb | 5.34 ± 1.31 | 0.270 ± 0.104 | 0.270 |
| 0 | random | 2.74 ± 1.05 | 0.032 ± 0.028 | 0.032 |
| 0 | diverse | 2.83 ± 0.94 | 0.028 ± 0.018 | 0.028 |
| 3 | greedy | 4.93 ± 1.47 | 0.204 ± 0.117 | **0.144** |
| 3 | ucb | 5.06 ± 1.29 | 0.213 ± 0.097 | **0.162** |

Model-guided acquisition beats random by roughly 8× on top-1% recall. A three-round
feedback delay costs about 25% of *retrospectively measured* recall but about 45% of
what has actually returned by the end of the budget.

**That asymmetry is the strongest defensible finding in the pilots.** It is directly
relevant to real assay turnaround and it is exactly the kind of effect the
emergency-scientist framing needs. Variability is across pools; it is not biological
replication across independent proteins.

Structural allocation (8 ATLAS evaluation entries, held-out R3 MD replicate as
reference): static baseline 0.215 nm² RMSE / 0.688 top-10 residue overlap; both uniform
and uncertainty reach 0.152 / 0.775. Uncertainty reached that at 68 replayed CPU-seconds
against uniform's 101 — roughly 33% cheaper for equal accuracy — then plateaued and
drifted slightly worse (0.157). A weak but real effect.

---

## 3. Biological debrief: what the pilots actually contain

### 3.1 Binding arm

`SPG1_STRSG_Wu_2016` is the **B1 domain of streptococcal protein G (GB1)**, residues
265–280 of the 448-residue parent, four co-varying sites, 149,360 measured variants.
ProteinGym records `selection_assay: Binding (IgG)` and `selection_type: binding` — an
affinity-selection readout against IgG Fc, not a growth or fitness proxy. That is the
correct *class* of readout for display work, and the originating Olson/Wu lab lineage
used mRNA display (verify the exact platform before relying on this in writing).

It is nonetheless a bacterial immunoglobulin-binding domain with 20 canonical amino
acids at four positions. Not a peptide, not cyclic, no non-canonical residues.

### 3.2 Structural arm

All 12 ATLAS entries are **human proteins**, selected for ECOD family diversity with no
therapeutic intent. `hpc_pilots/README.md` states this directly: "structural controls,
not designated therapeutic targets."

- *Evaluation (8):* TFIIF subunit 2, platelet basic protein (CXCL7), retinoic acid
  receptor γ, SKP2, non-secretory ribonuclease, IGF-1, UBR5, major prion protein.
- *Development (4):* galectin-3-binding protein, Axin-1, moesin, MEF2A.

### 3.3 Assessment

The pilots validate decision machinery on biologically inert substrates. That was a
defensible staging choice — it let the system-level benchmark progress while SynX data
and lab access were being arranged, exactly as `RESEARCH_BRIEF.md` intended. It is now
the constraint, because every molecular deliverable (C01–C06) and both case studies
depend on a family and target choice that has not been made.

---

## 4. Pathogen landscapes already downloaded and unused

From `data/raw/proteingym_metadata.csv`:

| DMS id | Target | Assay | Mutants |
| --- | --- | --- | --- |
| `R1AB_SARS2_Flynn_2022` | **SARS-CoV-2 Mpro** (3CLpro, nsp5) | FRET, Growth | 5,725 |
| `SPIKE_SARS2_Starr_2020_binding` | **SARS-CoV-2 Spike RBD** | ACE2 binding | 3,802 |
| `SPIKE_SARS2_Starr_2020_expression` | SARS-CoV-2 Spike RBD | ACE2 binding | 3,798 |
| `A0A2Z5U3Z0_9INFA_Doud_2016` | Influenza HA (H1N1) | Viral replication | 10,715 |
| `C6KNH7_9INFA_Lee_2018` | Influenza HA (H3N2) | Viral replication | 10,754 |
| `NRAM_I33A0_Jiang_2016` | Influenza neuraminidase | — | 298 |
| `POLG_DEN26_Suphatrakul_2023` | Dengue NS5 | Viral replication | 16,897 |
| `A0A140D2T1_ZIKV_Sourisseau_2019` | Zika envelope | Viral replication | 9,576 |

Two implications:

- The Spike RBD binding set has 3,802 variants, slightly under one pool at the current
  `pool_size: 4096`. The 20-disjoint-pool design needs rework for the smaller
  landscapes — fewer pools, or resampling with documented overlap.
- Flynn 2022 and Starr 2020 are both SARS-CoV-2, which makes them the natural substrate
  for the COVID historical replay (B08) rather than a separate exercise.

---

## 5. Recommended targets

### 5.1 The constraint that should drive selection

RaPID / mRNA display requires a **purified, soluble, immobilizable target** —
typically biotinylated and bead-captured. Separately, macrocyclic peptides are generally
poor at crossing membranes, which is why the reading list carries PeptideCLM
(Feller & Wilke 2025) and the 13-method permeability benchmark (PMC12392647).

This creates a tension worth naming explicitly rather than papering over:

> The best-behaved viral reagents are the **soluble intracellular enzymes** (proteases),
> which carry a permeability liability. The **extracellular glycoproteins** need no
> permeability story but are harder reagents and sit under strong positive selection,
> which undermines any conservation-based family claim.

### 5.2 Family 1 — Coronaviridae (prototype SARS-CoV-2)

**Primary recommendation: Mpro / 3CLpro (nsp5).** The most de-risked option available.

- Soluble homodimer, inexpensive *E. coli* expression.
- Cys145–His41 catalytic dyad; standard FRET substrate assay for functional follow-up.
- High conservation across sarbecoviruses, workable across betacoronaviruses.
- Published macrocyclic peptide campaigns exist, including RaPID-derived work — verify
  specific citations before treating any as a positive control, but precedent is there.
- Public DMS already on disk (`R1AB_SARS2_Flynn_2022`, FRET readout).

The cytosolic liability is arguably an asset for this project: it forces a genuinely
multi-objective decision (affinity against permeability) rather than single-score
ranking, which makes the co-scientist's task more interesting and better matched to the
proposal's framing.

*Secondary options:* PLpro (nsp3) — soluble, deubiquitinase/deISGylase activity,
drug-tractable; nsp13 helicase.

*On Spike RBD:* best public data, weakest conservation argument. It is an immune-evasion
surface under strong positive selection and is therefore the wrong choice for a
family-transfer claim, despite the data quality. For a conserved extracellular
coronavirus target, the **S2 stem-helix / fusion peptide** region is far better
conserved across betacoronaviruses — harder reagent, no public DMS.

### 5.3 Family 2 — two options, and the obvious one is not the better experiment

**Henipavirus (Nipah)** is the better narrative:

- WHO priority pathogen under the R&D Blueprint prototype-pathogen framework.
- Genuinely extracellular targets — **G attachment glycoprotein** ectodomain
  (ephrin-B2/B3 binding) and **F fusion glycoprotein**. This matches the "extracellular
  peptides" remark from the Oct 5 discussion.
- A real conservation ladder for transfer testing: NiV-Bangladesh → NiV-Malaysia →
  Hendra → Cedar → Langya.
- Against it: prefusion F requires conformational stabilisation to be a usable display
  reagent, and there is no public DMS, so the labeled molecular-transfer tests
  (E2, E3) remain blocked.

**Influenza** is the better experiment:

- The **HA stem** is extracellular, well conserved within group 1 / group 2, and has
  strong peptide-binder precedent. **NA's catalytic site** is a viable alternative.
- Decisive advantage: HA landscapes for both H1N1 (Doud 2016) and H3N2 (Lee 2018) are
  already on disk, so the H1→H3 transfer test specified in E2 is runnable this week on
  existing data rather than waiting on SynX.

**Recommendation: Coronaviridae (Mpro) + Orthomyxoviridae (HA stem / NA).** Both
families have public labeled data, which unblocks E2 and E3 immediately. Nipah is
retained as the stretch robustness family — better story, no labels.

### 5.4 The single display round (D01, due Oct 23)

**Recommend Mpro.** Cheapest and best-behaved reagent, unambiguous primary endpoint
(enrichment, with FRET inhibition as an orthogonal follow-up if the schedule allows),
and interpretable failure. Report enrichment as enrichment; a single round does not
establish affinity, functional inhibition, or antiviral activity.

### 5.5 Ambitious alternative worth raising

The reading list already contains PMC12464979 — macrocyclic binders, covalent modifiers
and degraders of a **structured RNA** by mRNA display. The SARS-CoV-2 **frameshift
stimulatory element** is highly conserved and functionally essential, setting the
ORF1a/ORF1b ratio. An RNA-targeting macrocycle campaign would be the most novel element
available and has direct precedent in the project's own bibliography. Higher risk, less
standard reagent and assay. Worth putting to Askar on Oct 23 even if declined.

---

## 6. Next step for the co-scientist

This is the largest gap between what is built and what the proposal promises.

The agent's current decision state (`hpc_pilots/common.py`, `hpc_pilots/binding.py`) is:

```
observed_score_summary, candidate_summary[{candidate: <int>, prediction,
uncertainty_heuristic, diversity}], remaining_label_budget, pending_labels,
feedback_delay_rounds, decision_round
```

Candidate identities are **deliberately opaque integer pool indices** — a documented
design rule in `hpc_pilots/README.md`, correct for isolating the decision policy from
hidden labels. The action space is `{greedy, explore, diverse}`.

So what exists is a **bandit-strategy selector**. It reasons competently about budget and
uncertainty — yesterday's smoke rationale was sound for that task — but it contains no
biology, and it is several abstraction layers from the proposal's Part 1: pathogen intake
from genomic sequence → evidence retrieval → structure prediction → target and epitope
ranking → library design.

**The step that closes this is B04 + B06:** give the agent a decision state whose
contents are the *target dossier schema from core deliverable 2* — target identity,
structure and confidence, epitope definition, conservation, accessibility, essentiality
evidence, reagent tractability, key uncertainties. The action space then becomes what
`RESEARCH_BRIEF.md` already specifies: retrieve missing evidence, request a structure
calculation, revisit a target ranking, invoke a design tool, or recommend an experiment.

The budgeted-acquisition machinery already validated carries over unchanged as the
resource accounting beneath that policy. The Oct 12 work becomes foundation rather than
detour.

### Ordered actions

1. **Oct 19 (B07 opens).** Put Mpro + influenza to Askar as the two families, with
   existing ProteinGym availability as the argument. Gates C06, D01 and A06.
2. **Re-point the binding pilot at Mpro and HA.** Largely a change to
   `hpc_pilots/configs/today.json` (`binding.dataset`) plus pool-size rework for the
   smaller landscapes. The decision policy is dataset-agnostic. This converts the pilot
   from a methods demonstration into a result about pathogen targets.
3. **Write the dossier schema before building the policy that consumes it.** The schema
   *is* deliverable 2. B06 without it will produce another bandit.
4. **Pull the Rogers, Passioura & Suga (2018) nonproteinogenic DMS data** (reading list
   section B; PNAS 115(43), 10959–10964). A deep mutational scan of linear *and cyclic*
   peptides including non-canonical residues — the only labeled macrocycle dataset with
   ncAAs in the bibliography, and a far better substrate for the low-label ranking work
   in C05/E3 than GB1.
5. **Restore the calibration scripts** so section 2.2 reproduces from source.
6. **Close the agent arm** (40 binding episodes, structural agent replay) using the
   hosted-API path, which no longer needs the cluster queue.

---

## 7. Literature the current design draws on

Load-bearing in code today:

- **Lakshminarayanan, Pritzel & Blundell (2017), deep ensembles** (arXiv:1612.01474) —
  direct ancestor of using ensemble member dispersion as the uncertainty signal. The
  RF calibration study in §2.2 asks whether that holds for a 64-tree forest at small
  budgets. Answer: ranking yes, calibration no.
- **Gonzalez-Duque, "Do Gaussian Processes scale well with dimension?"** — justifies a
  forest-plus-UCB design rather than a GP, given one-hot features over 20 residue
  identities.
- **Rogers, Passioura & Suga (2018)**, nonproteinogenic deep mutational scanning, PNAS —
  conceptual basis for replaying a DMS landscape as a sequential assay. GB1 is the
  public stand-in for a display round.

Staged but untouched in code:

- **Section D** (RFpeptides, Latent-X, BoltzGen, miniprotein antagonists) — design track.
- **Section E** (Flex ddG, FakeRotLib, MakeRotLib, Dunbrack rotamer libraries,
  RosettaCommons ncAA interface protocol) — the entire physics-based ncAA track, C01–C04.
  Nothing in the repository approaches this yet.
- **Section C** (PeptideCLM, PepDoRA, PepMLM, PepPrCLIP, permeability benchmark) —
  becomes directly relevant if Mpro is selected, as the permeability axis.

---

## 8. Open items for A06 (Oct 23)

Carry these into the gate honestly:

- Calibration results in §2.2 are not reproducible from source (no script committed).
- LLM-agent evidence is a single smoke episode, not a result. Agent arms incomplete:
  binding 160/200, structure 3/4.
- No prototype family, target, assay or chemical alphabet selected. B07 is the blocker.
- A06 additionally depends on A04 (lab route, lead times, data return), A05 (live Tinker
  models and spend cap), E02 and H02 — all external to this repository.
- ATLAS data is CC BY-NC 4.0; the pilot is scoped to noncommercial research. Commercial
  use requires separate terms from the dataset owner.
