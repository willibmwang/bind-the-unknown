# Emergency AI co-scientist — project overview and plan

**Prepared 20 October 2026 for discussion on Thursday 22 October.** Week 4 of 10; the final
week is 30 November – 4 December.

This document is the project overview and plan agreed in our last conversation. It is written
to be read in one sitting and to be argued with. Three companion documents carry the detail:
[ARCHITECTURE.md](ARCHITECTURE.md) for the system, [W4_DEBRIEF_AND_TARGET_SELECTION.md](W4_DEBRIEF_AND_TARGET_SELECTION.md)
for the October results and target analysis, and
[output/TEN_WEEK_DELIVERY_PLAN.md](output/TEN_WEEK_DELIVERY_PLAN.md) for the 82-task register
with dependencies, owners and acceptance criteria.

**Nothing below asserts binding affinity, functional inhibition, antiviral activity or
therapeutic efficacy.** No macrocyclic peptide and no non-canonical amino acid has yet been
measured or modelled in this project. Where a number appears, §4 states what it was measured
on and what it does not transfer to.

---

## Summary

The project builds an **emergency AI co-scientist**: a system that helps experts decide what
to pursue, what to measure next, and how to revise as evidence arrives, under the conditions
that actually characterise an outbreak — incomplete pathogen knowledge, conflicting evidence,
hard deadlines, finite compute, and very limited experimental capacity. Macrocyclic peptide
discovery on SynX's mRNA display platform is the application and the experimental testbed.

Four weeks in, the position is this.

**The research question has sharpened.** It began as "can an agent make better scientific
decisions under emergency constraints?" The October work suggests a more specific and more
defensible claim: *the binding constraint on scientific-decision agents is not policy
architecture but the absence of verifiable decision environments at the volume needed to
train and evaluate one* — and public deep mutational scanning (DMS) landscapes are exactly
such environments, already existing in quantity, and not currently used that way. Every
variant's outcome is known, so a proposed measurement, ranking or stopping decision can be
graded deterministically with no human and no wet lab. §1.3 and §2.2 make the argument.

**Two results survive scrutiny; one did not.** Grading a decision system on what it *selected*
overstates it by 1.1–1.4× relative to grading it on what had actually *returned* when each
decision was made — this replicated on every dataset tried, including SARS-CoV-2 Mpro and both
influenza haemagglutinin landscapes, and it is the metric the project will use. A second
measurement-design finding is that improvement-over-random is bounded by the labelled fraction
of the pool and is therefore not comparable across pool geometries. Against that, an earlier
headline — roughly eightfold recovery of strong candidates over random selection — is
substantially an artefact of pool size and **will not appear in any paper or talk**. §4 gives
the arithmetic.

**The system now holds biology.** Until 19 October the software chose among three abstract
strategies over numbered candidates and contained no representation of the problem domain. It
now operates on a seven-axis target dossier with per-claim provenance, refuses evidence dated
after the decision under consideration, retains contradictory evidence alongside the claim it
contests rather than overwriting it, and states what a single selection round can and cannot
establish. §3 and [ARCHITECTURE.md](ARCHITECTURE.md) describe it.

**Both families you endorsed are adopted.** Coronaviridae via SARS-CoV-2 Mpro and
Orthomyxoviridae via the haemagglutinin stem or neuraminidase, with Nipah retained as a stretch
robustness case. The deciding consideration was public measurement data, which lets method
development proceed without a dependency on SynX. §5 sets out the rubric and §5.4 responds to
your point on target identification. The prototype-versus-priority pathogen comparison you
suggested is in
[PROTOTYPE_AND_PRIORITY_PATHOGENS.md](PROTOTYPE_AND_PRIORITY_PATHOGENS.md).

**Three external dependencies are overdue and none has an internal workaround.** A named SynX
experimental owner with a booked slot; the campaign export, without which the mandatory
platform-improvement deliverable cannot begin; and confirmed trainable-model access with an
approved spending ceiling. §8 states the consequence of each, and §10 asks for a decision on
each.

**On visibility.** You are right that the output has been sitting locally. §9 proposes a
concrete fix: repository access, a fixed weekly written cadence, and a standing rule that
every claim in a document names the file and run that produced it.

---

## 1. The problem and the research question

### 1.1 What "emergency" means operationally

The word does a lot of work and is usually left vague, so it is worth pinning down. Five
conditions distinguish outbreak-time research from ordinary research, and the project treats
each as a thing to be represented in the system rather than as background colour:

- **Incomplete knowledge of the pathogen.** At the point decisions matter most, there may be a
  genome and little else — no structures, no assays, no reagents, no literature.
- **Evidence that evolves and conflicts.** Preprints contradict each other; early reports are
  revised; a claim that was reasonable on Tuesday is wrong by Friday. A system that overwrites
  its prior belief loses the information that the question was ever contested.
- **Hard deadlines.** The value of a decision decays. A correct answer in twelve months is
  worth less than a defensible answer in three weeks.
- **Finite compute and tool cost.** Structure prediction, molecular dynamics and large-model
  inference all cost money and time that is being spent instead of something else.
- **Very limited experimental capacity.** This is the binding one. A laboratory can run a small
  number of selection rounds. Choosing what to put into them is the decision that dominates
  the outcome.

The last two mean the central object is not a prediction but an **allocation**: given what is
known now, what is the next action worth paying for? That framing is what makes the problem
tractable as a measurable research question rather than an aspiration.

### 1.2 The question

> Can an AI co-scientist make better, faster, evidence-backed scientific decisions under
> emergency constraints, and revise appropriately as evidence and experimental results arrive?

Three levels of evaluation follow from it, and they are deliberately kept separate so that a
failure at one does not contaminate the others:

1. **Scientific decisions (primary).** Target prioritisation, next-action choice, uncertainty
   handling, adaptation to evidence, and decision quality against time and cost.
2. **Molecular tools (supporting).** Reliability of structure, design and scoring tools;
   molecular ranking; chemical and structural transfer.
3. **End-to-end validation.** A historical COVID-19 reconstruction from January–March 2020
   evidence, two prototype-family case studies, and the required prospective display round —
   with conclusions limited strictly to what was actually measured.

### 1.3 Where the contribution sits

A tool-using agent coupled to a structure-prediction stack is not by itself a novel
contribution; [Agent Rosetta](https://arxiv.org/abs/2603.15952) already couples an LLM to
Rosetta and evaluates non-canonical design. The project needs a sharper claim, and the October
work points at one.

Building and evaluating a decision policy requires **environments in which decisions can be
graded**. Expert review is the obvious grader, but it is slow, expensive, and has
inter-rater-reliability problems that are themselves a project risk. The alternative is an
environment where the ground truth is already known and the grading is exact.

Public DMS landscapes are that. A landscape such as `R1AB_SARS2_Flynn_2022` records a measured
outcome for each of 5,725 Mpro variants. Hide the outcomes behind an oracle, expose only a tool
API, let a system request measurements under a budget, and every decision it makes can be
scored deterministically. Vary pool composition, budget, feedback delay, initial evidence,
corrupted evidence and conflicting evidence, and one landscape yields an effectively unbounded
supply of graded episodes. Ten such landscapes are already on disk (§5.2).

The claim is therefore: **auto-gradeable scientific decision environments can be manufactured
at volume from existing public measurement data, and doing so is what makes a trainable,
evaluable decision policy possible.** The emergency co-scientist is the system that
demonstrates it; the SynX display round is the prospective check that the decisions survive
contact with a real laboratory.

This also rescues the October pilot work. The GB1 study was built as a biology pilot, which is
why it looks thin as biology. As infrastructure it is the first working instance of exactly the
pattern above.

---

## 2. Why the project has this shape

This section is the "reasoning around why you made certain choices" you asked for. Each
subsection states a decision, the alternatives, and what would make me reverse it.

### 2.1 Why decision quality rather than molecular prediction

**The choice.** The primary measured object is the quality of the *decision procedure*, not the
accuracy of a molecular predictor.

**The reasoning.** The October calibration study is the load-bearing evidence. Acquisition
policy produced gains over random selection *while the underlying predictor was only weakly
informative* — Spearman ≈ 0.31 against truth at a 128-label budget, with plain ridge regression
matching a 64-tree random forest. If decision gains had required a good predictor, the emergency
premise would collapse, because at emergency timescales there are no labels with which to fit
one. That the gains came from the policy rather than the model is the substrate-independent
part of the result and the reason the project points here.

**The caveat, stated because it matters.** §4.2 shows this argument is weaker than it looked a
week ago. On the pathogen landscapes at a 25% labelled fraction the policy gain is small and
inconsistent. The claim that survives is narrower: policy choice measurably affects outcomes
under some geometries, and the *metric* used to assess it is systematically biased in a way
that can be corrected. That is a real contribution to how such systems are evaluated, and it is
smaller than what I would have claimed on 12 October.

**What would reverse it.** If the policy gain disappears entirely under matched budget fractions
across all landscapes, the contribution becomes purely methodological — the evaluation-design
findings — and the system work is reframed as applied engineering rather than a research claim.
That is a defensible outcome and I would rather reach it honestly in week 6 than discover it at
review.

### 2.2 Why public measurement landscapes as the substrate

**The choice.** Method development runs on public DMS data rather than on SynX campaign data.

**The reasoning.** Three things follow from it. First, **it removes a blocking dependency**: the
SynX campaign export has not arrived, and a plan whose method development cannot start without
it is a plan with a single point of failure. Second, **the outcomes are already known for every
candidate**, which is what makes automatic grading possible at all. Third, **it is releasable**:
code, benchmark definitions and evaluation data can be published without negotiating permissions,
which matters for the artifact deliverable and for the paper.

**The cost, honestly.** Public landscapes are single-substitution scans of canonical residues.
They contain no macrocycle, no non-canonical chemistry and no enrichment readout — the two
features that define the SynX platform. No result from them can speak to either. The
landscapes are a decision-environment substrate, not a model of the platform, and §4.3 keeps
that boundary explicit.

**What would change it.** The arrival of the campaign export, which would let the same episode
machinery run over real macrocycle measurements. The request is live and overdue.

### 2.3 Why a target dossier as the decision state

**The choice.** The system's state is a seven-axis target dossier with per-claim provenance,
not a feature vector.

**The reasoning.** Before 19 October the state was `observed_score_summary`,
`candidate_summary[{candidate: int, prediction, uncertainty, diversity}]`,
`remaining_label_budget`, `pending_labels`, `feedback_delay_rounds`, `decision_round`, with an
action space of `{greedy, explore, diverse}`. That is a bandit-strategy selector. It reasons
competently about budget and ignorance and contains no biology whatsoever, and it is several
abstraction layers away from what the project is supposed to deliver: pathogen intake from
genomic sequence → evidence retrieval → structure prediction → target and epitope ranking →
library design.

Three properties of the dossier are deliberate and are the ones I would defend hardest:

- **Per-claim provenance.** Every assertion names its source. An assertion with no source is
  marked as requiring verification rather than being silently trusted — the system currently
  flags its own claim about prior macrocycle campaigns against Mpro as uncited.
- **Temporal refusal.** Evidence dated after the decision under consideration is rejected. This
  is what makes the historical COVID-19 replay meaningful rather than a demonstration of
  hindsight.
- **Conflict retention.** Contradictory evidence sits alongside the claim it contests, with both
  timestamps, rather than overwriting it. An emergency decision-maker needs to know that a
  question is contested; a system that overwrites loses exactly that.

The third has an under-appreciated cost: the dossier grows monotonically within an episode, and
what gets evicted from it becomes a decision the policy itself has to make. That is an open
design question, named in §10.

### 2.4 Why macrocyclic peptides and mRNA display

This was given by the project context rather than chosen, but it is a good fit and the reasons
are worth stating because they shape target selection.

RaPID-style mRNA display generates libraries of order 10¹²–10¹³ by flexizyme-charged initiator
incorporation with spontaneous thioether macrocyclisation. The chemistry is largely
target-independent, which means **the reagent is the binding constraint, not the library** — a
target is tractable if SynX can obtain soluble, correctly folded, immobilisable protein that
survives several selection rounds, and not otherwise. That single fact dominates the selection
rubric in §5.1.

It also creates a tension the project should own rather than finesse: the best-behaved viral
reagents are soluble intracellular enzymes, which carry a membrane-permeability liability;
extracellular glycoproteins need no permeability argument but are harder reagents and sit under
strong positive immune selection, which undermines conservation-based family claims. Choosing
a protease and owning the permeability problem is the more defensible position, because
permeability is predictable in silico with tools already in the bibliography whereas
conservation of an immune-evasion surface cannot be engineered around.

### 2.5 Why these two families

Coronaviridae via Mpro and Orthomyxoviridae via the haemagglutinin stem or neuraminidase, per
your endorsement. Beyond that endorsement the pair was chosen because:

- **Both have public labelled data**, which unblocks the molecular transfer tests immediately.
  The H1→H3 grouped holdout — the exact distribution-shift test the brief specifies — is
  runnable on data already on disk.
- **Together they span the §2.4 tension deliberately.** One soluble intracellular enzyme with a
  permeability problem; one conserved extracellular epitope without one. That makes the pair a
  designed contrast rather than two arbitrary picks, which is a better case study structure.
- **Nipah is retained as a stretch robustness family.** Better narrative — WHO priority
  pathogen, genuinely extracellular, a real conservation ladder for transfer testing — and no
  public DMS, so it cannot carry method development.

### 2.6 What was considered and set aside

Recording these matters as much as recording what was chosen.

- **The structural arm is retired as a scientific component.** It measured how fast mean
  per-residue solvent-accessible surface area converges to a held-out MD replicate under
  different allocation policies. The effect was real but small, and the deeper problem is that
  agreement with a sibling simulation is not an epitope-accessibility measure in any sense that
  feeds target ranking. Nothing in the theory of change consumes it. Re-pointing it at viral
  glycoproteins would not fix that. The cost-accounting and evidence-isolation code is kept.
- **Spike RBD was rejected as the family narrative target** despite having the cleanest binding
  landscape available. It is an immune-evasion surface under intense positive selection and is
  therefore the wrong choice for any conservation or family-transfer claim. It stays as a
  method-development substrate only.
- **The SARS-CoV-2 frameshift stimulatory element** — a structured-RNA target with mRNA display
  precedent in the project's own bibliography — is the most novel option available and
  sidesteps the conservation problem entirely, because RNA secondary structure is under
  stronger constraint than a glycoprotein surface epitope. Against it: non-standard reagent and
  assay, uncertain functional readout, no public landscape. Worth ten minutes on Thursday even
  if declined.

---

## 3. System architecture

[ARCHITECTURE.md](ARCHITECTURE.md) is the full description with module-level detail. This is
the shape and the rationale.

### 3.1 Layers

The system is a Python package, `hpc_pilots`, with a single CLI entry point and a strict
separation between four concerns:

| Layer | Responsibility | Why it is separate |
| --- | --- | --- |
| **Domain** | Target dossiers, pathogen target packets, the non-canonical building-block registry | The biology must be representable independently of any policy that reasons over it. Writing the dossier schema *before* the policy that consumes it was deliberate; the reverse order produces a schema shaped by whatever the first policy happened to need. |
| **Environment** | Decision episodes built from measurement landscapes: hidden outcomes, timed evidence release, budgets, deadlines, delayed feedback | This is the auto-gradeable oracle of §1.3. It owns the hidden truth and exposes only a request interface. |
| **Policy** | The rules being compared: random, greedy, UCB, diversity-seeking, and an LLM-mediated policy over the six-action space | Policies are interchangeable behind one interface so that comparisons are matched by construction rather than by care. |
| **Evaluation** | Scoring, calibration studies, summarisation, provenance and resource accounting | Kept out of the policy path entirely, so a policy cannot read a metric it has not earned. |

### 3.2 The isolation boundary, which is the part that matters

The single most important property of the system is that **a policy cannot reach an answer it
has not earned**, and it is enforced structurally rather than by convention:

- The full outcome vector never leaves the evaluator. The policy interface accepts only revealed
  values, enforced at the function boundary in `pick_batch` and covered by a test.
- Top-fraction membership is never supplied to a policy.
- Candidate identities in LLM prompts are opaque integer pool indices, so a model cannot
  recognise a variant and recall its phenotype from pretraining.
- For the structural data, one entire MD replicate is held out and used only inside the
  evaluator.

This is the same boundary that the containerised oracle of §7 generalises. Building it first was
the right call: an evaluation harness whose isolation is a matter of discipline rather than
structure will leak, and a leak discovered in week 9 invalidates everything after it.

### 3.3 Reproducibility, and why it was worth the time

Roughly a fifth of the engineering effort went into provenance machinery. The justification is
that a result I cannot regenerate is a result I cannot defend at review:

- SHA-256 hashes for every downloaded input and every frozen split, in
  `data/processed/*/manifest.json`.
- Every result file embeds `dataset_manifest_sha256` and `config_sha256`, so any result ties
  back to the exact configuration and data that produced it.
- Atomic JSON writes with `allow_nan=False`, so a NaN cannot silently enter a results file and a
  partial write cannot be mistaken for a result.
- Smoke-test outputs carry a `smoke` flag and are excluded from scientific summaries by
  construction, not by remembering to filter them.
- An inference or analysis failure fails its task explicitly. **A failed LLM call never degrades
  into a baseline policy result** — this one has caught real bugs.
- Seeded tie-breaking before a stable argsort. This sounds like trivia and is not: the GB1
  landscape has large exactly-tied plateaus at low budget, and unseeded tie-breaking silently
  injects a diversity bonus into the greedy policy.

The calibration studies regenerate from committed scripts. One column does not reproduce
bit-exactly and this is recorded rather than hidden: the original ridge penalty was not
recoverable, so the committed script uses `RidgeCV` over a log-spaced grid, reproducing 31 of
160 rows exactly with mean absolute difference 0.0022 on Spearman. No quoted figure changes.

### 3.4 Model access and compute

The LLM policy speaks to any OpenAI-compatible backend selected by `PILOT_LLM_PROVIDER` — local
vLLM, OpenRouter, Anthropic or OpenAI — with per-provider request shaping. Actions are
constrained by a strict JSON schema; an action outside the allowed set, or a reply containing no
JSON object, raises rather than falling back. Hosted runs get a provider-specific policy label so
they can never pool with or overwrite local-model runs.

Compute runs on the MATS cluster under SLURM, with dependency-gated job chains and a submission
ledger so that scheduler state is recorded from scheduler evidence rather than inferred from the
fact that a job was staged.

---

## 4. What has actually been measured

Four weeks of evidence, separated into what survives, what does not, and what has not been
attempted. The detail and arithmetic are in
[W4_DEBRIEF_AND_TARGET_SELECTION.md](W4_DEBRIEF_AND_TARGET_SELECTION.md) §2.

### 4.1 Two findings that survive

**Selected-versus-returned.** Grading a decision system on everything it *selected* overstates
its performance relative to grading it on what had actually *returned* at the time each decision
was made. The discrepancy is 1.12–1.38× at a three-round feedback delay across three landscapes
and four policies, and exactly 1.00× at zero delay, as it must be. On GB1 it was roughly 1.4×.

This is not bookkeeping. It separates "were these good selections in hindsight?" from "were
these good selections on the evidence then available?" Only the second is a fair test of an
emergency decision-maker, and the first is what most evaluations measure inadvertently. It is
substrate-independent, endpoint-independent, and it transfers directly into the episode rubric.

**Improvement-over-random is bounded by the labelled fraction.** Random top-fraction recall is
simply the labelled fraction of the pool, confirmed to two decimal places:

| Pool | Labelled fraction | Predicted random recall | Observed | Headroom |
| --- | --- | --- | --- | --- |
| GB1, 4,096 candidates | 128/4096 = 0.031 | 0.031 | 0.032 | 32× |
| Pathogen, 512 candidates | 128/512 = 0.250 | 0.250 | 0.223 | 4.0× |

So a multiple-over-random is not comparable across pool geometries, and the honest comparison
reports recall at a matched *budget fraction* rather than a matched absolute budget. This is a
second evaluation-design finding of the same kind as the first.

### 4.2 One finding that does not survive

An earlier headline reported that the selection rule recovered roughly **eight times** more of
the best candidates than random selection. That figure is substantially an artefact of the
bound above. Once corrected, the advantage on pathogen landscapes is small and inconsistent:

| Landscape | Greedy vs random | UCB vs random | Diverse vs random |
| --- | --- | --- | --- |
| Mpro | +0.065 ± 0.152 (7/10 pools) | +0.038 ± 0.120 (5/10) | −0.027 ± 0.065 (3/10) |
| HA H3 | +0.104 ± 0.085 (8/10) | +0.050 ± 0.093 (6/10) | −0.031 ± 0.065 (1/10) |
| HA H1 | −0.004 ± 0.118 (6/10) | +0.046 ± 0.107 (7/10) | +0.019 ± 0.084 (3/10) |

Pool-to-pool spread exceeds the mean difference everywhere; diversity-seeking is actively
harmful on two of three landscapes. The likely cause is representational: a single-substitution
scan encoded as (position, residue) over 303–566 positions gives a forest far less to generalise
from than GB1's four co-varying sites, where every candidate shares positions with every other.

**The eightfold figure will not appear in any paper or presentation.** I am raising it
prominently rather than quietly dropping it, because the correction is itself the kind of result
the project is trying to produce, and because I would rather you hear it from me than find it.

### 4.3 What has not been measured at all

- **No macrocycle, and no non-canonical amino acid.** The building-block registry carries
  literature block *classes* only; every chemical identifier, codon assignment, Rosetta parameter
  and rotamer library is null pending the campaign export.
- **The LLM policy has no non-smoke result anywhere.** Only the four heuristic baselines have
  scientific runs on disk. The emergency episodes have been run with deterministic policies
  only, and those are stable under irrelevant evidence by construction, so the stability metric
  has not yet been tested against anything that could fail it. This is week 5 work and it is the
  most important near-term gap.
- **The physics-based non-canonical track is untouched** — Flex ddG, rotamer library generation,
  the RosettaCommons non-canonical interface protocol. Nothing in the repository approaches it.
- **No prospective experimental measurement**, which is what the display round is for.

---

## 5. Targets

### 5.1 The selection rubric

Seven axes, which are simultaneously the selection criteria and the schema of the decision state
in §2.3:

| Axis | What it asks | Why it gates |
| --- | --- | --- |
| **Reagent tractability** | Can SynX obtain soluble, correctly folded, immobilisable protein surviving multiple selection rounds? | Hard gate. An untractable reagent ends the campaign regardless of biology. |
| **Essentiality** | Does loss of function block replication? | A binder to a dispensable protein cannot become a therapeutic. |
| **Conservation** | Is the epitope conserved across the family, and under what selective pressure? | Gates every family-wide and prototype-pathogen claim. |
| **Accessibility** | Intracellular or extracellular; is the epitope exposed in the relevant conformation? | Determines whether a permeability argument is required. |
| **Assay availability** | Is there a functional orthogonal assay beyond enrichment? | Without one, a single round substantiates enrichment and nothing more. |
| **Public label availability** | Is there a DMS or binding landscape for method development? | Determines whether transfer tests run now or block on SynX. |
| **Precedent** | Prior macrocycle or peptide campaigns against this target? | Supplies positive controls and de-risks failure interpretation. |

### 5.2 The recommendation, and the data behind it

**Coronaviridae via Mpro; Orthomyxoviridae via HA stem or NA; Nipah as stretch.**

Mpro (3CLpro, nsp5) scores strongly on reagent tractability — routine soluble *E. coli*
expression, well behaved, inexpensive — on essentiality, on active-site conservation across
sarbecoviruses, and on assay availability via the standard FRET cleavage assay. It is weak on
accessibility, being cytosolic, and so requires a permeability argument. That liability may be
an asset for this project: it forces a genuinely multi-objective decision, affinity against
predicted permeability, which is a harder and more interesting problem for a co-scientist than
single-score ranking.

Ten pathogen landscapes are already on disk:

| DMS id | Target | Mutants |
| --- | --- | --- |
| `R1AB_SARS2_Flynn_2022` | SARS-CoV-2 Mpro | 5,725 |
| `SPIKE_SARS2_Starr_2020_binding` / `_expression` | SARS-CoV-2 Spike RBD | 3,802 / 3,798 |
| `A0A2Z5U3Z0_9INFA_Doud_2016` | Influenza HA (H1N1) | 10,715 |
| `C6KNH7_9INFA_Lee_2018` | Influenza HA (H3N2) | 10,754 |
| `NRAM_I33A0_Jiang_2016` | Influenza neuraminidase | 298 |
| `A4D664_9INFA_Soh_2019` | Influenza PB2 | 14,421 |
| `RDRP_I33A0_Li_2023` | Influenza PB1 | 12,003 |
| `POLG_DEN26_Suphatrakul_2023` | Dengue NS5 | 16,897 |
| `A0A140D2T1_ZIKV_Sourisseau_2019` | Zika envelope | 9,576 |

**For the single display round** the proposal is Mpro: the least expensive well-behaved reagent,
the strongest conservation case, an orthogonal functional assay if scheduling permits, and
interpretable failure. One round substantiates enrichment and must be reported as enrichment.

### 5.3 Prototype and priority pathogens

The comparison you suggested is in
[PROTOTYPE_AND_PRIORITY_PATHOGENS.md](PROTOTYPE_AND_PRIORITY_PATHOGENS.md), covering WHO, UKHSA,
NIAID, CEPI, HERA/ECDC, the US operational and regulatory lists, other national frameworks and
the academic literature — with the conceptual distinction between the two kinds of list, an
overlap matrix, and an assessment of whether "tractability and data availability" is a defensible
prototype-selection criterion or a convenience argument.

### 5.4 On the target-identification analogy

You wrote that the analogue of target identification for a novel pathogen would be identifying
targets across novel disease biology, and that this is not useful because pathogen-relevant
biology differs from other disease biology.

Taken, and the question is withdrawn. The practical consequence is that the project does **not**
pursue a general target-identification method transferable across disease areas. Target
identification here is pathogen-specific and proceeds from the seven axes in §5.1, which are
mostly virology-specific: polyprotein processing essentiality, conservation under immune
pressure, the reagent behaviour of a viral protease versus a class-I fusion glycoprotein. The
transfer claim the project does make is the narrower and better-evidenced one — *within* a viral
family, from a prototype to its relatives — which is the same claim the prototype-pathogen
literature makes, and §5.3 examines how well it is actually evidenced.

---

## 6. Evaluation design

Four system-level studies, each with a decision attached, so that a negative result changes the
plan rather than being absorbed:

| Study | Question | Decision it forces |
| --- | --- | --- |
| **S0** | Do deadlines, incomplete evidence and limited capacity materially change which actions are useful? | If constraints do not affect the task, the emergency-specific contribution is not real and the benchmark is revised before any claim is made. |
| **S1** | Does structured evidence tracking beat simpler workflows on support, uncertainty and dossier quality? | If the agent offers no gain over a scripted workflow, identify the missing capability before adding orchestration. |
| **S2** | Does the system revise when evidence warrants and hold steady when it does not? | A changed ranking is not automatically a better ranking; this separates evidence-following from instability. |
| **S3** | Does an adaptive policy use a constrained budget better than a fixed action order? | Pursue the policy contribution only if gains survive held-out episodes and a simple cost-aware baseline. |
| **S4** | Do the recommendations survive a real laboratory? | The prospective check; predictions frozen before results arrive. |

Four commitments govern all of them. **Baselines are matched** — a fixed rubric, a scripted
workflow, retrieval with a single LLM response, and the adaptive agent, on identical document
access, tool capabilities and molecular models. **Splits are grouped, not random** — by parent
peptide, backbone and chemical similarity group, because a strong random-split result with weak
grouped transfer supports an interpolation claim and nothing more. **Prospective predictions are
frozen** before measurements arrive, with all submitted candidates, selection rules, failures and
controls recorded. **Hindsight leakage is audited** — date-limited retrieval does not remove
knowledge already in model weights, and the historical COVID-19 claims will state that limit
explicitly.

---

## 7. The plan

### 7.1 Required outputs, and where each is satisfied

No stretch goal has been dropped. If a collaborator-dependent obligation fails it is recorded as
incomplete rather than replaced with a simulation.

| Required output | Where it lands | Status |
| --- | --- | --- |
| Modular emergency co-scientist | Decision state and action space delivered 19 Oct; integrated replay W5; demo frozen W6 | Partially delivered |
| Evidence-backed target dossiers | Seven-axis packets for three families exist; both chosen families finalised W5 | Partially delivered |
| Non-canonical macrocyclic designs and focused library | Block registry exists as structure; chemistry blocked on campaign export | **Blocked** |
| Historical COVID-19 evaluation | W5, with hindsight audit | Planned |
| Two prototype-family case studies | W5–W6, scored outputs | Planned |
| SynX platform improvement | Blocked on campaign export; re-scope decision at Thursday's gate | **Blocked** |
| One mRNA display selection round, analysed | W7 execution; readout due 18 Nov; analysis by 20 Nov | **Blocked on slot** |
| Paper, reproducible artifact, presentation | W10 | Planned |

### 7.2 Remaining shape of the ten weeks

| Weeks | Focus | Gate |
| --- | --- | --- |
| **W4 · Oct 19–23** | Development episodes; evidence-state ledger; family sign-off; round question | Feasibility and novelty gate, 23 Oct, with the export blocker named |
| **W5 · Oct 26–30** | LLM policy over the six-action space; both-family dossiers; historical replay; expert rubric | Platform-improvement choice, 27 Oct |
| **W6 · Nov 2–6** | Held-out system evaluation; **prospective predictions frozen 4 Nov**; London readiness | Readiness gate, 6 Nov |
| **W7 · Nov 9–13** | London: facility and design review, the actual selection round, expert feedback | Round executed |
| **W8 · Nov 16–20** | Training harness; throughput and spend gate; **lab readout due 18 Nov** | Throughput gate, 19 Nov |
| **W9 · Nov 23–27** | Forecasts saved before held-out reveal; analysis; independent review | **Results freeze, 27 Nov** |
| **W10 · Nov 30–Dec 4** | Clean reproduction, manuscript, artifact, rehearsals, final handoff | Handoff, 4 Dec |

The register holds 82 tasks across eight streams and 273 estimated hands-on hours, of which 14
are delivered. A task reads `Planned` unless a status line names its evidence; only four
currently do. The generator checks for missing and cyclic dependencies and for window ordering,
and fails rather than emitting an inconsistent chart.

### 7.3 Capacity, stated plainly

**Week 4 does not fit.** Fourteen hours went into the domain layer on 19 October, which was not
in the baseline. The remaining 38.5 hours still include the molecular audit path, which cannot
start until the campaign export arrives. If the export has not arrived by 21 October the
molecular path slides to week 5 and the mandatory platform improvement loses its slack. That is
a blocker with an owner, not something to absorb silently.

Weeks 5 and 6 run at 38–40 hours against a 40-hour week — effectively no slack. Weeks 8 to 10
carry 6–12 hours of buffer each, deliberately reserved for API retries, late experimental QC and
reproduction fixes. Week 7 is on-site review and observation; it is not additional coding
capacity. Week 9 includes US Thanksgiving, so the results reviewer should be booked before
London or the review pulled to 25 November.

---

## 8. Risks

| Risk | Current state | Consequence | Mitigation |
| --- | --- | --- | --- |
| **No booked display round** | Overdue since 16 Oct. No named SynX owner, no slot. | The one mandatory experimental deliverable does not happen. Backward planning from a usable 18 Nov readout is already tight. | Needs a name and a date this week. No internal workaround exists. |
| **Campaign export not received** | Overdue. | The platform-improvement deliverable cannot begin; the molecular audit path is entirely blocked. Most exposed dependency in the project. | Re-scope decision at Thursday's gate: either the export arrives, or the platform improvement is re-specified against something available. |
| **Trainable model access unconfirmed** | No confirmed account, model list or spend ceiling. | Cannot commit to model identities or an experiment matrix. | Verify live availability; the local pilot model is not automatically a supported training model. |
| **Policy gain may not survive correction** | §4.2. Small and inconsistent on pathogen landscapes. | The headline claim shrinks to the evaluation-design findings. | Report matched budget fractions; decide pool geometry at Thursday's gate; frame the methodological contribution as primary. |
| **Expert rubric reliability** | Not yet measured. | If inter-rater reliability is poor, the primary evaluation is unreliable. | This is precisely why the auto-gradeable environments matter: they give a domain where ground truth is exact rather than assessed. |
| **Scope** | Eight workstreams, ten weeks, one person plus collaborators. | Thin delivery across everything rather than one solid result. | One paper narrative rather than two; mandatory outputs protected; a negative result retained and the claim narrowed, rather than the deliverable dropped. |

---

## 9. Working arrangements — addressing the visibility problem

You are right that the output has been sitting locally, and that is a problem worth fixing
properly rather than by sending more attachments. Proposed:

1. **Repository access.** The work is a git repository with full history — code, configurations,
   frozen splits, result manifests and the documents. I can grant access to whatever is most
   convenient for you, or mirror it to a private remote. Happy to do this immediately; I have
   not done so unilaterally because I would rather agree where it lives and who else sees it.
2. **A fixed weekly written update**, delivered the day before our call: what was decided, what
   was measured, what changed, what is blocked, and what I need. Short, and in the same honest
   register as §4.
3. **Claim-to-evidence traceability as a standing rule.** Every number in a document names the
   file and run that produced it, and anything asserted from memory is marked as requiring
   verification until a citation exists. There is one such flag outstanding right now — the
   claim about prior macrocycle campaigns against Mpro.
4. **Gate decisions in writing**, with the options, the recommendation and the reasoning, so a
   decision you make is recoverable later.

---

## 10. For Thursday

Ordered by how much they block.

**Decisions I need from you**

1. **The three overdue dependencies.** A named SynX experimental owner and a booked slot; the
   campaign export or an agreed substitute; confirmed trainable-model access with a spend
   ceiling. Each is past its date, each is outside my control, and the second and third both
   change the plan depending on the answer.
2. **The display-round question.** Mpro confirmed, or an alternative? This gates library design
   and everything downstream of it.
3. **Pool geometry.** Many small pools with compressed headroom, or few large pools with no
   pool-to-pool spread? There is no free choice at 5,725 rows; it is a genuine trade-off and I
   would like your view before fixing it.
4. **Scope.** Eight streams in six remaining weeks is aggressive. If something should be cut
   rather than delivered thinly, Thursday is the time to say so.

**Things I would value your judgement on**

5. **Is the sharpened contribution the right one?** "Verifiable decision environments
   manufactured from public measurement data" versus "an agent that helps scientists decide."
   The first is more defensible and less obviously useful to a clinician; I think it is right
   but I would like to be argued with.
6. **Is tractability a legitimate prototype-selection criterion?** The two families were chosen
   substantially because public data exists for them. §5.3 examines whether the literature
   supports that or whether it is convenience dressed as principle.
7. **The dossier eviction problem.** Evidence accumulates monotonically within an episode and
   conflicts are retained rather than overwritten, so context eventually has to be dropped —
   and what to drop is itself a decision the policy should be making. I do not have a good
   answer and it is a genuinely interesting design question.
8. **The frameshift element**, §2.6. Ten minutes, probably declined, worth asking.

**What I will bring**

The running system with a walkthrough of a single episode end to end; the 240 paired replays
behind §4; and the corrected arithmetic on the eightfold figure.
