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

The project is about **macrocyclic peptide drug discovery**: designing macrocycles that bind a
chosen viral target on SynX's mRNA display platform, well enough that the design process is an
improvement on what the platform does today. The primary research object is **molecular
prediction** — can we predict which macrocycles bind, and use that to design better libraries?
An emergency AI co-scientist sits above that as a sub-arm, deciding which target to pursue and
what to measure next when experimental capacity is the binding constraint.

Four weeks in, the position is this.

**The most important thing in this document is a problem, not a result.** Every
molecular-prediction deliverable is blocked on the SynX campaign export, which is overdue. Four
weeks in, **there is no macrocycle and no non-canonical amino acid anywhere in the repository** —
the building-block registry exists but every chemical field in it is null, and the physics track
is untouched. The work that could proceed without the export was the decision arm, so that is
what got built, and earlier drafts of this document then described the sub-arm as the
contribution. That was an inversion and §2.1 corrects it. The practical consequence is that the
export is the critical path rather than a risk-table row.

**The fix that does not need SynX, now in progress.** The bibliography contains one labeled
dataset of the right molecule class — Rogers, Passioura & Suga, *PNAS* 2018, a deep mutational
scan of linear **and cyclic** peptides **including non-canonical residues**. It was recorded as
an action in the Week 4 debrief and never done. Acquiring and characterising it is under way; it
would let binding prediction start on macrocycles with ncAAs this week rather than on the wrong
molecule class.

**Three of the project's headline claims failed this week, including one I made yesterday.**
The eightfold recovery over random selection is an artefact of pool size. The
selected-versus-returned ratio — until this morning the one result I thought had replicated
everywhere — is an accounting identity rather than a measurement: a feedback-blind policy
reproduces the whole band, and the matched comparison of delayed against undelayed feedback is
null. And my own claim from yesterday, that the gain tracks combinatorial encoding rather than
pathogen identity, was a mis-attribution with a circular control group. Each is now a permanent
check in the code rather than a note in a document. §4 gives the arithmetic and §4.3 says what
replaced them.

**What survives, read as a result about prediction rather than about choosing.** Binding is
learnable from a small number of labels **only when the candidates share sequence coordinates
with the labelled ones** — established by intervention in both directions, not by correlation,
then replicated on two unrelated proteins against a prediction committed before the replays ran
(predicted +0.25 and +0.24, observed +0.254 and +0.254, every falsifier passed; exact permutation
p = 1/70). The useful part for this project is that the favourable regime is *a randomised region
of a few positions over an alphabet* — which is what a RaPID library is. A weak predictor
suffices, Spearman ≈ 0.31, but there must be one, and whether one can be fitted is computable
from the library's design before any selection is run. The limits in §4.2 are real, and the
largest is that none of this is yet on macrocycles.

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

**Three external dependencies are overdue and none has an internal workaround.** In order of
damage: **the campaign export**, without which the primary molecular arm and the mandatory
platform-improvement deliverable cannot begin; a named SynX experimental owner with a booked
slot, without which the one display round does not happen; and confirmed trainable-model access
with an approved spending ceiling. §8 states the consequence of each and §10 asks for a decision
on each. If only one thing is resolved on Thursday it should be the export — and a minimal
subset with the chemistry definitions would unblock more than a complete export arriving in
three weeks.

**On visibility.** You are right that the output has been sitting locally. §9 proposes a
concrete fix: repository access, a fixed weekly written cadence, and a standing rule that
every claim in a document names the file and run that produced it.

---

## 1. The problem and the research question

### 1.1 The goal

The project is about **macrocyclic peptide drug discovery**: designing macrocycles that bind a
chosen viral target, on SynX's mRNA display platform, well enough that the design process
itself is an improvement over what the platform does today. The primary research object is
therefore **molecular prediction** — can we predict which macrocycles bind, and use that to
design better libraries?

Everything else in this document is in service of that. The emergency co-scientist (§1.3) is a
sub-arm: it decides *what to measure and which target to pursue* when experimental capacity is
the binding constraint. It matters because the project gets approximately one display round,
and choosing what goes into it determines what the round can establish. But it is the
allocation layer above the molecular work, not a replacement for it.

### 1.2 Why the molecular prediction problem is hard

Macrocyclic peptides sit in a gap between the two classes of molecule that prediction methods
are good at. They are not small molecules and not proteins, and almost every tool the field
offers assumes one or the other.

- **The chemistry has no settled representation.** A RaPID-style library is built by
  flexizyme-charged initiator incorporation with spontaneous thioether macrocyclisation, over an
  alphabet that includes non-canonical residues — N-methylated backbones, D-amino acids,
  β-amino acids. A 20-letter alphabet cannot express that, and neither can a model trained only
  on natural proteins. Every public landscape currently staged in this project is canonical
  residues on a folded domain, which is the wrong molecule class.
- **Cyclisation changes the problem.** Conformational entropy, the backbone constraint and the
  absence of free termini all matter for binding, and are exactly what sequence-only models
  ignore.
- **The scores available are not the quantity of interest.** Rosetta energies are not measured
  binding free energies, and a contact-distance filter is a geometric criterion rather than
  evidence of binding. Display selection returns *enrichment*, which is not affinity, which is
  not functional inhibition, which is not antiviral activity. Four distinct endpoints get
  conflated routinely and the project's rule is to report each as itself.
- **Labels are scarce and selected.** Measured candidates are typically the ones somebody chose
  to measure, so the labeled set is a biased sample of the design population.
- **And there is a membrane problem for the target we favour.** Macrocycles cross membranes
  poorly, and Mpro is cytosolic, so affinity alone is not the objective — predicted
  permeability has to be traded against it (§2.4).

So the research question underneath the application is: **what does it take to predict binding
for this molecule class, from few labels, well enough to design a library?** That question is
answerable with public data on the generic part — sequence-to-binding learnability — and needs
SynX chemistry for the part specific to the platform.

### 1.3 The emergency constraint, and where the co-scientist sits

The project's framing is pandemic preparedness, which adds a constraint rather than changing
the goal: the molecular work has to be doable *fast*, on a pathogen nobody has worked on, with
very little data. That rules out methods needing a large labeled set for the new target, and it
is why few-label prediction is the question rather than large-scale supervised fitting.

Within that, the co-scientist arm handles the allocation decisions — which target, which
candidates to measure next, when the evidence warrants revising a ranking. Five conditions
distinguish outbreak-time work and the system represents each explicitly rather than treating
it as background: incomplete pathogen knowledge; evidence that evolves and conflicts; hard
deadlines; finite compute and tool cost; and very limited experimental capacity. The last is
the binding one, and it is what makes the allocation layer worth building at all.

### 1.4 What "emergency" means operationally

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

### 1.5 The questions, in priority order

**Primary — molecular.**

> Can we predict binding for macrocyclic peptides with non-canonical residues, from few labels,
> well enough to design a better library than the platform designs today?

**Secondary — allocation.**

> Given that capacity, can a system decide well which target to pursue and what to measure
> next, and revise appropriately as evidence arrives?

Three levels of evaluation follow, kept separate so a failure at one does not contaminate the
others:

1. **Molecular prediction (primary).** Does a defined score rank measured outcomes within
   matched assay conditions? Does it transfer to new backbones and chemical classes? Do a few
   labels beat simple rescaling? Is there a measured failure mode that justifies a specific
   platform improvement?
2. **Scientific decisions (supporting).** Target prioritisation, next-action choice, uncertainty
   handling, adaptation to evidence, and decision quality against time and cost.
3. **End-to-end validation.** A historical COVID-19 reconstruction from January–March 2020
   evidence, two prototype-family case studies, and the required prospective display round —
   with conclusions limited strictly to what was actually measured.

### 1.6 Where the contribution sits

The honest answer is that it is not yet settled, and I would rather say so than pick one and
defend it.

The molecular claim is the one the project is *for*: that binding for this molecule class can be
predicted from few labels well enough to design a better library. It is the right contribution
and it is the one that is not yet started, because the chemistry is blocked (§8.1). The public
ncAA macrocycle set is the route to starting it without SynX.

The decision claim is the one the project can currently support. A tool-using agent coupled to a
structure stack is not itself novel — [Agent Rosetta](https://arxiv.org/abs/2603.15952) already
couples an LLM to Rosetta and evaluates non-canonical design — so the sharper version is about
*environments*: public measurement landscapes can be turned into auto-gradeable decision
environments at volume, which is what makes a decision policy trainable and evaluable at all.
That is a real claim and this week produced evidence for it. It is also the sub-arm.

My current view, for arguing with on Thursday: the paper is a **macrocycle design** paper in
which the decision layer is a component, and the environment-generation work is the methods
contribution that makes the evaluation credible. If the export does not arrive, that inverts by
force rather than by choice — which is a reason to treat §8.1 as urgent rather than as
housekeeping.

---

## 2. Why the project has this shape

This section is the "reasoning around why you made certain choices" you asked for. Each
subsection states a decision, the alternatives, and what would make me reverse it.

### 2.1 How the molecular and decision arms relate — and a correction

**The choice.** The primary measured object is **molecular prediction for macrocyclic peptides**.
The decision arm sits above it, allocating the scarce experimental capacity that the molecular
work consumes.

**A correction to how I have been describing this.** Earlier drafts of this document inverted
that, presenting decision quality as the primary contribution and molecular work as supporting.
That was wrong, and the drift has a cause worth naming: **every molecular-prediction deliverable
is blocked on the SynX campaign export**, so the only work that could proceed was decision-policy
work on public data. Four weeks in there is still no macrocycle and no non-canonical residue
anywhere in the repository (§4.5). The consequence of taking the goal seriously is that the
export is not a row in a risk table — it is the critical path, and §8 now says so.

**What the decision arm is for.** The project gets approximately one display round. Which target
it is run against, and which candidates go into the library, determine what the round can
establish. That is a real allocation problem and it is worth doing well, but it is the layer
above the molecular question rather than a substitute for it.

**What the decision work found, read as a molecular-prediction result.** This week's finding was
written up as being about acquisition policy, and under the corrected framing it reads more
usefully than that. What it actually establishes is a condition on **when binding is learnable
from few labels**: a weak model fitted on 128 measurements predicts held-out binding well enough
to be worth acting on *only when the candidates share sequence coordinates with the labelled
ones*. Where they do not, in-pool Spearman sits at 0.08–0.12 and nothing downstream helps. That
is a statement about the predictor, not about the chooser, and §4.2 gives the intervention tests.

**It points the right way for library design.** The favourable regime is a randomised region of
a few positions over an alphabet — which is what a RaPID library *is*. The highest in-pool
learnability and the largest measured gain in the project both come from the four-site randomised
GB1 library, the closest on-disk analogue to a display library, from a surrogate at Spearman 0.31.
A weak predictor is enough; but there must be one, and whether one can be fitted is a computable
property of the library's design, available before any selection is run. For a project whose job
is to design libraries, that is a more directly useful result than a claim about decision
procedure would have been.

**What it does not do**, and this is the limitation that matters most: every landscape behind it
is canonical residues on a folded protein domain. None contains a macrocycle, a non-canonical
residue or a cyclisation, so none of it speaks to the molecule class the project exists to design.
§2.2 is about closing exactly that gap.

### 2.2 Getting onto the right molecule class without waiting for SynX

**The choice.** Method development runs on public measurement data rather than on SynX campaign
data — but it has to move onto **macrocycles with non-canonical residues**, which the landscapes
staged so far are not.

**Why public data at all.** Three reasons, all still good. It removes a blocking dependency: a
plan whose method development cannot start without the campaign export has a single point of
failure, and four weeks of a ten-week project have now demonstrated that. The outcomes are known
for every candidate, so a predictor can be scored exactly. And it is releasable, which matters
for the artifact deliverable and the paper.

**The cost, stated plainly.** Everything staged to date is canonical amino acids on a folded
protein domain. No macrocycle, no non-canonical residue, no cyclisation, no enrichment readout —
which is to say, none of the features that define the molecule the project designs. These
landscapes support claims about sequence-to-binding learnability in general. They support no
claim about macrocyclic chemistry, and §4.5 keeps that boundary explicit.

**What is being done about it, this week.** The bibliography contains one labeled dataset of the
right class — Rogers, Passioura & Suga, *PNAS* 2018;115(43):10959–64, a deep mutational scan of
**linear and cyclic peptides including non-canonical residues**. It needs no SynX data. It was
recorded as an action in the Week 4 debrief and never done, which in hindsight is the single
clearest symptom of the inversion described in §2.1. Acquiring and characterising it is now in
progress, and §10 asks what to do if it turns out to be unusable.

**What would still change things.** The campaign export, which is the only route to the
platform's own chemistry — its building-block catalogue, codon mappings, stereochemistry and
existing Rosetta parameters. Public ncAA data lets the prediction work start; it does not
substitute for the platform-improvement deliverable.

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
| **Environment** | Decision episodes built from measurement landscapes: hidden outcomes, timed evidence release, budgets, deadlines, delayed feedback | This is the auto-gradeable oracle of §1.6. It owns the hidden truth and exposes only a request interface. |
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

Four weeks of evidence, separated into what has been withdrawn, what survives, and what has not
been attempted at all. This section supersedes §2 of the
[Week 4 debrief](W4_DEBRIEF_AND_TARGET_SELECTION.md), which is left unrevised as the record of
what was believed on 19 October and carries a correction notice at its head. The intervention
tests behind §4.2 are set out in
[ADJUDICATION_COMBINATORIAL_GAIN.md](ADJUDICATION_COMBINATORIAL_GAIN.md), and the replication
prediction in [PREREGISTRATION_COVERAGE.md](PREREGISTRATION_COVERAGE.md).

### 4.1 Three claims withdrawn

I am leading with these rather than burying them. Each was a number I believed, two of them
appear in the Week 4 debrief you may already have read, and one of them I generated yesterday.
All three are now permanent checks in the code, so they cannot quietly return.

**The eightfold gain over random (withdrawn 19 October).** Random top-fraction recall is simply
the labelled fraction of the pool, so an improvement-over-random is bounded by pool geometry:
32× of headroom at 128 labels from 4,096 candidates, but only 4× at 128 from 512. The figure
measured the pool, not the policy. A related error was found yesterday in the correction itself:
the attainable multiple is `min(1/top_fraction, 1/labelled_fraction)`, not `1/labelled_fraction`.
The two agree at the final checkpoint of every configured landscape, so no reported figure
changes, but the old formula overstated headroom precisely at the early checkpoints whose raw
multiples look most impressive.

**The selected-versus-returned ratio (withdrawn 20 October).** This was the project's best
result and the one I was most confident in. It is an identity. `returned` is always a subset of
`selected` and recall divides both by the same pool-wide hit count, so the ratio is monotone in
set inclusion — across 1,018 recorded delayed checkpoints it never once falls below 1. What it
reports is the share of the budget still in flight: with 128 labels in batches of 8 at a
three-round delay, 24 are pending at the final checkpoint and 128/104 = 1.23 falls out of the
bookkeeping before any policy acts.

The decisive check is the random policy. It reads no returned label, so its selections are
identical at every delay — confirmed in 99 of 99 pools — and yet it reproduces the entire band:

| | Mean ratio | SD | n |
| --- | --- | --- | --- |
| Random (feedback-blind) | 1.297 | 0.498 | 89 |
| Greedy / UCB / diverse (informed) | 1.312 | 0.356 | 279 |

A quantity a feedback-blind policy can produce is not a measurement of how delay degrades a
decision. And the matched comparison that *would* measure it — same budget, same metric, delay 0
against delay 3 — is null: mean paired difference −0.006 across 392 pool comparisons, Wilcoxon
p = 0.31 on the 243 non-zero pairs. **The 1.1–1.4× figure must not be presented as a finding.**

**That the gain tracks combinatorial encoding (withdrawn 20 October).** Mine, from yesterday's
runs. Two failures. The control group was circular: `spike_rbd_ace2` *is* SARS-CoV-2 spike RBD,
so across the six staged landscapes "viral protein" and "single-substitution scan" are the same
two-versus-four partition, and any statistic separating one separates the other identically.
And the explanatory variable was wrong — combined substitutions are collinear with, but not the
cause of, the quantity that actually moves the gain. Intervention breaks it in both directions
(§4.2). It is the same failure mode as the eightfold figure: a property of pool geometry read as
a property of the biology.

### 4.2 What survives — a mechanism, with an intervention test

Acquisition gain is governed by **how densely the candidate pool covers the variable positions
the surrogate must learn from** — that is, by whether a weak model can be fitted *inside* the
pool at all. This is a computable property of pool geometry, available before any assay is run.

Correlation would not establish that, so it was tested by intervention in both directions,
holding everything else fixed:

| Intervention | Greedy gain | Surrogate Spearman |
| --- | --- | --- |
| Influenza H3 scan, pools restricted to 28 variable positions | +0.104 → **+0.250** (8/8 pools) | 0.10 → 0.63 |
| …to 55 / 150 / 566 positions | +0.159 / +0.019 / 0.000 | 0.42 / 0.18 / 0.10 |
| GB1 doubles re-encoded so pool members share no feature levels | +0.242 → **−0.019** | 0.66 → 0.03 |

The first takes a pathogen protein on a replication-fitness endpoint — the very case that
appeared to fail — and recovers the full gain by narrowing the pool. The second keeps the same
variants, the same hidden scores and the same combinatorial structure, destroys only the feature
sharing, and removes the gain entirely. Combined substitutions are neither necessary nor
sufficient.

It is a *surrogate* effect: the surrogate-free diverse policy shows no systematic gain on any
landscape (−0.042 to +0.058 across all eight), and it requires at least 64 of 512 labels —
absent at 8, 16 and 32.

**It then held on two unrelated proteins, against a prediction written down first.** The
mechanism's weakest point was independence: it rested on one independent protein plus a
restaging of the pilot protein. Two further landscapes were staged — GRB2 SH3 domain and
KRAS against the DARPin K55 binder, neither related to anything already in the set, the second
carrying a binding rather than a fitness endpoint. The predictor (in-pool surrogate Spearman at
128 labels), the thresholds and the falsifiers were recorded in
[PREREGISTRATION_COVERAGE.md](PREREGISTRATION_COVERAGE.md) and committed **before** the replays
ran. Predicted ≥ +0.15 improving in ≥ 7 of 10 pools, with point estimates of +0.25 and +0.24;
observed +0.254 and +0.254. Every criterion passed, including diverse staying within ±0.05 of
zero on both.

Running every staged pool — no cell now rests on a selected half — the comparison is:

| Landscape | In-pool Spearman | Pools | Greedy gain | Pools improved |
| --- | ---: | ---: | ---: | ---: |
| GRB2 SH3 | 0.46 | 20 | **+0.265** | 19/20 |
| YAP1 WW | 0.35 | 18 | **+0.263** | 17/18 |
| GB1-Olson (matched) | 0.70 | 20 | **+0.244** | 17/20 |
| KRAS–DARPin | 0.68 | 20 | **+0.194** | 15/20 |
| Mpro | 0.09 | 10 | +0.065 | 7/10 |
| Influenza HA H3 | 0.08 | 20 | +0.060 | 13/20 |
| Spike RBD | 0.11 | 6 | +0.030 | 3/6 |
| Influenza HA H1 | 0.12 | 19 | −0.004 | 10/19 |

The groups do not overlap — lowest high-coverage +0.194 against highest low-coverage +0.065 —
and the exact one-sided permutation test over landscape labels now gives **p = 1/70 = 0.014**,
against a floor of 0.067 before these two were added. Finishing the partial sweeps was not
cosmetic: H3 fell from +0.104 on ten pools to +0.060 on twenty, and KRAS from +0.254 on ten to
+0.194 on twenty.

**The relevant analogue for this project is better than either.** The four-site GB1 library
(Wu 2016) is a randomised region over an alphabet, every member carrying a value at all four
randomised positions, members mutually distant (mean pairwise Hamming 0.95). At the same
512/128/5% geometry it gives **the largest gain measured anywhere in this work — greedy +0.373
and UCB +0.373, 10 of 10 pools — from a surrogate with Spearman 0.31.** What matters is coverage
of the randomised positions, not proximity to a wild-type reference, and a randomised macrocycle
region of a few positions sits at the favourable extreme of that variable.

**Six things that must travel with this.**

- It is an argument about pool structure, not a measurement of macrocycles. No landscape here
  contains a macrocycle, a non-canonical residue or a cyclisation.
- Coverage is necessary but not sufficient. Mpro reaches Spearman 0.68 with greedy gain −0.02;
  GB1 singles reach 0.61 with no gain. The surrogate must also resolve the top tail. Two
  conditions of unknown relative weight, not a rule.
- The gain decays steeply with pool size at fixed budget: normalised gain 0.58 → 0.22 → 0.09 →
  0.035 at 512 → 2,048 → 8,192 → 32,768 candidates. It therefore describes focused pools of
  hundreds to a few thousand — deconvolution and focused maturation — and says **nothing** about
  a selection round over a 10¹¹-member library. Over that same sweep the *multiple* over random
  rises from 2.9× to 12.9× while the policy loses almost all its real advantage, which is
  exactly how the eightfold figure misled. No multiple over random will be quoted again.
- Eight landscapes is still eight. The permutation test is at its floor (rank 1 of 70), so 0.014
  is the smallest p this design can produce, not a measurement of how much smaller it might be.
  The ± spreads quoted per landscape are variation across pools of one protein, not evidence
  about the landscape class.
- **This does not separate coverage from combinatorial structure.** Both new landscapes are
  multi-substitution and high-coverage, so the withdrawn hypothesis predicts the same outcome.
  It tests independence only. Separating the two needs the within-protein comparison — PDZ
  doubles against the already-staged PDZ singles with coverage matched — which is not yet run.
- Coverage is also not the whole story in the other direction: KRAS has the second-highest
  in-pool Spearman (0.68) and the smallest gain of the four (+0.194), and gain is not monotone
  in surrogate skill anywhere in the table.

### 4.3 What this does to the few-label premise

The project rests on a premise that matters for the molecular arm as much as the decision arm:
that useful work is possible *without* a good predictor, because at emergency timescales there
are no labels with which to fit one. That premise **partly survives, in a more useful form.**

What survives, now better supported: gains do not require a *good* predictor. The largest gain
in the project comes from a surrogate at Spearman 0.31, and gain is not even monotone in
surrogate skill — blocked H3 reaches 0.63 with a smaller gain. A modestly informative model is
enough.

What does not survive: the stronger reading that gains come from the policy *rather than* the
model, and are therefore substrate-independent. They are a surrogate effect. Every landscape
where the surrogate has essentially no in-pool skill shows essentially no gain, and the
surrogate-free policy never gains anywhere. The original argument was measured on the four-site
GB1 library — the highest-coverage substrate in the project — so "substrate-independent" was
never tested where coverage is poor. It has now been tested four times and it fails there.

The honest version, and the one I would defend on Thursday: *policy gains require only a weakly
informative surrogate, but they do require one — and whether one can be fitted at all is a
property of the candidate pool's geometry that can be computed in advance.* That is weaker than
the original and considerably more useful, because it is checkable before any budget is spent.

### 4.4 Why I am presenting it this way

Three self-caught errors in two days is not a comfortable thing to put in front of you, and the
temptation to lead with the surviving mechanism instead was real. But the methods claim this
project makes is that auto-gradeable environments built from public measurement data let you
find your own artefacts fast, and this week is the first evidence that it is true of the method
rather than just asserted about it. Each correction is now a control in the code — a delay-effect
table carrying
the identity and the null band beside the effect, a corrected headroom ceiling, a permuted-outcome
falsification run, and a zero-retrieval control for the dossier arm — so the specific mistakes
cannot recur silently. That is the part worth carrying into the paper.

### 4.5 What has not been measured at all

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

### 5.3 Prototype and priority pathogens — and an uncomfortable finding

The comparison you suggested is in
[PROTOTYPE_AND_PRIORITY_PATHOGENS.md](PROTOTYPE_AND_PRIORITY_PATHOGENS.md), covering WHO, UKHSA,
NIAID, CEPI, HERA, the US operational and regulatory lists, other national frameworks and the
academic literature. Three things from it belong here rather than in a companion document.

**The two kinds of list answer different questions and are not two tiers of one ranking.** A
*priority* pathogen is selected by risk: how bad, how likely. A *prototype* pathogen is selected
by representativeness and tractability: if we build tools against this one, do they transfer to
its relatives, and can we work on it today? The mismatch runs both ways and is explicit in the
sources. WHO names vaccinia a Poxviridae prototype partly because it offers a BSL-2 route, and
*Vesiculovirus* a Rhabdoviridae prototype because it is a useful vaccine vector — tool-value
criteria applied to a family WHO rates low risk. Conversely variola is a priority pathogen and
explicitly not a prototype. A prototype list is not a ranking of danger, and citing it as one
inverts its meaning.

**NIAID excluded both of the families we chose, by name, for a reason that applies to us.** I
verified this against the primary source rather than taking it from the survey. The NIAID
workshop that operationalised the prototype approach covered ten families — Arenaviridae,
Hantaviridae, Nairoviridae, Peribunyaviridae, Phenuiviridae, Filoviridae, Flaviviridae,
Paramyxoviridae, Picornaviridae and Togaviridae — and states:

> "Although Orthomyxoviridae and Coronaviridae have a very high pandemic risk, they were not
> included in this prototype pathogen exercise since significant resources are already allocated
> to characterize these families, and there are licensed or advanced clinical stage development
> MCMs available for viruses within them."
>
> — [Deschamps, DeRocco, Bok & Patterson, *J Infect Dis* 2023;228(Suppl 6):S355–S358](https://doi.org/10.1093/infdis/jiac465)

NIAID's criteria are *inverse* to resourcing: the better-studied a family, the lower it ranks as
a prototype. Our selection criterion — public measurement data exist — is close to the exact
quantity that caused the exclusion. A reviewer aligned with NIAID will not ask whether the choice
is risky enough; they will ask why not Nairoviridae or Picornaviridae, where the field is
demonstrably unprepared and where ReVAMPP actually spends.

That is not fatal, and the answer I would give is that NIAID's exclusion concerns *product*
resourcing for countermeasure development, not benchmark availability for *method* validation —
a family with dense public quantitative data is the right place to validate a method whose
intended destination is the data-poor families. But that answer is only credible if the method is
then shown to transfer out of distribution, and if the data-poor destination is stated as the
goal rather than left implicit. It is worth ten minutes on Thursday.

**Two corrections to how the project states its targets.** WHO prioritises Coronaviridae at
**subgenus** level, so the Mpro conservation argument should be made across *Sarbecovirus* and
*Merbecovirus*, not against SARS-CoV-2 alone — which is a stronger claim and the one WHO's own
molecular rationale invites. And "influenza haemagglutinin stem" is not a WHO unit: WHO
prioritises influenza by HA subtype. The project needs to say which subtype, and whether the
conservation claim is within an HA group or across groups. Both are free improvements to the
framing.

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

### 6.1 Molecular prediction — the primary arm

Each study has a decision attached, so a negative result changes the plan rather than being
absorbed. E0–E2 are gated on data: E0 and E1 can begin immediately on the public ncAA macrocycle
set if it proves usable (§2.2), and otherwise wait on the campaign export along with the rest.

| Study | Question | Decision it forces | Blocked on |
| --- | --- | --- | --- |
| **E0** | What do the labels actually mean — replicates, censoring, independent scaffolds, selection bias in which candidates were measured? | If there are too few independent chemical or backbone groups, the claim narrows to within-parent interpolation before any model is fitted. | ncAA set, or export |
| **E1** | Does a defined Rosetta metric rank experimental outcomes *within* matched assay conditions, rather than only separating campaigns or parents? | If the association disappears under grouping, physics is not a useful prior here and the contribution moves elsewhere. A null is informative. | Export |
| **E2** | Does any signal survive new backbones and new chemical classes? | A strong random-split result with weak grouped transfer supports interpolation, not generalisation. Prioritise the shift with the clearest reproducible failure. | ncAA set, or export |
| **E3** | Do a handful of labels beat simple score rescaling? | If affine calibration explains the whole gain, use the simpler correction and say so. | ncAA set, or export |
| **E4** | Is there a measured failure mode that justifies a specific platform improvement — rotamer coverage, or stage tracking and filtering? | Picks *which* platform change to implement. A better Rosetta score is not independent confirmation of a scoring improvement. | Export |
| **E5** | Does better prediction actually improve experimental choices under a fixed budget? | This is where the decision arm reconnects to the molecular arm. | A credible predictor |
| **E7** | Does the improvement survive new measurements? | The prospective check: candidates, ranks, model version and analysis frozen before the round. | Booked round |

The coverage result in §4.2 is a partial, early answer to the generic half of E0–E2: it says when
sequence-to-binding is learnable from few labels. It is on the wrong molecule class, which is
what §2.2 is for.

### 6.2 The decision arm

Four system-level studies, each likewise with a decision attached:

| Study | Question | Decision it forces |
| --- | --- | --- |
| **S0** | Do deadlines, incomplete evidence and limited capacity materially change which actions are useful? | If constraints do not affect the task, the emergency-specific contribution is not real and the benchmark is revised before any claim is made. |
| **S1** | Does structured evidence tracking beat simpler workflows on support, uncertainty and dossier quality? | If the agent offers no gain over a scripted workflow, identify the missing capability before adding orchestration. |
| **S2** | Does the system revise when evidence warrants and hold steady when it does not? | A changed ranking is not automatically a better ranking; this separates evidence-following from instability. |
| **S3** | Does an adaptive policy use a constrained budget better than a fixed action order? | Pursue the policy contribution only if gains survive held-out episodes and a simple cost-aware baseline. |
| **S4** | Do the recommendations survive a real laboratory? | The prospective check; predictions frozen before results arrive. |

### 6.3 Commitments governing both arms

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

Ordered with the molecular deliverables first, which is both the corrected priority and an
uncomfortable way round to read: the top three are the primary arm and all three are blocked.

| Required output | Where it lands | Status |
| --- | --- | --- |
| **Non-canonical macrocyclic designs and focused library** | Block registry exists as a container; every chemical field null | **Blocked — export** |
| **SynX platform improvement** | No substrate; re-scope decision at Thursday's gate | **Blocked — export** |
| **One mRNA display selection round, analysed** | W7 execution; readout due 18 Nov; analysis by 20 Nov | **Blocked — slot** |
| Binding prediction on the right molecule class | Public ncAA macrocycle set being characterised now (§2.2) | In progress |
| Evidence-backed target dossiers | Seven-axis packets for three families exist; both chosen families finalised W5 | Partially delivered |
| Modular emergency co-scientist | Decision state and action space delivered 19 Oct; integrated replay W5; demo frozen W6 | Partially delivered |
| Two prototype-family case studies | W5–W6, scored outputs | Planned |
| Historical COVID-19 evaluation | W5, with hindsight audit | Planned |
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

### 8.1 The one that dominates: the campaign export

This is not one risk among several. It is the critical path of the project's primary arm, and
it has been overdue for a week.

Without it there is no chemistry: no chemical identifiers, no codon mappings, no
stereochemistry, no molecular weights, no Rosetta parameters, no rotamer libraries. The
building-block registry is a correctly-shaped container with every field null. In consequence
**MOL-01 through MOL-04 cannot start**, the mandatory platform-improvement deliverable has no
substrate, and the physics-based non-canonical track — Flex ddG, rotamer generation, the
RosettaCommons ncAA interface protocol — remains entirely untouched. Six weeks remain.

Three things would help, in descending order of value and ascending order of how much they ask
of SynX:

1. **A minimal subset, this week**, beats a complete export in three weeks: the supported
   building-block catalogue with chemical identifiers, stereochemistry and codon assignments,
   plus whatever Rosetta parameters and rotamer assets already exist. That alone unblocks
   representation and the platform-improvement scoping.
2. **One complete campaign** with parent/variant identities, score provenance, assay endpoints,
   replicates and censoring — including the failed and low-scoring candidates, which are the
   ones that make an honest evaluation possible.
3. **A named owner and a date**, so that if neither arrives the platform-improvement deliverable
   can be re-specified at Thursday's gate against something that does exist, rather than
   slipping silently into week 8.

The public ncAA macrocycle dataset (§2.2) reduces the damage but does not remove it: it lets
binding prediction start on the right molecule class, and it says nothing about *this platform's*
chemistry, which is what the platform-improvement deliverable requires.

### 8.2 The rest

| Risk | Current state | Consequence | Mitigation |
| --- | --- | --- | --- |
| **No booked display round** | Overdue since 16 Oct. No named SynX owner, no slot. | The one mandatory experimental deliverable does not happen. Backward planning from a usable 18 Nov readout is already tight. | Needs a name and a date this week. No internal workaround exists. |
| **The public ncAA dataset may not be usable** | Being acquired and characterised now; the project's description of it is from memory and unverified. | If it has no quantitative per-variant endpoint, or too few independent scaffolds, the molecular arm has no substrate at all until the export lands. | §10 asks for the fallback. Candidates: other mRNA-display selection datasets, cyclic-peptide permeability sets, or accepting that prediction work waits on SynX. |
| **Trainable model access unconfirmed** | No confirmed account, model list or spend ceiling. | Cannot commit to model identities or an experiment matrix. | Verify live availability; the local pilot model is not automatically a supported training model. |
| **The mechanism rests on few independent landscapes** | §4.2. Now four against four with no overlap and a pre-registered replication on two unrelated proteins, but the permutation test sits at its floor (p = 1/70). | The claim is supported rather than established; a reviewer will ask for the within-protein control. | *Largely addressed on 20 October.* What remains is the 2×2 that separates coverage from combinatorial structure within one protein — PDZ doubles against the staged PDZ singles, coverage matched. ~60 replays, about two minutes. |
| **Further headline claims may be artefacts** | Three fell in two days. | Credibility, and wasted weeks building on a number that is not real. | The controls are now in the code rather than in a document: a delay-effect table carrying the identity and the null band, a corrected headroom ceiling, a permuted-outcome falsification run, and a zero-retrieval control. Every new claim gets an intervention test before it is written down. |
| **Expert rubric reliability** | Not yet measured. | If inter-rater reliability is poor, the primary evaluation is unreliable. | This is precisely why the auto-gradeable environments matter: they give a domain where ground truth is exact rather than assessed. |
| **Scope** | Eight workstreams, ten weeks, one person plus collaborators. | Thin delivery across everything rather than one solid result. | One paper narrative rather than two; mandatory outputs protected; a negative result retained and the claim narrowed, rather than the deliverable dropped. |

---

## 9. Working arrangements — addressing the visibility problem

You are right that the output has been sitting locally, and that is a problem worth fixing
properly rather than by sending more attachments. Proposed:

1. **Repository access.** The work is a git repository with full history — code, configurations,
   frozen splits, result manifests and these documents — and it already has a private GitHub
   remote, so this is one invitation away. Tell me which account to add, and whether anyone at
   SynX should have it too. I have not added anyone unilaterally because the repository will
   eventually carry campaign-derived material, and I would rather agree who sees it before that
   arrives than afterwards.
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

1. **The campaign export — or an agreed substitute — is the one that matters** (§8.1). The
   primary molecular arm has not started because of it. A minimal subset this week beats a
   complete export in three; if neither is realistic, I need to re-specify the
   platform-improvement deliverable on Thursday rather than let it slide to week 8.
2. **A named SynX experimental owner and a booked slot.** Backward planning from a usable
   18 November readout is already tight.
3. **The display-round question.** Mpro confirmed, or an alternative? This gates library design
   and everything downstream of it.
4. **Scope.** Eight streams in six remaining weeks is aggressive. If something should be cut
   rather than delivered thinly, Thursday is the time to say so — and my own view is that the
   decision arm is the thing to trim if the molecular arm unblocks.

**Things I would value your judgement on**

5. **Is the molecular/decision balance now right?** I had been describing the decision arm as
   the contribution, which was an inversion, and §2.1 explains how it happened. The corrected
   position is that molecular prediction for macrocycles is the research object and the
   co-scientist is the allocation layer. Tell me if that is still not the balance you want.
6. **The fallback if the public ncAA dataset is unusable.** It is being characterised now and
   the project's description of it is from memory. If it has no quantitative per-variant endpoint
   or too few independent scaffolds, the molecular arm has no substrate until the export lands,
   and I would rather agree the fallback in advance than improvise it.
7. **Have I scoped §4.2 correctly?** My reading is that the result describes focused pools of
   hundreds to a few thousand candidates — deconvolution and focused maturation — and says
   nothing about a selection round over a 10¹¹-member library. That is a deliberately narrow
   reading of my own result and I would rather you tell me it is too narrow than find out at
   review that it was too broad.
6. **NIAID excluded both our families from its prototype list, by name** (§5.3), because they
   are already well resourced — which is close to the exact reason we chose them. I have an
   answer, which is that their exclusion concerns product resourcing and ours concerns benchmark
   availability for method validation. I would like to know whether you find that convincing or
   whether it sounds like special pleading, because it is the one place where our target choice
   is in direct tension with a published framework rather than merely unsupported by one.
7. **The dossier eviction problem.** Evidence accumulates monotonically within an episode and
   conflicts are retained rather than overwritten, so context eventually has to be dropped —
   and what to drop is itself a decision the policy should be making. I do not have a good
   answer and it is a genuinely interesting design question.
8. **The frameshift element**, §2.6. Ten minutes, probably declined, worth asking.

**What I will bring**

Whatever the public ncAA macrocycle dataset turns out to contain, which is the thing most likely
to change what we do next; the running system with a walkthrough of one episode end to end; the
replays behind §4, including the two interventions; and the arithmetic on all three withdrawn
claims.

If you want one thing to read beforehand, make it §8.1 — the export — rather than the results.
The results are in better shape than the critical path is.
