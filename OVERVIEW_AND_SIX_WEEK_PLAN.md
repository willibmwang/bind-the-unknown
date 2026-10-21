# Emergency AI co-scientist — overview and plan for next six weeks

---

## Summary

The system is an **AI co-scientist that decides what to pursue and what to measure next**
under outbreak conditions: incomplete knowledge of the target, evidence that conflicts and
arrives late, a hard deadline, a fixed budget of actions, and one shot at the laboratory.
The central object is not a prediction but an allocation.

Four things changed in the last week, and they set the plan.

**The acquisition machinery now works on the right molecule class.** On two macrocyclic
libraries the policies reach 0.64–0.71 recall against a random baseline near 0.27 at matched
budget, significant across pools. This is the first result in the project on macrocyclic
substrates, and on one of them the features are named non-canonical monomers rather than
amino-acid letters. It did not require SynX.

**The decision arm's main readout turned out to be unmeasurable.** Forbidding evidence
retrieval entirely, the co-scientist still named the reference-best target in 16 of 16
episodes having bought nothing, while the evidence-using arm bought 90 items across 15
episodes for the identical score. The difference is 0.0. Target-selection agreement is
saturated by what the model already knows, so it cannot detect evidence-driven decision
quality in either direction. **Building a readout that is not saturated is the single most
important task of week 5.**

**Three claims were withdrawn, with the arithmetic shown.** They are listed in §4.2. Two
were mine and one was a formula error that had been reported in every summary.

**The engineering is now in a state where results regenerate rather than accumulate.** The
full grid is 2,556 graded cells, reproducible in four minutes on one laptop, every artifact
stamped with its host and commit, 39 tests on the invariants that would invalidate a result.

Three external dependencies remain overdue and none has an internal workaround. §6 states
each and what it blocks.

---

## 1. What the system is

### 1.1 The decision it makes

Each episode, under an eight-round deadline and a fixed action budget with the remaining
budget visible in its state, the co-scientist chooses one of six actions:

| Action | What it does |
| --- | --- |
| `retrieve_evidence` | Buy sourced evidence for one named target on one axis |
| `request_structure` | Request a structural or conformational calculation |
| `rerank_targets` | Revisit the ranking against evidence observed so far |
| `invoke_design_tool` | Propose a focused library against the leading target |
| `recommend_experiment` | Recommend the single round, and state what it can substantiate |
| `stop` | Stop and report the dossier with its open uncertainties |

The action budget binds independently of the deadline, and whichever binds first ends the
episode. Until this week the budget could never bind — with eight rounds, eight actions and
one action per round, the exhaustion branch first held at round nine, which the loop never
reached. The budget the prompt described to the policy was fiction; it now binds and is
tested.

### 1.2 What it reasons over

A **seven-axis dossier per candidate target**. The axes are the Week 4 selection rubric, and
each is in the state because it gates a decision rather than because it is interesting:

| Axis | What it gates |
| --- | --- |
| `reagent_tractability` | **Hard gate.** An untractable reagent ends the campaign regardless of the biology. |
| `essentiality` | A binder to a dispensable protein cannot become a therapeutic. |
| `conservation` | Gates every family-wide and prototype-pathogen claim. |
| `accessibility` | Sets whether a permeability story is required. |
| `assay_availability` | Without one, a single round can only substantiate enrichment. |
| `public_labels` | Determines whether transfer tests are runnable now or blocked on SynX. |
| `precedent` | Supplies positive controls and de-risks interpretation of a failure. |

Each claim on an axis carries a source, an availability date, one of four strength levels
(`unknown`, `weak`, `moderate`, `strong`), a `verified` flag, and an optional pointer to the
claim it contradicts. Only `reagent_tractability` is a hard gate: fail it and the campaign
ends however strong the other six look.

The rubric is also the reason the saturation problem in §4.2 has a tractable fix. Which axis
is currently gating a given decision depends on the evidence state the co-scientist has been
handed, not on the target's identity — so it is a question pretraining cannot answer, which
is what §5.1 builds on.

### 1.3 Three rules enforced in code, not by convention

- **Date-limited retrieval.** The dossier refuses evidence dated after the decision under
  consideration, and records the refusal rather than discarding it. The refusal count is
  deliberately *not* shown to the policy: telling it that three claims exist which it may not
  see is information from after the decision clock arriving through a side channel.
- **Contradictions are retained and consequential.** Conflicting claims coexist rather than
  overwrite. As of this week a contested axis no longer counts as substantiated — previously
  the ranking read a contested axis as settled while the state recorded that it was not,
  which made retention decorative.
- **An uncited claim cannot satisfy a gate.** Claims asserted without a citation are carried
  with `verified=false` and excluded from substantiation.

Each rule is pinned by a test that would fail if the rule were removed.

---

## 2. How it is graded without a laboratory or a reviewer

Knowing whether a scientific decision was a good one normally costs either a bench
experiment or an expert's time. Both are slow and expensive, and expert judgement has
inter-rater problems that are themselves a project risk.

The alternative used here: take a dataset in which someone has *already measured* the
outcome for every candidate, hide all of it behind an evaluator, and let the co-scientist pay
to reveal a few at a time under a budget. Every choice it makes is then scored exactly
against the outcomes that were withheld. Grading is deterministic, needs no bench and no
reviewer, and can be repeated as often as the question requires.

A note on one word, because it is overloaded. **"Policy" below means the decision rule — the
thing choosing the next action**, in the machine-learning sense. It has nothing to do with
public-health policy or with WHO guidance; the project's one WHO reference is the
prototype-pathogen rationale recorded in the evidence packets, which is a justification for
family choice and is unrelated to this section.

Four substrate families run through one replay engine: the original pilot landscape, the
pathogen landscapes, the method-development landscapes carrying measured binding endpoints,
and the macrocyclic display libraries added this week.

**Two falsification controls were added this week, and both are load-bearing.**

- **Permuted outcomes.** Shuffle the outcome vector before the episode and grade against the
  shuffle. A policy that learns must collapse to the random baseline. Measured over twenty
  episode coordinates: 0.804 on the real signal, 0.265 permuted, 0.263 random — the control
  lands 0.002 from random, which is what it is for. A policy that did *not* collapse would be
  scoring on index order or tie-breaking rather than on the evidence it paid for.
- **Zero retrieval.** Switch off the co-scientist's ability to look anything up, hand it an
  empty dossier, and require a decision anyway. In plain terms it is a cheating detector: if
  the system still names the target the full evidence base supports, it did not reason its
  way there, because it bought nothing to reason from — it already knew the answer from
  pretraining. It fired, in 16 of 16 episodes. This is the audit the architecture document
  named as necessary and left undone, and it is the reason §5.1 leads the plan.

Controls write to their own files and are excluded from every aggregate, because a
falsification check is not a result.

---

## 3. Architecture

One replay engine plus a registry, with the decision layer kept entirely separate.

| Component | Role |
| --- | --- |
| `arms.py` | The four substrate families as frozen records. Their differences are data — paths, outcome column, caveats, config hashes — not code. No I/O, no numerical dependencies. |
| `binding.py` | The replay engine, and one `run_arm` driven by that registry. The four named entry points are one-line delegations. |
| `streams.py` | One independent random stream per `(seed, purpose, index)` coordinate. |
| `data.py` | Staging, funnelled through a single exit that digests artifacts after writing them. |
| `sweep.py` | Grid enumeration and parallel execution, one code path for every arm. |
| `dossier.py`, `targets.py`, `emergency.py` | The decision layer. No numerical dependencies in the state object; it was written before the policy that consumes it. |
| `common.py` | The only inference client. Provider failures retry; a policy failure never does. |
| `summarize.py`, `calibration.py` | Evaluation, kept off the policy path. |

The consolidation happened this week because the copies had already drifted: the sweep knew
three arms, the summariser five names, and only one module knew that one arm's outcome column
is named differently. Equivalence of the new single path against each of the four
predecessors was verified field-for-field across twelve cells before the old bodies were
deleted.

**Reproducibility.** Every run records host, platform, commit, Slurm identifiers, and whether
the working tree was modified — the last being the one to read first, since a run from a
modified tree is not reproducible from the commit it names. 2,557 artifacts currently carry a
clean commit.

---

## 4. What has been measured

### 4.1 What survives

**The acquisition machinery transfers to macrocyclic substrates.** At 128 labels against a
random baseline near 0.27 at the same budget:

| Library | Pools | Greedy | UCB | Diverse | Significance |
| --- | --- | --- | --- | --- | --- |
| Selection-enrichment library | 20 | 0.640 | 0.657 | 0.027 | p = 0.0001 |
| Activity library, non-canonical monomer features | 15 | 0.688 | 0.711 | 0.260 | p = 0.0012, 0.0008 |

This is the first evidence in the project that the machinery works on the molecule class the
project is about, and the second row is the first result on non-canonical features.

**A sharp negative in the same table.** Pure-distance acquisition is not merely unhelpful on
the enrichment library — it is harmful: 0.027 against random's 0.267, worse in 20 of 20
pools, p = 0.0001 in the wrong direction. It spends the whole budget away from the hits. On
the activity library it is indistinguishable from random instead.

**The eight-label row is recorded before any policy acts, and is not about the surrogate.**
This section previously read "the surrogate has no discriminative power at the smallest
budget", on the grounds that every policy returns identical recall at eight labels in 21 of
105 rows of the gain table. The identity is real; the attribution was wrong. `replay` calls
`snapshot()` before the acquisition loop opens, so the eight-label record is the shared random
initialisation — `round: 0`, while the first acquisition is `round: 1` — taken before any
`pick_batch` call has executed. Every policy agrees there by construction at every seed, which
is the property `test_all_policies_still_open_from_identical_observations` exists to pin, and
it says nothing whatever about the surrogate's power.

The genuine smallest decision budget is **sixteen** labels: one batch of eight chosen by a
policy, with the surrogate fitted on eight. Measured there, the largest paired gain over
random on any landscape is **+0.0225**; signal emerges at thirty-two and is substantial by
sixty-four. The clause that survives is the last one — those rows carry no policy information
— and it is now pinned by a test rather than asserted.

This is the fourth claim of the same shape to be withdrawn here: a quantity a feedback-blind
policy reproduces by arithmetic, reported as a measurement. The others are in §4.2.

### 4.2 Three claims withdrawn

**The 1.1–1.4× selected-versus-returned gap is an accounting identity, not a measurement.**
The returned set is always a subset of the selected set and recall divides both by the same
pool-wide hit count, so the ratio is monotone in set inclusion and never falls below 1 in any
of 368 recorded runs. The feedback-blind random policy, whose selections are identical at
every delay, reproduces the entire band: 1.297 against 1.312 for the informed policies. A
quantity a feedback-blind policy can produce is not a measurement of how feedback delay
degrades a decision.

**The matched effect of delayed feedback is null.** Holding budget and metric fixed: mean
paired difference −0.011 over 534 pool comparisons, delay helping in 13 cells and hurting in
18, Wilcoxon p = 0.31. The random arm's paired difference is exactly 0.0, which is what makes
it usable as the null band.

**The attainable-headroom formula was wrong, and the metric correction is not novel.** The
reported bound was the inverse labelled fraction unconditionally; the attainable bound is the
tighter of that and the inverse hit threshold. The two agree at the terminal checkpoint of
every configured landscape, which is why no published figure changes, and diverge at the
early checkpoints where raw multiples look most impressive — at eight labels in a 512 pool it
claimed 64× of headroom where 20× exists. Separately, the chance-and-ceiling normalisation we
had treated as a contribution is published prior art from active-learning benchmarking and
from the enrichment-metrology literature. What remains ours is applying it per checkpoint and
showing it reorders *published* results.

### 4.3 What has not been measured at all

- The co-scientist's policy has **never run on a local model on GPU**. Every agent episode to
  date is a hosted model. Validating the local path is the only route to the volumes a
  trained policy needs.
- **No macrocycle from the SynX platform** has been represented or measured. Every
  platform-supplied chemistry field remains null pending the campaign export.
- The **physics-based non-canonical track** is untouched.
- The structural arm is retired as a scientific component.

---

## 5. The six-week plan

### 5.1 The problem that sets the order

The decision arm's readout is saturated. Until there is a readout the zero-retrieval control
does not saturate, no number from the decision arm can be attributed to evidence, and the
project's primary research question has no instrument. Everything in week 5 serves that, and
nothing downstream is worth building first.

Three candidate readouts, in descending order of how much they ask:

1. **Which axis it chooses to resolve next**, scored against which axis actually gates the
   decision. The co-scientist cannot know this from pretraining, because it depends on the
   evidence state it has been handed, not on the target's identity.
2. **Decision quality under corrupted and conflicting evidence.** Pretrained knowledge is a
   liability rather than an advantage when the supplied evidence contradicts it; this turns
   the saturation into a measurable failure mode.
3. **Anonymised targets.** Replace names with opaque identifiers and re-run. Compares
   directly against the named condition and isolates how much of the performance is the name.

### 5.2 Week by week

**W5, 26–30 October — make the decision arm measurable.**

| ID | Task | Acceptance |
| --- | --- | --- |
| COSCI-20 | Next-axis readout: score the chosen axis against the gating axis | Zero-retrieval control scores at or near chance; evidence arm beats it with a stated interval |
| COSCI-21 | Anonymised-target condition | Named versus anonymised run on the same packets, difference reported with its interval |
| COSCI-22 | Corrupted-evidence condition: supplied evidence contradicts the pretrained prior | Appropriate-update and hold rates separate the policies; no metric reads 1.0 for all of them |
| COSCI-23 | Procedural episode generator: vary packet, budget, delay, initial evidence, corruption | ≥200 graded episodes from the existing packets, with the generation parameters recorded per episode |
| FCAST-10 | Re-score one published table under the corrected attainment metric | A published ranking either moves or demonstrably does not; result written up either way |

Dependency: none external. This week is fully unblocked.

**W6, 2–6 November — the local-model policy, and the trained arm.**

| ID | Task | Acceptance |
| --- | --- | --- |
| TRAIN-10 | Stand the pinned local model up on GPU and validate the policy path end to end | One non-smoke agent episode per arm, artifact-recorded, from the local model |
| TRAIN-11 | Cost-matched comparison: policy versus deterministic baselines at equal spend | Comparison reported in actions and tokens, not wall-clock |
| TRAIN-12 | Supervised fine-tune on filtered successful episodes as the mandatory baseline | Held-out episode score against the untrained policy, ≥5 seeds |
| COSCI-24 | Full decision-arm sweep with the generator from COSCI-23 | Paired statistics with the unit of analysis stated |

Dependency: **confirmed trainable-model access with an approved spending ceiling.** Modal is
authenticated and is the fallback lane if cluster GPU does not materialise.

**W7, 9–13 November — London week. Reduced compute capacity; lab-facing.**

| ID | Task | Acceptance |
| --- | --- | --- |
| LAB-* | SynX engagement per the existing register | Named experimental owner and a booked slot, or a written re-specification |
| MOL-01..04 | Platform chemistry representation, if the export has arrived | Building-block registry populated; otherwise the deliverable is re-specified against public data at the W7 gate |
| GATE-* | Mid-point gate: confirm or re-specify the platform-improvement deliverable | A decision recorded either way |

Treat compute-heavy work as unavailable this week and schedule accordingly.

**W8, 16–20 November — consolidate and stress.**

| ID | Task | Acceptance |
| --- | --- | --- |
| COSCI-25 | Historical reconstruction episode on evidence restricted to an early window | Decision trace and its leakage audit, with the no-retrieval floor reported beside it |
| MOL-10 | Transfer test: canonical-trained surrogate applied to the non-canonical library | Reported with the geometry bound; a null stated as a null |
| SAFE-* | Safety benchmark preparation per the register | Per the register's acceptance criteria |
| FCAST-11 | Power analysis and the unit-of-analysis statement for every headline comparison | Every reported comparison carries an interval and a stated independent unit |

**W9, 23–27 November — writing, with results frozen.**

| ID | Task | Acceptance |
| --- | --- | --- |
| PAPER-01 | Draft the contribution: environments, the two controls, the metric audit | Complete draft, every claim naming the file and run that produced it |
| PAPER-02 | Reproducibility appendix: regenerate every figure from a clean checkout | A clean-clone regeneration that matches, or the differences enumerated |
| PAPER-03 | Prior-art section, written against sources actually read | No citation in the draft that has not been fetched and checked |

Freeze results at the start of this week. Anything not measured by 21 November does not go in.

**W10, 30 November–4 December — handoff.**

| ID | Task | Acceptance |
| --- | --- | --- |
| PAPER-04 | Presentation and final write-up | Delivered |
| GATE-99 | Final reproducibility check and artifact handoff | Repository, artifacts and documents handed over with provenance intact |

### 5.3 What is deliberately not in the plan

- Any multi-agent decomposition. The episode is a tightly coupled sequential decision over
  one evidence state, which is the regime where the current evidence favours a single policy
  with good tools; decomposition would be a cost multiplier.
- Any new benchmark-breadth claim. Our pools are sub-samples of single measurement tables,
  not independent replicates, so breadth is not ours to claim.
- The physics-based non-canonical track, unless the export arrives by W7.

---

## 6. Dependencies, risks, and what I need decided

| # | Dependency | Blocks | Consequence if it does not arrive |
| --- | --- | --- | --- |
| 1 | **Campaign export** | MOL-01..04, platform-improvement deliverable | The deliverable is re-specified against public data at the W7 gate. A minimal subset this week beats a complete export in three weeks. |
| 2 | **Trainable-model access with a spending ceiling** | TRAIN-10..12 | The trained-policy arm does not happen; the contribution narrows to environments plus the metric audit. Modal is the fallback. |
| 3 | **Named SynX experimental owner and a booked slot** | LAB-*, the prospective check | No prospective validation; conclusions stay retrospective and must say so. |

**Three decisions requested at Thursday's gate.**

1. Which of the three candidate readouts in §5.1 to treat as primary. My recommendation is
   the next-axis readout, because it is the one the co-scientist cannot answer from the
   target's name.
2. Whether to re-specify the platform-improvement deliverable now against public data rather
   than hold W7 open for the export.
3. Whether the trained-policy arm is in scope. If access is not confirmed this week, I would
   rather cut it deliberately than discover in W8 that it was never going to run.

---

## 7. Capacity, stated plainly

Six weeks, one person, with W7 substantially consumed by travel. The plan above is roughly
five productive weeks of work. The W5 block is the part I would protect at the expense of
everything else: without a readout the control does not saturate, the decision arm produces
numbers that cannot be attributed to evidence, and that is the project's primary question.

The macrocyclic result and the two falsification controls are, as of this week, the
contribution most likely to survive review. They are also the parts that required nothing
from SynX, which is worth weighing when deciding how much of the remaining six weeks to
spend waiting.
