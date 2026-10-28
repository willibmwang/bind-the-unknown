# bind-the-unknown

Designing macrocyclic peptides that bind a viral target, and deciding what to measure when you
only get one experiment.

This repository holds two connected pieces of work. The first decides which target and which
molecules are worth the one experimental round the project gets. The second predicts which
molecules bind. Everything here runs on public data, and every result is graded against outcomes that were
already known before the code ran, so nothing needs a human reviewer or a laboratory to score.

**Read this first, then pick a document from [the reading order](#documents) below.**

---

## The two arms

Numbered top down, by where each sits in the stack: Arm 1 allocates the scarce experimental
capacity that Arm 2 consumes.

**The numbering is not a ranking.** Arm 2 is the primary research object. The project exists to
design macrocycles, and the allocation layer is in service of that. Earlier drafts of the project
documents inverted this, and the correction is recorded in
[PROJECT_OVERVIEW_AND_PLAN.md](PROJECT_OVERVIEW_AND_PLAN.md) §2.1.

### Arm 1 · The tool-calling co-scientist (the allocation layer)

**The question.** A new pathogen appears. Evidence is incomplete, it conflicts, and it arrives
late. There is a deadline, a fixed budget of actions, and roughly one shot at the laboratory.
Which target do you pursue, and what do you measure?

The object of interest is not a prediction. It is an allocation.

**How it works.** A policy acts through a registry of tools against a seven-axis target dossier,
choosing from six actions: retrieve evidence, request structure, rerank, invoke a design tool,
recommend an experiment, stop. Every call is charged against the budget, and evidence dated after
the decision clock is refused with the refusal recorded.

**Where it stands.** This is the arm that has actually been running, because it can run on public
data without waiting for anyone. It currently goes from a raw genome to a ranked list of
countermeasure targets, and the recent work has been about finding where that chain breaks. See
[what the co-scientist arm found](#what-the-co-scientist-arm-found).

### Arm 2 · Molecular prediction (the primary research object)

**The question.** Can we predict which macrocyclic peptides bind a chosen viral target, and use
those predictions to design a better library for SynX's mRNA display platform?

This is what the project is ultimately for. A better design process, measured against what the
platform does today, is the deliverable.

**Where it stands.** Blocked, and honestly so. The macrocycle chemistry is represented and tested,
but it is wired into no experiment. Every platform-supplied chemistry field is null: chemical
identifiers, SMILES, stereochemistry, reprogrammed codons, Rosetta parameters, rotamer libraries.
All of it waits on the SynX campaign export. **No macrocycle and no non-canonical residue has been
measured anywhere in this repository.** The physics-based non-canonical track is untouched.

That blockage is the critical path, not a line in a risk table. It is also why the arm that sits
above is the one with results: when the molecular work could not proceed, decision-policy work on
public data could.

**What exists instead, and why it is worth having.** The available work was on public
canonical-protein datasets. It produced a result which turns out to be a statement about
prediction rather than about policy: a condition on *when sequence-to-binding is learnable from a
handful of labels*. See [the surviving result](#the-surviving-result).

| | Arm 1 · Co-scientist | Arm 2 · Molecular prediction |
| --- | --- | --- |
| **Decides** | Which target, and what to measure | Which molecules bind |
| **Where it sits** | The allocation layer, above Arm 2 | The work being allocated for |
| **Primary research object?** | No | Yes |
| **Runs today?** | Yes, on public data | No, chemistry is represented but unwired |
| **Blocked on** | Nothing | The SynX campaign export |
| **Evaluated by** | Studies S0–S4, [overview](PROJECT_OVERVIEW_AND_PLAN.md) §6.2 | Studies E0–E7, §6.1 |

---

## How the grading works

The same pattern produces every number here. Take a dataset where the outcome is already known for
every candidate. Hide all of it behind an evaluator. Let a decision procedure request a few
outcomes at a time under a fixed budget, optionally with results delayed by a few rounds. Score
what it chose against the hidden truth.

Because the truth was known in advance, grading is exact and automatic. Almost all of the
engineering exists to guarantee one property: **a policy cannot reach an answer it has not earned.**
Isolation is enforced by construction rather than by care, and
[ARCHITECTURE.md](ARCHITECTURE.md) §4 is about how.

---

## What has actually been measured

### What the co-scientist arm found

Each of these is a recorded experiment in [runs/](runs/) carrying its own question and answer.

| Experiment | Finding |
| --- | --- |
| `intake_validation` | Genome to ranked targets recovers the real countermeasure targets across three virus families, with the pathogen's own proteins removed from the reference. Every expected target lands in the top eight. |
| `holdout_ladder` | Push the nearest curated relative further away and the pipeline does not degrade, it **stops**. It needs a relative close enough to be conspecific or congeneric. The rank where this happens differs by family, because viral species boundaries are not calibrated to sequence identity. |
| `identification_cascade` | Past that cliff, two different mechanisms recover two different things. Conserved enzymes come back through remote homology because they genuinely have distant relatives. Family-specific surface proteins have none, so no homology method finds them. Those are the antibody targets, and they are recovered from their own sequence instead, as a class. |
| `cross_kingdom_intake` | Genome to targets transfers from viruses to a bacterium, but only after replacing the candidate-selection step. Identification and ranking carry over unchanged; six-frame calling and length ranking do not. |
| `surface_markers` | A target can be scored from structure without penalising the very interfaces macrocycles exist to bind. Scoring the largest contiguous exposed hydrophobic patch ranks spike RBD above the main protease, the reverse of a small-molecule druggability score and the right way round for this project. |
| `surface_in_ranker` | Those markers do **not** improve the ranking, in either form tried. The change was withdrawn. They are computed and reported, not scored. |
| `amenability_gate` | The macrocycle-amenability gate becomes honest rather than confident. The patch moves by up to 496 square angstroms across structures of the same domain, comparable to the 600 threshold, so most real targets fall in a borderline band. Reporting that band is the correct result, not a failure. |
| `coverage_channel` | Equalising axis coverage across targets does not remove a policy's advantage from counting offers. It relocates it to dossier order at almost unchanged size. |

Identification is the single bottleneck in the chain.

### The surviving result

Acquisition gain is governed by **how densely the candidate pool covers the variable positions the
model has to learn from**. In plainer terms: a weak model fitted on 128 measurements predicts
held-out binding well enough to act on *only when the candidates share sequence coordinates with
the labelled ones*. Where they do not, correlation sits near 0.1 and nothing downstream helps.

This is a computable property of a library's design, available before any assay is run.

Correlation would not have established it, so it was tested by intervention in both directions
with everything else held fixed:

| Intervention | Greedy gain | Model correlation |
| --- | --- | --- |
| Influenza H3, pools narrowed to 28 variable positions | +0.104 → **+0.250** | 0.10 → 0.63 |
| Same, widened to 55 / 150 / 566 positions | +0.159 / +0.019 / 0.000 | 0.42 / 0.18 / 0.10 |
| GB1 doubles re-encoded to share no feature levels | +0.242 → **−0.019** | 0.66 → 0.03 |

The first takes the pathogen case that appeared to fail and recovers the whole effect by narrowing
the pool. The second keeps the same variants, the same hidden scores and the same combinatorial
structure, destroys only the feature sharing, and removes the effect entirely.

It then held on two unrelated proteins against a prediction **written down and committed first**,
in [PREREGISTRATION_COVERAGE.md](PREREGISTRATION_COVERAGE.md). Predicted at least +0.15 improving
in at least 7 of 10 pools; observed +0.254 and +0.254. Every criterion passed. Across all eight
staged landscapes the high-coverage and low-coverage groups do not overlap, and the permutation
test over landscape labels gives p = 0.014.

**Why this matters for library design.** The favourable regime is a randomised region of a few
positions over an alphabet, which is what an mRNA display library *is*. A weak predictor is enough,
but there must be one, and whether one can be fitted is knowable in advance.

**The limitation that matters most.** Every landscape behind this result is canonical residues on a
folded protein domain. None contains a macrocycle, a non-canonical residue or a cyclisation, so
none of it speaks to the molecule class the project exists to design.

### Claims that were withdrawn

This repository withdraws results in writing when they fail, and each withdrawal becomes a
permanent check in the code so it cannot quietly return. Three are set out in full in
[PROJECT_OVERVIEW_AND_PLAN.md](PROJECT_OVERVIEW_AND_PLAN.md) §4.1, including the project's
then-best result, which turned out to be an arithmetic identity that a feedback-blind random
policy reproduces exactly. Two further axis claims were withdrawn in the most recent work.

The adversarial and confound reviews in [reviews/](reviews/) exist for the same purpose.

---

## Repository map

| Path | What is in it |
| --- | --- |
| [hpc_pilots/](hpc_pilots/) | All the code. 40 modules, about 18,900 lines, plus the SLURM execution layer. |
| [hpc_pilots/configs/](hpc_pilots/configs/) | Landscape, target, method and tool-set definitions. Treated as byte-immutable once a run has embedded a config hash, so new work goes into a new file. |
| [runs/](runs/) | Recorded results, one file per experiment, each stating its own question and answer. |
| [reviews/](reviews/) | Adversarial and confound reviews written against this project's own claims. |
| [output/](output/) | Figures and the Markdown-to-PDF builder for the documents. |
| `data/` | Staged datasets. Gitignored; rebuild with the `prepare` commands. |

Scale of what is staged: GB1 at 149,360 rows, a 536,962-row GB1-Olson landscape staged to restore
headroom the small pathogen pools cannot provide, three pathogen landscapes, four
method-development landscapes with measured binding endpoints, and evidence packets covering 11
targets across three viral families.

---

## Running it

Model access is configured by environment variable (`PILOT_LLM_PROVIDER`, `PILOT_LLM_MODEL` and a
provider key) read from a gitignored `.env`. [.env.example](.env.example) documents the keys. No
secret is in the repository.

```bash
python -m hpc_pilots prepare                              # GB1 + ATLAS selection manifest
python -m hpc_pilots prepare-target --landscape mpro      # one pathogen landscape
python -m hpc_pilots target --landscape mpro --pool 0 --policy ucb --delay 3
python -m hpc_pilots targets                              # build the evidence packets
python -m hpc_pilots chemistry                            # build the block registry
python -m hpc_pilots emergency --family coronaviridae --policy agent
python -m hpc_pilots calibration
python -m hpc_pilots summarize
python -m hpc_pilots.submit --with-agent                  # the SLURM chain
```

Provenance is checkable rather than promised. Every download carries a URL, SHA-256, byte count
and verification time; every run embeds its dataset and config hashes and the tool set it ran
against; and `python -m hpc_pilots manifest --check` reports drift without writing.

**Tests.** 191 tests, 190 passing. One failure is known and open:
`test_a_vocabulary_encoding_cannot_tell_two_unseen_monomers_apart` reports a tied rate of 1.0
against a threshold of 0.01.

```bash
python -m pytest hpc_pilots/tests -q
```

---

## Documents

In reading order. Items 1 to 6 are also committed as PDFs.

1. **[PROJECT_OVERVIEW_AND_PLAN.md](PROJECT_OVERVIEW_AND_PLAN.md)** — the project in one sitting:
   the goal, why it has this shape, what has been measured, targets, and the evaluation design.
   Start here.
2. **[ARCHITECTURE.md](ARCHITECTURE.md)** — the system and the reasoning behind it, organised
   around *why* rather than as an API listing. §9 states what runs, what is partial and what is a
   stub. §10 lists defects found while writing it.
3. **[W4_DEBRIEF_AND_TARGET_SELECTION.md](W4_DEBRIEF_AND_TARGET_SELECTION.md)** — the October
   results and target analysis. Left unrevised as the record of what was believed on 19 October,
   and carries a correction notice at its head.
4. **[ADJUDICATION_COMBINATORIAL_GAIN.md](ADJUDICATION_COMBINATORIAL_GAIN.md)** and
   **[PREREGISTRATION_COVERAGE.md](PREREGISTRATION_COVERAGE.md)** — the intervention tests behind
   the surviving result, and the replication prediction committed before the replays ran.
5. **[PROTOTYPE_AND_PRIORITY_PATHOGENS.md](PROTOTYPE_AND_PRIORITY_PATHOGENS.md)** — background on
   the prototype-pathogen framing, including a bounded negative finding about family
   generalisation.
6. **[reviews/](reviews/)** — the adversarial and confound reviews.
7. **[hpc_pilots/README.md](hpc_pilots/README.md)** — the pilots and domain layer, module by
   module, each stating what it does not establish.
