# Two benchmarks for the co-scientist

Task 1 asks whether the ranking is any good. Task 2 asks whether the tool selection is. They are
separate because they fail separately, and because only one of them needs the network.

```
python3 -m benchmarks.task2                     # offline, deterministic, ~1s
python3 -m benchmarks.task2 --states            # per-state accuracy + the discrimination report
python3 -m pytest benchmarks/tests -q           # 53 tests, offline
python3 -m benchmarks.task1 --labels            # label inventory, no network
python3 -m benchmarks.task1                     # reaches NCBI and UniProt on a cold cache
```

---

## Task 2 — tool selection under preconditions

**The environment is the precondition graph; the grade comes from outside it.** Scoring a
selector against `selector.rank` would be worthless — the system would be graded against its own
preference and would win by construction. So `selector.REQUIREMENTS` is split by what its fields
are claims *about*. `needs`, `needs_any` and `provides` are claims about the endpoints (UniProt
resolves an accession, a folding endpoint takes a sequence) and they are the environment. The
ordering rule — unblocking before axis coverage before name — is the system's policy, and it is
what gets graded, against a shortest legal path computed by breadth-first search.

Six arms: `oracle` (shortest path), `selector` (the system), `similarity_legal` (lexical TF-IDF
over tool descriptions, filtered to legal calls), `similarity` (the same, unfiltered),
`random_legal`, `random_any`. The ladder is built so that the gap between any two arms names one
idea: `similarity` → `similarity_legal` is the value of knowing what a tool needs;
`similarity_legal` → `oracle` is the value of knowing what a call unlocks.

### What it found

**1. The precondition model carries the result. The ordering rules are not yet measurable.**

| goal / seed | oracle | selector | similarity_legal | random_any | gap_ordering | gap_legality |
|---|---|---|---|---|---|---|
| unlock_all / onset | 1.000 | 1.000 | 1.000 | 0.125 | **0.000** | 0.875 |
| gate_all / onset | 1.000 | 1.000 | 1.000 | 0.263 | **0.000** | 0.738 |
| gate_all / known_protein | 1.000 | 0.857 | 1.000 | 0.167 | **0.000** | 0.833 |
| call_all / onset | 1.000 | 1.000 | 1.000 | 0.977 | **0.000** | 0.023 |

`gap_ordering` is zero on every instance: a lexical retriever handed the legality filter matches
an optimal planner everywhere. On a ten-tool registry with a single bottleneck there is almost no
ordering left to get right — identification unlocks five tools and after that everything is
callable. This is pinned by a test rather than reported as a win, because a benchmark where every
informed arm scores 1.000 has measured nothing about ordering however good the table looks.

So the honest claim is narrow and still worth making: **knowing what a tool needs is worth 0.74
to 0.88 in optimal-move rate, and knowing what a call unlocks is worth nothing measurable yet.**
To test the second claim the registry needs a deeper dependency graph, not a better policy.

**2. The selector pays exactly one call for preferring option value.** From a known protein with
both axes open, it calls `families.entries` first because that unblocks the literature lane,
where the shortest path to gating both axes is the surface measurement and the structural lookup.
`selector.rank`'s own docstring names `score[1]` as the term most likely to be wrong; this is the
number that says what it costs. One call, on one of four instances.

**3. From a bare sequence, four of ten tools are unreachable and no goal can be met.** Nothing in
the registry turns an amino-acid sequence into a reviewed protein accession — only
`intake.rank_targets` provides one and it needs a genome. For a protein nothing has catalogued,
which is the case this project is named for, the accession-keyed lanes are permanently shut. That
is a gap in the registry, no ordering rule repairs it, and it is the first thing the benchmark
found.

**4. `call_all` does not discriminate.** Once legality and no-repeat are given, every legal
ordering is optimal, so `random_legal` scores 1.000. It measures waste, not judgement. Kept for
the legality signal (`random_any` scores 0.977 there) and should not be quoted as a ranking result.

### A harness bug worth recording

The first version of this benchmark put a lexical retriever *first* and the repo's selector
*last*. The cause was entirely in the harness: the similarity arms kept their own no-repeat
bookkeeping, while `selector.first` was called statelessly and returns the same tool for ever once
the facts stop changing. Memory of what has already been called is a property of the agent loop,
not of a ranking rule, so the harness now owns it and passes every arm the same `mask`.
`test_every_arm_honours_the_mask` pins it.

---

## Task 1 — drugged-target retrieval

From one genome accession, rank the proteins the genome could encode; score where the proteins an
antiviral actually binds land. The candidate set comes from six-frame ORF calling and homology
identification, and **every arm ranks the same candidates**, so the only thing that varies is the
ordering. An arm that proposed its own candidate list could score well by recalling three proteins
it already knew about, which measures literature recall and not ranking.

Arms: `pipeline` (the rubric), `length` (longest protein first — the dumb prior, and a real risk
to the headline, since viral polymerases are both long and heavily drugged), `random`, and
`scripted` for replaying an external system's ordering.

**Blinding is the control the LLM arms rest on.** Asked to rank SARS-CoV-2 proteins by name, a
frontier model answers from memory. So each external arm runs twice: blinded, where candidates are
bare sequences with the virus and the protein names stripped, and named. The named score is a
memorisation ceiling, not a competitor; the gap between the two is the part that says anything.

### Label discipline

The label is **the protein, never the drug**. A drug acting on a host factor says nothing about
whether a viral protein was worth ranking, so bulevirtide (NTCP), maraviroc (CCR5) and the
interferons are absent by design — which is also why `DrugVirus.info` is not the source, since a
large share of broad-spectrum antivirals act on the host.

Two tiers, and the split is load-bearing. `verified=True` is the eleven rows already in
`runs/target_ranks.csv`, carried over unchanged and pinned by a test. `verified=False` rows are
drafted from standard antiviral pharmacology and **have not been checked against a primary source
in this repository**; they are excluded from every default number and each one carries the lookup
that would promote it. A benchmark whose labels were written from recollection and reported as
curated would be a fabrication channel pointed at the result.

Status is graded (`approved`, `approved (EUA)`, `investigational`, `research target`) because an
approved drug and a chemical probe are different evidence that a protein is worth binding.

### Status: unreconciled with `output/run_benchmark.py`

A second, independently written harness for this same task landed in the tree during this work:
`output/run_benchmark.py` over `data/benchmark_panel.json`, with **46 genomes and 154 target
patterns** against this module's 12 pathogens and 11 verified labels. Its scale is the thing to
keep. Four issues have to be settled before its numbers are quotable, and this module was written
against all four:

1. **Substring matching.** `hits()` tests `pattern.lower() in name`, so `Protease` matches
   `Serine protease` and `Envelope` matches `Small envelope protein`. A loose matcher finds every
   label somewhere and inflates every rank in the direction the author wants. This module matches
   exact normalised names and aliases, and reports a label it cannot match as unmatched rather
   than matching it approximately (`test_matcher_is_exact_and_refuses_a_substring`).
2. **The 46 genomes are not 46 independent instances.** Five coronaviruses carry identical target
   patterns, as do two alphaviruses and several flaviviruses. Any aggregate that treats them as
   independent is inflated. `stouffer()` here requires independence and says so.
3. **Class-effect and vaccine-antigen labels are mixed with approved-drug labels.** Roughly a
   quarter of the panel is justified by "class effect" or by a vaccine antigen (`Gardasil targets
   L1`, `17D vaccine targets E`). A homologous protein being drugged in a relative, and an antigen
   the immune system sees, are both much weaker than an approved antiviral binding this protein —
   and a vaccine antigen is close to a category error for a macrocycle-binding pipeline. They need
   the graded `status` field, not exclusion.
4. **recall@k with no null.** Recall@5 is uninterpretable without knowing the candidate-set size:
   three labels among 23 candidates give a random recall@5 near 22% before the ranker does
   anything. `rank_enrichment` supplies the null and the best attainable p at each size.

The merge, not yet done: take the 46-genome panel as the label source, re-key it onto this
module's schema with `status` and `verified` per row, keep exact matching, and aggregate over
*families* rather than genomes.
