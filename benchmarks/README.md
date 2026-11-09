# Two benchmarks for the co-scientist

Task 1 asks whether the ranking is any good. Task 2 asks whether the tool selection is. They are
separate because they fail separately, and because only one of them needs the network.

```
python3 -m benchmarks.task2                     # offline, deterministic, ~1s
python3 -m benchmarks.task2 --states            # per-state accuracy + the discrimination report
python3 -m pytest benchmarks/tests -q           # offline
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
`random_legal`, `random_any`. The ladder is built so the gap between any two arms names one idea:
`similarity` → `similarity_legal` is the value of knowing what a tool needs; `similarity_legal` →
`oracle` is the value of knowing what a call unlocks.

### What it found

**The ordering rule is a bet on option value, and it reverses sign depending on the goal.**

| goal / seed | oracle | selector | similarity_legal | random_any | gap_ordering |
|---|---|---|---|---|---|
| `unlock_all` / onset | 1.000 | **1.000** | 0.292 | 0.148 | **+0.708** |
| `gate_all` / onset | 1.000 | **0.505** | 1.000 | 0.214 | 0.000 |
| `gate_all` / known_protein | 1.000 | **0.429** | 1.000 | 0.162 | 0.000 |
| `call_all` / onset | 1.000 | 1.000 | 1.000 | 0.892 | 0.000 |

On `unlock_all`, where the goal is to widen the action set, the selector's unblocking-first rule
beats a lexical retriever handed the same legality filter by **+0.708** in optimal-move rate. On
`gate_all`, where the goal is axis coverage, **the identical rule costs it about half** — more
unlocking calls exist to be distracted by, and each is a detour from the two calls that actually
gate the axes. `selector.rank`'s own docstring calls `score[1]` "the assumption most likely to be
wrong". It is not wrong; it is a trade, and this is its price.

**This replaces an earlier all-zero result, and the reason it changed is the finding underneath.**
Before the structural chain was registered, `gap_ordering` was 0.000 on *every* instance: every
tool took an identifier, identification produced every identifier, so after one call everything
was callable and no ordering remained to get right. The cause was in `selector.FACTS` — all seven
kinds were identity-shaped ("something is named"), and none of the 47 declared parameters across
the ten specs accepted a structure, a marker dict, a family assignment or a row set. Nothing
consumed another tool's *payload*, only the identifiers it handed over.

Registering `structures.picks` → `geometry.markers_for_chain` → `geometry.macrocycle_score` gives
the graph a lane three deep and adds two payload-shaped facts (`pdb_chain`, `surface_markers`).
All three wrap functions that already existed in `geometry.py`; none reaches an endpoint the repo
was not already reaching. Registry 10 → 13, facts 7 → 9.

**`call_all` does not discriminate and should not be quoted.** Once legality and no-repeat are
given, every legal ordering is optimal, so `random_legal` scores 1.000. It measures waste, and is
kept only for the legality signal (`random_any` scores 0.892 there).

**From a bare sequence, no goal is reachable at all.** Nothing in the registry turns an
amino-acid sequence into a reviewed protein accession — only `intake.rank_targets` provides one,
and it needs a genome. For a protein nothing has catalogued, which is the case this project is
named for, the accession-keyed lanes are permanently shut. That is a gap in the registry, no
ordering rule repairs it, and it was the first thing the benchmark found.

### A harness bug worth recording

The first version of this benchmark put a lexical retriever *first* and the repo's selector
*last*. The cause was entirely in the harness: the similarity arms kept their own no-repeat
bookkeeping, while `selector.first` was called statelessly and returns the same tool for ever once
the facts stop changing. Memory of what has been called is a property of the agent loop, not of a
ranking rule, so the harness now owns it and passes every arm the same `mask`.
`test_every_arm_honours_the_mask` pins it.

### Still to do

`GATEABLE` is still only `accessibility` and `precedent` — 5 of the 7 dossier axes are gated by no
tool. `selector.PROPOSED`'s docstring argues four of those five are reachable with no new
endpoint, because three built lanes already retrieve material bearing on them and only the
*dating* is missing: an axis claim needs a dated claim, and a publication date, a bioactivity year
and a database release are all per-row. `dossier.evidence` already exists and already demands a
source and an `available_from`. Building two row-set tools over `literature.search` and
`bioactivity.prior_binders` would take `GATEABLE` from 2 to 6 and would change `gate_all`
substantially. `reagent_tractability` stays ungated deliberately — it is a wet-lab fact, and
`selector.py:487-490` argues a tool claiming it from public records would be the worst overclaim
available here.

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

**Blinding is the control the external arms rest on.** Asked to rank SARS-CoV-2 proteins by name, a
frontier model answers from memory. So each external arm runs twice: blinded, where candidates are
bare sequences with the virus and the protein names stripped, and named. The named score is a
memorisation ceiling, not a competitor; the gap between the two is the part that says anything.

### Label discipline

The label is **the protein, never the drug**. A drug acting on a host factor says nothing about
whether a viral protein was worth ranking, so bulevirtide (NTCP), maraviroc (CCR5) and the
interferons are absent by design — which is also why `DrugVirus.info` is not the source, since a
large share of broad-spectrum antivirals act on the host.

Three tiers, and the distinction between the second and third is the one that matters:

- **Strong** — an approved drug binds this protein in this virus.
- **Weak but real** — a class effect (the homologous protein is drugged in a relative) or a
  vaccine/immunoglobulin antigen. Real evidence of the wrong kind. Graded by `status`, reported as
  its own tier, never pooled with the strong rows.
- **Mislabelled** — simply wrong, and **grading does not catch these**. They score at whatever
  tier they are assigned and keep scoring. They have to be dropped, with the drop recorded.

### Status: reconciliation with `output/run_benchmark.py`, agreed with its author

A second harness for this same task was written independently and in parallel:
`output/run_benchmark.py` over `data/benchmark_panel.json`, **46 genomes and 154 target
patterns** against this module's 12 pathogens. Its scale is what to keep. Agreed split with its
author: **they own the runner and the per-genome records; this module owns the labels and the
scoring; and the panel is read-only here** — it stays the raw curation with its provenance, and
this schema is the graded derivative, so a weak label stays visible as a tier instead of being
silently upgraded.

Six defects had to be settled first. All six are confirmed by the panel's author, who
independently recomputed the null values.

1. **recall@k has no null, and the null is high.** Under the hypergeometric null at n=20
   candidates and k=5 matching chains, random ranking already scores 25.0% @1, 60.1% @3,
   **80.6% @5**, 98.4% @10. The first completed genomes reported 100% @5. `intake.rank_enrichment`
   (`intake.py:1538`) — the null with a best-attainable-p at each size — sits unused in a module
   the harness already imports from, and its own docstring describes this exact error.
2. **Chain-level counting triples the numerator.** `decompose` splits Spike into S1/S2/S2′ and
   `hits()` breaks only *within* a row, so one protein scores three hits. Each coronavirus has 3
   distinct biological targets, not the 5 scored — inflating numerator and null together.
3. **Substring matching.** `pattern.lower() in name` at `output/run_benchmark.py:80`. HBV's bare
   `"polymerase"` matches both `DNA polymerase` and `RNA-directed RNA polymerase`.
4. **26 of 46 genomes sit in a pattern-identical group.** Five coronaviruses share one pattern
   list; five biologically unrelated viruses share `RdRp / Glycoprotein / Nucleoprotein`. Only ~20
   rows have a unique list, so an aggregate over 46 independent instances is inflated. Combine
   within family first, then across families.
5. **About two-thirds of the labels are weak, and none is graded.** By enumeration of
   `validated_by`: 16 class-effect, 11 vaccine or immunoglobulin antigen, 5 investigational-only,
   3 presumptive ribavirin with no established protein target. Only **12–13 of 46** name an
   approved drug binding that protein in that virus. (An earlier draft of this file said "roughly
   a quarter"; that was wrong by a factor of nearly three.)
6. **Failed holdouts vanished from the denominator and errors cached for ever** — *fixed by the
   panel's author in `42e1555`*, not by this module. Error rows now go to
   `<accession>.error.json` so a transient timeout no longer drops a genome permanently, and
   holdout failures carry `kind: "holdout_removed_nothing"` so the previously-unreachable `leaked`
   branch has something real to report.

Three rows are mislabels rather than weak labels and are dropped on re-key with the reason
recorded: HSV-1's `terminase` (letermovir is HCMV-only — pattern-matched in from the panel's own
HCMV row) and HBV's `surface antigen` and `Large envelope` (vaccine antigens in a row whose
`validated_by` names tenofovir and entecavir).
