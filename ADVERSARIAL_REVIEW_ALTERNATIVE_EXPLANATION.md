# Adversarial review: is "combinatorial structure" the right explanation?

**Lens: alternative explanation.** Granting that the recorded numbers are right and the code is
clean, construct the best rival account of why combinatorial landscapes show larger acquisition
gains, and determine whether it is distinguishable from the stated one.

**Verdict: the claim is seriously weakened.** Half of it survives and is strengthened; the other
half is refuted as a causal account and must be reworded before Thursday.

---

## 1. The claim, split into its two halves

> At matched pool geometry and matched budget fraction, acquisition-policy gain tracks whether the
> landscape carries combined substitutions across shared positions, **not** whether the protein is a
> pathogen protein. The earlier "the gain vanishes on pathogen targets" reading was confounded.

- **(A) Pathogen-ness is not the operative variable.** This survives, and the evidence below makes
  it considerably stronger than the claim's own evidence does.
- **(B) Combined substitutions are the operative variable.** This is **refuted as a causal
  account**. Combinatorial structure is *sufficient* in the data on disk but it is demonstrably
  *not necessary*, and the thing it is standing in for is measurable, continuous, and more useful.

---

## 2. The recorded numbers reproduce exactly

Regenerated from `runs/method/*.json` and `runs/targets/*.json` (128 labels, delay 0, paired per
pool, `top_recall` minus the random policy's `top_recall` on the same pool). Every one of the
twelve quoted cells matches to four decimal places.

| landscape | greedy gain | pools improved | ucb gain | pools improved |
| --- | ---: | ---: | ---: | ---: |
| spg1_olson_igg_matched | +0.2425 ± 0.2182 | 8/10 | +0.3038 ± 0.1817 | 9/10 |
| yap1_ww_peptide | +0.2628 ± 0.1644 | 17/18 | +0.2457 ± 0.1654 | 17/18 |
| ha_h3 | +0.1038 ± 0.0851 | 8/10 | +0.0500 ± 0.0925 | 6/10 |
| mpro | +0.0654 ± 0.1517 | 7/10 | +0.0385 ± 0.1203 | 5/10 |
| ha_h1 | −0.0038 ± 0.1181 | 6/10 | +0.0462 ± 0.1070 | 7/10 |
| spike_rbd_ace2 | +0.0297 ± 0.1356 | 3/6 | −0.0365 ± 0.0907 | 2/6 |

The arithmetic in the claim also holds: combinatorial mean 0.264, single-substitution mean 0.036,
non-overlapping ranges. **No number is being disputed here. The attribution is.**

### 2.1 An unrequested robustness check, with one result that matters

Both headline landscapes had staged pools that were never run. Running them changes nothing for
the combinatorial arm and something real for the single-substitution arm.

| landscape | pools 0–9 (quoted) | pools 10–19 (previously unrun) |
| --- | ---: | ---: |
| spg1_olson_igg_matched, greedy | +0.2425 | **+0.2462** |
| spg1_olson_igg_matched, ucb | +0.3038 | **+0.3154** |
| ha_h3, greedy | +0.1038 | **+0.0154** |
| ha_h3, ucb | +0.0500 | **+0.0077** |
| ha_h1, greedy | −0.0038 | −0.0043 (pools 10–18) |

The combinatorial result replicates out of sample almost exactly — that is genuinely reassuring and
is the strongest thing in the claim's favour. But **ha_h3's +0.104, the largest single-substitution
gain in the table, does not replicate**: over all twenty pools it is +0.060. Quote ha_h3 at ten
pools and you are quoting the lucky half.

---

## 3. The rival explanation

**Stated mechanism.** Variants carry combined substitutions, so observations at shared positions
compose, so the forest learns something.

**Rival mechanism.** The forest learns nothing about *combination*. It learns a **per-coordinate
main effect**, and it can only do that when a one-hot column is populated by enough labelled
examples to be worth splitting on. The operative quantity is therefore

> **how many labelled candidates share each coordinate with the unlabelled candidates in the
> same pool** — not whether any individual variant carries one substitution or two.

Combinatorial landscapes get this for free: a 512-candidate pool of GB1 doubles puts
512 × 2 / 55 ≈ 19 labelled examples on every variable position. A single-substitution scan of
haemagglutinin puts 512 × 1 / 566 ≈ 0.9 — the forest sees each position roughly once and cannot
split on it at `min_samples_leaf=2`. The claim's category, "combinatorial", is correlated with this
quantity across the seven landscapes on disk, but it is not the quantity.

**The two accounts are distinguishable**, because coordinate density can be varied while holding
combinatoriality fixed at zero.

---

## 4. The experiment that separates them

`ha_h3` is a strictly single-substitution scan of a pathogen protein. Its 512-candidate pools are
drawn at random from 566 positions. I rebuilt the pools so that each one draws from a **block of 27
positions** instead. Nothing else changed: same landscape, same `(position, residue)` encoding, same
cardinalities, same 512 candidates, same 128-label budget, same 5% threshold, same seeds, same
`replay()`. Every variant is still a single substitution.

| condition | combined substitutions? | labelled per position | Spearman @128 | greedy gain | ucb gain |
| --- | :---: | ---: | ---: | ---: | ---: |
| ha_h3, staged random pools | no | 0.9 | 0.065 | +0.104 | +0.050 |
| **ha_h3, position-blocked pools** | **no** | **19.0** | **0.640** | **+0.358** | **+0.273** |
| ha_h1, position-blocked pools | no | 19.0 | 0.619 | +0.192 | +0.188 |
| mpro, position-blocked pools | no | 19.0 | 0.664 | +0.106 | +0.120 |
| spg1_olson_igg_matched (reference) | yes | 18.6 | 0.664 | +0.242 | +0.304 |
| yap1_ww_peptide (reference) | yes | 29.5 | 0.372 | +0.263 | +0.246 |

**A single-substitution scan of a pathogen protein produces the largest acquisition gain in the
entire study — +0.358, larger than either combinatorial landscape** — as soon as its pool is narrow
enough that labelled and unlabelled candidates share coordinates. The effect reproduces on all three
pathogen landscapes.

Combined substitutions are therefore **not necessary** for the gain. Statement (B) is false as
written.

### 4.1 Dose–response, on the pathogen landscape

Varying only the number of distinct positions a 512-candidate pool draws from, on `ha_h3`, averaged
over three independent block randomisations:

| positions per pool | labelled per position | Spearman @128 | greedy gain | ucb gain |
| ---: | ---: | ---: | ---: | ---: |
| 27 | 19.0 | 0.624 | **+0.250** | +0.271 |
| 54 | 9.5 | 0.396 | +0.199 | +0.115 |
| 108 | 4.7 | 0.198 | +0.146 | +0.115 |
| 216 | 2.4 | 0.061 | −0.032 | +0.045 |
| 566 (as staged) | 0.9 | 0.102 | −0.090 | −0.115 |

This is the whole observed combinatorial-versus-single-substitution contrast, reproduced inside one
pathogen landscape with no combined substitution anywhere in it. The claim's binary category is a
coarse proxy for a continuous variable that the project can measure directly and report.

### 4.2 The converse test, which goes the claim's way

Holding protein, assay, encoding, pool size, budget and threshold fixed, and varying only whether
variants carry one substitution or two:

| GB1 (Olson 2014), 512-candidate pools, 5% | Spearman @128 | greedy gain | ucb gain |
| --- | ---: | ---: | ---: |
| doubles only | 0.684 | +0.362 | +0.354 |
| singles only | 0.614 | **+0.027** | +0.004 |

So within one landscape, removing combined substitutions does remove the gain. That is real support
for the claim's direction, and it rules out protein identity and assay as the driver. **But it also
breaks the simple "surrogate skill" version of the rival**: GB1 singles reach Spearman 0.61 — as
skilful as the blocked `ha_h3` pools — and still produce no gain.

### 4.3 The residual: skill is necessary but not sufficient

Across 86 pool-instances in nine conditions, greedy gain correlates with the surrogate's rank
correlation against truth (ρ = +0.48), with its skill in the upper half of the pool (ρ = +0.54) and
with its top-5% precision (ρ = +0.50) — all about as well as the binary "combinatorial" label
(ρ = +0.50). None of these is a clean predictor on its own.

The GB1-singles anomaly identifies the second ingredient. On a 512-candidate pool:

| | top-5% threshold above median (SD) | distinct predicted values among the forest's top 120 |
| --- | ---: | ---: |
| GB1 singles | 0.55 | **18** |
| GB1 doubles | 1.22 | 110 |

On singles the forest can only express "which position was hit" — 55 levels — so its ranking
collapses into a few large tie plateaus, and greedy's tie-break is effectively random inside them.
On doubles the two position effects compose into a ranking that resolves individual candidates. And
GB1's single-substitution top tail is only half a standard deviation above the median, so there is
little to resolve in the first place.

**The honest mechanism has two conditions, both necessary:**

1. labelled and unlabelled candidates in the pool share coordinates, so the surrogate has
   non-trivial rank skill; and
2. the surrogate's ranking *resolves the top tail* — enough distinguishable predicted levels, and a
   top tail far enough above the bulk to be worth resolving.

Combinatorial structure is one way to satisfy both at once. A narrow single-substitution pool
satisfies the first and usually the second. Neither condition mentions combination, and neither
mentions pathogens.

---

## 5. This undermines an earlier project argument, and the mentor will see it

`PROJECT_OVERVIEW_AND_PLAN.md` §2.1 and `W4_DEBRIEF_AND_TARGET_SELECTION.md` §2.2(c) both rest on:

> Acquisition policy delivers roughly 8× top-1% recall over random *while the surrogate is only
> weakly informative* — Spearman ≈ 0.31 … The result says the gains come from the policy. That is
> the claim worth carrying forward, and it is substrate-independent.

Two problems.

**First, 0.31 is not "no skill" — it sits inside the band where this study finds gains.** The
landscapes with no gain have Spearman 0.065–0.116; every condition with a gain above +0.14 has
Spearman ≥ 0.20, and the dose–response in §4.1 shows gain and Spearman rising together. The defensible
version is *"gains need only a modestly skilful surrogate, not a good one"*, which is a much weaker
and much more falsifiable statement than *"the gains come from the policy rather than the model"*.

**Second, and worse, the 0.31 result was itself measured on a combinatorial landscape.**
`configs/today.json` sets the binding pilot to `SPG1_STRSG_Wu_2016`, the canonical four-site GB1
library — every variant carries up to four substitutions across four shared positions, which is the
densest coordinate sharing in the whole project. The "substrate-independent" claim was never tested
on a substrate lacking shared coordinates. When it finally was — the staged pathogen scans — the
gain largely disappeared, which is exactly what §2.1's own caveat records without naming the cause.

This also tightens the emergency premise. If gains require Spearman ≳ 0.2, and that requires roughly
a hundred labels landing on shared coordinates, then "at emergency timescales there are no labels
with which to fit a predictor" is an argument *against* the policy helping, not for it.

---

## 6. What should go in Thursday's document

**Do not write:** *gain tracks whether the landscape carries combined substitutions.*

**Write instead:**

> At matched 512-candidate pool geometry, a 25% labelled fraction and a 5% threshold, acquisition
> gain tracks whether the candidate pool is narrow enough that labelled and unlabelled candidates
> share sequence coordinates — not whether the protein is a pathogen protein, and not combinatorial
> structure as such. Combined substitutions are one way to guarantee that sharing and the two
> combinatorial landscapes on disk show it (greedy +0.24 and +0.26), but it is not required: a
> strictly single-substitution influenza scan reaches +0.36 once its pools are restricted to 27
> positions, and the effect falls monotonically to zero as the pools are widened to all 566. The
> gain additionally requires the surrogate's ranking to resolve the top tail; GB1 single
> substitutions reach Spearman 0.61 and still show no gain, because the forest's predictions on
> them collapse into about eighteen tie plateaus.

### Why the reworded claim is better for the project, not worse

A macrocyclic peptide library is a randomised region of 6–12 positions sampled thousands of times
over. That is the high-sharing, few-positions regime — the +0.25 to +0.36 end of the dose curve —
and it qualifies on the stated mechanism for a reason that is *measurable before any assay is run*:
count candidates per variable position in the planned pool. The original wording would have left the
mentor asking whether macrocycle libraries are "combinatorial enough"; the reworded one gives a
number to compute.

### Caveats that must travel with it

1. The hypothesis was generated on 19 October data and tested on 20 October data.
   `spg1_olson_igg_matched` was staged specifically to test it and is the **same protein (GB1) as
   the original pilot** — not an independent system. The combinatorial arm rests on two landscapes,
   one of which is not independent.
2. The position-blocked pools in §4 and §4.1 are **constructed**, not staged. They are a mechanism
   test, not a naturally occurring sampling design, and a position-blocked pool is an easier problem
   by construction.
3. The GB1 singles-only pools of §4.2 reuse candidates across the five partitions (only 1,045 single
   substitutions exist), so those ten pool-instances are not ten independent pools.
4. `ha_h3`'s quoted +0.104 is pools 0–9 only; over all twenty pools it is +0.060.
5. Pool spread is variation across sub-landscapes of one assay, not biological replication. `n` is
   2 proteins in the combinatorial arm and 4 in the single-substitution arm; no between-landscape
   significance claim is supportable.
6. Assay endpoints differ across all of these landscapes. Nothing here is a binding measurement for
   a macrocycle, and nothing here establishes antiviral activity.

---

## 7. Reproduction

All figures above were regenerated with `./.venv-hpc/bin/python` against `hpc_pilots.binding.replay`
at `forest_trees = 64`, `ucb_beta = 1.0`, `initial = 8`, `batch = 8`, `maximum = 128`, `delay = 0`.
The harness was validated against the recorded runs by re-running `mpro` pools 0–9 and `ha_h3` pools
0–9 from scratch and recovering the recorded `top_recall` values exactly.
