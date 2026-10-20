# Confound review: does acquisition gain track combined substitutions?

An adversarial re-examination of the claim that acquisition-policy gain at matched pool
geometry tracks whether a landscape carries combined substitutions across shared positions,
rather than whether its protein is a pathogen protein. Every figure below was recomputed
from `runs/` and from the staged landscapes in `data/processed/`; nothing is quoted from the
existing write-ups. Dated 20 October 2026.

**Verdict: the claim is half right and the half that is right is not the half it argues.**
The pathogen reading is dead, and more decisively than the claim says. The combinatorial
reading is refuted: a single-substitution pathogen landscape reaches the full combinatorial
gain once one other variable is changed, and a combinatorial landscape loses all of it when
only that variable is changed and the biology is left untouched.

---

## 1. The headline numbers reproduce exactly

Recomputed from the 404 complete run records in `runs/method` and `runs/targets`, taking the
final checkpoint (128 labels), zero feedback delay, and pairing each policy against `random`
on the same pool index.

| Landscape | Class | Pools | Random recall | Greedy gain | UCB gain | Greedy improved |
|---|---|---:|---:|---:|---:|---:|
| spg1_olson_igg_matched | combinatorial | 10 | 0.260 | **+0.242 ± 0.218** | **+0.304 ± 0.182** | 8/10 |
| yap1_ww_peptide | combinatorial | 18 | 0.218 | **+0.263 ± 0.164** | **+0.246 ± 0.165** | 17/18 |
| ha_h3 | single substitution | 10 | 0.219 | +0.104 ± 0.085 | +0.050 ± 0.093 | 8/10 |
| mpro | single substitution | 10 | 0.242 | +0.065 ± 0.152 | +0.038 ± 0.120 | 7/10 |
| ha_h1 | single substitution | 10 | 0.208 | −0.004 ± 0.118 | +0.046 ± 0.107 | 6/10 |
| spike_rbd_ace2 | single substitution | 6 | 0.265 | +0.030 ± 0.136 | −0.037 ± 0.091 | 3/6 |

Every cell matches to the last digit, pools-improved counts included. Random recall is
0.21–0.27 throughout, which is what a 25 percent labelled fraction should give, so the
geometry is genuinely matched and this is not the pool-size artefact that broke the earlier
"8x better than random" headline. The arithmetic is sound; the interpretation is not.

---

## 2. The variable that separates the two groups just as well

The claim's two groups are also perfectly separated by a second variable: **how many members
of a 512-candidate pool carry a mutation at each variable position**. Call it coverage.

Coverage is forced by the staging, not chosen. A single-substitution scan of haemagglutinin
spreads 512 candidates over 566 variable positions, so a position is represented about 1.5
times and no two candidates usually share one. A doubles library over 55 GB1 positions puts
about 18.6 mutated candidates on every position, and a 34-position WW domain about 29.6.

| Landscape | Variable positions | Coverage | Surrogate Spearman | Greedy gain |
|---|---:|---:|---:|---:|
| yap1_ww_peptide | 34 | 29.6 | 0.341 | +0.263 |
| spg1_olson_igg_matched | 55 | 18.6 | 0.663 | +0.242 |
| spike_rbd_ace2 | 201 | 2.55 | 0.055 | +0.030 |
| mpro | 303 | 1.69 | 0.103 | +0.065 |
| ha_h3 | 566 | 0.90 | 0.075 | +0.104 |
| ha_h1 | 564 | 0.91 | 0.099 | −0.004 |

Surrogate Spearman is the rank correlation between the shared 64-tree forest, trained on 128
randomly chosen labels in the pool, and the hidden outcomes of the other 384.

Across the twelve matched-geometry cells (six landscapes, greedy and ucb):

- r(gain, combinatorial flag) = 0.949
- r(gain, log coverage) = 0.874
- r(gain, surrogate Spearman) = 0.883
- **r(log coverage, combinatorial flag) = 0.962**

The two candidate explanations are collinear at 0.96 in the observed design. No amount of
re-analysis of these six landscapes can separate them. They have to be separated by
intervention, so they were.

---

## 3. Intervention A — a single-substitution pathogen landscape reaches the full gain

Same protein, same assay, same encoding, same surrogate, same 512-candidate pools, same 128
labels, same 5 percent threshold. The only thing changed is which positions a pool is drawn
from: instead of 512 candidates spread over all 566 haemagglutinin positions, each pool is
512 candidates drawn from a disjoint block of 28, 54 or 160 positions.

| Pools | Coverage | Spearman | Random | Greedy gain | UCB gain |
|---|---:|---:|---:|---:|---:|
| ha_h3, project's own random pools (n=8) | 1.48 | 0.072 | 0.240 | +0.091 | +0.024 |
| ha_h3, 160-position blocks (n=4) | 3.95 | 0.181 | 0.260 | +0.096 | +0.067 |
| ha_h3, 54-position blocks (n=8) | 9.48 | 0.428 | 0.293 | +0.187 | +0.202 |
| ha_h3, 28-position blocks (n=8) | **18.29** | 0.637 | 0.250 | **+0.284** | **+0.308** |

Influenza A H3N2 haemagglutinin. A pathogen protein. A replication-fitness endpoint, not a
binding endpoint. Every variant a single canonical substitution, encoded as (position,
residue) exactly as before. Greedy gain +0.284 and ucb gain +0.308 sit inside the
combinatorial band of 0.242–0.304, above both combinatorial landscapes on ucb.

The same sweep on Mpro is messier and is reported as found: coverage raises the surrogate
Spearman from 0.108 to 0.678, and ucb gain from +0.014 to +0.159, but greedy gain goes to
−0.024. Learnability is necessary and not sufficient; greedy can still be trapped on a
plateau. That weakens the corrected statement too, and it should not be presented as a law.

---

## 4. Intervention B — a combinatorial landscape loses all of it

The converse. Same 10 spg1_olson_igg_matched pools, same 512 doubles in each, same hidden
scores, same combinatoriality, same surrogate. The only change is the encoding: instead of
the residue identity at each of 55 positions, each variant is encoded as (which pair of
positions is mutated, which pair of residues it carries). That re-encoding is lossless — it
is a bijection onto the variant — but it stops pool members from sharing feature levels,
exactly as the single-substitution encoding does.

| spg1_olson_igg_matched, 10 pools | Spearman | Random | Greedy gain | UCB gain |
|---|---:|---:|---:|---:|
| Native encoding (residue at each position) | 0.664 | 0.260 | +0.242 | +0.304 |
| Sharing destroyed (position-pair, residue-pair) | **0.028** | 0.260 | **−0.019** | **−0.015** |

The entire gain disappears without touching a single variant, score or pool. Whatever is
producing the gain is a property of how the pool's features overlap, not of the fact that
the variants carry two substitutions.

A third cell completes the picture. The Olson library contains 1,045 single substitutions
over the same 55 GB1 positions. Pools of 512 drawn from them (overlapping, stated) have
coverage 9.3 and give Spearman 0.499, greedy +0.101, ucb +0.096 — single substitutions, same
protein, same IgG binding selection, an intermediate gain at an intermediate coverage.

---

## 5. A landscape already on disk that the claim's table omits

`dlg4_pdz3_peptide` is a single-substitution scan of the PSD-95 PDZ3 domain against its
CRIPT peptide ligand: a small interaction domain, a peptide-binding endpoint, 83 variable
positions, staged and replayed on 19 October. It is absent from the claim's table for one
reason: `method.json` stages it at `pool_size: 256`, so it is not at the matched geometry.

Re-pooled at the claim's own 512-candidate, 128-label, 5 percent geometry, with 24 pools
drawn from its 1,576 rows (overlapping, because only two disjoint 512-pools fit):

| dlg4_pdz3_peptide at 512/128/5% | Coverage | Spearman | Random | Greedy gain | UCB gain |
|---|---:|---:|---:|---:|---:|
| 24 overlapping pools | 6.17 | 0.266 | 0.231 | **+0.154 ± 0.169** | +0.119 |
| 2 strictly disjoint pools | 6.17 | — | 0.288 | 0.000 | +0.135 |

The overlapping estimate is four times the single-substitution group mean of 0.036 and lands
where the coverage curve in section 3 predicts for coverage ≈ 6. The two disjoint pools give
0.000 and are far too few to settle it. Either way, the claim's sentence "ranges do not
overlap" depends on a landscape being excluded by a staging parameter rather than by the
data, and that landscape is the one most similar to the application: a small domain binding
a peptide.

---

## 6. Confounds that were tested and ruled out

**Parent landscape size — ruled out.** `yap1_ww_peptide` has 10,075 rows, inside the
single-substitution range of 3,802–10,754, and gives the largest greedy gain in the set
(+0.263). `spg1_olson_igg_matched` is 53 times larger at 536,962 rows and gives +0.242.
Parent size does not order the gains and does not separate the groups.

**Score distribution shape — ruled out.** Measured per pool, averaged per landscape.
Separability, defined as the distance from the 95th percentile to the median in interquartile
ranges: yap1 2.01, ha_h3 1.15, ha_h1 1.14, dlg4 0.73, spg1_matched 0.54, spike 0.30, mpro
0.14. The high-gain spg1_matched is *less* separable than both influenza scans. Bimodality
coefficient does not order them either: mpro 0.86 is the highest and ha_h3 0.48 the lowest.
Exact ties: spike 0.60 and spg1_matched 0.24 are the two tie-heaviest landscapes and sit at
opposite ends of the gain range. Hits per pool are 26.0 ± 0.1 everywhere except spike, where
threshold ties give 30.5. No distributional statistic reproduces the split.

**Mutational load — ruled out.** The obvious mechanism, that combinatorial pools carry a
trivially learnable "number of substitutions" covariate that single-substitution pools cannot
carry by construction, is false here. Olson GB1 is a singles-and-doubles library: pools average
1.999 ± 0.021 substitutions with Spearman(load, score) = +0.013. YAP1 pools average
1.964 ± 0.187 with −0.008. Load is effectively constant in both and predicts nothing. This
matters, because if load had been the mechanism the application argument would have collapsed:
a fully randomised macrocycle library has no wild type and no load axis.

**Assay class — ruled out.** `spike_rbd_ace2` is a genuine binding-affinity assay and gives
+0.030 greedy, −0.037 ucb. `dlg4` is a peptide-binding assay. The constructed ha_h3 dense
pools are replication fitness and give +0.284. Binding versus fitness versus enzymatic
function does not order the gains in either direction.

**"Small interaction domain" — not separable, and subsumed.** Both combinatorial landscapes
are small domains and both low-gain influenza scans are large proteins. But the number of
variable positions is the denominator of the coverage statistic, so "small domain" and "high
coverage" are literally the same number here. It is not a distinct confound; it is the same
one under another name.

**A clean corroborating detail.** The `diverse` policy, which uses no surrogate at all, shows
no gain anywhere: spg1_matched −0.011, yap1 +0.026, ha_h3 −0.031, mpro −0.027, ha_h1 +0.019,
spike +0.025. The whole effect is a surrogate effect. That is consistent with the learnability
mechanism and inconsistent with any story about the biology of combined substitutions.

---

## 7. Two problems with the evidential base, independent of the confound

**The unit of analysis is the landscape, and there are six of them.** The ± figures in the
headline are standard deviations across pools of one protein. Pools are disjoint slices of a
single landscape, not independent landscapes, so they say how stable the estimate is within a
protein and nothing about how the class generalises. At the landscape level the comparison is
two means against four. An exact permutation test over all fifteen ways of splitting six
landscape means into a 2 and a 4 gives **p = 0.067**, which is 1/15 — the smallest p-value the
design can produce. The two combinatorial landscapes are indeed the top two of six, which is
the best possible outcome, and it is still not significant at the conventional threshold.

**Only one cell is a genuine out-of-sample test, and it is not independent.** File
modification times confirm the honesty context: `ha_h1`, `ha_h3`, `mpro` ran 2026-10-19
14:13–14:19; `spike_rbd_ace2`, `dlg4`, `yap1`, `spg1_olson_igg` ran 15:28–16:01 the same day;
`spg1_olson_igg_matched` ran 2026-10-20 10:26, after the hypothesis. So the confirmatory arm
is a single landscape, and that landscape is GB1 — the same protein as the original pilot.
The combinatorial group is effectively one hypothesis-generating landscape plus one
same-protein follow-up.

---

## 8. What can be said on Thursday

The mechanism is not combined substitutions. It is whether the acquisition surrogate can
generalise inside the pool, which at 512 candidates requires that pool members share feature
levels with each other. Combined substitutions over a few positions produce that sharing;
so does a single-substitution scan restricted to a few positions; and a combinatorial
landscape re-encoded without sharing produces none of it.

This is better news for the project than the original claim, not worse. The negative half is
now settled by intervention rather than by a coincidence of the sample: a pathogen protein,
on a replication-fitness endpoint, reaches +0.284 greedy gain. "The gain vanishes on pathogen
targets" is not merely confounded, it is false.

It is also a weaker basis for the macrocycle argument than the claim implies. The relevant
question becomes a design requirement that can be checked in advance, rather than a property
inherited from these six landscapes: a randomised macrocycle region of a few positions over a
building-block alphabet gives every library member a value at every position, so coverage is
high by construction and the regime should be favourable — but that is a prediction about the
library the project will build, and it has not been measured. The honest version is that the
pilot now identifies a measurable precondition for the policy to help, and states that the
planned library is expected to satisfy it.

**Suggested wording.** At matched 512-candidate geometry and a 25 percent labelled fraction,
acquisition-policy gain tracks how densely the candidate pool covers the landscape's variable
positions, and not whether the protein is a pathogen protein. Combined substitutions are a
proxy for dense coverage in the landscapes staged so far, not the cause: restricting an
influenza haemagglutinin single-substitution scan to 28 positions reproduces the full gain
(+0.284 greedy, 8 pools), and re-encoding GB1 doubles so that pool members share no feature
levels removes it (+0.242 to −0.019, 10 pools). With six landscapes the landscape-level
comparison cannot reach significance (exact permutation p = 0.067), and only one of the six
was run after the hypothesis was formed.

---

## 9. Gaps in the enforced invariants

`hpc_pilots/tests/test_pilots.py` enforces pool disjointness against the manifest, the
headroom ceiling, the encoding width, the surrogate-config wiring and a permuted control. It
does not record or constrain in-pool feature coverage, and nothing prevents a cross-landscape
comparison from silently omitting an eligible landscape because its configured `pool_size`
differs. Two cheap additions would have caught both problems found here: write the mean
in-pool coverage into each manifest's `pool_geometry` block, and have `summarize_method`
refuse to emit a cross-landscape gain table without listing the landscapes it excluded and why.
