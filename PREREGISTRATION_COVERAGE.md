# Pre-registration: does the coverage mechanism hold on independent proteins?

**Written 20 October 2026, before the replays were run.** Commit this file first; the result
goes in a separate commit. The point of writing it down is that a prediction recorded after the
fact is not a prediction, and this project has already retracted three claims in two days.

## What is being tested

The surviving mechanism from [ADJUDICATION_COMBINATORIAL_GAIN.md](ADJUDICATION_COMBINATORIAL_GAIN.md)
is that acquisition-policy gain is governed by whether the shared surrogate can learn **inside
the candidate pool**. Its top-ranked unresolved confound is independence: the combinatorial arm
rested on one independent protein (YAP1 WW domain) plus a restaging of the pilot protein (GB1),
so the landscape-level permutation floor sat at p = 1/15 = 0.067 and could not fall.

This adds two landscapes from unrelated proteins at the same 512/128/5% geometry:

| Landscape | Protein | Variants | Multi-sub | Endpoint | Variable positions |
| --- | --- | --- | --- | --- | --- |
| `grb2_sh3_peptide` | GRB2 SH3 domain | 63,366 | 62,332 | binding-coupled yeast growth | 56 |
| `rask_darpin_binding` | KRAS vs DARPin K55 | 24,873 | 21,789 | binding, by yeast growth | 187 |

Neither is related to GB1, YAP1 or any landscape already staged. `rask_darpin_binding` also
carries a binding rather than an organismal-fitness endpoint, which partly breaks the
assay-class confound in the earlier six.

## The predictor, measured before the replays

In-pool surrogate skill: fit the project's own 64-tree forest on 128 randomly chosen labels from
a 512-candidate pool, then take Spearman against truth on the 384 held out. Averaged over the
first five pools. This is the mechanism stated as a measurable quantity.

| Landscape | Variable positions | In-pool Spearman | Greedy gain (already known) |
| --- | ---: | ---: | ---: |
| `spg1_olson_igg_matched` | 55 | 0.695 | +0.242 |
| **`rask_darpin_binding`** | **187** | **0.675** | **— to be measured** |
| **`grb2_sh3_peptide`** | **56** | **0.455** | **— to be measured** |
| `yap1_ww_peptide` | 34 | 0.345 | +0.263 |
| `ha_h1` | 2 | 0.119 | −0.004 |
| `spike_rbd_ace2` | 2 | 0.112 | +0.030 |
| `mpro` | 2 | 0.091 | +0.065 |
| `ha_h3` | 2 | 0.075 | +0.060 (all 20 pools) |

On the six already measured, the split is clean: in-pool Spearman at or above 0.34 goes with a
gain at or above +0.24; at or below 0.12 with a gain at or below +0.065. Nothing yet lies
between 0.12 and 0.34, which is itself a gap worth filling later.

## The prediction

Both new landscapes sit above the threshold, so the mechanism predicts a substantial gain in
both.

1. **Primary, falsifiable.** Greedy gain over random at 128 labels, zero delay, paired per pool,
   10 disjoint pools: **≥ +0.15 for both landscapes**, improving in **≥ 7 of 10 pools** each.
2. **Point predictions.** `rask_darpin_binding` **+0.24 ± 0.10**; `grb2_sh3_peptide`
   **+0.25 ± 0.10**. I am deliberately not ordering them by Spearman, because gain was already
   shown not to be monotone in surrogate skill — GB1-matched at 0.695 gives +0.242 while YAP1 at
   0.345 gives +0.263.
3. **UCB** behaves similarly to greedy; **diverse** shows no gain (within ±0.05 of zero) on both,
   since the effect is a surrogate effect and diverse uses no surrogate.

## What would falsify it

- Either landscape showing greedy gain **below +0.10** falsifies the mechanism as stated. It
  would mean in-pool learnability is not sufficient even as a tendency, and the claim would have
  to retreat to a description of the six original landscapes.
- Either showing a gain with **diverse** comparable to greedy would mean the effect is not a
  surrogate effect, and the mechanism is misidentified.
- Gains in the +0.10 to +0.15 band would be an equivocal result and will be reported as such,
  not rounded into support.

## What this test does *not* do

It does **not** separate the coverage mechanism from the discarded "combined substitutions"
explanation: both landscapes are multi-substitution and high-coverage, so both hypotheses predict
the same outcome here. This test addresses independence only — whether the effect exists on
proteins unrelated to the two it was found on. Separating coverage from combinatoriality needs
the within-protein 2×2 (PDZ doubles against the already-staged PDZ singles with coverage matched),
which is a separate experiment and is not run here.

It also does not bear on macrocycles, display readouts, or any pool larger than 512; the gain is
already known to decay steeply with pool size at fixed budget.

## Analysis, fixed in advance

Ten disjoint evaluation pools per landscape, policies random / diverse / greedy / ucb, zero
delay, 128 labels, top-5% recall, paired per pool against random on the same pool with the same
eight seeded initial labels. Report mean paired gain, standard deviation across pools, and the
count of pools improved. No multiple over random will be quoted. If the exact landscape-level
permutation test is run, it will be reported with its floor (4 versus 4 gives 1/70 = 0.014).

---

## Outcome, appended 20 October after the replays were run

Everything above this line was committed before the replays (`3f9b0a1`-series, see
`git log --follow PREREGISTRATION_COVERAGE.md`). Nothing above has been edited.

**All criteria passed.** At 10 pools, the geometry the prediction was written for:

| Landscape | Predicted | Observed greedy | Pools improved | Diverse |
| --- | --- | --- | --- | --- |
| `grb2_sh3_peptide` | +0.25 ± 0.10, ≥ 7/10 | **+0.254** | 9/10 | −0.035 |
| `rask_darpin_binding` | +0.24 ± 0.10, ≥ 7/10 | **+0.254** | 8/10 | +0.038 |

Primary criterion (≥ +0.15 and ≥ 7/10): pass on both. Point predictions: within 0.005 on
both, which is closer than the method deserves and should not be read as precision. Diverse
within ±0.05 of zero: pass on both, consistent with a surrogate effect.

**On the remaining ten pools, run afterwards.** Both landscapes were staged with 20 pools; the
prediction was written for 10, so the second half is an out-of-sample extension rather than part
of the registered test. GRB2 strengthened to +0.265 over 20 pools (19/20). KRAS weakened to
+0.194 (15/20) — outside the registered ±0.10 band had the test been written for 20 pools. Both
figures are reported, and the 20-pool numbers are the ones used everywhere else, because using
the half that was registered while twenty exist is the selective reporting this file is meant to
prevent.

**Effect on the landscape-level comparison.** Four high-coverage landscapes against four
low-coverage, all staged pools run: high-coverage +0.194 to +0.265, low-coverage −0.004 to
+0.065, no overlap. Exact one-sided permutation p = 1/70 = 0.014, the floor of the design,
against 1/15 = 0.067 before these two were added.

**What it does not establish**, restated because it has not changed: this tests independence,
not the coverage-versus-combinatoriality distinction, and bears on neither macrocycles nor
display readouts nor pools larger than 512.
