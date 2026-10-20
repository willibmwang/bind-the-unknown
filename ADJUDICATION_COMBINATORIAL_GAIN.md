# Adjudication: does acquisition gain track combined substitutions?

*Adjudicator's review of five adversarial reports on the claim that, at matched pool
geometry, acquisition-policy gain tracks whether a landscape carries combined substitutions
across shared positions rather than whether its protein is a pathogen protein. Every number
below that is attributed to this review was recomputed or re-run here, with the project's own
`hpc_pilots.binding.replay`, the shared 64-tree surrogate, `ucb_beta` 1.0, 512-candidate pools,
8 seeded initial labels, batches of 8, 128 labels, top-5% threshold, zero delay, paired per
pool against `random` on the same pool.*

**Date:** 20 October 2026 · **Status:** adjudication of five reviews · **Scope:** one claim

---

## 1. Verdict

**The claim does not stand as stated.** Its two halves fail in different ways and neither
failure is repairable by re-analysis. The *negative* half — that the gain does not track
pathogen status — is true, but the six staged landscapes cannot establish it, because in those
six "viral protein" and "single-substitution scan" are the *same* two-versus-four partition:
`spike_rbd_ace2` is SARS-CoV-2 spike RBD, so all four single-substitution landscapes are viral
proteins, and `method.json` says so itself. Any statistic that separates the combinatorial
group separates the viral group identically; the stated rebuttal is circular. The *positive*
half — that combined substitutions are the operative variable — is refuted by intervention in
both directions, which I re-ran independently rather than taking from the reviewers.
Restricting `ha_h3` (a pure single-substitution scan of a pathogen protein, on a
replication-fitness endpoint) to 28 variable positions per pool and changing nothing else
raises greedy gain from +0.104 to **+0.250 (8/8 pools)** and ucb to +0.250, with surrogate
Spearman rising 0.10 → 0.63 — inside the combinatorial band. Re-encoding the `spg1_olson_igg_matched`
doubles as (position-pair, residue-pair), a lossless bijection that keeps the variants, the
scores and the combinatorial structure but stops pool members sharing feature levels, collapses
the gain from **+0.2425 → −0.019** (ucb +0.3038 → −0.015) and Spearman 0.660 → 0.025. Combined
substitutions are neither necessary nor sufficient; they are a proxy, collinear in these six
landscapes with the quantity that actually moves the gain, which is how densely the 512
candidates cover the variable positions the surrogate must learn from. All twelve reported
cells reproduce exactly from `runs/` — I regenerated them and every mean, standard deviation
and pools-improved count matched to four decimals — so this is not an arithmetic error. It is a
mis-attribution, and it is the same failure mode as the retracted "8× better than random": a
quantity governed by pool geometry read as a property of the biology. A better-supported claim
stands in its place, and it happens to be *better* news for the macrocycle application, not
worse.

---

## 2. The wording to use

> At matched pool geometry (512 candidates, 128 labels, a top-5% threshold, paired per pool),
> acquisition-policy gain tracks how densely the candidate pool covers the landscape's variable
> positions — that is, how much the shared surrogate can learn inside the pool — and not whether
> the protein is a pathogen protein. Combined substitutions are one way of producing dense
> coverage, and across the six landscapes staged so far the two are collinear, so those six
> cannot separate them. Direct intervention can, and does, in both directions. Restricting an
> influenza H3 haemagglutinin single-substitution scan — a pathogen protein on a
> replication-fitness endpoint — to 28 variable positions per pool, changing nothing else,
> raises greedy gain from +0.104 to +0.250 (8 of 8 pools) and surrogate Spearman from 0.10 to
> 0.63, matching the two combinatorial landscapes (+0.242 and +0.263). Conversely, re-encoding
> the GB1 double-mutant pools so that pool members share no feature levels — same variants,
> same hidden scores, same combinatorial structure — removes the gain entirely (+0.242 to
> −0.019, Spearman 0.66 to 0.03). The effect is a surrogate effect: the surrogate-free
> `diverse` policy shows no gain on any landscape (−0.03 to +0.03).
>
> Two qualifications travel with this. First, the landscape-level comparison cannot reach
> significance: two landscapes against four admits fifteen partitions, the observed split is
> rank 1 of 15, and the exact one-sided permutation p is 0.067, the floor of the design. The
> ± figures quoted per landscape are variation across pools of one protein, not evidence about
> the landscape class. Second, the "not pathogen status" conclusion is not supported by the six
> landscapes, in which "viral protein" and "single-substitution scan" are the same partition;
> it is supported by the intervention above, in which the pathogen scan itself delivers the
> full gain once its pools are narrow.
>
> For the macrocycle application the relevant on-disk analogue is not either of those two
> landscapes but the four-site GB1 library (Wu 2016): a randomised region over an alphabet,
> every member carrying a value at all four randomised positions, and members mutually distant
> (mean pairwise Hamming 0.95, against 0.07 for the GB1 doubles). At the same 512/128/5%
> geometry it gives the largest gain measured anywhere in this project — greedy +0.373 and ucb
> +0.373, 10 of 10 pools — from a surrogate with Spearman 0.31. What matters is therefore
> coverage of the randomised positions, not proximity to a wild-type reference, and a
> randomised macrocycle region of a few positions sits at the favourable extreme of that
> variable. This remains an argument about pool structure, not a measurement: no landscape here
> contains a macrocycle, a non-canonical residue or a cyclisation, and at a fixed 128-label
> budget the normalised gain on that same library falls from 0.58 at 512 candidates to 0.22 at
> 2,048, 0.09 at 8,192 and 0.035 at 32,768. The result therefore describes focused pools of
> hundreds to a few thousand candidates — deconvolution and focused maturation — and says
> nothing about a selection round over a 10^11-member library.

Do not write "combinatorial" as the explanatory variable anywhere, do not write "all three
pathogen landscapes" (there are four viral-protein landscapes), and do not quote any multiple
over random: across that pool-size sweep the multiple rises from 2.9× to 12.9× while the policy
is in fact losing almost all of its advantage, which is exactly how the earlier 8× figure
misled.

---

## 3. Caveats that must travel with it

- The hypothesis was generated by the 19 October runs and tested on data staged on 20 October
  for that purpose; `spg1_olson_igg_matched` is the only cell that postdates the hypothesis and
  it is GB1, the same protein as the original pilot, so the confirmatory arm is one
  non-independent landscape.
- The combinatorial arm is two landscapes and effectively one independent protein (YAP1 WW
  domain plus a restaging of the GB1 pilot); no between-landscape significance claim is
  supportable, and the exact permutation floor is p = 0.067.
- Both "combinatorial" landscapes are doubles only — 535,917 doubles and 1,045 singles for
  GB1-Olson, 9,713 doubles and 362 singles for YAP1 — i.e. perturbation scans two substitutions
  from a shared wild type, not randomised libraries.
- The headline `ha_h3` figure of +0.104 uses 10 of its 20 staged pools; the other ten give
  +0.015, so the honest all-pool figure is +0.060. (The combinatorial cell does not have this
  problem: `spg1_olson_igg_matched` pools 10–19 give greedy +0.246 and ucb +0.315, replicating
  the reported +0.242 / +0.304.) Report pool counts in the table.
- `dlg4_pdz3_peptide` is excluded from the comparison by a staging parameter, `pool_size` 256,
  not by the data; at its staged geometry 128 labels is a 50% labelled fraction and random
  recall is 0.538, so the exclusion is legitimate, but it must be stated rather than silent,
  and one reviewer's re-pooling of it at 512 gives greedy +0.154 over overlapping pools.
- Dense coverage is necessary but not sufficient: at high coverage `mpro` reaches surrogate
  Spearman 0.68 yet greedy gain −0.02 (ucb +0.16), and GB1 singles reach Spearman ~0.61 with no
  gain, because the surrogate's ranking does not resolve the top tail. State the mechanism as a
  tendency with two conditions, not a rule.
- The effect requires at least 64 of 512 labels; it is absent at 8, 16 and 32 labels and absent
  for the `diverse` policy at every budget.
- The separation is partly a property of the fixed 64-tree forest: a plain ridge on the same
  features beats it on every landscape and flips `ha_h1` from −0.004 to +0.073. The ordering
  survives the swap, but "the ranges do not overlap" is a statement about one surrogate.
- The position-blocked pools and the re-encoded GB1 pools are constructions made to break the
  collinearity. They establish which variable is operative; they are not representative samples
  of their landscapes and no landscape's gain should be quoted from them.
- `binding.py:enrichment()` is an exact algebraic rescaling of `top_recall` and models no count
  statistics, dropout or round dynamics; nothing in this repository has modelled a display
  readout.

---

## 4. Unresolved confounds, ranked by threat

1. **Independence of the combinatorial arm.** One independent protein, one restaging of the
   pilot protein, one post-hypothesis cell. *Settled by:* staging two further high-coverage
   combinatorial landscapes from unrelated proteins (see §6) at 512/128/5%, pre-registered
   before the replay is run. With 4 versus 4 the permutation floor falls from 1/15 to 1/70.
2. **Top-tail resolvability as a second necessary condition.** Learnability is necessary but
   not sufficient (`mpro` at Spearman 0.68 with greedy −0.02; GB1 singles at 0.61 with +0.03).
   Until this is pinned down, the mechanism is two conditions of unknown relative weight.
   *Settled by:* holding surrogate Spearman approximately fixed on the blocked `ha_h3` pools
   and varying only the top-tail separation, `(q95 − median)/sd`, by a rank-preserving transform
   of the outcomes.
3. **Whether any of this transfers to a selection round.** Per-candidate measurement is not an
   available action in mRNA display; the gain decays steeply with pool size at fixed budget;
   the enrichment metric is a rescaling of recall. *Settled by:* replacing the metric with a
   counts-based enrichment — multinomial resampling of the pool at a realistic sequencing depth
   with a dropout rate, over four to six rounds — and re-running the comparison. If the gain
   survives a pooled noisy readout, the transfer argument gains its first real support; if not,
   the result is confined to deconvolution, which is where I would scope it today.
4. **Scaling.** Normalised gain on the four-site library falls 0.58 → 0.22 → 0.09 → 0.035 as
   the pool grows 512 → 32,768 with the budget fixed at 128. *Settled by:* extending that sweep
   to 131,072 candidates (149,360 rows allow it) and fitting the exponent, and separately by
   sweeping the budget at fixed pool size to see whether gain is governed by labelled fraction
   alone.
5. **Surrogate dependence.** The ridge swap preserves the ordering but changes three of the six
   cells materially. *Settled by:* reporting the whole comparison under both surrogates as a
   standing second arm.
6. **Assay class.** Binding versus fitness is confounded with the split in the staged six, but
   is largely broken already: blocked `ha_h3` is a fitness endpoint with a large gain, and
   `spike_rbd_ace2` is a binding endpoint with none. *Settled by:* the §6 landscapes, which
   include both a binding and a growth endpoint.

---

## 5. Does this undermine the earlier "gains do not require a good predictor" argument?

**Partly — and it changes what that argument is allowed to say.**

What survives, and is now better supported than before: gains do not require a *good* predictor.
The four-site GB1 library delivers the largest gain in the project, greedy +0.373 over 10 of 10
pools, from a surrogate whose held-out Spearman is 0.31 — the same weak figure the earlier
argument rested on. Blocked `ha_h3` reaches Spearman 0.63 but a smaller gain, +0.250, so gain is
not even monotone in surrogate skill. A modestly informative surrogate is enough.

What does not survive: the stronger reading, that the gains come from the policy *rather than*
the model and are therefore substrate-independent. They are a surrogate effect. The
surrogate-free `diverse` policy shows no gain on any landscape in the set (−0.031 to +0.026,
recomputed here), and every landscape where the surrogate has essentially no in-pool skill
(Spearman 0.05–0.11) shows essentially no gain. Where the policy helps, it helps by exploiting
a weak model; where no weak model can be fitted inside the pool, no policy beats random. The
earlier argument was also measured on `SPG1_STRSG_Wu_2016` — the four-site library, the
highest-coverage substrate in the project — so "substrate-independent" was never tested on a
substrate lacking coverage. It has now been tested, four times, and it fails there.

The honest version for the document, which sits consistently beside the claim in §2: *policy
gains require only a weakly informative surrogate — Spearman ≈ 0.3 suffices — but they do
require one, and whether one can be fitted at all is a property of the candidate pool's
geometry that can be computed before any assay is run.* That is a weaker claim than the
original and a more useful one, because it is checkable in advance.

---

## 6. What to run next, ranked by value per unit of compute

1. **Two independent high-coverage combinatorial landscapes, pre-registered.** `GRB2_HUMAN_Faure_2021`
   (SH3 domain, 63,366 variants, 62,332 multi-substitution, binding-coupled growth readout) and
   `RASK_HUMAN_Weng_2022_binding-DARPin_K55` (24,873 variants, 21,789 multi, a binder-target
   binding endpoint) — both already reachable through `prepare_method`, both unrelated to GB1
   and YAP1, both large enough for 20 disjoint 512-candidate pools. Stage at 512/128/5%, run 10
   pools × 4 policies × delay 0, write the prediction down first. ~80 replays. This is the only
   thing that moves the landscape-level n, and it is cheap.
2. **The 2×2 that isolates coverage from combinatoriality within one protein.** `DLG4_HUMAN_Faure_2021`
   (PDZ doubles, 6,976 variants) against the already-staged `dlg4_pdz3_peptide` singles, same
   domain, same lab, both re-pooled at 512/128/5%, with the singles arm additionally restricted
   to the doubles' position set so coverage is matched. ~60 replays. Establishes how much
   combination adds *once coverage is held fixed* — the one number the current evidence cannot
   produce.
3. **Finish the pools already staged.** `ha_h3` pools 10–19, `ha_h1` pools 10–18 and
   `spg1_olson_igg_matched` pools 10–19 are staged and unrun; I ran the first and third while
   adjudicating (+0.015 and +0.246 greedy respectively). ~60 replays, and it removes a
   selective-reporting objection for free.
4. **Top-tail resolvability.** The rank-preserving tail transform on the blocked `ha_h3` pools
   described in §4.2. ~40 replays. Turns a two-condition hand-wave into a measured second axis.
5. **Counts-based enrichment readout.** Replace `enrichment()` with a multinomial-resampling
   version at a stated sequencing depth and dropout rate and re-run the six landscapes. A day of
   work, no new data, and it is the only item here that speaks to the display application
   rather than to the benchmark.
6. **Scaling extension.** Four-site GB1 at 65,536 and 131,072 candidates, budget fixed at 128,
   2 pools each. ~12 replays but slow ones. Confirms or breaks the decay exponent that bounds
   every transfer argument.

Do not spend compute re-running the six staged landscapes under new surrogates before item 1 is
done: it refines an estimate whose dominant uncertainty is the number of independent landscapes,
not the surrogate.

---

## 7. What I verified myself

| Check | Result |
| --- | --- |
| All twelve headline cells, regenerated from `runs/{method,targets}` at 128 labels, delay 0, paired per pool | Reproduce exactly, including every pools-improved count |
| Random recall across the six landscapes | 0.208–0.265 — the labelled fraction; geometry genuinely matched |
| `diverse` (surrogate-free) gain | −0.031 to +0.026 on all six; the effect is a surrogate effect |
| `ha_h3` restricted to 28 / 55 / 150 / 566 positions per pool | greedy +0.250 / +0.159 / +0.019 / 0.000; Spearman 0.63 / 0.42 / 0.18 / 0.10 |
| GB1-matched re-encoded to destroy feature sharing | +0.2425 → −0.019 greedy, +0.3038 → −0.015 ucb, Spearman 0.660 → 0.025 |
| GB1 four-site (Wu 2016) at 512/128/5%, 10 disjoint pools | greedy +0.373 (10/10), ucb +0.373 (10/10), Spearman 0.31, Hamming 0.948 |
| Unrun staged pools | `ha_h3` 10–19 greedy +0.015; `spg1_olson_igg_matched` 10–19 greedy +0.246, ucb +0.315 |
| Pool-size sweep, four-site GB1, budget fixed at 128 | normalised gain 0.575 / 0.225 / 0.093 / 0.035 at 512 / 2,048 / 8,192 / 32,768; multiple over random 2.9× → 12.9× |
| Mean pairwise Hamming within pools | GB1-matched 0.071, YAP1 0.110, four-site 0.948, single-sub scans 0.97 — dispersion does not separate gain from no-gain |
| `spike_rbd_ace2` is a viral protein | Yes, SARS-CoV-2 spike RBD, per `method.json`; four viral-protein landscapes, not three |
| Staged versus run pools | `ha_h3` 20/10, `ha_h1` 19/10, `spg1_olson_igg_matched` 20/10, `yap1` 18/18, `mpro` 10/10, `spike` 6/6 |
