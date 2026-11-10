"""Task 1: from one genome accession, how high does the ranking put the drugged proteins?

The instance is a pathogen genome. The candidate set is whatever six-frame ORF calling and
homology identification propose for it -- so every arm ranks the *same* proteins, and the only
thing that varies is the ordering. The score is where the labelled targets land.

**Every arm sees the same candidates, and that is what makes the comparison fair.** An arm that
produced its own candidate list could score well by proposing three proteins it already knew
about, which measures recall of the literature and not ranking. Fixing the candidate set upstream
of every arm removes that move. It also means a target the pipeline never proposed is scored as a
miss for every arm alike, which is the honest treatment: it is a failure of the candidate stage,
reported separately in `unmatched`.

**Blinding is the control that makes the LLM arms interpretable.** Asked to rank proteins from
SARS-CoV-2 by name, a frontier model answers from memory -- the drugged targets of that virus are
in every training corpus -- and scores near-perfectly without reading the genome. So each LLM arm
runs twice: blinded, where candidates are bare amino-acid sequences with identifiers stripped, and
unblinded, where they carry names. The unblinded score is a memorisation ceiling rather than a
competitor, and the gap between the two is the only part that says anything about ranking.
"""

from __future__ import annotations

import json
import math
import os
import random as _random
import re
import time
from dataclasses import dataclass, field

from . import task1_labels

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
CACHE = os.path.join(RUNS, "benchmarks")

TOP_K = (1, 3, 5, 10)


# ---------------------------------------------------------------------------
# Matching a label to a candidate row
# ---------------------------------------------------------------------------

_PUNCT = re.compile(r"[^a-z0-9]+")


def normalise(text):
    """Lowercase, strip punctuation, collapse spaces. The whole normalisation, kept visible."""
    return _PUNCT.sub(" ", (text or "").lower()).strip()


def matches(label, row):
    """Whether this candidate row is the labelled target, by whole-token run.

    An alias matches when its tokens appear as a *contiguous run of whole tokens* in the
    candidate's name. Two shapes in the pipeline's own output force something between strict
    equality and substring containment:

    * the candidate carries an affix the label does not -- `3C-like proteinase nsp5`,
      `RNA-directed RNA polymerase nsp12`, `NCAP_SEOUS Nucleoprotein` -- so strict equality
      misses nearly every real chain;
    * a longer name merely starts the same -- `NS5A` and `NS5B` against `NS5`.

    Substring matching gets the second one wrong in the flattering direction: `"ns5" in "ns5a"`
    is true, so a bare NS5 label would claim HCV's NS5A and NS5B as two hits. Token-run matching
    rejects it, because `ns5` is not one of `ns5a`'s tokens. It also still refuses the trap that
    retired the substring version: `f protein` is inside the string `transmembrane protein gp41`
    but is not a token run in it.

    A label that matches nothing is reported as unmatched rather than matched approximately. The
    inflation runs one way -- a looser matcher finds every label somewhere -- so the bias of
    being too strict is the one worth having.
    """
    aliases = {normalise(label["name"])} | {normalise(a) for a in label.get("aliases", ())}
    aliases = {a for a in aliases if a}
    for field_name in ("name", "parent", "target", "chain"):
        tokens = normalise(row.get(field_name)).split()
        if not tokens:
            continue
        for alias in aliases:
            want = alias.split()
            if not want or len(want) > len(tokens):
                continue
            if any(tokens[i:i + len(want)] == want
                   for i in range(len(tokens) - len(want) + 1)):
                return True
    return False


def rank_of(label, ordering):
    """1-based position of the labelled target in this ordering, or None if absent."""
    for position, row in enumerate(ordering, 1):
        if matches(label, row):
            return position
    return None


# ---------------------------------------------------------------------------
# The candidate set, fetched once per pathogen and cached
# ---------------------------------------------------------------------------

def candidate_set(entry, top=200, refresh=False, record=True):
    """The pathogen's identified, ranked candidates -- the shared instance for every arm.

    Cached to `runs/benchmarks/`, keyed on accession and holdout, because the live call reaches
    NCBI and UniProt and a benchmark that re-fetches on every arm would be neither reproducible
    nor welcome at the endpoints. The pipeline's own ordering is kept alongside the rows, so the
    `pipeline` arm needs no second call.

    `top` is deliberately large. The ranker's default of 12 would truncate the candidate list
    below the number of labels for the big herpesvirus genomes, and a truncated denominator
    would quietly turn a hard instance into an easy one.
    """
    os.makedirs(CACHE, exist_ok=True)
    key = normalise(f"{entry['accession']} {entry.get('holdout') or 'none'}").replace(" ", "_")
    path = os.path.join(CACHE, f"candidates_{key[:120]}.json")
    if os.path.exists(path) and not refresh:
        with open(path) as handle:
            return json.load(handle)

    from hpc_pilots.intake import decompose, run_intake

    started = time.time()
    intake = run_intake(entry["accession"], exclude_organism=entry.get("holdout"),
                        top=top, record=False)
    ranked = decompose(intake["targets"])
    rows = []
    for position, row in enumerate(ranked, 1):
        if not row.get("identified"):
            continue
        rows.append({"rank": position, "name": row.get("name"), "parent": row.get("parent"),
                     "tier": row.get("tier"), "score": row.get("score"),
                     "accession": row.get("nearest_homolog"),
                     "sequence": row.get("sequence") or row.get("protein") or "",
                     "length_aa": row.get("length_aa") or len(row.get("sequence") or "")})
    payload = {
        "pathogen": entry["pathogen"], "accession": entry["accession"],
        "holdout": entry.get("holdout"),
        "orfs_found": intake.get("orfs_found"), "orfs_ranked": intake.get("orfs_ranked"),
        "reference_entries": intake.get("reference_entries"),
        "reference_entries_removed": intake.get("reference_entries_removed"),
        "identified": len(rows), "unidentified": intake.get("unidentified"),
        "seconds": round(time.time() - started, 1),
        "candidates": rows,
    }
    if record:
        with open(path, "w") as handle:
            json.dump(payload, handle, indent=2)
    return payload


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

class Arm:
    """A ranking policy. Given the candidate rows, return them ordered best-first."""

    name = "arm"
    blinded = None

    def order(self, candidates, entry):  # pragma: no cover - interface
        raise NotImplementedError


class Pipeline(Arm):
    """The system under test: the rubric's own ordering, as the ranker produced it."""

    name = "pipeline"

    def order(self, candidates, entry):
        return sorted(candidates, key=lambda r: r["rank"])


class RandomOrder(Arm):
    """Uniform shuffle. The null the rank-sum p-value is computed against, run explicitly.

    Run as an arm as well as being the analytic null, because the two should agree and a gap
    between them would mean the p-value's assumptions do not hold on these candidate sets.
    """

    name = "random"

    def __init__(self, seed=0):
        self.seed = seed

    def order(self, candidates, entry):
        rows = list(candidates)
        _random.Random(self.seed).shuffle(rows)
        return rows


class LengthOrder(Arm):
    """Longest protein first. The dumb structural prior, and a real risk to the headline.

    Viral polymerases and polyproteins are long and are heavily drugged, so sorting by length
    alone may place labelled targets high. If this arm is close to the pipeline, the rubric is
    buying little over a feature available with no lookups at all -- which is worth knowing
    before claiming the ranking works.
    """

    name = "length"

    def order(self, candidates, entry):
        return sorted(candidates, key=lambda r: (-(r.get("length_aa") or 0), r["rank"]))


class Scripted(Arm):
    """Replays an ordering recorded from an external system.

    How Biomni and the LLM arms enter the benchmark: their transcript is saved as a list of
    candidate names per pathogen and replayed here. Candidates the external system did not
    mention keep the pipeline's relative order and are appended after the ones it did, so an
    arm is never credited for a target it never ranked and never destroyed by omitting one.
    """

    name = "scripted"

    def __init__(self, orderings, name="scripted", blinded=None):
        self.orderings, self.name, self.blinded = dict(orderings), name, blinded

    def order(self, candidates, entry):
        wanted = [normalise(n) for n in self.orderings.get(entry["pathogen"], [])]
        position = {name: i for i, name in enumerate(wanted)}

        def key(row):
            for field_name in ("name", "parent"):
                hit = position.get(normalise(row.get(field_name)))
                if hit is not None:
                    return (0, hit, row["rank"])
            return (1, 0, row["rank"])

        return sorted(candidates, key=key)


# ---------------------------------------------------------------------------
# Prompts for the external arms
# ---------------------------------------------------------------------------

def prompt(entry, candidates, blinded=True, limit=60):
    """The task statement handed to an external system, blinded or not.

    Blinded, a candidate is an index and a sequence: no protein name, no accession, no organism,
    and the pathogen is not named. That is the only form in which a frontier model's answer is
    about the proteins rather than about what it remembers of this virus. Unblinded, the same
    candidates carry their names and the pathogen is stated.
    """
    rows = sorted(candidates, key=lambda r: r["rank"])[:limit]
    if blinded:
        head = ("Below are proteins encoded by one viral genome. The virus is not named and the "
                "proteins are not named. Rank every candidate by how promising it is as a "
                "target for an antiviral that binds it directly. Answer with the candidate "
                "indices, best first, one per line, and nothing else.")
        body = "\n".join(f"[{r['rank']}] {len(r.get('sequence') or '')} aa\n"
                         f"{(r.get('sequence') or '')[:400]}" for r in rows)
    else:
        head = (f"Below are proteins encoded by {entry['pathogen']} ({entry['accession']}). "
                "Rank every candidate by how promising it is as a target for an antiviral that "
                "binds it directly. Answer with the candidate names, best first, one per line, "
                "and nothing else.")
        body = "\n".join(f"[{r['rank']}] {r.get('name')}"
                         f"{' (' + r['parent'] + ')' if r.get('parent') else ''}"
                         f" - {r.get('length_aa')} aa" for r in rows)
    return f"{head}\n\n{body}\n"


def emit_prompts(include_draft=False, blinded=True, top=200, refresh=False):
    """Write one prompt per pathogen, for running an external system by hand or by script."""
    os.makedirs(CACHE, exist_ok=True)
    written = []
    for entry in task1_labels.pathogens(include_draft=include_draft):
        payload = candidate_set(entry, top=top, refresh=refresh)
        text = prompt(entry, payload["candidates"], blinded=blinded)
        tag = "blinded" if blinded else "named"
        path = os.path.join(CACHE, f"prompt_{tag}_{normalise(entry['pathogen']).replace(' ', '_')}.txt")
        with open(path, "w") as handle:
            handle.write(text)
        written.append(path)
    return written


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_pathogen(arm, entry, payload):
    """One arm on one pathogen: where each label landed, and what that is worth.

    The p-value is `intake.rank_enrichment`, reused rather than reimplemented so the benchmark
    and the existing experiment cannot disagree. It reports the best attainable p at this size,
    which is why a Nipah-shaped result can be read as underpowered instead of as a failure.
    """
    from hpc_pilots.intake import rank_enrichment

    candidates = payload["candidates"]
    ordering = arm.order(candidates, entry)
    total = len(ordering)
    ranks, unmatched, rows = [], [], []
    for label in entry["targets"]:
        position = rank_of(label, ordering)
        if position is None:
            unmatched.append(label["name"])
        else:
            ranks.append(position)
        rows.append({"target": label["name"], "drug": label.get("drug"),
                     "status": label.get("status"), "verified": bool(label.get("verified")),
                     "rank": position,
                     "percentile": round(100 * position / total, 1) if position else None})

    enrichment = rank_enrichment(ranks, total) if ranks and total else {}
    return {
        "arm": arm.name, "blinded": arm.blinded, "pathogen": entry["pathogen"],
        "family": entry.get("family"), "accession": entry["accession"],
        "candidates": total, "labels": len(entry["targets"]),
        "matched": len(ranks), "unmatched": unmatched,
        "ranks": ranks, "targets": rows,
        # Hit rate at k is capped by how many labels exist, so a pathogen with one label can
        # never exceed 1/1 at any k. Reported as a fraction of labels matched, which keeps it
        # comparable across pathogens with different label counts.
        "hit_at_k": {f"top{k}": (sum(1 for r in ranks if r <= k) / len(ranks))
                     if ranks else None for k in TOP_K},
        "mrr": (sum(1 / r for r in ranks) / len(ranks)) if ranks else None,
        "mean_percentile": enrichment.get("mean_percentile"),
        "p_vs_random": enrichment.get("p"),
        "best_attainable_p": enrichment.get("best_attainable_p"),
        "adequately_powered": enrichment.get("adequately_powered"),
        "verdict": enrichment.get("verdict"),
    }


def stouffer(pvalues):
    """Combine per-pathogen p-values into one, by Stouffer's z.

    Stouffer rather than Fisher because Fisher is dominated by the single smallest p and this
    set has one pathogen with far more candidates than the rest, so Fisher would report the
    aggregate as whatever SARS-CoV-2 did. Equal weights, stated rather than tuned: weighting by
    candidate count or by label count would both be defensible and choosing between them after
    seeing the result is the move this note exists to prevent.

    Valid only because the pathogens are independent instances -- different genomes, different
    reference entries removed. It would not be valid across two arms on the same pathogen.
    """
    usable = [p for p in pvalues if p is not None and 0 < p < 1]
    if not usable:
        return {"p": None, "k": 0, "method": "none usable"}
    z = sum(_inverse_normal(1 - p) for p in usable) / math.sqrt(len(usable))
    return {"p": 1 - _normal_cdf(z), "z": z, "k": len(usable),
            "method": "Stouffer, equal weights", "dropped": len(pvalues) - len(usable)}


def _normal_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _inverse_normal(p, lo=-12.0, hi=12.0):
    """Probit by bisection. Deliberately dependency-free and plenty accurate at this scale."""
    for _ in range(200):
        mid = (lo + hi) / 2
        if _normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def arms(scripted=None):
    """The comparison ladder. `scripted` carries any external transcripts to include."""
    built = [Pipeline(), LengthOrder(), RandomOrder(seed=0)]
    for name, orderings in (scripted or {}).items():
        blinded = "blind" in name.lower()
        built.append(Scripted(orderings, name=name, blinded=blinded))
    return built


def run(include_draft=False, top=200, refresh=False, scripted=None, random_draws=20):
    """Every arm on every pathogen, with the aggregate each arm earns.

    The random arm is averaged over `random_draws` shuffles for the same reason Task 2 averages
    its random arms: one shuffle is a number with no error bar.
    """
    entries = task1_labels.pathogens(include_draft=include_draft)
    payloads = {e["pathogen"]: candidate_set(e, top=top, refresh=refresh) for e in entries}

    rows, aggregates = [], []
    for arm in arms(scripted):
        if isinstance(arm, RandomOrder):
            per_pathogen = []
            for entry in entries:
                draws = [score_pathogen(RandomOrder(seed=i), entry, payloads[entry["pathogen"]])
                         for i in range(random_draws)]
                per_pathogen.append(_mean_pathogen(draws, random_draws))
            rows.extend(per_pathogen)
        else:
            per_pathogen = [score_pathogen(arm, entry, payloads[entry["pathogen"]])
                            for entry in entries]
            rows.extend(per_pathogen)
        aggregates.append(_aggregate(arm.name, arm.blinded, per_pathogen))

    return {
        "benchmark": "task1-drugged-target-retrieval",
        "labels": task1_labels.counts(),
        "include_draft": include_draft,
        "pathogens": [{"pathogen": p["pathogen"], "candidates": len(p["candidates"]),
                       "orfs_found": p.get("orfs_found"), "identified": p.get("identified")}
                      for p in payloads.values()],
        "aggregates": aggregates, "rows": rows,
    }


def _aggregate(name, blinded, per_pathogen):
    """One arm's headline: combined p, mean percentile, hit rates, and what went unmatched."""
    ranks = [r for row in per_pathogen for r in row["ranks"]]
    percentiles = [t["percentile"] for row in per_pathogen for t in row["targets"]
                   if t["percentile"] is not None]
    return {
        "arm": name, "blinded": blinded, "pathogens": len(per_pathogen),
        "labels": sum(row["labels"] for row in per_pathogen),
        "matched": sum(row["matched"] for row in per_pathogen),
        "unmatched": sorted({u for row in per_pathogen for u in row["unmatched"]}),
        "combined_p": stouffer([row["p_vs_random"] for row in per_pathogen]),
        "mean_percentile": round(sum(percentiles) / len(percentiles), 1) if percentiles else None,
        "mrr": round(sum(1 / r for r in ranks) / len(ranks), 4) if ranks else None,
        "hit_at_k": {f"top{k}": round(sum(1 for r in ranks if r <= k) / len(ranks), 3)
                     if ranks else None for k in TOP_K},
        "pathogens_beating_chance": sum(1 for row in per_pathogen
                                        if row.get("verdict") == "beats chance"),
        "pathogens_underpowered": sum(1 for row in per_pathogen
                                      if row.get("verdict") == "not established (underpowered)"),
    }


def _mean_pathogen(draws, n):
    """Average one random arm's draws on one pathogen."""
    def avg(key):
        vals = [d[key] for d in draws if d.get(key) is not None]
        return (sum(vals) / len(vals)) if vals else None

    first = dict(draws[0])
    first.update({"arm": "random", "draws": n,
                  "ranks": [r for d in draws for r in d["ranks"]],
                  "mrr": avg("mrr"), "mean_percentile": avg("mean_percentile"),
                  "p_vs_random": avg("p_vs_random"),
                  "hit_at_k": {f"top{k}": (sum(1 for d in draws
                                               for r in d["ranks"] if r <= k)
                                           / max(1, sum(len(d["ranks"]) for d in draws)))
                               for k in TOP_K}})
    return first


# ---------------------------------------------------------------------------
# Scoring the 46-genome panel from the records the runner already wrote
# ---------------------------------------------------------------------------

PANEL_RUNS = os.path.join(RUNS, "benchmark")


def panel_records(directory=PANEL_RUNS):
    """Every clean per-genome record the panel runner has written, by pathogen.

    Error rows live beside the results under `<accession>.error.json` and are skipped here by
    extension rather than by inspecting a field, so a genome that failed is absent from the
    denominator instead of counted as a ranking that found nothing. The two are different
    claims and only one of them is about the ranker.
    """
    out = {}
    if not os.path.isdir(directory):
        return out
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json") or name.endswith(".error.json"):
            continue
        with open(os.path.join(directory, name)) as handle:
            record = json.load(handle)
        if record.get("error"):
            continue
        out[record["pathogen"]] = record
    return out


def hypergeometric_at_k(candidates, labels, k):
    """Chance that a random ranking puts at least one label in the top k.

    The number without which a recall@k is unreadable. With 20 candidates and 5 matching
    chains, random ranking already scores 80.6% at k=5 -- so a reported 100% is worth about
    nineteen points, not a hundred. Computed per genome because the panel's candidate sets run
    from 0 to 22 and a single headline null would be wrong for nearly all of them.
    """
    if not candidates or not labels:
        return None
    k = min(k, candidates)
    miss = 1.0
    for i in range(k):
        remaining = candidates - i
        if remaining <= 0:
            break
        miss *= (remaining - labels) / remaining if remaining > labels else 0.0
    return round(100 * (1 - miss), 1)


def score_panel_row(entry, record):
    """One pathogen: where each label landed among the candidates the ranker proposed.

    **One label contributes one rank, its best.** `rank_of` returns the first match, so a label
    matching several mature chains of one protein -- Spike as S1, S2 and S2' -- is counted once
    at the best of them. That is the whole fix for chain-level counting: the inflation came from
    counting matching *chains* rather than matching *labels*, which turned three views of one
    protein into three hits and raised the null along with the numerator.
    """
    from hpc_pilots.intake import rank_enrichment

    ordering = sorted(record.get("chains") or [], key=lambda r: r["rank"])
    total = len(ordering)
    ranks, rows, unmatched = [], [], []
    for label in entry["targets"]:
        position = rank_of(label, ordering)
        if position is None:
            unmatched.append(label["name"])
        else:
            ranks.append(position)
        rows.append({"target": label["name"], "status": label["status"],
                     "why": label.get("why"), "rank": position,
                     "percentile": round(100 * position / total, 1) if position and total
                                   else None})

    enrichment = rank_enrichment(ranks, total) if ranks and total else {}
    return {
        "pathogen": entry["pathogen"], "family": entry["family"],
        "accession": entry["accession"],
        "candidates": total, "labels": len(entry["targets"]),
        "matched": len(ranks), "unmatched": unmatched, "ranks": sorted(ranks),
        "targets": rows,
        "best_rank": min(ranks) if ranks else None,
        "holdout_removed": record.get("reference_entries_removed"),
        "segmented": entry.get("segmented"),
        "mean_percentile": enrichment.get("mean_percentile"),
        "p_vs_random": enrichment.get("p"),
        "best_attainable_p": enrichment.get("best_attainable_p"),
        "adequately_powered": enrichment.get("adequately_powered"),
        "verdict": enrichment.get("verdict"),
        # Recall@k never alone. The chance value travels with it or the number is unreadable.
        "recall_at_k": {f"top{k}": (1.0 if ranks and min(ranks) <= k else 0.0) for k in TOP_K},
        "chance_at_k": {f"top{k}": hypergeometric_at_k(total, len(ranks), k) for k in TOP_K},
    }


def run_panel(tier="strong", directory=PANEL_RUNS):
    """Score the panel, combining within family before combining across families.

    **Why the family step is not optional.** 26 of the 47 genomes share a target pattern list
    with at least one sibling -- five coronaviruses carry one list between them, and five
    biologically unrelated viruses share `RdRp / Glycoprotein / Nucleoprotein`. Stouffer's
    method needs independent instances, and five coronaviruses ranked against each other's
    relatives are not five independent tests of anything. Combining within family first and
    then across families gives roughly the number of instances the panel actually contains.
    """
    from . import task1_labels

    records = panel_records(directory)
    rows, missing = [], []
    for entry in task1_labels.pathogens(tier=tier):
        record = records.get(entry["pathogen"])
        if record is None:
            missing.append(entry["pathogen"])
            continue
        rows.append(score_panel_row(entry, record))

    families = {}
    for row in rows:
        families.setdefault(row["family"], []).append(row)
    per_family = []
    for family, group in sorted(families.items()):
        combined = stouffer([r["p_vs_random"] for r in group])
        per_family.append({
            "family": family, "genomes": len(group),
            "pathogens": [r["pathogen"] for r in group],
            "labels": sum(r["labels"] for r in group),
            "matched": sum(r["matched"] for r in group),
            "combined_p": combined,
        })

    across = stouffer([f["combined_p"]["p"] for f in per_family])
    all_ranks = [r for row in rows for r in row["ranks"]]
    return {
        "benchmark": "task1-drugged-target-retrieval",
        "tier": tier,
        "labels": task1_labels.counts(),
        "genomes_scored": len(rows), "genomes_without_a_record": missing,
        "independent_instances": len(per_family),
        "labels_total": sum(r["labels"] for r in rows),
        "labels_matched": sum(r["matched"] for r in rows),
        "unmatched": [(r["pathogen"], u) for r in rows for u in r["unmatched"]],
        "mrr": round(sum(1 / r for r in all_ranks) / len(all_ranks), 4) if all_ranks else None,
        "recall_at_k": {f"top{k}": round(
            100 * sum(1 for r in rows if r["best_rank"] and r["best_rank"] <= k) / len(rows), 1)
            if rows else None for k in TOP_K},
        "chance_at_k": {f"top{k}": round(
            sum(r["chance_at_k"][f"top{k}"] for r in rows if r["chance_at_k"][f"top{k}"]
                is not None) / max(1, sum(1 for r in rows
                                          if r["chance_at_k"][f"top{k}"] is not None)), 1)
            for k in TOP_K},
        "combined_p_across_families": across,
        "per_family": per_family,
        "rows": rows,
        "unscoreable": task1_labels.UNSCOREABLE,
    }


def main(argv=None):
    """`python -m benchmarks.task1` -- reaches NCBI and UniProt on a cold cache."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--include-draft", action="store_true",
                        help="score the unverified label rows too (excluded by default)")
    parser.add_argument("--top", type=int, default=200, help="candidates to rank per genome")
    parser.add_argument("--refresh", action="store_true", help="re-fetch cached candidate sets")
    parser.add_argument("--labels", action="store_true", help="print the label inventory only")
    parser.add_argument("--emit-prompts", choices=("blinded", "named"),
                        help="write per-pathogen prompts for an external system")
    parser.add_argument("--scripted", metavar="JSON",
                        help="a {arm: {pathogen: [names...]}} file of external orderings")
    args = parser.parse_args(argv)

    if args.labels:
        print(json.dumps(task1_labels.counts(), indent=2))
        return 0
    if args.emit_prompts:
        written = emit_prompts(include_draft=args.include_draft,
                               blinded=args.emit_prompts == "blinded",
                               top=args.top, refresh=args.refresh)
        print(json.dumps({"written": written}, indent=2))
        return 0

    scripted = None
    if args.scripted:
        with open(args.scripted) as handle:
            scripted = json.load(handle)
    payload = run(include_draft=args.include_draft, top=args.top, refresh=args.refresh,
                  scripted=scripted)
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
