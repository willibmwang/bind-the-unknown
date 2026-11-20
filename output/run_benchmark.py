"""Leave-one-organism-out ranking benchmark over the panel in `data/benchmark_panel.json`.

**What this measures.** For each virus the organism's own curated entries are deleted from
the reference before identification, so every protein has to be named through a relative.
Then: does the ranker put a protein an approved antiviral actually acts on near the top?
Recall@k over 47 genomes, rather than the three the project had before.

**Why it runs the ranking stages and not the chain.** The question is about the ordering, and
the ordering is settled by the end of stage 2. The surface stage costs about thirty seconds a
genome and the fold stage calls ESMFold, and neither changes a rank -- the surface term was
measured against the ranking and made it worse, which is why it is reported and not scored.
Running them here would multiply the wall clock without moving a single number this asks for.

**The hold-out is verified, not assumed.** `reference_entries_removed` is recorded per genome
and a zero is a failed hold-out, not a clean one: it means the organism name did not match
UniProt's spelling and the virus identified itself through its own entries. Those rows are
reported separately rather than averaged in, because a self-identified genome makes the
ranker look good for the wrong reason.

Resumable: one file per genome under `runs/benchmark/`, and a genome already written is
skipped, so a network failure halfway costs only what had not finished.

    python3 output/run_benchmark.py              # run every genome not yet done
    python3 output/run_benchmark.py --report     # summarise what is on disk, run nothing
"""
import argparse
import json
import pathlib
import re
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PANEL = ROOT / "data/benchmark_panel.json"
OUT = ROOT / "runs/benchmark"



def stem_of(accession):
    """A filename for one genome, segmented or not.

    A segmented genome arrives as comma-joined accessions, so the stem is taken from the
    first segment with a count appended. Keeping the commas would put them in a filename and
    in every glob that reads this directory."""
    parts = [a.strip() for a in accession.split(",") if a.strip()]
    stem = re.sub(r"[^A-Za-z0-9]+", "_", parts[0]).strip("_")
    return stem if len(parts) == 1 else f"{stem}_plus{len(parts) - 1}"


def rank_one(entry):
    """Stages 1 and 2 for one genome, returning the ranked chains and the hold-out check."""
    from hpc_pilots.intake import decompose, rank_score, run_intake, topology_overlap

    started = time.time()
    intake = run_intake(entry["accession"], exclude_organism=entry["holdout"],
                        top=12, record=False)
    ranked = decompose(intake["targets"])

    rows, position = [], 0
    for row in ranked:
        if not row.get("identified"):
            continue
        position += 1
        rows.append({"rank": position, "target": row.get("name"),
                     "nearest_homolog": row.get("nearest_homolog"),
                     "nearest_organism": row.get("nearest_organism"),
                     "tier": row.get("tier"),
                     "score": rank_score(row["features"]),
                     "features": row["features"]})
    return {
        "accession": entry["accession"], "pathogen": entry["pathogen"],
        "holdout": entry["holdout"],
        "genome_length_nt": intake["genome_length_nt"],
        "orfs_found": intake["orfs_found"], "orfs_ranked": intake["orfs_ranked"],
        # After decomposition, which is where it is large: a polyprotein is one row in the
        # intake payload and a dozen mature chains here, so the intake payload's own copy of
        # this number understates it on every single-polyprotein genome.
        "topology_rows_above_identified": topology_overlap(rows),
        "reference_entries": intake["reference_entries"],
        # Zero here means the hold-out matched nothing and the genome identified itself.
        "reference_entries_removed": intake["reference_entries_removed"],
        "holdout_applied": bool(intake["reference_entries_removed"]),
        "chains": rows, "seconds": round(time.time() - started, 2),
    }


def hits(result, patterns):
    """The rank of every chain whose name matches a ground-truth pattern."""
    found = []
    for row in result["chains"]:
        name = (row.get("target") or "").lower()
        for pattern in patterns:
            if pattern.lower() in name:
                found.append({"rank": row["rank"], "target": row["target"],
                              "matched": pattern, "score": row["score"]})
                break
    return found


def report(with_families=False):
    files = sorted(OUT.glob("*.json"))
    if not files:
        print("nothing in runs/benchmark yet")
        return
    panel = {e["accession"]: e for e in json.loads(PANEL.read_text())["panel"]}
    rows, leaked = [], []
    for path in files:
        res = json.loads(path.read_text())
        if res.get("error"):
            continue
        entry = panel.get(res["accession"])
        if not entry:
            continue
        found = hits(res, entry["targets"])
        best = min((h["rank"] for h in found), default=None)
        row = {"pathogen": res["pathogen"], "accession": res["accession"],
               "chains": len(res["chains"]), "matching": len(found),
               "removed": res["reference_entries_removed"], "best": best,
               "applied": res["holdout_applied"], "seconds": res["seconds"],
               "top": res["chains"][0]["target"] if res["chains"] else None}
        (rows if res["holdout_applied"] else leaked).append(row)

    def at(k, group):
        hit = sum(1 for r in group if r["best"] is not None and r["best"] <= k)
        return hit, len(group), (100.0 * hit / len(group) if group else 0.0)

    print(f"\n{'pathogen':<42}{'chains':>7}{'removed':>9}{'best rank':>11}  top-ranked")
    print("-" * 110)
    for r in sorted(rows, key=lambda r: (r["best"] is None, r["best"] or 0)):
        print(f"{r['pathogen']:<42}{r['chains']:>7}{r['removed']:>9}"
              f"{str(r['best'] or '-'):>11}  {str(r['top'])[:38]}")
    print()
    # PROVISIONAL, and never to be quoted on its own. Recall@k against no null is the exact
    # claim `intake.rank_enrichment` was written to replace: with k matching chains among n
    # candidates a random ranking already scores most of it, so a number here that looks like
    # a pass can be chance. The baseline below is the hypergeometric probability that a random
    # ranking puts at least one matching chain in the top k, averaged over the genomes as they
    # actually came out. The authoritative scorer is the graded-label one; this is a guard
    # rail so nobody reads the left column alone.
    from math import comb

    def null_at(k, group):
        vals = []
        for r in group:
            n, m = r["chains"], r["matching"]
            if not n or not m:
                continue
            miss = comb(n - m, k) / comb(n, k) if n - m >= k else 0.0
            vals.append(1.0 - miss)
        return 100.0 * sum(vals) / len(vals) if vals else 0.0

    print(f"  {'':9}{'observed':>10}{'random':>10}{'lift':>9}   (observed alone means nothing)")
    for k in (1, 3, 5, 10):
        hit, total, pct = at(k, rows)
        base = null_at(k, rows)
        print(f"  recall@{k:<3}{pct:>9.1f}%{base:>9.1f}%{pct - base:>+8.1f}  {hit}/{total}")
    if leaked:
        print(f"\n  {len(leaked)} genome(s) where the hold-out matched nothing, "
              f"reported separately and NOT counted above:")
        for r in leaked:
            print(f"    {r['pathogen']:<42} best rank {r['best']}")
    if with_families:
        # Only the rows whose hold-out applied, for the reason the split above exists: a
        # self-identified genome makes the ranker look good for the wrong reason, and
        # re-weighting by family would not make it look good for a better one.
        families_report({r["accession"]: r for r in rows}, record=with_families)


FAMILIES = ROOT / "data/benchmark_families.json"


def families_report(rows_by_accession, record=False):
    """Recall@k with the family as the unit of analysis, and an interval on the lift.

    **The defect this answers is already written down.** `benchmarks/README.md`: "26 of 46
    genomes sit in a pattern-identical group. Five coronaviruses share one pattern list; five
    biologically unrelated viruses share RdRp / Glycoprotein / Nucleoprotein. Only ~20 rows have
    a unique list, so an aggregate over 46 independent instances is inflated. Combine within
    family first, then across families." A per-genome mean over 47 genomes counts the five
    coronaviruses as five draws, when the hold-out makes each of them name its proteins through
    a relative that is usually another coronavirus and scores them against the same three target
    names. They are closer to one draw than to five.

    **Mean of family means, not a weighted mean.** Weighting each family by its genome count
    reproduces the per-genome figure exactly and answers nothing. So each of the 21 families
    carries one vote: Flaviviridae's eight genomes are averaged into one number before that
    number is averaged with Hepadnaviridae's single one. Eleven families hold exactly one
    genome, so this is a re-weighting rather than a variance reduction, and the re-weighting is
    the correction -- the panel's largest families are also its best-curated ones.

    **The interval is over families, bootstrapped, because that is the unit being claimed.**
    Resampling genomes would put the dependence back. 10,000 draws, seeded, so the number is
    reproducible rather than merely plausible; percentile interval, because a proportion near
    its ceiling has no business with a symmetric one.

    **Read the lift, not the recall.** `null_at` already explains why recall@k alone can look
    like a pass by chance: with m matching chains among n candidates a random ranking scores
    most of it. The lift is observed minus that hypergeometric null, family-combined the same
    way, and its interval is the one that says whether there is a result here at all.
    """
    import random
    from math import comb

    if not FAMILIES.exists():
        print("\n  data/benchmark_families.json is absent, so no family-combined figure")
        return
    payload = json.loads(FAMILIES.read_text())
    assignment = payload["families"]

    def null_for(row, k):
        n, m = row["chains"], row["matching"]
        if not n or not m:
            return None
        miss = comb(n - m, k) / comb(n, k) if n - m >= k else 0.0
        return 1.0 - miss

    # One bucket per family, holding its genomes' (hit, null) at each k.
    groups = {}
    unscoreable = []
    for accession, row in rows_by_accession.items():
        family = assignment.get(accession)
        if family is None:
            continue
        if null_for(row, 1) is None:
            # No matching chain among the candidates, so there is no ranking to score. The
            # panel's own `unscoreable_note` says this is a finding about reach and belongs in
            # its own tier rather than in a recall denominator, and a zero here would be a
            # claim that the ranker put the target low rather than that it never had one.
            unscoreable.append((family, row["pathogen"]))
            continue
        groups.setdefault(family, []).append(row)

    print(f"\n  Unit of analysis: FAMILY. {sum(len(v) for v in groups.values())} scoreable "
          f"genome(s) in {len(groups)} family(ies); the per-genome table above treats each "
          f"genome as independent and this does not.")
    if unscoreable:
        for family, pathogen in unscoreable:
            print(f"    dropped as unscoreable (no matching chain among its candidates): "
                  f"{pathogen} [{family}]")

    rng = random.Random(20261119)
    names = sorted(groups)
    recorded = {
        "what": "Recall@k for the genome-to-ranked-targets stage with the ICTV family as the "
                "unit of analysis, and a bootstrap interval on the lift over a "
                "hypergeometric null.",
        "why": "benchmarks/README.md states the defect this answers: 26 of the panel's genomes "
               "sit in a pattern-identical group, so an aggregate over them as independent "
               "instances is inflated, and the prescription is to combine within family first "
               "and then across families. The per-genome figure is reported beside this one "
               "and is not replaced by it.",
        "unit_of_analysis": "family",
        "weighting": "mean of family means; each family carries one vote whatever its genome "
                     "count, because weighting by count reproduces the per-genome figure",
        "interval": "percentile bootstrap over FAMILIES, 10,000 draws, seeded 20261119",
        "null": "hypergeometric probability that a random ranking puts at least one matching "
                "chain in the top k, averaged within family before across families",
        "genomes_scoreable": sum(len(v) for v in groups.values()),
        "families": {name: [r["pathogen"] for r in groups[name]] for name in names},
        "dropped_as_unscoreable": [
            {"pathogen": pathogen, "family": family,
             "because": "no matching chain among its candidates, so there is no ranking to "
                        "score; a zero would claim the ranker put the target low when it "
                        "never had one to put anywhere"}
            for family, pathogen in unscoreable],
        "recall": {},
    }
    print(f"\n  {'':9}{'observed':>10}{'random':>10}{'lift':>9}"
          f"{'95% CI on the lift':>22}")
    for k in (1, 3, 5, 10):
        # Each family collapses to one observed proportion and one null proportion.
        per_family = []
        for family in names:
            members = groups[family]
            obs = sum(1.0 for r in members
                      if r["best"] is not None and r["best"] <= k) / len(members)
            null = sum(null_for(r, k) for r in members) / len(members)
            per_family.append((obs, null))
        observed = 100.0 * sum(o for o, _ in per_family) / len(per_family)
        random_at = 100.0 * sum(n for _, n in per_family) / len(per_family)
        draws = []
        for _ in range(10000):
            sample = [per_family[rng.randrange(len(per_family))]
                      for _ in range(len(per_family))]
            draws.append(100.0 * sum(o - n for o, n in sample) / len(sample))
        draws.sort()
        low, high = draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]
        print(f"  recall@{k:<3}{observed:>9.1f}%{random_at:>9.1f}%"
              f"{observed - random_at:>+8.1f}{f'[{low:+.1f}, {high:+.1f}]':>22}")
        recorded["recall"][f"at_{k}"] = {
            "observed_percent": round(observed, 2),
            "random_percent": round(random_at, 2),
            "lift_percentage_points": round(observed - random_at, 2),
            "lift_ci95": [round(low, 2), round(high, 2)],
            "excludes_zero": bool(low > 0),
            "per_family": {name: {"observed": round(o, 4), "random": round(n, 4)}
                           for name, (o, n) in zip(names, per_family)},
        }
    print(f"\n  An interval that excludes zero is a lift this panel supports at the family "
          f"level. One that includes it is a lift the per-genome table may still show, which "
          f"is the whole reason for printing both.")
    print(f"  Families, and how many genomes each carries: "
          + ", ".join(f"{f} {len(groups[f])}" for f in names))
    if record:
        # Written to the tree, not only printed. `runs/benchmark/` is a directory of 47
        # per-genome files that nothing tracks, so until now the panel-level figure existed
        # only in whatever terminal last ran `--report`. This project's standing rule is that
        # a number on a page or in a document is checkable against a record in the tree, and a
        # headline statistic kept nowhere is the case that rule exists for.
        sys.path.insert(0, str(ROOT))
        from hpc_pilots.common import provenance, write_json

        recorded["provenance"] = provenance()
        write_json(ROOT / "runs/benchmark_families.json", recorded)
        print(f"\n  wrote runs/benchmark_families.json")
    return recorded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="summarise what is on disk")
    ap.add_argument("--families", action="store_true",
                    help="also combine within family before combining across families, with a "
                         "bootstrap interval over families. The per-genome aggregate treats 47 "
                         "genomes as 47 independent draws and they are not")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new genomes")
    args = ap.parse_args()
    if args.report:
        return report(with_families=args.families)

    OUT.mkdir(parents=True, exist_ok=True)
    panel = json.loads(PANEL.read_text())["panel"]
    done = 0
    for i, entry in enumerate(panel, 1):
        path = OUT / f"{stem_of(entry['accession'])}.json"
        if path.exists():
            print(f"[{i}/{len(panel)}] {entry['pathogen']}: already on disk", flush=True)
            continue
        print(f"[{i}/{len(panel)}] {entry['pathogen']} ({entry['accession']}) ...", flush=True)
        try:
            result = rank_one(entry)
        except Exception as broke:
            # One unreachable endpoint must not cost the other forty-six genomes -- and must
            # not cost THIS one either. A failure is written beside the result path, never to
            # it: `path.exists()` is the resume check, so persisting an error row here made a
            # transient timeout drop that genome permanently and silently.
            detail = f"{type(broke).__name__}: {broke}"
            kind = ("holdout_removed_nothing" if "removed no entries" in str(broke)
                    else "error")
            path.with_suffix(".error.json").write_text(json.dumps(
                {"accession": entry["accession"], "pathogen": entry["pathogen"],
                 "kind": kind, "error": detail[:400],
                 "traceback": traceback.format_exc()[-1200:]}, indent=1))
            print(f"      {kind.upper()} {detail[:120]} (will retry on next run)", flush=True)
            continue
        found = hits(result, entry["targets"])
        best = min((h["rank"] for h in found), default=None)
        print(f"      {len(result['chains'])} chains, {result['reference_entries_removed']} "
              f"removed, best ground-truth rank {best}, {result['seconds']}s", flush=True)
        path.write_text(json.dumps(result, indent=1))
        path.with_suffix(".error.json").unlink(missing_ok=True)
        done += 1
        if args.limit and done >= args.limit:
            break
    report(with_families=args.families)


if __name__ == "__main__":
    main()
