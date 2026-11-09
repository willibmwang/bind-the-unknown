"""Leave-one-organism-out ranking benchmark over the panel in `data/benchmark_panel.json`.

**What this measures.** For each virus the organism's own curated entries are deleted from
the reference before identification, so every protein has to be named through a relative.
Then: does the ranker put a protein an approved antiviral actually acts on near the top?
Recall@k over 46 genomes, rather than the three the project had before.

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
    from hpc_pilots.intake import decompose, rank_score, run_intake

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


def report():
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
        row = {"pathogen": res["pathogen"], "chains": len(res["chains"]),
               "matching": len(found),
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true", help="summarise what is on disk")
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new genomes")
    args = ap.parse_args()
    if args.report:
        return report()

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
            # One unreachable endpoint must not cost the other forty-five genomes -- and must
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
    report()


if __name__ == "__main__":
    main()
