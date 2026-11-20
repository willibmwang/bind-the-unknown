"""The whole onset chain on every genome in `data/benchmark_panel.json`, folds uncapped.

**Why this exists, when `chain.one` already runs one genome.** The bounds `chain` folds under
are a visitor's bounds. `FOLD_BUDGET_SECONDS` is 240 and `FOLD_REQUESTS_PER_RUN` is 40, both
sized so a run fits the wall-clock cap on the workflow that serves somebody waiting on a page,
and both are right for that. Summed over the panel they mean something else: 1,198 fold units
offered and 333 folded, 827 annotated domain units offered and 28 folded, 39 chains over the
folding ceiling and 60 left with no coordinates at all. The published coverage of the whole
panel was a statement about a web request's patience rather than about what the pipeline
reaches. This runs the same stages with `fold_limit=None`, `fold_budget=None` and the units of
one genome folded concurrently against the GPU deployment, so the coverage numbers become a
property of the pipeline.

**One genome at a time, and that is a rate limit rather than caution.** `intake` counts every
NCBI call against one per-source budget -- 3 requests a second without a key, 10 with one --
so running genomes in parallel divides the ceiling rather than multiplying it, and the excess
comes back as 429s that the lane retries and then records as absences. A rate limit does not
look like a rate limit downstream; it looks like a protein nobody could fetch. The folds are
where the parallelism goes, because they leave this machine for a host that autoscales.

**Resumable, one file per genome, and a genome already finished is skipped.** The pattern
`output/run_benchmark.py` already uses, for the same reason: a network failure halfway through
a panel should cost only what had not finished. The progress file is a summary and a marker;
the scientific artifact is the chain record `chain.one` writes to `runs/`.

**A run that broke is not recorded, and is not skipped either.** `chain.one` raises rather
than writing when `fold_integrity` says the missing structures are infrastructure failures
instead of declared refusals. That exception is caught here and written to `<stem>.error.json`
beside the results, so the genome is visibly outstanding rather than silently absent, and
re-running the panel retries exactly it.

    python3 output/run_panel.py                 # every genome not yet done
    python3 output/run_panel.py --workers 12    # folds in flight per genome
    python3 output/run_panel.py --report        # summarise what is on disk, run nothing
    python3 output/run_panel.py --force         # re-run genomes already finished
"""
import argparse
import json
import pathlib
import re
import socket
import subprocess
import sys
import time
import traceback

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PANEL = ROOT / "data/benchmark_panel.json"
OUT = ROOT / "runs/panel"


def route_everything_to_our_own_gpu():
    """Send every fold of a panel run to the deployment we pay for, and say why in one place.

    **Concurrency against a free public host is not a thing to do.** `folded_structure` routes
    a sequence of 400 residues or fewer to `api.esmatlas.com`, which is keyless and costs its
    operators money, and most units of most genomes are inside that. Its `RETRY_PAUSE` is
    described in the module as the inter-request pause owed a free host. A panel of 47 genomes
    folding twelve units at a time would put twelve concurrent requests on it, repeatedly, for
    hours. That is the whole argument: the parallelism this runner exists for is only honest
    against a host whose bill we receive.

    So for an offline panel run `ESMFOLD_ENDPOINT` is pointed at the same deployment
    `ESMFOLD_LONG_ENDPOINT` names and the local ceiling is raised to match what that deployment
    accepts. `_is_ours` keys the bearer token and the longer timeout on the host rather than on
    which variable named it, so both follow automatically.

    **What this costs, stated rather than discovered later.** The same weights on different
    hardware put the main protease 0.841 angstrom from `runs/folded_structure/mpro_nsp5.pdb`,
    the artifact this repository reproduces byte for byte against the public host. Folds made
    here are therefore not that artifact, and `folded_structure.citation` already says so by
    naming the host that served each one rather than crediting the Atlas. The committed
    reproduction is untouched because it is a separate artifact tested separately; what changes
    is that a panel's folds all come from one host, which is the condition under which coverage
    numbers across 47 genomes are comparable at all.

    Must run before `hpc_pilots.folded_structure` is first imported: every constant there is
    evaluated at import, which is the same reason that module now reads `.env` itself.
    """
    import os

    from hpc_pilots.common import load_env

    load_env()
    long_endpoint = os.environ.get("ESMFOLD_LONG_ENDPOINT")
    if not long_endpoint:
        sys.exit("ESMFOLD_LONG_ENDPOINT is unset, so there is no GPU deployment to route to. "
                 "A panel run will not fan concurrent requests at the public host instead; "
                 "set it in .env, or pass --workers 1 to run serially against the Atlas.")
    os.environ["ESMFOLD_ENDPOINT"] = long_endpoint
    os.environ.setdefault("ESMFOLD_MAX_RESIDUES",
                          os.environ.get("ESMFOLD_LONG_MAX_RESIDUES") or "1200")
    return long_endpoint

# What the record is asked to report back, so the progress file answers "did the folds actually
# arrive" without reopening a 10 MB chain record. Every one of these is a coverage field the
# chain already computes; none is derived here, because a number computed in two places is a
# number that will disagree in one of them.
COVERAGE = ("chains_ranked", "chains_with_no_coordinates", "chains_with_any_surface",
            "chains_over_the_folding_ceiling", "chains_rescued_by_the_domain_route",
            "fold_units_offered", "fold_units_folded",
            "domain_units_offered", "domain_units_folded",
            "surface_measured", "site_patch_measured_on_a_prediction")


def stem_of(accession):
    """The filename `chain.one` will write this genome's record under.

    Derived the same way rather than guessed: `one` slugs the first accession and appends
    nothing, so a segmented genome supplied as every segment comma-joined lands under its first
    segment. Duplicating the rule here is what lets the resume check look for a record without
    running the chain to find out where it went.
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", accession.split(",")[0]).strip("_")


def commit():
    """The commit this run was made at, or None outside a checkout."""
    try:
        sha = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        dirty = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain"],
                               capture_output=True, text=True, timeout=30)
        if sha.returncode:
            return None
        return {"commit": sha.stdout.strip(),
                "dirty": bool(dirty.stdout.strip())}
    except Exception:                                      # noqa: BLE001
        return None


def run_one(entry, workers):
    """One genome, every stage, folds uncapped. Returns the progress summary."""
    from hpc_pilots.chain import one

    started = time.time()
    case = one(entry["accession"], pathogen=entry["pathogen"], holdout=entry["holdout"],
               record=True, fold_workers=workers, fold_limit=None, fold_budget=None)
    coverage = case.get("coverage") or {}
    return {
        "accession": entry["accession"],
        "pathogen": entry["pathogen"],
        "holdout": entry["holdout"],
        "record": f"runs/onset_chain_{stem_of(entry['accession'])}.json",
        "seconds": round(time.time() - started, 2),
        "chain_seconds": case.get("seconds"),
        "fold_workers": workers,
        "fold_limit": None,
        "fold_budget": None,
        # The hold-out is verified rather than assumed, exactly as the ranking benchmark
        # verifies it: a zero here means the organism name did not match UniProt's spelling and
        # the virus identified itself through its own entries, which makes the run look good
        # for the wrong reason.
        "reference_entries_removed": next(
            (s["output"].get("reference_entries_removed")
             for s in (case.get("stages") or []) if s.get("stage") == 1), None),
        "fold_integrity": case.get("fold_integrity"),
        "coverage": {k: coverage.get(k) for k in COVERAGE},
        "host": socket.gethostname(),
        "provenance": commit(),
    }


def report():
    """What is on disk, and the panel-wide coverage it adds up to."""
    done = sorted(OUT.glob("*.json"))
    results = [json.loads(p.read_text()) for p in done if not p.name.endswith(".error.json")]
    errors = [p for p in done if p.name.endswith(".error.json")]
    panel = json.loads(PANEL.read_text())["panel"]
    print(f"{len(results)} of {len(panel)} genomes finished, {len(errors)} with an error\n")
    if errors:
        for p in errors:
            payload = json.loads(p.read_text())
            print(f"  ERROR {payload.get('pathogen', p.stem)}: "
                  f"{(payload.get('error') or '')[:150]}")
        print()
    if not results:
        return
    print(f"{'pathogen':<40}{'sec':>7}{'ranked':>8}{'no coord':>10}{'surface':>9}"
          f"{'units':>14}{'domains':>12}")
    for r in sorted(results, key=lambda r: r["pathogen"]):
        c = r["coverage"]
        print(f"{r['pathogen'][:39]:<40}{r['seconds']:>7.0f}"
              f"{str(c['chains_ranked']):>8}{str(c['chains_with_no_coordinates']):>10}"
              f"{str(c['chains_with_any_surface']):>9}"
              f"{str(c['fold_units_folded']) + '/' + str(c['fold_units_offered']):>14}"
              f"{str(c['domain_units_folded']) + '/' + str(c['domain_units_offered']):>12}")
    # Sums and not means. A mean over genomes of different sizes weights a five-chain virus the
    # same as a thirty-chain one, which is the error the site's own matrix is laid out to avoid.
    print()
    for key in COVERAGE:
        total = sum(r["coverage"][key] or 0 for r in results)
        print(f"  {key:<46}{total}")
    untrustworthy = [r["pathogen"] for r in results
                     if not (r.get("fold_integrity") or {}).get("trustworthy", True)]
    if untrustworthy:
        print(f"\n  recorded but not trustworthy: {', '.join(untrustworthy)}")
    self_identified = [r["pathogen"] for r in results
                       if not r.get("reference_entries_removed")]
    if self_identified:
        print(f"\n  hold-out removed nothing (self-identified): {', '.join(self_identified)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12,
                    help="Folds in flight per genome. Must stay at or under the folding "
                         "deployment's own container cap, or the surplus queues at the host "
                         "and the concurrency buys nothing")
    ap.add_argument("--limit", type=int, help="Stop after this many genomes this invocation")
    ap.add_argument("--report", action="store_true", help="Summarise what is on disk, run nothing")
    ap.add_argument("--force", action="store_true", help="Re-run genomes already finished")
    ap.add_argument("--only", help="Run only genomes whose accession or pathogen contains "
                                   "this text, case-insensitively. For bringing one genome up "
                                   "before trusting a panel to it")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if args.report:
        report()
        return

    if args.workers > 1:
        host = route_everything_to_our_own_gpu()
        print(f"every fold routed to {host}")
    import os
    print(f"local ceiling {os.environ.get('ESMFOLD_MAX_RESIDUES', '400')} residues")

    panel = json.loads(PANEL.read_text())["panel"]
    if args.only:
        needle = args.only.lower()
        panel = [e for e in panel if needle in e["accession"].lower()
                 or needle in e["pathogen"].lower()]
        if not panel:
            sys.exit(f"no panel genome matches {args.only!r}")
    todo = [e for e in panel
            if args.force or not (OUT / f"{stem_of(e['accession'])}.json").exists()]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(todo)} genome(s) to run, {len(panel) - len(todo)} already on disk, "
          f"{args.workers} folds in flight each")

    started = time.time()
    for index, entry in enumerate(todo, 1):
        stem = stem_of(entry["accession"])
        error_path = OUT / f"{stem}.error.json"
        print(f"[{index}/{len(todo)}] {entry['pathogen']} ({entry['accession'][:40]}) ...",
              flush=True)
        try:
            summary = run_one(entry, args.workers)
        except Exception as failure:                       # noqa: BLE001
            # Written beside the results rather than raised, so one genome's outage does not
            # end a panel that has forty-six others to run. The next invocation retries it.
            error_path.write_text(json.dumps(
                {"accession": entry["accession"], "pathogen": entry["pathogen"],
                 "error": f"{type(failure).__name__}: {failure}",
                 "traceback": traceback.format_exc()[-3000:],
                 "host": socket.gethostname(), "provenance": commit()}, indent=2))
            print(f"    FAILED, recorded in {error_path.name}: "
                  f"{type(failure).__name__}: {str(failure)[:200]}", flush=True)
            continue
        error_path.unlink(missing_ok=True)
        (OUT / f"{stem}.json").write_text(json.dumps(summary, indent=2))
        c = summary["coverage"]
        print(f"    {summary['seconds']:.0f}s  ranked {c['chains_ranked']}  "
              f"no coordinates {c['chains_with_no_coordinates']}  "
              f"units {c['fold_units_folded']}/{c['fold_units_offered']}  "
              f"domains {c['domain_units_folded']}/{c['domain_units_offered']}", flush=True)
    print(f"\n{len(todo)} genome(s) in {(time.time() - started) / 60:.1f} min")
    report()


if __name__ == "__main__":
    main()
