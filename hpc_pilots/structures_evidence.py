"""Structural evidence per target, fetched from public records and carrying its own dates.

Written after the scope narrowed to pathogen intake through target generation. The reason
this module exists rather than a wider one: structural evidence is the only target-quality
signal in this project that can be computed for a protein nobody has studied. A pocket can be
scored on a structure the day it is deposited; an antibody count, a citation count and a
measured affinity all require that somebody already cared. That asymmetry is the whole of the
project's saturation problem, and keeping the two kinds of evidence in separate columns is
what lets a ranker be built on one and validated against the other.

So every quantity here is recorded with the attention it required, not just its value. A
structure count *is* a research-attention proxy and is labelled as one. A pocket score
computed on a structure is not. Both come out of the same fetch and must not be pooled.

**Three things are fetched and none is asserted.** UniProt supplies the mature-chain and
domain boundaries, so `Mpro` means residues 3264-3569 of P0DTD1 because the entry says so and
not because this file says so. PDBe SIFTS supplies the residue range each deposited chain
actually covers, so a structure is assigned to a target by overlap rather than by its title.
PDBe supplies deposition dates, which the date boundary needs: a claim that a reagent was
tractable is only usable in an episode if it carries the date it became true.

**Why ranges rather than names.** The obvious implementation searches structure titles for
"main protease". It fails in both directions -- a polyprotein construct is titled for the
polyprotein, and a fragment screen is titled for the fragment -- and it fails silently, which
is worse. Overlap against a UniProt range is checkable, reproducible from the accession, and
wrong in ways that show up as a count rather than as a wrong structure quietly included.

**What a range cannot do.** Several targets in the packets are not mature chains. The S2
stem-helix is an epitope, the PA-PB1 interface is a binding site on a domain, the NA catalytic
site is a pocket inside a chain. UniProt has features for some and not others, so the config
carries two kinds of range: one read from the entry's own features, and one declared with a
citation. A declared range without a citation raises rather than defaulting, for the same
reason `targets.py` carries `verified=False` on the three claims asserted from memory -- the
distinction between a derived number and an asserted one is the thing worth keeping.
"""
import json
import urllib.error
import urllib.parse
import urllib.request

from .common import ROOT, content_digest, provenance, write_json

UNIPROT = "https://rest.uniprot.org/uniprotkb/"
SIFTS = "https://www.ebi.ac.uk/pdbe/api/mappings/best_structures/"
RESIDUE_MAPPING = "https://www.ebi.ac.uk/pdbe/api/mappings/uniprot/"
SUMMARY = "https://www.ebi.ac.uk/pdbe/api/pdb/entry/summary/"

# PDBe rejects very long POST bodies, and a failed batch is indistinguishable from an empty
# one in the response, so the batch is kept small enough that it has never been exercised.
DATE_BATCH = 400


def config():
    return json.loads((ROOT / "hpc_pilots/configs/structures.json").read_text())


# Statuses that mean "ask again", as opposed to "there is no such thing". A 503 from PDBe is
# a minute of maintenance and a 404 is a fact about the accession, and treating the first like
# the second is how a target came back unmeasurable in `runs/amenability_gate.json` with a
# reason that described the server rather than the protein.
TRANSIENT = (429, 500, 502, 503, 504)


def _get(url, timeout=120, attempts=4, backoff=2.0):
    """One JSON GET, retried on the statuses that mean the server is busy rather than sure.

    Linear backoff rather than exponential: these endpoints recover in seconds and the point
    is to outlast a blip, not to wait politely for minutes inside a measurement. A status
    outside `TRANSIENT` is raised on the first attempt, because retrying a 404 only delays
    the correct answer.
    """
    import time

    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as handle:
                if handle.status == 204:
                    return None
                return json.load(handle)
        except urllib.error.HTTPError as error:
            if error.code not in TRANSIENT or attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts - 1:
                raise
        time.sleep(backoff * (attempt + 1))


def entry(accession):
    """The UniProt entry, whole, so the caller can read features and the release date."""
    return _get(f"{UNIPROT}{urllib.parse.quote(accession)}.json")


def ranges(spec, record):
    """Resolve one target's declared ranges to residue intervals, or say why it could not.

    Returns intervals and the provenance of each, because a range read from the entry and a
    range asserted in the config are different kinds of claim and a consumer that cannot tell
    them apart will quote both the same way.
    """
    features = record.get("features", [])
    out = []
    for want in spec["ranges"]:
        if want["source"] == "uniprot_feature":
            hits = [f for f in features
                    if f["type"] == want["type"]
                    and f.get("description", "") == want["description"]]
            if len(hits) != 1:
                raise ValueError(
                    f"{spec['accession']}: {len(hits)} features match "
                    f"{want['type']}/{want['description']!r}; a range must resolve to exactly "
                    f"one feature or the assignment is a guess")
            location = hits[0]["location"]
            out.append({"start": location["start"]["value"], "end": location["end"]["value"],
                        "derived_from": f"UniProt {want['type']} {want['description']!r}",
                        "asserted": False})
        elif want["source"] == "declared":
            if not want.get("citation"):
                raise ValueError(f"{spec['accession']}: a declared range carries no citation; "
                                 f"an asserted interval without a source is not evidence")
            out.append({"start": want["start"], "end": want["end"],
                        "derived_from": want["citation"], "asserted": True})
        else:
            raise ValueError(f"Unknown range source {want['source']!r}")
    return out


def chains(accession, cache={}):
    """Every deposited chain mapped to this accession, with the range each one covers.

    Memoized per accession, following `residue_mapping` two functions below. This was the one
    uncached fetch on the decomposition path, and the shape of that path makes the omission
    expensive: `identify_local` returns at most three hits, so taking one polyprotein apart
    needs at most three distinct SIFTS payloads, while `family_chain_structures` asked for one
    per hit *per mature chain* -- up to three times sixteen for a coronavirus replicase, all but
    three of them identical. The cap on how many chains were scored at all existed because of
    that cost.

    A 404 is cached as the empty list it returns, which is correct: "no structure maps to this
    accession" is a fact about the accession. Other HTTP errors still raise and are not cached,
    so a transient failure is not frozen into the process.

    A 404 means no structure maps to the accession, and is returned as an empty list rather
    than raised. The distinction matters and was found the hard way: P03470, the neuraminidase
    of the strain whose deep mutational scan the packet cites, has no deposited structure at
    all, and the endpoint answers that with a 404 rather than an empty body. Treating it as a
    transport error would have dropped a target whose true structure count is zero -- which is
    a finding about the target, and one the packet's `reagent_tractability` assessment of
    `strong` does not obviously anticipate.
    """
    if accession in cache:
        return cache[accession]
    try:
        payload = _get(f"{SIFTS}{urllib.parse.quote(accession)}")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            cache[accession] = []
            return cache[accession]
        raise
    cache[accession] = (payload or {}).get(accession, [])
    return cache[accession]


def residue_mapping(pdb_id, accession, cache={}):
    """Per chain, the UniProt-to-author residue correspondence for one entry.

    **Why this is a second call and not a field on `chains`.** `best_structures` reports
    `start` and `end` beside `unp_start` and `unp_end`, which reads like the author numbering
    and is not: it is the residue's index within the entity. For the spike receptor-binding
    domain the entity indices are 4 to 198 where the author numbers are 333 to 527, and a clip
    applied in the wrong numbering selects no residue at all -- which is what it did, silently,
    on every chain whose construct does not start at residue 1.

    Returned as `{chain_id: [(unp_start, unp_end, author_start, author_end), ...]}`, one tuple
    per contiguous mapped segment. A chain whose entry gives no author number is absent, and
    the caller measures it unclipped rather than guessing an offset.
    """
    key = (pdb_id.lower(), accession)
    if key in cache:
        return cache[key]
    try:
        payload = _get(f"{RESIDUE_MAPPING}{urllib.parse.quote(pdb_id)}")
    except urllib.error.HTTPError as error:
        if error.code == 404:
            cache[key] = {}
            return {}
        raise
    block = ((payload or {}).get(pdb_id.lower(), {}).get("UniProt", {})
             .get(accession, {}) or {})
    out = {}
    for mapping in block.get("mappings", []):
        chain = mapping.get("chain_id")
        start, end = mapping.get("start") or {}, mapping.get("end") or {}
        first, last = start.get("author_residue_number"), end.get("author_residue_number")
        if chain is None or first is None or last is None:
            continue
        if mapping.get("unp_start") is None or mapping.get("unp_end") is None:
            continue
        out.setdefault(chain, []).append(
            (mapping["unp_start"], mapping["unp_end"], first, last))
    cache[key] = out
    return out


def dates(pdb_ids):
    """Deposition dates, batched. The date boundary is unusable without them."""
    found = {}
    ids = sorted(set(pdb_ids))
    for start in range(0, len(ids), DATE_BATCH):
        batch = ids[start:start + DATE_BATCH]
        request = urllib.request.Request(SUMMARY, data=",".join(batch).encode(), method="POST")
        with urllib.request.urlopen(request, timeout=180) as handle:
            payload = json.load(handle)
        for pdb, records in payload.items():
            stamp = records[0].get("deposition_date")
            if stamp:
                found[pdb.lower()] = f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
    return found


def covers(row, interval, minimum=0.5):
    """Does a deposited chain cover enough of an interval to be evidence about it?

    A threshold rather than any overlap at all. A polyprotein construct overlaps nsp5 by a
    single residue and is not a structure of the main protease; requiring half the interval
    keeps fragments of a domain out while admitting the constructs that are genuinely of it.
    The fraction is reported with every count so a reader can see what it did rather than
    inferring it, and the whole table can be rebuilt at another threshold.
    """
    start, end = row.get("unp_start"), row.get("unp_end")
    if start is None or end is None:
        return False
    overlap = max(0, min(end, interval["end"]) - max(start, interval["start"]) + 1)
    return overlap >= minimum * (interval["end"] - interval["start"] + 1)


def build(minimum=0.5, record=True):
    """Assemble the per-target structural table and write it with provenance."""
    spec = config()
    entries, out = {}, {}
    for target, declared in spec["targets"].items():
        accession = declared["accession"]
        if accession not in entries:
            entries[accession] = {"record": entry(accession), "chains": chains(accession)}
        record_ = entries[accession]["record"]
        intervals = ranges(declared, record_)
        rows = entries[accession]["chains"]

        # Grouped into entries before the decision, not after. An obligate complex is two
        # polymer entities and therefore two chains, so no single chain covers both ranges and
        # a per-chain test returns zero for every nsp10/nsp16 structure ever deposited -- which
        # is what the first version of this did, and the zero looked exactly like the genuine
        # zero that NA returns.
        by_entry = {}
        for row in rows:
            by_entry.setdefault(row["pdb_id"].lower(), []).append(row)
        matched = {}
        for pdb, entry_rows in by_entry.items():
            hit = [any(covers(r, i, minimum) for r in entry_rows) for i in intervals]
            keep = all(hit) if declared.get("requires_all_ranges") else any(hit)
            if keep:
                matched[pdb] = [r for r in entry_rows
                                if any(covers(r, i, minimum) for i in intervals)]
        stamps = dates(matched) if matched else {}
        dated = sorted((stamps[p], p) for p in matched if p in stamps)
        resolutions = [r.get("resolution") for rows_ in matched.values() for r in rows_
                       if r.get("resolution")]
        methods = sorted({r.get("experimental_method") for rows_ in matched.values()
                          for r in rows_ if r.get("experimental_method")})

        out[target] = {
            "accession": accession,
            "uniprot_length": record_["sequence"]["length"],
            "uniprot_release": record_.get("entryAudit", {}).get("lastSequenceUpdateDate"),
            "intervals": intervals,
            "requires_all_ranges": bool(declared.get("requires_all_ranges")),
            "coverage_threshold": minimum,
            # Labelled, not merely reported. This number is how much work has been done on the
            # target, which is what a pretrained policy recalls without reasoning, so it is a
            # covariate to regress out and never an input to a ranker that claims to predict
            # target quality.
            "structures": len(matched),
            "structures_are_an_attention_proxy": True,
            "earliest_deposition": dated[0][0] if dated else None,
            "earliest_pdb": dated[0][1] if dated else None,
            "latest_deposition": dated[-1][0] if dated else None,
            "best_resolution": min(resolutions) if resolutions else None,
            "methods": methods,
            "pdb_ids": sorted(matched),
            "caveat": declared.get("caveat"),
        }

    payload = {"experiment": "structural-evidence", "status": "complete",
               "coverage_threshold": minimum,
               "sources": [{"name": "UniProt", "url": UNIPROT,
                            "licence": "CC BY 4.0"},
                           {"name": "PDBe SIFTS best_structures", "url": SIFTS,
                            "licence": "EMBL-EBI terms of use; PDB data CC0 1.0"},
                           {"name": "PDBe entry summary", "url": SUMMARY,
                            "licence": "EMBL-EBI terms of use; PDB data CC0 1.0"}],
               "targets": out, "provenance": provenance(),
               "interpretation": "Counts of deposited structures per target, assigned by "
                                 "residue-range overlap against UniProt features rather than "
                                 "by title. A structure count is a record of research "
                                 "attention and is labelled as one; it is not a measure of "
                                 "target quality and must not be used as a ranker input. "
                                 "Nothing here predicts binding, affinity or inhibition."}
    # The same identity every other staged manifest carries: a hash over the payload with the
    # volatile timestamps stripped, so refetching unchanged records reproduces it. Without it
    # this table would be the one artifact in `data/processed` whose data cannot be identified
    # independently of when it was fetched, and the suite refuses that -- correctly.
    payload["content_sha256"] = content_digest(payload)
    if record:
        write_json(ROOT / "data/processed/structures/manifest.json", payload)
    return payload
