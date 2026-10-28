"""Family and domain assignment for one protein, keyed on its sequence rather than its name.

What this answers is narrow and worth stating precisely: *which curated families and domains
claim this sequence, over which residues, and how large is the thing doing the claiming*. It
is the step that turns a claim about a protein into a claim about a site, which is the only
form a binder campaign can act on -- `structures_evidence.ranges` already refuses a target
whose range is neither derived from a UniProt feature nor declared with a citation, and for a
pathogen absent from `configs/structures.json` there is at present nothing that can derive
one. This module derives them.

**It does not reach the endpoint `selector.PROPOSED` names, and the reason is measured.** The
proposed lane was InterPro's entry-by-protein route, `entry/all/protein/uniprot/{acc}`. That
route fails on this repository's own flagship accession. P0DTD1 -- SARS-CoV-2 pp1ab, the
accession behind four of the eleven targets in `configs/structures.json` -- carries 206
entries, and the route returned HTTP 408 at `page_size=200` after 66 s, HTTP 500 at
`page_size=100`, and took 25 s for a single page of 50. Served safely it needs eleven
paginated requests at `page_size=20`, around 75 s, which is worse than the structural tool
this one is supposed to feed. The InterPro **Matches API**, shipped with InterPro 110.0 and
undocumented in the proposal, answers the same question for the same protein in **one request
and 2.2 s**, and returns 267 matches where the paginated route returns 206. So the lane is
implemented against the better endpoint and the proposal's scope note is wrong on its facts.

The Matches API is keyed on the **MD5 of the amino-acid sequence**, not on an accession, and
three consequences follow that are worth more than the speed.

*It works without an accession.* A translated ORF off `intake`'s six-frame caller can be
looked up directly, with no UniProt entry and no job queue, provided the exact sequence is in
UniParc. That is the one route that reaches a pathogen protein nothing has catalogued.

*It carries site-level residues.* CDD matches arrive with named `sites` -- "receptor binding
site", "putative RNA binding site" -- enumerated per residue. A domain boundary says where to
cut a construct; a site says where to aim.

*It mixes three epistemic classes in one list, and pooling them would be the costliest
mistake available here.* Of the 267 matches on P0DTD1, **124 come from InterPro-N**, a neural
network from Google DeepMind. An InterPro-N hit borrows a member-database accession -- it will
say `Pfam`, `CDD`, `SUPERFAMILY` -- but carries no E-value, no HMM bounds, no integrated
InterPro entry, and a `score` that is a probability between 0.53 and 1.0 rather than a bit
score. Read naively it is indistinguishable from a curated HMM match by accession alone. So
this module splits the return three ways and never merges them:

  `assignments`         curated signature matches, from databases InterPro types as `entry`.
                        An annotation transferred from relatives. Not a measurement.
  `predicted`           InterPro-N. A model's assertion, `unverified=True`, always.
  `sequence_features`   transmembrane, signal-peptide, disorder and coil predictions, from
                        databases InterPro types as `feature`. Computed from *this* sequence
                        rather than transferred from relatives, and still not measured.

The classification is read from InterPro's own `/api/` root at call time rather than written
into this file, for the reason `configs/structures.json` exists: a list of database names in
source drifts from the service and nothing notices.

**What no part of this establishes.** A family assignment is a statement about homologues. It
is not evidence that this pathogen's protein folds, is soluble, is expressible, is exposed, or
is essential -- those are the axes the dossier actually gates on, and a signature match speaks
to none of them. The size of a family is likewise not a measure of its usefulness and runs the
other way as often as not: on the Nipah G glycoprotein the CDD model `cd15468` holds 17
proteins and is nearly pathogen-specific, while `SSF50939` holds 43,086 and is a fold-level
grouping that says nothing about Nipah. And the dates are shallower than they look. SUPERFAMILY
is frozen at v1.75, released **2010-11-08**; CATH-Gene3D at 4.3.0, 2020-10-21. A SUPERFAMILY
assignment fetched today is a fifteen-year-old annotation, so the date attached to every row
here is the release date of the database serving it -- an upper bound on when the assignment
became available, and never the date of an experiment.

Absence is recorded and never filled in. An MD5 that UniParc does not hold returns HTTP 404
`{"detail":"Sequence not found"}`, and the payload then says so in `absences` with a real zero
in `counts`. A protein with no family is a finding.
"""
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request

from .common import ROOT, content_digest, provenance, write_json

# InterPro 110.0's pre-computed match store, keyed on the MD5 of the uppercased sequence.
# One request answers the whole question; see the module docstring for why the entry-by-protein
# route this replaces could not.
MATCHES = "https://www.ebi.ac.uk/interpro/matches/api/matches/"
# The classic API, used for two things the match store does not carry: how large a family is,
# and which databases are families rather than predictors.
INTERPRO = "https://www.ebi.ac.uk/interpro/api/"
# Accession to sequence. UniProt publishes `sequence.md5` itself, which this module
# nonetheless recomputes locally -- a free integrity check on the only join in the pipeline.
UNIPROT = "https://rest.uniprot.org/uniprotkb/"

# Sent on every request. The structural tool sends the default urllib agent to EMBL-EBI 36
# times per call with no contact string; that is a defect and not a precedent.
USER_AGENT = "hpc-pilots/1.0 (research prototype; contact via repository)"
# Between successive requests. EMBL-EBI publishes no hard limit on these endpoints, which is
# a reason to be careful rather than a licence. At six requests a call it costs two seconds
# and removes any question of whether we are the burst in somebody's log.
PAUSE = 0.34

# Per request. The spec's `worst_case_seconds` is arithmetic over this and the request
# ceiling -- 3 fixed requests plus `max_families` -- rather than an estimate: at the default
# cap that is 15 x 25 + 15 x 0.34, so 381 s against a 600 s wall budget. Observed times on
# these three endpoints run 0.65-6.5 s, so 25 is about four times the slowest measured. A
# 90-second timeout, which is what this module was first written with, would have declared a
# worst case of 23 minutes that the gate would refuse on every call.
TIMEOUT = 25
# Requests made before the family lookups: UniProt, the match store, the release table.
FIXED_REQUESTS = 3

# The match store names its neural source in this field. Not inferred from a missing E-value,
# because a missing E-value is also how CDD reports some curated hits.
PREDICTED_SOURCE = "InterPro-N"

# Used only if the `/api/` root cannot be read, and marked as such in `absences` when it is.
# Derived at call time in the normal path, for the reason the docstring gives.
FALLBACK_FEATURES = ("tmbed", "phobius", "signalp_p", "signalp_e", "mobidblt", "elm", "coils",
                     "cathfunfam")


def _get(url, calls, sources, allow_absent=()):
    """One GET, counted, with the agent attached and the absence codes handed back rather than raised.

    `urllib.request.urlopen` is spelled out at the call site on purpose. The project counts
    requests by patching that attribute, so a bound local name imported once would make this
    module record as zero network calls -- which is exactly the false-cost defect the
    structural tool's spec was corrected for.
    """
    calls.append(url)
    sources.append(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as handle:
            code = handle.getcode()
            body = handle.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        if error.code in allow_absent:
            return None, error.code
        raise RuntimeError(f"{url} returned HTTP {error.code} {error.reason}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"{url} could not be reached: {error.reason}") from error
    finally:
        time.sleep(PAUSE)
    # A 204 with an empty body is how the classic API reports an accession it does not
    # annotate, and an empty body has also come back from a 200. Both parse to nothing, and
    # `json.loads` on either raises `JSONDecodeError` from inside the caller's fifth round.
    # Neither is an error here; both are an absence, and an absence is reported.
    if not body.strip():
        return None, code
    try:
        return json.loads(body), code
    except json.JSONDecodeError:
        return None, code


def md5_of(sequence):
    """UniParc's key for a sequence: MD5 of the uppercased residues, hex, uppercased."""
    residues = "".join(sequence.split()).upper()
    if not residues:
        raise ValueError("Cannot key an empty sequence")
    return hashlib.md5(residues.encode("ascii", "strict")).hexdigest().upper()


def _databases(calls, sources, absences):
    """InterPro's own release table: which databases are families, and when each last shipped.

    Two questions answered by one request, and both are claims about the service rather than
    about this pathogen, so neither belongs in source.
    """
    payload, _ = _get(INTERPRO, calls, sources, allow_absent=(404, 500, 503))
    if not payload or "databases" not in payload:
        absences.append({"what": "InterPro database release table", "url": INTERPRO,
                         "consequence": "Feature databases are separated by a hard-coded list "
                                        "instead of by InterPro's own type field, and no "
                                        "assignment carries a release date."})
        return {name: {"kind": "feature" if name in FALLBACK_FEATURES else "entry",
                       "version": None, "released": None, "derived": False}
                for name in FALLBACK_FEATURES}
    out = {}
    for name, row in payload["databases"].items():
        if not isinstance(row, dict):
            continue
        out[name.lower()] = {"kind": row.get("type"), "version": row.get("version"),
                             "released": row.get("releaseDate"),
                             "name": row.get("name"), "derived": True}
    return out


# The match store spells a library the way a human reads it; the release table keys the same
# database by its API slug. Nothing joins them, so the join is written down here and the
# failure to join is recorded rather than guessed at.
LIBRARY_SLUG = {"pfam": "pfam", "cdd": "cdd", "cath-gene3d": "cathgene3d",
                "cath-funfam": "cathfunfam", "superfamily": "ssf", "smart": "smart",
                "panther": "panther", "prosite profiles": "profile",
                "prosite patterns": "prosite", "prosite": "prosite", "hamap": "hamap",
                "pirsf": "pirsf", "prints": "prints", "sfld": "sfld", "ncbifam": "ncbifam",
                "antifam": "antifam", "mobidb lite": "mobidblt", "mobidb-lite": "mobidblt",
                "phobius": "phobius", "tmbed": "tmbed", "coils": "coils", "elm": "elm",
                "signalp": "signalp_e", "signalp_euk": "signalp_e",
                "signalp_prok": "signalp_p", "interpro-n": "interpro_n",
                # PIRSR is a site-rule library, typed `other` by InterPro rather than `entry`.
                # It annotates residues and not families, so it is routed with the predictors
                # below and not with the curated family assignments.
                "pirsr": "pirsr"}


def _classify(source, library, databases):
    """Which of the three epistemic buckets a match belongs in, and the date behind it.

    Returns `(bucket, slug, release)`. `bucket` is one of `predicted`, `feature`, `entry`.
    An unrecognised library is typed `entry` and flagged, because silently dropping a match
    is worse than carrying one whose class is unresolved.
    """
    if source == PREDICTED_SOURCE:
        slug = "interpro_n"
    else:
        slug = LIBRARY_SLUG.get((library or "").lower().strip())
    row = databases.get(slug or "", {})
    release = row.get("released")
    version = row.get("version")
    if source == PREDICTED_SOURCE:
        return "predicted", slug, release, version, True
    kind = row.get("kind")
    # `feature` is InterPro's own type for the topology and disorder predictors; `other`
    # covers the site-rule libraries, which annotate residues rather than assign a family.
    # Neither is a family assignment, so neither may land in `assignments`.
    if kind in ("feature", "other"):
        return "feature", slug, release, version, bool(slug)
    return "entry", slug, release, version, bool(slug) and kind is not None


def _row(match, databases):
    """One match, flattened, with its class and its date attached and nothing merged."""
    signature = match.get("signature") or {}
    release = (signature.get("signatureLibraryRelease") or {})
    library = release.get("library")
    source = match.get("source")
    bucket, slug, released, version, resolved = _classify(source, library, databases)
    entry = signature.get("entry") or {}
    spans, sites = [], []
    for location in match.get("locations") or []:
        fragments = location.get("location-fragments") or [
            {"start": location.get("start"), "end": location.get("end")}]
        for fragment in fragments:
            if fragment.get("start") is not None and fragment.get("end") is not None:
                spans.append({"start": int(fragment["start"]), "end": int(fragment["end"]),
                              "discontinuity": fragment.get("dc-status")})
        for site in location.get("sites") or []:
            residues = [{"position": int(s["start"]), "residue": s.get("residue")}
                        for s in (site.get("siteLocations") or [])
                        if s.get("start") is not None]
            sites.append({"description": site.get("description"),
                          "residue_count": site.get("numLocations"),
                          "residues": sorted(residues, key=lambda r: r["position"])})
        # An E-value is the thing that makes a curated match falsifiable, and InterPro-N has
        # none. Kept per location rather than summarised, because a two-domain match has two.
    evalues = [l.get("evalue") for l in (match.get("locations") or [])
               if l.get("evalue") is not None]
    scores = [l.get("score") for l in (match.get("locations") or [])
              if l.get("score") is not None]
    bounds = [l.get("hmmBounds") for l in (match.get("locations") or []) if l.get("hmmBounds")]
    return {
        "signature": signature.get("accession"),
        "signature_name": signature.get("name"),
        "signature_description": signature.get("description"),
        "signature_type": signature.get("type"),
        "source": source,
        "database": slug,
        "database_version": version,
        "database_release_date": released,
        "database_resolved": resolved,
        "interpro_entry": entry.get("accession"),
        "interpro_name": entry.get("name"),
        "interpro_description": entry.get("description"),
        "interpro_type": entry.get("type"),
        "spans": spans,
        "sites": sites,
        "evalue": min(evalues) if evalues else None,
        "score": max(scores) if scores else None,
        "hmm_bounds": sorted(set(bounds)) or None,
        # The whole point of the three-way split, carried on every row so that a consumer
        # that flattens the payload cannot lose it.
        "unverified": bucket == "predicted",
        "basis": {"predicted": f"{PREDICTED_SOURCE} neural-network prediction; the score is a "
                               f"probability, not a bit score, and no E-value exists",
                  "feature": "Predicted from this sequence by a topology, signal-peptide or "
                             "disorder model; not transferred from relatives and not measured",
                  "entry": "Curated signature match; an annotation transferred from the "
                           "family's members and not a measurement on this pathogen",
                  }[bucket],
    }, bucket


def _merge(spans):
    """Collapse overlapping or abutting intervals, so one family yields one range list."""
    ordered = sorted(({"start": s["start"], "end": s["end"]} for s in spans),
                     key=lambda s: (s["start"], s["end"]))
    out = []
    for span in ordered:
        if out and span["start"] <= out[-1]["end"] + 1:
            out[-1]["end"] = max(out[-1]["end"], span["end"])
        else:
            out.append(dict(span))
    return out


def _family(accession, calls, sources, absences):
    """One InterPro entry's size, hierarchy, cited literature and description provenance.

    The size is fetched here and not from the match store, and not from the per-protein route
    with `extra_fields=counters` either -- on that route the counters are scoped to the
    filtered protein, so every family comes back holding one protein. A size of 1 for every
    family would read as a measurement and be an artefact of the query.
    """
    url = f"{INTERPRO}entry/interpro/{urllib.parse.quote(accession)}?extra_fields=counters"
    payload, code = _get(url, calls, sources, allow_absent=(204, 404, 410, 500, 503))
    if not payload or "metadata" not in payload:
        absences.append({"what": f"counters for InterPro entry {accession}", "url": url,
                         "http_status": code,
                         "consequence": "Family size is unknown for this entry; it is not "
                                        "zero and must not be read as zero."})
        return None
    meta = payload["metadata"]
    counters = (payload.get("extra_fields") or {}).get("counters") or {}
    members = meta.get("member_databases") or {}
    name = meta.get("name") or {}
    hierarchy = meta.get("hierarchy") or {}
    pmids = sorted({int(v["PMID"]) for v in (meta.get("literature") or {}).values()
                    if v.get("PMID")})
    return {
        "accession": accession,
        "name": name.get("name"),
        "short_name": name.get("short"),
        "type": meta.get("type"),
        # How many entries the family holds, in the two senses that differ: how many curated
        # signatures InterPro integrates into it, and how many sequences carry it.
        "member_database_entries": sum(len(v) for v in members.values()),
        "member_databases": {db: sorted(entries) for db, entries in members.items()},
        "proteins_in_family": counters.get("proteins"),
        "taxa_in_family": counters.get("taxa"),
        "structures_in_family": counters.get("structures"),
        "proteomes_in_family": counters.get("proteomes"),
        "domain_architectures": counters.get("domain_architectures"),
        "alphafold_models": (counters.get("structural_models") or {}).get("alphafold"),
        "children": [c.get("accession") for c in (hierarchy.get("children") or [])],
        "go_terms": [{"id": g.get("identifier"), "name": g.get("name"),
                      "category": (g.get("category") or {}).get("name")}
                     for g in (meta.get("go_terms") or [])],
        "cited_pmids": pmids,
        # InterPro 110 serves machine-written entry descriptions, flagged in its own metadata.
        # A family blurb a language model wrote is not curation and is marked as not curation.
        "description_is_llm_written": bool(meta.get("is_llm")),
        "description_llm_reviewed": bool(meta.get("is_reviewed_llm")),
    }


def entries(accession, sequence="", include_predicted=True, max_families=12, record=True):
    """Families, domains and residue ranges claiming one protein, split by what each claim is.

    `accession` is a UniProt accession, or -- when `sequence` is supplied -- the caller's label
    for a sequence that may have no accession at all. `sequence` is the amino-acid string; pass
    it to key a translated ORF directly and skip UniProt, which is the route that reaches a
    protein nothing has catalogued. Nothing checks that a supplied sequence belongs to the
    label, and the payload says so.

    `include_predicted` keeps InterPro-N's neural predictions in the return. They are segregated
    under `predicted` and flagged `unverified` either way; False drops them and records the count
    dropped, because a silently shorter list is a different answer to the same question.

    `max_families` caps the per-entry size lookups, which is what makes the worst case something
    a wall gate can refuse. 0 skips them and records the skip.
    """
    calls, sources, absences, limitations = [], [], [], []

    if sequence:
        residues = "".join(sequence.split()).upper()
        digest = md5_of(residues)
        length, organism, protein_name, source = len(residues), None, None, "caller"
        limitations.append("The sequence was supplied by the caller and nothing here verified "
                           f"that it is the sequence of {accession!r}. The label is the "
                           "caller's claim; only the MD5 was used.")
    else:
        url = (f"{UNIPROT}{urllib.parse.quote(accession)}.json"
               f"?fields=sequence,protein_name,organism_name")
        payload, code = _get(url, calls, sources, allow_absent=(400, 404))
        if not payload or not (payload.get("sequence") or {}).get("value"):
            absences.append({"what": f"UniProt sequence for {accession}", "url": url,
                             "http_status": code,
                             "consequence": "No sequence, so no MD5 and no lookup. This is an "
                                            "unresolved accession, not a protein without "
                                            "families."})
            return _finish(accession, None, None, None, None, "uniprot", [], [], [], {},
                           absences,
                           limitations + ["The accession did not resolve at UniProt, so this "
                                          "payload reports nothing about any protein."],
                           calls, sources, include_predicted, max_families, 0, record)
        block = payload["sequence"]
        residues = block["value"].upper()
        digest = md5_of(residues)
        length = block.get("length") or len(residues)
        organism = (payload.get("organism") or {}).get("scientificName")
        protein_name = (((payload.get("proteinDescription") or {}).get("recommendedName")
                         or {}).get("fullName") or {}).get("value")
        source = "uniprot"
        # The only join in this pipeline is accession -> sequence -> MD5, and it is checkable
        # for free because UniProt publishes its own MD5. A mismatch is recorded, never fixed.
        if (block.get("md5") or "").upper() not in ("", digest):
            absences.append({"what": f"MD5 agreement for {accession}",
                             "url": url, "http_status": code,
                             "consequence": f"UniProt reports {block['md5']} and the sequence "
                                            f"it served hashes to {digest}. The lookup used "
                                            f"the locally computed value."})

    # The match store is asked before the release table, so a sequence UniParc does not hold
    # costs one request and not two. The release table is only needed to classify matches,
    # and an absence has none to classify.
    url = f"{MATCHES}{digest}"
    payload, code = _get(url, calls, sources, allow_absent=(404, 422))
    if payload is None:
        absences.append({"what": f"InterPro matches for MD5 {digest}", "url": url,
                         "http_status": code,
                         "consequence": "UniParc does not hold this exact sequence, so no "
                                        "pre-computed matches exist. A real zero: this protein "
                                        "has no recorded family assignment at this endpoint, "
                                        "which is not the same as having no family."})
        return _finish(accession, digest, length, organism, protein_name, source, [], [], [],
                       {}, absences,
                       limitations + ["The sequence is not in UniParc, so the pre-computed "
                                      "match store has nothing for it. InterProScan would "
                                      "compute matches de novo; it is a queued job and asks "
                                      "for a contact email, so it is not reached from here."],
                       calls, sources, include_predicted, max_families, 0, record)

    databases = _databases(calls, sources, absences)

    buckets = {"entry": [], "predicted": [], "feature": []}
    unresolved = []
    for match in payload.get("matches") or []:
        row, bucket = _row(match, databases)
        buckets[bucket].append(row)
        if not row["database_resolved"]:
            unresolved.append(row["source"])
    dropped = 0
    if not include_predicted:
        dropped = len(buckets["predicted"])
        buckets["predicted"] = []

    # Residue ranges, consolidated per integrated InterPro entry. This is the fact the rest of
    # the system is short of: `structures_evidence` needs an interval and today derives one
    # only for the eleven targets hand-written into `configs/structures.json`.
    by_entry = {}
    for row in buckets["entry"]:
        key = row["interpro_entry"]
        if not key:
            continue
        slot = by_entry.setdefault(key, {"entry": key, "name": row["interpro_name"],
                                         "type": row["interpro_type"], "spans": [],
                                         "from_signatures": [], "databases": [],
                                         "database_release_date": None})
        slot["spans"].extend(row["spans"])
        slot["from_signatures"].append(row["signature"])
        if row["database"]:
            slot["databases"].append(row["database"])
        dates = [d for d in (slot["database_release_date"], row["database_release_date"]) if d]
        slot["database_release_date"] = min(dates) if dates else None
    ranges = []
    for key in sorted(by_entry):
        slot = by_entry[key]
        ranges.append({**slot, "ranges": _merge(slot["spans"]),
                       "from_signatures": sorted(set(slot["from_signatures"])),
                       "databases": sorted(set(slot["databases"]))})
        ranges[-1].pop("spans")

    # Family sizes, for the integrated entries only and capped. Member-database entries are
    # not followed: the cap has to bite somewhere and the integrated entry is the identity a
    # literature query can actually use.
    families = {}
    wanted = [r["entry"] for r in ranges]
    for key in wanted[:max(0, int(max_families))]:
        row = _family(key, calls, sources, absences)
        if row:
            families[key] = row
    if len(wanted) > max(0, int(max_families)):
        absences.append({"what": "family sizes beyond the cap",
                         "url": None, "http_status": None,
                         "consequence": f"{len(wanted) - max(0, int(max_families))} of "
                                        f"{len(wanted)} integrated entries were not sized "
                                        f"because max_families={max_families}. Their size is "
                                        "unknown, not zero."})

    return _finish(accession, digest, length, organism, protein_name, source,
                   buckets["entry"], buckets["predicted"], buckets["feature"], databases,
                   absences, limitations, calls, sources, include_predicted, max_families,
                   dropped, record, ranges=ranges, families=families, unresolved=unresolved,
                   protein_name_hint=protein_name)


def _finish(accession, digest, length, organism, protein_name, source, assignments, predicted,
            features, databases, absences, limitations, calls, sources, include_predicted,
            max_families, dropped, record, ranges=(), families=None, unresolved=(),
            protein_name_hint=None):
    """Assemble the payload. One exit, so every return carries the same fields."""
    families = families or {}
    sites = [{"interpro_entry": row["interpro_entry"], "signature": row["signature"],
              "database": row["database"], "description": site["description"],
              "residue_count": site["residue_count"], "residues": site["residues"],
              "unverified": row["unverified"]}
             for row in list(assignments) + list(predicted)
             for site in row["sites"]]

    # Whether the family route gives a literature tool something to search. The question is
    # not rhetorical: a novel target has no name anything has published on, and the family
    # does. A key is a string a query can carry, so names and accessions count and a count
    # does not.
    terms = sorted({t for row in families.values()
                    for t in (row.get("name"), row.get("short_name")) if t})
    accessions = sorted(families) or sorted({r["entry"] for r in ranges if r["entry"]})
    pmids = sorted({p for row in families.values() for p in row.get("cited_pmids") or []})
    literature = {
        "usable": bool(terms or accessions),
        "query_terms": terms,
        "interpro_accessions": accessions,
        "cited_pmids": pmids,
        "why": ("A family name and an InterPro accession are both searchable strings, so a "
                "literature tool can query the family when it cannot query the target. "
                "Europe PMC indexes InterPro accessions unevenly, so the name is the term to "
                "search and the accession is the term to report."
                if terms or accessions else
                "No integrated InterPro entry was resolved and sized, so there is no family "
                "name or accession to query with. The family route is unavailable for this "
                "protein and a literature search must fall back to the target name."),
        "pmids_are_not_precedent": "The PMIDs are what InterPro's curators cited when building "
                                   "the entry. They are literature about the family, not "
                                   "evidence of a macrocycle or peptide campaign against this "
                                   "target, and counting them measures attention only.",
    }

    standing = [
        "A family or domain assignment is an annotation transferred from the family's other "
        "members. It is never a measurement on this pathogen, and it is not evidence that this "
        "protein folds, is soluble, is expressible, survives immobilisation, is surface "
        "exposed, or is essential.",
        "Residue ranges are the boundaries of a signature match, not the boundaries of a "
        "folded domain and not an epitope. They are where to cut a construct, subject to "
        "experiment, and a match boundary routinely disagrees with a crystallised one.",
        f"Every date here is `database_release_date`: when the release currently serving an "
        f"assignment shipped. It is an upper bound on when the assignment became available and "
        f"has nothing to do with when anything was measured. Two member databases are long "
        f"frozen -- SUPERFAMILY at v1.75 (2010) and CATH-Gene3D at 4.3.0 (2020) -- so a hit "
        f"from either is a decade-old annotation served today.",
        "Family size is not family usefulness, and the two often run opposite. A narrow model "
        "built for one pathogen pair can hold fewer than twenty proteins and be the most "
        "informative row in the payload; a fold-level superfamily can hold tens of thousands "
        "and say nothing about this organism.",
        f"`predicted` holds {PREDICTED_SOURCE} matches: neural-network predictions whose score "
        f"is a probability and which carry no E-value, no HMM bounds and no integrated InterPro "
        f"entry. They borrow member-database accessions, so by accession alone they are "
        f"indistinguishable from curated hits. They are flagged `unverified` and must not be "
        f"pooled with `assignments`. They also routinely re-predict signatures the curated "
        f"HMMs already found, so pooling would double-count as well as launder: on the Nipah "
        f"G glycoprotein all five predictions duplicate curated hits, which would inflate the "
        f"assignment count from six to eleven.",
        f"A {PREDICTED_SOURCE} row's `database_version` is the release currently serving that "
        f"database, not the release the model was trained against, and the two differ -- the "
        f"predictions borrow Pfam 37.3 accessions while the curated Pfam matches are served "
        f"from 38.2. A predicted accession is not guaranteed to still exist upstream.",
        "`sequence_features` holds transmembrane, signal-peptide, disorder and coil "
        "predictions together with site-rule annotations. These are computed from this "
        "sequence or applied as a rule, rather than transferred from relatives, which makes "
        "them a different claim from an assignment and still not a measurement. A predicted "
        "topology is not a localisation experiment.",
        "An entry's description may be machine-written; `description_is_llm_written` says "
        "when it is. A flagged description is not curation and must not be quoted as such.",
        "This reports no binding, no affinity, no inhibition and no druggability for anything, "
        "and nothing here ranks one target above another.",
    ]
    if not include_predicted:
        standing.append(f"include_predicted=False: {dropped} {PREDICTED_SOURCE} matches were "
                        f"dropped from this payload and are not reported individually.")
    if unresolved:
        standing.append(f"{len(unresolved)} matches came from a library this module could not "
                        f"map to an InterPro database slug, so they carry no release date and "
                        f"were typed as curated by default: "
                        f"{sorted(set(unresolved))}. An unmapped library is a drift between "
                        f"the match store's names and the release table's keys, not a reason "
                        f"to drop a match.")

    payload = {
        "accession": accession,
        "protein_name": protein_name,
        "organism": organism,
        "sequence_md5": digest,
        "sequence_length": length,
        "sequence_source": source,
        "assignments": tuple(sorted(assignments, key=lambda r: (r["database"] or "",
                                                                r["signature"] or ""))),
        "predicted": tuple(sorted(predicted, key=lambda r: (r["source"] or "",
                                                            r["signature"] or ""))),
        "sequence_features": tuple(sorted(features, key=lambda r: (r["database"] or "",
                                                                   r["signature"] or ""))),
        "residue_ranges": tuple(ranges),
        "families": families,
        "functional_sites": tuple(sites),
        "literature_key": literature,
        "database_releases": {k: v for k, v in sorted(databases.items())},
        "counts": {
            "assignments": len(assignments),
            "predicted": len(predicted),
            "predicted_dropped": dropped,
            "sequence_features": len(features),
            "integrated_entries": len(ranges),
            "families_sized": len(families),
            "functional_sites": len(sites),
            "transmembrane_predictions": sum(1 for r in features
                                             if (r["database"] or "") in ("tmbed", "phobius")),
            "network_calls": len(calls),
        },
        "absences": tuple(absences),
        "limitations": tuple(standing + list(limitations)),
        "parameters": {"accession": accession, "sequence_supplied": source == "caller",
                       "include_predicted": include_predicted, "max_families": max_families},
        "sources": [
            {"name": "InterPro Matches API (InterPro 110.0)", "url": MATCHES,
             "licence": "EMBL-EBI terms of use; InterPro data CC0 1.0"},
            {"name": "InterPro REST API", "url": INTERPRO,
             "licence": "EMBL-EBI terms of use; InterPro data CC0 1.0"},
            {"name": "UniProt REST", "url": UNIPROT, "licence": "CC BY 4.0"},
        ],
        "requests": tuple(sources),
        "interpretation": "Which curated families and domains claim this sequence, over which "
                          "residues, how large each family is, and which of the three kinds of "
                          "claim each row is. Not a measurement on this pathogen and not a "
                          "target ranking.",
        "provenance": provenance(),
    }
    payload["content_sha256"] = content_digest(payload)
    if record:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in accession) or "unnamed"
        write_json(ROOT / f"data/processed/families/{safe}.json", payload)
    return payload
