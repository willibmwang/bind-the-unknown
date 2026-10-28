"""Homology search as a callable capability, on the fastest keyless endpoint that exists.

`intake.py` already reaches a remote aligner, and says in its own docstring at line 26 why no
recorded path ever used it: "It is a job queue, and a single 306-residue query sat at RUNNING
for" minutes on end. That is the defect this module exists to fix, and the fix is not a better
poll loop -- it is a different service.

**What was surveyed, and what each attempt actually returned.** Every line below is a live
result from this machine, not a recollection.

| candidate | keyless? | measured |
|---|---|---|
| EBI HMMER API v1 `phmmer` | yes, no email | **2.0 s** total for a 306-aa query vs SwissProt; 10.2 s vs all of UniProtKB |
| EBI job dispatcher `ncbiblast` | **no** -- `/run` without `email` returns HTTP 400 `<description>Please enter an email address</description>` | not run |
| EBI job dispatcher `hmmer3_phmmer` | **no** -- same HTTP 400 on the same field | not run |
| NCBI BLAST URL API | yes | still `Status=WAITING` at 62 s and beyond, with a 60-s minimum poll interval NCBI asks clients to honour |
| MMseqs2 `search.mmseqs.com/api/ticket` | n/a | HTTP 503 -- the public demo is not serving |
| HHpred `toolkit.tuebingen.mpg.de/api/` | n/a | HTTP 404 -- no documented keyless REST surface at that path |
| old HMMER `Tools/hmmer/search/phmmer` | n/a | HTTP 405 Not Allowed on POST -- the pre-v1 API is retired |
| UniProt UniRef clusters | yes, sub-second | returns cluster *membership*, not a search: no per-hit E-value and no pairwise identity to the query |

So the service the repo named is the one option that is **not** usable here, and for a reason
that has nothing to do with speed: EMBL-EBI's job dispatcher requires an `email` field, and a
tool that has to carry somebody's address is not an unauthenticated tool. The HMMER v1 API
needs no email, no key and no signup, and it is two orders of magnitude faster besides.

**The job model is pollable with a real worst case.** `POST /search/phmmer` returns a UUID
immediately; `GET /result/{id}` carries `status`, which was already `SUCCESS` on the *first*
poll for SwissProt. That is what lets `worst_case_seconds` be a number the wall gate can
refuse against, rather than the `None` that `structures.evidence` has to declare. The bound is
enforced here too: `max_wait_seconds` is a real ceiling, and crossing it raises rather than
returning an empty hit list, because a timeout is a failure and not a finding.

**What this does not establish, restated because it is the easiest thing in this file to get
wrong.** A homolog is somebody else's accession. The repo's standing warning is that treating
a 28%-identity hit as the query's own identity is a conflation, so every row here carries its
own `identity` and the payload carries `identity_warning` naming the best one; a caller that
reads the hit list without reading that field is reproducing the error. Beyond that: a Pfam
architecture is an annotation transferred from relatives and never a measurement on this
pathogen; conservation across curated database entries is not conservation across circulating
isolates, because SwissProt is not a sample of anything; and an E-value is a statement about
chance similarity, never about function, essentiality or whether a surface is bindable.

**Two silent-200 traps are handled explicitly** because both were triggered on purpose while
writing this. NCBI's `efetch` answers a nonexistent accession with HTTP **200** and a body of
`Error: F a i l e d  t o  u n d e r s t a n d  i d : ...`, and EBI's dbfetch answers one with
HTTP 200 and `ERROR 11 Unable to connect to database`. A resolver that trusted the status code
would hand that prose to an aligner as a query sequence. Both are detected and raised.
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .common import ROOT, content_digest, provenance, write_json

HMMER = "https://www.ebi.ac.uk/Tools/hmmer/api/v1"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
UNIPROT = "https://rest.uniprot.org/uniprotkb"

# Sent on every outbound request. `structures_evidence` sends the stock urllib agent to
# EMBL-EBI 36 times a call with no contact string, which is the politeness defect this
# module is written not to repeat.
AGENT = "hpc-pilots/1.0 (research prototype; contact via repository)"

# Between successive requests, for the same reason the agent string exists.
PAUSE = 0.34

# Floor on any single request's socket timeout. `max_wait_seconds` bounds the whole call by
# becoming each request's timeout, but a budget of zero cannot be handed to `urlopen`, and a
# ceiling below one round trip is unenforceable anyway: a 0.5s ceiling was measured exiting
# at 3.5s because the first poll had already been issued. So the declared worst case is
# `max_wait_seconds` plus at most this, and that is the number the wall gate can refuse.
MIN_TIMEOUT = 5.0

# `type: "seq"` databases phmmer accepts, as `GET /search/databases` listed them on
# 2026-10-27. `pfam` is deliberately absent: it is `type: "hmm"` and belongs to hmmscan, so
# naming it here would advertise a search this entry point cannot run. The live list is
# fetched on every call anyway and any drift from this tuple is reported in the payload
# rather than silently accepted.
DATABASES = ("swissprot", "uniprot", "refprot", "pdb", "rp15", "rp35", "rp55", "rp75",
             "mgnify30_c2", "mgnify30_c5_fl", "mgnify30_c5_ppfam")

# Identity at or below which a hit is a remote relative whose accession says nothing about
# the query's own identity. Not a cutoff -- nothing is dropped on it -- only the threshold
# `identity_warning` is phrased against.
CONFLATION_IDENTITY = 0.40

# One-letter amino acids plus the ambiguity, selenocysteine/pyrrolysine and gap characters a
# FASTA protein record may legally carry.
_AMINO = set("ACDEFGHIKLMNPQRSTVWYBXZUO*-.")
# The nucleotide alphabet is a subset of the above, so "is it protein" cannot be answered by
# membership alone; composition decides, and a nucleotide query is refused rather than aligned.
_NUCLEOTIDE = set("ACGTUN-.")

# UniProtKB accession, both the 6- and the 10-character forms, isoform suffix allowed.
_UNIPROT_ACCESSION = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[0-9]+)?$")

# Qualifiers NCBI writes into a `fasta_cds_aa` defline, e.g.
# `>lcl|NC_045512.2_prot_YP_009724389.1_1 [gene=ORF1ab] [protein=ORF1ab polyprotein] ...`
_QUALIFIER = re.compile(r"\[([a-z_]+)=([^\]]*)\]")


class Absent(ValueError):
    """A query that resolved to nothing retrievable.

    Distinct from a search that ran and found no homolog: that is a real zero and is
    returned, recorded and charged. This is "there was no query", which is a contract
    violation and must not be reported as an empty hit list.
    """


def _request(url, calls, data=None, accept="application/json"):
    """One outbound request, counted, with a contact string and a pause behind it.

    `calls` is a one-element list used as a counter, so the count in the payload is the
    number of requests the process actually issued rather than the number the author
    intended. `urllib.request.urlopen` and nothing else: the project counts network calls
    by patching `urlopen`, so a tool that reached for `requests` would certify a cost of
    zero while hitting EMBL-EBI, which is exactly the false certification this codebase has
    already been bitten by.
    """
    headers = {"User-Agent": AGENT, "Accept": accept}
    body = None
    if data is not None:
        body = json.dumps(data).encode()
        headers["Content-Type"] = "application/json"
    if calls[0]:
        time.sleep(PAUSE)
    calls[0] += 1
    request = urllib.request.Request(url, data=body, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        detail = ""
        try:
            detail = error.read().decode("utf-8", "replace")[:400]
        except Exception:  # noqa: BLE001 - the body is a courtesy, never the signal
            pass
        raise RuntimeError(
            f"HTTP {error.code} from {url}: {error.reason}. {detail}".strip()) from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"no response from {url}: {error.reason}") from error


def _json(url, calls, data=None):
    text = _request(url, calls, data=data)
    try:
        return json.loads(text)
    except ValueError as error:
        raise RuntimeError(
            f"{url} returned {len(text)} bytes that are not JSON: {text[:200]!r}") from error


def _guard_silent_error(text, url):
    """Raise on the two endpoints that answer a bad accession with HTTP 200 and prose.

    NCBI returns `Error: F a i l e d  t o  u n d e r s t a n d  i d : ...` -- the spacing is
    theirs -- and EBI's dbfetch returns `ERROR 11 Unable to connect to database [...]`, both
    under a 200. Collapsing whitespace is what makes the first one matchable.
    """
    head = " ".join(text[:400].split())
    squeezed = re.sub(r"(?<=\b\w) (?=\w\b)", "", head)
    for candidate in (head, squeezed):
        if re.match(r"^(Error|ERROR)\b", candidate):
            raise Absent(f"{url} answered HTTP 200 with an error body: {candidate[:220]!r}")
    return text


def _parse_fasta(text):
    """FASTA to `[(defline, sequence)]`, with no tolerance for an empty record."""
    records, defline, chunks = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if defline is not None:
                records.append((defline, "".join(chunks)))
            defline, chunks = line[1:].strip(), []
        elif defline is not None:
            chunks.append(line.strip())
    if defline is not None:
        records.append((defline, "".join(chunks)))
    return [(head, body.upper()) for head, body in records if body]


def _classify(text):
    """`"protein"`, `"nucleotide"` or `"accession"` for a bare string.

    Composition, not membership: `ACGT` is a legal protein alphabet subset, so a nucleotide
    query would otherwise sail through to an aligner that cannot use it.
    """
    body = "".join(text.split()).upper()
    if not body:
        return "accession"
    if len(body) >= 20 and all(character in _AMINO for character in body):
        nucleotide = sum(1 for character in body if character in _NUCLEOTIDE)
        return "nucleotide" if nucleotide / len(body) >= 0.9 else "protein"
    return "accession"


def _cds_catalogue(records, accession):
    """NCBI `fasta_cds_aa` records to rows carrying what the deflines actually say.

    The qualifiers are the point: `gene`, `protein`, `protein_id` and `location` are
    `target_name`, `protein_accession` and `residue_ranges` for free, from an annotation that
    already exists, with no ORF calling and no second opinion about where a gene starts.
    """
    catalogue = []
    for index, (defline, sequence) in enumerate(records):
        fields = dict(_QUALIFIER.findall(defline))
        catalogue.append({
            "cds_index": index,
            "gene": fields.get("gene"),
            "protein": fields.get("protein"),
            "protein_accession": fields.get("protein_id"),
            "location": fields.get("location"),
            "pseudo": "pseudo" in fields,
            "length": len(sequence),
            "source_accession": accession,
        })
    return catalogue


def _resolve(query, cds_index, calls, sources, attempts):
    """A query string to one protein sequence, saying how it got there.

    Four accepted shapes, in this order: FASTA text, a bare protein sequence, a nucleotide
    accession whose annotated CDS translations are fetched, and a protein accession. A
    nucleotide *sequence* is refused -- phmmer compares proteins, and silently translating
    one here would be this module inventing a reading frame.
    """
    text = query.strip()
    if not text:
        raise ValueError("query is empty; pass a protein sequence, a FASTA record, or an "
                         "accession")

    if text.startswith(">"):
        records = _parse_fasta(text)
        if not records:
            raise ValueError("query begins with '>' but carries no sequence under it")
        if not 0 <= cds_index < len(records):
            raise ValueError(f"cds_index {cds_index} is outside the {len(records)} record(s) "
                             f"in the supplied FASTA")
        defline, sequence = records[cds_index]
        kind = _classify(sequence)
        if kind == "nucleotide":
            raise ValueError(
                f"FASTA record {defline[:60]!r} is {len(sequence)} residues over the "
                f"nucleotide alphabet; phmmer compares proteins and this module will not "
                f"pick a reading frame for you")
        attempts.append({"route": "supplied_fasta", "outcome": "used"})
        return sequence, {"route": "supplied_fasta", "defline": defline,
                          "length": len(sequence), "cds_index": cds_index,
                          "records_available": len(records), "catalogue": []}

    kind = _classify(text)
    if kind == "nucleotide":
        raise ValueError(
            f"query is {len(''.join(text.split()))} residues over the nucleotide alphabet; "
            f"phmmer compares proteins and this module will not pick a reading frame for you")
    if kind == "protein":
        attempts.append({"route": "supplied_sequence", "outcome": "used"})
        return "".join(text.split()).upper(), {
            "route": "supplied_sequence", "defline": None,
            "length": len("".join(text.split())), "cds_index": None,
            "records_available": 1, "catalogue": []}

    accession = text
    routes = []
    if _UNIPROT_ACCESSION.match(accession):
        routes.append("uniprot")
    routes += ["ncbi_nuccore_cds", "ncbi_protein"]
    if "uniprot" not in routes:
        routes.append("uniprot")

    for route in routes:
        if route == "uniprot":
            url = f"{UNIPROT}/{urllib.parse.quote(accession)}.fasta"
        elif route == "ncbi_nuccore_cds":
            url = EFETCH + "?" + urllib.parse.urlencode(
                {"db": "nuccore", "id": accession, "rettype": "fasta_cds_aa",
                 "retmode": "text", "tool": "hpc-pilots"})
        else:
            url = EFETCH + "?" + urllib.parse.urlencode(
                {"db": "protein", "id": accession, "rettype": "fasta", "retmode": "text",
                 "tool": "hpc-pilots"})
        sources.append(url)
        try:
            body = _guard_silent_error(_request(url, calls, accept="text/plain"), url)
            records = _parse_fasta(body)
            if not records:
                raise Absent(f"{url} returned no FASTA record")
        except (Absent, RuntimeError) as error:
            attempts.append({"route": route, "outcome": "no_record", "detail": str(error)[:240]})
            continue

        catalogue = _cds_catalogue(records, accession) if route == "ncbi_nuccore_cds" else []
        if not 0 <= cds_index < len(records):
            raise ValueError(
                f"cds_index {cds_index} is outside the {len(records)} record(s) {accession} "
                f"resolved to via {route}; the catalogue of what is there is in "
                f"resolved_query.catalogue")
        defline, sequence = records[cds_index]
        if _classify(sequence) == "nucleotide":
            attempts.append({"route": route, "outcome": "not_protein"})
            continue
        attempts.append({"route": route, "outcome": "used"})
        fields = dict(_QUALIFIER.findall(defline))
        return sequence, {
            "route": route, "defline": defline, "length": len(sequence),
            "cds_index": cds_index, "records_available": len(records),
            "gene": fields.get("gene"), "protein": fields.get("protein"),
            "protein_accession": fields.get("protein_id"),
            "location": fields.get("location"), "catalogue": catalogue}

    raise Absent(
        f"{accession!r} resolved to no protein sequence on any of {routes}; what each route "
        f"returned is recorded. This is an absence of a query, not an absence of homologs.")


def _best_domain(hit):
    """The included domain with the best independent E-value, or None.

    `hit['domains']` is `None` unless `with_domains=true` was asked for, and a hit can report
    domains none of which are included, so both are real cases rather than defensive noise.
    """
    domains = hit.get("domains") or []
    scored = [d for d in domains if d.get("alignment_display")]
    if not scored:
        return None
    included = [d for d in scored if d.get("is_included")] or scored
    return min(included, key=lambda d: d.get("ievalue") if d.get("ievalue") is not None
               else float("inf"))


def _row(hit):
    """One hit to one row: accession, identity, E-value, organism, and nothing invented.

    `identity` arrives as `[fraction, positions]` on the alignment, so a hit whose domains
    were not returned gets `None` and not `0.0`: an unmeasured identity reported as zero is
    the same class of error as an unmeasured cost reported as zero, which this repo has
    already paid for once.
    """
    metadata = hit.get("metadata") or {}
    domain = _best_domain(hit)
    display = (domain or {}).get("alignment_display") or {}
    identity = display.get("identity") or [None, None]
    similarity = display.get("similarity") or [None, None]
    architecture = (metadata.get("architecture") or "").split()
    return {
        "accession": metadata.get("accession"),
        "uniprot_accession": metadata.get("uniprot_accession"),
        "identifier": metadata.get("identifier") or metadata.get("uniprot_identifier"),
        "description": metadata.get("description"),
        "species": metadata.get("species"),
        "taxonomy_id": metadata.get("taxonomy_id"),
        "kingdom": metadata.get("kingdom"),
        "phylum": metadata.get("phylum"),
        "evalue": hit.get("evalue"),
        "bitscore": hit.get("score"),
        "identity": identity[0],
        "identity_positions": identity[1],
        "similarity": similarity[0],
        "domains_reported": hit.get("ndom"),
        "domains_included": hit.get("nincluded"),
        "query_from": display.get("hmmfrom"),
        "query_to": display.get("hmmto"),
        "target_from": display.get("sqfrom"),
        "target_to": display.get("sqto"),
        "target_length": display.get("l"),
        "domain_evalue": (domain or {}).get("ievalue"),
        "pfam_accessions": architecture,
        "pfam_architecture": metadata.get("architecture"),
        "pdb_entries": metadata.get("structures") or [],
        "entry_url": metadata.get("external_link"),
        "included": bool(hit.get("is_included")),
    }


def search(query, database="swissprot", max_hits=50, min_identity=0.0, cds_index=0,
           max_wait_seconds=120.0, record=True):
    """Homologs of one protein, with identity, E-value and source organism, from EBI HMMER.

    `query` may be a protein sequence, a FASTA record, a nucleotide genome accession whose
    annotated CDS translations are fetched and indexed by `cds_index`, or a protein
    accession. The genome-accession route is what makes this callable from the one fact a
    campaign starts with: `efetch?rettype=fasta_cds_aa` returns the translations already in
    the annotation, so no reading frame is guessed here and nothing duplicates `intake`'s ORF
    caller. A genome with no annotated CDS is an explicit absence, not a silent zero.

    Returns homolog rows and the counts behind them. It does not return the query's identity:
    a 28%-identity hit is a remote relative's accession, `identity_warning` says which case
    this call is, and reading the hit list past that field is the conflation the repo warns
    about. `min_identity` filters rows and the count it removed is reported, so a filtered
    search cannot be mistaken for an empty one.
    """
    started = time.time()
    calls, sources, attempts = [0], [], []
    if database not in DATABASES:
        raise ValueError(f"database {database!r} is not one of {DATABASES}; 'pfam' is an HMM "
                         f"library for hmmscan and is not searchable by phmmer")
    if max_hits < 1:
        raise ValueError(f"max_hits must be at least 1, got {max_hits}")
    if not 0.0 <= min_identity <= 1.0:
        raise ValueError(f"min_identity is a fraction in [0, 1], got {min_identity}")
    if max_wait_seconds <= 0:
        raise ValueError(f"max_wait_seconds must be positive, got {max_wait_seconds}")

    catalogue_url = f"{HMMER}/search/databases"
    sources.append(catalogue_url)
    served = _json(catalogue_url, calls)
    served = {entry["id"]: entry for entry in served if isinstance(entry, dict)}
    drift = sorted(set(DATABASES) - set(served))
    if database not in served:
        raise RuntimeError(
            f"{database!r} is declared in this module but the service lists "
            f"{sorted(served)}; the database set has moved and the declaration is stale")
    serving = served[database]
    if serving.get("status") != "enabled":
        raise RuntimeError(f"{database!r} is listed with status {serving.get('status')!r}")

    sequence, resolved = _resolve(query, cds_index, calls, sources, attempts)

    submit_url = f"{HMMER}/search/phmmer"
    sources.append(submit_url)
    submitted = _json(submit_url, calls,
                      data={"database": database, "input": f">query\n{sequence}"})
    job = submitted.get("id")
    if not job:
        raise RuntimeError(f"{submit_url} returned no job id: {submitted!r}")

    result_url = (f"{HMMER}/result/{job}?"
                  + urllib.parse.urlencode({"with_domains": "true", "page_size": max_hits}))
    sources.append(result_url)
    polls, wait, status, payload = 0, 1.0, None, None
    while True:
        payload = _json(result_url, calls)
        polls += 1
        status = payload.get("status")
        if status == "SUCCESS":
            break
        if status in ("ERROR", "FAILURE"):
            raise RuntimeError(f"job {job} on {database} returned status {status!r} after "
                               f"{polls} poll(s)")
        if time.time() - started > max_wait_seconds:
            raise RuntimeError(
                f"job {job} on {database} was still {status!r} after {polls} poll(s) and "
                f"{time.time() - started:.1f}s, over the {max_wait_seconds}s ceiling; the "
                f"job id is retrievable at {HMMER}/result/{job}")
        time.sleep(min(wait, max(0.0, max_wait_seconds - (time.time() - started))))
        wait = min(wait * 1.5, 5.0)

    body = payload.get("result") or {}
    stats = body.get("stats") or {}
    hits = body.get("hits") or []
    rows = [_row(hit) for hit in hits]
    kept = [row for row in rows
            if row["identity"] is not None and row["identity"] >= min_identity] \
        if min_identity > 0.0 else list(rows)
    unmeasured = [row for row in rows if row["identity"] is None]
    dropped = len(rows) - len(kept)

    identities = [row["identity"] for row in kept if row["identity"] is not None]
    best = max(identities) if identities else None
    if not kept:
        identity_warning = ("No homolog was returned, so no identity was measured and no "
                            "identification is supported by this call.")
    elif best is None:
        identity_warning = ("Hits were returned but none carried an alignment, so no "
                            "identity was measured. Treat every accession here as "
                            "unidentified.")
    elif best <= CONFLATION_IDENTITY:
        identity_warning = (
            f"Best identity is {best:.3f}, at or below {CONFLATION_IDENTITY}. These are "
            f"remote relatives. Each accession is somebody else's protein and naming the "
            f"query after any of them is the identity conflation this repository warns "
            f"about.")
    else:
        identity_warning = (
            f"Best identity is {best:.3f}. Even at this level the accession belongs to the "
            f"database entry and not to the query; identity is evidence about similarity, "
            f"not a transfer of identity.")

    absence = None
    if not hits:
        absence = {
            "what": "no homolog at or above the service reporting threshold",
            "query_length": len(sequence),
            "database": database,
            "database_entries_searched": stats.get("nseqs"),
            "reading": "A real zero, recorded and charged. It is evidence that this query "
                       "has no detectable relative in this database, not evidence that the "
                       "search failed.",
        }
    elif not kept:
        absence = {
            "what": f"no homolog survived min_identity={min_identity}",
            "hits_before_filter": len(rows),
            "database": database,
            "reading": "The search found relatives and the caller's identity floor removed "
                       "all of them. Not the same zero as an unproductive search.",
        }

    species = sorted({row["species"] for row in kept if row["species"]})
    families = sorted({accession for row in kept for accession in row["pfam_accessions"]})
    payload_out = {
        "query": {"query": query, "database": database, "max_hits": max_hits,
                  "min_identity": min_identity, "cds_index": cds_index,
                  "max_wait_seconds": max_wait_seconds},
        "resolved_query": resolved,
        "resolution_attempts": attempts,
        "database_id": database,
        "database_name": serving.get("name"),
        "database_version": serving.get("version"),
        "database_release_date": serving.get("release_date"),
        "declared_databases_not_served": drift,
        "homologs": kept,
        "homolog_count": len(kept),
        "hits_returned": len(rows),
        "hits_dropped_by_min_identity": dropped,
        "hits_without_measured_identity": len(unmeasured),
        "hits_reported_by_service": stats.get("nreported"),
        "hits_above_inclusion_threshold": stats.get("nincluded"),
        "database_entries_searched": stats.get("nseqs"),
        "distinct_species": species,
        "distinct_pfam_accessions": families,
        "best_identity": best,
        "identity_warning": identity_warning,
        "absence": absence,
        "job": {"id": job, "algorithm": stats.get("algo") or "phmmer", "status": status,
                "polls": polls, "engine_elapsed_seconds": stats.get("elapsed"),
                "result_url": f"{HMMER}/result/{job}"},
        "unverified": [
            "Species, Pfam architecture and description are UniProt/Pfam annotations on the "
            "hit entry. They are curated assertions about that entry, not measurements on "
            "this pathogen, and this module did not check any of them.",
            "A Pfam architecture is an annotation transferred from relatives. Using it as "
            "protein_family is a hypothesis about the query, not an observation of it.",
        ],
        "limitations": [
            "A homolog is somebody else's accession. This call establishes similarity "
            "between the query and database entries and establishes no identity for the "
            "query itself.",
            "An E-value bounds the chance of a similarity this good arising at random in a "
            "database this size. It says nothing about function, essentiality, expression, "
            "or whether any surface is bindable.",
            "Identity is the best included domain's alignment identity over the aligned "
            "region only, not over the full length of either sequence. A high identity on a "
            "short domain is not a high identity overall.",
            "Conservation here is conservation across curated database entries. SwissProt "
            "and UniProt are not samples of circulating isolates, so this cannot report "
            "allele frequency, selective pressure, or whether an epitope is conserved in "
            "the field.",
            "database_release_date is the release date of the searched database, not the "
            "date any individual entry became public. Entries were available earlier, so "
            "using it as available_from is late and therefore conservative: it will treat "
            "genuinely older evidence as unknowable rather than the reverse.",
            "phmmer is a profile-free single-sequence HMM search. It is more sensitive than "
            "BLAST at low identity and it is not a structural alignment; nothing here speaks "
            "to fold, conformation or solvent exposure.",
            "No structure, assay, essentiality screen, deep mutational scan or prior "
            "macrocycle campaign is retrieved or implied by this call.",
            "The CDS route reads the submitter's annotation. An unannotated or "
            "mis-annotated genome yields no CDS or a wrong one, and this module does not "
            "second-guess the annotation or call ORFs of its own.",
            "Only the first page of hits is retrieved, bounded by max_hits. A query with "
            "more relatives than that returns a truncated and best-scoring view, which "
            "page_count in the service response reveals and this payload does not paginate.",
        ],
        "sources": list(dict.fromkeys(sources)),
        "service_terms": [
            "EMBL-EBI HMMER web service: free, no key, no registration. "
            "https://www.ebi.ac.uk/Tools/hmmer/",
            "NCBI E-utilities: free, no key; keyless clients are held to 3 requests/second. "
            "https://www.ncbi.nlm.nih.gov/books/NBK25497/",
            "UniProt REST: free, no key, CC-BY 4.0. https://www.uniprot.org/help/about",
        ],
        "network_calls": calls[0],
        "wall_seconds": round(time.time() - started, 3),
        "provenance": provenance(),
    }
    # `common.VOLATILE_KEYS` strips timestamps but knows nothing about this payload's
    # per-call fields, and a digest that moves because a poll took 1.3s instead of 1.1s is
    # not a content identity. The job id, the poll count, the wall clock and the request
    # count are facts about this invocation; the hit list is the content.
    payload_out["content_sha256"] = content_digest(
        {key: value for key, value in payload_out.items()
         if key not in ("job", "wall_seconds", "network_calls", "provenance")})
    if record:
        label = (resolved.get("protein_accession") or resolved.get("gene")
                 or query.strip()[:24] or "query")
        slug = re.sub(r"[^A-Za-z0-9._-]+", "_", label)
        write_json(ROOT / "runs" / "homologs" / f"{slug}.{database}.json", payload_out)
    return payload_out
