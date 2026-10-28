"""Dated literature retrieval, which this repository did not have at all.

Fifty-four of the fifty-nine evidence claims in `targets.py` are hand-authored strings
citing this project's own unreviewed write-up. Nothing anywhere retrieved a publication. So
the first question is not which endpoint is nicest but which one can be asked a *falsifiable*
question, and for a project that replays historical decisions that means one thing: a
publication-date ceiling. Without it a hindsight audit reads today's literature and reports
it as what was knowable in 2015, which is leakage dressed as foresight.

Five candidates were surveyed live on 2026-10-27 against four criteria -- keyless access, a
publication-date range filter, abstract availability, and whether a query can be built from a
protein *family* rather than only a target name. Measured, not recalled:

    Europe PMC       keyless 200. `FIRST_PDATE:[a TO b]`, 18041 -> 1173. Abstracts in
                     24 of 25 hits. MED + PMC + preprints + patents, fielded query
                     language.
    OpenAlex         keyless 200. `to_publication_date`. Abstracts in 21 of 25, as an
                     inverted index only. Not biomedical-specific.
    NCBI E-utilities keyless 200. `datetype=pdat&mindate&maxdate`. No abstract without a
                     second efetch. 25 hits where Europe PMC found 1173.
    Crossref         keyless 200. `until-pub-date`. Abstracts in 6 of 20. Bibliographic
                     metadata with no biomedical index.
    Semantic Scholar **429 on the first unauthenticated call.** Disqualified: a key is
                     the only reliable route, and a key needs a signup.

Europe PMC wins on all four and is not close. The decisive one is the fourth: a fielded query
language means `(TITLE:"RNA-dependent RNA polymerase" OR ABSTRACT:"...")` is expressible, so
the family route works when the target itself is unknown -- which is the whole premise of a
prototype-pathogen argument. The NCBI row is the one worth staring at: 25 against 1173 on the
same query. PubMed indexes neither preprints nor full text, and at onset, when a pathogen is
days old, the preprints are the literature.

**The defect this module exists to defend against.** Europe PMC silently discards a date
clause it cannot parse. Measured:

    nsp12                                              -> 45481
    nsp12 AND FIRST_PDATE:[1900-01-01 TO 2015-12-31]   -> 12995
    nsp12 AND FIRST_PDATE:[1900-01-01 TO notadate]     -> 45481, HTTP 200, no warning

The third response is the first response wearing the second one's query string: the server
echoes `notadate` back in `request.queryString` and returns the unfiltered corpus. A tool
that trusted HTTP 200 would hand an auditor the 2026 literature labelled 2015 and nothing in
the artifact would show it. An unbalanced bracket fails the other way: a query truncated
after `FIRST_PDATE:[1900-01-01 TO` returns `hitCount: 0` at HTTP 200, a false absence
indistinguishable from a real one. And an invalid `pageSize` returns `{"errCode": 404}`
*inside* a 200.

Hence three guards, and they are the reason this file is longer than a wrapper needs to be.
Dates are validated against a strict ISO pattern here, before dispatch, so `notadate` can
never be sent. Subject strings are balance-checked, so the false-absence path is unreachable.
And every dated call is paired with a control call that omits the clause, with both counts
recorded in `date_filter_verified`: the ceiling is then a *measurement* in the artifact rather
than an assumption behind it. When the two counts are equal that is recorded as
non-discriminating and not as an error -- a query whose every hit predates the ceiling
legitimately gives equality -- but `filtered > unfiltered` is impossible and raises.

**What a count is.** `hit_count` is evidence of attention and of nothing else. Four thousand
papers on a family says the field is crowded, which bears on whether a method can be
developed against it; it says nothing about whether this protein is essential, conserved,
exposed or expressible. The per-axis counts are worse: the term sets in `AXIS_TERMS` are an
authored heuristic that no one has validated against a labelled corpus, so every axis row
carries `verified: False` and the composed query string that produced it. A number whose
query is not recorded cannot be checked, and an unchecked number from a heuristic is the
thing `targets.py` is already full of.

What does carry weight is a hit: an identifier, a title and a first-publication date. That is
a dated, citable claim, which is why `hits` is a mapping of separately dated rows and the
counts are not.

One more thing the survey turned up, and it is why this tool is callable at onset. A campaign
starts holding one fact, a nucleotide accession, and the selector's `PROPOSED` entry assumed
literature needed a target name or a family first. Europe PMC indexes accession numbers
text-mined out of article bodies, so the one fact a campaign starts with is already a query:
`ACCESSION_ID:"NC_045512"` returns 1577 records, `"NC_002549"` returns 67. That makes this
the only registered tool that can be called from the starting state and return dated
evidence. The versioned and bare forms are different tokens with different counts -- 3354
against 1577 -- so the accession route ORs both and records the string it sent.
"""
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date

from .common import ROOT, provenance, write_json

SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
PROFILE = "https://www.ebi.ac.uk/europepmc/webservices/rest/profile"

# Sent on every request. The structural tool reaches EMBL-EBI thirty-six times a call under
# the default `Python-urllib/3.12`, which gives an operator throttling us no way to find out
# who we are. EBI asks for a contact string; this is the contact string.
AGENT = "hpc-pilots/1.0 (research prototype; contact via repository)"

# Seconds between successive requests. Europe PMC publishes no hard limit for the article
# API, so this is politeness and not compliance: at seven requests a call it costs under two
# seconds and removes any question of whether we are the burst in somebody's log.
PAUSE = 0.34

# Per-request ceiling. Low on purpose: `worst_case_seconds` in the spec is this number times
# the request count, and a 180-second timeout would declare a worst case of twenty-five
# minutes that the wall gate would have to refuse on every call.
TIMEOUT = 30

# Open ends of a date range. Europe PMC needs both bounds of `FIRST_PDATE:[a TO b]`, and
# these two were checked to reproduce the unfiltered count exactly: `nsp12` alone and
# `nsp12 AND FIRST_PDATE:[1800-01-01 TO 3000-12-31]` both return 45481.
OPEN_PAST = "1800-01-01"
OPEN_FUTURE = "3000-12-31"

# A date as Europe PMC will accept it and as `datetime.date` will parse it. Checked with a
# regex and not only with `date.fromisoformat`, because from 3.11 that function also accepts
# `20150101` and `2015-W01-1`, neither of which Europe PMC parses -- and a form it cannot
# parse is the silent-drop path this module exists to close.
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# An INSDC-style nucleotide or protein accession, optionally versioned.
ACCESSION = re.compile(r"^[A-Za-z]{1,6}_?\d{4,}(\.\d+)?$")

SCOPES = ("title_abstract", "full_text")
SUBJECT_KINDS = ("term", "accession", "query")

# Per-axis term sets. **An authored heuristic, validated against nothing.** They are here
# rather than in a config because they are a claim this module makes and should be diffable
# with it, and every row they produce is stamped `verified: False`.
#
# Only the five axes `selector.PROPOSED` grants `literature.search` appear. `accessibility`
# and `reagent_tractability` are constructible -- "solvent exposed", "soluble expression" --
# and are deliberately absent: `selector.py` says an overclaimed axis "would make an
# unresolved gate read as resolved, which is the one error that changes a recommendation
# rather than merely a ranking", and `reagent_tractability` is the hard gate. Widening this
# mapping is the orchestrator's decision to take in the open, not this file's to take quietly.
AXIS_TERMS = {
    "essentiality": ("essential", "essentiality", "knockout", "knock-out",
                     '"loss of function"', '"gene deletion"', "CRISPR",
                     '"replication deficient"'),
    "conservation": ("conserved", "conservation", '"sequence conservation"',
                     '"positive selection"', '"selective pressure"', "phylogenetic"),
    "assay_availability": ("assay", '"enzyme inhibition"', '"surface plasmon resonance"',
                           "FRET", '"fluorescence polarization"', "ELISA", '"IC50"',
                           '"biolayer interferometry"'),
    "public_labels": ('"deep mutational scanning"', '"mutational scanning"',
                      '"saturation mutagenesis"', '"binding landscape"',
                      '"deep sequencing selection"'),
    "precedent": ("macrocycle", "macrocyclic", '"cyclic peptide"', '"peptide inhibitor"',
                  '"mRNA display"', '"phage display"', "RaPID", '"stapled peptide"'),
}


def _request(url):
    """One GET, with a contact header, returning parsed JSON or naming why it could not.

    Everything goes through `urllib.request.urlopen` because that is the only transport
    `chief._install_request_counter` wraps. A `requests` call here would make this tool
    certify a network cost of zero while reaching EMBL-EBI seven times, which is the defect
    the counter's own docstring warns is indistinguishable from a tool that touches nothing.
    """
    headers = {"User-Agent": AGENT, "Accept": "application/json"}
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                    timeout=TIMEOUT) as handle:
            body = handle.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Europe PMC returned HTTP {error.code} for {url!r}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Europe PMC was unreachable for {url!r}: {error.reason}") from error
    try:
        payload = json.loads(body)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Europe PMC returned a non-JSON body for {url!r}: "
                           f"{body[:200]!r}") from error
    # An invalid parameter comes back as `{"errCode": 404, "errMsg": ...}` with the HTTP
    # status still 200. Reading that as an empty result would record a fabricated absence.
    if "errCode" in payload:
        raise RuntimeError(f"Europe PMC refused {url!r} inside a 200 response: "
                           f"errCode={payload['errCode']} {payload.get('errMsg', '')!r}")
    time.sleep(PAUSE)
    return payload


def _search_url(query, result_type="lite", page_size=1):
    return SEARCH + "?" + urllib.parse.urlencode(
        {"query": query, "format": "json", "resultType": result_type,
         "pageSize": page_size})


def _profile_url(query):
    return PROFILE + "?" + urllib.parse.urlencode({"query": query, "format": "json"})


def _hit_count(payload, url):
    """`hitCount`, refusing to default it to zero.

    A missing key is a changed response shape, and the whole point of this module is that an
    unexplained zero and a real zero must not look alike.
    """
    if "hitCount" not in payload:
        raise RuntimeError(f"Europe PMC response for {url!r} carries no hitCount; the "
                           f"response shape has changed and a count cannot be read from it")
    return int(payload["hitCount"])


def _iso(value, label):
    """A date Europe PMC will parse, or a refusal naming the field.

    This is guard number one. `FIRST_PDATE:[1900-01-01 TO notadate]` is accepted by the
    server, which then returns the whole unfiltered corpus at HTTP 200, so the only place a
    malformed bound can be caught is before it is sent.
    """
    if not isinstance(value, str) or not ISO_DATE.match(value):
        raise ValueError(
            f"{label}={value!r} is not an ISO yyyy-mm-dd date. Europe PMC accepts a date "
            f"clause it cannot parse and silently returns the unfiltered corpus, so a "
            f"malformed bound would produce a date-limited answer that is not date-limited.")
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{label}={value!r} is not a real calendar date: {error}") from error
    return value


def _balanced(text, label):
    """Refuse an unbalanced subject string rather than let it return a false zero.

    Guard number two. `FIRST_PDATE:[1900-01-01 TO` returns `hitCount: 0` at HTTP 200, which
    is indistinguishable in an artifact from a target nobody has published on -- and the
    second of those is a finding this project acts on.
    """
    if text.count('"') % 2:
        raise ValueError(f"{label}={text!r} has an odd number of double quotes. Europe PMC "
                         f"answers a malformed query with hitCount 0 at HTTP 200, which "
                         f"would be recorded as a real absence.")
    for opener, closer in (("[", "]"), ("(", ")")):
        if text.count(opener) != text.count(closer):
            raise ValueError(f"{label}={text!r} has unbalanced {opener}{closer}; Europe PMC "
                             f"would answer it with a false hitCount of 0 at HTTP 200.")
    return text


def _subject_clause(subject, subject_kind, scope):
    """The query fragment naming what is being asked about, and nothing about dates."""
    if not isinstance(subject, str) or not subject.strip():
        raise ValueError("subject must be a non-empty string: a target name, a protein "
                         "family, an accession, or a Europe PMC query.")
    subject = subject.strip()
    if subject_kind not in SUBJECT_KINDS:
        raise ValueError(f"subject_kind={subject_kind!r} is not one of {SUBJECT_KINDS}")
    if scope not in SCOPES:
        raise ValueError(f"scope={scope!r} is not one of {SCOPES}")

    if subject_kind == "accession":
        if not ACCESSION.match(subject):
            raise ValueError(
                f"subject={subject!r} does not look like an INSDC accession, and the "
                f"accession route builds ACCESSION_ID clauses that would silently match "
                f"nothing. Pass subject_kind='term' for a name or a family.")
        # The bare and versioned forms are distinct tokens in this index with different
        # counts -- NC_045512 gives 1577, NC_045512.2 gives 3354 -- so neither form is the
        # canonical key and asking for one alone silently discards the other's records.
        forms = [subject]
        bare = subject.split(".")[0]
        if bare != subject:
            forms.append(bare)
        return "(" + " OR ".join(f'ACCESSION_ID:"{form}"' for form in forms) + ")"

    if subject_kind == "query":
        _balanced(subject, "subject")
        if "FIRST_PDATE" in subject.upper():
            raise ValueError(
                "subject carries its own FIRST_PDATE clause. This tool composes and then "
                "verifies the date clause, and a second one it did not write would make "
                "date_filter_verified a statement about the wrong query.")
        return f"({subject})"

    _balanced(subject, "subject")
    if ":" in subject:
        raise ValueError(f"subject={subject!r} contains a colon, which Europe PMC reads as a "
                         f"field qualifier. Pass subject_kind='query' to send a fielded "
                         f"query deliberately.")
    phrase = f'"{subject}"'
    if scope == "title_abstract":
        # Scoping the subject to title or abstract is what makes a count a count of papers
        # *about* the subject: measured, "RNA-dependent RNA polymerase" matches 47941 records
        # in full text and 8244 in title or abstract, and the two numbers answer different
        # questions. The axis terms stay unscoped, because an assay or a mutational-scanning
        # dataset is routinely named only in the methods.
        return f"(TITLE:{phrase} OR ABSTRACT:{phrase})"
    return phrase


def _date_clause(date_from, date_to):
    """`FIRST_PDATE:[a TO b]` with validated bounds, or empty when neither was asked for."""
    if date_from is None and date_to is None:
        return ""
    low = OPEN_PAST if date_from is None else _iso(date_from, "date_from")
    high = OPEN_FUTURE if date_to is None else _iso(date_to, "date_to")
    if date.fromisoformat(low) > date.fromisoformat(high):
        raise ValueError(f"date_from={low} is after date_to={high}, which no record can "
                         f"satisfy; an empty result from an impossible range would be "
                         f"recorded as an absence of literature.")
    return f"FIRST_PDATE:[{low} TO {high}]"


def _compose(subject_clause, date_clause):
    return f"{subject_clause} AND {date_clause}" if date_clause else subject_clause


def _row(result, rank):
    """One hit as a separately dated, citable row."""
    abstract = result.get("abstractText") or ""
    source = result.get("source") or "UNKNOWN"
    identifier = result.get("id") or ""
    journal = result.get("journalInfo") or {}
    return {
        "rank": rank,
        "source": source,
        "id": identifier,
        "pmid": result.get("pmid"),
        "pmcid": result.get("pmcid"),
        "doi": result.get("doi"),
        "title": (result.get("title") or "").strip(),
        "author_string": result.get("authorString"),
        "journal": (result.get("journalTitle")
                    or (journal.get("journal") or {}).get("title")),
        # The dossier's `available_from`. A paper published in 2015 is evidence from 2015
        # however recently it was fetched, so this is the publication date and never the
        # time of the call.
        "first_publication_date": result.get("firstPublicationDate"),
        "pub_year": result.get("pubYear"),
        # Not the same question as the one above, and worth carrying: a record retro-indexed
        # years after publication was not discoverable in between, so a date-limited query is
        # a bound on publication and only an approximation of what was findable.
        "first_index_date": result.get("firstIndexDate"),
        "is_preprint": source == "PPR",
        "is_open_access": result.get("isOpenAccess") == "Y",
        "cited_by_count": result.get("citedByCount"),
        "has_abstract": bool(abstract),
        "abstract_chars": len(abstract),
        "abstract": abstract[:1500],
        "abstract_truncated": len(abstract) > 1500,
        "url": (f"https://europepmc.org/article/{source}/{identifier}"
                if identifier else None),
    }


def _axis_counts(subject_clause, date_clause, sources_seen):
    """Per-axis hit counts, each with the query that produced it and `verified: False`."""
    rows = {}
    for axis, terms in AXIS_TERMS.items():
        clause = "(" + " OR ".join(terms) + ")"
        query = f"{_compose(subject_clause, date_clause)} AND {clause}"
        url = _search_url(query)
        sources_seen.append(url)
        rows[axis] = {
            "hit_count": _hit_count(_request(url), url),
            "query": query,
            # The term set is an authored heuristic with no validation against a labelled
            # corpus, and a count from an unvalidated query is exactly the kind of claim
            # `targets.py` marks unverified. The query string is recorded so the number can
            # be checked by hand, which is the only thing that makes it worth keeping.
            "verified": False,
            "means": f"Records matching the subject and at least one {axis} term. Evidence "
                     f"that the axis has been written about, not that it is satisfied.",
        }
    return rows


def search(subject, subject_kind="term", scope="title_abstract", date_from=None,
           date_to=None, top=10, per_axis=True, record=True):
    """Date-limited Europe PMC retrieval for a target name, protein family or accession.

    Returns counts, a per-source breakdown, the top hits as separately dated rows, optional
    per-axis counts, and -- when a date bound was given -- a recorded verification that the
    ceiling actually bit. Writes nothing when `record=False`, which is how a dispatcher calls
    it, because the episode writes the record.

    What it does not establish: a hit count is evidence of attention. Nothing here measures
    essentiality, conservation, exposure, expressibility or affinity, and a paper about a
    family is not a measurement on this pathogen.
    """
    if not isinstance(top, int) or isinstance(top, bool) or not 1 <= top <= 100:
        raise ValueError(f"top={top!r} must be an integer in 1..100; Europe PMC refuses a "
                         f"pageSize outside 1..1000 inside a 200 response.")

    subject_clause = _subject_clause(subject, subject_kind, scope)
    date_clause = _date_clause(date_from, date_to)
    query = _compose(subject_clause, date_clause)
    sources_seen = []

    # The dated query, with full records so abstract availability is read and not assumed.
    hits_url = _search_url(query, result_type="core", page_size=top)
    sources_seen.append(hits_url)
    payload = _request(hits_url)
    hit_count = _hit_count(payload, hits_url)
    echoed = (payload.get("request") or {}).get("queryString", "")
    results = (payload.get("resultList") or {}).get("result") or []
    hits = {}
    for rank, result in enumerate(results, start=1):
        row = _row(result, rank)
        key = f"{row['source']}:{row['id']}" if row["id"] else f"rank:{rank}"
        hits[key] = row

    # Guard number three. The echo is necessary and not sufficient: the server echoed
    # `notadate` back and still returned the unfiltered corpus, so the clause being present
    # in the echo proves only that it was received. The control call is what proves it was
    # applied, and both numbers go in the artifact so the claim is checkable there.
    verification = None
    if date_clause:
        if "FIRST_PDATE" not in echoed.upper():
            raise RuntimeError(
                f"Europe PMC echoed {echoed!r}, which carries no FIRST_PDATE clause although "
                f"one was sent. The result cannot be treated as date-limited.")
        control_url = _search_url(subject_clause)
        sources_seen.append(control_url)
        unfiltered = _hit_count(_request(control_url), control_url)
        if hit_count > unfiltered:
            raise RuntimeError(
                f"The date-limited query returned {hit_count} records and the same query "
                f"without the date clause returned {unfiltered}. A ceiling cannot add "
                f"records, so the clause was not applied as a filter.")
        verification = {
            "clause": date_clause,
            "echoed_query": echoed,
            "hit_count_with_clause": hit_count,
            "hit_count_without_clause": unfiltered,
            "removed": unfiltered - hit_count,
            # Equality is not an error: a subject whose every record predates the ceiling
            # legitimately gives it. It is reported so a reader can tell a filter that bit
            # from one that had nothing to remove.
            "discriminating": hit_count < unfiltered,
            "control_query": subject_clause,
        }

    # Per-source and per-type counts for the same query, in one call: MED is PubMed, PPR is
    # preprints, PAT is patents. The preprint count is the one to read at onset, and the one
    # a PubMed-only backend cannot report at all.
    profile_url = _profile_url(query)
    sources_seen.append(profile_url)
    profile = (_request(profile_url).get("profileList") or {})
    by_source = {entry["name"]: entry["count"] for entry in profile.get("source") or []}
    by_type = {entry["name"]: entry["count"] for entry in profile.get("pubType") or []}

    axes = _axis_counts(subject_clause, date_clause, sources_seen) if per_axis else {}

    payload = {
        "subject": subject,
        "subject_kind": subject_kind,
        "scope": scope,
        "query": query,
        "date_from": date_from,
        "date_to": date_to,
        "hit_count": hit_count,
        "returned": len(hits),
        "hits": hits,
        "hits_with_abstract": sum(1 for row in hits.values() if row["has_abstract"]),
        "preprints_returned": sum(1 for row in hits.values() if row["is_preprint"]),
        "by_source": by_source,
        "by_publication_type": by_type,
        "axes": axes,
        "date_filter_verified": verification,
        # A zero is a result and is recorded as one. Both guards above exist so that a zero
        # here cannot have come from a malformed query, which is what makes it reportable.
        "absence": {
            "subject_zero": hit_count == 0,
            "axes_zero": tuple(sorted(name for name, row in axes.items()
                                      if row["hit_count"] == 0)),
            "means": "A zero is a recorded absence of indexed literature for this query at "
                     "this date bound, not a failed lookup: the date bounds and the subject "
                     "string were validated before dispatch and the response was checked "
                     "for an in-body error code.",
        },
        "limitations": [
            "A hit count is evidence of attention, not of a fact. It does not say the "
            "protein is essential, conserved, exposed, expressible or bindable.",
            "The per-axis term sets in AXIS_TERMS are an authored heuristic validated "
            "against nothing. Every axis row carries verified=False and the query string "
            "that produced it.",
            "A family-level query returns literature about relatives. An annotation or a "
            "finding transferred from a relative is not a measurement on this pathogen.",
            "FIRST_PDATE bounds first publication, which is an approximation of what was "
            "knowable: a record retro-indexed later was not discoverable in between, and "
            "first_index_date is carried per hit so the gap is visible.",
            "Europe PMC indexes full text for its open-access subset only, so an unscoped "
            "term is searched over more fields for some records than for others and counts "
            "are not comparable across scopes.",
            "The accession route matches accession numbers text-mined from article bodies. "
            "A match means the accession was mentioned, not that the paper is about this "
            "pathogen, and the bare and versioned forms are different tokens.",
            "Nothing here reads a paper. Relevance is unjudged, retracted records are not "
            "filtered, and the returned abstracts are truncated at 1500 characters.",
            "Only the five axes selector.PROPOSED grants this tool are queried. "
            "accessibility and the reagent_tractability hard gate are deliberately absent.",
        ],
        "sources": sources_seen,
        "citation": "Europe PMC RESTful Web Service, EMBL-EBI. Each hit is cited by its "
                    "source and identifier (PMID, PMCID or DOI) together with its "
                    "first_publication_date.",
        "provenance": provenance(),
    }
    if record:
        stamp = date_to or "open"
        slug = re.sub(r"[^A-Za-z0-9]+", "_", f"{subject}_{stamp}").strip("_").lower()
        write_json(ROOT / "data/processed/literature" / f"{slug}.json", payload)
    return payload
