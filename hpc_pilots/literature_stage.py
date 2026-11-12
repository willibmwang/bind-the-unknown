"""Dated, citable literature for each shortlisted target and each domain that was folded.

`chain.evidence_rows` already calls `literature.search` once per target, and then throws away
the only part of the return that carries weight. `_evidence_summary` reduces any tool return
to ten headline scalars -- deliberately, so a chain record does not become a copy of five
external databases -- and for this tool that leaves `hit_count`, which the module it came from
says is "evidence of attention and of nothing else". The identifiers, titles, journals, DOIs
and publication dates were fetched over the network, counted, and dropped. This stage is the
same retrieval with the rows kept.

It is deterministic and holds no model. The fifth ask was phrased as a question about
connecting a subagent; the answer taken was that a literature *review* is a judgement and a
literature *retrieval* is a measurement, and only the second belongs in a replayable pipeline.
Nothing here reads a paper, ranks a paper, or decides that a paper supports a claim.

**The hard part is composing the query, and a bare domain description is a terrible one.**
Measured live against Europe PMC on 12 November 2026, with the composed-query form this module
sends shown beside it:

    "Y1"                                       7939   adrenocortical cells, B. subtilis
        the nsp3 region, asked bare                   Pol Y1, neuropeptide Y receptors
    "Y1" AND (nsp3 OR papain-like protease)      45   Ubl2 flexibility in the SARS-CoV-2
        the same region, anchored                     papain-like protease; nsp3 domains
                                                      by AlphaFold2
    "Dimerization"                            31632   Ebola VP30, myeloperoxidase,
        the nucleoprotein region, asked bare          receptor-engineering hydrogels
    "Dimerization" AND nucleoprotein AND
        this genome's accession                   5   SARS-CoV-2 nucleoprotein in complex
                                                      with nsp3a; its NTD assignments
    "C2HC"                                       82   spermatogenesis, papain-induced
        the nsp2 zinc finger, asked bare              airway inflammation
    "C2HC" AND nsp2                               2   Structure and Function of the
                                                      N-Terminal Zinc Finger Domain of
                                                      SARS-CoV-2 NSP2

So the description is anchored to the protein it belongs to, as a fielded Europe PMC query
sent through `subject_kind="query"` -- the fielded query language being the capability
`literature.py`'s own five-endpoint survey chose Europe PMC for. Three things about that
composition were measured rather than assumed, and each replaced a form that was tried first:

1. **The domain phrase is searched unscoped, over full text.** These are database-register
   names -- Pfam, MEROPS, UniProt's own -- and they appear in methods, tables and figure
   captions rather than in abstracts. Scoping `"Peptidase C16"` to title or abstract returns 0
   and would have recorded a false absence for the protease nsp3 is named after; unscoped and
   anchored to nsp3 it returns 3. The same holds for `BetaCoV S1-CTD` and `4Ecto`.
2. **The anchor phrases are scoped to title or abstract**, so the paper has to be *about* the
   protein rather than merely to mention it. This is the distinction `literature._subject_clause`
   already measures for a plain term: "RNA-dependent RNA polymerase" matches 47941 records in
   full text and 8244 in title or abstract.
3. **The target's name is split into anchor phrases rather than passed through.** The name in
   the record is a UniProt designation and often nobody's search term. Measured:
   `"3C-like proteinase nsp5"` as one phrase returns 1 record; `(nsp5 OR "3C-like proteinase")`
   returns 775. Worse, a chain identified from a relative's entry carries that entry's
   identifier in its name -- `NCAP_BCRP3 Nucleoprotein`, `VME1_BC279 Membrane protein` -- and
   both of those as written return **0**, a false absence caused entirely by a token that is a
   database key and not a word anyone publishes. But splitting a name has its own failure, and
   it is the largest number in this module: the remainder of `ORF9b protein` is `protein`, and
   OR'ing that in takes a 119-record query to **3,726,147** records, which is the entire index.
   A remainder that names a class rather than a protein is therefore refused -- see
   `GENERIC_NAMES`, which carries the count behind each entry.
4. **The domain phrase is never shortened, even when it returns nothing.** nsp13's
   `(+)RNA virus helicase ATP-binding` returns 0 anchored, and the tempting fix is to drop the
   qualifier. It was measured and it is wrong: the brackets are not the cause -- the same
   phrase without them also returns 0 -- and shortening to `helicase ATP-binding` returns 99
   while shortening the sibling region to `helicase C-terminal` returns 134 whose top hits are
   chromodomain helicases in orofacial cleft and DEAD-box helicases in the rubber tree. A
   recorded zero under a query a reader can see beats 134 records about another kingdom, so
   the annotation's phrase is sent as the annotation wrote it.

**What it actually retrieved.** Run live on 12 November 2026 over the six top-ranked chains of
the held-out case: 20 composed queries, 30 network requests, 78 dated citable rows, no undated
ones, and 4 measured zeros. All four zeros are database-register domain names nobody publishes
under -- `BetaCoV S1-NTD`, `BetaCoV S1-CTD`, `RdRp Palm C-ter`, `CoV Nsp3 Y`. The sixteen that
retrieved are on target: `Heptad repeat 2` leads with a recombinant HR212 construct aimed at
that very repeat, `Nsp12 Interface` with interface-mimicking peptides against the nsp12-nsp8
complex, `Y1` with the AlphaFold2 nsp3 domain determination, and `Peptidase C16` with the three
papers that use the MEROPS name at all.

**The anchor of last resort, and why it is this genome's accession.** A one-word protein noun
does not identify a protein. `"Dimerization" AND nucleoprotein` returns 400 records whose top
three are Hazara virus, a DNA-superhelicity sensor and influenza A -- obviously wrong
literature for a coronavirus nucleoprotein. So where a chain's name yields no systematic
designation (`nsp12`, `ORF9b`) the clause `(ACCESSION_ID:"<genome>" OR ACCESSION_ID:"<bare>")`
is added, which takes that query from 400 records to 5 correct ones. The accession is read from
the case, not written here, and the route is the one `literature.py` documents at length: Europe
PMC indexes accession numbers text-mined out of article bodies, the bare and versioned forms are
different tokens, so both are OR'd. Measured 12 November 2026, the clause alone holds 4725
records for this genome.

It is added only to the chains that need it, and the reason is the premise of this project. A
genome three days old has no literature citing its accession, so a stage that anchored every
query on the accession would return measured zeros at exactly the onset it exists to serve.
Anchoring on `nsp12` does not have that failure, because the designation is nomenclature rather
than a citation. Which route each row took is recorded in `anchor_route`.

**Which domains are asked about.** The ones stage 3 actually folded, read off the fold record,
and not every annotated region. A domain with coordinates is a domain a binder could be designed
against this week and its literature is actionable; a region refused as a fold unit is one a
reader cannot act on yet, and nsp3 offers twenty-one regions whose literature would spend the
whole budget on one chain. Where a chain has no folded domain the chain-level query still runs
and the row says there were no domain queries and why.

**No date ceiling, stated rather than left to be noticed.** `date_to` defaults to None, so this
stage retrieves today's literature, which is the right default: the ask was for literature that
informs a binder design now, not a hindsight replay. But `literature.py` exists largely to stop
undated retrieval being passed off as dated, so `date_ceiling_applied: false` is written
explicitly on the stage and on the record, with the sentence that follows from it -- these hits
include work published after the onset this chain replays, and they are not a claim about what
was knowable then. Where `date_to` is given it is passed through and `date_filter_verified`, the
paired control call that is the whole point of that module's third guard, is kept on the row.

This stage changes no rank. The ranking is final at the end of stage 2, the same as stages 3, 4
and 6, and a hit count may not reorder a shortlist it is not a term in.

    python -m hpc_pilots.literature_stage [--top N] [--date-to YYYY-MM-DD]

Writes `runs/literature_stage.json`.
"""

import argparse
import json
import re
import time
from pathlib import Path

from .common import ROOT, content_digest, provenance, write_json

RECORD = ROOT / "runs" / "literature_stage.json"
LANDING = ROOT / "runs" / "onset_chain.json"
FOLDS = ROOT / "runs" / "domain_folds.json"

# Targets this stage asks about. More than `evidence_stage`'s 3 and fewer than all 21: this
# stage makes one to four cheap calls per target where the evidence stage makes five expensive
# ones -- measured, two network requests per call here against the twenty-odd an InterPro or a
# ChEMBL lane spends -- and the shortlist a binder designer actually reads is longer than
# three. Stated, not fitted: no retrieval quality was compared across values of it.
#
# One thing to know about this number, measured on the held-out run of 12 Nov 2026 and left to
# the caller rather than decided here. Rank 6 is `Papain-like protease nsp3`, whose
# `Peptidase C16` domain folds at 76.4 mean confidence -- among the three best structures in
# the run -- so at 5 the page offers a structure for it and no literature. Covering it costs
# four queries and about eight seconds: `top=6`.
LITERATURE_TOP_TARGETS = 5

# Dated rows kept per query. A citable row is the unit of value in this stage and five is what
# a reader will follow; `hit_count` already carries the magnitude, and keeping fifty rows per
# query would put a bibliography in a run record to be read by nobody.
LITERATURE_HITS_PER_QUERY = 5

# Seconds the whole stage may spend, in both of the roles a budget plays here: the wall budget
# the gate compares `literature.search`'s declared 90-second worst case against -- anything
# under 90 would have the gate refuse every call before dispatch -- and the elapsed ceiling that
# stops the stage asking.
#
# It was written as 180, on the arithmetic that Europe PMC answers in about a second and
# `literature.PAUSE` waits 0.34 s between requests, so sixteen queries at two requests each is
# well inside it. That arithmetic is right on a good afternoon and was measured: 20 queries in
# 35.9 seconds on 12 Nov 2026. The same stage run an hour later, against a slower Europe PMC,
# spent 45 seconds on its first query and 195 seconds on eleven of sixteen -- one query timed
# out at `literature.TIMEOUT` and the ceiling then refused the remaining four, which were the
# whole of nsp12, the polymerase. Every refusal was recorded with its reason, so the run was
# honest; it was also missing its headline target.
#
# So 180 was never a budget for this stage, it was a budget for two of its queries: the tool
# itself declares 90 seconds of worst case per call and the stage makes sixteen of them. 600
# covers a bad afternoon at the measured bad-afternoon rate of about 18 seconds a query, is
# seventeen times the measured good-afternoon cost, and still cannot be the reason a run misses
# the twenty-five minute cap on the workflow that serves a visitor -- which is the argument
# `EVIDENCE_BUDGET_SECONDS` makes for the stage before it, at 700.
LITERATURE_BUDGET_SECONDS = 600.0

# A UniProt entry identifier sitting in a target's name, left there by the identification
# cascade when a chain was named from a relative's entry rather than from a mature-chain
# annotation: `NCAP_BCRP3`, `VME1_BC279`, `ORF9B_SARS`. It is a database key and not a word
# anybody publishes, and measured 12 Nov 2026 a phrase containing one returns 0 records in
# Europe PMC where the same phrase without it returns 11055. Stripped for that reason, and the
# strip is recorded on the row so a reader can see the name was not the name that was sent.
_ENTRY_ID = re.compile(r"^[A-Z0-9]+_[A-Z0-9]+$")

# A systematic designation specific enough to anchor a query on its own, and to settle that the
# accession clause is not needed. Deliberately narrow: only the replicase `nspN` series and the
# `ORFNx` series qualify. `S1`, `S2` and `7a` are designations too and are **not** matched here,
# because as search tokens they are not specific -- `S1` retrieves a nuclease and a ribosomal
# protein -- and anchoring on one would be the same error as asking for "Y1" bare.
_DESIGNATION = re.compile(r"^(?:nsp\d+|orf\d+[a-z]?)$", re.IGNORECASE)

# A token to split off the name to find the descriptive part behind it, which is a wider set
# than the one above: `S2'` is not specific enough to anchor on and still has to come off
# `Spike protein S2'` for `Spike protein` to be found underneath it. The length bound is what
# keeps `3C-like` -- which carries a digit and must stay attached to `proteinase` -- out of it.
_SUBUNIT = re.compile(r"^(?:nsp\d+|orf\d+[a-z]?|[A-Za-z]{0,2}\d{1,2}[A-Za-z]?)'?$",
                      re.IGNORECASE)

# UniProt's long form for the same thing. `Non-structural protein 10` and `nsp10` are one
# protein under two names; the long form is kept as an anchor beside the short one rather than
# replaced by it, because a paper may use either -- measured 12 Nov 2026, `nsp10` alone matches
# 374 records and the two OR'd together match 380.
_LONG_FORM = re.compile(r"non[- ]structural protein\s+(\d+)", re.IGNORECASE)

# Descriptive remainders that name a CLASS of protein rather than a protein, and are therefore
# refused as anchors. An AUTHORED list, in the sense `literature.AXIS_TERMS` and
# `domain_inventory.RELEVANCE_TERMS` are authored -- nothing was optimised against a labelled
# corpus -- but every entry has a count behind it, measured live on 12 November 2026, and the
# counts are what the list is for. A generic remainder is OR'd with the designation, so it
# cannot narrow a query and can only widen it:
#
#     nsp13 alone                                366
#     nsp13 OR Helicase                       20318   RNA helicase HDX-MS, substrate libraries
#     nsp10 alone                                374
#     nsp10 OR "Non-structural protein"         4040   nervous necrosis virus, alphaviruses
#     nsp10 OR "Non-structural protein 10"       380   -- the FULL name costs nothing
#     ORF9b alone                                119
#     ORF9b OR protein                      3726147   the entire index
#
# Three million records is not a widening, it is the loss of the query. Only the remainder is
# refused and never the full name, which is why `Non-structural protein 10` survives while
# `Non-structural protein` does not. Deliberately NOT included: `Nucleoprotein` (11055) and
# `Membrane protein` (57285), which are broad and are also the only name those two chains have;
# both take the accession route, which bounds them to this genome's 4725 records.
GENERIC_NAMES = frozenset((
    "protein", "proteins", "polyprotein", "peptide", "chain", "domain", "region",
    "non-structural protein", "nonstructural protein", "structural protein",
    "helicase", "protease", "proteinase", "polymerase", "replicase", "enzyme",
))


def _phrase(text):
    """Text safe to put inside a quoted Europe PMC phrase: no double quotes, no runs of space.

    The double quote is the one character that cannot survive, and `literature._balanced` does
    not catch it here for an arithmetic reason worth writing down. That guard refuses an ODD
    number of quotes, because an odd number is what truncates a clause. This composition quotes
    each phrase and uses every anchor twice, once under TITLE and once under ABSTRACT, so a
    single stray quote in a name arrives as a pair and the count comes out even. Measured
    12 Nov 2026 with a deliberately broken name: `Bad "name protein` composed to
    `(TITLE:"Bad "name protein" OR ABSTRACT:"Bad "name protein")`, four quotes, balanced,
    dispatched, and answered with hitCount 0 at HTTP 200 -- a false absence of exactly the kind
    `literature.py`'s second guard exists to make unreachable, slipped past it by doubling.

    So the quote is removed from the phrase before it is wrapped, and `_balanced` stays as the
    backstop for the brackets it does catch. The removal is visible because the composed query
    is recorded on every row.
    """
    return " ".join(str(text or "").replace('"', " ").split())


def _scoped(phrase):
    """One anchor phrase, scoped so the paper has to be about the protein.

    Unscoped, a phrase matches anywhere Europe PMC indexes, which for its open-access subset is
    the whole article. That is the right scope for a domain name and the wrong one for a
    protein name: it counts every paper that cites one.
    """
    phrase = _phrase(phrase)
    return f'(TITLE:"{phrase}" OR ABSTRACT:"{phrase}")'


def _accession_clause(genome_accession):
    """`ACCESSION_ID` over both forms of this genome's accession, or empty without one.

    Both forms, for the reason `literature._subject_clause` gives for its own accession route:
    the bare and versioned strings are different tokens in this index with different counts, so
    asking for one alone silently discards the other's records.
    """
    if not genome_accession:
        return ""
    forms = [genome_accession]
    bare = genome_accession.split(".")[0]
    if bare != genome_accession:
        forms.append(bare)
    return "(" + " OR ".join(f'ACCESSION_ID:"{form}"' for form in forms) + ")"


def chain_anchors(chain_name):
    """The phrases that name one chain's protein, and whether any of them is specific.

    Returns `{"anchors", "designations", "entry_id_stripped", "name_sent",
    "generic_remainder_refused"}`. The anchors are what goes in the query's scoped OR;
    `designations` says whether the chain carries a systematic name, which is what decides
    whether the accession clause is needed.

    Three reductions, each measured and each replacing a form that was tried first. The entry
    identifier is dropped -- see `_ENTRY_ID`. A descriptive remainder that names a class rather
    than a protein is refused -- see `GENERIC_NAMES`, and note that the FULL name is never
    refused, so a chain whose whole name is broad still has one. And an anchor that contains
    another anchor as a substring is dropped, because in a disjunction the longer phrase is
    subsumed by the shorter one: a record matching `"Spike protein S2'"` matches
    `"Spike protein"` as well, so keeping both widens nothing and only makes the recorded query
    harder for a reader to check. That last is the deduplication
    `domain_inventory._relevance_terms` performs, run in the opposite direction because this is
    an OR and that was a score.
    """
    raw = (chain_name or "").strip()
    tokens = raw.split()
    stripped = [t for t in tokens if _ENTRY_ID.match(t)]
    kept = [t for t in tokens if not _ENTRY_ID.match(t)]
    name = " ".join(kept).strip()

    def bare(token):
        return token.strip("'\",.()")

    designations = [bare(t) for t in kept if _DESIGNATION.match(bare(t))]
    long_form = _LONG_FORM.search(name)
    if long_form:
        short = f"nsp{long_form.group(1)}"
        if not any(d.lower() == short for d in designations):
            designations.append(short)

    # The name with its subunit tokens taken out: "3C-like proteinase" out of "3C-like
    # proteinase nsp5", "Spike protein" out of "Spike protein S2'". Papers name the function
    # far more often than the systematic number, so this is the anchor that carries the recall
    # while the designation carries the precision.
    descriptive = " ".join(t for t in kept if not _SUBUNIT.match(bare(t))).strip(" -,")
    refused = None
    if descriptive and descriptive.lower() in GENERIC_NAMES and descriptive != name:
        refused, descriptive = descriptive, ""

    candidates = []
    for phrase in designations + [descriptive, name]:
        phrase = (phrase or "").strip()
        if phrase and phrase not in candidates:
            candidates.append(phrase)
    anchors = [a for a in candidates
               if not any(other != a and other.lower() in a.lower() for other in candidates)]
    return {"anchors": anchors, "designations": designations,
            "entry_id_stripped": stripped or None, "name_sent": name or None,
            "generic_remainder_refused": refused}


def literature_subject(chain_name, domain=None, genome_accession=None):
    """The Europe PMC query for one chain, or for one domain of it.

    A domain's UniProt description is not a search term on its own, and that is the whole
    difficulty of this stage: "Y1" retrieved 7939 records about adrenocortical cells and
    neuropeptide receptors, "Dimerization" retrieved 31632, and "RdRp catalytic" retrieved the
    right field only by luck. The description has to be anchored to the protein it belongs to.

    So the query is composed rather than passed through -- the domain's description AND the
    chain's own anchor phrases, as a fielded Europe PMC query. The domain phrase goes in
    unscoped because these are database-register names that live in methods sections; the
    anchors are scoped to title or abstract so the paper is about the protein. Where the chain
    carries no systematic designation the genome's accession is ANDed as the anchor of last
    resort, which is the module docstring's 400-records-to-5 measurement.

    `genome_accession` is last and optional so the two-argument form in the written spec still
    calls. Returns `(subject, subject_kind)`; `subject_kind` is always `"query"`, because a
    composed fielded query is not a term and `literature._subject_clause`'s term route would
    refuse it for carrying a colon.
    """
    spec = chain_anchors(chain_name)
    if not spec["anchors"]:
        raise ValueError(
            f"chain_name={chain_name!r} yields no anchor phrase once its entry identifier is "
            f"removed, so a domain query for it could only be the bare description -- which "
            f"is the retrieval this stage exists to replace.")
    # One anchor needs no wrapper -- `_scoped` already brackets it -- and `((...))` in a query
    # a reader is meant to check by hand is noise.
    clause = (_scoped(spec["anchors"][0]) if len(spec["anchors"]) == 1
              else "(" + " OR ".join(_scoped(a) for a in spec["anchors"]) + ")")
    parts = []
    if domain:
        domain = _phrase(domain)
        if not domain:
            raise ValueError("domain is empty once it is made safe to quote; an empty phrase "
                             "would compose to the chain-level query under a domain's label, "
                             "and the row would report a chain's hit count as a domain's.")
        parts.append(f'"{domain}"')
    parts.append(clause)
    if not spec["designations"]:
        accession = _accession_clause(genome_accession)
        if accession:
            parts.append(accession)
    return " AND ".join(parts), "query"


def _anchor_route(chain_name, genome_accession):
    """Which anchor carried this target's queries, for the record and for the page."""
    spec = chain_anchors(chain_name)
    if spec["designations"]:
        return "designation", spec
    if _accession_clause(genome_accession):
        return "genome_accession", spec
    return "descriptive_name_only", spec


def _hit_row(row):
    """One Europe PMC hit, trimmed to what makes it citable.

    The abstract is dropped and `has_abstract` kept: `literature._row` truncates an abstract at
    1500 characters, and five of those per query across four queries and five targets would add
    the better part of a hundred kilobytes to a record for a reader who has the DOI and the URL.
    Its absence is still visible, which is the part worth keeping.

    A hit with no `first_publication_date` is KEPT and marked uncitable, never dropped. The
    suite already holds that behaviour for a different table -- a target with no deposited
    structure has no earliest deposition, and
    `test_a_table_row_with_no_date_is_kept_and_marked_uncitable` requires the row to survive
    marked rather than vanish -- and the reason is the same here: dropping undated rows would
    make the retrieval read as though every record Europe PMC returned were datable.
    """
    date = row.get("first_publication_date") or None
    return {
        "rank": row.get("rank"),
        "source": row.get("source"),
        "id": row.get("id"),
        "title": row.get("title"),
        "journal": row.get("journal"),
        # `literature._row` comments this as "The dossier's `available_from`": the publication
        # date, never the time of the call.
        "first_publication_date": date,
        "doi": row.get("doi"),
        "is_preprint": bool(row.get("is_preprint")),
        "has_abstract": bool(row.get("has_abstract")),
        "url": row.get("url"),
        "citable": date is not None,
        "uncitable_because": (
            None if date else
            "Europe PMC returned this record with no firstPublicationDate, so it cannot be "
            "placed on a decision clock. Kept and marked rather than dropped."),
    }


def _unasked(scope, domain_label, domain_span, subject, subject_kind, status, note):
    """A query entry for a query that was never sent, with the same keys as one that was.

    One constructor rather than four literals, because the page reads these rows and a refusal
    missing a key the others carry would make a refused query render as a missing one -- which
    is the distinction this repository spends most of its records on. Where a subject was
    composed before the refusal it is kept, so what was not asked is still visible.
    """
    return {"scope": scope, "domain_label": domain_label, "domain_span": domain_span,
            "subject": subject, "subject_kind": subject_kind, "status": status, "note": note,
            "hit_count": None, "preprints": None, "seconds": 0.0, "network_calls": None,
            "reused_from": None, "date_filter_verified": None, "hits": []}


def _query_entry(scope, domain_label, domain_span, subject, subject_kind, date_to, budget,
                 cache=None, asked_by=None):
    """Dispatch one composed query through `chief`, and return its row whatever happened.

    Through the envelope and not through `literature.search` directly, exactly as
    `chain.evidence_rows` does it: `tool_spec`, `schema_conformance`, `chief.call` with a
    context, inside `chief.counting_requests()`. The envelope is what makes the call a
    recorded, charged, policy-visible event, and a stage that reached the function behind it
    would be the one lane in this pipeline whose cost is invisible.

    `per_axis=False` on every call. `literature.AXIS_TERMS` is by its own module's admission
    "an authored heuristic that no one has validated against a labelled corpus", every axis row
    it produces carries `verified: False`, and each axis costs a request. This stage buys
    citable rows and does not buy unvalidated counts.

    `cache` keys on the composed query, because this run genuinely repeats one. Spike S2' is a
    cleavage product of S2, so ranks 1 and 3 are two ranked chains of one glycoprotein that
    compose the same anchor and fold the same three regions at the same spans -- the collision
    `domain_folds._unit_slug` had to qualify filenames against. Five of this run's seventeen
    queries are duplicates of another target's, and asking Europe PMC the same question five
    times would be a cost with no answer behind it. A reused row says so in `reused_from` and
    carries no network cost, so the record never claims a call that was not made.
    """
    from . import chief
    from .tools import tool as tool_spec

    row = {"scope": scope, "domain_label": domain_label, "domain_span": domain_span,
           "subject": subject, "subject_kind": subject_kind,
           "status": None, "note": None, "hit_count": None, "preprints": None,
           "seconds": None, "network_calls": None, "reused_from": None,
           "date_filter_verified": None, "hits": []}

    if cache is not None and subject in cache:
        done, owner = cache[subject]
        row.update({k: v for k, v in done.items()
                    if k not in ("scope", "domain_label", "domain_span", "subject",
                                 "subject_kind", "seconds", "network_calls", "reused_from")})
        row.update({"seconds": 0.0, "network_calls": 0, "reused_from": owner,
                    "note": f"the identical composed query was already sent for {owner!r}; "
                            f"this row reuses that answer and cost no request"})
        return row

    arguments = {"subject": subject, "subject_kind": subject_kind,
                 "top": LITERATURE_HITS_PER_QUERY, "per_axis": False}
    if date_to:
        arguments["date_to"] = date_to

    started = time.time()
    try:
        with chief.counting_requests() as tally:
            spec = tool_spec("literature.search")
            chief.schema_conformance(spec, arguments)
            envelope = chief.call(spec, arguments, context=chief.context(
                None, now=0, wall_budget_seconds=budget))
    except Exception as broke:                                           # noqa: BLE001
        # An endpoint failure is already an envelope, so this is the unanticipated one. It is
        # written down rather than raised, for the reason `evidence_rows` gives: one bad query
        # may not cost the other nineteen their rows.
        row.update({"status": "error", "note": f"{type(broke).__name__}: {broke}"[:200],
                    "seconds": round(time.time() - started, 2)})
        return row

    row.update({"status": envelope.get("status"), "note": envelope.get("note") or None,
                "seconds": round(time.time() - started, 2), "network_calls": tally["made"]})
    value = envelope.get("value")
    if isinstance(value, dict):
        hits = sorted((value.get("hits") or {}).values(),
                      key=lambda h: h.get("rank") or 10 ** 6)
        row.update({
            "hit_count": value.get("hit_count"),
            # The PPR count: preprints are the literature at onset, and a PubMed-only backend
            # cannot report it at all, which is one of the four criteria Europe PMC was chosen
            # on.
            "preprints": (value.get("by_source") or {}).get("PPR", 0),
            "date_filter_verified": value.get("date_filter_verified"),
            "hits": [_hit_row(h) for h in hits[:LITERATURE_HITS_PER_QUERY]],
        })
    elif row["status"] == "empty":
        # `chief.ONSET_FOUND_KEYS` declares `hit_count` as this tool's found key, so a query
        # matching nothing comes back `empty` with no value at all. That zero is a result and is
        # recorded as one: `literature.py`'s two pre-dispatch guards exist precisely so a zero
        # here cannot have come from a malformed query, which is what makes it reportable.
        #
        # The note is rewritten rather than passed through. `chief._passthrough` sets it to
        # `str(absence)`, which for this tool is the repr of a Python dict -- a page rendering
        # `{'subject_zero': True, 'axes_zero': (), ...}` to a reader would be reporting a data
        # structure instead of a finding. The sentence below says the same thing in the
        # register the rest of the record uses.
        row.update({
            "hit_count": 0, "preprints": 0,
            "note": "a measured absence: Europe PMC holds no record matching this composed "
                    "query. The date bounds and the query string were validated before "
                    "dispatch and the response was checked for an in-body error code, so this "
                    "zero is a recorded absence of indexed literature and not a failed "
                    "lookup."})
    if cache is not None and row["status"] in ("ok", "empty"):
        cache[subject] = (row, asked_by)
    return row


def _folded_domains(fold_row):
    """The units of one chain that have coordinates, in the order the fold record put them.

    Two row shapes are accepted on purpose. `domain_folds.run` writes `units` carrying
    `span_in_this_genome` and an explicit `folded` flag; `chain.fold_stage` is being given
    `fold_units` carrying `span`, and whether a unit was folded is the per-chain cap's business
    rather than the ceiling's. So `folded` decides when it is present, and the absence of a
    refusal decides when it is not -- which keeps this stage callable from the record today and
    from the pipeline once it writes one.

    The order is not re-sorted. The fold record's order is the inventory's relevance order,
    which `CONTRACT-ADDENDUM-ORDERING` establishes is a claim about function; re-sorting by
    anything here would turn it into a claim about retrieval.
    """
    units = fold_row.get("units") or fold_row.get("fold_units") or ()
    out = []
    for unit in units:
        if unit.get("kind") != "domain":
            continue
        folded = unit.get("folded")
        if folded is False:
            continue
        if folded is None and unit.get("refused_because"):
            continue
        label = unit.get("label") or unit.get("description") or unit.get("type")
        span = unit.get("span_in_this_genome") or unit.get("span")
        out.append({"label": label, "span": list(span) if span else None,
                    "length_aa": unit.get("length_aa") or unit.get("length"),
                    "mean_plddt": unit.get("mean_plddt")})
    return out


def literature_rows(scored, fold_rows, top=LITERATURE_TOP_TARGETS, date_to=None,
                    budget=LITERATURE_BUDGET_SECONDS, genome_accession=None):
    """Dated, citable literature for the top targets and each of their folded domains.

    One row per target, each carrying one entry per query: the composed query string that was
    actually sent, the status the gate or the endpoint produced, what it cost, the hit count,
    the preprint count and the dated rows. A refusal is an entry with its reason and never an
    omission -- a target with no name, a composed query that fails the balance check, a call
    the gate refuses on its declared cost, an endpoint error, and a query the stage's own
    budget stopped before it was sent all produce one.
    """
    from .literature import _balanced

    by_target = {}
    for fold_row in fold_rows or ():
        key = fold_row.get("target")
        if key is not None:
            by_target[key] = fold_row

    rows, cache = [], {}
    started = time.time()
    for score in sorted(scored, key=lambda r: r.get("rank") or 10 ** 6)[:top]:
        target = score.get("target")
        folded = _folded_domains(by_target.get(target) or {})
        route, anchors = _anchor_route(target, genome_accession)
        row = {
            "rank": score.get("rank"), "target": target,
            "nearest_homolog": score.get("nearest_homolog"),
            "anchor_phrases": anchors["anchors"],
            "anchor_route": route,
            "entry_id_stripped_from_the_name": anchors["entry_id_stripped"],
            "generic_remainder_refused": anchors["generic_remainder_refused"],
            "domains_folded": len(folded),
            "no_domain_queries_because": None,
            "queries": [], "dated_hits": 0, "undated_hits": 0,
        }
        if not folded:
            row["no_domain_queries_because"] = (
                "stage 3 folded no domain of this chain, so there is no domain with "
                "coordinates whose literature a reader could act on. The chain-level query "
                "still ran.")

        wanted = [("chain", None, None)] + [("domain", d["label"], d["span"]) for d in folded]
        for scope, label, span in wanted:
            if not target:
                row["queries"].append(_unasked(
                    scope, label, span, None, None, "refused",
                    "the ranked row carries no target name, so no anchor phrase exists and "
                    "the query could only have been the bare description"))
                continue
            try:
                subject, subject_kind = literature_subject(
                    target, domain=label, genome_accession=genome_accession)
                # Balance-checked here as well as inside the tool, so an unbalanced composed
                # query becomes a row with its reason instead of a network call that comes back
                # as the tool's own failure. `literature._balanced` is the same guard the
                # dispatch would apply; a bracket in a UniProt description -- nsp13 carries
                # `(+)RNA virus helicase ATP-binding` -- is the reason it can fire at all.
                _balanced(subject, "subject")
            except ValueError as refused:
                row["queries"].append(_unasked(scope, label, span, None, "query", "refused",
                                               str(refused)[:300]))
                continue
            spent = time.time() - started
            if spent > budget:
                row["queries"].append(_unasked(
                    scope, label, span, subject, subject_kind, "not_asked",
                    f"the stage had spent {spent:.1f}s of its {budget:.0f}s budget before "
                    f"this query; the composed query is recorded so what was not asked is "
                    f"visible"))
                continue
            row["queries"].append(_query_entry(scope, label, span, subject, subject_kind,
                                               date_to, budget, cache=cache, asked_by=target))

        row["dated_hits"] = sum(1 for q in row["queries"] for h in q["hits"] if h["citable"])
        row["undated_hits"] = sum(1 for q in row["queries"] for h in q["hits"]
                                  if not h["citable"])
        rows.append(row)
    return rows


def stage(scored, fold_rows, top=LITERATURE_TOP_TARGETS, date_to=None,
          budget=LITERATURE_BUDGET_SECONDS, genome_accession=None):
    """`literature_rows`, plus the stage entry that goes in the record beside stages 1 to 6.

    Returns `(stage, rows)`, the shape `chain.evidence_stage` returns, so the call site in
    `chain._case` is the one line `stage, literature = literature_stage.stage(...)` beside the
    one it already has for stage 6.
    """
    started = time.time()
    rows = literature_rows(scored, fold_rows, top=top, date_to=date_to, budget=budget,
                           genome_accession=genome_accession)
    queries = [q for r in rows for q in r["queries"]]
    ok = [q for q in queries if q["status"] in ("ok", "empty")]
    domain_queries = [q for q in queries if q["scope"] == "domain"]
    entry = {
        "stage": 7, "name": "Dated literature for each target and each folded domain",
        "tool": "literature.search",
        "endpoints": ["Europe PMC"],
        "input": {"targets_in": len(rows), "ranked_available": len(scored),
                  "domains_folded": sum(r["domains_folded"] for r in rows),
                  "hits_per_query": LITERATURE_HITS_PER_QUERY,
                  "wall_budget_seconds": budget},
        "output": {
            "targets_covered": len(rows), "queries_made": len(queries),
            "domain_queries": len(domain_queries),
            "ok": len(ok), "not_ok": len(queries) - len(ok),
            "queries_reused_from_an_identical_one":
                sum(1 for q in queries if q.get("reused_from")),
            "network_calls": sum(q["network_calls"] or 0 for q in queries),
            "dated_hits": sum(r["dated_hits"] for r in rows),
            "undated_hits": sum(r["undated_hits"] for r in rows),
            "per_axis_requested": False,
            "changes_any_rank": False,
            "every_target_has_a_chain_query":
                all(any(q["scope"] == "chain" for q in r["queries"]) for r in rows),
        },
        # Stated rather than left to be inferred from a null. `literature.py` exists largely to
        # stop undated retrieval being passed off as dated, so the absence of a ceiling is a
        # recorded property of this stage and not a gap in it.
        "date_ceiling_applied": bool(date_to),
        "date_to": date_to,
        "date_ceiling_means": (
            f"Every hit was published on or before {date_to}, verified by the paired control "
            f"call literature.search records in date_filter_verified."
            if date_to else
            "No publication-date ceiling was applied, so these hits include literature "
            "published after the onset this chain replays. They are literature that informs a "
            "binder design now, and they are not a claim about what was knowable at onset."),
        "seconds": round(time.time() - started, 2),
        "hands_to_next": "nothing. This annotates the shortlist and never reorders it",
    }
    return entry, rows


def run(case=None, path=None, record=True, landing=None, folds=None,
        top=LITERATURE_TOP_TARGETS, date_to=None, budget=LITERATURE_BUDGET_SECONDS):
    """The stage over one case of the landing record, written as a record.

    `case` is a case block from a chain record. With none supplied the landing record is read
    and its HELD-OUT case picked, for the reason `predicted_inventory` and `domain_inventory`
    both give: the landing page is the run with SARS-CoV-2 deleted from its own reference, so
    its targets resolve to relatives, and reading the un-held-out case would put the deleted
    organism's own records on a page whose premise is that they were deleted before the run.
    """
    payload_in = json.loads(Path(landing or LANDING).read_text())
    if case is None:
        case = next((c for c in payload_in["cases"] if c.get("reference_excluded_organism")),
                    payload_in["cases"][0])
    fold_record = json.loads(Path(folds or FOLDS).read_text())
    fold_rows = fold_record.get("chains") or ()

    entry, rows = stage(case.get("scores") or (), fold_rows, top=top, date_to=date_to,
                        budget=budget, genome_accession=case.get("genome_accession"))

    queries = [q for r in rows for q in r["queries"]]
    dated = sum(r["dated_hits"] for r in rows)
    undated = sum(r["undated_hits"] for r in rows)
    zero = [q for q in queries if q["hit_count"] == 0]

    payload = {
        "experiment": "literature_stage",
        "status": "complete",
        "question": "For each shortlisted target and each domain that was folded, what dated "
                    "literature exists, and under which query?",
        "answer": (f"{dated} dated citable rows and {undated} undated ones across "
                   f"{len(queries)} composed queries over {len(rows)} targets, of which "
                   f"{len([q for q in queries if q['scope'] == 'domain'])} were per-domain. "
                   + (f"{len(zero)} queries returned a measured zero."
                      if zero else "No query returned zero.")),
        "case": case.get("case"),
        "genome_accession": case.get("genome_accession"),
        "reference_excluded_organism": case.get("reference_excluded_organism"),
        "top_targets": top,
        "hits_per_query": LITERATURE_HITS_PER_QUERY,
        "date_ceiling_applied": entry["date_ceiling_applied"],
        "date_to": date_to,
        "date_ceiling_means": entry["date_ceiling_means"],
        "stage": entry,
        "targets": rows,
        "composition": {
            "domain_phrase": "the annotated description, verbatim and unscoped, so it matches "
                             "the methods and tables where a database-register name actually "
                             "appears. Scoping `Peptidase C16` to title or abstract returns 0 "
                             "and would record a false absence; unscoped and anchored it "
                             "returns 3.",
            "anchor_phrases": "the target's name split into a systematic designation and a "
                              "descriptive name, each scoped to TITLE or ABSTRACT so the "
                              "paper is about the protein. The name is not passed through: "
                              "`3C-like proteinase nsp5` as one phrase returns 1 record and "
                              "`(nsp5 OR \"3C-like proteinase\")` returns 775.",
            "entry_identifier": "stripped. `NCAP_BCRP3 Nucleoprotein` and `VME1_BC279 "
                                "Membrane protein` both return 0 as written, a false absence "
                                "caused by a database key in a target's name.",
            "generic_remainder": "refused as an anchor, because it is OR'd with the "
                                 "designation and so can only widen the query. `ORF9b` alone "
                                 "matches 119 records and `ORF9b OR protein` matches "
                                 "3726147 -- the whole index. `nsp13` alone matches 366 and "
                                 "`nsp13 OR Helicase` matches 20318. The full name is never "
                                 "refused, which is why `Non-structural protein 10` survives "
                                 "where `Non-structural protein` does not.",
            "genome_accession_clause": "added only where the chain's name yields no systematic "
                                       "designation, because a one-word protein noun does not "
                                       "identify a protein: `\"Dimerization\" AND "
                                       "nucleoprotein` returns 400 records whose top three "
                                       "are Hazara virus, a DNA-superhelicity sensor and "
                                       "influenza A, and the accession clause takes it to 5 "
                                       "correct ones.",
            "measured_on": "2026-11-12, live against Europe PMC",
            "verified": False,
        },
        "limitations": [
            "A hit count is evidence of attention and of nothing else. It does not say the "
            "protein is essential, conserved, exposed, expressible or bindable, and it must "
            "not be read as a measure of a target's quality -- `literature.py` is emphatic "
            "about this and nothing here may undo it.",
            "No publication-date ceiling was applied unless `date_to` was passed, so by "
            "default these hits include literature published after the onset this chain "
            "replays. `date_ceiling_applied` states which, on the stage and on this record.",
            "The composed queries are an authored composition, in the sense "
            "`literature.AXIS_TERMS` and `domain_inventory.RELEVANCE_TERMS` are authored: no "
            "retrieval quality was measured against a labelled corpus. Every query string is "
            "recorded on its row, because a number whose query is not recorded cannot be "
            "checked.",
            "A domain is queried under the name its annotation gives it, which is often a "
            "database-register name rather than the one the field uses, and that is where "
            "every measured zero in this record comes from. The relative's entry calls the "
            "receptor-binding domain `BetaCoV S1-CTD`; anchored and asked under that name it "
            "returns 0 records, where the field's own name for the same 194 residues -- the "
            "RBD -- has a large literature. `RdRp Palm C-ter` and `CoV Nsp3 Y` return 0 for "
            "the same reason. The hit count is a count for the query that was sent, and the "
            "query is on the row so that can be seen.",
            "The genome-accession clause buys precision with recall, and it is the anchor of "
            "last resort for exactly that reason. A genome three days old has no literature "
            "citing its accession, so for a chain whose own name is too generic to anchor on, "
            "this stage would return measured zeros at onset -- a property of a pathogen "
            "nobody has published on, recorded rather than hidden.",
            "Only the domains stage 3 folded are queried. A functionally important region that "
            "the per-chain cap left unfolded has literature this record does not hold, and "
            "`domain_inventory` is where what the annotation offers is counted.",
            "Nothing here reads a paper. Relevance is unjudged, retracted records are not "
            "filtered, and the abstract is dropped from every row -- `has_abstract` says "
            "whether there was one.",
            "This stage changes no rank. The ranking is final at the end of stage 2 and a "
            "retrieval that annotates the shortlist may not reorder it.",
        ],
        "provenance": provenance(),
    }
    payload["content_sha256"] = content_digest(payload)
    if record:
        out = Path(path) if path else RECORD
        write_json(out, payload)
        # Relative to the tree where it is one, absolute where it is not. The sibling record
        # modules print `out.relative_to(ROOT)` unguarded, which raises for a `path=` outside
        # the repository -- after the record has already been written, so the write succeeds
        # and the run reports a traceback. That happened writing this stage's own verification
        # record to a scratch directory on 12 Nov 2026.
        try:
            shown = out.relative_to(ROOT)
        except ValueError:
            shown = out
        print(f"wrote {shown}  "
              f"({len(queries)} queries over {len(rows)} targets, "
              f"{dated} dated rows, {undated} undated, "
              f"{len(zero)} measured zeros)")
    return payload


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--top", type=int, default=LITERATURE_TOP_TARGETS,
                        help="how many ranked targets to cover")
    parser.add_argument("--date-to", default=None,
                        help="ISO publication-date ceiling; absent means today's literature, "
                             "recorded as date_ceiling_applied false")
    parser.add_argument("--budget", type=float, default=LITERATURE_BUDGET_SECONDS,
                        help="seconds the whole stage may spend")
    parser.add_argument("--no-record", action="store_true", help="do not write the record")
    args = parser.parse_args(argv)
    run(top=args.top, date_to=args.date_to, budget=args.budget, record=not args.no_record)


if __name__ == "__main__":
    main()
