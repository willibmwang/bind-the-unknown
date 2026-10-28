"""Predicted structural models per accession, with per-range confidence and a hard wall against experiment.

The axis this serves is `accessibility`: is the epitope solvent-exposed in the relevant
conformation. A predicted model is the only way to ask that question about a protein nobody
has deposited, which is the same asymmetry `structures_evidence` is built around -- a pocket
can be scored the day a model appears, an antibody count cannot.

**This tool must never touch `precedent`, and the separation is structural rather than
advisory.** `structures_evidence` counts *deposited* entries and labels that count a
research-attention proxy. A prediction is not a deposition: AlphaFold modelled every
accession in a reference proteome whether or not a single person ever cared about it, so a
predicted-model count carries no attention signal at all. Pooling the two would turn "nobody
has solved this structure" into "we have a structure". So this module partitions the models
it retrieves on `model_category` and puts everything marked `EXPERIMENTALLY DETERMINED` into
`excluded_experimental`, counted separately, never once in `predicted_models`. The exclusion
is reported rather than silent, because a reader who sees eighteen models vanish deserves to
know which tool owns them.

**Why 3D-Beacons and not AlphaFold DB alone.** The brief for this module named
`https://alphafold.ebi.ac.uk/api/prediction/<accession>`, and that endpoint alone reports a
false absence on the one accession this repository already singles out. P03470 -- the
neuraminidase whose true deposited-structure count is zero, which `structures_evidence`
documents as the case that taught it to distinguish a 404 from an empty body -- answers the
AlphaFold prediction endpoint with HTTP 404 and a body of `{}`. It is not unmodelled. The
3D-Beacons Hub reports a SWISS-MODEL template-based model over residues 66-453 at QMEANDisCo
0.913, created 2026-09-28. An AlphaFold-only tool would have recorded "no predicted model"
for a target that has one, on the very target whose zero this project reasons about. A false
absence is the costliest return available here, so the inventory comes from the aggregator.

**Why AlphaFold DB as well, and not the aggregator alone.** 3D-Beacons' summary endpoint
carries only a model-level average, and its residue-level endpoint
(`/uniprot/<accession>.json`) silently covers one provider: for P0DTC1 it returns 32
structures and all 32 are SWISS-MODEL, for P0DTD1 52 of the 113 the summary lists, with every
AlphaFold DB, PED and SASBDB model dropped. So per-residue pLDDT is reachable *only* through
the AlphaFold prediction endpoint's `plddtDocUrl`, and per-residue QMEANDisCo *only* through
the aggregator's residue endpoint. Two sources is not redundancy; neither one answers the
question alone.

**What the models actually are, which the headline does not say.** For the SARS-CoV-2
accessions this project uses, the AlphaFold prediction endpoint serves no DeepMind model at
all. P0DTC1, P0DTC2 and P0DTD1 return `ColabFold v1.5.2` models from provider `VR3D`, with
identifiers like `AF-0000000365840313` and confidence files versioned `_v1.json`. Human
P04637 returns nine `AlphaFold Monomer v2.0 pipeline` models from provider `GDM`, identifiers
`AF-P04637-F1`, confidence `_v6.json`. The endpoint's host name is not the provenance of what
it returns, so `tool_used` and `provider_id` are carried per model and a caller that wants to
say "AlphaFold predicts this" has to look at them first.

**Two numbering traps, both avoided here by construction.**

The first is residue offset. A confidence document numbers its residues 1..N from the start of
the *model*, and these models are fragments: `AF-0000000365840313` covers UniProt 1566-1868
of a 4405-residue polyprotein, and its `residueNumber` runs 1..303. A caller asking about
UniProt 3264-3569 who indexed `confidenceScore` by UniProt number would read somebody else's
residues, off by 1565, and get a plausible number back. The trap is invisible on the proteins
people test with -- every full-length single-domain model has `uniprotStart == 1`, where the
wrong arithmetic is right -- and springs on exactly the viral polyproteins this project
targets. So the mapping is explicit, and when a document's length disagrees with the range it
claims the model is recorded as `mapping_unreliable` and contributes nothing, rather than
being mapped on an assumption.

The second is the confidence *category* vocabulary, which is not stable across file versions
and was measured to disagree. In `AF-0000000365840313-confidence_v1.json` the letter `H`
labels residues scoring 71.38-89.94. In `AF-P04637-F1-confidence_v6.json` the letter `H`
labels residues scoring 90.00-98.69, and 70-90 is `M`. Reading the letters would therefore
report merely-confident residues as very-high for every v1 document on the strength of a
vocabulary learned from a v6 one. Nothing here reads `confidenceCategory`. Every band is
derived numerically from `confidenceScore` against `PLDDT_BANDS`, which is declared in this
file and travels in the payload so a reader can recompute it.

**Scales are kept apart.** pLDDT runs 0-100 and QMEANDisCo runs 0-1, and they are not the same
quantity rescaled -- one is a predicted per-residue error estimate from a neural network, the
other a statistical potential's agreement score against a template. A mean over both is
meaningless. Averages are grouped by `confidence_type` and never merged, and the per-range
summary reports a pLDDT block and a QMEANDisCo block that no code here combines.

**What a return does not establish.** That a protein adopts the modelled conformation. A
prediction is a hypothesis about structure, not an observation of one; a high pLDDT means the
network is confident, which is a statement about the network. Low pLDDT is the more
informative direction and is reported as its own finding, because it frequently marks genuine
intrinsic disorder -- and a disordered epitope is a real and actionable answer on
`accessibility`, not a missing one.
"""
import json
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request

from .common import ROOT, content_digest, provenance, write_json

# The aggregator. One request answers "does any provider model this at all", which is the
# question an AlphaFold-only lookup gets wrong on P03470.
BEACONS_SUMMARY = "https://www.ebi.ac.uk/pdbe/pdbe-kb/3dbeacons/api/uniprot/summary/"
# Residue-level mappings. Measured to carry SWISS-MODEL only, so it is the QMEANDisCo route
# and not a general one; off by default because it returned 2.45 MB for P0DTD1.
BEACONS_RESIDUES = "https://www.ebi.ac.uk/pdbe/pdbe-kb/3dbeacons/api/uniprot/"
# The only route to per-residue pLDDT, via each entry's `plddtDocUrl`.
ALPHAFOLD = "https://alphafold.ebi.ac.uk/api/prediction/"

# Sent on every request. The structural tool reaches EMBL-EBI thirty-six times a call under
# the default Python urllib agent with no contact string; an endpoint that wants to throttle
# or warn an abusive client has to be able to name it.
USER_AGENT = "hpc-pilots/1.0 (research prototype; contact via repository)"

# Between successive requests. Not a rate limit either host publishes -- neither does -- but
# the fair-use posture a keyless public endpoint is entitled to.
SLEEP_SECONDS = 0.34

# Per request. The ceiling in the Tool spec's `worst_case_seconds` is derived from this and
# `MAX_CONFIDENCE_DOCS`, so a declared worst case is arithmetic over enforced timeouts rather
# than an estimate. Observed times run 0.35-6.9 s, so this is roughly three times the slowest
# response measured.
TIMEOUT = 20

# Confidence documents fetched per call. The cap is what makes the worst case finite: a
# polyprotein can carry 28 AlphaFold-served fragments and a range spanning the chain would
# otherwise fetch all of them. When it bites, `confidence_docs_truncated` says so.
MAX_CONFIDENCE_DOCS = 8

# AlphaFold's published pLDDT bands, lower bound inclusive. In the payload, so a reader can
# recompute every fraction instead of trusting one. Deliberately not read from the files'
# own `confidenceCategory`, whose letters were measured to mean different bands in different
# file versions -- see the module docstring.
PLDDT_BANDS = (("very_high", 90.0, 100.0), ("confident", 70.0, 90.0),
               ("low", 50.0, 70.0), ("very_low", 0.0, 50.0))

# 3D-Beacons' category for a deposited or measured structure. Everything carrying it is
# excluded from the predicted set and counted on its own, because it belongs to
# `structures.evidence` and its count is an attention proxy this tool must not launder.
EXPERIMENTAL_CATEGORY = "EXPERIMENTALLY DETERMINED"

_ACCESSION = re.compile(r"^[A-Za-z0-9._-]{1,32}$")
_RANGE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")


class _Budget:
    """Requests made and seconds spent, so the payload reports cost it observed."""

    def __init__(self):
        self.calls = 0
        self.seconds = 0.0
        self.bytes = 0
        self.urls = []


def _get(url, budget, timeout=TIMEOUT):
    """One GET with a descriptive agent. Returns None on 404; every other status raises.

    A 404 from either host means "valid identifier, nothing modelled", which is a finding and
    not a transport error -- the distinction `structures_evidence` documents learning on
    P03470, whose AlphaFold 404 carries a body of `{}`. A 400 from the AlphaFold host means
    the identifier is not an accession, which is a contract violation and is raised as one by
    the caller.
    """
    if budget.calls:
        time.sleep(SLEEP_SECONDS)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                                   "Accept": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as handle:
            raw = handle.read()
            status = handle.status
    except urllib.error.HTTPError as error:
        budget.calls += 1
        budget.seconds += time.perf_counter() - start
        budget.urls.append(url)
        if error.code == 404:
            return None
        raise
    budget.calls += 1
    budget.seconds += time.perf_counter() - start
    budget.bytes += len(raw)
    budget.urls.append(url)
    if status == 204 or not raw.strip():
        return None
    return json.loads(raw)


def parse_ranges(residue_ranges):
    """`("3264-3569", ...)` to validated inclusive intervals, raising on anything else.

    One contract, strictly. A malformed interval is a caller error and never an input to
    guess at, and an inverted one would silently match no residue and read as a confident
    "this site is unmodelled" -- the false absence this module exists to avoid.
    """
    out = []
    for item in residue_ranges:
        if not isinstance(item, str):
            raise ValueError(f"A residue range must be a 'start-end' string, got {item!r}. "
                             f"This tool declares one form on purpose; a pair that happens to "
                             f"unpack is a different contract.")
        match = _RANGE.match(item)
        if not match:
            raise ValueError(f"Residue range {item!r} is not 'start-end' with integers.")
        start, end = int(match.group(1)), int(match.group(2))
        if start < 1 or end < start:
            raise ValueError(f"Residue range {item!r} is not a forward interval from 1; an "
                             f"inverted range matches nothing and would read as an absence.")
        out.append((start, end))
    return out


def _overlap(a_start, a_end, b_start, b_end):
    """Residues common to two inclusive intervals."""
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def _bands(scores):
    """Fraction of residues in each pLDDT band, derived from the scores themselves."""
    if not scores:
        return {name: None for name, _, _ in PLDDT_BANDS}
    out = {}
    for name, low, high in PLDDT_BANDS:
        hits = sum(1 for s in scores if low <= s < high or (high == 100.0 and s >= 100.0))
        out[name] = round(hits / len(scores), 4)
    return out


def _longest_run_below(scores, threshold):
    """The longest consecutive stretch scoring below `threshold`.

    A mean hides this and the shape is what matters: twenty scattered low residues and one
    twenty-residue low stretch average identically, and only the second is a disordered
    segment a binder would have to contend with.
    """
    best = run = 0
    for score in scores:
        run = run + 1 if score < threshold else 0
        best = max(best, run)
    return best


def _summarise_scores(scores, threshold, banded):
    """The per-range confidence block, on whatever scale `threshold` is stated in.

    `banded` exists because `PLDDT_BANDS` is a pLDDT vocabulary and nothing else. Running it
    over QMEANDisCo was a real defect in this file's first version, caught by calling it: a
    0-1 score is below every band's floor, so a SWISS-MODEL range whose median was 0.93 came
    back reported as 100% `very_low`. That is the scale conflation this module's docstring
    forbids, arriving as a plausible number. Bands are None for any other metric rather than
    computed on a vocabulary that does not apply to it.
    """
    bands = _bands(scores) if banded else None
    if not scores:
        return {"residues_scored": 0, "min": None, "median": None, "mean": None, "max": None,
                "fraction_at_or_above_threshold": None, "bands": bands,
                "longest_run_below_threshold": None}
    return {"residues_scored": len(scores),
            "min": round(min(scores), 2), "median": round(statistics.median(scores), 2),
            "mean": round(statistics.fmean(scores), 2), "max": round(max(scores), 2),
            "fraction_at_or_above_threshold":
                round(sum(1 for s in scores if s >= threshold) / len(scores), 4),
            "bands": bands,
            "longest_run_below_threshold": _longest_run_below(scores, threshold)}


def _date(value):
    """`2025-07-03T00:00:00Z` and `2023-05-05` both to `2025-07-03`-shaped, or None.

    The providers disagree on format -- AlphaFold DB stamps an ISO instant, SWISS-MODEL and
    PED a bare date -- and an unnormalised mixture cannot be sorted, which would make the
    earliest-model date this tool offers as `available_from` depend on string collation.
    """
    if not isinstance(value, str) or len(value) < 10:
        return None
    head = value[:10]
    return head if re.match(r"^\d{4}-\d{2}-\d{2}$", head) else None


def beacons_summary(accession, budget):
    """Every model any 3D-Beacons provider holds for the accession, or None when it holds none."""
    return _get(f"{BEACONS_SUMMARY}{urllib.parse.quote(accession)}.json", budget)


def alphafold_entries(accession, budget):
    """AlphaFold-DB-served entries, or None when the host has nothing for the accession.

    A 400 here is the host rejecting the identifier's *format*, which is a different thing
    from having no model for it, and is raised rather than recorded as an absence.
    """
    try:
        payload = _get(f"{ALPHAFOLD}{urllib.parse.quote(accession)}", budget)
    except urllib.error.HTTPError as error:
        if error.code == 400:
            raise ValueError(
                f"AlphaFold DB rejected {accession!r} as an identifier format, so this is not "
                f"a recorded absence: the host never looked for a model. Supply a UniProt "
                f"accession.") from error
        raise
    if isinstance(payload, dict):
        # The host answers an unknown-but-well-formed accession with `{}` under a 404, which
        # `_get` already turns into None; a dict here means an unexpected envelope.
        return None
    return payload or None


def _alphafold_index(entries):
    """Model identifier to the fields only the AlphaFold host supplies."""
    index = {}
    for entry in entries or ():
        identifier = entry.get("entryId")
        if identifier:
            index[identifier] = entry
    return index


def _model_row(summary, af_entry):
    """One predicted model, with its provenance separated from its host."""
    confidence_type = summary.get("confidence_type")
    return {
        "model_identifier": summary.get("model_identifier"),
        "provider": summary.get("provider"),
        "model_category": summary.get("model_category"),
        "model_url": summary.get("model_url"),
        "model_page_url": summary.get("model_page_url"),
        "model_format": summary.get("model_format"),
        "created": _date(summary.get("created")),
        "uniprot_start": summary.get("uniprot_start"),
        "uniprot_end": summary.get("uniprot_end"),
        "coverage": summary.get("coverage"),
        "sequence_identity": summary.get("sequence_identity"),
        "oligomeric_state": summary.get("oligomeric_state"),
        # Kept beside the value, always. 0.911 QMEANDisCo and 82.21 pLDDT are not comparable
        # numbers, and a bare average with no metric name invites exactly that comparison.
        "confidence_type": confidence_type,
        "confidence_scale": {"pLDDT": "0-100", "QMEANDisCo": "0-1"}.get(confidence_type),
        "confidence_avg_local_score": summary.get("confidence_avg_local_score"),
        # The host is not the provenance. These two are why.
        "tool_used": (af_entry or {}).get("toolUsed"),
        "provider_id": (af_entry or {}).get("providerId"),
        "plddt_doc_url": (af_entry or {}).get("plddtDocUrl"),
        "pae_doc_url": (af_entry or {}).get("paeDocUrl"),
        "pae_image_url": (af_entry or {}).get("paeImageUrl"),
        "cif_url": (af_entry or {}).get("cifUrl"),
        "pdb_url": (af_entry or {}).get("pdbUrl"),
        "global_metric_value": (af_entry or {}).get("globalMetricValue"),
        "per_residue_available": bool((af_entry or {}).get("plddtDocUrl")),
    }


def _confidence_by_uniprot(doc, model_start, model_end):
    """UniProt residue number to pLDDT, or a stated reason the mapping was refused.

    The document numbers residues from the start of the model, so a UniProt number is
    `model_start + residueNumber - 1`. When the document's length disagrees with the range the
    summary claims, that arithmetic is not known to hold and the model is refused rather than
    mapped on an assumption -- a wrong offset returns a plausible score for the wrong site,
    which is undetectable downstream.
    """
    numbers = doc.get("residueNumber")
    scores = doc.get("confidenceScore")
    if not isinstance(numbers, list) or not isinstance(scores, list) or len(numbers) != len(scores):
        return None, "confidence document carries no aligned residueNumber/confidenceScore arrays"
    if model_start is None or model_end is None:
        return None, "summary declares no uniprot_start/uniprot_end to offset against"
    expected = model_end - model_start + 1
    if len(numbers) != expected:
        return None, (f"confidence document holds {len(numbers)} residues but the summary "
                      f"claims UniProt {model_start}-{model_end} ({expected}); the offset "
                      f"is not known to hold, so no score is mapped")
    if numbers and numbers[0] != 1:
        return None, (f"confidence document starts at residueNumber {numbers[0]} rather than "
                      f"1, so model-local numbering is not what this mapping assumes")
    return {model_start + n - 1: s for n, s in zip(numbers, scores)}, None


def _swissmodel_residues(accession, budget):
    """Per-residue QMEANDisCo by model identifier, from the aggregator's residue endpoint.

    Measured to return SWISS-MODEL models only, so this extends per-residue coverage to
    template-based models and to nothing else. Reported on its own scale under its own key.
    """
    payload = _get(f"{BEACONS_RESIDUES}{urllib.parse.quote(accession)}.json", budget)
    out = {}
    for structure in (payload or {}).get("structures", ()):
        summary = structure.get("summary", {})
        identifier = summary.get("model_identifier")
        if not identifier:
            continue
        mapped = {}
        for chain in structure.get("chains", ()) or ():
            for segment in chain.get("segments", ()) or ():
                start = (segment.get("uniprot") or {}).get("from")
                if start is None:
                    continue
                for offset, residue in enumerate(segment.get("residues", ()) or ()):
                    score = residue.get("confidence")
                    number = residue.get("uniprot_residue_number")
                    if score is None:
                        continue
                    mapped[number if number is not None else start + offset] = score
        if mapped:
            out[identifier] = mapped
    return out


def _range_block(interval, predicted, plddt_maps, qmean_maps, confident_plddt):
    """One requested range: which models reach it, and how well each one models it."""
    start, end = interval
    width = end - start + 1
    models = []
    for row in predicted:
        m_start, m_end = row.get("uniprot_start"), row.get("uniprot_end")
        if m_start is None or m_end is None:
            continue
        shared = _overlap(start, end, m_start, m_end)
        if not shared:
            continue
        identifier = row["model_identifier"]
        block = {"model_identifier": identifier, "provider": row["provider"],
                 "tool_used": row["tool_used"], "created": row["created"],
                 "residues_of_range_covered": shared,
                 "fraction_of_range_covered": round(shared / width, 4),
                 "confidence_type": row["confidence_type"],
                 "confidence_avg_local_score": row["confidence_avg_local_score"],
                 "per_residue": None, "per_residue_metric": None,
                 "per_residue_threshold": None, "per_residue_absent_because": None}
        if identifier in plddt_maps:
            scores = [plddt_maps[identifier][r] for r in range(start, end + 1)
                      if r in plddt_maps[identifier]]
            block["per_residue"] = _summarise_scores(scores, confident_plddt, banded=True)
            block["per_residue_metric"] = "pLDDT (0-100)"
            block["per_residue_threshold"] = float(confident_plddt)
        elif identifier in qmean_maps:
            scores = [qmean_maps[identifier][r] for r in range(start, end + 1)
                      if r in qmean_maps[identifier]]
            # Its own threshold on its own scale, and no bands. `confident_plddt` is a pLDDT
            # number: applying it unconverted would mark every residue as failing, and
            # applying PLDDT_BANDS would report a 0.93-median range as wholly `very_low`.
            block["per_residue"] = _summarise_scores(scores, confident_plddt / 100.0,
                                                     banded=False)
            block["per_residue_metric"] = "QMEANDisCo (0-1)"
            block["per_residue_threshold"] = round(confident_plddt / 100.0, 4)
        else:
            block["per_residue_absent_because"] = (
                "no per-residue document was retrieved for this model: pLDDT is reachable "
                "only through the AlphaFold host's plddtDocUrl, and QMEANDisCo only with "
                "residue_detail=True")
        models.append(block)
    models.sort(key=lambda m: (-m["fraction_of_range_covered"],
                               m["model_identifier"] or ""))
    # The best pLDDT-scored model over this range, chosen on coverage then median. Reported
    # as `best_plddt_model` and not as `the` answer: it is one model's opinion.
    scored = [m for m in models if m["per_residue_metric"] == "pLDDT (0-100)"
              and m["per_residue"]["residues_scored"]]
    best = max(scored, key=lambda m: (m["residues_of_range_covered"],
                                      m["per_residue"]["median"]), default=None)
    return {"range": f"{start}-{end}", "start": start, "end": end, "width": width,
            "models_overlapping": len(models),
            "models": models,
            "best_plddt_model": None if best is None else {
                "model_identifier": best["model_identifier"],
                "provider": best["provider"], "tool_used": best["tool_used"],
                "median_plddt": best["per_residue"]["median"],
                "min_plddt": best["per_residue"]["min"],
                "fraction_at_or_above_threshold":
                    best["per_residue"]["fraction_at_or_above_threshold"],
                "longest_run_below_threshold":
                    best["per_residue"]["longest_run_below_threshold"],
                "fraction_of_range_covered": best["fraction_of_range_covered"]},
            "modelled": bool(models),
            "absence_is_recorded": not models}


def coverage(accession, residue_ranges=(), confident_plddt=70.0, residue_detail=False,
             record=True):
    """Predicted models for one accession, with a per-range confidence summary.

    `residue_ranges` are `"start-end"` strings in UniProt numbering; with none supplied the
    return is the model inventory and no confidence document is fetched. `confident_plddt` is
    the pLDDT a residue must reach to count as modelled well enough to reason about;
    `residue_detail` adds per-residue QMEANDisCo for template-based models at the cost of one
    further request that measured 2.45 MB on P0DTD1.

    An absence is recorded and charged, never inferred away: `predicted_models` empty with
    `any_prediction_exists` False is a real zero about the accession, and it is reported with
    the providers that were asked.
    """
    if not isinstance(accession, str) or not _ACCESSION.match(accession):
        raise ValueError(f"Accession {accession!r} is not a plain identifier; this tool is "
                         f"keyed on a UniProt accession and does not search by name.")
    if not isinstance(confident_plddt, (int, float)) or not 0.0 <= confident_plddt <= 100.0:
        raise ValueError(f"confident_plddt must be a pLDDT in 0-100, got {confident_plddt!r}.")
    intervals = parse_ranges(residue_ranges)
    budget = _Budget()

    summary = beacons_summary(accession, budget)
    structures = (summary or {}).get("structures", ()) or ()
    entry = (summary or {}).get("uniprot_entry", {}) or {}

    experimental, predicted_summaries = [], []
    for structure in structures:
        block = structure.get("summary", {}) or {}
        if (block.get("model_category") or "").upper() == EXPERIMENTAL_CATEGORY:
            experimental.append({"model_identifier": block.get("model_identifier"),
                                 "provider": block.get("provider"),
                                 "model_category": block.get("model_category"),
                                 "created": _date(block.get("created")),
                                 "uniprot_start": block.get("uniprot_start"),
                                 "uniprot_end": block.get("uniprot_end")})
        else:
            predicted_summaries.append(block)

    # Always asked, never conditioned on what the aggregator returned. Two reasons, both
    # found by running this. Gating the AlphaFold call on a non-empty aggregator response
    # meant an accession the aggregator knows nothing about reported
    # `alphafold_host_served_models: False` without the host ever having been asked, which is
    # an unasked absence dressed as a measured one. And it made a 3D-Beacons miss or outage
    # silently suppress the one provider that supplies per-residue pLDDT.
    af_entries = alphafold_entries(accession, budget)
    af_index = _alphafold_index(af_entries)
    predicted = [_model_row(block, af_index.get(block.get("model_identifier")))
                 for block in predicted_summaries]

    # The union, not the aggregator's view of it. A model the AlphaFold host serves but the
    # aggregator's summary omits would otherwise disappear from a payload that claims to
    # answer whether any prediction exists.
    seen = {row["model_identifier"] for row in predicted}
    for identifier, af_entry in sorted(af_index.items()):
        if identifier in seen:
            continue
        row = _model_row({"model_identifier": identifier, "provider": "AlphaFold DB",
                          "model_category": None,
                          "model_url": af_entry.get("cifUrl"),
                          "model_format": "MMCIF",
                          "created": af_entry.get("modelCreatedDate"),
                          "uniprot_start": af_entry.get("uniprotStart"),
                          "uniprot_end": af_entry.get("uniprotEnd"),
                          "confidence_type": "pLDDT",
                          "confidence_avg_local_score": af_entry.get("globalMetricValue")},
                         af_entry)
        row["discovered_via"] = "AlphaFold DB only; absent from the 3D-Beacons summary"
        predicted.append(row)

    plddt_maps, refusals, truncated = {}, [], False
    if intervals:
        wanted = [row for row in predicted if row["plddt_doc_url"]
                  and any(_overlap(s, e, row["uniprot_start"] or 0, row["uniprot_end"] or -1)
                          for s, e in intervals)]
        wanted.sort(key=lambda r: -sum(_overlap(s, e, r["uniprot_start"], r["uniprot_end"])
                                       for s, e in intervals))
        truncated = len(wanted) > MAX_CONFIDENCE_DOCS
        for row in wanted[:MAX_CONFIDENCE_DOCS]:
            doc = _get(row["plddt_doc_url"], budget)
            if doc is None:
                refusals.append({"model_identifier": row["model_identifier"],
                                 "reason": "confidence document returned 404 or an empty body"})
                continue
            mapped, why = _confidence_by_uniprot(doc, row["uniprot_start"], row["uniprot_end"])
            if mapped is None:
                row["mapping_unreliable"] = True
                refusals.append({"model_identifier": row["model_identifier"], "reason": why})
            else:
                plddt_maps[row["model_identifier"]] = mapped

    qmean_maps = _swissmodel_residues(accession, budget) if (residue_detail and intervals) else {}

    ranges_out = [_range_block(i, predicted, plddt_maps, qmean_maps, confident_plddt)
                  for i in intervals]
    dates = sorted(r["created"] for r in predicted if r["created"])

    # Grouped by metric, never merged. See the module docstring.
    by_metric = {}
    for row in predicted:
        metric = row["confidence_type"]
        value = row["confidence_avg_local_score"]
        if metric and isinstance(value, (int, float)):
            by_metric.setdefault(metric, []).append(value)

    payload = {
        "experiment": "predicted-structure-coverage",
        "status": "complete",
        "accession": accession,
        "uniprot_entry_id": entry.get("id"),
        "sequence_length": entry.get("sequence_length"),
        "requested_ranges": [f"{s}-{e}" for s, e in intervals],
        "confident_plddt": float(confident_plddt),
        "plddt_bands": [{"name": n, "min": lo, "max": hi} for n, lo, hi in PLDDT_BANDS],
        "residue_detail": bool(residue_detail),

        # The absence question, answered explicitly in both directions.
        "any_prediction_exists": bool(predicted),
        "predicted_model_count": len(predicted),
        "providers_queried": ["3D-Beacons Hub (aggregates AlphaFold DB, SWISS-MODEL, PED, "
                              "SASBDB and other registered providers)", "AlphaFold DB"],
        "providers_with_models": sorted({r["provider"] for r in predicted if r["provider"]}),
        # Queried unconditionally, so False here is a measured absence and not an unasked one.
        "alphafold_host_queried": True,
        "alphafold_host_served_models": bool(af_entries),
        "models_found_only_by_alphafold_host":
            [r["model_identifier"] for r in predicted if r.get("discovered_via")],
        "model_categories": sorted({r["model_category"] for r in predicted
                                    if r["model_category"]}),
        "tools_used": sorted({r["tool_used"] for r in predicted if r["tool_used"]}),
        "predicted_models": predicted,
        "mean_confidence_by_metric": {m: round(statistics.fmean(v), 4)
                                      for m, v in sorted(by_metric.items())},
        "ranges": ranges_out,
        "ranges_with_no_model": [r["range"] for r in ranges_out if not r["modelled"]],

        # Retrieved, partitioned off, and counted where a reader can see it.
        "excluded_experimental_count": len(experimental),
        "excluded_experimental": experimental,
        "excluded_experimental_reason":
            "Models marked EXPERIMENTALLY DETERMINED are deposited or measured structures. "
            "They are excluded from every count and every average here and belong to "
            "structures.evidence, whose count is a research-attention proxy. A prediction is "
            "not a deposition and pooling the two would turn 'nobody has solved this' into "
            "'we have a structure'.",

        "earliest_model_created": dates[0] if dates else None,
        "latest_model_created": dates[-1] if dates else None,
        "per_residue_refusals": refusals,
        "confidence_docs_truncated": truncated,
        "confidence_docs_cap": MAX_CONFIDENCE_DOCS,
        # Deterministic given the same upstream records -- measured identical across five
        # consecutive calls -- so both belong in the digested body. Wall-clock seconds
        # deliberately do not: `content_digest` strips three timestamp key names and nothing
        # else, so any per-call quantity left in the payload makes the digest identify the
        # fetch rather than the data, and the first version of this file did exactly that.
        # A digest that cannot be recomputed by refetching is a label and not a record.
        # Timing is the episode's to record, and `cost_seconds` on the Tool spec to declare.
        "network_calls": budget.calls,
        "bytes_received": budget.bytes,
        "sources": [
            {"name": "3D-Beacons Hub summary", "url": BEACONS_SUMMARY,
             "licence": "EMBL-EBI terms of use; per-provider licences apply"},
            {"name": "3D-Beacons Hub residue mappings", "url": BEACONS_RESIDUES,
             "licence": "EMBL-EBI terms of use; SWISS-MODEL data CC BY-SA 4.0",
             "requested": bool(residue_detail and intervals)},
            {"name": "AlphaFold DB prediction", "url": ALPHAFOLD,
             "licence": "EMBL-EBI terms of use; AlphaFold DB data CC BY 4.0"},
        ],
        "urls_hit": budget.urls,

        "limitations": [
            "A predicted model is not experimental evidence that the protein adopts the "
            "modelled conformation. It is a hypothesis about structure, and a high pLDDT is "
            "a statement about the predictor's confidence, not an observation of the fold.",
            "Nothing here bears on precedent. A predicted-model count is not an "
            "attention proxy: AlphaFold modelled entire reference proteomes irrespective of "
            "whether anyone ever studied the protein, so the count carries no signal about "
            "prior work and must never be pooled with deposited-structure counts.",
            "pLDDT (0-100) and QMEANDisCo (0-1) are different quantities, not one quantity "
            "rescaled. Averages are grouped by confidence_type and never merged, and "
            "confident_plddt is applied to pLDDT only.",
            "The AlphaFold host's name is not the provenance of what it serves. For the "
            "SARS-CoV-2 accessions used here it returns ColabFold v1.5.2 models from "
            "provider VR3D, not DeepMind AlphaFold; read tool_used and provider_id before "
            "attributing a model to AlphaFold.",
            "A template-based model is an annotation transferred from a relative's solved "
            "structure, not a measurement on this pathogen. sequence_identity and the "
            "template are what qualify it, and a coverage number does not.",
            "Confidence is reported per model over the requested range. It says whether a "
            "site is modelled well enough to reason about; it does not say the site is "
            "solvent-exposed, is a pocket, is bindable, or is accessible in vivo, and it "
            "predicts no affinity.",
            "A low-pLDDT region is a positive finding about the site, not missing data. It "
            "frequently marks genuine intrinsic disorder, which is a real answer on "
            "accessibility and often means no single conformation exists to target.",
            "Absence is recorded, not inferred. An empty predicted_models means the "
            "providers queried hold nothing, which is a finding about the accession; it does "
            "not establish that no model exists anywhere, since only the listed providers "
            "were asked.",
            "Per-residue pLDDT reaches only models the AlphaFold host serves a plddtDocUrl "
            "for; per-residue QMEANDisCo reaches only SWISS-MODEL and only with "
            "residue_detail=True. Models with neither carry a model-level average and say so "
            "in per_residue_absent_because.",
            "Band fractions are computed from confidenceScore against the declared "
            "plddt_bands. The files' own confidenceCategory letters are not read, because "
            "the vocabulary was measured to disagree between file versions: H labels 71-90 "
            "in a _v1 document and 90-100 in a _v6 one.",
            "plddt_bands apply to pLDDT only. A QMEANDisCo block carries bands=None and its "
            "own per_residue_threshold on the 0-1 scale, because banding a 0-1 score against "
            "a 0-100 vocabulary reports a well-modelled range as wholly very_low.",
            "Residue numbering is UniProt numbering, mapped through each model's "
            "uniprot_start. Models whose confidence document length disagrees with their "
            "declared range are marked mapping_unreliable and contribute no score.",
            "No PAE matrix is parsed. The PAE URLs are reported so a caller can fetch them, "
            "and inter-domain confidence is not assessed here, so a two-domain range may "
            "read as well modelled while their relative orientation is unresolved.",
        ],
        "interpretation":
            "Predicted structural models for one accession, aggregated across 3D-Beacons "
            "providers and summarised per requested residue range on each model's own "
            "confidence metric. Answers whether a site is modelled well enough to reason "
            "about, and records a real zero when no provider models it. Experimentally "
            "determined entries are partitioned out and counted separately. Nothing here is "
            "evidence that the protein adopts the modelled conformation, and nothing here "
            "bears on precedent.",
        "provenance": provenance(),
    }
    payload["content_sha256"] = content_digest(payload)
    if record:
        write_json(ROOT / f"data/processed/predicted_structure/{accession}.json", payload)
    return payload
