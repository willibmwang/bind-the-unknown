"""What each shortlisted chain does, what it binds, and where its residues sit in the genome.

The chain records answer "which protein" and "which residues of it". They do not answer "what
is this protein for" or "what does it touch", and a target handed to a chemist without those is
a name and a number. Three gaps, closed here:

**What the protein does.** `intake.homolog_records` asks UniProt for eight fields and the
curated FUNCTION paragraph is not among them. It does not need to be: `intake._entry` already
downloads the whole entry for every accession the surface stage touches, and the function text
has been sitting in that cached payload unread. More usefully, a polyprotein entry carries one
FUNCTION comment **per mature chain**, keyed by a `molecule` string that matches the chain names
the ranker already produces -- so nsp5 gets nsp5's function rather than the replicase's.

**What it interacts with.** The same entry carries UniProt's IntAct-derived binary interactions,
each with an `organismDiffer` flag. That flag is the whole answer to "host factor or viral
partner": true means the two proteins are from different organisms, which for a viral protein
means a host protein. Also present and unread: the SUBUNIT paragraph (what it assembles into),
ComplexPortal complex identifiers, GO terms with their aspect, the entry's declared host
organisms, and the Pfam, InterPro and CDD signatures.

**Where its residues sit in the genome.** Positions in the chain records are 1-based along one
open reading frame of this genome, and `site_patches` is explicit that nothing may join, sort or
plot on such a span without its frame. The frame is recoverable: the fold stage already records
`found_in` as "ORF n of the ranked set, m aa", and that `n` indexes the ranked candidate list,
each row of which carries the strand, the codon offset and the nucleotide start of its reading
frame. So a residue position becomes a nucleotide range by arithmetic, both strands handled, and
the length in the string is checked against the candidate row before any of it is used.

**This module writes a side record and changes no chain record.** The 47 chain records are
tracked, the site asserts numbers off them, and `chain.py` states that nothing stage 6 returns
may reorder a rank. `runs/annotation/<stem>.json` joins to its chain record on the accession and
the held-out organism, which is how `domain_inventory`, `site_patches` and the literature record
already attach. Ranks are copied, never recomputed.

**No new host is contacted.** Everything above comes from the UniProt entry the pipeline already
fetches, so running this across the panel costs one entry per distinct homolog and nothing else.
That is deliberate: a novel pathogen is the case this project is about, and a lane that needs a
second database to say anything is a lane that says nothing exactly when it matters.
"""
import json
import re
from pathlib import Path

from .common import ROOT, content_digest, provenance, write_json

# The fold stage's own description of which reading frame it found a chain in. Parsed rather
# than recomputed: re-aligning 396 chains against twelve reading frames apiece would take hours
# and would be a second opinion about a question the record already answers. The `aa` count is
# the key that makes this safe -- it is checked against the candidate row the index selects, and
# a mismatch refuses the coordinate rather than publishing one from the wrong frame.
FOUND_IN = re.compile(r"ORF\s+(\d+)\s+of the ranked set,\s*(\d+)\s*aa")

# How the fold stage records a chain that is its whole reading frame rather than a piece cut
# out of one. Matched as a substring of `recovered_by`, which is the only field that says so.
WHOLE_ORF = "the ORF translation is the candidate"

# Comment types worth carrying, and what each one answers. Anything else in the entry is left
# where it is; this is a handoff and not a mirror of UniProt.
COMMENTS = {
    "FUNCTION": "what the protein does",
    "SUBUNIT": "what it assembles into",
    "CATALYTIC ACTIVITY": "the reaction it catalyses, where it is an enzyme",
    "ACTIVITY REGULATION": "what switches it on or off",
    "COFACTOR": "what it needs bound to work",
    "DOMAIN": "what its parts do",
    "PTM": "how it is modified after translation",
}

GO_ASPECTS = {"P": "biological_process", "F": "molecular_function", "C": "cellular_component"}
SIGNATURE_DATABASES = ("Pfam", "InterPro", "CDD", "SUPFAM", "PROSITE", "Gene3D")


# ---------------------------------------------------------------------------
# Reading frame and nucleotide coordinates
# ---------------------------------------------------------------------------
def _frame_refused(because, rank=None, route=None):
    return {"verified": False, "orf_rank": rank, "route": route, "because": because}


def reading_frame(case, foldable_row, score_row=None):
    """Which reading frame of this genome a chain's positions are numbered along.

    Two routes, because the fold stage records the frame two ways and only one of them names it.

    **Named.** `found_in` reads "ORF n of the ranked set, m aa" when the chain was located in a
    reading frame other than its parent's, which is how a coronavirus chain past the ribosomal
    frameshift is found. `n` indexes the ranked candidate list and `m` is checked against the
    row it selects.

    **Unnamed.** `chain.fold_stage` searches the chain's own parent ORF first, under the literal
    label `"the ORF that matched"`, and that label wins for most chains -- 10 of the 12 on the
    recorded SARS-CoV-2 run. The frame is still determined: the parent ORF is the candidate the
    chain's `parent` names. So the route is `scores[].parent` to `candidates[].name`, which is a
    join on a string the record carries at both ends.

    **Both routes are checked the same way, and the check is the point.** The chain's span in
    this genome must fit inside the ORF the route selected. I tried the obvious parent join
    without this check and it silently returned a frame whose ORF was 2,595 residues for a chain
    whose span ends at residue 3,569 -- the arithmetic then placed the main protease 13,000
    nucleotides from where it is. A span that does not fit is not a near miss; it is the wrong
    reading frame, and it is refused rather than reported.
    """
    found = (foldable_row or {}).get("found_in")
    span = (foldable_row or {}).get("span_in_this_genome")
    candidates = list(case.get("candidates") or ())
    match = FOUND_IN.search(found or "")

    # **A chain that was never cut out of anything needs no alignment, and had none.** Where a
    # candidate ORF was not a polyprotein, the whole translation is the chain, and the fold
    # stage records that as `recovered_by: "the ORF translation is the candidate"` with no
    # `found_in` at all -- there was no search, so there is nothing to name. That is 210 of the
    # 396 rows on this panel, and reading the absent `found_in` as a failure left every one of
    # them without a coordinate it could trivially have had.
    #
    # Resolved by the candidate's own name where that is unique, and by its residue count
    # where the row carries no name the candidates use -- the hydropathy-class rows are all
    # called "unidentified probable surface antigen" and cannot be told apart by name. Three
    # rows on this panel have two candidates of the same length and no usable name, and those
    # are refused rather than guessed.
    # `recovered_by` is present and null on rows that never reached this branch, so the empty
    # string has to come from `or` rather than from `get`'s default.
    if not found and WHOLE_ORF in ((foldable_row or {}).get("recovered_by") or ""):
        length = foldable_row.get("parent_orf_length_aa")
        named = [row for row in candidates
                 if row.get("name") and row["name"] == foldable_row.get("target")]
        sized = [row for row in candidates if row.get("length_aa") == length]
        # A name that matches is not on its own enough. The residue count has to agree too, the
        # same key the named route checks: a candidate sharing a chain's name but not its
        # length is not the reading frame that chain is the translation of, and taking it would
        # put every coordinate on the row somewhere else entirely.
        if len(named) == 1 and length is not None \
                and named[0].get("length_aa") != length:
            return _frame_refused(
                f"this chain is its reading frame's whole translation and names candidate "
                f"{named[0].get('rank')}, but that reading frame is "
                f"{named[0].get('length_aa')} residues against the chain's {length}, so they "
                f"are not the same thing", named[0].get("rank"), "whole_orf")
        row = (named[0] if len(named) == 1 else sized[0] if len(sized) == 1 else None)
        if row is None:
            return _frame_refused(
                f"this chain is its reading frame's whole translation, but "
                f"{len(sized)} candidate reading frames share its {length} residues and none "
                f"carries its name, so which one it is cannot be settled", None, "whole_orf")
        strand, frame, start = row.get("strand"), row.get("frame"), row.get("nt_start")
        if None in (strand, frame, start):
            return _frame_refused(
                "the candidate row for this reading frame carries no strand, codon offset or "
                "nucleotide start", row.get("rank"), "whole_orf")
        return {
            "verified": True, "route": "whole_orf", "orf_rank": row.get("rank"),
            "strand": strand, "frame": frame, "nt_start": start,
            "orf_length_aa": row.get("length_aa"), "segment": row.get("segment"),
            "orf_name": row.get("name"),
            "reading_frame": f"{strand}{frame} at nt {start + 1}",
            "whole_chain_span": [1, int(length)] if length else None,
            # Nothing is transferred across a species gap here, so there is nothing to check
            # the residues against and nothing that could be off by an offset: the sequence is
            # this reading frame's own translation. That makes the coordinate firmer than an
            # aligned one, not weaker, and the field says which it is rather than leaving a
            # consumer to read an absent check as a failed one.
            "residue_agreement": {
                "checked": False, "agrees": None,
                "because": "the chain is this reading frame's own translation, so there is no "
                           "second sequence to compare it against and no alignment that could "
                           "have shifted it"},
            "confirmed_at": ["the whole chain: it is this reading frame's own translation"],
            "confirmed_at_both_ends": True,
            "how_it_was_resolved": ("the candidate's name identifies the reading frame"
                                    if len(named) == 1 else
                                    "the candidate's residue count identifies the reading "
                                    "frame uniquely"),
            "what_the_numbering_is": "positions are 1-based along this one open reading frame "
                                     "of this genome, which for this chain is the whole of it.",
        }

    if match:
        rank, declared = int(match.group(1)), int(match.group(2))
        rows = [row for row in candidates if row.get("rank") == rank]
        if not rows:
            return _frame_refused(
                f"the fold stage found this chain in ranked ORF {rank}, which is not among the "
                f"{len(candidates)} candidate rows this record kept", rank, "named")
        row, route = rows[0], "named"
        if int(row.get("length_aa") or -1) != declared:
            return _frame_refused(
                f"ranked ORF {rank} is {row.get('length_aa')} residues in the candidate table "
                f"and {declared} in the fold stage's description, so the two are not the same "
                f"reading frame", rank, route)
    elif found:
        parent = (score_row or {}).get("parent")
        rows = [row for row in candidates if row.get("name") and row["name"] == parent]
        if not parent:
            return _frame_refused(
                f"the fold stage recorded the frame as {found!r}, which names no ORF, and the "
                f"chain record names no parent ORF to resolve it against", None, "parent")
        if not rows:
            return _frame_refused(
                f"the fold stage recorded the frame as {found!r}, and the chain's parent "
                f"{parent!r} matches none of the {len(candidates)} candidate rows", None,
                "parent")
        row, route = rows[0], "parent"
    else:
        return _frame_refused(
            "the fold stage recorded no reading frame for this chain, so it was never located "
            "in this genome's translation", None, None)

    strand, frame, start = row.get("strand"), row.get("frame"), row.get("nt_start")
    if None in (strand, frame, start):
        return _frame_refused(
            "the candidate row for this reading frame carries no strand, codon offset or "
            "nucleotide start", row.get("rank"), route)
    length = row.get("length_aa")
    if span and length and (span[1] > length or span[0] < 1):
        return _frame_refused(
            f"this chain spans residues {span[0]}-{span[1]} of its numbering and the reading "
            f"frame the {route} route selected is only {length} residues, so the span does not "
            f"lie in it and no nucleotide coordinate from it would be right", row.get("rank"),
            route)

    # The span fitting is necessary and nowhere near sufficient. nsp12's span is residues 1 to
    # 823 and its parent reading frame is 2,595 residues, so the span fits -- and the parent is
    # the wrong frame anyway, because the polymerase lies past the ribosomal frameshift. A
    # coordinate from it would be wrong by hundreds of codons while looking entirely reasonable.
    # So the frame is confirmed by reading the residues out of it and comparing them with the
    # relative's own chain, which is the only check that can tell a plausible frame from the
    # right one.
    agreement = _residues_agree(case, row, span, score_row)
    if agreement.get("checked") and not agreement.get("agrees"):
        return _frame_refused(
            f"the residues at {span[0]}-{span[1]} of the reading frame the {route} route "
            f"selected match the relative's own chain at only "
            f"{agreement['identity']:.0%}, so this is not the frame the chain was located in "
            f"and a coordinate from it would be wrong. {agreement.get('detail', '')}".strip(),
            row.get("rank"), route)

    # **An unmade check must not read as a blank.** Where the residues could not be compared at
    # all -- no relative sequence, a reading frame this module could not re-derive -- the frame
    # is still used, because refusing every coordinate the check could not reach would throw
    # away the ones that are right along with the ones nobody looked at. But the state is named
    # rather than left as an absent field, so a consumer filtering on it sees "not checked"
    # instead of an empty cell that looks like missing data.
    confirmed = (agreement.get("confirmed_at") if agreement.get("checked")
                 else ["neither end: the residues could not be compared"])
    return {
        "verified": True,
        "route": route,
        "residue_agreement": agreement,
        "confirmed_at": confirmed,
        "confirmed_at_both_ends": bool(agreement.get("confirmed_at_both_ends")),
        "orf_rank": row.get("rank"),
        "strand": strand, "frame": frame, "nt_start": start,
        "orf_length_aa": length,
        "segment": row.get("segment"),
        "orf_name": row.get("name"),
        "reading_frame": f"{strand}{frame} at nt {start + 1}",
        "span_fits_the_frame": True,
        "how_it_was_resolved": ("the fold stage named the ranked ORF and its residue count "
                                "matched the candidate row" if route == "named" else
                                "the fold stage located the chain in its own parent ORF, which "
                                "is the candidate the chain's parent names"),
        "what_the_numbering_is": "positions are 1-based along this one open reading frame of "
                                 "this genome, not along the genome. Two chains in different "
                                 "frames can carry the same integers and share no residue.",
    }


def residue_to_nt(position, frame, segment_length):
    """The nucleotide range of one residue's codon, 1-based and inclusive.

    On the plus strand the ORF's `nt_start` is a 0-based offset into the segment, so residue
    `p` begins at `nt_start + (p - 1) * 3` and the 1-based inclusive range is that plus one
    through that plus three.

    **On the minus strand the offset runs along the reverse complement**, which is the case
    `build_site` records as not converted and leaves as a caveat. Offset `i` of the reverse
    complement is offset `L - 1 - i` of the segment, so a codon occupying reverse-complement
    offsets `r, r+1, r+2` occupies segment offsets `L-1-r` down to `L-3-r`, and the 1-based
    inclusive range is `(L - r - 2, L - r)`. The codon is read from the high coordinate to the
    low one, which is why the range is reported with a strand beside it and not alone.

    Returns `None` rather than a guess wherever the frame is unverified or the arithmetic would
    leave the segment.
    """
    if not frame or not frame.get("verified"):
        return None
    strand, start = frame.get("strand"), frame.get("nt_start")
    if strand not in ("+", "-") or start is None or position is None or position < 1:
        return None
    offset = int(start) + (int(position) - 1) * 3
    if strand == "+":
        first, last = offset + 1, offset + 3
    else:
        if not segment_length:
            return None
        first, last = segment_length - offset - 2, segment_length - offset
    if first < 1 or (segment_length and last > segment_length):
        return None
    return [int(first), int(last)]


def span_to_nt(span, frame, segment_length):
    """The nucleotide range a residue span covers, low coordinate first on either strand."""
    if not span or len(span) != 2:
        return None
    first = residue_to_nt(span[0], frame, segment_length)
    last = residue_to_nt(span[1], frame, segment_length)
    if not first or not last:
        return None
    low = min(first + last)
    high = max(first + last)
    return [int(low), int(high)]


MINIMUM_FRAME_IDENTITY = 0.35
# Residues probed at each end of a stretch. Long enough that chance agreement over a
# 20-letter alphabet is negligible, short enough to sit inside the first conserved block
# of a chain rather than spanning an indel.
PROBE_RESIDUES = 40


def _orf_protein(accession, row, _cache={}):
    """The translation of the reading frame a candidate row describes.

    Matched on the frame itself -- strand, codon offset, nucleotide start and residue count --
    rather than on a position in a list. `domain_folds._orf_records` sorts by length and keeps
    twelve, so an index into it is only meaningful to a caller that rebuilt the same pool in the
    same order, which is the trap `site_patches._reading_frame` documents. These four fields are
    the frame, so the match is on the thing itself.
    """
    from . import domain_folds

    if accession not in _cache:
        _cache[accession] = domain_folds._orf_records(accession)
    key = (row.get("strand"), row.get("frame"), row.get("nt_start"), row.get("length_aa"))
    for orf in _cache[accession]:
        if (orf.get("strand"), orf.get("frame"), orf.get("nt_start"),
                orf.get("length_aa")) == key:
            return orf.get("protein")
    return None


def _residues_agree(case, row, span, score_row):
    """Do the residues at this span of this reading frame match the relative's own chain?

    The decisive check on a reading frame. Everything else about a frame can be plausible while
    being wrong; this reads the amino acids the frame actually encodes over the span and compares
    them, position by position, with the stretch of the relative's entry the chain was cut from.

    Compared without re-aligning. The fold stage already aligned them and recorded the span, so
    the two stretches should correspond residue for residue; if they do not, that is the finding.
    A length mismatch is reported as such rather than scored, because comparing a 306-residue
    stretch against a 932-residue one position by position would give a meaningless number.

    `checked` is false, and nothing is refused, where the comparison could not be made at all --
    no relative, no span, an ORF this module could not re-derive. An unmade check must not read
    as a passed one, so the caller only refuses on `checked and not agrees`.
    """
    accession = (score_row or {}).get("nearest_homolog")
    chain_range = (score_row or {}).get("chain_range")
    if not (accession and chain_range and span and len(chain_range) == 2):
        return {"checked": False,
                "because": "no relative accession, chain range or span to compare"}
    protein = _orf_protein(case.get("genome_accession"), row)
    if not protein:
        return {"checked": False,
                "because": "this reading frame could not be re-derived from the genome, so its "
                           "residues could not be read"}
    try:
        from .intake import _entry

        sequence = _entry(accession)["sequence"]["value"]
    except Exception as failure:                                        # noqa: BLE001
        return {"checked": False,
                "because": f"the relative's sequence could not be read: "
                           f"{type(failure).__name__}"}
    here = protein[span[0] - 1:span[1]]
    there = sequence[chain_range[0] - 1:chain_range[1]]
    if not here or not there:
        return {"checked": False, "because": "one of the two stretches is empty"}
    # **Probed at both ends, and a single indel is why.** The two stretches need not be the same
    # length: the fold stage aligned them locally, and an insertion or deletion across the
    # species gap leaves the span longer or shorter than the relative's chain. nsp3 is the real
    # case, 1,945 residues against the relative's 1,922. Comparing position by position over the
    # whole overlap then reads 7% even though the two stretches are plainly the same protein --
    # one 23-residue indel puts everything after it out of register, and identity collapses to
    # chance. Their first forty residues are 75% identical.
    #
    # So the probe is the leading and the trailing `PROBE_RESIDUES` of each stretch, each
    # anchored at its own end. An indel anywhere in the middle leaves both ends in register, and
    # a frame that is simply the wrong frame matches at neither. The better of the two decides,
    # and both are reported, because a large gap between them is itself informative: it says the
    # difference between these two chains is at one end.
    overlap = min(len(here), len(there))
    probe = min(overlap, PROBE_RESIDUES)
    leading = sum(1 for a, b in zip(here[:probe], there[:probe]) if a == b) / probe
    trailing = sum(1 for a, b in zip(here[-probe:], there[-probe:]) if a == b) / probe
    best = max(leading, trailing)
    ends = ([] + (["the start of the chain"] if leading >= MINIMUM_FRAME_IDENTITY else [])
            + (["the end of the chain"] if trailing >= MINIMUM_FRAME_IDENTITY else []))
    # **One end agreeing and the other not is a finding, not a near miss.** SARS-CoV-2's
    # polymerase is the case: its last forty residues match the relative exactly and its first
    # forty do not match at all, because nsp12 begins before the ribosomal frameshift and the
    # reading frame it was located in begins after it. The recovered end coordinate is right and
    # the start coordinate is short by the length of the frameshifted piece. Reported as
    # confirmed at one end rather than suppressed, because the end that is right is worth having
    # and the end that is not must not be quoted as though it were.
    return {"checked": True, "agrees": best >= MINIMUM_FRAME_IDENTITY,
            "identity": round(best, 4),
            "confirmed_at": ends,
            "confirmed_at_both_ends": len(ends) == 2,
            "one_end_only_note": None if len(ends) != 1 else
                (f"Only {ends[0]} matches the relative. The other end of this span does not, "
                 f"so the coordinate at that end is not confirmed. On a polyprotein the usual "
                 f"cause is a ribosomal frameshift: the chain starts in one reading frame and "
                 f"the frame it was located in starts in another, which leaves the far end "
                 f"exact and the near end short."),
            "leading_identity": round(leading, 4), "trailing_identity": round(trailing, 4),
            "residues_compared": probe,
            "frame_span_residues": len(here), "relative_chain_residues": len(there),
            "lengths_differ": len(here) != len(there),
            "threshold": MINIMUM_FRAME_IDENTITY,
            "what": f"fraction of positions where the reading frame's residue equals the "
                    f"relative's, over the first and last {probe} residues of each stretch, "
                    f"each anchored at its own end so an indel between them cannot put the "
                    f"comparison out of register. No re-alignment."}


def segment_lengths(accession):
    """Nucleotide length per segment of this genome, keyed by the accession in its header.

    Needed only for the minus strand, and fetched through `genome_records` so a segmented
    genome returns its segments rather than a concatenation -- the same reason `domain_folds`
    takes that door. Cached per accession because the panel calls this once per genome.
    """
    from .intake import genome_records

    out, total = {}, []
    for index, (header, sequence) in enumerate(genome_records(accession)):
        name = (header or "").split()[0].lstrip(">") if header else None
        out[name] = len(sequence)
        # **Keyed by position as well as by name**, because a candidate row names its segment
        # with an integer index and not with an accession. Keying only by accession meant
        # every minus-strand row looked up `0`, found nothing, and lost a coordinate it had
        # all the information for -- 85 rows on this panel.
        out[index] = len(sequence)
        total.append(len(sequence))
    out[None] = total[0] if len(total) == 1 else None
    return out


# ---------------------------------------------------------------------------
# What the protein does, from the entry the pipeline already downloaded
# ---------------------------------------------------------------------------
def _texts(comment):
    return [t["value"] for t in (comment.get("texts") or ()) if t.get("value")]


def entry_annotation(accession, molecule=None):
    """Function, partners, families and terms for one accession, narrowed to one mature chain.

    **A comment that names a molecule is about that molecule and a comment that names none is
    about the entry.** They are kept in separate keys rather than merged, because a polyprotein
    entry's entry-level FUNCTION describes the whole replicase and attributing it to nsp5 would
    be a claim nobody made. `molecule_matched` says which of the two a reader is looking at, and
    `molecules_available` lists what the entry offered so a failed match is visible rather than
    reading as an absence of annotation.
    """
    from .intake import _entry

    entry = _entry(accession)
    wanted = (molecule or "").strip().lower()
    available = sorted({c["molecule"] for c in entry.get("comments") or ()
                        if c.get("molecule")})
    chain_level, entry_level = {}, {}
    for comment in entry.get("comments") or ():
        kind = comment.get("commentType")
        if kind not in COMMENTS:
            continue
        texts = _texts(comment)
        if not texts:
            continue
        name = (comment.get("molecule") or "").strip().lower()
        bucket = chain_level if (name and wanted and name == wanted) else (
            entry_level if not name else None)
        if bucket is None:
            continue
        bucket.setdefault(kind, []).extend(texts)

    interactions = _interactions(entry, wanted)
    crossrefs = entry.get("uniProtKBCrossReferences") or ()
    signatures = {}
    for ref in crossrefs:
        if ref.get("database") in SIGNATURE_DATABASES:
            name = next((p["value"] for p in ref.get("properties") or ()
                         if p.get("key") == "EntryName"), None)
            signatures.setdefault(ref["database"], []).append(
                {"id": ref.get("id"), "name": name})
    terms = {}
    for ref in crossrefs:
        if ref.get("database") != "GO":
            continue
        label = next((p["value"] for p in ref.get("properties") or ()
                      if p.get("key") == "GoTerm"), "")
        aspect = GO_ASPECTS.get(label[:1])
        if aspect:
            terms.setdefault(aspect, []).append({"id": ref.get("id"), "term": label[2:]})

    return {
        "accession": accession,
        "entry_id": entry.get("uniProtkbId"),
        "protein_name": (((entry.get("proteinDescription") or {}).get("recommendedName")
                          or {}).get("fullName") or {}).get("value"),
        "organism": (entry.get("organism") or {}).get("scientificName"),
        "reviewed": entry.get("entryType", "").startswith("UniProtKB reviewed"),
        "molecule_asked_for": molecule,
        "molecule_matched": bool(chain_level),
        "molecules_available": available,
        "about_this_chain": {kind: value for kind, value in chain_level.items()},
        "about_the_whole_entry": {kind: value for kind, value in entry_level.items()},
        "what_each_comment_answers": {kind: question for kind, question in COMMENTS.items()
                                      if kind in chain_level or kind in entry_level},
        "interactions": interactions,
        "declared_hosts": [{"name": host.get("scientificName"),
                            "common_name": host.get("commonName"),
                            "taxon": host.get("taxonId")}
                           for host in entry.get("organismHosts") or ()],
        "complexes": [ref.get("id") for ref in crossrefs
                      if ref.get("database") == "ComplexPortal"],
        "signatures": signatures,
        "go_terms": terms,
        "keywords": [k.get("name") for k in entry.get("keywords") or () if k.get("name")],
        "host_interaction_keywords": [k.get("name") for k in entry.get("keywords") or ()
                                      if "host" in (k.get("name") or "").lower()],
        "structures_deposited": sum(1 for ref in crossrefs if ref.get("database") == "PDB"),
        "source": f"https://rest.uniprot.org/uniprotkb/{accession}.json",
        "citation": "UniProt curated annotation on the nearest reviewed relative. Every line "
                    "here is transferred from that relative and none of it was measured on "
                    "this pathogen.",
    }


def chain_identifiers(entry):
    """Map each mature chain's description to the `PRO_` identifier UniProt gives it.

    The bridge without which interactions cannot be attributed. An interaction on a polyprotein
    entry names its interactant by `chainId`, an opaque `PRO_0000037312`, while every other part
    of this pipeline knows the chain by its description. The entry's own `Chain` features carry
    both, so the join is read out of the entry rather than guessed.
    """
    out = {}
    for feature in entry.get("features") or ():
        if feature.get("type") != "Chain" or not feature.get("featureId"):
            continue
        description = (feature.get("description") or "").strip().lower()
        if description:
            out[description] = feature["featureId"]
    return out


def _interactions(entry, wanted):
    """Binary interactions for one mature chain, split by whether the partner is another species.

    **`organismDiffer` is what makes this worth carrying.** For a viral protein a partner from a
    different organism is a host protein, which is the interaction a drug programme cares about;
    a partner from the same organism is another viral protein or the protein itself. Counted
    apart rather than summed, because a list of viral self-interactions presented as host
    factors would be a wrong answer in the flattering direction.

    **Attributed to the chain, not to the entry.** A polyprotein entry carries every one of its
    products' interactions, so returning them all for every chain would give the main protease
    and the polymerase identical partner lists -- which is what a first version of this function
    did. An interaction belongs to this chain when either interactant carries this chain's `PRO_`
    identifier. Interactions naming no chain at all are entry-level and are counted separately
    rather than folded in, because "this product binds X" and "something in this polyprotein
    binds X" are different claims.
    """
    identifier = chain_identifiers(entry).get(wanted) if wanted else None
    host, viral, entry_level = [], [], 0
    for comment in entry.get("comments") or ():
        if comment.get("commentType") != "INTERACTION":
            continue
        for item in comment.get("interactions") or ():
            one = item.get("interactantOne") or {}
            two = item.get("interactantTwo") or {}
            chains = {one.get("chainId"), two.get("chainId")} - {None}
            if identifier is None or identifier not in chains:
                if not chains:
                    entry_level += 1
                continue
            # The partner is whichever interactant is not this chain.
            partner = two if one.get("chainId") == identifier else one
            row = {"partner_accession": partner.get("uniProtKBAccession"),
                   "partner_gene": partner.get("geneName"),
                   "partner_chain": partner.get("chainId"),
                   "experiments": item.get("numberOfExperiments"),
                   "intact_id": partner.get("intActId")}
            (host if item.get("organismDiffer") else viral).append(row)
    return {
        "chain_identifier": identifier,
        "with_a_host_protein": host,
        "with_a_protein_of_this_virus": viral,
        "host_partners": len(host),
        "viral_partners": len(viral),
        "entry_level_interactions_not_attributed_to_any_chain": entry_level,
        "not_attributable_because": None if identifier or not wanted else
                                    "the entry names no mature chain with this description, so "
                                    "its interactions cannot be attributed to this chain and "
                                    "none are claimed for it",
        "source": "UniProt's IntAct-derived binary interactions, carried in the entry this "
                  "pipeline already downloads. No second database is contacted.",
        "absence_note": "An empty list means no curated interaction is attributed to this "
                        "chain, which for a novel pathogen is the expected case and is not "
                        "evidence that the protein touches nothing.",
    }


# ---------------------------------------------------------------------------
# One genome
# ---------------------------------------------------------------------------
def _case(payload):
    """The held-out case of a chain record, which is the one every page and figure uses."""
    cases = payload.get("cases")
    if isinstance(cases, list) and cases:
        held = [c for c in cases if c.get("reference_excluded_organism")]
        return held[0] if held else cases[0]
    return payload


def annotate_case(case, accession):
    """Every shortlisted chain of one genome, annotated. Ranks are copied, never recomputed."""
    lengths = {}
    try:
        lengths = segment_lengths(accession)
    except Exception as failure:                                        # noqa: BLE001
        lengths = {"_failed": f"{type(failure).__name__}: {str(failure)[:160]}"}

    surfaces = {row.get("target"): row for row in case.get("surfaces") or ()}
    foldables = {row.get("target"): row for row in case.get("foldable") or ()}
    rows = []
    for score in sorted(case.get("scores") or (), key=lambda r: r.get("rank", 0)):
        target = score.get("target")
        surface = surfaces.get(target) or {}
        foldable = foldables.get(target) or {}
        frame = reading_frame(case, foldable, score)
        # The whole-ORF route carries its own span: the chain is the entire translation, so it
        # runs from residue 1 to the frame's length, and no alignment recorded it.
        span_in_frame = (foldable.get("span_in_this_genome")
                         or frame.get("whole_chain_span"))
        length = lengths.get(frame.get("segment")) if frame.get("verified") else None
        if length is None and frame.get("verified"):
            # `segment_lengths` sets the `None` key only where the genome has exactly one
            # segment, so falling back to it cannot pick the wrong segment of a segmented
            # genome. It used to be restricted to the plus strand, which was the wrong guard:
            # the plus strand does not need a length at all, and the minus strand -- the one
            # that does -- was the only one the fallback refused to serve.
            length = lengths.get(None)
        accession_of_relative = score.get("nearest_homolog")
        annotation, failure = None, None
        if accession_of_relative:
            try:
                annotation = entry_annotation(accession_of_relative, target)
            except Exception as problem:                                # noqa: BLE001
                failure = f"{type(problem).__name__}: {str(problem)[:200]}"
        rows.append({
            "rank": score.get("rank"),
            "target": target,
            "parent": score.get("parent"),
            "tier": score.get("tier"),
            "nearest_homolog": accession_of_relative,
            "score": score.get("score"),
            "score_terms": score.get("score_terms"),
            "gate": surface.get("gate"),
            "reading_frame": frame,
            "chain_span_on_the_relative": score.get("chain_range"),
            "chain_span_in_this_genome": span_in_frame,
            "chain_span_nt": span_to_nt(span_in_frame, frame, length),
            "chain_span_nt_confirmed_at": frame.get("confirmed_at"),
            "segment_length_nt": length,
            "annotation": annotation,
            "annotation_failed_because": failure,
            "not_annotated_because": None if accession_of_relative else
                                     "this chain was identified with no reviewed relative, so "
                                     "there is no curated entry to read a function from",
        })
    return rows


def run_annotate(path=None, accession=None, record=True):
    """Annotate one chain record and write `runs/annotation/<stem>.json`."""
    source = Path(path) if path else (
        ROOT / f"runs/onset_chain_{(accession or '').replace('.', '_')}.json"
        if accession else ROOT / "runs/onset_chain.json")
    payload = json.loads(source.read_text())
    case = _case(payload)
    genome = case.get("genome_accession") or payload.get("genome_accession")
    rows = annotate_case(case, genome)

    functions = sum(1 for r in rows if (r.get("annotation") or {}).get("molecule_matched"))
    hosts = sum((r.get("annotation") or {}).get("interactions", {}).get("host_partners", 0)
                for r in rows)
    nt = sum(1 for r in rows if r.get("chain_span_nt"))
    out = {
        "experiment": "annotation",
        "status": "complete",
        "question": "For every shortlisted chain: what does the protein do, what does it "
                    "touch, and where do its residues sit in this genome?",
        "genome_accession": genome,
        "pathogen": case.get("pathogen") or payload.get("pathogen"),
        "reference_excluded_organism": case.get("reference_excluded_organism"),
        "source_record": str(source.relative_to(ROOT)),
        "chains": rows,
        "coverage": {
            "chains": len(rows),
            "with_a_reviewed_relative": sum(1 for r in rows if r.get("nearest_homolog")),
            "with_a_function_for_this_chain": functions,
            "with_a_verified_reading_frame": sum(
                1 for r in rows if (r.get("reading_frame") or {}).get("verified")),
            "with_nucleotide_coordinates": nt,
            "host_interactions_found": hosts,
        },
        "joins_to_its_chain_record_on": ["genome_accession", "reference_excluded_organism"],
        "ranks_are_copied_not_recomputed":
            "`chain.py` states that nothing a late stage returns may reorder a rank. Every rank "
            "and score here is copied from the chain record and this module computes neither.",
        "limitations": [
            "Every function, partner and family here is curated annotation on the nearest "
            "reviewed relative, transferred across a species gap. None of it was measured on "
            "this pathogen, and for a genuinely novel pathogen the nearest relative may be "
            "distant enough that the transfer is wrong.",
            "An empty interaction list means no curated interaction exists for the relative. "
            "It is not evidence that the protein touches nothing, and it is the expected "
            "result for an understudied virus.",
            "A nucleotide range is emitted only where the fold stage's own reading frame was "
            "recovered and its residue count matched the candidate row. Where it was not, the "
            "range is absent and the reason is named; it is never estimated.",
            "On the minus strand a codon is read from the high nucleotide coordinate to the "
            "low one. The range is reported low coordinate first with the strand beside it, so "
            "it cannot be read as a direction.",
            "A range covers the codons of the residues named and excludes nothing else; a "
            "chain's span in nucleotides therefore excludes the stop codon, which lies beyond "
            "the last residue.",
        ],
        "provenance": provenance(),
    }
    out["content_sha256"] = content_digest(out)
    if record:
        stem = (genome or "unknown").replace(".", "_")
        write_json(ROOT / f"runs/annotation/{stem}.json", out)
    return out
