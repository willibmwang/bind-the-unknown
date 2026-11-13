"""Fold one amino-acid sequence with ESMFold, keyed on the sequence and not on an accession.

Every structural lookup this repository already makes is keyed on an accession.
`structures_evidence.build` resolves a UniProt entry and asks PDBe what covers it; `intake`
asks UniProt for chain locations. That shape has a floor built into it, and `selector.FACTS`
states the floor out loud, under `protein_accession`: "a sequence is not a substitute, because
the endpoints this repository reaches resolve accessions and not sequences."

This module exists because that sentence is a statement about the endpoints that were reached,
not about the world. A genuinely unknown protein -- the premise of the whole project -- has no
reviewed accession, therefore no PDBe coverage and no AlphaFold DB entry, and every
accession-keyed lane returns a true and useless zero. It does have a sequence. ESMFold folds a
sequence with no MSA and no template, so it is the one structural lane whose input a novel
pathogen can actually supply. The sentence in FACTS should be amended rather than deleted: a
sequence is still not a substitute for an accession *for the endpoints named there*, and it is
now sufficient for this one.

What it is not, stated before anything else because the coverage is the easy part:

**A prediction is not a structure.** Nothing here was measured. `structures.evidence` returns
deposition dates for experiments somebody performed; this returns coordinates a model emitted.
The two must never be pooled into one structural count, and a reader who sees "has a structure"
for a target whose only structure came from this module has been misled.

**pLDDT is the model's confidence, not its accuracy.** It is calibrated on the kind of protein
the model was trained on. ESMFold runs single-sequence: for a sequence with few homologs --
which is the case this module was written for -- accuracy is lower *and* the calibration of
pLDDT is itself less trustworthy, so a low number on a novel pathogen protein is the expected
reading rather than a malfunction, and a high number is weaker evidence than the same number on
a well-represented family. Reporting the mean alone would hide both facts, so the payload
carries the per-residue vector, the confident fraction, and the low-confidence runs.

**The neighbour count is a burial proxy and is not SASA.** No solvent model, no sidechain
packing past CB, no probe radius. It is a count of CB atoms inside a radius, which the
literature uses as a contact number and which this module labels a proxy everywhere it appears.
It is computed on predicted coordinates, so it inherits the prediction's error, and over a
low-pLDDT stretch it is not interpretable at all.

**One monomer, one conformation, in vacuum.** No partners, no glycans, no membrane, no ligand,
no ensemble. For a viral glycoprotein the biological unit is usually a trimer, so a surface this
module scores as least-buried may be an oligomeric interface that is never solvent-exposed in
the assembled particle -- which is the precise question `accessibility` asks, answered wrongly.
A fragment's fold in isolation is also not its fold in the chain it was cut from.

Three decisions about how it fails are worth defending.

**The 400-residue ceiling is enforced locally and returns an absence, not an exception.** The
ceiling is measured, not read off documentation: 400 residues returns 200, and 401 returns HTTP
413 with the body `Sequence is longer than 400.`. Spending a request to be told a deterministic
no is waste, so the check runs first. It returns a recorded absence rather than raising because
a 602-residue Nipah G glycoprotein being out of range is a *finding* about what this lane can
reach -- `chief`'s rule that a missing backend "returns an explicit absence, charged" -- whereas
an exception would read as a caller error and invite a retry that cannot succeed.

**The endpoint being down also returns an absence, under a reason code that cannot be confused
with the first.** This host has been unreliable historically. Conflating "we could not fold
this" with "this target is unfoldable" is the one conflation that would do real damage here, so
`absence["reason"]` distinguishes `sequence_too_long` from `endpoint_unavailable`, and a caller
that reads the boolean alone and not the reason gets a conservative answer either way.

**`timeout` and `retries` are module constants and deliberately not parameters.** The wall gate
refuses a call whose `worst_case_seconds` exceeds the budget, and a declared worst case is only
sound if the caller cannot move the quantity it is computed from. Exposing a timeout would let
one argument falsify the spec's own gate, which is the same defect class as a tool declaring
zero network calls while reaching four hosts.

Measured behaviour, so the spec is written from observation rather than from assumption:

* Cold (cache-miss) wall time scales steeply with length -- 1.8 s at 50 residues, 5.0 s at 200,
  12.1 s at 300, 25.6 s at 400 -- and the host caches, so the *second* call on the same sequence
  returns in about 0.7 s. SARS-CoV-2 Mpro (306 aa) took 12.8 s cold and 0.64 s warm. The cold
  number is the one to budget: in this project every target sequence is novel on first sight,
  and declaring the warm number would certify a cost no real campaign pays.
* The B-factor column carries pLDDT on a **0-1** scale, not the conventional 0-100. This module
  multiplies by 100 and says so in `plddt_scale`; reading the column as 0-100 would report every
  residue as catastrophically unconfident.
* The accepted alphabet is the 20 standard residues plus the ambiguity codes B, J, X and Z.
  `U` (selenocysteine) and `O` (pyrrolysine) are refused with HTTP 422 although both occur in
  real proteins.
* **The host silently drops X residues from the model.** An 11-residue input containing one X
  returns ten residues. The `resSeq` numbering is preserved against the input -- the returned
  residues were numbered 1-7, 9, 10, 11 -- so the dropped position is recoverable, which is why
  this module keys pLDDT on `resSeq` rather than on the order atoms arrive in. Anything that
  zipped the B-factors against the input sequence positionally would silently misalign every
  residue after the first X, and report the confidence of residue 9 as the confidence of
  residue 8. `unmodelled_positions` records the gap and `alignment_verified` records whether
  every returned residue's identity matches the input at the position it claims.
"""
import hashlib
import json
import os
import math
import re
import time
import urllib.error
import urllib.request

from .common import ROOT, content_digest, provenance, write_json

ENDPOINT = (os.environ.get("ESMFOLD_ENDPOINT")
            or "https://api.esmatlas.com/foldSequence/v1/pdb/")
"""ESM Atlas's keyless ESMFold v1 endpoint. POST the raw sequence as the body; it is not JSON.

Chosen over the alternatives after checking each one live on 2026-10-27. AlphaFold DB is a
lookup keyed on a UniProt accession, so it answers a different question and returns nothing for
the novel targets this lane exists for. EBI's and NVIDIA's folding services and
api.openprotein.ai all require a key or an account -- openprotein answered `RBAC: access denied`
unauthenticated, biolm.ai hung and then refused. `api.colabfold.com` is an MSA server, not a
folding service, and single-sequence folding is the property being bought here anyway. No
keyless ESMFold host other than this one survived the check, so there is no fallback to declare:
when this host is down the capability is absent, and the payload says so.
"""

USER_AGENT = "hpc-pilots/1.0 (research prototype; contact via repository)"
"""Sent on every request. `structures_evidence` sends the default urllib agent to EMBL-EBI
thirty-six times per call with no contact string, which is the politeness defect not copied."""

MAX_RESIDUES = int(os.environ.get("ESMFOLD_MAX_RESIDUES") or 400)
"""The ceiling above which a sequence is refused locally, without spending a request.

400 is the PUBLIC HOST's limit, measured rather than documented: 400 returns 200 and 401
returns HTTP 413 `Sequence is longer than 400.`. Re-checked 2026-11-09 against
api.esmatlas.com -- 401 and 600 both still 413 -- so raising this number alone does not fold
a longer chain, it only converts a free local refusal into a wasted request that fails.

It is settable because the limit belongs to that host and not to ESMFold. A local or
cluster deployment has no such ceiling, and the nine SARS-CoV-2 chains this refuses include
the top-ranked spike at 458 residues and the copying machine at 823, which is the target an
approved antiviral already hits. Point `ESMFOLD_ENDPOINT` at a deployment that can take them
and raise `ESMFOLD_MAX_RESIDUES` to match what it will actually accept; attention is
quadratic in length, so the practical ceiling is the GPU's memory rather than a constant.

Do not raise it against the public host. The refusal it produces is a recorded absence with
a stated reason, which is worth more than a 413 with the same meaning and a round trip."""

ACCEPTED = frozenset("ACDEFGHIKLMNPQRSTVWY" "BJXZ")
"""The host's own alphabet, quoted from its 422 body. `U` and `O` are absent from it."""

AMBIGUOUS = frozenset("BJXZ")
"""Accepted by the host but not a residue. X is dropped from the model outright; B, J and Z are
modelled as some concrete residue the host chooses and does not report, so a position holding
one of these is recorded in `ambiguous_positions` and its pLDDT is a confidence about a residue
identity that was never pinned down."""

REQUEST_TIMEOUT = 45
"""Seconds per request. The host sits behind an API gateway that appears to cut requests near
30 s -- the historic "Endpoint request timed out" reports cluster at 400 residues, which is
exactly where the measured cold time reaches 25.6 s -- so 45 is past the gateway's own ceiling
and a timeout here means the gateway never answered."""

LONG_REQUEST_TIMEOUT = int(os.environ.get("ESMFOLD_LONG_TIMEOUT") or 170)
"""Seconds per request to the SECOND host, which is a different machine with different
numbers and must not inherit the Atlas's.

45 seconds is right for a gateway that cuts at 30. It is wrong for a GPU deployment that
cold-starts in about 30 seconds before folding anything and then takes 37 seconds on a
932-residue chain and 109 on a 1,200. Measured end to end on 13 Nov 2026, a cold 932-residue
fold timed out at 45 s on the first attempt and only succeeded because the retry found the
container warm -- which is luck, and with `RETRIES` at 2 a slower chain would have spent both
attempts and been recorded as an absence the host never actually refused. A false absence is
the most expensive error this lane can make: it is indistinguishable from a real one.

170 sits just past the 150 seconds a Modal web endpoint will hold a connection for, so what
comes back is the HOST's own answer about being too slow rather than ours about waiting.
"""

RETRIES = 2
"""Total attempts, not additional ones. One retry, because the observed failure is a flaky host
rather than a malformed request, and more than one would push the declared worst case past what
a wall budget can sensibly admit."""

RETRY_PAUSE = 3.0
"""Backoff between attempts, and the inter-request pause this module owes a free host."""

RETRY_STATUS = frozenset({429, 500, 502, 503, 504})
"""Statuses worth a second attempt. 400, 413 and 422 are refusals of this exact body and will
be refused identically however often they are sent."""

LONG_ENDPOINT = os.environ.get("ESMFOLD_LONG_ENDPOINT")
"""A second ESMFold deployment for the chains the public host refuses on length, or None.

**The split, and why it is a split rather than a replacement.** `api.esmatlas.com` stops at
400 residues, which on the held-out SARS-CoV-2 run refuses eight of the twelve shortlisted
chains whole -- all three spike chains, the polymerase, nsp3, nsp13, nsp14 and the
nucleoprotein. A GPU deployment has no such limit. It is also not the same artifact: the same
weights on different hardware put the main protease 0.841 angstrom from the committed fold at
`runs/folded_structure/mpro_nsp5.pdb`, which this repository reproduces byte for byte against
the public host. Moving every fold to a GPU would quietly end that reproduction.

So short sequences keep going to the host the committed artifact came from, and only the
chains that host will not take are sent elsewhere. Nothing a recorded run already contains
changes; what was an absence becomes a structure.
"""

LONG_MAX_RESIDUES = int(os.environ.get("ESMFOLD_LONG_MAX_RESIDUES") or 1200)
"""What the second host can return inside one request, measured on a Modal H100 on 13 Nov
2026: 932 residues answered in 37 s and 1,200 in 109 s, while 1,500 and 1,922 both hit the
150-second limit a web endpoint holds a connection for and came back 303 to a polling URL
that a POST cannot follow. So this is a WALL-CLOCK ceiling wearing a length's clothes, and
raising it without making the host faster converts a local refusal into a lost fold. nsp3 at
1,922 residues stays above it and keeps the domain route."""

LONG_TOKEN = os.environ.get("ESMFOLD_LONG_TOKEN")
"""Bearer token for the second host, when it is deployed behind one. Absent is allowed so the
lane works against an open deployment, which is what a first bring-up produces."""

MODEL = "ESMFold v1"


def route(length):
    """Which host should take a sequence of this length, and why. Returns (endpoint, why).

    `(None, reason)` means neither host will take it, which is a real zero for the target and
    is recorded as one rather than attempted.
    """
    if length <= MAX_RESIDUES:
        return ENDPOINT, "inside the public host's ceiling"
    if LONG_ENDPOINT and length <= LONG_MAX_RESIDUES:
        return LONG_ENDPOINT, (f"over the public host's {MAX_RESIDUES}-residue ceiling and "
                               f"inside the second host's {LONG_MAX_RESIDUES}")
    return None, (f"{length} residues is over the public host's {MAX_RESIDUES}"
                  + (f" and over the second host's {LONG_MAX_RESIDUES}" if LONG_ENDPOINT
                     else " and no second host is configured"))


def _served_by(endpoint=None):
    """Who actually answered, not who usually answers.

    The citation named api.esmatlas.com unconditionally. Once ENDPOINT became settable that
    was a provenance error waiting to happen: a fold produced by a local deployment would
    have carried a citation crediting a host that never saw the sequence."""
    host = re.sub(r"^https?://([^/]+).*$", r"\1", endpoint or ENDPOINT)
    if host == "api.esmatlas.com":
        return ("Served by the ESM Metagenomic Atlas (api.esmatlas.com) under CC-BY-4.0, "
                "(c) Meta Platforms.")
    variable = ("ESMFOLD_LONG_ENDPOINT" if LONG_ENDPOINT and host == re.sub(
        r"^https?://([^/]+).*$", r"\1", LONG_ENDPOINT) else "ESMFOLD_ENDPOINT")
    return f"Served by {host}, configured through {variable}, not by the ESM Atlas."


CITATION = ("Lin et al., Evolutionary-scale prediction of atomic-level protein structure with a "
            "language model, Science 379:1123 (2023), doi:10.1126/science.ade2574; preprint "
            "doi:10.1101/2022.07.20.500902. " + _served_by() + " A prediction cited without "
            "the model and its version is not a cited claim.")

MODEL_DATE_FALLBACK = "2022-10-18"
"""Parsed from the response HEADER in practice. Every response observed carried `18-OCT-22`,
which is the ESMFold v1 weights and not the moment of the call: a prediction from a 2022 model
is evidence available from 2022 however recently it was requested. Kept as a fallback only so a
HEADER-less response does not lose its date."""

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # What the host emits for the ambiguity codes it accepts, where it emits anything at all.
    "ASX": "B", "GLX": "Z", "XLE": "J", "UNK": "X",
}

MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
          "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}

LIMITATIONS = (
    "A predicted structure is not an experimental one. Nothing in this payload was measured, "
    "and it must never be pooled with the PDBe deposition counts structures.evidence returns.",
    "pLDDT is the model's own confidence, not its accuracy, and not a measurement of anything. "
    "It is least trustworthy on exactly the sequences this lane exists for.",
    "ESMFold runs single-sequence with no MSA and no template. For a sequence with few "
    "homologs, accuracy is lower than published benchmark averages, which are dominated by "
    "well-represented families. A low pLDDT on a novel pathogen protein is the expected "
    "reading, not a defect.",
    "The neighbour count is a burial proxy, not SASA: a count of CB atoms within a radius, with "
    "no solvent model, no probe radius and no sidechain packing past CB. It is computed on "
    "predicted coordinates and inherits their error, and over a low-pLDDT stretch it carries no "
    "information at all.",
    "One monomer, one conformation, in vacuum. No partners, glycans, membrane, ligand or "
    "ensemble. For a viral glycoprotein the biological unit is usually a trimer, so a surface "
    "scored here as least-buried may be an oligomeric interface that is never solvent-exposed "
    "in the assembled particle.",
    "A fragment folded in isolation is not that fragment's fold inside the chain it was cut "
    "from, and the 400-residue ceiling forces most full-length glycoproteins to be cut.",
    "This bears on accessibility weakly, as a hypothesis about which residues are exposed in a "
    "predicted monomer. It establishes nothing whatever about essentiality, conservation, assay "
    "availability, public labels or precedent, and it is not reagent tractability: a confident "
    "fold is not evidence that the protein expresses solubly or survives immobilisation.",
    "The ESMFold v1 training corpus has its own cutoff, which this module did not verify. Any "
    "claim that a prediction for this pathogen is free of hindsight is unverified here.",
    "The host caches by sequence, so a repeated call is roughly twenty times faster than the "
    "first. Only the cold number is a cost a campaign pays.",
)


class _Opened:
    """What one HTTP attempt produced: the body, or the refusal and its exact text."""

    def __init__(self, body=None, status=None, message="", transport=""):
        self.body, self.status, self.message, self.transport = body, status, message, transport


def clean_sequence(sequence):
    """Uppercase, drop FASTA headers and whitespace, and refuse anything the host would refuse.

    Uppercasing is not cosmetic: the host answers a lowercase sequence with HTTP 422, so a
    sequence pasted out of a lowercased FASTA file would fail for a reason the error text does
    not name. Characters outside the host's alphabet raise rather than being stripped, because
    silently deleting a residue would renumber everything after it and the caller would read
    per-residue confidences against the wrong positions -- which is the same defect the host's
    own X-dropping creates and this module exists to detect.
    """
    if not isinstance(sequence, str):
        raise ValueError(f"sequence must be a str, not {type(sequence).__name__}")
    text = "\n".join(line for line in sequence.splitlines() if not line.startswith(">"))
    cleaned = re.sub(r"\s+", "", text).upper()
    if not cleaned:
        raise ValueError("sequence is empty after removing FASTA headers and whitespace; the "
                         "host answers an empty body with HTTP 400 'Sequence is empty.'")
    bad = sorted(set(cleaned) - ACCEPTED)
    if bad:
        raise ValueError(
            f"sequence holds {bad!r}, which api.esmatlas.com refuses with HTTP 422. It accepts "
            f"the twenty standard residues plus the ambiguity codes B, J, X and Z. Note that U "
            f"(selenocysteine) and O (pyrrolysine) are among the refused characters although "
            f"both occur in real proteins, so a selenoprotein cannot be folded by this lane.")
    return cleaned


def _attempt(body, endpoint=None):
    """One POST of the raw sequence. Never raises for an HTTP status; records it."""
    endpoint = endpoint or ENDPOINT
    headers = {"User-Agent": USER_AGENT, "Content-Type": "text/plain", "Accept": "*/*"}
    # Only to the second host, and only when one is configured. Sending a bearer token to
    # the public Atlas would be leaking a credential to a third party for no purpose.
    if LONG_TOKEN and endpoint == LONG_ENDPOINT:
        headers["Authorization"] = f"Bearer {LONG_TOKEN}"
    request = urllib.request.Request(
        endpoint, data=body.encode("ascii"), method="POST", headers=headers)
    timeout = (LONG_REQUEST_TIMEOUT if LONG_ENDPOINT and endpoint == LONG_ENDPOINT
               else REQUEST_TIMEOUT)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return _Opened(body=response.read().decode("utf-8", "replace"),
                           status=getattr(response, "status", 200))
    except urllib.error.HTTPError as error:
        # The refusal text is the most useful thing the host returns, so it is read rather
        # than discarded in favour of the status code.
        try:
            detail = error.read().decode("utf-8", "replace").strip()
        except Exception:
            detail = ""
        return _Opened(status=error.code, message=detail or str(error))
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return _Opened(transport=f"{type(error).__name__}: {error}")


def _post(sequence, sources, endpoint=None):
    """Fetch the predicted PDB, retrying only a transport failure or a retryable status.

    Returns `(pdb_text, attempts, absence)`. `absence` is None on success and otherwise the
    recorded refusal, never an exception: whether this lane could reach the host is a fact about
    the lane, and a caller that has already spent an action on it needs that fact recorded.
    """
    attempts, last = 0, None
    for attempt in range(1, RETRIES + 1):
        if attempt > 1:
            time.sleep(RETRY_PAUSE)
        attempts += 1
        sources.append(endpoint or ENDPOINT)
        last = _attempt(sequence, endpoint)
        if last.body is not None:
            return last.body, attempts, None
        retryable = last.transport or (last.status in RETRY_STATUS)
        if not retryable:
            break
    if last.transport:
        detail = last.transport
        reason = "endpoint_unavailable"
    elif last.status in RETRY_STATUS:
        detail = f"HTTP {last.status}: {last.message}"
        reason = "endpoint_unavailable"
    else:
        detail = f"HTTP {last.status}: {last.message}"
        reason = "endpoint_refused_sequence"
    return None, attempts, {
        "reason": reason, "detail": detail, "http_status": last.status,
        "attempts": attempts,
        "note": "No structure was obtained. This is a statement about this endpoint on this "
                "call and not a statement about the target: it must not be read as the "
                "sequence being unfoldable, and no fallback keyless ESMFold host exists to "
                "retry against."}


def _parse_pdb(text):
    """Pull per-residue pLDDT and CB coordinates out of the returned PDB.

    Keyed on `resSeq`, never on arrival order. The host drops X residues from the model while
    preserving the numbering of everything else, so a positional zip against the input sequence
    would shift every residue after the first X by one and silently report the wrong residue's
    confidence.
    """
    residues, model_date = {}, None
    for line in text.splitlines():
        if line.startswith("HEADER") and model_date is None:
            stamp = re.search(r"(\d{2})-([A-Z]{3})-(\d{2})", line)
            if stamp and stamp.group(2) in MONTHS:
                day, month, year = int(stamp.group(1)), MONTHS[stamp.group(2)], int(stamp.group(3))
                # Two-digit PDB years: the format's own convention, and no ESMFold release
                # predates 2000 in any case.
                model_date = f"{2000 + year:04d}-{month:02d}-{day:02d}"
            continue
        if not line.startswith("ATOM"):
            continue
        atom, resname = line[12:16].strip(), line[17:20].strip()
        try:
            resseq = int(line[22:26])
            x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
            bfactor = float(line[60:66])
        except ValueError:
            continue
        row = residues.setdefault(resseq, {"resseq": resseq, "resname": resname,
                                           "plddt_raw": None, "point": None})
        if atom == "CA":
            # The B-factor column is per-residue here; CA is the one atom every residue has.
            row["plddt_raw"] = bfactor
            if row["point"] is None:
                row["point"] = (x, y, z)
        elif atom == "CB":
            # CB is the burial-proxy centre wherever it exists; glycine keeps its CA.
            row["point"] = (x, y, z)
    return [residues[key] for key in sorted(residues)], model_date


def _neighbour_counts(rows, cutoff):
    """Residues whose proxy centre lies within `cutoff` angstrom of each residue's own.

    A contact number, computed as a plain O(n^2) sweep -- 400 residues is 80,000 distance
    comparisons and needs no spatial index. It is a proxy for burial and is not SASA; see the
    module docstring and `limitations`.
    """
    points = [row["point"] for row in rows]
    counts, limit = [], cutoff * cutoff
    for i, here in enumerate(points):
        if here is None:
            counts.append(None)
            continue
        total = 0
        for j, there in enumerate(points):
            if i == j or there is None:
                continue
            dx, dy, dz = here[0] - there[0], here[1] - there[1], here[2] - there[2]
            if dx * dx + dy * dy + dz * dz <= limit:
                total += 1
        counts.append(total)
    return counts


def _quantile(values, fraction):
    """Linear-interpolated quantile over a sorted copy. Empty input gives None."""
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return float(ordered[0])
    position = fraction * (len(ordered) - 1)
    low = math.floor(position)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def _runs(positions):
    """Contiguous [start, end] spans over a sorted position list."""
    spans = []
    for position in positions:
        if spans and position == spans[-1][1] + 1:
            spans[-1][1] = position
        else:
            spans.append([position, position])
    return [list(span) for span in spans]


def fold(sequence, target_name="", neighbour_cutoff=10.0, confident_plddt=70.0, record=True):
    """Fold one sequence with ESMFold and return the structure, its pLDDT, and a burial proxy.

    `sequence` is an amino-acid string; a FASTA block is accepted and its header dropped. A
    sequence over `MAX_RESIDUES` returns a recorded absence without spending a request, as does
    a host that will not answer -- under distinguishable reason codes, because "we could not
    fold this" and "this is unfoldable" are different claims and conflating them is the costliest
    error available here. Characters the host refuses raise `ValueError`.

    `record=False` returns the payload and writes nothing, which is what a dispatcher inside an
    episode passes: the episode writes the record.
    """
    started = time.perf_counter()
    cleaned = clean_sequence(sequence)
    if not isinstance(neighbour_cutoff, (int, float)) or neighbour_cutoff <= 0:
        raise ValueError(f"neighbour_cutoff must be a positive distance in angstrom, "
                         f"not {neighbour_cutoff!r}")
    sources = []
    payload = {
        "tool": "structure.fold_sequence", "model": MODEL,
        "target_name": target_name or None,
        "sequence_length": len(cleaned),
        "sequence_sha256": hashlib.sha256(cleaned.encode()).hexdigest(),
        "max_residues": MAX_RESIDUES,
        "neighbour_cutoff_angstrom": float(neighbour_cutoff),
        "confident_plddt": float(confident_plddt),
        "ambiguous_positions": [i for i, residue in enumerate(cleaned, 1)
                                if residue in AMBIGUOUS],
        "folded": False, "absence": None, "model_date": None,
        "limitations": list(LIMITATIONS),
    }

    # Which host takes this one, decided on length alone and written onto the record before
    # anything is sent. The citation follows the host that actually answered rather than the
    # one this module usually uses, so a GPU fold is never cited to the Atlas.
    endpoint, why = route(len(cleaned))
    payload["endpoint"] = endpoint or ENDPOINT
    payload["routed_because"] = why
    payload["citation"] = (CITATION if endpoint in (None, ENDPOINT) else
                           CITATION.replace(_served_by(ENDPOINT), _served_by(endpoint)))
    payload["max_residues"] = MAX_RESIDUES if endpoint in (None, ENDPOINT) else LONG_MAX_RESIDUES

    if endpoint is None:
        payload["absence"] = {
            "reason": "sequence_too_long", "detail": f"{len(cleaned)} residues exceeds the "
            f"measured {MAX_RESIDUES}-residue ceiling, above which the host returns HTTP 413 "
            f"'Sequence is longer than 400.'", "http_status": None, "attempts": 0,
            "routed_because": why,
            "note": "Refused locally, so no request was spent on a deterministic no. This is a "
                    "real limit on the lane and a real zero for this target: a full-length "
                    "viral glycoprotein cannot be folded here, and folding a fragment of it "
                    "answers a different question."}
        payload["network_calls"] = 0
        payload["sources"] = []
        payload["seconds"] = round(time.perf_counter() - started, 3)
        payload["provenance"] = provenance()
        return _finish(payload, None, target_name, record)

    text, attempts, absence = _post(cleaned, sources, endpoint)
    payload["network_calls"] = attempts
    payload["sources"] = list(dict.fromkeys(sources))
    if absence is not None:
        payload["absence"] = absence
        payload["seconds"] = round(time.perf_counter() - started, 3)
        payload["provenance"] = provenance()
        return _finish(payload, None, target_name, record)

    rows, model_date = _parse_pdb(text)
    payload["model_date"] = model_date or MODEL_DATE_FALLBACK
    payload["model_date_is_parsed"] = model_date is not None
    if not rows:
        payload["absence"] = {
            "reason": "unparseable_response", "detail":
            f"the host returned {len(text)} characters with no parseable ATOM records; it "
            f"began {text[:120]!r}", "http_status": 200, "attempts": attempts,
            "note": "A 200 whose body is not a PDB. Recorded rather than guessed at."}
        payload["seconds"] = round(time.perf_counter() - started, 3)
        payload["provenance"] = provenance()
        return _finish(payload, text, target_name, record)

    modelled = [row["resseq"] for row in rows]
    unmodelled = [position for position in range(1, len(cleaned) + 1) if position not in set(modelled)]
    mismatched = [row["resseq"] for row in rows
                  if 1 <= row["resseq"] <= len(cleaned)
                  and THREE_TO_ONE.get(row["resname"], "?") != cleaned[row["resseq"] - 1]]
    out_of_range = [position for position in modelled if not 1 <= position <= len(cleaned)]

    counts = _neighbour_counts(rows, float(neighbour_cutoff))
    plddt_rows, scores = [], []
    for row, count in zip(rows, counts):
        raw = row["plddt_raw"]
        # The column is 0-1 here, not the conventional 0-100. Reading it as 0-100 would call
        # every residue catastrophically unconfident.
        score = None if raw is None else round(raw * 100.0, 2)
        if score is not None:
            scores.append(score)
        plddt_rows.append({
            "position": row["resseq"],
            "residue": THREE_TO_ONE.get(row["resname"], "X"),
            "plddt": score, "neighbour_count": count})

    confident = [row["position"] for row in plddt_rows
                 if row["plddt"] is not None and row["plddt"] >= float(confident_plddt)]
    unconfident = sorted(row["position"] for row in plddt_rows
                         if row["plddt"] is not None and row["plddt"] < float(confident_plddt))
    usable = [row for row in plddt_rows
              if row["neighbour_count"] is not None and row["plddt"] is not None
              and row["plddt"] >= float(confident_plddt)]
    low_quartile = _quantile([row["neighbour_count"] for row in usable], 0.25)

    payload.update({
        "folded": True,
        "pdb": text,
        "pdb_bytes": len(text.encode()),
        "pdb_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "residues_modelled": len(rows),
        "unmodelled_positions": unmodelled,
        "out_of_range_positions": out_of_range,
        "alignment_verified": not mismatched and not out_of_range,
        "alignment_mismatches": mismatched,
        "plddt_scale": "0-100. The endpoint writes pLDDT into the PDB B-factor column on a 0-1 "
                       "scale; these values are that column multiplied by 100.",
        "plddt": plddt_rows,
        "mean_plddt": round(sum(scores) / len(scores), 2) if scores else None,
        "median_plddt": None if not scores else round(_quantile(scores, 0.5), 2),
        "min_plddt": min(scores) if scores else None,
        "confident_fraction": round(len(confident) / len(plddt_rows), 4) if plddt_rows else None,
        "low_confidence_runs": [span for span in _runs(unconfident) if span[1] - span[0] >= 4],
        "exposure_proxy": {
            "method": f"count of other residues whose CB (CA for glycine) lies within "
                      f"{float(neighbour_cutoff)} angstrom. A contact number.",
            "is_sasa": False,
            "lower_is_more_exposed": True,
            "computed_over_confident_residues_only": True,
            "confident_residues_used": len(usable),
            "first_quartile_count": low_quartile,
            "least_buried_positions": sorted(
                row["position"] for row in usable
                if low_quartile is not None and row["neighbour_count"] <= low_quartile),
            "caveat": "A proxy for burial on predicted coordinates, not a solvent-accessible "
                      "surface area and not a measurement. Positions are reported only where "
                      "pLDDT clears the threshold, because a contact number over a "
                      "low-confidence stretch describes a fold the model did not commit to. "
                      "A monomer surface is not necessarily exposed in an oligomer.",
        },
        "unverified": ["Every coordinate, every pLDDT value and every neighbour count. A "
                       "prediction and a proxy computed on it are model assertions, not "
                       "measurements, and nothing here was checked against an experiment."],
    })
    payload["seconds"] = round(time.perf_counter() - started, 3)
    payload["provenance"] = provenance()
    return _finish(payload, text, target_name, record)


def _finish(payload, text, target_name, record):
    """Stamp the content digest and, when recording, write the JSON and any PDB beside it."""
    payload["content_sha256"] = content_digest(payload)
    if not record:
        return payload
    slug = re.sub(r"[^a-z0-9]+", "_", (target_name or payload["sequence_sha256"][:12]).lower())
    base = ROOT / "runs" / "folded_structure"
    write_json(base / f"{slug.strip('_') or 'sequence'}.json", payload)
    if text:
        path = base / f"{(slug.strip('_') or 'sequence')}.pdb"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        payload["pdb_path"] = str(path.relative_to(ROOT))
    return payload


def main(argv=None):
    """Fold a sequence or a FASTA file from the command line, for checking the lane by hand."""
    import argparse
    import pathlib

    parser = argparse.ArgumentParser(description="Fold one sequence with ESMFold.")
    parser.add_argument("sequence", help="an amino-acid string, or a path to a FASTA file")
    parser.add_argument("--target-name", default="")
    parser.add_argument("--neighbour-cutoff", type=float, default=10.0)
    parser.add_argument("--confident-plddt", type=float, default=70.0)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args(argv)
    text = args.sequence
    candidate = pathlib.Path(text)
    if len(text) < 4096 and candidate.exists():
        text = candidate.read_text()
    payload = fold(text, target_name=args.target_name,
                   neighbour_cutoff=args.neighbour_cutoff,
                   confident_plddt=args.confident_plddt, record=not args.no_record)
    terse = {key: value for key, value in payload.items()
             if key not in ("pdb", "plddt", "provenance", "limitations")}
    print(json.dumps(terse, indent=2))
    return 0 if payload["folded"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
