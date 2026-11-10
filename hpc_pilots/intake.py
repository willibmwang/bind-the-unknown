"""Pathogen genome in, candidate target proteins out.

The stage the project is scoped to as of 22 October is intake through target generation: a
sequence arrives, and what comes back is a ranked list of proteins worth trying to bind. This
module is the first half of that -- getting from nucleotides to a set of identified proteins
-- and it deliberately does not read anybody's existing annotation of the genome.

**Why call the ORFs rather than fetch the annotation.** NCBI will return a fully annotated
protein set for any genome already in the database, and for SARS-CoV-2 that answer is correct
and instant. It is also unavailable for the case the system exists for. A genuinely novel
pathogen off a metagenomic assembly has no annotation, so a pipeline that reads one is
measuring the curator's work rather than its own, and would degrade silently -- not loudly --
the first time it met something new. Calling ORFs and identifying them by homology is the
same operation in both cases, which is the property worth having.

**What the frameshift costs, and why it is not patched over.** Six-frame translation finds
ORF1a of SARS-CoV-2 at 4,405 residues and ORF1b at 2,595, where the biological product is a
single 7,096-residue polyprotein produced by a -1 ribosomal frameshift. No ORF caller reading
stop codons can see that, and inferring it would mean encoding coronavirus-specific knowledge
into a step that is supposed to be family-agnostic. The two fragments are reported as what
they are, and `identify` recovers the relationship anyway, because both halves hit the same
UniProt polyprotein entry and the mature-protein ranges in that entry are what the target
layer uses. The limitation is recorded per ORF rather than hidden.

**Why the homology search runs locally.** EBI's BLAST service is the obvious tool and was the
first implementation. It is a job queue, and a single 306-residue query sat at RUNNING for
over fifteen minutes on the afternoon this was written, which makes a whole-genome scan
unbounded in a way an emergency tool cannot be. So identification runs against a reference set
fetched once -- the reviewed viral proteins, 17,535 entries and 10 MB -- indexed on 5-mers and
searched locally. A genome resolves in about ten seconds. BLAST remains available behind
`identify`, reached through `annotate`, which `run_intake` does not call -- so the service is
reachable and is off every recorded path.

The case where k-mer identity is too blunt -- remote homology below roughly 25% identity -- is
answered locally instead, by `identify_remote`: reduced-alphabet seeding then Smith-Waterman
with Karlin-Altschul expectations. That function's name is about the *homology* being remote
and not about the *service* being remote, and this paragraph previously named it as the BLAST
escape hatch, which sent a reader to the one tier that never leaves the machine.

**How far the control actually reaches, measured rather than assumed.** The organism-name
holdout below is an *isolate* holdout: ICTV places SARS-CoV, SARS-CoV-2 and three bat
coronaviruses in one species, so removing the name "severe acute respiratory syndrome
coronavirus 2" leaves four conspecific relatives in the reference. Taking the holdout up the
ranks, in `runs/holdout_ladder.json`, finds the limit. SARS-CoV-2 survives the isolate
holdout with all five targets and collapses at the species rung to one identified ORF and
none of them. Nipah survives a species holdout -- Hendra is a separate species in the same
genus -- and collapses at the genus rung to zero.

Two things follow. The limit is sequence identity and taxonomic rank is only a proxy for it,
which is why the breaking rung differs between families. And the failure is a cliff rather
than a slope: one rung below success nothing is identified at all. That it fails *empty*
rather than *wrong* is the one comfortable part -- with no identifiable homolog the pipeline
returns no targets instead of a confident wrong list -- and it is worth keeping. The remedy
for the cross-family case is remote homology, which means profile methods or structure
prediction and structural search, not a lower k-mer threshold.

**The control that makes this a novel-pathogen pipeline rather than a lookup.** Every protein
of SARS-CoV-2 is in the reference, so identifying its ORFs against the full set is a database
query with a known answer. `exclude_organism` removes the pathogen from the reference before
indexing, which is what the system would face for something genuinely new. Held out, all eight
real ORFs still resolve to the right protein by function -- the replicase to bat coronavirus
HKU3, the spike to HKU3's spike, the nucleoprotein to Rp3/2004 -- with scores falling from
1.000 to between 0.30 and 0.84. The assignment survives and its confidence visibly does not,
which is the behaviour wanted. The spurious reverse-strand ORF scores 0.0417 against a bat
coronavirus replicase and is rejected by the same threshold.

**Identification is homology, not naming.** Each ORF is searched against UniProtKB with
BLAST. The hit supplies an accession, which is the key everything downstream needs: the
structural evidence table is keyed on accession, and so are the mature-chain boundaries that
turn "this is a polyprotein" into "this is the protease domain of it". A hit is kept with its
identity and E-value so that a weak, family-level assignment -- which is what a novel pathogen
will produce -- is visibly weaker than an exact one rather than arriving as the same kind of
claim.
"""
import concurrent.futures
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request

from .common import ROOT, content_digest, provenance, write_json

EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BLAST = "https://www.ebi.ac.uk/Tools/services/rest/ncbiblast"

# The standard genetic code, built rather than tabulated so the ordering is checkable.
_BASES = "TCAG"
_AMINO = ("FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG")
CODONS = {a + b + c: _AMINO[i] for i, (a, b, c) in enumerate(
    (a, b, c) for a in _BASES for b in _BASES for c in _BASES)}

# An ORF shorter than this is noise at genome scale: six-frame translation of 30 kb throws off
# dozens of short open stretches by chance. Sixty residues keeps every real SARS-CoV-2 product
# including ORF10 at 38 -- which it does not keep, and that is the honest cost of the cutoff,
# recorded here rather than discovered later.
MINIMUM_AA = 60


def revcomp(seq):
    return seq.translate(str.maketrans("ACGTN", "TGCAN"))[::-1]


def translate(nt):
    return "".join(CODONS.get(nt[i:i + 3], "X") for i in range(0, len(nt) - 2, 3))


# Statuses that mean "ask again" rather than "there is no such thing". NCBI answers a burst
# of unauthenticated efetch calls with 429, and UniProt and the rest answer maintenance with
# 503; a 404 is a fact about the accession and retrying it only delays the right answer.
#
# This matters most at the one place a stranger touches this code. Someone pasting their own
# accession into the notebook and getting `HTTP Error 429` reads it as "my accession is
# wrong", which is the single most discouraging wrong conclusion available here, and two of
# the five example genomes in that cell returned it while being checked.
TRANSIENT_HTTP = (429, 500, 502, 503, 504)


def _fetch(url, timeout=180, data=None, attempts=4, backoff=2.0):
    """One request, retried on the statuses that mean the server is busy rather than sure."""
    import time as _time

    for attempt in range(attempts):
        try:
            request = urllib.request.Request(url, data=data) if data is not None else url
            with urllib.request.urlopen(request, timeout=timeout) as handle:
                return handle.read()
        except urllib.error.HTTPError as error:
            if error.code not in TRANSIENT_HTTP or attempt == attempts - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == attempts - 1:
                raise
        _time.sleep(backoff * (attempt + 1))


def fetch_genome(accession):
    """One genome from NCBI, as (header, sequence). No annotation is requested.

    `accession` may name several records separated by commas, which is how a segmented
    genome arrives. Influenza carries its eight segments as eight nucleotide records, as do
    the bunyaviruses and arenaviruses at two or three, so a tool that takes one accession
    cannot read a large part of the pathogen families it exists for. Segments are returned
    separately rather than concatenated: joining them would create reading frames across a
    junction that does not exist in any molecule, and those frames would be called as ORFs.
    """
    query = urllib.parse.urlencode({"db": "nuccore", "id": accession,
                                    "rettype": "fasta", "retmode": "text"})
    text = _fetch(f"{EFETCH}?{query}", timeout=180).decode()
    records = fasta_records(text)
    if not records:
        raise ValueError(f"{accession} did not return FASTA; got {text[:80]!r}")
    if len(records) == 1:
        return records[0]
    return records


def fasta_records(text):
    """[(header, sequence)] from FASTA text, in order."""
    records, header, chunks = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if header is not None:
                records.append((header, "".join(chunks).upper()))
            header, chunks = line[1:].strip(), []
        elif header is not None:
            chunks.append(line.strip())
    if header is not None:
        records.append((header, "".join(chunks).upper()))
    return records


def genome_records(accession):
    """Always a list of (header, sequence), whether one accession or many."""
    fetched = fetch_genome(accession)
    return [fetched] if isinstance(fetched, tuple) else fetched


def orfs(seq, minimum_aa=MINIMUM_AA):
    """Open reading frames on both strands, longest first.

    Overlapping ORFs in different frames are all returned. That is deliberate: which of two
    overlapping reading frames is real is a question homology answers, and discarding one here
    would decide it on length alone.
    """
    found = []
    for strand, strand_seq in (("+", seq), ("-", revcomp(seq))):
        for frame in range(3):
            protein = translate(strand_seq[frame:])
            for match in re.finditer(r"M[^*]{%d,}" % (minimum_aa - 1), protein):
                found.append({
                    "strand": strand, "frame": frame,
                    "nt_start": frame + match.start() * 3,
                    "length_aa": len(match.group()), "protein": match.group()})
    return sorted(found, key=lambda o: -o["length_aa"])


def _post(path, fields):
    data = urllib.parse.urlencode(fields).encode()
    with urllib.request.urlopen(urllib.request.Request(f"{BLAST}/{path}", data=data),
                                timeout=120) as handle:
        return handle.read().decode().strip()


def _get(path, timeout=120):
    with urllib.request.urlopen(f"{BLAST}/{path}", timeout=timeout) as handle:
        return handle.read().decode()


def identify(protein, email, database="uniprotkb", poll=8, limit=90):
    """Homology-identify one ORF against UniProtKB. Returns the best hit, or None.

    Returns `None` on no hit rather than raising, because a genuinely novel protein having no
    homolog is a result about the pathogen and not an error in the pipeline. A transport
    failure does raise, so the two cannot be confused.
    """
    job = _post("run", {"email": email, "program": "blastp", "stype": "protein",
                        "database": database, "matrix": "BLOSUM62", "exp": "1e-3",
                        "alignments": "5", "scores": "5", "sequence": protein})
    for _ in range(limit):
        status = _get(f"status/{job}", timeout=60).strip()
        if status == "FINISHED":
            break
        if status in ("ERROR", "FAILURE", "NOT_FOUND"):
            raise RuntimeError(f"BLAST job {job} reported {status}")
        time.sleep(poll)
    else:
        raise TimeoutError(f"BLAST job {job} still {status} after {limit * poll}s")

    payload = json.loads(_get(f"result/{job}/json"))
    hits = payload.get("hits") or []
    if not hits:
        return None
    best = hits[0]
    hsp = (best.get("hit_hsps") or [{}])[0]
    return {"job": job,
            "accession": best.get("hit_acc"),
            "uniprot_id": best.get("hit_id"),
            "description": best.get("hit_desc"),
            "organism": best.get("hit_os"),
            "identity": hsp.get("hsp_identity"),
            "evalue": hsp.get("hsp_expect"),
            "align_length": hsp.get("hsp_align_len")}


def annotate(accession, email, minimum_aa=MINIMUM_AA, top=None, workers=8, record=True):
    """Genome accession -> ORFs -> homology-identified proteins, with provenance.

    `top` bounds how many of the longest ORFs are searched. BLAST is a job queue and each
    submission costs minutes, so a whole six-frame call is a lot of waiting for ORFs that are
    short and spurious. What is dropped is recorded in the artifact, because a bounded sweep
    reported without its bound reads as a complete one.
    """
    header, seq = fetch_genome(accession)
    found = orfs(seq, minimum_aa)
    searched = found if top is None else found[:top]

    def run(index_orf):
        index, orf = index_orf
        try:
            return index, identify(orf["protein"], email), None
        except Exception as error:                       # noqa: BLE001 - reported, not raised
            return index, None, f"{type(error).__name__}: {error}"

    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for index, hit, failure in pool.map(run, list(enumerate(searched))):
            results[index] = (hit, failure)

    proteins = []
    for index, orf in enumerate(searched):
        hit, failure = results.get(index, (None, "not searched"))
        proteins.append({k: orf[k] for k in ("strand", "frame", "nt_start", "length_aa")}
                        | {"sequence": orf["protein"], "hit": hit, "search_error": failure})

    payload = {
        "experiment": "pathogen-intake", "status": "complete",
        "genome_accession": accession, "genome_header": header,
        "genome_length_nt": len(seq),
        "minimum_orf_aa": minimum_aa,
        "orfs_found": len(found), "orfs_searched": len(searched),
        "orfs_dropped_by_top": len(found) - len(searched),
        "identified": sum(1 for p in proteins if p["hit"]),
        "search_failures": sum(1 for p in proteins if p["search_error"]),
        "proteins": proteins,
        "sources": [{"name": "NCBI efetch", "url": EFETCH,
                     "licence": "NCBI public domain; see NLM copyright notice"},
                    {"name": "EBI NCBI-BLAST REST", "url": BLAST,
                     "licence": "EMBL-EBI terms of use"}],
        "provenance": provenance(),
        "limitations": [
            "Six-frame ORF calling cannot see a ribosomal frameshift. SARS-CoV-2 ORF1ab is "
            "returned as ORF1a and ORF1b, two ORFs, where the product is one polyprotein.",
            f"ORFs shorter than {minimum_aa} residues are not called, which excludes real "
            "small products; SARS-CoV-2 ORF10 at 38 residues is one.",
            "Overlapping ORFs in different frames are all retained and identified; no frame "
            "is chosen by length.",
            "A homology hit is an assignment, not a measurement. Identity and E-value travel "
            "with it so a family-level assignment is visibly weaker than an exact one."],
        "interpretation": "Proteins are identified by homology to UniProtKB, not read from "
                          "the genome's existing annotation. No binding, affinity, "
                          "inhibition or antiviral property is predicted for any protein."}
    payload["content_sha256"] = content_digest(payload)
    if record:
        write_json(ROOT / "data/processed/intake" / f"{accession}.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Local identification against a cached reference set.

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"

# Reviewed entries only. TrEMBL would multiply the reference by three orders of magnitude for
# sequences nobody has curated, and the point of the reference is that a hit against it carries
# a name somebody stands behind.
#
# One scope, because this project is antiviral work. The reference is reviewed viral proteins
# and nothing else, which is a limitation worth stating rather than a default worth hiding: a
# genome from outside that set identifies nothing at all -- not distantly, not weakly, nothing
# -- and the failure looks exactly like the cross-family failure the cascade exists to handle
# while having an entirely different cause. `run_intake` refuses such a genome on size rather
# than ranking it, so the two never get confused.
#
# Measured on 22 October 2026: 17,529 reviewed viral entries, 6M residues, indexed in about
# five seconds. The exact-seed index is a dict of sets and its memory grows with residues.
SCOPES = {"viral": "(taxonomy_id:10239) AND (reviewed:true)"}
REFERENCE_QUERY = SCOPES["viral"]

KMER = 5

# Below this, a hit is noise. Calibrated on the spurious ORFs in the SARS-CoV-2 six-frame call,
# the highest of which scores 0.0417, against 0.3011 for the weakest real assignment when the
# pathogen itself is held out of the reference.
#
# The noise figure was 0.048 against an unrelated phage protein while `identify_local` divided
# by the smaller of the two k-mer sets. Normalising by the query alone moved it down, because
# that phage entry is 46 residues and was supplying its own 42-k-mer denominator: the same two
# shared k-mers now read 2/96 rather than 2/42. The weakest real assignment did not move at all,
# since its query was already the smaller side. So the band this threshold sits in widened from
# 0.048-0.30 to 0.0417-0.3011 and 0.12 became more conservative without being re-derived --
# worth stating, because a threshold that survives a change to the quantity it gates should be
# shown to have survived rather than assumed to have.
MINIMUM_SCORE = 0.12


def fetch_reference(query=None, path=None, scope="viral"):
    """The reference protein set for one scope, downloaded once and cached as FASTA."""
    import gzip

    if query is None:
        if scope not in SCOPES:
            raise ValueError(f"Unknown scope {scope!r}; known scopes are {sorted(SCOPES)}")
        query = SCOPES[scope]
    path = path or ROOT / "data/raw/reference" / f"{scope}_reviewed.fasta"
    if path.exists():
        return path
    url = f"{UNIPROT_STREAM}?query={urllib.parse.quote(query)}&format=fasta&compressed=true"
    with urllib.request.urlopen(url, timeout=600) as handle:
        raw = handle.read()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(gzip.decompress(raw).decode())
    return path


def read_fasta(path):
    """(accession, name, organism, sequence) per entry, from a UniProt FASTA."""
    entries, accession, name, organism, taxon, chunks = [], None, None, None, None, []

    def flush():
        if accession:
            entries.append({"accession": accession, "name": name, "organism": organism,
                            "taxid": taxon, "sequence": "".join(chunks)})

    for line in open(path):
        if line.startswith(">"):
            flush()
            head = line[1:].strip()
            parts = head.split("|")
            accession = parts[1] if len(parts) > 2 else head.split()[0]
            rest = parts[2] if len(parts) > 2 else head
            name = rest.split(" OS=")[0]
            match = re.search(r"OS=(.*?)(?: OX=| GN=| PE=| SV=|$)", rest)
            organism = match.group(1) if match else None
            taxid = re.search(r"OX=(\d+)", rest)
            taxon = taxid.group(1) if taxid else None
            chunks = []
        else:
            chunks.append(line.strip())
    flush()
    return entries


def build_index(entries, exclude_organism=None, k=KMER, require_removal=True):
    """A k-mer index over the reference, optionally with one organism removed.

    `exclude_organism` is the control that makes this a pipeline rather than a lookup. Every
    protein of a sequenced pathogen is already in UniProt, so identifying its own ORFs against
    the full reference answers a question nobody has. Removing it reproduces what the system
    faces for something new, where the nearest relative is a different species.
    """
    index, kept = {}, []
    for entry in entries:
        if exclude_organism and entry["organism"] \
                and exclude_organism.lower() in entry["organism"].lower():
            continue
        position = len(kept)
        kept.append(entry)
        sequence = entry["sequence"]
        for offset in range(len(sequence) - k + 1):
            index.setdefault(sequence[offset:offset + k], set()).add(position)
    # A name holdout that removed nothing is the most dangerous outcome available here,
    # because its output is indistinguishable from one that worked: the pathogen is still in
    # the reference, every protein identifies perfectly, and the run reads as a passing test
    # of something never tested. Two real cases, found while assembling benchmark families.
    # NCBI titles the dengue reference "Dengue virus 2" while UniProt writes "Dengue virus
    # type 2", so the obvious string removes nought of thirteen entries. And holding out
    # "West Nile virus" leaves Kunjin virus, which is West Nile lineage 1b at 97.6% identity
    # under a name sharing no substring with it.
    #
    # So a holdout removing nothing raises. The second failure it cannot catch, which is why
    # `build_index_excluding_clade` is the better instrument: taxonomy knows Kunjin is West
    # Nile and a substring never will.
    if exclude_organism and require_removal and len(kept) == len(entries):
        raise ValueError(
            f"Holding out {exclude_organism!r} removed no entries, so the pathogen is still "
            f"in the reference and this run is a lookup wearing a control's name. Check the "
            f"string against UniProt's OS= field, or hold out by clade instead.")
    return index, kept


def identify_local(protein, index, kept, k=KMER, top=3):
    """Best reference hits for one ORF, by shared k-mer fraction.

    The score is shared k-mers over the query's own k-mer count, so a short query against a
    long polyprotein is not penalised for the length it does not cover -- which matters here,
    because a frameshifted ORF half is exactly that case.

    **It used to divide by the smaller of the two sets, and the mirror case is what that
    broke.** Taking the minimum protects a short query inside a long entry, which is the case
    the paragraph above is about and which this still handles identically -- the query side was
    already the smaller one there. The case it did not survive is the reverse: a short reference
    ENTRY inside a long query, where the denominator becomes the entry's length and a fragment
    can score far higher than the full-length protein it is a fragment of.

    Two recorded instances. `P35985`, `CAPSH_ADECU Hexon protein (Fragment)`, is 18 residues --
    14 five-mers -- so ten shared k-mers scored 10/14 = 0.7143 and it was recorded as the
    nearest relative of a human adenovirus 2 hexon ORF, beating four full-length hexons at
    0.4759 to 0.5037, in `runs/benchmark/NC_001405_1.json`. And `A8C8X2`, 11 residues, scored
    exactly 1.0 on seven shared k-mers against influenza PB1-F2, over the true ortholog
    `A4GCJ5` at 0.8675. Because `hits[0]` is also what `decompose` reads mature chains from and
    what the structural stage fetches, the whole downstream lane for such a candidate pointed at
    an 18-residue entry.

    The reference holds 485 entries shorter than this pipeline's own `MINIMUM_AA` of 60 and 450
    named "(Fragment)", so this was not an isolated row.

    Normalising by the query alone needs no threshold, which is why it is preferred to flooring
    the denominator at some length: a floor would move the cliff rather than remove it -- 382 of
    those fragments are over 60 residues and could do the same thing -- and it would import
    `MINIMUM_AA`, a number argued for as a statement about six-frame translation noise in a
    query genome, into a role its rationale says nothing about.
    """
    import collections

    query = {protein[i:i + k] for i in range(len(protein) - k + 1)}
    counts = collections.Counter()
    for kmer in query:
        for position in index.get(kmer, ()):
            counts[position] += 1

    # Totally ordered, and the truncation is the half that matters. `counts.most_common(50)` is
    # `heapq.nlargest`, stable on first-touch insertion order -- and insertion order here is the
    # order `for kmer in query` walks a set of strings, which Python randomises per process. So
    # *which fifty* entries reached the scoring below was a function of PYTHONHASHSEED, and the
    # boundary is crowded: influenza's nuclear export protein cuts at 77 shared k-mers with
    # twelve positions tied there, PB1-F2 at 7 with seven tied.
    #
    # That was not cosmetic, because the score is not monotone in the count the cut ranks by.
    # Replaying PB1-F2 across seeds, A8C8X2 -- an 11-residue fragment -- misses the cut at seeds
    # 0, 1 and 2 and survives at 11 and 42, and when it survives it takes `hits[0]` outright.
    # The recorded run was the second kind, so `conservation`, `nearest_homolog` and
    # `nearest_organism` on a published row, and the rank it sorted to, all depended on the
    # interpreter's hash seed. Tie-breaking only the final sort leaves that untouched: an entry
    # that never entered the candidate set cannot be reordered back into it.
    #
    # `common.content_digest` hashes the whole payload bar the timestamps, and the standard it
    # was written to is stated in its own docstring: "A provenance record that cannot be
    # recomputed is a label, not a record." `hits`, `nearest_homolog` and `nearest_organism` all
    # participate in that digest, so this is the same requirement reaching one function later.
    #
    # Cost, measured on the real reference: influenza PA counts 2,860 candidate positions, and
    # `sorted` over them takes 1.70 ms against `most_common`'s 1.33 ms. Against an index build
    # of seconds, free.
    ordered = sorted(counts.items(),
                     key=lambda item: (-item[1], kept[item[0]]["accession"]))[:50]
    scored = []
    for position, shared in ordered:
        entry = kept[position]
        denominator = max(1, len(query))
        scored.append({"accession": entry["accession"], "name": entry["name"],
                       "organism": entry["organism"],
                       "score": round(shared / denominator, 4), "shared_kmers": shared,
                       # Recorded so the score can be recomputed from the row rather than
                       # trusted, and so a short entry is visible as one.
                       "denominator": denominator,
                       "entry_length": len(entry["sequence"]),
                       "score_scale": "shared_kmer_fraction_of_query"})
    scored.sort(key=lambda hit: (-hit["score"], hit["accession"]))
    return scored[:top]


# ---------------------------------------------------------------------------
# Ranking. Every feature is read off the homologs, never off the query protein.

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

# The subcellular locations that put a protein where a binder can reach it without crossing a
# membrane live in `geometry.EXPOSED_LOCATIONS`, and this module reads them only through
# `exposed_from_locations`.
#
# There used to be a second copy here, named `EXPOSED`, byte-identical to that one and matched
# with `any()` instead of a majority. Two copies of a vocabulary is a trap rather than a
# duplication: whichever one a later reader edits, half the module keeps the old answer, and the
# half that kept the old answer here was the half worth 2.0 points in the rubric. Removed rather
# than left unused, because an unreferenced constant that looks authoritative is the same trap
# with an extra step.

# Keyword families with an established record of being druggable, and the record is the point:
# these are not a judgement about this pathogen but about what has historically yielded
# inhibitors. Viral proteases and polymerases are the two classes with approved antivirals
# against them, and envelope glycoproteins are the class antibodies and peptides reach.
# Synonyms are the whole of the difficulty. These terms are matched against free text -- a
# UniProt keyword on one path, a mature-chain description on the other -- and the two
# vocabularies disagree. The first version of this table carried "protease" and not
# "proteinase", so SARS-CoV-2's main protease, which UniProt describes as "3C-like proteinase
# nsp5" and against which there is an approved drug, scored zero for druggability and ranked
# eleventh behind four proteins of unknown function. Matching free text is brittle and the
# brittleness is not hypothetical, so every term here carries its variants.
DRUGGABLE = {"protease": 3, "proteinase": 3, "peptidase": 3,
             "polymerase": 3, "nucleotidyltransferase": 3,
             "helicase": 2, "methyltransferase": 2, "endonuclease": 2, "exonuclease": 2,
             "hydrolase": 2, "transferase": 1,
             "glycoprotein": 2, "spike": 2, "viral envelope protein": 2,
             "fusion protein": 2, "fusion glycoprotein": 2, "attachment": 2,
             # Influenza names its two surface antigens after their activities rather than
             # after their class, so neither matched anything above. Haemagglutinin is the
             # target of every broadly neutralising antibody programme against the virus and
             # ranked sixth, below the matrix protein, until these three lines existed -- the
             # same failure as "proteinase", one family later.
             "hemagglutinin": 2, "haemagglutinin": 2, "neuraminidase": 3,
             # General enzyme classes a virus can carry. This is the THIRD time this table
             # has been wrong in the same way -- "proteinase" demoted the main protease of
             # SARS-CoV-2 to eleventh, "hemagglutinin" demoted influenza's antibody target to
             # sixth, and a whole class of targets scored zero for being absent from one
             # author's vocabulary rather than for being undruggable.
             #
             # The pattern is the finding and a fourth patch will not end it. A hand-written
             # keyword list encodes whichever domain its author had in mind, and the failure
             # is silent -- a missing term reads as "not a druggable class" rather than as
             # "not in my vocabulary". The principled replacement is an external, curated
             # source of target classes: an EC number means the protein is an enzyme, and
             # ChEMBL and DrugBank both publish what has actually been drugged. Until that
             # exists here, this list is a known weak point and is documented as one rather
             # than presented as complete.
             "topoisomerase": 3, "reductase": 3, "synthase": 2, "synthetase": 2,
             "dehydrogenase": 2, "kinase": 2, "ligase": 2, "isomerase": 1, "lyase": 1,
             "carboxylase": 1, "oxidase": 1}

# Keywords implying the protein is required for the virus to replicate at all. Binding a
# dispensable protein cannot become a therapeutic, which is the same argument the dossier's
# essentiality axis makes.
ESSENTIAL = ("polymerase", "helicase", "protease", "proteinase", "peptidase", "capsid",
             "nucleoprotein", "spike", "fusion", "attachment", "entry into host",
             "hemagglutinin", "haemagglutinin", "neuraminidase",
             # Replication and transcription, named the several ways entries spell them.
             "topoisomerase", "replication initiator", "dna polymerase", "rna polymerase")


def homolog_records(accessions, chunk=25):
    """UniProt entries for a set of accessions, with the fields the ranking reads."""
    fields = ("accession,id,protein_name,cc_subcellular_location,keyword,xref_pdb,"
              "length,organism_name")
    out = {}
    accessions = sorted(set(a for a in accessions if a))
    for start in range(0, len(accessions), chunk):
        batch = accessions[start:start + chunk]
        query = " OR ".join(f"accession:{a}" for a in batch)
        url = (f"{UNIPROT_SEARCH}?query={urllib.parse.quote(query)}"
               f"&fields={fields}&format=json&size={len(batch)}")
        with urllib.request.urlopen(url, timeout=120) as handle:
            payload = json.load(handle)
        for entry in payload.get("results", []):
            locations = []
            for comment in entry.get("comments", []):
                if comment["commentType"] == "SUBCELLULAR LOCATION":
                    for location in comment.get("subcellularLocations", []):
                        value = location.get("location", {}).get("value")
                        if value:
                            locations.append(value)
            out[entry["primaryAccession"]] = {
                "uniprot_id": entry.get("uniProtkbId"),
                "organism": entry.get("organism", {}).get("scientificName"),
                "locations": locations,
                "keywords": [k["name"] for k in entry.get("keywords", [])],
                "pdb": len([x for x in entry.get("uniProtKBCrossReferences", [])
                            if x["database"] == "PDB"])}
    return out


def score_candidate(hits, records):
    """Target-quality features for one ORF, read across its homologs.

    Read across all of them rather than off the best one, because the nearest relative is
    often the least studied. Bat coronavirus HKU3's replicase is the closest match to
    SARS-CoV-2's and carries no deposited structure, while SARS-CoV's carries 104 -- and the
    question the structure count answers is whether the *fold* has ever been solved, which is
    a property of the family and not of whichever entry happened to score highest.

    The weights are a stated rubric and not a fitted model. Nothing here is trained, and the
    combination is published in the artifact so a reader can disagree with it arithmetically
    rather than having to guess at it.
    """
    keywords, locations, structures = set(), set(), 0
    for hit in hits:
        # The protein's own name is matched alongside its keywords, because the two
        # vocabularies do not overlap where it matters. UniProt's keywords for influenza
        # neuraminidase are Glycoprotein, Hydrolase and Host membrane; the word
        # "neuraminidase" -- the name of the enzyme oseltamivir inhibits -- appears only in
        # the protein name, so a rule reading keywords alone scored it as non-essential.
        if hit.get("name"):
            keywords.add(hit["name"].lower())
        record = records.get(hit["accession"])
        if not record:
            continue
        keywords.update(k.lower() for k in record["keywords"])
        locations.update(l.lower() for l in record["locations"])
        structures = max(structures, record["pdb"])

    # One rule for this verdict, in both places it is reached. This read used to be its own
    # `any()` over its own copy of the vocabulary, while `exposed_from_locations` -- written
    # later, for the gate -- used a majority over a vocabulary byte-identical to it. So the same
    # quantity was computed two ways in one module, and the weaker way was the one worth 2.0
    # points: `any()` makes a protein annotated in four compartments surface-exposed on the
    # strength of one, which is the exact case that function's docstring was written about
    # (SARS-CoV-2's main protease, annotated in four, one of them `Host extracellular space`,
    # reached by a cell-permeable drug). `decompose` now scopes this per chain, and if the
    # entry-level read kept `any()` the two would differ by rule as well as by scope.
    verdict = exposed_from_locations(sorted(locations))
    exposed = verdict["exposed"]
    druggable = max([weight for term, weight in DRUGGABLE.items()
                     if any(term in k for k in keywords)] or [0])
    essential = any(any(term in k for k in keywords) for term in ESSENTIAL)
    conservation = hits[0]["score"] if hits else 0.0
    # The scale is recorded, not reconciled, and the two tiers do not share one.
    #
    # `identify_local` returns shared 5-mers over the query's k-mer count. `identify_remote`
    # returns min(1, bits/200). Both land in this one variable and are multiplied by the
    # rubric's heaviest weight, 3.0, and nothing anywhere said so. Across every artifact on
    # disk the remote tier has median conservation 1.0000 with 43 of 74 rows at exactly 1.0,
    # against the exact tier's median 0.5397 with 11 of 448 -- so the two are not comparable
    # and the remote one is mostly saturated.
    #
    # It inverts real pairs. In the recorded SARS-CoV-2 holdout, ORF8's homolog is 57.0%
    # identical over 121 residues and scores 0.8300 on the remote scale, while ORF7a's is 89.3%
    # identical and scores 0.5385 on the exact scale; every other feature is zero on both rows,
    # so conservation alone orders them and puts the worse match above the better one. Spike at
    # 1964.9 bits reads 1.0000, the same credit as a hit at exactly 200.
    #
    # Rescaling is refused rather than postponed. A conversion between a k-mer fraction and a
    # bit score needs a calibration, and `rank_score` states that there is no labelled set of
    # correct rankings to calibrate against -- so a divisor chosen to make one recorded run read
    # better is the fitted threshold this module refuses three times elsewhere. The honest
    # replacement is one quantity on both tiers, alignment identity, which `identify_remote`
    # already has the alignment for; it belongs in its own field measured against
    # `runs/surface_in_ranker.json`'s instrument, not smuggled into `score`, because
    # MINIMUM_SCORE is calibrated on the k-mer scale and redefining `score` would move the
    # exact/remote tier boundary as a side effect.
    #
    # The ceiling flag is on the quantity that matters and not on the tier. A k-mer fraction
    # reaches 1.0 too -- ten EBV rows do, all exact -- and at the ceiling the heaviest term
    # stops ordering the list whichever scale produced it.
    scale = hits[0].get("score_scale") if hits else None
    if scale is None and hits and hits[0].get("bits") is not None:
        scale = "bits_over_200"

    return {
        "conservation": round(conservation, 4),
        # Inside `features`, not only in the payload, because that is the one insertion point
        # that travels: `decompose` copies this dict for every mature chain, and
        # `runs/benchmark/*.json` carries `features` while dropping `hits` and having no
        # `limitations` block at all. A bare 1.0 in those artifacts was unannotatable from
        # anywhere else.
        "conservation_scale": scale,
        "conservation_at_ceiling": round(conservation, 4) >= 1.0,
        "conservation_bits": hits[0].get("bits") if hits else None,
        "conservation_shared_kmers": hits[0].get("shared_kmers") if hits else None,
        "conservation_denominator": hits[0].get("denominator") if hits else None,
        "surface_exposed": exposed,
        "structures_in_family": structures,
        "druggable_class": druggable,
        "essential_class": essential,
        "keywords_matched": sorted(k for k in keywords
                                   if any(t in k for t in DRUGGABLE) or
                                   any(t in k for t in ESSENTIAL))[:8],
        # Not truncated any more, and the fraction is carried beside it. `sorted(...)[:4]` made
        # the 2.0-point verdict unauditable from its own record in the worst possible way: the
        # verdict is now a fraction over this list, so a reader recomputing it from four of six
        # compartments gets a different number from the one scored. It was also biased against
        # the evidence that earned the term -- `secreted`, `virion membrane` and `virion
        # surface` all sort after every `host ...` value, so the compartment responsible for a
        # `true` was the one most likely to be cut, leaving rows that read as four
        # intracellular locations beside `surface_exposed: true`. Locations are a handful of
        # short strings; the truncation never bought much.
        "locations": sorted(locations),
        "locations_scored": verdict["n"],
        "exposed_fraction_of_locations": verdict["fraction"]}


def chain_locations(accession, description=None):
    """Subcellular locations for one mature chain, not for the entry it sits in.

    UniProt scopes these per molecule and the entry-level read was discarding it. Every
    SUBCELLULAR LOCATION comment on a polyprotein carries a `molecule` naming the chain it
    describes: SARS-CoV-2's replicase holds fourteen of them, and the main protease's says
    host cytoplasm, Golgi and nucleus while nsp3's says endoplasmic reticulum membrane.
    Flattening them to the entry made Mpro read as surface-exposed through nsp3, nsp4 and
    nsp6, which genuinely are membrane proteins.

    With no description, or none matching, the entry-wide union is returned and the caller is
    told which it got. A union over a polyprotein is close to meaningless and a consumer
    should know it is looking at one.
    """
    payload = _entry(accession)
    per_molecule, general = {}, []
    for comment in payload.get("comments", []):
        if comment["commentType"] != "SUBCELLULAR LOCATION":
            continue
        values = [spot.get("location", {}).get("value")
                  for spot in comment.get("subcellularLocations", [])]
        values = [v for v in values if v]
        molecule = comment.get("molecule")
        if molecule:
            per_molecule.setdefault(molecule.lower(), []).extend(values)
        else:
            general.extend(values)

    if description:
        wanted = description.lower()
        for molecule, values in per_molecule.items():
            if molecule == wanted or wanted in molecule or molecule in wanted:
                return {"locations": values, "scope": "chain", "molecule": molecule}
    union = list(general) + [v for values in per_molecule.values() for v in values]
    return {"locations": union,
            "scope": "entry" if not per_molecule else "entry union over chains",
            "molecule": None}


def exposed_from_locations(locations):
    """Is this protein predominantly somewhere a macrocycle can reach without crossing a
    membrane?

    A majority rather than an `any`, and the difference is not pedantic. SARS-CoV-2's main
    protease is annotated in four compartments and one of them is `Host extracellular space`
    -- a real annotation, not an error -- so `any` made the target of an intracellular drug
    read as a surface protein. A protein found in four places and exposed in one is
    predominantly inside, and nirmatrelvir reaches it by being cell-permeable rather than by
    meeting it outside.

    Returns the fraction alongside the verdict, because a protein at exactly half is a
    different kind of claim from one at all four of four and the gate should be able to say so.
    """
    from .geometry import EXPOSED_LOCATIONS

    lowered = [spot.lower() for spot in locations]
    if not lowered:
        return {"exposed": False, "fraction": 0.0, "n": 0}
    hits = sum(1 for spot in lowered
               if any(term in spot for term in EXPOSED_LOCATIONS))
    fraction = hits / len(lowered)
    return {"exposed": fraction >= 0.5, "fraction": round(fraction, 3), "n": len(lowered)}


def amenable(features, max_patch_fraction=None):
    """Can a macrocycle bind this target at all? A gate, not a rank.

    The surface markers failed as a scoring term in both forms tried, and the reason they
    failed is that they answer a different question from the one the ranking asks. The
    ranking asks which target to pursue. This asks whether the molecule class can reach it,
    and those come apart: a target can be the best in the genome and still have no surface a
    macrocycle can lie across, which is a reason to pick a different modality rather than a
    reason to demote the target.

    Three conditions, and all are necessary. The protein has to be somewhere a macrocycle can
    get to, the patch has to be large enough to host the footprint, and it has to be small
    enough to be a site rather than a region.

    The third was absent for a long time and its absence pointed the wrong way: with a floor
    at half a footprint and no ceiling, the larger the patch the more confidently this read
    `yes`. Patch membership is spatial adjacency at ten angstroms, so on a large chain the
    exposed hydrophobic residues connect into one component wrapping the protein --
    SARS-CoV-2's nsp4 reports 7,558 square angstroms that way, an order of magnitude over the
    footprint, marking nothing. The quantity that separates a site from a region is the share
    of the domain's own surface and not an absolute area, because percolation scales with the
    chain while a binding site does not.

    The second is doing more work than it appears. Nipah's matrix protein carries the largest
    patch measured anywhere, 1,236 square angstroms, and is not a drug target: that surface is
    an oligomerisation interface on the inner face of the virion. UniProt separates it
    correctly -- matrix is `Virion`, the glycoproteins are `Virion membrane` -- which is why
    the gate reads localisation rather than hydropathy. A signal peptide would get this wrong:
    Nipah's G is a type II membrane protein with an uncleaved signal anchor and shows no
    signal peptide at all, while being the most exposed protein on the particle.

    Returns None rather than False when there is no structure, because "no surface was
    measured" and "the surface is unsuitable" are different statements and only one of them is
    about the target.

    **Two limitations that make this advisory rather than decisive, both measured.**

    The localisation it reads is annotated on the entry, and for a polyprotein the entry is
    the polyprotein. SARS-CoV-2's replicase lists `Host cell membrane` among its locations --
    correctly, since nsp3, nsp4 and nsp6 are membrane proteins -- so the main protease, which
    is cytosolic, reads as surface-exposed through its parent. That is the same parent-versus-
    chain defect already fixed for structure selection, still present in this axis, and it
    means the gate is unreliable for any mature chain of a polyprotein until localisation is
    resolved per chain.

    And the patch itself moves with the structure chosen. The main protease measures 343
    square angstroms through one homolog's entry and 825 through another, a factor of 2.4 on
    the quantity the gate thresholds. A single structure is a sample of a conformational
    ensemble and is treated here as if it were the protein.
    """
    if max_patch_fraction is None:
        max_patch_fraction = PATCH_FRACTION_CEILING
    patch = features.get("hydrophobic_patch_area")
    if patch is None:
        return None
    if not features.get("surface_exposed"):
        return "no"
    # Three-valued, because a binary gate puts a hard edge where the evidence has none.
    # A macrocyclic peptide buries four to eight hundred square angstroms in total, and the
    # hydrophobic patch is the core of that site rather than all of it -- hot-spot analyses
    # put roughly half the binding energy in a small apolar core, with polar contacts around
    # it. So a patch at half the footprint is a plausible site and a patch at the full
    # footprint is a comfortable one, and the band between them is genuinely uncertain.
    #
    # Nipah's G is the case that forced this. It is the most exposed protein on the particle
    # and the target of the only antibody ever given to a human for the disease, and it
    # measures 534 square angstroms against a 600 threshold. Lowering the threshold to admit
    # it would be fitting the gate to a target already known to be good, which is the error
    # this file has now recorded three times. Reporting it as uncertain is what the
    # measurement supports.
    # The measured spread decides the verdict, not the median alone. Measuring the patch over
    # five structures of the same domain rather than one showed the quantity moving by 200
    # square angstroms on the main protease and 496 on the spike receptor-binding domain --
    # a spread comparable to the 600 threshold itself, with the range 592 to 1088 straddling
    # it. A hard cut on a median that uncertain reports a confidence the measurement does not
    # have, so a range crossing the threshold is reported as borderline whatever the median
    # says.
    # A patch can fail upwards, and the gate used to have no way of saying so: the band had a
    # floor at half a footprint and no ceiling at all, so the larger the patch the more
    # confidently it read `yes`. The largest patches in a viral genome belong to percolated
    # surfaces and to membrane proteins measured out of their bilayer, and neither is a site.
    # A patch covering more than `max_patch_fraction` of the domain's accessible surface is
    # therefore reported as uncertain rather than as a pass: there is a great deal of exposed
    # greasy area and the measurement has not located a macrocycle-sized place within it.
    fraction = features.get("patch_fraction_of_surface")
    if fraction is not None and fraction >= max_patch_fraction:
        return "borderline"

    low = features.get("patch_area_min", patch)
    high = features.get("patch_area_max", patch)
    if low >= MACROCYCLE_FOOTPRINT_A2:
        return "yes"
    if high < 0.5 * MACROCYCLE_FOOTPRINT_A2:
        return "no"
    return "borderline"


# A macrocyclic peptide buries roughly this much on binding. Repeated from `geometry` so the
# gate can be evaluated without importing the structural stack.
MACROCYCLE_FOOTPRINT_A2 = 600.0

# The share of a domain's accessible surface above which the largest patch is no longer a
# site. Chosen, not derived, and reported at three values in `runs/amenability_gate.json` for
# the same reason the contact cutoff is: a threshold that produces the verdict is worth being
# suspicious of, and what matters is whether the direction survives moving it.
#
# The quantity is a fraction and not an area because the failure it catches is percolation.
# Patch membership is spatial adjacency at ten angstroms, so on a large enough chain the
# exposed hydrophobic residues connect into one component that wraps the protein -- nsp4
# reports 7,558 square angstroms this way. In absolute terms that clears a 600 square-angstrom
# footprint by an order of magnitude and reads as the most macrocycle-amenable surface in the
# genome; as a share of the domain's own surface it is a quarter of it, which is a region and
# not a place to put a molecule.
PATCH_FRACTION_CEILING = 0.15

# Mean macrocycle surface score over the six structures in `runs/surface_markers.json`, which
# span two enzyme pockets and four interfaces. Used to centre the surface term so that a
# candidate with no structure is neutral rather than penalised.
SURFACE_BASELINE = 3.24


def enrich_with_surface(ranked, top=12):
    """Attach structure-derived surface markers to the highest-ranked candidates.

    Only the top few, because the accessibility calculation costs seconds per structure and a
    segmented genome decomposed into mature chains produces more candidates than are worth
    measuring. The name-derived score orders the field cheaply; this re-scores the head of it
    with something that does not depend on what the protein is called.
    """
    from .geometry import surface_for_accession

    for row in ranked[:top]:
        accession = row.get("nearest_homolog")
        if not row.get("identified") or not accession or not row.get("features"):
            continue
        span = row.get("chain_range")
        interval = {"start": span[0], "end": span[1]} if span else None
        markers = surface_for_accession(accession, interval=interval)
        if not markers or markers.get("surface_score") is None:
            row["features"]["surface_score"] = None
            continue
        row["features"]["surface_score"] = markers["surface_score"]
        row["features"]["hydrophobic_patch_area"] = markers.get("hydrophobic_patch_area")
        row["features"]["patch_area_min"] = markers.get("patch_area_min")
        row["features"]["patch_area_max"] = markers.get("patch_area_max")
        row["features"]["patch_area_spread"] = markers.get("patch_area_spread")
        row["features"]["patch_fraction_of_surface"] = markers.get(
            "patch_fraction_of_surface")
        row["features"]["structures_measured"] = markers.get("structures_measured")
        row["features"]["structure_used"] = f"{markers['pdb_id']}:{markers.get('chain')}"
        # Localisation resolved for this chain, not inherited from the polyprotein.
        described = (row["features"].get("chain_description") or row.get("name") or "")
        scoped = chain_locations(accession, described)
        row["features"]["chain_locations"] = scoped["locations"][:4]
        row["features"]["location_scope"] = scoped["scope"]
        if scoped["scope"] == "chain":
            verdict = exposed_from_locations(scoped["locations"])
            row["features"]["surface_exposed"] = verdict["exposed"]
            row["features"]["exposed_fraction_of_locations"] = verdict["fraction"]
        row["features"]["macrocycle_amenable"] = amenable(row["features"])
    # Not re-sorted. The markers are reported beside the ranking and do not change it.
    return ranked


# The five weights, in one place. They were literals here and literals again in the `rubric`
# block of the record, which is two statements of one rubric: an edit to the weight that
# actually ranks could leave the published rubric asserting the old one, and the published
# rubric is what the site renders and what a reader checks the arithmetic against.
RUBRIC = {"conservation": 3.0, "surface_exposed": 2.0, "essential_class": 1.5,
          "druggable_class": 0.8, "structures_in_family_capped": 0.5}

# The most each term can pay, which is what lets a term be read as a fraction of what was
# available rather than as a bare number. Three of the five are their own weight:
# `conservation` is a 0-1 fraction, and the two booleans earn the weight or nothing. The
# structure count is divided by its 50 cap first, so it tops out at its weight too.
# `druggable_class` is the exception and the reason this table exists -- it is an integer
# class, not a fraction, so its ceiling is the weight times the highest class in DRUGGABLE,
# and a reader given only the weight would score a 2.4-point term out of 0.8.
TERM_CEILING = {"conservation": RUBRIC["conservation"],
                "surface_exposed": RUBRIC["surface_exposed"],
                "essential_class": RUBRIC["essential_class"],
                # Rounded, because 0.8 * 3 is 2.4000000000000004 in binary floating point and
                # this number is published as the denominator a reader divides by.
                "druggable_class": round(RUBRIC["druggable_class"] * max(DRUGGABLE.values()),
                                         4),
                "structures_in_family": RUBRIC["structures_in_family_capped"]}


def rank_score(features):
    """One number from the features, with every term visible.

    Deliberately simple and deliberately not fitted. There is no labelled set of "correct
    target rankings for a novel pathogen" to fit against -- that absence is why the project
    grades decisions rather than predictions -- so a learned weighting here would be fitted to
    whatever its author already believed. A stated rubric is honest about being one.
    """
    base = (RUBRIC["conservation"] * features["conservation"]
            + RUBRIC["surface_exposed"] * float(features["surface_exposed"])
            + RUBRIC["essential_class"] * float(features["essential_class"])
            + RUBRIC["structures_in_family_capped"]
            * min(1.0, features["structures_in_family"] / 50.0))

    # Where a structure exists, the surface markers carry the druggability term and the
    # keyword class does not. The keyword class scores zero on six of this project's own
    # eleven targets -- including the main protease and the neuraminidase, whose packet names
    # contain no enzyme word -- and it is biased against interfaces, which is the half of the
    # target set macrocycles exist for. The surface score is computed from a deposited
    # structure and does not know what the protein is called.
    #
    # The keyword class remains the fallback rather than being deleted, because most
    # candidates never have a structure fetched for them -- the surface stage reaches the head
    # of the list and no further -- and a ranker that scores the rest identically is worse
    # than one using a flawed proxy.
    # The surface markers are computed and reported and are deliberately NOT scored. Two
    # forms were measured against the ranking they were meant to improve and both made it
    # worse.
    #
    #                        SARS-CoV-2 p      Nipah p
    #   no surface term          0.0014         0.0714
    #   added as a bonus         0.0009         0.3929
    #   centred on a baseline    0.0201         0.1250
    #
    # Added, the term cannot be earned by a candidate with no deposited structure, so every
    # such candidate sinks below every candidate that has one -- a penalty for being
    # understudied, aimed at the novel targets this exists to find. The spike went first to
    # fifth on `None`. Centred on the mean of six reference structures, the mirror failure:
    # most real candidates score below that mean, so having a structure became the penalty and
    # the top five were again everything with no structure at all. There is no baseline to
    # centre on that is not invented from the targets under test.
    #
    # What survives is the measurement in `runs/surface_markers.json`: scoring a contiguous
    # hydrophobic patch ranks the spike receptor-binding domain above the main protease, the
    # reverse of a small-molecule druggability score, and the direction holds at every contact
    # cutoff tried. That signal is real and is too weak to improve a ranking that already
    # works. Reported as a feature so a reader can see it; not mixed into the score.
    return round(base + RUBRIC["druggable_class"] * features["druggable_class"], 4)


def topology_overlap(rows):
    """How many identified rows the hydropathy-class rows outrank, summed over both.

    A topology row carries a hard-coded feature vector and therefore always scores 3.6, which
    is not commensurable with a score a homology row earned from measured features. The comment
    on that row used to claim it was "scored below any homology hit so it never outranks one",
    which was never true and could not be made true by any fixed number: an exact-tier hit only
    has to clear 3.0 * MINIMUM_SCORE = 0.36, and remote-tier hits are not filtered on that
    threshold at all.

    Ordering it below every named row instead was measured and refused -- it sends the class row
    to last place on every genome that produces one, which is the penalty for being understudied
    that `rank_score` already rejected two surface-term variants for. So the overlap is counted
    and published per run. It is a cost being accepted knowingly, and this is the number that
    says how large it is; it is also the before-measurement any later reordering needs.
    """
    classes = [r for r in rows
               if r.get("tier") == "topology" and r.get("score") is not None]
    named = [r for r in rows
             if r.get("tier") not in ("topology", None) and r.get("score") is not None]
    return sum(1 for klass in classes for row in named
               if row["score"] < klass["score"])


def score_terms(features, rubric=None):
    """`rank_score` split into the five terms it is a sum of.

    One number sorts the list and five numbers explain it, and the explanation is only worth
    having if it is the same arithmetic rather than a second rubric dressed as a reading of
    the first: every term here is a weight from `RUBRIC` times a feature already in the row,
    so the sum recovers the score exactly. `chain._decomposed_score` asserts that it does,
    per row, on every recorded run.

    Worth the two lines because the top of the list is routinely a near-tie reached by
    opposite routes. On the held-out SARS-CoV-2 run the spike and the replicase finish within
    a tenth of a point of each other, and the spike gets there on a full 2.0 for sitting on
    the outside of the virion and under half the conservation term, while the replicase scores
    nothing for exposure and most of the conservation term. A single number says they are
    equally good and hides that one is an antibody target and the other is reached by a
    cell-permeable drug. The exact decimals move between runs, because four of the five
    features are read off live UniProt annotations on whichever homolog was matched.
    """
    rubric = rubric or RUBRIC
    terms = {
        "conservation": round(rubric["conservation"] * features["conservation"], 4),
        "surface_exposed": round(rubric["surface_exposed"]
                                 * float(features["surface_exposed"]), 4),
        "essential_class": round(rubric["essential_class"]
                                 * float(features["essential_class"]), 4),
        "structures_in_family": round(rubric["structures_in_family_capped"]
                                      * min(1.0, features["structures_in_family"] / 50.0), 4),
        "druggable_class": round(rubric["druggable_class"]
                                 * features["druggable_class"], 4),
    }
    return {"terms": terms, "total": round(sum(terms.values()), 4)}


def candidate_table(payload):
    """The ORFs one intake run ranked, one flat row each, with the score split five ways.

    `run_intake` already returns all of this, in the shape the stages downstream need: a full
    translated sequence per ORF, every homolog hit under it, and the features nested a level
    down. That is the right shape for a pipeline and an unreadable one for a person, and the
    only published form of this stage was five counts -- ORFs found, ORFs ranked, how many
    were identified. So the page could say that nine of twelve ORFs resolved and could not say
    which nine, what any of them scored, or what the other three were.

    Every ORF the run ranked is here, the unidentified ones included. Those three are the
    operating envelope rather than a gap in the table: no relative at either homology tier and
    no surface signal in their own sequence either, which is the state a genuinely novel
    protein arrives in. A table that listed the nine that worked would be the pipeline marking
    its own coverage.
    """
    rubric = payload.get("rubric") or RUBRIC
    rows = []
    for position, row in enumerate(payload["targets"], 1):
        orf = row["orf"]
        # The ORF's own coordinates, flattened up beside the score. Which strand and frame a
        # candidate was read in is the one fact here that comes from the genome rather than
        # from a relative, and it is what shows that overlapping frames are ranked separately:
        # ORFs 10 and 11 of the held-out run sit inside the spike's nucleotides on the other
        # strand and in another frame, and score nothing.
        out = {"rank": position,
               "identified": bool(row.get("identified")),
               "name": row.get("name"),
               "identification_route": row.get("identification_route") or row.get("tier"),
               "nearest_homolog": row.get("nearest_homolog"),
               "nearest_organism": row.get("nearest_organism"),
               "strand": orf["strand"], "frame": orf["frame"],
               "nt_start": orf["nt_start"], "length_aa": orf["length_aa"],
               "segment": orf.get("segment", 0)}
        features = row.get("features")
        if features is None:
            out |= {"features": None, "terms": None, "score": None,
                    "unidentified_because": row.get("unidentified_because"),
                    "topology": row.get("topology")}
        else:
            split = score_terms(features, rubric)
            score = row.get("score")
            out |= {"features": {k: features[k] for k in
                                 ("conservation", "surface_exposed", "essential_class",
                                  "druggable_class", "structures_in_family")},
                    "terms": split["terms"], "score": score,
                    # Asserted per row rather than trusted, for the same reason the chain
                    # asserts it: five numbers that do not add to the number the list sorted
                    # by would be an explanation of a different ranking.
                    "terms_sum_to_score": (score is not None
                                           and abs(split["total"] - score) < 5e-4),
                    # Why the two keyword terms fired, which is the half of the rubric a
                    # reader is most likely to want to argue with. `locations` is the evidence
                    # under `surface_exposed`, which is a majority over that list.
                    "keywords_matched": features.get("keywords_matched"),
                    "locations": features.get("locations"),
                    "exposed_fraction_of_locations":
                        features.get("exposed_fraction_of_locations"),
                    # Which quantity the heaviest term is carrying on this row. The two
                    # identification tiers do not report the same one -- exact returns a
                    # shared-5-mer fraction of the query, remote returns min(1, bits/200) --
                    # and a column of bare decimals that mixes the two invites a reader to
                    # compare numbers that are not on one scale. Carried per row rather than
                    # per run because one run's table can hold both.
                    "conservation_scale": features.get("conservation_scale"),
                    "conservation_at_ceiling": features.get("conservation_at_ceiling")}
        rows.append(out)
    return rows


def candidate_record(accession, exclude_organism=None, top=12, record=True, **kwargs):
    """Stage 1 on its own, recorded as the table behind its counts.

    Separate from the onset chain's record and not a replacement for it. The chain runs this
    same call as its first stage, with these same arguments, and keeps the counts; this keeps
    the rows. It is worth having as its own entry point because it is the cheap end of the
    pipeline -- about fifteen seconds against a warm reference cache, where the whole chain is
    thirty-five minutes -- so the published table can be refreshed without re-running six
    stages of folding and surface measurement to do it.

    The vocabulary that fires the two keyword terms is written into the record beside the
    weights. `druggable_class` is an integer from a hand-written keyword table that has been
    wrong three times in the same way, and a reader cannot audit a 1.6-point term without
    seeing which words earn which class.
    """
    started = time.time()
    payload = run_intake(accession, exclude_organism=exclude_organism, top=top,
                         record=False, **kwargs)
    rows = candidate_table(payload)
    out = {
        "experiment": "intake-candidate-table", "status": "complete",
        "question": "Which ORFs did stage 1 rank, what did each of the five stated weights "
                    "pay each of them, and which ORFs did it fail to identify at all?",
        "genome_accession": payload["genome_accession"],
        "genome_header": payload["genome_header"],
        "genome_length_nt": payload["genome_length_nt"],
        "segments": payload["segments"],
        "reference_excluded_organism": payload["reference_excluded_organism"],
        "reference_entries": payload["reference_entries"],
        "reference_entries_removed": payload["reference_entries_removed"],
        "reference_scope": payload["reference_scope"],
        "orfs_found": payload["orfs_found"],
        "orfs_ranked": payload["orfs_ranked"],
        "orfs_dropped_by_top": payload["orfs_dropped_by_top"],
        "orfs_rescued_by_topology": payload["orfs_rescued_by_topology"],
        "identified": sum(1 for r in rows if r["identified"]),
        "unidentified": payload["unidentified"],
        "minimum_orf_aa": payload["minimum_orf_aa"],
        "minimum_hit_score": payload["minimum_hit_score"],
        "rubric": dict(RUBRIC),
        "term_ceiling": dict(TERM_CEILING),
        "vocabulary": {
            "druggable_class": {str(klass): sorted(t for t, k in DRUGGABLE.items()
                                                   if k == klass)
                                for klass in sorted(set(DRUGGABLE.values()), reverse=True)},
            "essential_class": sorted(ESSENTIAL)},
        "candidates": rows,
        "every_score_re_adds_from_its_terms":
            all(r["terms_sum_to_score"] for r in rows if r["identified"]),
        "seconds": round(time.time() - started, 2),
        "provenance": provenance(),
        "limitations": [
            "The ranking is a stated rubric, not a fitted model. No labelled set of correct "
            "target rankings exists to fit against.",
            "`druggable_class` and `essential_class` are read from a hand-written keyword "
            "table against UniProt annotations on homologs. A term the table does not list "
            "reads as 'not druggable' rather than as 'not in my vocabulary'.",
            "`structures_in_family` is a count of deposited structures and is therefore a "
            "record of research attention as much as of tractability. It carries the smallest "
            "weight for that reason and is capped at 50.",
            "Six-frame calling misses frameshifted products: SARS-CoV-2's replicase is one "
            "7,096-residue polyprotein and is ranked here as two fragments, because no ORF "
            "caller reading stop codons can see a ribosomal frameshift.",
            "These are whole ORFs. A polyprotein is one row here and is cut into its mature "
            "chains by the stage after this one, so the scores below are not the per-chain "
            "scores the shortlist is drawn from."],
        "interpretation": "The candidate proteins stage 1 proposes for one genome, ranked by "
                          "a stated rubric. No binding, affinity, inhibition or antiviral "
                          "property is predicted for any of them."}
    out["content_sha256"] = content_digest(out)
    if record:
        write_json(ROOT / "runs/intake_candidates.json", out)
    return out


def run_intake(accession, exclude_organism=None, minimum_aa=MINIMUM_AA, top=12,
               minimum_score=MINIMUM_SCORE, record=True,
               exclude_taxid=None, exclude_rank=None, remote=True, scope="viral"):
    """Genome accession to a ranked list of candidate binding targets.

    Two holdouts, and they ask different questions. `exclude_organism` removes one organism
    by name, which tests identification from a close relative -- SARS-CoV-2 still resolves
    through bat coronavirus HKU3, same subgenus, 46% identity. `exclude_taxid` with
    `exclude_rank` removes an entire clade, so a family-level holdout leaves no coronavirus in
    the reference at all. The second is the case the theory of change describes and the one a
    species holdout cannot speak to.
    """
    segments = genome_records(accession)
    # A genome larger than this pipeline's heuristics are calibrated for is refused here
    # rather than ranked wrongly. The reference is reviewed viral proteins, `orfs` returns
    # every stretch between stops, and the selection below keeps the longest. All three are
    # right for a 30 kb virus and all three fail silently on a chromosome -- nothing
    # identifies against a viral reference, and length ranking returns a confident shortlist
    # of whichever long reading frames happened to be longest. Refusing is the only honest
    # answer, because a confident wrong ranking is the one output worse than none.
    #
    # **Measured on the longest segment, not the sum.** The motivating case in this module's
    # own docstring is a novel pathogen off a metagenomic assembly, and an assembly arrives
    # as contigs. Summing them refused a perfectly viral-scale genome for having been
    # delivered in pieces -- sixty 20 kb contigs sum past any viral limit while no single
    # contig is near it -- which rejected exactly the input the module exists to serve.
    # Influenza's eight segments sum to 13.5 kb and passed either way, so nothing recorded
    # moves; the contig case is the one this changes.
    longest_nt = max(len(part) for _, part in segments)
    total_nt = sum(len(part) for _, part in segments)
    if longest_nt > VIRAL_GENOME_LIMIT_NT:
        raise ValueError(
            f"{accession} has a {longest_nt:,} nt segment, over the "
            f"{VIRAL_GENOME_LIMIT_NT:,} nt limit this pipeline's heuristics are calibrated "
            f"for. This is not a claim that the input is not viral -- Mimivirus is 1.2 Mb and "
            f"Pandoravirus 2.5 Mb, and both are viruses. It is a statement about this code: "
            f"six-frame calling, length ranking and a reviewed-viral reference are all unsafe "
            f"at that size, so the honest answer is to refuse rather than to rank.")
    found = []
    for index, (segment_header, segment) in enumerate(segments):
        for orf in orfs(segment, minimum_aa):
            found.append({**orf, "segment": index,
                          "segment_header": segment_header})
    found.sort(key=lambda o: -o["length_aa"])
    for position, orf in enumerate(found):
        orf["_found_rank"] = position
    # Per segment when the genome is segmented, because the longest ORFs of influenza all sit
    # on the three polymerase segments and a flat top-N would rank those and never reach
    # haemagglutinin or neuraminidase, which are the targets.
    if len(segments) > 1:
        per_segment = max(1, top // len(segments) + 1)
        kept, counts = [], {}
        for orf in found:
            if counts.get(orf["segment"], 0) >= per_segment:
                continue
            counts[orf["segment"]] = counts.get(orf["segment"], 0) + 1
            kept.append(orf)
        searched = kept[:top] if top else kept
    else:
        searched = found[:top]

    # Length is a proxy for importance, and it is a proxy biased toward the proteins already
    # known. Truncating to the longest `top` ORFs before anything has looked at them drops a
    # short novel surface protein on a criterion that has nothing to do with whether it is a
    # target -- replicases and polymerases are long, and accessory and surface proteins are
    # often not. So every ORF the cut discarded is tested on its own sequence, and one that
    # reads as a probable surface protein is admitted even though it is short.
    #
    # Affordable because `membrane_topology` is arithmetic over the sequence: no network, no
    # reference, no index. The homology search these rescued ORFs then go through is the
    # expensive part, which is why the rescue is capped at `top` of them, longest first.
    rescued = []
    if top:
        taken = {orf["_found_rank"] for orf in searched}
        for orf in found:
            if len(rescued) >= top:
                break
            if orf["_found_rank"] in taken:
                continue
            if membrane_topology(orf["protein"])["likely_surface"]:
                rescued.append(orf)
        searched = list(searched) + rescued

    header = segments[0][0] if len(segments) == 1 else \
        f"{len(segments)} segments: " + "; ".join(h.split()[0] for h, _ in segments)
    seq = "".join(part for _, part in segments)

    entries = read_fasta(fetch_reference(scope=scope))
    removed, clade = None, None
    if exclude_taxid:
        if not exclude_rank:
            raise ValueError("A clade holdout needs the rank to take it at")
        known = lineages([exclude_taxid] + [e.get("taxid") for e in entries])
        index, kept, removed, clade = build_index_excluding_clade(
            entries, exclude_taxid, exclude_rank, known)
    else:
        index, kept = build_index(entries, exclude_organism)
        # Counted, so a holdout that matched nothing is visible. `removed` was left None on
        # this branch, which meant a misspelled organism name -- the only part of the input a
        # caller has to get exactly right, and UniProt's spelling is not always the obvious
        # one -- produced a run that identified the pathogen through its own curated entries
        # while reporting that it had been held out. Zero here is the signal that the two
        # differ, and it costs one subtraction to say so.
        removed = (len(entries) - len(kept)) if exclude_organism else None

    # Three tiers, tried in order of cost. Exact 5-mers answer the easy case in milliseconds.
    # Reduced-alphabet seeding plus Smith-Waterman answers the remote case in seconds, and is
    # only paid for when the first tier finds nothing. Hydropathy answers the case neither can
    # -- a protein with no relative anywhere -- and answers it with a class rather than a name.
    reduced = build_reduced_index(kept) if remote else None
    aligner = _aligner() if remote else None

    candidates = []
    for orf in searched:
        hits = [h for h in identify_local(orf["protein"], index, kept)
                if h["score"] >= minimum_score]
        tier = "exact" if hits else None
        if not hits and remote:
            hits = identify_remote(orf["protein"], reduced, kept, aligner)
            tier = "remote" if hits else None
        # Computed for every ORF, not only for the ones homology failed on. It used to read
        # `None if hits else membrane_topology(...)`, so any hit at all consumed the ORF's
        # class-level answer -- including a hit that was wrong. RSV B's attachment glycoprotein
        # G and human metapneumovirus's G both pass `membrane_topology` as likely surface on
        # their own sequence, by the anchored-and-glycosylated route, which is the correct
        # biology for a type II membrane mucin carrying thirty to forty O-glycans, and "Major
        # surface glycoprotein" is RSV B's ground-truth label in `data/benchmark_panel.json`.
        # That answer was computed, discarded, and replaced by a fish nidovirus.
        #
        # It is carried beside the homology result rather than substituted for it. Which of the
        # two should be published when they disagree is a ranking change and needs the
        # measurement this module requires of ranking changes; the evidence is recorded first so
        # that measurement is possible at all. `membrane_topology` is arithmetic over the
        # sequence -- no network, no reference -- so computing it always costs nothing.
        topology = membrane_topology(orf["protein"])
        if not hits and topology and topology["likely_surface"]:
            tier = "topology"
        # `protein` is carried now, and the omission it replaces was the single break in the
        # onset chain. `orfs` translates every reading frame and this dict then kept the
        # coordinates and dropped the sequence, so the one input a genuinely unknown target can
        # supply was computed and discarded. Everything downstream is keyed on an accession --
        # PDBe coverage, UniProt locations, AlphaFold DB -- and a novel protein has none, so
        # every structural lane returned a true and useless zero for exactly the case this
        # project is about. A single-sequence folding endpoint needs the sequence and nothing
        # else, and this is the only place it exists.
        #
        # The coordinates alone are not a substitute. They are sufficient to re-derive the
        # sequence, but only by re-fetching the genome and re-translating, which makes a second
        # caller reimplement the translation and the frame arithmetic this function already did.
        #
        # Cost of carrying it: the ranked output is `top` rows, twelve by default, so this adds
        # roughly twenty kilobytes to a payload that is already hundreds. The payload shape
        # changes, so `content_sha256` moves for runs recorded from here on; artifacts already
        # on disk keep the digest they were written with, which is what that field is for.
        candidates.append({"orf": {k: orf[k] for k in
                                   ("strand", "frame", "nt_start", "length_aa", "protein")}
                                  | {"segment": orf.get("segment", 0)},
                           "hits": hits, "tier": tier, "topology": topology})

    records = homolog_records([h["accession"] for c in candidates for h in c["hits"]])
    ranked = []
    for candidate in candidates:
        if not candidate["hits"]:
            topology = candidate.get("topology") or {}
            if topology.get("likely_surface"):
                # Named by class, because that is all the evidence supports. A surface antigen
                # with no identifiable relative is still the protein antibodies would be
                # raised against, and returning nothing for it loses the target the system
                # exists to find.
                #
                # **It is NOT scored below every homology hit, and this comment used to say it
                # was.** The features here are fixed, so `rank_score` always returns
                # 3.0*0.0 + 2.0*1 + 1.5*0 + 0.5*0 + 0.8*2 = 3.6, and a homology row clears 3.6
                # only by earning it. Across the recorded panel this row ranks above at least
                # one identified row in 14 of the 17 runs that produce one. In
                # `runs/benchmark/NC_001437_1.json` it sits at rank 5, one place above
                # `Envelope protein E` -- a ground-truth label in `data/benchmark_panel.json`
                # -- and above seven further identified rows.
                #
                # No fixed score can deliver that invariant, which is why the sentence was
                # never true rather than merely stale: the floor a homology row has to clear is
                # 3.0 * MINIMUM_SCORE = 0.36 on the exact tier, and remote-tier hits are not
                # filtered by MINIMUM_SCORE at all.
                #
                # Enforcing it in the ordering instead was measured and refused. Keying either
                # sort on tier sends this row to last place -- adenovirus 4 to 13, RSV B 10 to
                # 22, Japanese encephalitis 5 to 13, hMPV 8 to 20, MERS 9 to 19 -- which is "a
                # penalty for being understudied, aimed at exactly the novel targets this
                # exists to find", the argument `rank_score` already used to reject two surface
                # terms, and it re-creates the loss this row exists to prevent. It also moves a
                # panel verdict across 0.05 (JEV 0.0559 to 0.0385). So the overlap is counted
                # and published rather than sorted away, and any later reordering has to ship
                # with that measurement instead of a claim.
                # `conservation` is 0.0 because there is no relative to measure it against,
                # which is not the same fact as a relative that turned out to conserve
                # nothing -- and the rubric's heaviest term, weight 3.0, reads the two
                # identically. The score stays where it is, deliberately: a class must not
                # outrank a name. What changes is that the record now says the term was
                # never measured, so a reader comparing this row's score with a homology
                # row's can see that the gap is mostly an absence of evidence rather than
                # evidence of a worse target.
                features = {"conservation": 0.0, "surface_exposed": True,
                            "structures_in_family": 0, "druggable_class": 2,
                            "essential_class": False, "keywords_matched": [],
                            "locations": ["predicted surface, from hydropathy"],
                            "from_topology": True,
                            "conservation_is_measured": False,
                            # Explicitly None, not absent. These rows never pass through
                            # `score_candidate`, so with any consumer that defaults a missing
                            # scale their 0.0 would read as a measured k-mer fraction -- which
                            # is the thing `conservation_is_measured: False` exists to prevent,
                            # arriving through the field added to prevent it.
                            "conservation_scale": None,
                            "conservation_at_ceiling": False,
                            "conservation_absent_because":
                                "no identifiable relative, so the term has no reference to "
                                "be measured against and scores zero by default"}
                ranked.append({**candidate, "identified": True,
                               "identification_route": "topology",
                               "surface_route": topology.get("surface_route"),
                               "name": "unidentified probable surface antigen",
                               "nearest_homolog": None, "nearest_organism": None,
                               "features": features, "score": rank_score(features),
                               "conservation": features["conservation"]})
                continue
            ranked.append({**candidate, "identified": False,
                           "identification_route": None,
                           # Named, because this row is the one the pipeline loses and the
                           # reason it loses it is the operating envelope: no relative at any
                           # sensitivity, and no surface signal in its own sequence either.
                           "unidentified_because":
                               "no homolog at either tier and neither topology route fired",
                           "features": None,
                           "score": None, "conservation": None})
            continue
        features = score_candidate(candidate["hits"], records)
        # Conservation is reported beside the score and not only inside it. It is the one
        # feature read from this genome's relationship to the reference; the other four are
        # read off whatever homolog was matched and do not move as that homolog gets more
        # distant. Summed, a close assignment and a remote one differ by at most three of the
        # rubric's nine-odd points, so the number the list sorts by is the number that hides
        # how far the evidence reached. The holdout ladder is why it matters: within a genus
        # conservation carries the assignment, one rank further out there is no assignment at
        # all, and the two cases must not present as the same kind of answer.
        ranked.append({**candidate, "identified": True,
                       # "exact" or "remote": which tier found it, carried beside the score
                       # so the three routes into this list are distinguishable in the record
                       # rather than only in the tier field the cascade artifact reads.
                       "identification_route": candidate.get("tier"),
                       "name": candidate["hits"][0]["name"],
                       "nearest_homolog": candidate["hits"][0]["accession"],
                       "nearest_organism": candidate["hits"][0]["organism"],
                       "features": features, "score": rank_score(features),
                       "conservation": features["conservation"]})
    ranked.sort(key=lambda r: (r["score"] is not None, r["score"] or 0), reverse=True)

    payload = {
        "experiment": "intake-and-target-ranking", "status": "complete",
        "genome_accession": accession, "genome_header": header,
        "genome_length_nt": len(seq), "segments": len(segments),
        "reference_excluded_organism": exclude_organism,
        "reference_excluded_taxid": exclude_taxid,
        "reference_excluded_rank": exclude_rank,
        "reference_excluded_clade": clade,
        "reference_entries_removed": removed,
        "reference_entries": len(kept),
        "minimum_orf_aa": minimum_aa, "minimum_hit_score": minimum_score,
        "reference_scope": scope,
        "orfs_found": len(found), "orfs_ranked": len(searched),
        "orfs_dropped_by_top": len(found) - len(searched),
        # How many of the ranked ORFs are there because they read as surface proteins rather
        # than because they were long. Zero on a genome whose targets are all long anyway,
        # which is most of the recorded ones; non-zero is the case the length cut was losing.
        "orfs_rescued_by_topology": len(rescued),
        # How many identified rows each class row outranks, summed. A topology row's 3.6 is a
        # hard-coded feature vector and is not commensurable with a score a homology row earned
        # -- its `conservation: 0.0` is an absence of evidence under the heaviest term, and its
        # `druggable_class: 2` is asserted from the class on a row whose `keywords_matched` is
        # empty, which is the one term a homology row has to earn. The comment at the topology
        # row says why that is still the right trade. This counts the cost of it per run, so a
        # reader sees the overlap instead of the claim that there is none, and so any later
        # attempt to reorder has a before to measure against.
        # Counted here before decomposition, where a polyprotein is still one row, so this is
        # usually 0 or 1 even on the genomes where the overlap is large. The number that
        # matters is the post-decomposition one, and `output/run_benchmark.py` records it from
        # the same function for exactly that reason.
        "topology_rows_above_identified": topology_overlap(ranked),
        "unidentified": sum(1 for r in ranked if not r["identified"]),
        "rubric": dict(RUBRIC),
        "targets": ranked,
        "sources": [{"name": "NCBI efetch", "url": EFETCH, "licence": "NCBI public domain"},
                    {"name": "UniProt", "url": UNIPROT_SEARCH, "licence": "CC BY 4.0"}],
        "provenance": provenance(),
        "limitations": [
            "A topology row's score is one hard-coded feature vector, always 3.6, and is not "
            "commensurable with a score a homology row earned. It is not below every homology "
            "row and no fixed number could be: the floor an exact-tier hit has to clear is "
            "3.0 * 0.12 = 0.36, and remote-tier hits are not filtered on that threshold at "
            "all. `topology_rows_above_identified` counts the overlap for this run. The row is "
            "kept because a surface antigen with no identifiable relative is still the protein "
            "antibodies would be raised against, and ordering it below every named row was "
            "measured and rejected for burying exactly the novel candidates it exists to find.",
            "`conservation` is not comparable between tiers. The exact tier reports shared "
            "5-mers over the query's k-mer count; the remote tier reports a bit score divided "
            "by 200 and capped at 1.0, so it saturates at 200 bits and a hit ten times past "
            "that reads the same as one exactly at it. Each row carries `conservation_scale` "
            "saying which it is, and `conservation_at_ceiling` saying whether the rubric's "
            "heaviest term is at its ceiling and therefore no longer ordering the list.",
            "The ranking is a stated rubric, not a fitted model. No labelled set of correct "
            "target rankings exists to fit against.",
            "`structures_in_family` is a count of deposited structures and is therefore a "
            "record of research attention as much as of tractability. It carries the smallest "
            "weight for that reason and is capped.",
            "Essentiality and druggability are inferred from UniProt keywords on homologs, "
            "which is an annotation and not a measurement.",
            "Six-frame calling misses frameshifted products and anything under the length "
            "cutoff; overlapping frames are ranked separately."],
        "interpretation": "A ranked list of candidate proteins to attempt to bind, derived "
                          "from homology to curated relatives. No binding, affinity, "
                          "inhibition or antiviral property is predicted for any protein."}
    payload["content_sha256"] = content_digest(payload)
    if record:
        name = f"{accession}{'_heldout' if exclude_organism else ''}.json"
        write_json(ROOT / "data/processed/intake" / name, payload)
    return payload


# ---------------------------------------------------------------------------
# Polyprotein decomposition.

# Below this many residues a mature chain is a cleavage product too small to present a binding
# site worth designing against -- nsp7 at 83 and nsp10 at 139 are retained, the short peptides
# either side of them are not.
MINIMUM_CHAIN_AA = 70

# How close to the top hit a homolog has to score before its annotation may be preferred over
# the top hit's. `decompose` chooses the best-annotated of the top three hits, which is right
# -- the closest sequence is often not the one carrying mature-chain boundaries -- but only
# among homologs that are near-equivalent identifications. The cases that fixed this number,
# and the reason it is a near-tie rather than a tolerance, are recorded at the point of use.
NEAR_EQUIVALENT_RATIO = 0.9


def mature_chains(accession):
    """Mature chains of one UniProt entry, or an empty list if it is not a polyprotein.

    This is the step that turns a ranked ORF into a ranked target. Six-frame calling returns
    ORF1a of a coronavirus as one 4,405-residue product, and no useful countermeasure targets
    a 4,405-residue product: what is targeted is the main protease inside it, at 3264-3569 of
    the polyprotein, which has an inhibitor in clinical use. The entry already carries those
    boundaries, so the decomposition is read rather than inferred -- the same mechanism the
    structural evidence table uses to assign deposited structures to a domain.
    """
    payload = _entry(accession)
    chains = []
    for feature in payload.get("features", []):
        if feature["type"] != "Chain":
            continue
        start = feature["location"]["start"]["value"]
        end = feature["location"]["end"]["value"]
        if None in (start, end) or end - start + 1 < MINIMUM_CHAIN_AA:
            continue
        chains.append({"description": feature.get("description") or "", "start": start,
                       "end": end, "length": end - start + 1})
    # An entry whose only Chain feature spans the whole sequence is not a polyprotein; it is a
    # single protein whose chain happens to be annotated. Returning it would duplicate the ORF
    # it came from as a child of itself.
    whole = payload["sequence"]["length"]
    chains = [c for c in chains if c["length"] < whole * 0.95]
    return chains


def _entry(accession, _cache={}):
    if accession not in _cache:
        url = f"https://rest.uniprot.org/uniprotkb/{urllib.parse.quote(accession)}.json"
        with urllib.request.urlopen(url, timeout=120) as handle:
            _cache[accession] = json.load(handle)
    return _cache[accession]


def family_chain_structures(accessions, description, minimum=0.5):
    """Structures covering the chain with this description, across every homolog that has it.

    Across the family, because a single entry's count answers the wrong question twice over.
    The nearest homolog is often the least studied -- bat coronavirus HKU3's spike carries no
    deposited structure at all, so counting on it alone turned the parent ORF's 84 into a
    zero for every spike chain. And a long chain under-counts against the coverage threshold:
    nsp3 spans 1,922 residues, almost no construct covers half of it, and the entry that holds
    104 structures reports none for that range. Both failures look like "this domain has never
    been solved", which is the opposite of true.

    The chain is matched by description across accessions, so `3C-like proteinase nsp5` is
    found wherever a relative annotates it, and the count is the union of PDB entries covering
    any of those ranges.
    """
    entries = set()
    for accession in accessions:
        try:
            chains = mature_chains(accession)
        except Exception:                                  # noqa: BLE001
            continue
        for chain in chains:
            if chain["description"].lower() != description.lower():
                continue
            try:
                entries |= _covering(accession, chain["start"], chain["end"], minimum)
            except Exception:                              # noqa: BLE001
                continue
    return len(entries)


def _covering(accession, start, end, minimum):
    from .structures_evidence import chains as sifts_chains, covers

    interval = {"start": start, "end": end}
    return {row["pdb_id"].lower() for row in sifts_chains(accession)
            if covers(row, interval, minimum)}


def chain_structures(accession, start, end, minimum=0.5):
    """Deposited structures covering a chain's residue range, via PDBe SIFTS.

    Reuses the assignment rule the structural evidence table established: overlap against a
    range, not a match against a title. A polyprotein's structures are overwhelmingly of one
    domain or another, so counting them against the whole entry tells a ranker nothing about
    which domain is tractable.
    """
    from .structures_evidence import chains as sifts_chains, covers

    interval = {"start": start, "end": end}
    entries = set()
    for row in sifts_chains(accession):
        if covers(row, interval, minimum):
            entries.add(row["pdb_id"].lower())
    return len(entries)


def decompose(ranked, limit=None):
    """Expand polyprotein hits into their mature chains and re-rank.

    Conservation is inherited from the parent ORF, because that is what was measured: the
    k-mer identity was computed against the whole homolog and nothing finer is available
    without aligning the chain itself. Structure counts are recomputed per chain, which is
    where the parent's number was most misleading -- 140 structures against a 4,405-residue
    replicase says nothing about which of its sixteen products anyone has ever solved.
    """
    expanded = []
    for row in ranked:
        if not row.get("identified"):
            continue
        # The best-scoring homolog is not necessarily the best-annotated one, and for
        # decomposition the annotation is what matters. Hepatitis C's polyprotein matches a
        # UniProt entry marked "(Fragment)" at 0.899 -- correctly, it is the closest sequence
        # -- and a fragment carries two mature chains where a complete entry carries nine. The
        # pipeline therefore identified HCV perfectly and then returned three candidates with
        # none of NS3, NS5A or NS5B among them, because the chains it needed were annotated on
        # an entry it had not chosen.
        #
        # So the hit used for decomposition is the best-annotated of the top hits rather than
        # the highest-scoring one. Identity picks the homolog; annotation picks which of the
        # near-equivalent homologs can actually be taken apart.
        #
        # **And near-equivalence is now enforced, because it was only ever asserted.** The
        # sentence above was the rule; the code took `max` on chain count with no reference to
        # score at all, so the worst hit in the top three won outright by being the most
        # annotated. Measured on the recorded benchmark, where it emitted another family's
        # protein inventory as the pathogen's targets:
        #
        #   CCHFV   Dugbe virus Q02004 at 1.000 and 1012 bits, 0 chains, lost to White bream
        #           virus Q008X6 at 0.231 and 46 bits with 12 -- a fish nidovirus supplying all
        #           twelve of a tick-borne bunyavirus's top candidates, with CCHFV's own L
        #           protein pushed to rank 13.
        #   HCV     GB virus B Q69422 at 1.000 and 1142 bits, 9 chains -- the closest known
        #           relative of HCV -- lost to bovine viral diarrhea virus Q96662 at 0.514 and
        #           103 bits with 12, so an `E(rns) glycoprotein` was returned for a virus that
        #           has no Erns.
        #
        # The ratio only has to say "the annotation may break a near-tie and nothing more",
        # which is what the sentence above always claimed. The two swaps worth keeping sit far
        # from the two worth refusing -- HIV-1 reaches a second SIV entry at 0.929 for one more
        # chain, and an un-held-out HCV reaches a second genotype 1a entry at 0.934 -- so the
        # line is not delicately placed. It is stated rather than fitted, and deliberately not
        # chosen by what it does to any target's rank.
        #
        # It is tight, and that is the intended direction. HIV-1's benign swap clears it by
        # 0.029, so a future one could fall below and keep the top hit's annotation instead:
        # one fewer mature chain from the right organism, against twelve from the wrong one.
        hits = row.get("hits", [])
        best_score = (hits[0].get("score") or 0.0) if hits else 0.0
        floor = best_score * NEAR_EQUIVALENT_RATIO
        candidates_by_chain, refused = [], []
        for hit in hits[:3]:
            try:
                count_for_hit = len(mature_chains(hit["accession"]))
            except Exception:                             # noqa: BLE001
                continue
            if (hit.get("score") or 0.0) < floor:
                refused.append({"accession": hit["accession"], "score": hit.get("score"),
                                "organism": hit.get("organism"), "chains": count_for_hit})
                continue
            candidates_by_chain.append((count_for_hit, hit))
        if not candidates_by_chain:
            expanded.append({**row, "decompose_error": "no hit could be read for chains"})
            continue
        count, best_hit = max(candidates_by_chain, key=lambda pair: pair[0])
        accession = best_hit["accession"]
        if accession != row["nearest_homolog"]:
            # The organism moves with the accession. It did not, and the mismatch was the one
            # visible symptom of the swap going wrong: the CCHFV rows above named Dugbe virus
            # while carrying a White bream virus accession, so the row read as a sound
            # same-genus identification and a reader had no way to see otherwise.
            row = {**row, "decomposed_via": accession,
                   "decomposed_via_organism": best_hit.get("organism"),
                   "decomposed_via_score": best_hit.get("score"),
                   "decomposed_via_reason": "better annotated, and within "
                                            f"{NEAR_EQUIVALENT_RATIO:.0%} of the top hit",
                   "decomposed_displaced": row["nearest_homolog"]}
        if refused:
            row = {**row, "decompose_refused_hits": refused}
        try:
            chains = mature_chains(accession)
        except Exception as error:                        # noqa: BLE001
            row = {**row, "decompose_error": f"{type(error).__name__}: {error}"}
            expanded.append(row)
            continue
        if not chains:
            expanded.append(row)
            continue
        parent_length = _entry(accession)["sequence"]["length"]
        inherited_druggable = row["features"]["druggable_class"]
        inherited_essential = row["features"]["essential_class"]
        # Every mature chain, not the first twelve. `limit` defaulted to 12 and sliced
        # `chains` in UniProt feature order, which is N-to-C position on the polyprotein, so
        # what it dropped was whatever the virus encodes last -- and the cut was nowhere
        # documented. Poliovirus P22055 has 14 chains over MINIMUM_CHAIN_AA, so 3C (the
        # protease rupintrivir targets) and 3D (the polymerase) were cut from Poliovirus,
        # Enterovirus A71 and Rhinovirus A: not demoted, never evaluated. On the project's own
        # lead genome the replicase kept nsp1-nsp10, nsp12 and nsp13 and lost nsp14, nsp15 and
        # nsp16 -- all three in DRUGGABLE.
        #
        # Ordering the chains before slicing was the obvious alternative and both candidate
        # orderings are refuted by this module's own prose. Longest-first is the proxy the ORF
        # rescue argues against ("length is a proxy for importance, and it is a proxy biased
        # toward the proteins already known"), and it would drop nsp7 at 83, nsp9 at 113 and
        # nsp10 at 139 -- nsp10 being half of a target this project declares. Ordering by the
        # best DRUGGABLE weight in the description pre-selects the candidate set with the
        # keyword table documented three times as a known weak point, which turns that blind
        # spot from a demotion into an erasure and makes it self-confirming: nsp14 and nsp15
        # match no term in it, so they would be first to go.
        #
        # So the cap is removed rather than reordered, and what paid for it was caching. The
        # only per-chain cost is `family_chain_structures`, whose SIFTS fetch was the one
        # uncached call on this path; memoized per accession, a parent needs three payloads
        # rather than three per chain, and scoring every chain costs about what scoring twelve
        # cost. `limit` stays as a parameter for a caller that wants a cap, and defaults to
        # none; no caller ever passed it.
        considered = chains if limit is None else chains[:limit]
        # A hit that aligned to one region of a polyprotein licenses the chains in that region
        # and no others. Where the hit carries an `entry_span`, chains it does not overlap are
        # dropped -- `overlap > 0` between two intervals the pipeline already holds, with no
        # constant to choose and nothing to calibrate.
        #
        # Measured on `runs/benchmark/NC_001781_1.json`, the worst case: RSV B's attachment
        # glycoprotein G reaches the remote tier, because G is hypervariable enough that exact
        # 5-mers miss it, and lands a 228-column, 22.4%-identity alignment against the White
        # bream virus replicase -- the pneumovirus mucin-like O-glycosylated stalk against a
        # Ser/Thr/Pro/Glu-rich linker, two intrinsically disordered regions sharing no fold and
        # no ancestry. One ORF, one marginal alignment, and `decompose` published twelve mature
        # chains of a fish nidovirus as RSV B's top candidates, with the RdRp row among them
        # being credited as a ground-truth hit for RSV's L target. The alignment covers
        # residues 962-1185, so on span overlap eleven of those twelve rows go.
        #
        # It is also a correctness gain on the capability this tier exists for, which is the
        # part worth stating: the flagship cross-family case in this module's own docstring is
        # SARS-CoV-2 ORF1b against that same accession at E = 8.9e-68, and its span 4645-6800
        # licenses exactly the six ORF1b products and drops the seven ORF1a ones. An ORF1b
        # query cannot contain 3CLpro or nsp2, and those rows were being emitted. Nothing
        # genuine is lost elsewhere: HCV keeps 9 of 9, VEEV 4 of 4, CCHFV 5 of 5, VZV gB 1 of 1.
        #
        # Exact-tier hits carry no span and are unaffected, which is correct -- a k-mer hit is
        # a whole-sequence statement and has no region to scope to.
        span = best_hit.get("entry_span")
        if span:
            licensed = [c for c in considered
                        if c["start"] <= span[1] and c["end"] >= span[0]]
            unlicensed = [c["description"] for c in considered if c not in licensed]
            considered = licensed
        else:
            unlicensed = []
        for chain in considered:
            try:
                structures = family_chain_structures(
                    [h["accession"] for h in row["hits"]], chain["description"])
            except Exception:                             # noqa: BLE001
                structures = 0
            # Recomputed from the chain's own description, not inherited. Inheriting them
            # was the first version and it collapsed the ranking: every mature product of a
            # replicase carries the parent's `essential_class` and the parent's best
            # druggable keyword, so nsp4 scored within 0.02 of nsp12 and the decomposition
            # bought nothing. What differentiates a 3C-like proteinase from a non-structural
            # protein of unknown function is the description, which is the one field that is
            # per-chain.
            features = dict(row["features"])
            features["structures_in_family"] = structures
            # Localisation resolved for this chain, on the same argument the structure count
            # above already won. `score_candidate` reads SUBCELLULAR LOCATION off the whole
            # entry, because at that point there is no chain to scope to -- but a polyprotein's
            # comments are scoped per molecule, so the entry-wide read is a union over products
            # in different compartments and `any()` over it makes every chain of one parent
            # surface-exposed together. On a flavivirus that is the entire job of the term:
            # envelope E, the target of every neutralising antibody, scores the same 2.0 as
            # cytosolic NS3. The right answer was already being computed one stage later and
            # used only for the amenability gate, so the two verdicts sat in the same published
            # record disagreeing.
            #
            # Guarded on the scope, following the one existing precedent for this call. An
            # entry with no `molecule`-scoped comments returns its union, and overwriting a
            # union with itself under a fraction rule would silently restate the entry read as
            # a chain read. Where the scope is not "chain" the parent value stands and the
            # record says so.
            #
            # Free in network terms: `_entry(accession)` is already warm from `parent_length`
            # two lines above, and `chain_locations` reads that same cached payload.
            scoped = chain_locations(accession, chain["description"])
            features["location_scope"] = scoped["scope"]
            if scoped["scope"] == "chain":
                verdict = exposed_from_locations(scoped["locations"])
                features["surface_exposed_from_parent_entry"] = row["features"].get(
                    "surface_exposed")
                features["surface_exposed"] = verdict["exposed"]
                features["exposed_fraction_of_locations"] = verdict["fraction"]
                features["locations"] = sorted(scoped["locations"])
                features["locations_scored"] = len(scoped["locations"])
            described = chain["description"].lower()
            from_description = max(
                [weight for term, weight in DRUGGABLE.items() if term in described] or [0])
            essential_from_description = any(term in described for term in ESSENTIAL)
            # A chain that is most of its parent keeps the parent's character; a small
            # cleavage product does not. Haemagglutinin cleaves into HA1 and HA2 at 58% and
            # 39% of the precursor and both are still haemagglutinin, so dropping the parent's
            # keywords there discards the only evidence those chains have. A coronavirus
            # replicase cleaves into sixteen products at a few percent each, and letting nsp4
            # inherit "protease" from the polyprotein is what collapsed the ranking before.
            major = chain["length"] >= 0.3 * parent_length if parent_length else False
            features["druggable_class"] = (max(from_description, inherited_druggable)
                                           if major else from_description)
            features["essential_class"] = ((essential_from_description or inherited_essential)
                                           if major else essential_from_description)
            features["chain_description"] = chain["description"]
            expanded.append({
                "orf": row["orf"], "identified": True,
                "name": chain["description"] or row["name"],
                "nearest_homolog": accession,
                # The organism of the accession this chain was actually read from, which is not
                # always the ranked hit's. These two disagreed on 32 recorded rows, and the
                # disagreement was silent: every CCHFV candidate named Dugbe virus while
                # carrying a White bream virus accession, so the rows read as a sound
                # same-genus identification. The swap is now floored at a near-tie, which stops
                # the bad cases, and reporting the organism that goes with the accession is
                # what makes a surviving swap visible rather than merely permitted.
                "nearest_organism": best_hit.get("organism") or row["nearest_organism"],
                "decomposed_via": row.get("decomposed_via"),
                "decomposed_displaced": row.get("decomposed_displaced"),
                # Carried onto the chain rows, because the parent row they replace is the only
                # thing that held it. The first version of this fix recorded the refused hits on
                # the parent and then built chain rows from an explicit key list that omitted
                # them, so a run whose swap had been correctly refused reported no refusals at
                # all -- the same way `decomposed_via` was being written onto a dict that was
                # discarded. A guard that leaves no trace when it fires is indistinguishable
                # from one that never fired.
                "decompose_refused_hits": row.get("decompose_refused_hits"),
                # So a cap, if a caller sets one, is visible in the record instead of looking
                # like a parent with no further products. That is how the old default read for
                # three picornaviruses and for SARS-CoV-2.
                "chains_in_parent": len(chains),
                "chains_scored": len(considered),
                # Which chains the hit's aligned region did not reach, and the span that
                # decided it. Recorded rather than silently absent, because "this hit covers
                # residues 962-1185 of a 6,872-residue entry" is the fact that made eleven of
                # RSV B's twelve published candidates wrong, and no artifact carried it.
                "chains_outside_aligned_span": unlicensed,
                "aligned_span_on_homolog": span,
                "parent": row["name"], "chain_range": [chain["start"], chain["end"]],
                "length_aa": chain["length"],
                "hits": row["hits"], "features": features,
                "score": rank_score(features),
                # Carried from the parent ORF, which is the only thing that was matched. A
                # mature chain is cut out of a hit, never aligned on its own, so its
                # conservation is the parent's and the tier that found the parent is the
                # tier that found it. Dropping them here is what made the decomposed list
                # look like it had been identified more directly than it was.
                "tier": row.get("tier"),
                "conservation": features["conservation"]})
    # A polyprotein reached by two ORFs -- ORF1a and ORF1b of a coronavirus both hit a
    # replicase entry -- decomposes twice into the same chains. Keep the better-scoring copy.
    best = {}
    for row in expanded:
        # Keyed on the description alone, not on the accession it came from. The same mature
        # product is reached through several relatives -- nsp5 arrives once via bat
        # coronavirus HKU3 and again via SARS-CoV -- and those are one target, assigned twice
        # with different confidence, not two candidates to rank against each other.
        key = (row.get("name") or "").lower()
        # Class rows key on their ORF as well as their name, because every one of them is
        # called "unidentified probable surface antigen". Keyed on the name alone they all
        # collapsed into one: three rescued surface antigens at three positions in a genome
        # published as a single candidate, and because they all score exactly 3.6 the survivor
        # was whichever the iteration reached first. The name-only key is right for the case it
        # was written for -- one mature product reached through several relatives is one target
        # assigned twice -- and wrong here, where the name carries no identity at all.
        if not row.get("identified") or row.get("tier") == "topology":
            orf = row.get("orf") or {}
            key = (key, orf.get("segment"), orf.get("strand"), orf.get("nt_start"))
        if key not in best or (row["score"] or 0) > (best[key]["score"] or 0):
            best[key] = row
    expanded = sorted(best.values(), key=lambda r: r["score"] or 0, reverse=True)
    return expanded


# ---------------------------------------------------------------------------
# Graded taxonomic holdout.

TAXONOMY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Ranks a holdout can be taken at, outermost last. Holding out a species asks whether the
# pipeline can identify a protein it has never seen from a close relative; holding out a
# family asks whether it can do so from a distant one, which is the case the theory of change
# actually describes and the one the first validation did not test.
RANKS = ("species", "genus", "family", "order")


def lineages(taxids, chunk=200, cache=None):
    """taxid -> {rank: taxid} for every rank in RANKS, from NCBI taxonomy.

    Cached on disk because the reference carries a few thousand distinct taxa and the lineage
    of a species does not change between runs. The cache is keyed on taxid, so adding a
    pathogen later costs only its own lookups.
    """
    import xml.etree.ElementTree as ET

    cache = cache or ROOT / "data/raw/reference/lineages.json"
    known = json.loads(cache.read_text()) if cache.exists() else {}
    missing = sorted({str(t) for t in taxids if t and str(t) not in known})

    for start in range(0, len(missing), chunk):
        batch = missing[start:start + chunk]
        # POSTed, not in the URL. A few hundred ids exceeds what a GET query string carries
        # comfortably, and NCBI answers an over-long one with a 414 or silently truncates.
        body = urllib.parse.urlencode({"db": "taxonomy", "id": ",".join(batch),
                                       "retmode": "xml"}).encode()
        # NCBI permits three requests a second without a key and answers a burst with 429.
        # Retried with backoff rather than failed, because a lineage fetch is thousands of
        # taxa and losing the run to one throttle is avoidable; the sleep between batches
        # keeps it under the limit in the first place.
        for attempt in range(6):
            try:
                request = urllib.request.Request(TAXONOMY, data=body)
                with urllib.request.urlopen(request, timeout=180) as handle:
                    tree = ET.fromstring(handle.read())
                break
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503) or attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        time.sleep(0.4)
        for taxon in tree.findall("Taxon"):
            identifier = taxon.findtext("TaxId")
            ranks = {}
            own = taxon.findtext("Rank")
            if own in RANKS:
                ranks[own] = identifier
            for parent in taxon.findall("./LineageEx/Taxon"):
                rank = parent.findtext("Rank")
                if rank in RANKS:
                    ranks[rank] = parent.findtext("TaxId")
            known[identifier] = ranks
        # Written after each batch: a taxonomy fetch of several thousand ids takes minutes and
        # losing all of it to one timeout at the end is avoidable.
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(known))
    return known


def clade_of(taxid, rank, known):
    """The ancestor of `taxid` at `rank`, or None if the lineage does not reach it."""
    return (known.get(str(taxid)) or {}).get(rank)


def build_index_excluding_clade(entries, taxid, rank, known, k=KMER):
    """Index the reference with everything in one clade removed.

    The graded version of `exclude_organism`, and the difference is the whole point. Excluding
    an organism by name leaves its genus and family in place, so SARS-CoV-2 is still
    identified through bat coronavirus HKU3 at 46% identity -- a close relative in the same
    subgenus. Excluding the family removes every coronavirus, which is what a genuinely new
    pathogen looks like, and whether identification survives that is not something the first
    validation could answer.

    Returns the index, the kept entries, and how many were removed, because a holdout that
    removed nothing is indistinguishable in its output from one that worked.
    """
    target = clade_of(taxid, rank, known)
    if target is None:
        raise ValueError(f"taxid {taxid} has no {rank} in its lineage, so a {rank}-level "
                         f"holdout would silently remove nothing")
    kept = [e for e in entries if clade_of(e.get("taxid"), rank, known) != target]
    removed = len(entries) - len(kept)
    index = {}
    for position, entry in enumerate(kept):
        sequence = entry["sequence"]
        for offset in range(len(sequence) - k + 1):
            index.setdefault(sequence[offset:offset + k], set()).add(position)
    return index, kept, removed, target


# ---------------------------------------------------------------------------
# Remote homology, and the proteins no homology search can reach.

# Murphy's ten-letter reduced alphabet. Exact 5-mers over the twenty-letter alphabet need
# roughly 40% identity before two homologs share any, which is why identification falls off a
# cliff one taxonomic rank out. Grouping residues by physicochemical class keeps seeds alive
# at much lower identity, at the cost of many more chance matches -- which is what the
# alignment stage after it exists to throw away.
REDUCED_GROUPS = ("LVIM", "C", "A", "G", "ST", "P", "FYW", "EDNQ", "KR", "H")
REDUCED = {residue: chr(ord("a") + group)
           for group, residues in enumerate(REDUCED_GROUPS) for residue in residues}

# Karlin-Altschul parameters for BLOSUM62 with an 11/1 affine gap, the values BLAST uses.
# They turn a raw Smith-Waterman score into a bit score and an expectation, which is the only
# way to compare a 4,400-residue replicase against a 222-residue membrane protein and say
# which hit means something.
LAMBDA, KAPPA = 0.267, 0.041

# An expectation below this counts as a hit. Calibrated against the Coronaviridae-removed run,
# where the replicase finds a genuine nidovirus homolog at E near zero while the spike and the
# nucleoprotein -- which have no relatives outside the family -- score E of 20 and 0.35.
MAX_EVALUE = 1e-3


def reduce_sequence(seq):
    return "".join(REDUCED.get(residue, "x") for residue in seq)


def bit_score(raw):
    import math

    return (LAMBDA * raw - math.log(KAPPA)) / math.log(2)


def expectation(raw, query_len, database_residues):
    return query_len * database_residues * (2.0 ** -bit_score(raw))


def _aligner():
    from Bio import Align
    from Bio.Align import substitution_matrices

    aligner = Align.PairwiseAligner()
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score, aligner.extend_gap_score = -11, -1
    aligner.mode = "local"
    return aligner


def aligned_span(aligner, query, subject):
    """Which residues of `subject` the local alignment actually covers, 1-based inclusive.

    Recovered only for the handful of hits that already passed the E-value gate, never for the
    whole prefilter: `aligner.score` is what the gate needs and is three times cheaper than
    `aligner.align`, so scoring 120 subjects and then aligning the two or three survivors costs
    essentially what scoring alone cost.

    The span is what makes a remote hit's *scope* legible. Without it a 223-residue alignment
    against one stretch of a 6,872-residue polyprotein was indistinguishable, downstream, from
    an identification of the whole entry -- and `decompose` then emitted every mature chain of
    that entry as a separate candidate for the query. Returns None if no alignment can be
    recovered, and callers must treat that as "scope unknown" rather than as "covers
    everything".
    """
    try:
        alignment = aligner.align(query, subject)[0]
    except Exception:                                      # noqa: BLE001
        return None
    # `alignment.aligned` is a numpy array, so every test here is on a length rather than on
    # truthiness: `not blocks` raises "truth value of an array with more than one element is
    # ambiguous" on exactly the alignments that have something to report.
    blocks = getattr(alignment, "aligned", None)
    if blocks is None or len(blocks) < 2 or len(blocks[1]) == 0:
        return None
    starts = [int(start) for start, _end in blocks[1]]
    ends = [int(end) for _start, end in blocks[1]]
    return [min(starts) + 1, max(ends)]


def build_reduced_index(entries, k=KMER):
    index = {}
    for position, entry in enumerate(entries):
        reduced = reduce_sequence(entry["sequence"])
        for offset in range(len(reduced) - k + 1):
            index.setdefault(reduced[offset:offset + k], set()).add(position)
    return index


def identify_remote(protein, index, kept, aligner=None, k=KMER, prefilter=120, top=3):
    """Identify one ORF by remote homology: reduced-alphabet seeds, then alignment.

    The two stages do different jobs and neither works alone. Reduced-alphabet seeding is
    sensitive and indiscriminate -- it proposes candidates a twenty-letter seed would never
    reach, and most of them are chance. Smith-Waterman with BLOSUM62 scores each properly, and
    the Karlin-Altschul expectation makes scores comparable across query lengths so a
    threshold means the same thing for a replicase and for a membrane protein.

    What it cannot do is the point of the next function. With all of Coronaviridae removed
    this finds the replicase through a torovirus at E near zero, because nidovirus replicases
    are genuinely homologous across families. It finds nothing for the spike, because the
    spike has no homolog outside the family. That is biology and not a tuning failure, and no
    sensitivity setting changes it.

    **The torovirus in that sentence is also this tier's worst false positive, and the
    coincidence is exact rather than illustrative.** The entry is Q008X6, the White bream virus
    replicase 1ab. Against SARS-CoV-2's ORF1b it is the celebrated hit, at E = 8.9e-68. Against
    the attachment glycoprotein G of RSV B and of human metapneumovirus it is a 22.4%-identity
    alignment of one disordered mucin stalk against another, clearing `MAX_EVALUE` by 0.86 bits
    -- less than a fifth of a single BLOSUM62 match -- and it supplied twelve mature chains of a
    fish nidovirus as each pathogen's top candidates. One reference entry, both roles, 64 orders
    of magnitude apart in E.

    So a gate on hit quality could not have told them apart, and the measurement in
    `runs/remote_span_licensing.json` says nothing else could either: of thirteen quantities
    tried none separates the two populations, and the one that appeared to is
    entry-length-over-query-length in disguise. What separates them is *scope* -- the celebrated
    hit covers residues 4645-6800 of the entry and the artifact covers 962-1185 -- which is why
    `entry_span` is recorded here and why `decompose` licenses only the chains a span reaches.

    The sharper finding in that record is not fixed and is worth reading before trusting this
    tier: RSV G's correct identification, bovine RSV G at E = 3.11e-31, was in the reference the
    whole time at reduced-alphabet seed rank 962, outside `prefilter`. Seed count scales with
    entry length, so the selection step prefers long polyproteins -- median selected entry 4,060
    residues against a database median of 289 -- and discarded a same-genus homolog in favour of
    a cross-order artifact. The tier's specificity problem is a symptom; its sensitivity problem
    is the disease.
    """
    import collections

    aligner = aligner or _aligner()
    database_residues = sum(len(entry["sequence"]) for entry in kept)
    reduced = reduce_sequence(protein)
    seeds = {reduced[i:i + k] for i in range(len(reduced) - k + 1)}

    counts = collections.Counter()
    for seed in seeds:
        for position in index.get(seed, ()):
            counts[position] += 1

    # Totally ordered for the same reason as `identify_local`, and with the same half mattering
    # most: the prefilter decides which `prefilter` entries get an alignment at all, and seeds
    # tying at the boundary were resolved by set iteration order. Measured across seeds, two
    # entries tied at 28 reduced-alphabet seeds swapped prefilter places and moved Q0Q469
    # between indices 29, 28 and 23, which decides `hits[0]` whenever their E-values also tie --
    # and the recorded NS8 hit list holds two hits at an identical E-value of 1.027e-41.
    ordered = sorted(counts.items(),
                     key=lambda item: (-item[1], kept[item[0]]["accession"]))[:prefilter]
    scored = []
    for position, _shared in ordered:
        entry = kept[position]
        raw = aligner.score(protein, entry["sequence"])
        evalue = expectation(raw, len(protein), database_residues)
        if evalue > MAX_EVALUE:
            continue
        scored.append({"accession": entry["accession"], "name": entry["name"],
                       "organism": entry["organism"],
                       "bits": round(bit_score(raw), 1), "evalue": evalue,
                       # `score` here is NOT the same quantity as `identify_local`'s, and the
                       # scale is recorded rather than reconciled. See the note on
                       # `conservation_scale` in `score_candidate` for why rescaling is refused.
                       "score": round(min(1.0, bit_score(raw) / 200.0), 4),
                       "score_scale": "bits_over_200",
                       # Which residues of the entry the alignment covers. Recorded because a
                       # remote hit against a polyprotein is a claim about a region, and
                       # nothing downstream could previously tell a 223-residue local
                       # alignment from an identification of all 6,872 residues.
                       "entry_span": aligned_span(aligner, protein, entry["sequence"]),
                       "entry_length": len(entry["sequence"])})
    # Equal E-value implies equal raw score implies equal bits, so the accession is the only
    # further key this needs.
    scored.sort(key=lambda hit: (hit["evalue"], hit["accession"]))
    return scored[:top]


# Kyte-Doolittle hydropathy, for the proteins homology cannot reach.
HYDROPATHY = {"A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5, "Q": -3.5, "E": -3.5,
              "G": -0.4, "H": -3.2, "I": 4.5, "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8,
              "P": -1.6, "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2}


def membrane_topology(protein, window=19, threshold=1.6):
    """Signal peptide and transmembrane segments, from hydropathy alone.

    This is the path for a protein with no identifiable relative, and it exists because the
    holdout ladder showed what the alternative is: for a pathogen outside every sequenced
    family, the spike-equivalent returns no hit and therefore no target, and the
    spike-equivalent is the antibody target. Losing it is losing the thing the system is for.

    Homology is not the only evidence. A viral surface glycoprotein is recognisable from its
    own sequence: an N-terminal signal peptide, one or more hydrophobic segments long enough
    to cross a membrane, and N-linked glycosylation sequons on the ectodomain. None of that
    requires a relative. It identifies a *class* rather than a protein, which is weaker than
    a homology hit and is the right kind of weaker -- it says "this is probably a surface
    antigen worth raising antibodies against" without claiming to know which one.
    """
    scores = [HYDROPATHY.get(residue, 0.0) for residue in protein]
    if len(scores) < window:
        return {"signal_peptide": False, "transmembrane_segments": 0, "sequons": 0,
                "likely_surface": False}

    means = [sum(scores[i:i + window]) / window for i in range(len(scores) - window + 1)]
    segments, inside = 0, False
    for value in means:
        if value >= threshold and not inside:
            segments, inside = segments + 1, True
        elif value < threshold:
            inside = False

    # A signal peptide is a hydrophobic core in the first ~40 residues. Counted separately
    # from the transmembrane segments it would otherwise be mistaken for.
    head = means[:max(1, 40 - window)]
    signal = bool(head) and max(head) >= threshold

    sequons = len(re.findall(r"N[^P][ST]", protein))
    # A polyprotein is not a surface antigen, however many membrane segments it carries. The
    # SARS-CoV-2 replicase 1a has twenty-four and twenty-eight sequons -- nsp3, nsp4 and nsp6
    # are genuinely membrane-bound -- and reads as a surface protein on hydropathy alone. In
    # the cascade this rarely matters, because a replicase is identified by homology long
    # before the de-novo path is reached, but the function is also called directly and should
    # not return an answer it cannot support.
    plausible_length = 100 <= len(protein) <= 2000

    # Two routes, because requiring glycosylation made this lane narrower than the paragraph
    # above it claims. `sequons >= 2` describes an enveloped virus's glycoprotein, and that
    # was the only way through: a novel surface protein that is anchored but not glycosylated
    # failed the gate and left the pipeline with no target at all -- the same loss the
    # docstring calls "losing the thing the system is for", reintroduced one condition lower
    # down. The second route asks for a signal peptide *and* a membrane segment, which is
    # stronger evidence of a surface protein than either alone and does not depend on
    # glycosylation. Recorded by name, because the two routes are not equally strong and a
    # reader has to be able to see which one fired.
    route = None
    if plausible_length and (signal or segments >= 1) and sequons >= 2:
        route = "anchored_and_glycosylated"
    elif plausible_length and signal and segments >= 1:
        route = "signal_peptide_with_membrane_anchor"

    return {"signal_peptide": signal, "transmembrane_segments": segments,
            "sequons": sequons, "plausible_antigen_length": plausible_length,
            "surface_route": route,
            # What neither route reaches, stated here rather than left to be discovered: a
            # non-enveloped virus's capsid has no signal peptide and no membrane anchor, so
            # it is recovered by neither. That needs a fold-and-search method, not a
            # hydropathy threshold.
            "likely_surface": route is not None}


# ---------------------------------------------------------------------------
# The size beyond which this pipeline's heuristics stop being safe.

# The largest genome in the families this project models is around 200 kb. Above this, every
# heuristic here -- six-frame calling, ranking by length, a reviewed viral reference -- stops
# being safe at once, so `run_intake` refuses rather than ranking: all three fail silently and
# a confident wrong shortlist is the one output worse than none.
#
# **This is a limit on the code, not a definition of a virus.** The name and the comment here
# previously said a genome over this size "is something else", which is false: Mimivirus is
# ~1.18 Mb, Pandoravirus ~2.47 Mb and Pithovirus ~610 kb, and all of them are viruses. The
# gate therefore does not reject non-viral input, it rejects *viruses this pipeline was never
# calibrated on* -- and a comment claiming virology was hiding that behind a fact. Covering
# them is real work (gene calling that is not six-frame, a reference that includes them), not
# a larger constant, which is why the number stays and the justification changes.
VIRAL_GENOME_LIMIT_NT = 500_000


# ---------------------------------------------------------------------------
# Whether a ranking beat chance, which "top 8" does not answer.

def rank_enrichment(ranks, candidates):
    """Probability a random ranking would place these targets at least this high.

    The statistic exists because the claim it replaces could not fail. "Every expected target
    ranks in the top eight" was reported for three pathogens, two of which produced eight and
    eleven candidates in total -- so for Nipah it says only that the targets were somewhere in
    the list, and it would have read as a pass had the ranking been random.

    Computed exactly over all placements of k targets among n candidates, which is tractable
    at viral scale, and by normal approximation when the binomial coefficient is too large to
    enumerate, which a few thousand candidates reaches.

    Returns the one-sided p, the mean percentile, and which method was used, because a p of
    0.07 on eight candidates and a p of 0.0004 on sixteen hundred are different kinds of
    evidence and the reader needs to see which they have.
    """
    import itertools
    import math

    k, total = len(ranks), int(candidates)
    if not k or total < k:
        return {"p": None, "mean_percentile": None, "method": "not computable",
                "targets": k, "candidates": total}
    observed = sum(ranks)
    mean_percentile = round(sum(r / total for r in ranks) / k * 100, 1)

    if math.comb(total, k) > 5_000_000:
        mu = k * (total + 1) / 2
        variance = k * (total - k) * (total + 1) / 12
        z = (observed - mu) / math.sqrt(variance)
        p = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        method = "normal approximation"
    else:
        placements = math.comb(total, k)
        atleast = sum(1 for combo in itertools.combinations(range(1, total + 1), k)
                      if sum(combo) <= observed)
        p, method = atleast / placements, "exact"

    # What the test could achieve at this size if the ranking were perfect. Without it a
    # non-significant result is unreadable, because two very different things produce one:
    # a ranking that did badly, and a test with no room to show that it did well. Nipah
    # scores three targets among eight candidates, where the best attainable p is 0.018 --
    # so even a flawless ranking would be weak evidence, and the observed 0.071 says almost
    # nothing about the ranking.
    floor = sum(range(1, k + 1))
    if math.comb(total, k) > 5_000_000:
        mu = k * (total + 1) / 2
        variance = k * (total - k) * (total + 1) / 12
        z = (floor - mu) / math.sqrt(variance)
        best = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    else:
        best = sum(1 for combo in itertools.combinations(range(1, total + 1), k)
                   if sum(combo) <= floor) / math.comb(total, k)
    powered = best < 0.01

    # Three outcomes, not two. A test that cannot produce strong evidence in its best case
    # must not be reported as a failure when it produces weak evidence in the actual one.
    if p < 0.05:
        verdict = "beats chance"
    elif powered:
        verdict = "does not beat chance"
    else:
        verdict = "not established (underpowered)"

    return {"p": p, "mean_percentile": mean_percentile, "method": method,
            "targets": k, "candidates": total, "rank_sum": observed,
            "chance_rank_sum": k * (total + 1) / 2,
            "best_attainable_p": best, "adequately_powered": powered,
            "verdict": verdict,
            "beats_chance_at_05": bool(p is not None and p < 0.05),
            # The denominator moves the answer and must therefore be fixed by a rule applied
            # to every pathogen, never chosen per pathogen. Nipah's observed ranks of 1, 3 and
            # 4 give p = 0.071 against eight candidates and p = 0.006 against seventeen, with
            # nothing about the ranking changed -- so counting unidentified reading frames as
            # candidates would rescue it and would be selecting the denominator that gives
            # the wanted answer. The rule here is candidates the pipeline identified and
            # ranked, and it is the same rule at every genome size.
            "denominator_rule": "candidates identified and ranked by the pipeline"}
