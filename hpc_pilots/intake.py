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
`identify_remote` for the case where k-mer identity is too blunt, which is remote homology
below roughly 25% identity.

**The control that makes this a novel-pathogen pipeline rather than a lookup.** Every protein
of SARS-CoV-2 is in the reference, so identifying its ORFs against the full set is a database
query with a known answer. `exclude_organism` removes the pathogen from the reference before
indexing, which is what the system would face for something genuinely new. Held out, all eight
real ORFs still resolve to the right protein by function -- the replicase to bat coronavirus
HKU3, the spike to HKU3's spike, the nucleoprotein to Rp3/2004 -- with scores falling from
1.000 to between 0.30 and 0.84. The assignment survives and its confidence visibly does not,
which is the behaviour wanted. The spurious reverse-strand ORF scores 0.048 against a phage
protein and is rejected by the same threshold.

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
    with urllib.request.urlopen(f"{EFETCH}?{query}", timeout=180) as handle:
        text = handle.read().decode()
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
REFERENCE_QUERY = "(taxonomy_id:10239) AND (reviewed:true)"

KMER = 5

# Below this, a hit is noise. Calibrated on the one spurious ORF in the SARS-CoV-2 six-frame
# call, which scores 0.048 against an unrelated phage protein, against 0.30 for the weakest
# real assignment when the pathogen itself is held out of the reference.
MINIMUM_SCORE = 0.12


def fetch_reference(query=REFERENCE_QUERY, path=None):
    """The reference protein set, downloaded once and cached as FASTA."""
    import gzip

    path = path or ROOT / "data/raw/reference/viral_reviewed.fasta"
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
    entries, accession, name, organism, chunks = [], None, None, None, []

    def flush():
        if accession:
            entries.append({"accession": accession, "name": name,
                            "organism": organism, "sequence": "".join(chunks)})

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
            chunks = []
        else:
            chunks.append(line.strip())
    flush()
    return entries


def build_index(entries, exclude_organism=None, k=KMER):
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
    return index, kept


def identify_local(protein, index, kept, k=KMER, top=3):
    """Best reference hits for one ORF, by shared k-mer fraction.

    The score is shared k-mers over the smaller of the two k-mer sets, so a short query against
    a long polyprotein is not penalised for the length it does not cover -- which matters here,
    because a frameshifted ORF half is exactly that case.
    """
    import collections

    query = {protein[i:i + k] for i in range(len(protein) - k + 1)}
    counts = collections.Counter()
    for kmer in query:
        for position in index.get(kmer, ()):
            counts[position] += 1

    scored = []
    for position, shared in counts.most_common(50):
        entry = kept[position]
        denominator = max(1, min(len(query), len(entry["sequence"]) - k + 1))
        scored.append({"accession": entry["accession"], "name": entry["name"],
                       "organism": entry["organism"],
                       "score": round(shared / denominator, 4), "shared_kmers": shared})
    scored.sort(key=lambda hit: -hit["score"])
    return scored[:top]


# ---------------------------------------------------------------------------
# Ranking. Every feature is read off the homologs, never off the query protein.

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

# Subcellular locations that put a protein where a binder can reach it without crossing a
# membrane. Matched as substrings against UniProt's own location values.
EXPOSED = ("virion membrane", "host cell membrane", "virion surface", "secreted",
           "cell surface", "host extracellular")

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
             "hemagglutinin": 2, "haemagglutinin": 2, "neuraminidase": 3}

# Keywords implying the protein is required for the virus to replicate at all. Binding a
# dispensable protein cannot become a therapeutic, which is the same argument the dossier's
# essentiality axis makes.
ESSENTIAL = ("polymerase", "helicase", "protease", "proteinase", "peptidase", "capsid",
             "nucleoprotein", "spike", "fusion", "attachment", "entry into host",
             "hemagglutinin", "haemagglutinin", "neuraminidase")


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

    exposed = any(any(e in location for e in EXPOSED) for location in locations)
    druggable = max([weight for term, weight in DRUGGABLE.items()
                     if any(term in k for k in keywords)] or [0])
    essential = any(any(term in k for k in keywords) for term in ESSENTIAL)
    conservation = hits[0]["score"] if hits else 0.0

    return {
        "conservation": round(conservation, 4),
        "surface_exposed": exposed,
        "structures_in_family": structures,
        "druggable_class": druggable,
        "essential_class": essential,
        "keywords_matched": sorted(k for k in keywords
                                   if any(t in k for t in DRUGGABLE) or
                                   any(t in k for t in ESSENTIAL))[:8],
        "locations": sorted(locations)[:4]}


def rank_score(features):
    """One number from the features, with every term visible.

    Deliberately simple and deliberately not fitted. There is no labelled set of "correct
    target rankings for a novel pathogen" to fit against -- that absence is why the project
    grades decisions rather than predictions -- so a learned weighting here would be fitted to
    whatever its author already believed. A stated rubric is honest about being one.
    """
    return round(
        3.0 * features["conservation"]
        + 2.0 * float(features["surface_exposed"])
        + 1.5 * float(features["essential_class"])
        + 0.8 * features["druggable_class"]
        + 0.5 * min(1.0, features["structures_in_family"] / 50.0), 4)


def run_intake(accession, exclude_organism=None, minimum_aa=MINIMUM_AA, top=12,
               minimum_score=MINIMUM_SCORE, record=True):
    """Genome accession to a ranked list of candidate binding targets.

    `exclude_organism` holds the pathogen out of the reference, which is the only honest way
    to test the pipeline on a genome whose every protein is already curated.
    """
    segments = genome_records(accession)
    found = []
    for index, (segment_header, segment) in enumerate(segments):
        for orf in orfs(segment, minimum_aa):
            found.append({**orf, "segment": index,
                          "segment_header": segment_header})
    found.sort(key=lambda o: -o["length_aa"])
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

    header = segments[0][0] if len(segments) == 1 else \
        f"{len(segments)} segments: " + "; ".join(h.split()[0] for h, _ in segments)
    seq = "".join(part for _, part in segments)

    entries = read_fasta(fetch_reference())
    index, kept = build_index(entries, exclude_organism)

    candidates = []
    for orf in searched:
        hits = [h for h in identify_local(orf["protein"], index, kept)
                if h["score"] >= minimum_score]
        candidates.append({"orf": {k: orf[k] for k in
                                   ("strand", "frame", "nt_start", "length_aa")}
                                  | {"segment": orf.get("segment", 0)},
                           "hits": hits})

    records = homolog_records([h["accession"] for c in candidates for h in c["hits"]])
    ranked = []
    for candidate in candidates:
        if not candidate["hits"]:
            ranked.append({**candidate, "identified": False, "features": None, "score": None})
            continue
        features = score_candidate(candidate["hits"], records)
        ranked.append({**candidate, "identified": True,
                       "name": candidate["hits"][0]["name"],
                       "nearest_homolog": candidate["hits"][0]["accession"],
                       "nearest_organism": candidate["hits"][0]["organism"],
                       "features": features, "score": rank_score(features)})
    ranked.sort(key=lambda r: (r["score"] is not None, r["score"] or 0), reverse=True)

    payload = {
        "experiment": "intake-and-target-ranking", "status": "complete",
        "genome_accession": accession, "genome_header": header,
        "genome_length_nt": len(seq), "segments": len(segments),
        "reference_excluded_organism": exclude_organism,
        "reference_entries": len(kept),
        "minimum_orf_aa": minimum_aa, "minimum_hit_score": minimum_score,
        "orfs_found": len(found), "orfs_ranked": len(searched),
        "orfs_dropped_by_top": len(found) - len(searched),
        "unidentified": sum(1 for r in ranked if not r["identified"]),
        "rubric": {"conservation": 3.0, "surface_exposed": 2.0, "essential_class": 1.5,
                   "druggable_class": 0.8, "structures_in_family_capped": 0.5},
        "targets": ranked,
        "sources": [{"name": "NCBI efetch", "url": EFETCH, "licence": "NCBI public domain"},
                    {"name": "UniProt", "url": UNIPROT_SEARCH, "licence": "CC BY 4.0"}],
        "provenance": provenance(),
        "limitations": [
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


def decompose(ranked, limit=12):
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
        accession = row["nearest_homolog"]
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
        for chain in chains[:limit]:
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
                "nearest_organism": row["nearest_organism"],
                "parent": row["name"], "chain_range": [chain["start"], chain["end"]],
                "length_aa": chain["length"],
                "hits": row["hits"], "features": features,
                "score": rank_score(features)})
    # A polyprotein reached by two ORFs -- ORF1a and ORF1b of a coronavirus both hit a
    # replicase entry -- decomposes twice into the same chains. Keep the better-scoring copy.
    best = {}
    for row in expanded:
        # Keyed on the description alone, not on the accession it came from. The same mature
        # product is reached through several relatives -- nsp5 arrives once via bat
        # coronavirus HKU3 and again via SARS-CoV -- and those are one target, assigned twice
        # with different confidence, not two candidates to rank against each other.
        key = (row.get("name") or "").lower()
        if key not in best or (row["score"] or 0) > (best[key]["score"] or 0):
            best[key] = row
    expanded = sorted(best.values(), key=lambda r: r["score"] or 0, reverse=True)
    return expanded
