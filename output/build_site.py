"""Build the GitHub Pages site under `docs/` from the records in `runs/`.

The site has three parts and this script owns two of them.

* `docs/index.html` is written by hand. It is the SARS-CoV-2 walkthrough, and it is prose, so
  generating it from a template would make it worse. This script only wraps it: it reads
  `site/body.html`, puts a real `<head>` on it and links the shared stylesheet, so the page
  that ships is a complete document rather than a fragment.
* `docs/results/` is generated, one page per chain record. A visitor's run arrives as a new
  `runs/onset_chain_<stem>.json` written by `python -m hpc_pilots chain --accession ...`, and
  this script turns it into a page without anyone editing HTML. The records are tracked, so
  `runs/` is the list; each one is copied next to its page so the site can serve it, and that
  copy is what a page links as "the raw record" for a reader to check it against.
* `docs/notebook.html` is the walkthrough notebook, rendered by nbconvert with its committed
  outputs, so a visitor sees every number without running anything.

**Everything a visitor supplies is escaped on the way in.** `pathogen` is free text typed into
a GitHub issue form by a stranger, and it reaches this script through the chain record. It is
written into a public page, so every interpolation here goes through `html.escape`, including
the ones that look like they came from our own code: the record is not a trusted document once
a workflow has written a stranger's words into it.

    python3 output/build_site.py            # wrap the index, build every result page
    python3 output/build_site.py --notebook # also re-render the notebook (slow)
"""
import argparse
import html
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
DOCS = ROOT / "docs"
RESULTS = DOCS / "results"

SITE = "bind-the-unknown"
REPO = "willibmwang/bind-the-unknown"

# The record the hand-written landing page describes. It is excluded from the generated
# result listing, because the landing page is its page and two routes to one run would
# disagree the moment one of them is edited.
LANDING_RECORD = "onset_chain.json"

# Stage 1's own rows. The chain record keeps that stage as five counts, and records written
# before `chain._case` carried `candidates` have nothing else; this is where the twelve rows
# behind the counts come from for those. Written by `python -m hpc_pilots candidates`.
CANDIDATES_RECORD = "intake_candidates.json"

# The five terms in the order the rubric is argued in, which is heaviest first, and in plain
# words beside their keys. The weight is NOT written here -- it is read from the record's own
# `rubric` block, so the page cannot assert a weight the run did not use.
#
# `ceiling` is the key in the record's `term_ceiling` whose value this column is a fraction of.
# Four of the five ceilings are the weight itself; `druggable_class` is a 0-3 integer class, so
# its column tops out at 2.4 and shading it against 0.8 would put three cells past full.
TERM_COLUMNS = (
    ("conservation", "Match to a known relative",
     "Share of 5-mers the ORF has in common with its nearest curated relative. "
     "1.0 would be the same sequence; anything under 0.12 is not kept as a match at all."),
    ("surface_exposed", "On the outside",
     "Whether most of the compartments its relatives are annotated in are places a binder "
     "can reach without crossing a membrane. All or nothing."),
    ("essential_class", "The virus needs it",
     "Whether the protein's name or keywords name a class the virus cannot replicate "
     "without — a polymerase, a protease, a capsid, an entry protein. All or nothing."),
    ("druggable_class", "Class that gets drugged",
     "A 0–3 class from the words in its annotations: proteases and polymerases score 3, "
     "surface glycoproteins and helicases 2. The historical record for approved antivirals."),
    ("structures_in_family", "Structures already solved",
     "Deposited structures across its relatives, capped at 50. The smallest weight, because "
     "a structure count measures research attention as much as tractability."),
)


def heat(earned, ceiling):
    """Which of six shades a term's cell gets: none earned, then five bands up to the cap.

    Banded rather than a continuous alpha so the contrast of the text on top is decided here
    and once. The two darkest bands carry light text; a continuous ramp would put mid-grey
    text on mid-violet somewhere in the middle of the scale and fail contrast quietly.
    """
    if not ceiling or earned <= 0:
        return 0
    share = min(1.0, earned / ceiling)
    return min(5, int(share * 5) + 1)


def esc(value):
    """Every string that reaches the page goes through here. See the module docstring."""
    return html.escape("" if value is None else str(value), quote=True)


def num(value):
    """Thousands separators for ints, three decimals for floats, em dash for nothing."""
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.4g}"
    return esc(value)


def head(title, description, depth=0, current=""):
    """A complete document head. `depth` is how many directories deep the page sits.

    `current` is the nav key for the page being written, so that link is marked
    `aria-current="page"` and a screen reader says which page this is rather than reading
    five indistinguishable links.
    """
    up = "../" * depth
    nav = [("flowchart", f"{up}flowchart.html", "Diagram"),
           ("runs", f"{up}results/index.html", "Runs"),
           ("notebook", f"{up}notebook.html", "Notebook"),
           ("code", f"https://github.com/{REPO}", "Code")]
    CUR = ' aria-current="page"'
    links = "\n  ".join(
        f'<a href="{href}"{CUR if key == current else ""}>{label}</a>'
        for key, href, label in nav)
    brand_cur = CUR if current == "index" else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700;800&family=Manrope:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<link rel="stylesheet" href="{up}style.css">
</head>
<body>
<a class="skip" href="#main">Skip to the content</a>
<nav class="sitenav" aria-label="Site"><div class="sitenav-in">
  <a class="brand" href="{up}index.html"{brand_cur}>bind-the-unknown</a>
  {links}
  <button class="themer" id="themer" type="button">Dark</button>
</div></nav>
"""


# The theme the reader picked, remembered per browser. Everything here is wrapped in
# try/catch: localStorage throws outright in a private window with site data blocked, and a
# page that cannot remember a preference must still render in the one the OS asks for.
THEME_JS = """
<script>
(function () {
  var root = document.documentElement, btn = document.getElementById("themer");
  function stored() { try { return localStorage.getItem("theme"); } catch (e) { return null; } }
  function system() {
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches
      ? "dark" : "light";
  }
  function show(mode) {
    // The button names the theme it switches to, which is the opposite of the current one.
    var next = mode === "dark" ? "light" : "dark";
    btn.textContent = next === "dark" ? "Dark" : "Light";
    btn.setAttribute("aria-label", "Switch to the " + next + " theme");
  }
  var mode = stored() || system();
  root.setAttribute("data-theme", mode);
  show(mode);
  btn.addEventListener("click", function () {
    mode = root.getAttribute("data-theme") === "dark" ? "light" : "dark";
    root.setAttribute("data-theme", mode);
    try { localStorage.setItem("theme", mode); } catch (e) {}
    show(mode);
  });
})();
</script>
"""


FOOT = """
<footer class="sitefoot"><div class="wrap">
  Built from the records in <code>runs/</code> by <code>output/build_site.py</code>.
  Every number on this site is read from a record, not retyped.
</div></footer>
""" + THEME_JS + """</body>
</html>
"""


# --------------------------------------------------------------------- the toolkit
# Each tool's plain-English name and the services it actually reaches, read off the import
# URLs in its own module. The order is the order the chips are shown in.
TOOLKIT = [
    ("intake_rank_targets",       "intake.rank_targets",       "Find proteins in a genome",  "NCBI E-utilities \u00b7 UniProt \u00b7 InterPro",  True),
    ("literature_search",         "literature.search",         "Search the literature",      "Europe PMC",                                        False),
    ("similarity_homologs",       "similarity.homologs",       "Find relatives",             "EBI HMMER \u00b7 NCBI E-utilities \u00b7 UniProt", False),
    ("families_entries",          "families.entries",          "Look up protein families",   "InterPro \u00b7 UniProt",                          False),
    # Not on the chain path, despite reading like it: `chain.gate_table` imports the module
    # for the amenability-gate experiment, which writes its own record and reads its targets
    # from `configs/structures.json` rather than from the ranked list. None of the five
    # stages calls the registered tool, so it belongs with the tools that get a card.
    ("structures_evidence",       "structures.evidence",       "Check known structures",     "PDBe \u00b7 UniProt",                              False),
    ("structures_predicted",      "structures.predicted",      "Check predicted structures", "AlphaFold DB \u00b7 3D-Beacons",                   False),
    ("structures_folded",         "structures.folded",         "Fold a sequence",            "ESM Atlas (ESMFold v1)",                            True),
    ("geometry_surface_markers",  "geometry.surface_markers",  "Measure a surface",          "PDBe \u00b7 RCSB \u00b7 UniProt",                 True),
    ("bioactivity_prior_binders", "bioactivity.prior_binders", "Find known binders",         "ChEMBL \u00b7 IEDB \u00b7 MaveDB",                False),
    ("chemistry_library_spec",    "chemistry.library_spec",    "Define a library",           "local tables, no network",                          True),
]


def _args_summary(call):
    """One line per recorded call: what it was asked, and what came back.

    Arguments are a stranger-free dict written by our own exercise grid, but they can be a
    306-residue sequence, so long values are replaced by their length rather than printed.
    """
    bits = []
    for k, v in (call.get("arguments") or {}).items():
        v = str(v)
        bits.append(f"{esc(k)}: " + (f"<i>{len(v)} characters</i>" if len(v) > 28 else esc(v)))
    return ", ".join(bits) or "<i>no arguments</i>"


def build_toolkit():
    """Chips for all ten tools, then a card for each one the chain does not call.

    Every number here is read from `runs/toolcalls/<tool>.json`, which is why this is
    generated rather than written into `site/body.html` by hand: the page claims that no
    number on it was retyped, and the toolkit counts are numbers like any other.
    """
    records = {}
    for stem, tool, label, dbs, on_chain in TOOLKIT:
        path = RUNS / "toolcalls" / f"{stem}.json"
        if path.exists():
            records[stem] = json.loads(path.read_text())

    chips, total = [], 0
    for stem, tool, label, dbs, on_chain in TOOLKIT:
        rec = records.get(stem)
        if not rec:
            continue
        total += rec.get("calls") or 0
        chips.append(f'<span class="chip">{esc(label)} <b>{num(rec.get("calls"))}</b></span>')

    cards = []
    for stem, tool, label, dbs, on_chain in TOOLKIT:
        rec = records.get(stem)
        if not rec or on_chain:
            continue                      # the chain's own tools are detailed in section 04
        dec, mea = rec.get("declared") or {}, rec.get("measured") or {}
        status = rec.get("by_status") or {}
        cells = [
            ("Calls recorded", rec.get("calls")),
            ("Declared cost", f'{num(dec.get("cost_seconds"))} s \u00b7 '
                              f'{num(dec.get("cost_network_calls"))} requests'),
            ("Measured, median", f'{num(round(mea.get("seconds_median") or 0, 1))} s \u00b7 '
                                 f'{num(mea.get("network_calls_median"))} requests'),
            ("Slowest call", f'{num(round(mea.get("seconds_max") or 0, 1))} s'),
        ] + [(k.title(), v) for k, v in sorted(status.items())]
        ret = "".join(f"<div><dt>{esc(k)}</dt><dd>{num(v) if not isinstance(v, str) else v}</dd></div>"
                      for k, v in cells)

        # The grid repeats a call to separate a cold request from warm ones, so identical
        # (arguments, status) pairs are collapsed with a count rather than listed twice.
        # The note is dropped when it only restates the status, which is the common case.
        seen = {}
        for c in (rec.get("trace") or []):
            note = (c.get("result") or "")
            status = c.get("tool_status") or ""
            if note.startswith(status + ":"):
                note = note[len(status) + 1:].strip()
            if note == "call returned with no note":
                note = ""
            # Some notes are a dict repr written by the exercise grid. Prefer its own
            # explanation field over printing Python at the reader; literal_eval is safe
            # here because the value is a literal we wrote, and it cannot execute code.
            if note.startswith("{"):
                try:
                    import ast
                    parsed = ast.literal_eval(note)
                    note = str(parsed.get("means") or parsed.get("note") or "")
                except Exception:
                    note = ""
            if len(note) > 170:
                note = note[:168].rstrip() + "\u2026"
            key = (_args_summary(c), status, note)
            seen[key] = seen.get(key, 0) + 1
        calls = "".join(
            f'<div class="pdbrow{"" if st == "ok" else " none"}">'
            f'<span class="who">{args}{f" <i>&#215;{n}</i>" if n > 1 else ""}</span>'
            f'<span class="ids">{esc(st)}{" \u00b7 " + esc(note) if note else ""}</span></div>'
            for (args, st, note), n in seen.items())

        mis = rec.get("mispredictions") or []
        mis_html = ""
        if mis:
            rows = "".join(
                f'<div class="pdbrow none"><span class="who">{_args_summary(m)}</span>'
                f'<span class="ids">expected {esc(m.get("expected"))}, observed '
                f'{esc(m.get("observed"))}</span></div>' for m in mis)
            mis_html = (f'<div class="accs"><span class="lead">Where the tool\u2019s own declared '
                        f'spec was wrong \u2014 {len(mis)} of {rec.get("calls")} calls. These are '
                        f'kept rather than corrected away:</span>{rows}</div>')

        cards.append(f"""
  <div class="tool-card" id="t-{esc(tool.split('.')[1]).replace('_', '-')}">
    <div class="tc-head"><span class="tn">{esc(tool)}</span>
      <span class="plain">{esc(label)}</span>
      <span class="db">{dbs}</span></div>
    <div class="tc-body">
      <dl class="ret">{ret}</dl>
      <div class="accs"><span class="lead">Every call on record, including the deliberately
        bad ones:</span>{calls}</div>
      {mis_html}
    </div>
  </div>""")

    return (f'<div class="chips">{"".join(chips)}</div>\n'
            f'<p class="after">Databases reached: NCBI E-utilities, UniProt, Europe PMC, '
            f'EBI HMMER, InterPro, PDBe, RCSB, 3D-Beacons, AlphaFold DB, ESM Atlas, '
            f'ChEMBL, IEDB and MaveDB.</p>\n'
            f'<p class="cap" style="margin-top:34px">The {num(len(cards))} tools the five steps '
            f'never call are below. They were exercised on their own to record what they cost '
            f'and how '
            f'they fail, so the failure modes are measured rather than assumed, and deliberately '
            f'bad inputs are included. Some of those calls were about this pathogen — the '
            f'arguments below name this page’s own input accession and two of the targets '
            f'ranked above — but the grid keeps each call’s cost, status and failure '
            f'mode and discards the payload, so nothing any of them returned reached the result '
            f'above.</p>'
            + "".join(cards)), total


def build_offchain():
    """One generated block for section 04: the registered tools the five steps do not call.

    Section 04 details what the chain itself called, so a reader counting tool names there
    finds four of the ten, and no Europe PMC, InterPro, AlphaFold DB or ChEMBL. That is
    accurate, and it is also the first question the section raises, so the answer belongs
    beside it rather than only in section 06. It is generated because it carries counts.
    """
    chips, calls, tools = [], 0, 0
    for stem, tool, label, dbs, on_chain in TOOLKIT:
        if on_chain:
            continue
        path = RUNS / "toolcalls" / f"{stem}.json"
        if not path.exists():
            continue
        rec = json.loads(path.read_text())
        calls += rec.get("calls") or 0
        tools += 1
        chips.append(f'<span class="k">{esc(tool)} · {dbs}</span>')
    if not chips:
        return ""
    return (f'<div class="fields"><span class="lead">The other {num(tools)} registered tools '
            f'reach further databases, and none of them is on this path. They were exercised '
            f'on a grid of their own — {num(calls)} calls, each one listed in the toolkit '
            f'below with the arguments it was given — which records what a call cost and '
            f'how it failed and keeps no payload, so nothing they returned is in the result '
            f'above:</span>{"".join(chips)}</div>')


# The row section 07 gets in the table of contents, shown only when the record has a
# stage 6 to point at. It lives here rather than in `site/body.html` for that reason.
EVIDENCE_TOC = """<a class="tocrow" href="#evidence"><span class="n">07</span>
    <span class="ti">Evidence on the top targets<span class="ds">The five tools the chain \
refuses, run on the three it ranked highest.</span></span>
    <span class="ar" aria-hidden="true">→</span></a>"""


# --------------------------------------------------------------------- the landing page
def build_index():
    """Wrap the hand-written body in a real document. The prose is not generated."""
    body = (ROOT / "site" / "body.html").read_text()
    landing = json.loads((RUNS / LANDING_RECORD).read_text())
    # Stage 6 is absent from any record written before it existed, and `evidence_block`
    # answers that with an empty string. Its table-of-contents row has to vanish with it,
    # or the page ships a row pointing at an anchor that is not there.
    # The landing page is the held-out case throughout: its ranking, its accessions, its
    # refusals. Its evidence section has to be the same case. `cases[0]` is the one that
    # identifies through SARS-CoV-2's own curated entries, and annotating the shortlist from
    # it would put the held-out organism's own InterPro, AlphaFold and ChEMBL records on a
    # page whose premise is that those entries were deleted before the run.
    holdout = next((c for c in landing["cases"] if c.get("reference_excluded_organism")),
                   landing["cases"][0])
    body = body.replace("<!--CANDIDATES-->", candidate_matrix(holdout))
    evidence = evidence_block(holdout)
    body = body.replace("<!--EVIDENCE-->", evidence)
    body = body.replace("<!--EVIDENCE-TOC-->", EVIDENCE_TOC if evidence else "")
    body = body.replace("<!--OFFCHAIN-->", build_offchain())
    toolkit, calls = build_toolkit()
    body = body.replace("<!--TOOLKIT-->", toolkit)
    body = body.replace("<!--TOOLCALLS-->", f"{len(TOOLKIT)} tools \u00b7 {calls} calls")
    # Last, so a {{token}} introduced by any generated block above is filled too. The
    # landing page is the held-out case throughout, so its numbers come from that case.
    body = fill(body, facts(holdout) | candidate_facts(holdout))
    out = (head(f"{SITE} · one pathogen genome in, every next step out",
                "Deciding which of a new virus's proteins a drug should aim at, from the "
                "genome accession alone, with every step on the record. This is the "
                "walkthrough on SARS-CoV-2, held out of its own reference.",
                current="index")
           + body + FOOT)
    (DOCS / "index.html").write_text(out)
    print(f"wrote docs/index.html  ({len(out):,} bytes)")


def build_flowchart():
    """The pipeline as a diagram, for a reader who wants the shape before the detail.

    Hand-written for the same reason the landing page is: it is one figure making one claim,
    and generating it from the records would make it specific to a genome when the point of
    it is that it is not.
    """
    body = (ROOT / "site" / "flowchart.html").read_text()
    out = (head(f"From an accession to ranked targets · {SITE}",
                "The shape of the pipeline for any virus: one accession in, ranked targets "
                "out, with nothing fitted anywhere on the path.", current="flowchart")
           + body + FOOT)
    (DOCS / "flowchart.html").write_text(out)
    print(f"wrote docs/flowchart.html  ({len(out):,} bytes)")


# --------------------------------------------------------------------- one result page
def coverage_strip(case):
    cov = case.get("coverage", {})
    cells = [
        ("Accession", case.get("genome_accession")),
        ("Chains ranked", cov.get("chains_ranked")),
        ("Surfaces measured", f"{cov.get('surface_measured')} of {cov.get('surface_rows')}"),
        ("Nothing dropped", cov.get("surface_covers_every_chain")),
        ("Seconds", round(case.get("seconds") or 0)),
    ]
    inner = "".join(
        f'<div><dt>{esc(k)}</dt><dd>{num(v)}</dd></div>' for k, v in cells)
    return f'<dl class="strip">{inner}</dl>'


def stage_cards(case):
    """Each stage's own input and output dicts, rendered as they were recorded."""
    out = []
    for st in case.get("stages", []):
        ins = "".join(f"<li><span class=\"k\">{esc(k)}</span> {num(v)}</li>"
                      for k, v in (st.get("input") or {}).items()
                      if v not in (None, [], {}))
        ins = ins or "<li>nothing from upstream</li>"
        outs = "".join(f"<li><span class=\"k\">{esc(k)}</span> {num(v)}</li>"
                       for k, v in (st.get("output") or {}).items()
                       if not isinstance(v, (dict, list)))
        hands = st.get("hands_to_next")
        hands_html = (f'<div class="hands"><b>hands on</b> &nbsp;{esc(hands)}</div>'
                      if hands else "")
        out.append(f"""
  <div class="stage">
    <div class="stage-head"><span class="sn">{esc(st.get('stage'))}</span>
      <span class="what">{esc(st.get('name'))}</span>
      <span class="secs">{esc(st.get('tool'))} · {num(round(st.get('seconds') or 0, 2))} s</span></div>
    <div class="io">
      <div><h3>In</h3><ul>{ins}</ul></div>
      <div class="out"><h3>Out</h3><ul>{outs}</ul></div>
    </div>{hands_html}
  </div>""")
    return "\n".join(out)


def score_bars(case, limit=8):
    rows = sorted(case.get("scores", []), key=lambda r: r.get("rank", 999))[:limit]
    if not rows:
        return ""
    top = max((r.get("score") or 0) for r in rows) or 1
    bars = []
    for r in rows:
        score = r.get("score") or 0
        bars.append(f"""
    <div class="bi"><div class="bi-top">
      <span class="nm">{esc(r.get('rank'))} · {esc(r.get('target'))}</span>
      <span class="vl">{score:.2f}</span></div>
      <div class="track"><div class="fill" style="width:{max(2, score / top * 100):.1f}%"></div></div></div>""")
    rest = len(case.get("scores", [])) - len(rows)
    tail = (f'<p class="after">{rest} further chains were ranked and scored; '
            f'the full list is in the record.</p>' if rest > 0 else "")
    return f'<div class="bars">{"".join(bars)}</div>{tail}'


def candidate_rows(case):
    """Stage 1's rows, and the record they came from, with the counts cross-checked.

    Prefers the chain record's own `candidates`, so a page built from a run that carried them
    is reading one record throughout. Falls back to the stage-1 record for the runs written
    before that key existed, which includes the landing page's own 35-minute run.

    The fallback is only honest if the two are the same stage of the same genome under the
    same reference, so the four counts the chain record does keep are checked against the
    stage-1 record's. They are the whole point of the cross-check: if the ORF caller found a
    different number of ORFs, or identified a different number of them, the table is not the
    one the rest of the page is describing and a reader has no way to see that from the page.
    Mismatches stop the build rather than shipping, for the same reason an unfilled {{token}}
    does.
    """
    st1 = {s["tool"]: s for s in case["stages"]}["intake.rank_targets"]
    counts = st1["output"]
    if case.get("candidates"):
        return case["candidates"], None
    path = RUNS / CANDIDATES_RECORD
    if not path.exists():
        raise SystemExit(
            f"The landing record keeps stage 1 as counts only, and {path} is not there to "
            f"supply the rows. Run `python -m hpc_pilots candidates` (about 17 s against a "
            f"warm reference cache).")
    record = json.loads(path.read_text())
    checked = ("orfs_found", "orfs_ranked", "identified", "unidentified")
    drift = {k: (counts.get(k), record.get(k)) for k in checked
             if counts.get(k) != record.get(k)}
    if drift or record.get("genome_accession") != case.get("genome_accession"):
        raise SystemExit(
            f"{CANDIDATES_RECORD} is not the same stage 1 as the landing record: "
            f"{drift or 'different genome accession'} (chain, stage-1 record). Re-run "
            f"`python -m hpc_pilots candidates` so the table matches the run on the page.")
    return record["candidates"], record


def candidate_facts(case):
    """The stage-1 thresholds the prose quotes, which the chain record does not keep.

    The gene caller's length cutoff and the identity floor are arguments to the stage, not
    outputs of it, so the chain record's five counts do not carry them and the prose beside
    the table was the only place on the page they appeared. Read from the record that does
    carry them, for the same reason every other number on the page is: a cutoff typed into a
    sentence is a cutoff that stays at 60 after someone changes it.
    """
    rows, record = candidate_rows(case)
    source = record or case
    out = {"minimum_orf_aa": source.get("minimum_orf_aa"),
           "minimum_hit_score": source.get("minimum_hit_score")}
    return {k: v for k, v in out.items() if v is not None}


def candidate_matrix(case):
    """Every ORF stage 1 ranked, with the score split into what each weight paid it.

    The table this replaces was five integers. A reader could learn that twelve ORFs were
    ranked and nine identified, and had no way to ask which nine, what any of them scored, or
    which of the five weights put the spike at the top -- which is the one question the
    ranking exists to answer and the one a reader is most entitled to disagree with.

    Three things it is built to make visible, each of which a sorted list of scores hides:

    * The top of the list is a near-tie reached by opposite routes. The spike and the
      replicase finish a few hundredths apart; the spike earns a full 2.0 for sitting on the
      outside and under half the conservation term, the replicase earns nothing for exposure
      and most of the conservation term. Read down the columns and those are different
      proteins. Read down the scores and they are the same.
    * Conservation is the only one of the five read from this genome's relationship to the
      reference. The other four are read off whichever homolog was matched and do not move as
      that homolog gets more distant, which is why a held-out run's scores fall by about three
      points rather than collapsing, and why the conservation column is the one that says how
      far the evidence actually reached.
    * Three of the twelve score nothing at all. They are in the table with their reason,
      because they are the operating envelope rather than a gap: no relative at either
      homology tier and no surface signal in their own sequence either. A table of the nine
      that worked would be the pipeline grading its own coverage.
    """
    rows, record = candidate_rows(case)
    if not rows:
        return ""
    source = record or case
    rubric, ceiling = source["rubric"], source["term_ceiling"]
    scored = [r for r in rows if r.get("score") is not None]
    top = max((r["score"] for r in scored), default=0) or 1
    total_ceiling = round(sum(ceiling[key] for key, _, _ in TERM_COLUMNS), 2)

    head_cells = []
    for key, label, note in TERM_COLUMNS:
        weight = rubric["structures_in_family_capped" if key == "structures_in_family"
                        else key]
        head_cells.append(
            f'<th scope="col" title="{esc(note)}"><span class="cl">{esc(label)}</span>'
            f'<span class="ck">{esc(key)}</span>'
            f'<span class="cw">× {num(weight)} → max {num(ceiling[key])}</span></th>')

    body = []
    for row in rows:
        where = (f'{esc(row["strand"])}{esc(row["frame"])} · nt {row["nt_start"]:,} '
                 f'· {row["length_aa"]:,} aa')
        if row.get("score") is None:
            # One cell across the five terms. Writing five zeroes here would read as five
            # measurements that came back zero, and nothing was measured: the ranker never
            # reached the features because nothing identified the protein.
            body.append(f"""
      <tr class="unid">
        <td class="rk">{esc(row['rank'])}</td>
        <td class="nm"><span class="t">Not identified</span><span class="w">{where}</span></td>
        <td class="why" colspan="5">{esc(row.get('unidentified_because'))}</td>
        <td class="sc">&mdash;</td>
      </tr>""")
            continue
        features, terms = row["features"], row["terms"]
        cells = []
        for key, label, _ in TERM_COLUMNS:
            earned, raw = terms[key], features[key]
            if isinstance(raw, bool):
                shown = "yes" if raw else "no"
            elif key == "conservation":
                shown = f"{raw:.3f}"
            elif key == "druggable_class":
                shown = f"class {raw}"
            else:
                shown = f"{raw:,} in PDB"
            cells.append(f'<td class="h h{heat(earned, ceiling[key])}">'
                         f'<span class="pt">{earned:.2f}</span>'
                         f'<span class="rw">{esc(shown)}</span></td>')
        route = row.get("identification_route")
        homolog = row.get("nearest_organism") or row.get("nearest_homolog")
        via = (f'<span class="w">{esc(homolog)}'
               f'{" · " + esc(route) if route else ""}</span>' if homolog else
               f'<span class="w">{esc(route or "")}</span>')
        body.append(f"""
      <tr>
        <td class="rk">{esc(row['rank'])}</td>
        <td class="nm"><span class="t">{esc(row['name'])}</span>{via}
          <span class="w">{where}</span></td>
        {"".join(cells)}
        <td class="sc"><span class="v">{row['score']:.2f}</span>
          <span class="tr"><i style="width:{max(2, row['score'] / top * 100):.1f}%"></i></span></td>
      </tr>""")

    note = ""
    if record is not None:
        note = (f'<p class="after">The five weights and every number in this table are read '
                f'from <span class="mono">runs/{esc(CANDIDATES_RECORD)}</span>, which is '
                f'this same stage run on its own — the chain record keeps stage 1 as '
                f'counts, and the build refuses to ship this table unless those counts '
                f'match. The later sections are from '
                f'<span class="mono">runs/{esc(LANDING_RECORD)}</span>.</p>')

    return f"""
  <div class="cmxscroll" tabindex="0" role="region" aria-label="The ranked ORFs and their scores">
  <table class="cmx">
    <caption>Every ORF stage 1 ranked, highest composite score first. Each cell is the points
      that weight paid, with the measurement it was computed from underneath; the shading is
      the share of that column's maximum. Nothing here is fitted — the five weights were
      fixed before the run.</caption>
    <thead>
      <tr>
        <th scope="col" class="rk">#</th>
        <th scope="col" class="nm">ORF, and what it matched</th>
        {"".join(head_cells)}
        <th scope="col" class="sc">Composite<span class="cw">of {num(total_ceiling)}</span></th>
      </tr>
    </thead>
    <tbody>{"".join(body)}</tbody>
  </table>
  </div>{note}"""


def surface_block(case):
    rows = case.get("surfaces", [])
    if not rows:
        return ""
    def tile_class(row):
        if row.get("measured"):
            return "m"
        return "p" if (row.get("predicted_surface") or {}).get("measured") else "r"

    tiles = "".join(f'<span class="tile {tile_class(r)}"></span>' for r in rows)
    measured = [r for r in rows if r.get("measured")]
    refused = [r for r in rows if not r.get("measured")]
    bars = ""
    if measured:
        measured = sorted(measured, key=lambda r: -(r.get("patch_median_a2") or 0))
        top = max((r.get("patch_median_a2") or 0) for r in measured) or 1
        bars = "".join(f"""
    <div class="bi"><div class="bi-top">
      <span class="nm">{esc(r.get('target'))}</span>
      <span class="vl">{(r.get('patch_median_a2') or 0):.0f} Å²</span></div>
      <div class="track"><div class="fill" style="width:{max(2, (r.get('patch_median_a2') or 0) / top * 100):.1f}%"></div></div></div>"""
                        for r in measured)
        bars = f'<div class="bars">{bars}</div>'
    # Every refusal keeps its own sentence. Grouping them would hide the distinction the
    # stage exists to make: "nothing is deposited" and "something is, but not here". A chain
    # whose only surface came from a prediction says so on its own row: it is still refused on
    # the deposited lane, and reading it as measured would pool the two kinds of evidence this
    # stage keeps apart.
    def predicted_note(row):
        predicted = row.get("predicted_surface") or {}
        if not predicted.get("measured"):
            return ""
        return (f'<br><i class="gl">measured on predicted coordinates instead: '
                f'{predicted.get("patch_a2")} Å²'
                + (' — the only surface this chain has'
                   if predicted.get("only_route_to_a_surface") else '')
                + '</i>')

    why = "".join(f"""
    <div class="pdbrow none"><span class="who">{esc(r.get('target'))}</span>
      <span class="ids">{esc(r.get('unmeasured_because'))}{predicted_note(r)}</span></div>"""
                  for r in refused)
    why_block = (f'<div class="accs"><span class="lead">Why each of the '
                 f'{len(refused)} had no deposited structure to measure:</span>{why}</div>'
                 ) if refused else ""
    rescued = sum(1 for r in refused
                  if (r.get("predicted_surface") or {}).get("measured"))
    rescued_line = (f'<span><span class="key p"></span> {rescued} of those measured on '
                    f'predicted coordinates instead</span>' if rescued else "")
    return f"""
  <div class="tiles" aria-hidden="true">{tiles}</div>
  <div class="legend">
    <span><span class="key m"></span> {len(measured)} measured on deposited structures</span>
    <span><span class="key r"></span> {len(refused)} with none, each with a reason</span>
    {rescued_line}
  </div>
  {bars}
  {why_block}"""



PREDICTED_RECORD = "predicted_models.json"


def build_predicted():
    """What the endpoint label "AlphaFold DB" on the rows above is actually counting.

    Section 07 prints `structures.predicted` as scalars, and `predicted_model_count 50`
    beside the endpoint "AlphaFold DB, 3D-Beacons" reads as "AlphaFold modelled this
    protein". It is not what the tool returned. The tool carries `provider` (the host that
    served a model) apart from `tool_used` (what computed it) precisely because the two
    disagree here, and `chain._evidence_summary` reduces the model list to a count by
    design, which is the step that loses the distinction.

    So the number gets its provenance printed next to it, from a record written by
    `hpc_pilots.predicted_inventory`, which re-asks the same tool for the same accessions.
    Generated rather than written out because every figure in it is a count.
    """
    path = RUNS / PREDICTED_RECORD
    if not path.exists():
        return ""
    rec = json.loads(path.read_text())
    blocks = rec.get("accessions") or []
    if not blocks:
        return ""

    rows = []
    for b in blocks:
        served = b.get("alphafold_host_model_count") or 0
        # Read `tool_used` out loud. A provider breakdown alone still lets a reader assume
        # the 35 models under "AlphaFold DB" are AlphaFold's.
        tools = " · ".join(f"{esc(tool)} {num(count)}"
                               for tool, count in (b.get("models_by_tool_used") or {}).items())
        providers = " · ".join(f"{esc(name)} {num(count)}"
                                   for name, count in (b.get("models_by_provider") or {}).items())
        if served:
            host = (f"the AlphaFold host served {num(served)} of them, every one computed by "
                    + " and ".join(esc(t) for t in
                                   (b.get("alphafold_host_models_by_tool_used") or {}))
                    + f" — <b>{num(b.get('alphafold_authored_model_count'))} were computed "
                    f"by AlphaFold</b>")
        else:
            # An absence that was measured, not skipped. The tool asks the host
            # unconditionally for exactly this reason.
            host = ("the AlphaFold host was asked and holds nothing for it — a measured "
                    "absence, not a request that was skipped")

        # pLDDT is absent, not zero, when every model is template-based. The page prints
        # `confident_plddt 70` on these rows and it never applied to this accession.
        plddt = []
        for r in b.get("ranges") or []:
            best = r.get("best_plddt_model")
            if best:
                plddt.append(f'<div class="pdbrow"><span class="who">{esc(r.get("range"))} '
                             f'<i>{num(r.get("width"))} res</i></span><span class="ids">'
                             f'best pLDDT {num(best.get("median_plddt"))} median, '
                             f'{num(best.get("min_plddt"))} min · '
                             f'{esc(best.get("model_identifier"))} '
                             f'({esc(best.get("tool_used"))})</span></div>')
            else:
                plddt.append(f'<div class="pdbrow none"><span class="who">'
                             f'{esc(r.get("range"))} <i>{num(r.get("width"))} res</i></span>'
                             f'<span class="ids">no pLDDT at all · '
                             f'{esc(r.get("no_plddt_because"))}</span></div>')

        targets = ", ".join(f'#{esc(t.get("rank"))} {esc(t.get("target"))}'
                            for t in (b.get("asked_about") or []))
        rows.append(
            f'<div class="accs"><span class="lead"><b>{esc(b.get("accession"))}</b> '
            f'— {num(b.get("sequence_length"))} residues, asked about for {targets}. '
            f'{num(b.get("predicted_model_count"))} predicted models, and {host}.</span>'
            f'<div class="pdbrow"><span class="who">served by</span>'
            f'<span class="ids">{providers}</span></div>'
            f'<div class="pdbrow"><span class="who">computed by</span>'
            f'<span class="ids">{tools}</span></div>'
            f'{"".join(plddt)}</div>')

    total = rec.get("predicted_model_count")
    by_host = rec.get("alphafold_host_model_count")
    authored = rec.get("alphafold_authored_model_count")
    return (
        f'<div class="tool-card" id="predicted-provenance">\n'
        f'    <div class="tc-head"><span class="tn">What &#8220;AlphaFold DB&#8221; '
        f'served</span>\n'
        f'      <span class="plain">the provenance of the counts above</span>\n'
        f'      <span class="db">{num(total)} models · {num(authored)} from '
        f'AlphaFold</span></div>\n'
        f'    <div class="tc-body">\n'
        f'      <p class="cap" style="margin:0 0 18px">Every <span class="mono">'
        f'structures.predicted</span> row above is labelled <span class="mono">AlphaFold DB '
        f'· 3D-Beacons</span> and reports a bare <span class="mono">'
        f'predicted_model_count</span>, which reads as though AlphaFold modelled these '
        f'proteins. It did not. The tool reports the host that <i>served</i> each model '
        f'separately from the tool that <i>computed</i> it, and for these accessions the two '
        f'disagree: of {num(total)} predicted models, {num(by_host)} were served by the '
        f'AlphaFold host and <b>{num(authored)} were computed by AlphaFold</b> — they '
        f'are ColabFold v1.5.2 models, which run the same weights behind a different '
        f'alignment pipeline. The chain record cannot carry this, because its summariser '
        f'reduces the model list to a count on purpose; it is read here from <span '
        f'class="mono">runs/predicted_models.json</span>.</p>\n'
        f'      {"".join(rows)}\n'
        f'    </div>\n'
        f'  </div>')



def facts(case):
    """Every number the hand-written prose states, read out of the record it came from.

    `site/body.html` is prose and stays prose, but the numbers in it were literals typed to
    match one run. The page's own contract is that every number on it is read from a record
    and not retyped, and for the hand-written half that was not true: re-running the chain
    moved the record and left the prose asserting the previous run's values, with nothing
    to show a reader which half was stale. The literals are now `{{tokens}}` filled from the
    record at build time, so the prose follows whichever run is current.

    A token that has no value raises rather than rendering empty. A silent blank in a
    sentence like "<b></b> of 20 measured" is worse than a build that stops.
    """
    # Keyed on the tool each stage ran, not on its position. The pipeline's order is a design
    # decision and it has already changed once -- folding moved ahead of the surface stage so
    # that a chain with no deposited structure has coordinates to measure -- and a token table
    # written against `st[3]` silently re-points every number on the page when that happens.
    # It also keeps the page buildable from records written under the previous numbering,
    # because the tool names did not move.
    by_tool = {s["tool"]: s for s in case["stages"]}
    st1 = by_tool["intake.rank_targets"]
    st2 = by_tool["intake.decompose + intake.rank_score"]
    surface = by_tool.get("geometry.surface_markers") or {}
    folded = by_tool.get("structures.folded") or {}
    library = by_tool.get("chemistry.library_spec") or {}
    out1, out2 = st1["output"], st2["output"]
    out3 = surface.get("output") or {}
    out4 = folded.get("output") or {}
    out5 = library.get("output") or {}
    plddt = out4.get("mean_plddt")
    confident = out4.get("at_or_above_70")
    diversity = out5.get("theoretical_diversity")
    values = {
        "genome_nt": f"{out1['genome_length_nt']:,}",
        "segments": out1.get("segments"),
        "orfs_found": out1["orfs_found"],
        "orfs_ranked": out1["orfs_ranked"],
        "reference_entries": f"{out1['reference_entries']:,}",
        "identified": out1["identified"],
        "unidentified": out1["unidentified"],
        "chains": out2["chains_out"],
        "rows_out": out3["rows_out"],
        "measured": out3["measured"],
        "unmeasured": out3["unmeasured"],
        "fold_residues": out4.get("residues"),
        "mean_plddt": round(plddt, 1) if plddt is not None else None,
        "confident_pct": f"{confident * 100:.1f}" if confident is not None else None,
        "alphabet": out5.get("alphabet_size"),
        "diversity": f"{diversity:,}" if diversity else None,
        "diversity_short": (f"{diversity / 1e9:.0f} billion" if diversity else None),
        "seconds": round(case.get("seconds") or 0),
        # Named after the stage's job rather than its position, for the same reason the
        # lookup above is.
        "intake_seconds": round(st1.get("seconds") or 0),
        "score_seconds": round(st2.get("seconds") or 0),
        "surface_seconds": round(surface.get("seconds") or 0) if surface else None,
        "fold_seconds": round(folded.get("seconds") or 0) if folded else None,
        # The tool cards quote the wall clock to one decimal, which is the only place the
        # page reports a timing precisely enough for rounding to lose information.
        "intake_exact": f"{st1.get('seconds', 0):.1f}",
        "score_exact": f"{st2.get('seconds', 0):.1f}",
        "surface_exact": f"{surface.get('seconds', 0):.1f}" if surface else None,
        "fold_exact": f"{folded.get('seconds', 0):.1f}" if folded else None,
    }
    return {k: v for k, v in values.items() if v is not None}


def fill(body, values):
    """Substitute every {{token}}, and refuse to ship one that has no value."""
    import re as _re

    def one(match):
        key = match.group(1)
        if key not in values:
            raise KeyError(
                f"site/body.html asks for {{{{{key}}}}} and the record has no value for it. "
                f"Available: {sorted(values)}")
        return esc(values[key])

    return _re.sub(r"\{\{(\w+)\}\}", one, body)

def evidence_block(case):
    """The five off-chain tools, per target, as they came back.

    A refusal is rendered as loudly as a result. The whole point of the stage is that a tool
    the chain's budget turns away is recorded rather than omitted, and a page that only
    showed the successes would undo that.
    """
    rows = case.get("evidence") or []
    if not rows:
        return ""
    cards = []
    for row in rows:
        calls = "".join(f"""
      <div class="pdbrow{'' if c.get('status') == 'ok' else ' none'}">
        <span class="who">{esc(c.get('tool'))}<em>{esc(c.get('endpoint'))}</em></span>
        <span class="ids">{esc(c.get('status'))}
          {('\u00b7 ' + num(c.get('seconds')) + ' s') if c.get('seconds') is not None else ''}
          {('\u00b7 ' + num(c.get('network_calls')) + ' requests') if c.get('network_calls') else ''}
          {('<br>' + esc(c.get('note'))) if c.get('note') and c.get('status') != 'ok' else ''}
          {('<br>' + ' \u00b7 '.join(esc(k) + ' ' + num(v) for k, v in (c.get('returned') or {}).items())) if c.get('returned') else ''}
        </span>
      </div>""" for c in row.get("calls", []))
        cards.append(f"""
  <div class="tool-card">
    <div class="tc-head"><span class="sn">{esc(row.get('rank'))}</span>
      <span class="tn">{esc(row.get('target'))}</span>
      <span class="db">{esc(row.get('nearest_homolog'))}</span></div>
    <div class="tc-body"><div class="accs">{calls}</div></div>
  </div>""")
    cov = case.get("coverage", {})
    # Under a holdout, every accession in this section belongs to a relative, because that is
    # the only accession the chain holds. A reader who takes an InterPro assignment or a
    # ChEMBL binder here for a fact about the virus that went in has been misled by the
    # layout, so the section says whose records these are before it shows them.
    held = case.get("reference_excluded_organism")
    whose = (f' Each accession below is the relative the chain identified that target '
             f'through, not {esc(held)}, which was deleted from the reference before the run: '
             f'every annotation here is a record of that relative.') if held else ""
    return f"""
<section id="evidence">
  <div class="sec-head"><h2>Evidence on the top targets</h2>
    <span class="tag">{num(cov.get('evidence_calls'))} calls</span></div>
  <p class="cap">Five tools the chain itself never calls, run on the highest-ranked targets
  after the ranking was already final. Nothing refuses them: the cost gate admits all five,
  because it turns a tool away only when its declared worst case is <em>strictly greater</em>
  than the budget, and the default budget is 600 seconds \u2014 so
  <span class="mono">bioactivity.prior_binders</span>, declaring exactly 600, is admitted. What
  keeps them off the path is measured cost rather than declared cost: about 210 seconds for one
  target across the five, against a chain that finishes in {{{{seconds}}}} seconds.
  <b>Nothing they return changes a rank.</b>{whose}</p>
  {"".join(cards)}
  {build_predicted()}
</section>"""

def result_page(case, stem):
    label = case.get("pathogen") or case.get("case") or case.get("genome_accession")
    acc = case.get("genome_accession")
    lib = case.get("library") or {}
    folded = case.get("folded") or {}
    cov = case.get("coverage", {})
    title = f"{label} · {acc}"

    # The step number is read from the record, not typed. This page prints "Every number on
    # this page is read from it. Nothing here is retyped", and a hand-typed `step 4` broke that
    # the moment folding moved to stage 3: the stage card above it prints the record's own
    # number, so one page would have carried both.
    by_tool = {st.get("tool"): st for st in case.get("stages", [])}
    fold_step = (by_tool.get("structures.folded") or {}).get("stage")
    surface_step = (by_tool.get("geometry.surface_markers") or {}).get("stage")
    fold_tag = f"step {fold_step}" if fold_step else "the folding stage"

    fold_html = ""
    if folded.get("sequence_length"):
        also = case.get("folded_also") or []
        folded_line = (f'<div class="sub">{len(also) + 1} chains folded this run; '
                       f'this is the one the record reports in full</div>' if also else "")
        fold_html = f"""
<section>
  <div class="sec-head"><h2>The structure it folded</h2><span class="tag">{fold_tag}</span></div>
  {folded_line}
  <div class="two">
    <div class="card acc">
      <div class="label">{esc(folded.get('target_name'))}</div>
      <div class="big">{num(folded.get('sequence_length'))} res</div>
      <div class="sub">confidence {num(folded.get('mean_plddt'))} of 100 ·
        {(folded.get('confident_fraction') or 0) * 100:.1f} % at or above 70<br>
        confidence is not accuracy</div>
    </div>
    <div class="card acc">
      <div class="label">The library defined over the randomised region</div>
      <div class="big">{num(lib.get('theoretical_diversity'))}</div>
      <div class="sub">{esc(lib.get('initiator'))} initiator ·
        {esc((lib.get('random_region') or ['?', '?'])[0])}–{esc((lib.get('random_region') or ['?', '?'])[1])} positions<br>
        a definition only — no binding or antiviral property is predicted</div>
    </div>
  </div>
</section>"""

    return (head(f"{title} · {SITE}",
                 f"Every stage of the chain run on {acc}.", depth=1,
                 current="runs") + f"""
<main class="wrap" id="main">
<header class="top">
  <div class="eyebrow"><a href="index.html">runs</a> · generated from the record</div>
  <h1>{esc(label)}</h1>
  <p class="standfirst">One genome accession went in. This is what each stage of the chain
  handed the next.</p>
  {coverage_strip(case)}
  <p class="reclink"><a href="{esc(stem)}.json">Download the raw record · JSON</a>
  <span>Every number on this page is read from it. Nothing here is retyped.</span></p>
</header>

<section>
  <div class="sec-head"><h2>Every stage, in and out</h2><span class="tag">as recorded</span></div>
  {stage_cards(case)}
</section>

<section>
  <div class="sec-head"><h2>What it picked</h2><span class="tag">top {min(8, len(case.get('scores', [])))}</span></div>
  {score_bars(case)}
</section>

{fold_html if (fold_step or 99) < (surface_step or 0) else ""}
<section>
  <div class="sec-head"><h2>Surfaces</h2><span class="tag">one row per ranked chain</span></div>
  <div class="good"><p><b>{num(cov.get('surface_rows'))} answers for
  {num(cov.get('chains_ranked'))} chains.</b> A chain can be refused. It cannot disappear.</p></div>
  {surface_block(case)}
</section>
{fold_html if (fold_step or 99) >= (surface_step or 0) else ""}

{evidence_block(case)}

<section>
  <div class="sec-head"><h2>What is still missing</h2><span class="tag">on every run</span></div>
  <div class="note"><p><b>The library ignores the surface.</b> Step 5 reads nothing steps 1 to 4
  measured, on this genome as on any other.</p></div>
  <div class="note"><p><b>Nothing comes back from the lab.</b> No screening result feeds the next
  design yet.</p></div>
</section>
</main>""" + FOOT)


# --------------------------------------------------------------------- the results index
def results_index(entries):
    if entries:
        rows = "".join(f"""
    <a class="runrow" href="{esc(stem)}.html">
      <span class="who">{esc(label)}<em>{esc(acc)}</em></span>
      <span class="stat">{num(ranked)} chains · {num(measured)} measured · {num(round(secs))} s</span>
    </a>""" for stem, label, acc, ranked, measured, secs in entries)
        body = f'<div class="runlist">{rows}</div>'
    else:
        body = ('<div class="good"><p>No runs have been submitted yet. Yours would be the '
                'first — the form is on the front page.</p></div>')
    return (head(f"Runs · {SITE}", "Every pathogen genome this system has been run on.",
                 depth=1, current="runs") + f"""
<main class="wrap" id="main">
<header class="top">
  <div class="eyebrow">generated · one page per record</div>
  <h1>Every genome this has been run on</h1>
  <p class="standfirst">Each page below was written from a chain record, not by hand. Submit an
  accession on the <a href="../index.html">front page</a> and yours joins the list.</p>
</header>
<section>{body}</section>
</main>""" + FOOT)


def build_results():
    """One page per chain record. `runs/` is the list; `docs/results/` is a copy of it.

    The records are tracked, so this directory is complete on a fresh clone and on a CI
    runner alike, and a run is published by running it rather than by editing HTML. Each
    record is copied next to its page so the site can serve it: the page says every number on
    it is read from that record, and a reader should be able to open the record and check.
    """
    entries = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    # `onset_chain_*.json` with the underscore: it excludes the landing page's own record,
    # `onset_chain.json`, which is a two-reference payload and has its own hand-written page.
    for path in sorted(RUNS.glob("onset_chain_*.json")):
        if path.name == "onset_chain_genomes.json":
            continue
        case = json.loads(path.read_text())
        if "cases" in case:            # a multi-reference payload, not a single run
            continue
        if not case.get("stages"):
            print(f"  skipped {path.name}: no stages recorded")
            continue
        stem = re.sub(r"[^A-Za-z0-9]+", "_", path.stem.replace("onset_chain_", "")).strip("_")
        (RESULTS / f"{stem}.html").write_text(result_page(case, stem))
        shutil.copyfile(path, RESULTS / f"{stem}.json")
        cov = case.get("coverage", {})
        entries.append((stem,
                        case.get("pathogen") or case.get("case") or stem,
                        case.get("genome_accession"),
                        cov.get("chains_ranked"), cov.get("surface_measured"),
                        case.get("seconds") or 0))
        print(f"  wrote docs/results/{stem}.html")
    (RESULTS / "index.html").write_text(results_index(entries))
    print(f"wrote docs/results/index.html  ({len(entries)} run(s))")
    return entries


# --------------------------------------------------------------------- the other assets
def build_hero_image():
    """The hero artwork: the committed fold, drawn as a soft violet volume.

    This is the real backbone from `runs/folded_structure/mpro_nsp5.pdb`, not an illustration,
    which is the only reason it is allowed to be decorative. Two details matter if you change
    it. The backbone is straight line segments between alpha carbons, so it has to be smoothed
    or it reads as a wire diagram. And every pass must be ONE continuous path: drawing the
    segments separately (a LineCollection, say) composites each overlap on its own and the
    alpha accumulates until the whole thing is a solid blob.
    """
    pdb = RUNS / "folded_structure/mpro_nsp5.pdb"
    if not pdb.exists():
        print("  no committed fold on this machine; leaving docs/hero.png as it is")
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-site")
    os.environ.setdefault("MPLBACKEND", "Agg")
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xyz = np.array([(float(l[30:38]), float(l[38:46]), float(l[46:54]))
                    for l in pdb.read_text().splitlines()
                    if l.startswith("ATOM") and l[12:16].strip() == "CA" and l[16] in (" ", "A")])

    def chaikin(pts, n=4):
        """Corner cutting: each pass replaces a segment with points at 1/4 and 3/4."""
        for _ in range(n):
            a, b = pts[:-1], pts[1:]
            out = np.empty((2 * len(a), pts.shape[1]))
            out[0::2] = .75 * a + .25 * b
            out[1::2] = .25 * a + .75 * b
            pts = out
        return pts

    centred = xyz - xyz.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    p = chaikin((centred @ vt[:2].T)[:, ::-1], 4)     # longest axis vertical

    fig = plt.figure(figsize=(6, 7.6), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_aspect("equal")
    for lw, alpha in ((58, .030), (44, .038), (32, .050), (22, .065), (14, .085), (8, .11)):
        ax.plot(p[:, 0], p[:, 1], color="#ac31ff", lw=lw, alpha=alpha,
                solid_capstyle="round", solid_joinstyle="round")
    m = 9
    ax.set_xlim(p[:, 0].min() - m, p[:, 0].max() + m)
    ax.set_ylim(p[:, 1].min() - m, p[:, 1].max() + m)
    fig.savefig(DOCS / "hero.png", transparent=True, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    print("wrote docs/hero.png")


def build_fold_image():
    """The committed fold, drawn through the same function the notebook uses."""
    rec = RUNS / "folded_structure/mpro_nsp5.json"
    pdb = RUNS / "folded_structure/mpro_nsp5.pdb"
    if not (rec.exists() and pdb.exists()):
        print("  no committed fold on this machine; leaving docs/fold.png as it is")
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-site")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "notebooks"))
    import panels                                                      # noqa: E402
    payload = json.loads(rec.read_text())
    resnum, xyz, bfac = panels.ca_trace(pdb.read_text())
    plddt, _ = panels.plddt_on_0_100(bfac)
    fig = panels.structure_figure(
        resnum, xyz, plddt,
        title="Main protease, folded from sequence alone",
        subtitle=f"ESMFold v1 · {len(resnum)} residues · mean confidence "
                 f"{plddt.mean():.1f} of 100 · coloured by per-residue confidence",
        runs=tuple(tuple(r) for r in payload.get("low_confidence_runs", [])))
    fig.savefig(DOCS / "fold.png", facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=150)
    print("wrote docs/fold.png")


def build_notebook():
    """nbconvert with the committed outputs, so the page needs no kernel."""
    nb = ROOT / "notebooks/double_arm_mvp.ipynb"
    if not nb.exists():
        print("  no notebook on this machine; skipped")
        return
    cmd = [sys.executable, "-m", "nbconvert", "--to", "html", "--template", "lab",
           "--output", str(DOCS / "notebook.html"), str(nb)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("  nbconvert failed:\n" + (result.stderr or "")[-800:])
        return
    print("wrote docs/notebook.html")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--notebook", action="store_true", help="also re-render the notebook")
    args = ap.parse_args()

    DOCS.mkdir(exist_ok=True)
    (DOCS / ".nojekyll").write_text("")      # serve the tree as files, no Jekyll pass
    build_index()
    build_flowchart()
    build_hero_image()
    build_fold_image()
    build_results()
    if args.notebook:
        build_notebook()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"site built {stamp}")


if __name__ == "__main__":
    main()
