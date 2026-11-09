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
    ("structures_evidence",       "structures.evidence",       "Check known structures",     "PDBe \u00b7 UniProt",                              True),
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
            f'EBI HMMER, InterPro, PDBe, RCSB, 3D-Beacons, AlphaFold DB, ESM Atlas and '
            f'ChEMBL.</p>\n'
            f'<p class="cap" style="margin-top:34px">The five the chain never calls are below. '
            f'They were exercised on their own to record what they cost and how they fail, so '
            f'the failure modes are measured rather than assumed. Nothing here ran on the '
            f'SARS-CoV-2 result above.</p>'
            + "".join(cards)), total


# --------------------------------------------------------------------- the landing page
def build_index():
    """Wrap the hand-written body in a real document. The prose is not generated."""
    body = (ROOT / "site" / "body.html").read_text()
    landing = json.loads((RUNS / LANDING_RECORD).read_text())
    body = body.replace("<!--EVIDENCE-->", evidence_block(landing["cases"][0]))
    toolkit, calls = build_toolkit()
    body = body.replace("<!--TOOLKIT-->", toolkit)
    body = body.replace("<!--TOOLCALLS-->", f"{len(TOOLKIT)} tools \u00b7 {calls} calls")
    out = (head(f"{SITE} · one genome in, every step out",
                "What came back when the system was given one SARS-CoV-2 genome accession "
                "and nothing else.", current="index")
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


def surface_block(case):
    rows = case.get("surfaces", [])
    if not rows:
        return ""
    tiles = "".join(
        f'<span class="tile {"m" if r.get("measured") else "r"}"></span>' for r in rows)
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
    # stage exists to make: "nothing is deposited" and "something is, but not here".
    why = "".join(f"""
    <div class="pdbrow none"><span class="who">{esc(r.get('target'))}</span>
      <span class="ids">{esc(r.get('unmeasured_because'))}</span></div>""" for r in refused)
    why_block = (f'<div class="accs"><span class="lead">Why each of the '
                 f'{len(refused)} was refused:</span>{why}</div>') if refused else ""
    return f"""
  <div class="tiles" aria-hidden="true">{tiles}</div>
  <div class="legend">
    <span><span class="key m"></span> {len(measured)} measured</span>
    <span><span class="key r"></span> {len(refused)} refused, each with a reason</span>
  </div>
  {bars}
  {why_block}"""



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
    return f"""
<section id="evidence">
  <div class="sec-head"><h2>Evidence on the top targets</h2>
    <span class="tag">{num(cov.get('evidence_calls'))} calls</span></div>
  <p class="cap">Five tools the chain itself never calls, run on the highest-ranked targets
  after the ranking was already final. They are refused on the chain path because of what
  they declare they can cost \u2014 <span class="mono">bioactivity.prior_binders</span> advertises
  a 600-second worst case \u2014 so they run here, over a few targets, under a larger budget.
  <b>Nothing they return changes a rank.</b></p>
  {"".join(cards)}
</section>"""

def result_page(case, stem):
    label = case.get("pathogen") or case.get("case") or case.get("genome_accession")
    acc = case.get("genome_accession")
    lib = case.get("library") or {}
    folded = case.get("folded") or {}
    cov = case.get("coverage", {})
    title = f"{label} · {acc}"

    fold_html = ""
    if folded.get("sequence_length"):
        fold_html = f"""
<section>
  <div class="sec-head"><h2>The structure it folded</h2><span class="tag">step 4</span></div>
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

<section>
  <div class="sec-head"><h2>Surfaces</h2><span class="tag">one row per ranked chain</span></div>
  <div class="good"><p><b>{num(cov.get('surface_rows'))} answers for
  {num(cov.get('chains_ranked'))} chains.</b> A chain can be refused. It cannot disappear.</p></div>
  {surface_block(case)}
</section>
{fold_html}

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
