"""Build the GitHub Pages site under `docs/` from the records in `runs/`.

The site has three parts and this script owns two of them.

* `docs/index.html` is written by hand. It is the SARS-CoV-2 walkthrough, and it is prose, so
  generating it from a template would make it worse. This script only wraps it: it reads
  `docs/_body.html`, puts a real `<head>` on it and links the shared stylesheet, so the page
  that ships is a complete document rather than a fragment.
* `docs/results/` is generated, one page per chain record. A visitor's run arrives as a new
  `runs/onset_chain_<stem>.json` written by `python -m hpc_pilots chain --accession ...`, and
  this script turns it into a page without anyone editing HTML. `runs/` is gitignored, so the
  record is copied next to its page and served from there; that copy is what the site links as
  "the raw record", and it is the thing a reader can check the page against.
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


def head(title, description, depth=0):
    """A complete document head. `depth` is how many directories deep the page sits."""
    up = "../" * depth
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{esc(title)}</title>
<meta name="description" content="{esc(description)}">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:wght@400;600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<link rel="stylesheet" href="{up}style.css">
</head>
<body>
<nav class="sitenav"><div class="sitenav-in">
  <a class="brand" href="{up}index.html">bind-the-unknown</a>
  <a href="{up}results/index.html">Runs</a>
  <a href="{up}notebook.html">Notebook</a>
  <a href="https://github.com/{REPO}">Code</a>
</div></nav>
"""


FOOT = """
<footer class="sitefoot"><div class="wrap">
  Built from the records in <code>runs/</code> by <code>output/build_site.py</code>.
  Every number on this site is read from a record, not retyped.
</div></footer>
</body>
</html>
"""


# --------------------------------------------------------------------- the landing page
def build_index():
    """Wrap the hand-written body in a real document. The prose is not generated."""
    body = (ROOT / "site" / "body.html").read_text()
    out = (head(f"{SITE} · one genome in, every step out",
                "What came back when the system was given one SARS-CoV-2 genome accession "
                "and nothing else.")
           + body + FOOT)
    (DOCS / "index.html").write_text(out)
    print(f"wrote docs/index.html  ({len(out):,} bytes)")


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
  <div class="tiles">{tiles}</div>
  <div class="legend">
    <span><span class="key m"></span> {len(measured)} measured</span>
    <span><span class="key r"></span> {len(refused)} refused, each with a reason</span>
  </div>
  {bars}
  {why_block}"""


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
                 f"Every stage of the chain run on {acc}.", depth=1) + f"""
<div class="wrap">
<header class="top">
  <div class="eyebrow"><a href="index.html">runs</a> · generated from the record</div>
  <h1>{esc(label)}</h1>
  <p class="standfirst">One genome accession went in. This is what each stage of the chain
  handed the next, read from <a href="{esc(stem)}.json">the record</a> of that run.</p>
  {coverage_strip(case)}
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

<section>
  <div class="sec-head"><h2>What is still missing</h2><span class="tag">on every run</span></div>
  <div class="note"><p><b>The library ignores the surface.</b> Step 5 reads nothing steps 1 to 4
  measured, on this genome as on any other.</p></div>
  <div class="note"><p><b>Nothing comes back from the lab.</b> No screening result feeds the next
  design yet.</p></div>
</section>
</div>""" + FOOT)


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
                 depth=1) + f"""
<div class="wrap">
<header class="top">
  <div class="eyebrow">generated · one page per record</div>
  <h1>Every genome this has been run on</h1>
  <p class="standfirst">Each page below was written from a chain record, not by hand. Submit an
  accession on the <a href="../index.html">front page</a> and yours joins the list.</p>
</header>
<section>{body}</section>
</div>""" + FOOT)


def build_results():
    entries = []
    RESULTS.mkdir(parents=True, exist_ok=True)
    for path in sorted(RUNS.glob("onset_chain*.json")):
        if path.name in (LANDING_RECORD, "onset_chain_genomes.json"):
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
    build_fold_image()
    build_results()
    if args.notebook:
        build_notebook()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"site built {stamp}")


if __name__ == "__main__":
    main()
