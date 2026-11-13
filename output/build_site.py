"""Build the GitHub Pages site under `docs/` from the records in `runs/`.

The site has three parts and this script owns two of them.

* `docs/index.html` is written by hand. It is the SARS-CoV-2 walkthrough, and it is prose, so
  generating it from a template would make it worse. This script only wraps it. It reads the
  fragment in `site/body.html`, puts a real `<head>` on it and links the shared stylesheet,
  so the page that ships is a complete document.
* `docs/results/` is generated, one page per chain record. A visitor's run arrives as a new
  `runs/onset_chain_<stem>.json` written by `python -m hpc_pilots chain --accession ...`, and
  this script turns it into a page without anyone editing HTML. The records are tracked, so
  `runs/` is the list; each one is copied next to its page so the site can serve it, and that
  copy is what a page links as "the raw record" for a reader to check it against.
* `docs/notebook.html` is the walkthrough notebook, rendered by nbconvert with its committed
  outputs, so a visitor sees every number without running anything.

**Everything a visitor supplies is escaped on the way in.** `pathogen` is free text typed into
a GitHub issue form by a stranger, and it reaches this script through the chain record, bound
for a public page. Every interpolation here goes through `html.escape`, including the ones
that look like they came from our own code. The record is not a trusted document once a
workflow has written a stranger's words into it.

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

# What the curated annotation offers for the chains the folding ceiling refuses whole. The
# chain record says which chains were folded; this one says which functionally important
# regions were SHORT ENOUGH to fold, which is the larger set and the one this page has to
# quote. nsp3 alone offers eighteen foldable regions against a per-chain cap of three, so a
# page reading only the fold results could say three domains of nsp3 were folded and could
# not say that the ceiling costs nothing in reachable targets -- which is the finding.
# Written by `python -m hpc_pilots.domain_inventory`.
DOMAIN_RECORD = "domain_inventory.json"

# The five terms in the order the rubric is argued in, which is heaviest first, and in plain
# words beside their keys. The weight is NOT written here -- it is read from the record's own
# `rubric` block, so the page cannot assert a weight the run did not use.
#
# `ceiling` is the key in the record's `term_ceiling` whose value this column is a fraction of.
# Four of the five ceilings are the weight itself; `druggable_class` is a 0-3 integer class, so
# its column tops out at 2.4 and shading it against 0.8 would put three cells past full.
TERM_COLUMNS = (
    ("conservation", "Match to a known relative",
     "Share of the five-residue runs in the protein it comes from that its nearest curated "
     "relative also has. 1.0 would be the same sequence, and under 0.12 is not kept as a "
     "match at all. Rows with only a distant relative carry a different measure, labelled "
     "in the cell as a bit score, because the two are not one scale."),
    ("surface_exposed", "On the outside",
     "Whether a binder could reach most of the places its relatives are recorded as sitting "
     "in without crossing a membrane, scored all or nothing."),
    ("essential_class", "The virus needs it",
     "Whether the protein's name or keywords put it in a class the virus cannot "
     "replicate without, such as a polymerase, a protease, a capsid or an entry protein. "
     "All or nothing."),
    ("druggable_class", "Class that gets drugged",
     "A 0–3 class read off the words in its annotations. Proteases and polymerases score 3, "
     "surface glycoproteins and helicases 2, because that is where approved antivirals have "
     "historically landed."),
    ("structures_in_family", "Structures already solved",
     "Deposited structures across its relatives, capped at 50. It carries the smallest "
     "weight, because a count like that measures how much a protein has been studied as "
     "much as how easy it is to work with."),
)


# `druggable_class` is the one feature that is not already a 0-1 number. It is an integer
# class from the ranker's keyword table, whose top class is 3. The chain record keeps the
# five weights but not the ceiling they imply, so the ceiling is derived from the weights
# here, and cross-checked against the stage-1 record's own `term_ceiling` wherever that
# record is beside the page. A change to the class range then stops the build instead of
# shading two of the columns against the wrong maximum.
DRUGGABLE_CLASS_MAX = 3


def term_ceilings(rubric, record=None):
    """The most each of the five terms can pay, derived from the weights that paid it."""
    if record and record.get("term_ceiling"):
        return record["term_ceiling"]
    return {"conservation": rubric["conservation"],
            "surface_exposed": rubric["surface_exposed"],
            "essential_class": rubric["essential_class"],
            "druggable_class": round(rubric["druggable_class"] * DRUGGABLE_CLASS_MAX, 4),
            "structures_in_family": rubric["structures_in_family_capped"]}


def term_head_cells(rubric, ceiling):
    """The five column headings, each carrying the weight the record says it used."""
    cells = []
    for key, label, note in TERM_COLUMNS:
        weight = rubric["structures_in_family_capped" if key == "structures_in_family"
                        else key]
        cells.append(
            f'<th scope="col" title="{esc(note)}"><span class="cl">{esc(label)}</span>'
            f'<span class="ck">{esc(key)}</span>'
            f'<span class="cw">× {num(weight)} → max {num(ceiling[key])}</span></th>')
    return "".join(cells)


def term_cells(features, terms, ceiling, scale=None):
    """The five scored cells of one row: points paid, and the measurement underneath.

    Shared by the ORF table and the chain table because the two are the same arithmetic on
    two altitudes, and a reader comparing a chain's score with its parent ORF's is comparing
    these cells. Rendering them twice would let the two drift apart in formatting, which on
    this page would read as a difference in the scoring.
    """
    cells = []
    for key, label, _ in TERM_COLUMNS:
        earned, raw = terms[key], features[key]
        if isinstance(raw, bool):
            shown = "yes" if raw else "no"
        elif key == "conservation":
            # The scale, not just the number. The exact tier reports a shared-5-mer fraction
            # of the query and the remote tier reports min(1, bits/200), so a column of bare
            # decimals mixing the two would invite a reader to compare 0.83 on one scale with
            # 0.54 on the other -- which is the inversion the record's `conservation_scale`
            # field exists to make visible. Named where it is not the k-mer fraction, and left
            # bare where it is, so the common case stays readable.
            shown = f"{raw:.3f}"
            if scale and scale != "shared_kmer_fraction_of_query":
                shown = f"{shown} · {scale.replace('_', ' ')}"
        elif key == "druggable_class":
            shown = f"class {raw}"
        else:
            shown = f"{raw:,} in PDB"
        cells.append(f'<td class="h h{heat(earned, ceiling[key])}">'
                     f'<span class="pt">{earned:.2f}</span>'
                     f'<span class="rw">{esc(shown)}</span></td>')
    return "".join(cells)


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
    four indistinguishable links.
    """
    up = "../" * depth
    nav = [("flowchart", f"{up}flowchart.html", "Diagram"),
           ("runs", f"{up}results/index.html", "Runs"),
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


# ------------------------------------------------------- the tools the chain does not call
# Each tool's plain-English name and the services it actually reaches, read off the import
# URLs in its own module.
#
# The toolkit section this table was written for is gone, and only `build_offchain` reads it
# now -- so the rows that matter are the ones with `on_chain` False, which is the half the
# off-chain block in section 04 lists. The on-chain rows are kept because the flag is what
# distinguishes them, and a list of tools that only held the off-path ones could not say that
# the other four are on the path.
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


def build_offchain():
    """One generated block for section 04: the registered tools the five steps do not call.

    Section 04 details what the chain itself called, so a reader counting tool names there
    finds four of the ten, and no Europe PMC, InterPro, AlphaFold DB or ChEMBL. That is
    accurate, and it is also the first question the section raises, so the answer belongs
    beside it. It is generated because it carries counts.

    This block used to point at the toolkit section for the per-call detail. That section is
    gone, so it points at the records the section was itself generated from -- which is where
    a reader following the claim has to end up anyway.
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
            f'on a grid of their own, {num(calls)} calls recorded under '
            f'<span class="mono">runs/toolcalls/</span> with the arguments each was given. '
            f'Those records keep what a call cost and how it failed. They hold no payload, '
            f'so nothing the tools returned is in the result above:</span>{"".join(chips)}</div>')


# --------------------------------------------------------------------- the landing page
def build_index():
    """Wrap the hand-written body in a real document. The prose is not generated."""
    body = (ROOT / "site" / "body.html").read_text()
    landing = json.loads((RUNS / LANDING_RECORD).read_text())
    # The landing page is the held-out case throughout: its ranking, its accessions, its
    # refusals. `cases[0]` is the one that identifies through SARS-CoV-2's own curated
    # entries, and reading the shortlist from it would put the held-out organism's own
    # records on a page whose premise is that those entries were deleted before the run.
    holdout = next((c for c in landing["cases"] if c.get("reference_excluded_organism")),
                   landing["cases"][0])
    body = body.replace("<!--CANDIDATES-->", candidate_matrix(holdout))
    body = body.replace("<!--CHAINS-->", chain_matrix(holdout))
    body = body.replace("<!--PICKED-->", picked_table(holdout))
    body = body.replace("<!--FOLD-TRACE-->", fold_trace())
    body = body.replace("<!--FOLD-LANE-->", fold_lane(holdout))
    body = body.replace("<!--LITERATURE-->", literature_block(holdout))
    body = body.replace("<!--SURFACE-LANE-->", surface_lane(holdout))
    body = body.replace("<!--SITE-PATCH-->", site_patch_block(holdout))
    body = body.replace("<!--OFFCHAIN-->", build_offchain())
    # Last, so a {{token}} introduced by any generated block above is filled too. The
    # landing page is the held-out case throughout, so its numbers come from that case.
    body = fill(body, facts(holdout) | candidate_facts(holdout) | chain_facts(holdout)
                | picked_facts(holdout) | lane_facts(holdout)
                | site_patch_facts(holdout))
    # The fold renderer once for the page, after every fold placeholder it draws into.
    body += FOLD_TRACE_JS
    out = (head(f"{SITE} · one pathogen genome in, every next step out",
                "Deciding which of a new virus's proteins a drug should aim at, from the "
                "genome accession alone, with every step on the record. The walkthrough "
                "runs on SARS-CoV-2, held out of its own reference.",
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
    # Filled from the pipeline's own constants rather than from a run. This page is about the
    # shape of the chain on any virus, so the record of one genome is the wrong source -- but
    # the residue ceiling and the macrocycle footprint are properties of the code, identical
    # on every genome, and typed into this page as literals they would go on drawing the old
    # line after either one moved. Both are passed whether or not today's copy asks for them,
    # because `fill` refuses a token it has no value for and the page is hand-written.
    body = fill(body, {"fold_ceiling": FOLD_CEILING,
                       "footprint": f"{MACROCYCLE_FOOTPRINT_A2:,.0f}"})
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


def _cut_summary(case):
    """"N identified ORFs cut into M chains", counted from the record, for a heading tag.

    `identified` and not `orfs_ranked`, because an ORF that reached no relative is cut into
    nothing: stage 2 needs a homolog's mature-chain boundaries to cut by, so the unidentified
    rows leave the list here. Against `orfs_ranked` the tag read "12 ORFs cut into 8 chains"
    on Nipah, which describes cutting twelve things into eight. Against `identified` it reads
    7 into 8, and Zika -- the genome where only the polyprotein resolves -- reads 1 into 12,
    which is the clearest statement of what this stage does on the whole site.
    """
    by_tool = {s["tool"]: s for s in case.get("stages", [])}
    orfs = (by_tool.get("intake.rank_targets", {}).get("output") or {}).get("identified")
    chains = len(case.get("scores") or [])
    if orfs and chains:
        return (f"{orfs} identified ORF{'' if orfs == 1 else 's'} "
                f"cut into {chains} chains")
    return f"{chains} chains" if chains else "no chains ranked"


def gate_block(case, tool):
    """One stage's gate block, or None for a record written before the gate existed.

    Every caller treats None as "this run had no gate" and prints nothing, which is what
    keeps the pages built from the records already on disk identical to what they were. A
    note asserting a floor the run never applied would be the page inventing a stage.
    """
    stage = {s["tool"]: s for s in case.get("stages", [])}.get(tool)
    return ((stage or {}).get("output") or {}).get("gate")


def gate_note(case, tool, noun):
    """What the gate removed from the table above, named and with the rule it applied.

    The table is the kept rows, and a table that is quietly shorter than the count in the
    stage block beside it is the page hiding a filter. So the rows are named here. Naming
    them is also the only form of this note worth printing: "six chains were dropped" cannot
    be argued with, and "Protein C, prM, NS2A, NS2A-alpha, NS4A and NS4B were dropped, none
    paying anything on any of the four" can be -- by a reader who thinks the capsid belongs
    in the list, which is the disagreement the floor is there to provoke.
    """
    block = gate_block(case, tool)
    if not block or not block.get("gated_out"):
        return ""
    dropped = block["dropped"]
    named = ", ".join(esc(row["name"]) for row in dropped if row.get("name"))
    terms = ", ".join(f'<span class="mono">{esc(t)}</span>' for t in block["terms"])
    return (f'<p class="after">{len(dropped)} more {esc(noun)} were scored and are not in '
            f'this table. A row is kept when at least one of {terms} pays '
            f'{num(block["floor"])} points or more, and these paid under that on all '
            f'{len(block["terms"])}. They leave the pipeline here: nothing after this stage '
            f'folds them, measures a surface on them or reads literature for them. '
            f'Conservation is not one of the four, because it says how well this genome was '
            f'recognised rather than whether what was recognised is worth aiming at. The '
            f'numbering keeps each row its place in the full ranking, so a gap in the # column '
            f'is one of these. They are: {named}.</p>')


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
    * Only the ORFs stage 1 actually identified. Three of the twelve score nothing at all --
      no relative at either homology tier and no surface signal in their own sequence -- and
      they used to sit in the table under their reason, as the operating envelope rather than
      a gap. The project lead asked on 12 Nov 2026 for the nine instead: a row that reads
      "Not identified" across every column is not a ranking, and the table is the ranking.

      The counts they were standing in for must therefore survive somewhere a reader sees.
      They do, in stage 1's own In/Out block beside this table, which states that twelve were
      ranked and three came back unrecognised and off the table. If that block is ever
      rewritten, the three stop being reported anywhere on the page, which is the one outcome
      this filter must not produce.
    * And, since 13 Nov 2026, only the ORFs that cleared the evidence gate. That is a second
      and narrower filter than the one above: these rows WERE scored, and were dropped for
      paying under half a point on all four of exposure, essentiality, drugged class and
      solved relatives. They are named under the table rather than counted, for the same
      reason the three unidentified ones are counted beside it -- a row removed from a ranking
      has to be findable on the page that published the ranking.
    """
    rows, record = candidate_rows(case)
    rows = [r for r in rows if r.get("score") is not None]
    if not rows:
        return ""
    # Where the weights come from depends on which record supplied the rows. The stage-1
    # record carries `rubric` and `term_ceiling` at its top level; a chain record does not --
    # it keeps the weights inside stage 2's output, because stage 2 is the stage that publishes
    # them. Reading `case["rubric"]` worked only for as long as no chain record carried its own
    # `candidates` and this function always fell back to the stage-1 record; the first chain
    # record to carry them raised KeyError here.
    if record is not None:
        rubric, ceiling = record["rubric"], record["term_ceiling"]
    else:
        rubric, ceiling = _chain_rubric(case)
    top = max((r["score"] for r in rows), default=0) or 1
    total_ceiling = round(sum(ceiling[key] for key, _, _ in TERM_COLUMNS), 2)

    head_cells = term_head_cells(rubric, ceiling)

    body = []
    for row in rows:
        where = (f'{esc(row["strand"])}{esc(row["frame"])} · nt {row["nt_start"]:,} '
                 f'· {row["length_aa"]:,} aa')
        cells = term_cells(row["features"], row["terms"], ceiling,
                           row.get("conservation_scale"))
        route = row.get("identification_route")
        homolog = row.get("nearest_organism") or row.get("nearest_homolog")
        via = (f'<span class="w">{esc(homolog)}'
               f'{" · " + esc(route) if route else ""}</span>' if homolog else
               f'<span class="w">{esc(route or "")}</span>')
        # Anchored per row, so the shortlist two sections down can link a mature chain back
        # to the ORF it was cut out of. That link is the one relationship the two tables
        # cannot show on their own: twelve of the twenty chains come from this list's second
        # row, and without the anchor a reader has to match protein names by eye.
        body.append(f"""
      <tr id="orf-{esc(row['rank'])}">
        <td class="rk">{esc(row['rank'])}</td>
        <td class="nm"><span class="t">{esc(row['name'])}</span>{via}
          <span class="w">{where}</span></td>
        {cells}
        <td class="sc"><span class="v">{row['score']:.2f}</span>
          <span class="tr"><i style="width:{max(2, row['score'] / top * 100):.1f}%"></i></span></td>
      </tr>""")

    note = ""
    if record is not None:
        note = (f'<p class="after">The five weights and every number in this table are read '
                f'from <span class="mono">runs/{esc(CANDIDATES_RECORD)}</span>, which is '
                f'this same stage run on its own. The chain record keeps stage 1 as counts, '
                f'and the build refuses to ship this table unless those counts match. '
                f'The later sections are from '
                f'<span class="mono">runs/{esc(LANDING_RECORD)}</span>.</p>')

    # Only when these rows are the gated run's own. On the fallback path the table is read
    # from a separate stage-1 record that was never gated, so quoting the chain record's gate
    # beside it would caption one run's rows with another run's filter.
    gated = gate_note(case, "intake.rank_targets", "identified ORFs") if record is None else ""
    note += gated
    kept = " and the gate kept" if gated else ""

    return f"""
  <div class="cmxscroll" tabindex="0" role="region" aria-label="The ranked ORFs and their scores">
  <table class="cmx">
    <caption>Every ORF stage 1 identified{kept}, highest composite score first. Each cell is the points
      that weight paid, with the measurement it was computed from underneath; the shading is
      the share of that column's maximum. Nothing here is fitted — the five weights were
      fixed before the run.</caption>
    <thead>
      <tr>
        <th scope="col" class="rk">#</th>
        <th scope="col" class="nm">ORF, and what it matched</th>
        {head_cells}
        <th scope="col" class="sc">Composite<span class="cw">of {num(total_ceiling)}</span></th>
      </tr>
    </thead>
    <tbody>{"".join(body)}</tbody>
  </table>
  </div>{note}"""


def _chain_rubric(case):
    """Stage 2's own five weights, and the ceiling they imply, from the chain record."""
    st2 = {s["tool"]: s for s in case["stages"]}["intake.decompose + intake.rank_score"]
    rubric = st2["output"]["rubric"]
    return rubric, term_ceilings(rubric)


def chain_matrix(case, link_orfs=True):
    """Every mature chain stage 2 scored, with the score split into what each weight paid.

    Stage 2's output block used to be three sentences: how many chains came out, that the
    weights were fixed in advance, and that every score re-adds from its parts. All three are
    true and none of them is the stage's result. The result is twenty chains, each with a
    residue range, a parent and a score, and the one question a reader has at this altitude --
    why is the main protease 6.90 and the protein next to it in the same polyprotein 2.50 --
    could not be asked of a count.

    So the arithmetic is the table. Two of the five terms are the reason the cut changes the
    ranking: `essential_class` and `druggable_class` are read from each chain's own annotation,
    and `structures_in_family` is counted over each chain's own range, so chains sharing a
    parent separate on those three while inheriting its conservation and its exposure. The
    inheritance is visible as a repeated number down the column rather than claimed in prose,
    which is also the honest way to show it: conservation is measured once per ORF, and twelve
    chains carrying 0.835 is a limit of the cut, not twelve measurements.

    `link_orfs` is off for the per-genome result pages, which carry no stage-1 table to link
    to. Those rows name the parent instead -- the same decision `score_bars` made before this
    table replaced it, for the same reason: a dead anchor is worse than plain text.
    """
    rows = case.get("scores") or []
    if not rows:
        return ""
    rubric, ceiling = _chain_rubric(case)
    orf = {}
    if link_orfs:
        candidates, record = candidate_rows(case)
        if record and record.get("rubric") and record["rubric"] != rubric:
            raise SystemExit(
                "stage 1 and stage 2 scored on different weights "
                f"({record['rubric']} vs {rubric}). The two tables cannot be read against "
                "each other. The build stops rather than shipping them side by side.")
        ceiling = term_ceilings(rubric, record)
        orf = {c["name"]: c for c in candidates if c.get("name")}
    top = max((r.get("score") or 0) for r in rows) or 1
    total_ceiling = round(sum(ceiling[key] for key, _, _ in TERM_COLUMNS), 2)

    body = []
    for r in rows:
        # A chain with no `parent` is a protein that was already whole -- the ORF and the
        # chain are the same object -- so it matches the stage-1 row by its own name.
        parent = r.get("parent") or r.get("target")
        row1 = orf.get(parent)
        if row1 is not None:
            # The rank, and not the ORF's own score. Stage 1's table is read from a separate
            # record of the same stage, so the two scores are from two runs and differ in the
            # third decimal; printed side by side in one row they would read as the cut moving
            # a score, which is the one thing this table is here to let a reader judge.
            cut = (f'<a href="#orf-{esc(row1["rank"])}">{esc(parent)}</a>'
                   f'<span class="w">stage&nbsp;1 rank {esc(row1["rank"])}</span>')
        else:
            cut = esc(parent)
        if not r.get("parent"):
            cut += '<span class="w">not a polyprotein</span>'
        start_end = r.get("chain_range") or []
        where = (f'residues {start_end[0]:,}–{start_end[1]:,} of the parent'
                 if len(start_end) == 2 else "the whole protein")
        features = {key: r.get(key) for key, _, _ in TERM_COLUMNS}
        cells = term_cells(features, r["score_terms"], ceiling,
                           None if r.get("tier") == "exact" else "bit_score_over_200")
        score = r.get("score") or 0
        body.append(f"""
      <tr id="chain-{esc(r.get('rank'))}">
        <td class="rk">{esc(r.get('rank'))}</td>
        <td class="nm"><span class="t">{esc(r.get('target'))}</span>
          <span class="w">{esc(r.get('nearest_homolog'))}</span>
          <span class="w">{where}</span></td>
        <td class="cut">{cut}</td>
        {cells}
        <td class="sc"><span class="v">{score:.2f}</span>
          <span class="tr"><i style="width:{max(2, score / top * 100):.1f}%"></i></span></td>
      </tr>""")

    linked = (" and links to its row in the table above" if link_orfs else "")
    note = gate_note(case, "intake.decompose + intake.rank_score", "mature chains")
    kept = " and the gate kept" if note else ""
    return f"""
  <div class="cmxscroll" tabindex="0" role="region" aria-label="Every mature chain and its score">
  <table class="cmx">
    <caption>Every mature chain stage 2 produced{kept}, highest composite score first. Each cell is
      the points that weight paid, with the measurement it was computed from underneath; the
      shading is the share of that column's maximum. &ldquo;Cut out of&rdquo; names the ORF the
      chain was taken from{linked}, so a chain's score can be read against its parent's.
      Same five weights as stage 1, fixed before the run.</caption>
    <thead>
      <tr>
        <th scope="col" class="rk">#</th>
        <th scope="col" class="nm">Mature chain &mdash; what a binder would be aimed at</th>
        <th scope="col" class="cut">Cut out of</th>
        {term_head_cells(rubric, ceiling)}
        <th scope="col" class="sc">Composite<span class="cw">of {num(total_ceiling)}</span></th>
      </tr>
    </thead>
    <tbody>{"".join(body)}</tbody>
  </table>
  </div>{note}"""


def chain_facts(case):
    """What stage 2's output block says, counted from the chains it is describing.

    The two lines this replaces -- five ingredients fixed in advance, twenty scores that
    re-add from their parts -- described the scorer rather than its result, and both are
    stated where they are checkable anyway: the weights head the table's own columns, and
    section 04's card carries the re-add count. These are the facts a reader needs to read
    the table: how the twenty divide, how far the scores spread, and whether cutting a
    polyprotein up ever buys one of its pieces a higher score than the whole.
    """
    rows = case.get("scores") or []
    if not rows:
        return {}
    _, ceiling = _chain_rubric(case)
    scores = [r.get("score") or 0 for r in rows]
    cut = [r for r in rows if r.get("parent")]
    out = {
        "cut_chains": len(cut),
        "whole_chains": len(rows) - len(cut),
        "score_high": f"{max(scores):.2f}",
        "score_low": f"{min(scores):.2f}",
        "score_ceiling": num(round(sum(ceiling[key] for key, _, _ in TERM_COLUMNS), 2)),
    }
    # The spread inside one parent, which is what a single score at the ORF altitude hides.
    # Taken from the ORF that yielded the most chains, which is the row the prose points at.
    # Both ends are read from this record and not from stage 1's, so the comparison stays
    # inside one run: the two records are the same stage of the same genome, but they are two
    # runs, and their scores differ in the third decimal.
    parents = [r["parent"] for r in cut]
    if parents:
        biggest = max(set(parents), key=parents.count)
        inside = [r.get("score") or 0 for r in cut if r["parent"] == biggest]
        out["cut_high"] = f"{max(inside):.2f}"
        out["cut_low"] = f"{min(inside):.2f}"
    return out


def picked_table(case):
    """The shortlist stage 2 actually produced, with each chain tied to the ORF it came from.

    This section used to be eight hand-typed bars: literal scores, literal CSS bar widths,
    and plain-language names invented for the page. Three things were wrong with that, and
    only one of them was the retyping.

    It was truncated to the top eight of twenty with no way to see the rest. It carried the
    scores of a run two scoring changes ago, under a headline -- "Spike and the main protease
    come out level at the top" -- that had stopped being true. And it renamed every chain, so
    `nsp12` appeared as "Copying machine (RdRp)" with nothing linking it to anything above it
    on the page; a reader could not tell that this list and the ORF list are different cuts of
    the same genome rather than two unrelated rankings.

    The last of those is the one worth fixing properly, because the relationship it hides is
    the point of the stage. Twelve of these twenty chains are cut out of a single ORF, the
    replicase polyprotein, and the two targets that carry the project's claim -- the main
    protease and the polymerase, which nirmatrelvir and remdesivir hit -- are among them. They
    do not exist at the ORF altitude at all: stage 1 can only ever rank the whole polyprotein,
    which is why a page that stopped at the ORF table could not show that the system found
    them. So every row here names its parent ORF and links to that ORF's row in the stage-1
    table, which makes the cut traceable in the one direction a reader needs it.
    """
    rows = case.get("scores") or []
    if not rows:
        return ""
    # The stage-1 ranks to link to. A chain's parent is the ORF it was cut from; a chain that
    # is a whole protein has no parent and matches the stage-1 row by its own name. Where
    # neither resolves the cell is plain text rather than a link -- a wrong anchor is worse
    # than no anchor, and the two tables can legitimately come from different runs.
    candidates, _ = candidate_rows(case)
    orf_rank = {c["name"]: c["rank"] for c in candidates if c.get("name")}
    top = max((r.get("score") or 0) for r in rows) or 1

    body = []
    for r in rows:
        parent = r.get("parent") or r.get("target")
        rank = orf_rank.get(parent)
        cut = (f'<a href="#orf-{esc(rank)}">{esc(parent)}</a>'
               f'<span class="w">stage&nbsp;1 rank {esc(rank)}</span>'
               if rank is not None else f'{esc(parent)}')
        whole = ' <span class="w">not a polyprotein</span>' if not r.get("parent") else ""
        score = r.get("score") or 0
        body.append(f"""
      <tr>
        <td class="rk">{esc(r.get('rank'))}</td>
        <td class="nm"><span class="t">{esc(r.get('target'))}</span>{whole}</td>
        <td class="cut">{cut}</td>
        <td class="sc"><span class="v">{score:.2f}</span>
          <span class="tr"><i style="width:{max(2, score / top * 100):.1f}%"></i></span></td>
      </tr>""")

    return f"""
  <div class="cmxscroll" tabindex="0" role="region" aria-label="The ranked shortlist of mature chains">
  <table class="cmx pick">
    <caption>Every mature chain stage 2 produced, highest score first. &ldquo;Cut out of&rdquo;
      names the ORF from the stage-1 table above and links to its row there, so a chain can be
      traced back to the stretch of genome it came from. Scores are on the same five weights.</caption>
    <thead>
      <tr>
        <th scope="col" class="rk">#</th>
        <th scope="col" class="nm">Mature chain &mdash; what a binder would be aimed at</th>
        <th scope="col" class="cut">Cut out of</th>
        <th scope="col" class="sc">Score</th>
      </tr>
    </thead>
    <tbody>{"".join(body)}</tbody>
  </table>
  </div>"""


def picked_facts(case):
    """The numbers the shortlist's prose quotes, read from the record it describes.

    The claim this section exists to make is that two of the chains near the top are the
    targets of approved antivirals, and the ranks in that sentence were typed by hand. They
    are looked up here instead, by the `nsp` designation in the chain's own name, so the
    sentence follows the ranking rather than asserting where it used to sit. A designation
    that no longer matches any chain raises through `fill` rather than printing a blank rank.
    """
    rows = case.get("scores") or []
    out = {}
    for token, key in (("nsp5", "mpro_rank"), ("nsp12", "rdrp_rank")):
        hit = next((r for r in rows
                    if re.search(rf"\b{token}\b", r.get("target") or "")), None)
        if hit:
            out[key] = hit["rank"]
    # How concentrated the cut is: the number of chains taken out of whichever single ORF
    # yielded the most of them. It is the figure that makes the stage legible -- one row of
    # the stage-1 table becoming most of this one -- and it is counted, not asserted.
    parents = [r.get("parent") for r in rows if r.get("parent")]
    if parents:
        top_parent = max(set(parents), key=parents.count)
        out["biggest_cut"] = parents.count(top_parent)
        out["biggest_cut_orf"] = top_parent
    return out


def surface_block(case):
    """One tile per chain and what the three colours mean. The detail is `surface_lane`.

    This used to carry the bars and the refusal sentences as well. `surface_lane` draws both,
    per chain, against the threshold the gate applies and on a scale shared with the predicted
    patches -- so keeping them here would have printed every measured chain twice on one page
    with two different sets of bar widths, which is worse than either.

    The strip stays because it is the one thing the lane cannot do: twenty outcomes in one
    line, before a reader commits to reading twenty rows.
    """
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
  </div>"""



# ------------------------------------------------------ the two lanes stages 3 and 4 draw
# Stages 3 and 4 were five bullets each on the landing page, and the numbers in them were
# literals: "11 of those are short enough", "5 of the 11 with nothing deposited", "14 of 20".
# Every one of those is a join between the fold stage's rows and the surface stage's rows --
# the join the reorder exists to create -- so the page was asserting the pipeline's new
# behaviour in prose with the record sitting beside it unread. These two functions draw the
# join: one row per chain in each stage, on a shared scale, with the threshold that stage
# applies drawn through every row.
#
# Both thresholds are imported and neither is typed. The residue ceiling decides which chains
# reach the folding endpoint and therefore which chains stage 4 can rescue; the macrocycle
# footprint is what the gate thresholds on. A literal in this file would go on drawing the
# old line after either one moved, which is the failure the {{token}} rework was for.
sys.path.insert(0, str(ROOT))
from hpc_pilots.chain import FOLD_CEILING                                      # noqa: E402
from hpc_pilots.intake import MACROCYCLE_FOOTPRINT_A2                          # noqa: E402


def _ceiling(case):
    """The residue ceiling stage 3 applied, from the record where the record states it.

    Records written before the stage recorded its own ceiling fall back to the constant the
    stage still uses, which is the same number those runs applied. The fallback is an import
    and not a literal, so if the ceiling moves, a page rebuilt from an old record moves with
    it rather than keeping a line the code no longer draws.
    """
    folded = {s["tool"]: s for s in case.get("stages", [])}.get("structures.folded") or {}
    return (folded.get("input") or {}).get("residue_ceiling") or FOLD_CEILING


def _lane_rows(case):
    """Every ranked chain, with both stages' rows for it joined on the name they share.

    Keyed on the target and not on position. Both lists are in rank order today, but pairing
    row 7 with row 7 would attribute one chain's coordinates to another the first time one of
    the two stages emits its rows in a different order, and that is a mistake no reader of the
    page could see.

    `predicted` is the recorded measurement on stage 3's coordinates. `inside` is the weaker
    statement that the chain is short enough for stage 3 to have folded it, which is all a
    record written before the reorder can support -- `_predicted_is_recorded` is what
    distinguishes them, so the page can say which of the two it is showing instead of
    presenting a projection as a measurement.
    """
    ceiling = _ceiling(case)
    folds = {f.get("target"): f for f in (case.get("foldable") or [])}
    rows = []
    for surface in (case.get("surfaces") or []):
        fold = folds.get(surface.get("target")) or {}
        length = fold.get("recovered_length_aa")
        located = bool(fold.get("sequence_recovered"))
        predicted = surface.get("predicted_surface") or {}
        rows.append({
            "target": surface.get("target"), "rank": surface.get("rank"),
            "surface": surface, "fold": fold, "length": length, "located": located,
            "inside": bool(located and length and length <= ceiling),
            "measured": bool(surface.get("measured")),
            "predicted": predicted if predicted.get("measured") else None,
        })
    return ceiling, rows


def _predicted_is_recorded(case):
    """Whether this run measured any predicted surface at all, rather than how many.

    A run from before folding moved ahead of the surface stage has no `predicted_surface` on
    any row, and the difference between "no chain was rescued" and "this record cannot say"
    is the difference between a result and a projection.
    """
    return any("predicted_surface" in r for r in (case.get("surfaces") or []))


def domain_inventory(case):
    """The annotated functional regions for this case's chains, or None if not recorded.

    Refused outright unless the record describes the same run as the page. The inventory is
    keyed by target name and so are the lane rows, so an inventory of a different genome --
    or of the same genome identified through its own curated entries rather than held out of
    them -- would join cleanly and silently, and attach one run's domains to another run's
    chains. The two differ in exactly the way that matters here: held out, the spike's
    relative is a bat coronavirus whose entry does not name a receptor-binding domain at all.

    Both keys are checked because neither is sufficient. The accession alone admits the
    un-held-out case of the same genome, and the excluded organism alone admits a different
    genome held out of the same reference.
    """
    path = RUNS / DOMAIN_RECORD
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if (payload.get("genome_accession") != case.get("genome_accession")
            or payload.get("reference_excluded_organism")
            != case.get("reference_excluded_organism")):
        return None
    return payload


FOLD_RECORD = "domain_folds.json"
FOLD_STORE = "domain_folds"

# Dated, citable literature for each shortlisted target and each domain that was folded.
# Written by `python -m hpc_pilots.literature_stage`.
LITERATURE_RECORD = "literature_stage.json"


def literature_record(case):
    """The literature stage's record for this case, or None.

    Guarded on the same two keys the other two records are, and for the same reason: a
    literature row keyed by target name would join a different run's chains silently.
    """
    path = RUNS / LITERATURE_RECORD
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if (payload.get("genome_accession") != case.get("genome_accession")
            or payload.get("reference_excluded_organism")
            != case.get("reference_excluded_organism")):
        return None
    return payload


# Inline markup a publisher put inside a record title. Europe PMC returns titles with
# `<em>` and friends in them -- "Effect of Drug Binding to the Variant-<em>in Silico</em>
# Analysis" -- and escaping the title, which is the only safe thing to do with a string
# this page did not write, prints those tags at the reader. The tags are dropped and their
# text kept, so the title reads as a title. Everything that survives is still escaped on
# the way out; this never puts a stranger's angle bracket back into the page.
_TITLE_TAG = re.compile(r"</?(?:em|i|b|strong|sub|sup|u|span)\b[^<>]*>", re.I)


def _plain_title(text):
    """A record title with a publisher's inline tags dropped and their text kept."""
    return re.sub(r"\s{2,}", " ", _TITLE_TAG.sub("", text)).strip()


def _hit_rows(query):
    """The dated hits of one query, as links.

    A hit with no publication date is drawn and marked rather than dropped. The record keeps
    it under `citable: false` with a reason, and a page that silently showed only the dated
    ones would make the retrieval look cleaner than it was.
    """
    out = []
    for hit in query.get("hits") or ():
        date = hit.get("first_publication_date") or ""
        marks = []
        if hit.get("is_preprint"):
            marks.append('<span class="pp">preprint</span>')
        if not hit.get("citable"):
            marks.append(f'<span class="pp un">{esc(hit.get("uncitable_because") or "undated")}</span>')
        title = esc(_plain_title(hit.get("title") or "untitled"))
        url = hit.get("url")
        linked = f'<a href="{esc(url)}" rel="noopener">{title}</a>' if url else title
        out.append(f"""
        <li><span class="lt">{linked}</span>
          <span class="lm">{esc(date[:10]) or '&mdash;'}
            {('&middot; ' + esc(hit.get('journal'))) if hit.get('journal') else ''}
            {' '.join(marks)}</span></li>""")
    return "".join(out)


def _query_block(query):
    """One composed query: what was asked, what came back, and the rows worth citing."""
    scope = query.get("scope")
    label = query.get("domain_label")
    span = query.get("domain_span") or []
    where = (f' <span class="w">{span[0]:,}&ndash;{span[1]:,}</span>'
             if len(span) == 2 else "")
    name = (f'<b>{esc(label)}</b>{where}' if scope == "domain"
            else "<b>the whole chain</b>")
    status = query.get("status")
    count = query.get("hit_count")
    if status != "ok":
        return f"""
      <div class="lq x">
        <div class="lqh">{name}<span class="lqn">{esc(status)}</span></div>
        <div class="lqq">{esc(query.get('subject'))}</div>
        <div class="lqz">{esc(query.get('note') or 'not asked')}</div>
      </div>"""
    preprints = query.get("preprints")
    tally = (f"{num(count)} record{'s' if count != 1 else ''}"
             + (f", {num(preprints)} of them preprints" if preprints else ""))
    empty = ('<div class="lqz">No record matches this query. A measured zero, not a '
             'question that was skipped.</div>' if not count else "")
    return f"""
      <div class="lq">
        <div class="lqh">{name}<span class="lqn">{tally}</span></div>
        <div class="lqq">{esc(query.get('subject'))}</div>
        {empty}
        <ol class="lh">{_hit_rows(query)}</ol>
      </div>"""


def literature_block(case):
    """Dated literature for the shortlist, per target and per folded domain.

    Placed between stages 3 and 4 because that is where it becomes answerable. Before stage 3
    a domain is a span on somebody else's entry; after it the domain has coordinates, and a
    region with coordinates is one a binder could be designed against this week -- so the
    literature for it is the literature a designer would actually open. That is also why the
    queries are per folded domain rather than per annotated region: the stage asks about the
    things stage 3 made real.

    The composed query is printed on every row. This repository says repeatedly that a number
    whose query is not recorded cannot be checked, and these are the numbers most able to
    mislead: a hit count is evidence of attention and of nothing else, which the retrieval
    module's own docstring is emphatic about, so the page shows the question rather than
    letting the count stand on its own.
    """
    payload = literature_record(case)
    if not payload:
        return ""
    targets = payload.get("targets") or []
    if not targets:
        return ""
    blocks = []
    for row in targets:
        queries = row.get("queries") or []
        asked = [q for q in queries if q.get("status") == "ok"]
        dated = row.get("dated_hits") or 0
        route = row.get("anchor_route")
        # Why this target's query is shaped the way it is. The two routes are a measured
        # decision in the record, not a quirk, and a reader comparing two targets' queries
        # will otherwise see an inconsistency rather than a rule.
        why = ("anchored on the genome accession, because the chain's name yields no "
               "systematic designation and a one-word protein noun does not identify a "
               "protein" if route == "genome_accession" else
               "anchored on the chain's own designation, which identifies the protein "
               "without needing the accession")
        # One disclosure per target, shut. Open, this section is four hundred lines of
        # citations -- thirteen queries, five records apiece -- and a reader scrolling to
        # stage 4 has to pass every one of them. Shut, the same records are two clicks from
        # anywhere and the section is a list of targets with its counts on it, which is what
        # a reader checks first anyway. Nothing is dropped and nothing is fetched on demand:
        # the markup is all here, the browser is merely not drawing it.
        # The shut row carries the name and the counts and nothing else. How the query was
        # anchored is two lines of explanation that belong beside the queries themselves,
        # which is where it now sits -- on the shut row it was longer than everything else
        # on it put together.
        blocks.append(f"""
    <details class="lrow">
      <summary class="lrh"><span class="t">{esc(row.get('target'))}</span>
        <span class="w"><em>{esc(row.get('nearest_homolog'))}</em>
          <b>{num(dated)}</b> dated record{'s' if dated != 1 else ''} kept
          &middot; <b>{len(asked)}</b> of <b>{len(queries)}</b>
          {'queries' if len(queries) != 1 else 'query'} answered</span></summary>
      <div class="lwhy">{why}</div>
      {"".join(_query_block(q) for q in queries)}
    </details>""")

    # Stated as what the retrieval IS rather than as what it is not. The record's own
    # `date_ceiling_means` spells the consequence out at length, which belongs in the record;
    # on the page the fact that these are today's records is the whole of it, and a reader
    # who wants it argued has the record. Where a ceiling WAS applied the sentence has to
    # carry the date, because then it is the measurement the control query exists to make.
    # Distinct papers, not hit rows. The record's own headline counts rows, and the two are
    # not close: 65 rows across this run are 33 papers. Two targets that compose the same
    # query are served the same hits -- S2' and S2 are both the spike and anchor identically,
    # so four of S2's queries are answered from the cache and repeat S2's records row for row.
    # Counting rows is right for the record, which is per query; a reader of the page counts
    # papers, and 65 would read as twice the literature there is.
    seen = {(h.get("source"), h.get("id"))
            for row in targets for q in (row.get("queries") or ())
            for h in (q.get("hits") or ()) if h.get("id")}
    rows_shown = sum(len(q.get("hits") or ()) for row in targets
                     for q in (row.get("queries") or ()))
    asked = sum(1 for row in targets for q in (row.get("queries") or ())
                if q.get("status") == "ok")
    headline = (f"{num(len(seen))} distinct records over {num(asked)} composed "
                f"{'queries' if asked != 1 else 'query'}"
                + (f", listed {num(rows_shown)} times because targets that compose the same "
                   f"query are shown the same records" if rows_shown != len(seen) else ""))
    ceiling = payload.get("date_ceiling_applied")
    note = ("Retrieved without a publication-date limit, so these are the records as they "
            "stand today." if not ceiling else
            f"Limited to work published on or before {esc(payload.get('date_to'))}, checked "
            f"against a control query that omits the limit.")
    return f"""
  <div class="lit">
    <div class="lith">{headline}<span>{note}</span></div>
    {"".join(blocks)}
  </div>"""


def domain_folds(case):
    """The folds made for this case's chains and domains, keyed by target, or None.

    Guarded on the same two keys `domain_inventory` is guarded on and for the same reason: a
    fold of one genome's chain attached to another genome's row would join silently.
    """
    path = RUNS / FOLD_RECORD
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if (payload.get("genome_accession") != case.get("genome_accession")
            or payload.get("reference_excluded_organism")
            != case.get("reference_excluded_organism")):
        return None
    return {c.get("target"): c for c in (payload.get("chains") or ())}


def build_domain_fold_images():
    """One figure per stored fold, through the same function the committed fold uses.

    Reusing `panels.structure_figure` rather than drawing something lighter is deliberate.
    These sit on the same page as the main protease's figure, and two renderers would differ
    in projection, in the confidence bands and in how a low-confidence stretch is marked --
    so a reader comparing a domain against the chain figure above it would be comparing two
    drawing conventions as well as two structures.

    Written at a lower dpi than the committed figure. There is one of these per folded unit
    rather than one on the page, and the full-resolution figure is 240 KB.
    """
    record = RUNS / FOLD_RECORD
    if not record.exists():
        print("  no domain folds recorded on this machine; no figures written")
        return
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-site")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "notebooks"))
    import panels                                                      # noqa: E402
    import matplotlib.pyplot as plt                                     # noqa: E402

    out = DOCS / FOLD_STORE
    out.mkdir(parents=True, exist_ok=True)
    payload = json.loads(record.read_text())
    written = 0
    for chain in payload.get("chains") or ():
        for unit in chain.get("units") or ():
            slug = unit.get("slug")
            if not unit.get("folded") or not slug:
                continue
            pdb = RUNS / FOLD_STORE / f"{slug}.pdb"
            store = RUNS / FOLD_STORE / f"{slug}.json"
            if not (pdb.exists() and store.exists()):
                continue
            image = out / f"{slug}.png"
            # Skipped when the figure is already newer than the coordinates it draws. A
            # rebuild of the site should not re-render twenty figures that have not changed.
            if image.exists() and image.stat().st_mtime >= pdb.stat().st_mtime:
                continue
            stored = json.loads(store.read_text())
            resnum, xyz, bfac = panels.ca_trace(pdb.read_text())
            if not len(resnum):
                continue
            plddt, _ = panels.plddt_on_0_100(bfac)
            kind = ("the whole chain" if unit.get("kind") == "whole_chain"
                    else f"{unit.get('region_type') or 'region'} of {chain.get('target')}")
            fig = panels.structure_figure(
                resnum, xyz, plddt,
                title=f"{unit.get('label')}, folded from sequence alone",
                subtitle=f"ESMFold v1 · {kind} · {len(resnum)} residues · mean "
                         f"confidence {plddt.mean():.1f} of 100",
                runs=tuple(tuple(r) for r in stored.get("low_confidence_runs", [])))
            fig.savefig(image, facecolor=fig.get_facecolor(), bbox_inches="tight", dpi=110)
            plt.close(fig)
            written += 1
    print(f"wrote {written} structure figure(s) to docs/{FOLD_STORE}/")


def _fold_panel(unit, chain_target):
    """One folded structure, with the numbers that say what it is and how good it is.

    The two spans are both printed, which is the point of the row rather than a detail: the
    annotation is numbered on the relative and the fold is numbered on this genome, and the
    two differ by more than a constant. Printing only one of them would leave a reader unable
    to tell which they were looking at.
    """
    slug = unit.get("slug")
    if not unit.get("folded") or not slug:
        return ""
    entry = unit.get("entry_span_on_the_relative") or []
    here = unit.get("span_in_this_genome") or []
    plddt = unit.get("mean_plddt")
    where = []
    if len(entry) == 2:
        where.append(f"{entry[0]:,}&ndash;{entry[1]:,} on the relative&rsquo;s entry")
    if len(here) == 2:
        where.append(f"{here[0]:,}&ndash;{here[1]:,} in this genome")
    adjusted = ("" if not unit.get("ends_adjusted") else
                " &middot; one boundary had no counterpart in this genome and the window was "
                "taken inward to the nearest residue that did")
    sites = unit.get("site_residues_inside")
    marked = (f" &middot; {sites} residue{'s' if sites != 1 else ''} the curators marked as "
              f"doing the chemistry" if sites else "")
    confidence = (f"mean confidence <b>{plddt:.1f}</b> of 100"
                  if isinstance(plddt, (int, float)) else "confidence not recorded")
    image = f"""<img src="{FOLD_STORE}/{esc(slug)}.png" loading="lazy"
          alt="The predicted backbone of {esc(unit.get('label'))}, drawn on its two principal
            axes and coloured by per-residue confidence, beside a plot of that confidence
            against residue position.">"""
    # Traced by residue like the committed fold, with this PNG for a reader without scripts.
    return f"""
      <div class="dfold">
        {unit_trace(unit, image)}
        <div class="cc"><b>{esc(unit.get('label'))}</b> &middot;
          {num(unit.get('length_aa'))} residues &middot; {confidence}{marked}<br>
          {' &middot; '.join(where)}{adjusted}</div>
      </div>"""


def _rescued(inventory):
    """How many over-ceiling chains the inventory found a foldable region for.

    Read from the record's own summary rather than recounted from its rows. The two would
    agree today; the point is that if they ever stopped agreeing, the page would be the wrong
    place to discover it, and recounting here would hide the disagreement instead.
    """
    return ((inventory or {}).get("summary") or {}).get(
        "chains_over_the_ceiling_with_a_foldable_region")


def _region_rows(chain, fold_chain=None):
    """One chain's annotated functional regions, foldable ones first and refusals named.

    **The `where` column is the relative's numbering and used to say nothing else.** The
    inventory record is annotated on the homolog by construction -- it states in its own
    limitations that mapping those coordinates needs stage 3's alignment and that it will not
    approximate one -- so every span in this panel is a position in somebody else's protein.
    The fold record is where the other numbering lives, and joining the two puts this genome's
    span under the relative's on the rows that have one: of the 143 regions the held-out run
    was offered, the 26 that were folded carry a mapped span and the rest were never sent
    through `domains.map_span`, so they have none to show and are not given one.

    The join is on `(type, entry_span)` and not on the label, which is the key
    `chain._in_relevance_order` uses against the same two records. A label join would be
    wrong here for a reason already measured on this page: Spike protein S2 carries two
    `Coiled coil` features with no description between them, and a name that cannot tell
    918-962 from 1144-1172 cannot key a join either.

    The order is the record's, which is the inventory's stated ordering -- contained
    annotated site residues first, then an authored term list, then length -- and is not
    recomputed here. A page that re-sorted would be a second opinion on the question the
    record exists to answer, and the two would drift.

    Every region is drawn, including the refused ones, with the reason on the row. A region
    refused for being five residues long is not noise: its positions are a site a macrocycle
    could be designed at, which is the distinction `domains.MINIMUM_DOMAIN_AA` is careful
    about, and a page that showed only the foldable ones would make the annotation look
    thinner than it is.
    """
    mapped = {(u.get("region_type"), tuple(u.get("entry_span_on_the_relative") or ())):
              u.get("span_in_this_genome")
              for u in (fold_chain or {}).get("units") or ()
              if u.get("kind") == "domain" and u.get("span_in_this_genome")}
    out = []
    for region in chain.get("regions") or ():
        span = region.get("entry_span_on_the_relative") or []
        where = f"{span[0]:,}&ndash;{span[1]:,}" if len(span) == 2 else "&mdash;"
        # The caption under the span says which numbering it is in, and it has to say two
        # different things: a region that was folded has both numberings and the second one is
        # the useful line, while a region that was not has only the relative's and saying so
        # is the whole point of the caption.
        here = mapped.get((region.get("type"), tuple(span)))
        where += ('<span class="dq">' + (f'{here[0]:,}&ndash;{here[1]:,} in this genome'
                                         if here else 'on the relative&rsquo;s entry')
                  + '</span>')
        sites = region.get("site_residues_inside") or 0
        # The reason the row sorts where it does, in the reader's words rather than the
        # record's key names. Shown only when there is one: a region with no contained site
        # residues and no matched term is sorted by length alone, and inventing a phrase for
        # that would read as evidence it does not have.
        why = []
        if sites:
            why.append(f"{sites} annotated site residue{'s' if sites != 1 else ''} inside")
        terms = region.get("relevance_terms_matched") or []
        if terms:
            why.append("names " + ", ".join(f"&ldquo;{esc(t)}&rdquo;" for t in terms[:2]))
        if region.get("describes_position_not_function"):
            why.append("describes position, not function")
        refused = region.get("refused_because")
        out.append(f"""
      <div class="dmn{' x' if refused else ''}">
        <span class="dn">{esc(region.get('description') or region.get('type'))}
          <span class="dt">{esc(region.get('type'))}</span></span>
        <span class="dl">{num(region.get('length_aa'))} aa</span>
        <span class="dw">{where}<span class="dq">on the relative&rsquo;s entry</span></span>
        <span class="dy">{' &middot; '.join(why) if why else '&mdash;'}</span>
        <span class="dv">{'refused' if refused else 'within the ceiling'}
          {f'<span class="dq">{esc(refused)}</span>' if refused else ''}</span>
      </div>""")
    return "".join(out)


def chain_fold_block(fold_chain):
    """The folded structure of a chain that was inside the ceiling all along.

    These were folded by the recorded run and their coordinates dropped at the end of it --
    only the main protease was kept, because it was the one the page drew. So every other row
    inside the ceiling said it had been folded and could not show what came back. The fold is
    made again here, per chain, and the row opens onto it.
    """
    units = [u for u in (fold_chain or {}).get("units") or ()
             if u.get("kind") == "whole_chain" and u.get("folded")]
    if not units:
        return ""
    unit = units[0]
    plddt = unit.get("mean_plddt")
    rating = (f" &middot; mean confidence {plddt:.1f} of 100"
              if isinstance(plddt, (int, float)) else "")
    return f"""
    <details class="dmns">
      <summary>Folded structure &mdash; <b>{num(unit.get('length_aa'))}</b> residues{rating}</summary>
      {_fold_panel(unit, (fold_chain or {}).get('target'))}
    </details>"""


def domain_block(chain, fold_chain=None):
    """The expandable panel of functional regions under one over-ceiling chain.

    A `<details>` and not a `<select>`, for a reason the rest of this page shares: the site is
    static HTML served from a repository with no build step on the client, so a dropdown that
    swapped content would need JavaScript to do it, and a reader with JavaScript off would get
    a control that moves and a page that does not. `<details>` is a disclosure the browser
    implements, keyboard reachable and announced by a screen reader without any help from us.
    """
    foldable = chain.get("regions_foldable") or 0
    offered = chain.get("regions_offered") or 0
    best = chain.get("best_region") or {}
    if not offered:
        return ""
    headline = (f"{foldable} of {offered} annotated regions are within the ceiling"
                if foldable else
                f"none of its {offered} annotated regions is within the ceiling either")
    lead = ""
    if best:
        span = best.get("entry_span_on_the_relative") or []
        # "Offered first" and not "shortest": the order is functional relevance, so the
        # leading region is routinely not the shortest one. Calling it the shortest route in
        # was wrong on nsp3 the moment it was written -- a 369-residue domain leads there
        # while a 56-residue one sits further down the same panel.
        lead = (f' &mdash; offered first is <b>{esc(best.get("description") or best.get("type"))}</b>'
                f' at {num(best.get("length_aa"))} aa'
                + (f' ({span[0]:,}&ndash;{span[1]:,} on the relative)' if len(span) == 2 else ''))
    # The structure first and the table under it. A reader opening the row wants to see that
    # there IS something here before reading twenty-seven rows about what might have been.
    folded = [u for u in (fold_chain or {}).get("units") or ()
              if u.get("kind") == "domain" and u.get("folded")]
    structure = "".join(_fold_panel(u, (fold_chain or {}).get("target")) for u in folded)
    return f"""
    <details class="dmns">
      <summary>{headline}{lead}</summary>
      {structure}
      <div class="dmn-hd">
        <span class="dn">region</span><span class="dl">length</span>
        <span class="dw">where</span><span class="dy">why it is offered here</span>
        <span class="dv">fold unit</span>
      </div>
      {_region_rows(chain, fold_chain)}
    </details>"""


def _bar(width, left=0.0, cls="b"):
    return (f'<i class="{cls}" style="left:{left:.2f}%;width:{max(width, 0.6):.2f}%"></i>')


def _tick(value, scale, faint=False):
    """The threshold line, at the same fraction of every row's width.

    `faint` is for a line that is a reference rather than the threshold itself -- half the
    macrocycle footprint, which is where the gate stops saying no rather than where it starts
    saying yes. Two lines of equal weight would leave a reader unable to tell which one the
    verdict column is about.
    """
    left = min(max(value / scale * 100, 0), 100)
    return f'<i class="thr{" faint" if faint else ""}" style="left:{left:.2f}%"></i>'


def _whisker(low, high, scale):
    """The spread across every structure of one domain, drawn over its own bar.

    The bar is the median and the whisker is how far the structures disagree, so bar LENGTH
    always means the same thing on every row of the lane and the whisker never has to be read
    as a magnitude. Before this they were one mark: a floating segment from smallest to
    largest with a dot on the median and nothing at zero, so the same horizontal extent meant
    "how big" on a predicted row and "how uncertain" on a deposited one, and neither was
    anchored to anything a reader could point at.

    Nothing is drawn for a domain with one structure. A zero-width whisker is not a measured
    agreement -- it is the absence of a second opinion -- and drawing it as a dot would put a
    mark on the page for a quantity that was never measured.
    """
    if not high or high <= low:
        return ""
    left = min(max(low / scale * 100, 0), 100)
    width = min(max((high - low) / scale * 100, 0.4), 100 - left)
    return f'<i class="whisk" style="left:{left:.2f}%;width:{width:.2f}%"></i>'


def _share_axis():
    """The scale for a lane of meters, where each row's track is its own whole.

    A share lane takes no shared absolute scale and must not be given one. The site patches
    run from 0 to 10,262 square angstroms, so one outlying unit on a common axis squeezes
    every other row's pair of bars into the first tenth of the track -- and the quantity that
    lane exists to show is the RATIO, which then becomes the hardest thing on it to see. Each
    row is a meter against its own unit's patch instead, so the fills are directly comparable
    as shares and the absolute areas travel in the figure beside them, which is the meter-list
    contract: a name, a bar and a figure per row.
    """
    return """
    <div class="lane-axis" aria-hidden="true">
      <span></span>
      <span class="ax"><i class="lim" style="left:50%"></i>
        <b class="l0">none of it</b>
        <b class="ll c" style="left:50%">half</b>
        <b class="lm">the unit&rsquo;s whole patch</b></span>
      <span></span>
    </div>"""


def _lane_axis(scale, limit, label, unit="&Aring;&sup2;"):
    """One scale strip above the rows, so the bars below are read against a visible line.

    The scale used to live in the lane's footer, in words, under twenty rows of bars -- so a
    reader met the marks before the thing they were drawn against, and the threshold itself
    was a faint dashed rule with no number on it anywhere near the plot. This puts zero, the
    limit and the top of the scale over the first bar, on the same grid as every row, and the
    per-row rule becomes a gridline continuing a labelled line rather than a mark a reader has
    to decode.

    The limit's label is centred on it and pulled inside the strip at either end, because a
    label centred on a limit near zero or near the maximum hangs off the plot.
    """
    at = min(max(limit / scale * 100, 0), 100)
    # Centred on the limit only where a centred label has room on both sides. The fold lane
    # is the case: its ceiling sits at 400 of a 1,945-residue scale, 20.6 % along, and a label
    # centred there reached back over the `0` and printed on top of it. A third of the track
    # at either end is the band where the label has to hang off its rule rather than straddle.
    anchor = "l" if at < 33 else ("r" if at > 67 else "c")
    return f"""
    <div class="lane-axis" aria-hidden="true">
      <span></span>
      <span class="ax"><i class="lim" style="left:{at:.2f}%"></i>
        <b class="l0">0</b>
        <b class="ll {anchor}" style="left:{at:.2f}%">{limit:,.0f} {unit} {label}</b>
        <b class="lm">{scale:,.0f} {unit}</b></span>
      <span></span>
    </div>"""


def fold_lane(case):
    """Stage 3's rows: where each chain was found, how long it is, and the ceiling.

    The ceiling is the whole picture. Nine of the twenty chains on the SARS-CoV-2 holdout sit
    above it -- the three spike chains, the polymerase, the papain-like protease -- and a
    reader cannot be expected to take that on trust from a sentence when the record carries
    the length of every one of them. Drawn as one dashed line through twenty bars, the
    eleven short enough for the endpoint and the nine too long are a single glance.

    **What the nine are no longer.** This lane used to say of them that nothing could be
    predicted for them, and that was a true statement about the pipeline and a false one
    about the proteins. Every one of the nine carries an annotated functional region at or
    under the ceiling -- the polymerase's catalytic domain is 163 residues inside 823, and
    nsp3 is hopeless at 1,945 and offers eighteen regions that are not. So each over-ceiling
    row now opens onto what its relative's entry actually offers, read from
    `runs/domain_inventory.json`, and the group says the length is a bar to folding the chain
    whole rather than to reaching the target at all.

    Rows are sorted by length rather than by rank, because length is what this stage sorts on.
    """
    rows = list(case.get("foldable") or [])
    if not rows:
        return ""
    ceiling = _ceiling(case)
    # Keyed by target, and absent when no inventory has been recorded for this case. The lane
    # renders identically without it, minus the panels: a page built from a record that
    # predates the inventory must not fail, and must not imply it read one.
    inventory = domain_inventory(case) or {}
    regions = {c.get("target"): c for c in (inventory.get("chains") or ())}
    folds = domain_folds(case) or {}
    scale = max([r.get("recovered_length_aa") or 0 for r in rows] + [ceiling]) or 1
    recorded = any("folded" in r for r in rows)

    def plot(row):
        length = row.get("recovered_length_aa") or 0
        inside = bool(row.get("sequence_recovered") and 0 < length <= ceiling)
        bar = (_bar(length / scale * 100, cls="b" if inside else "b over")
               if length else "")
        return f'<span class="plot">{_tick(ceiling, scale)}{bar}</span>'

    def where(row):
        """How the chain was located, which is the only claim this stage makes about it."""
        if not row.get("sequence_recovered"):
            return esc(row.get("not_recovered_because") or "not located in this genome")
        if row.get("found_in"):
            covered = row.get("chain_covered_by_alignment")
            identity = row.get("identity_over_aligned_columns")
            detail = (f" &middot; {covered * 100:.0f} % of the relative&rsquo;s chain at "
                      f"{identity * 100:.0f} % identity"
                      if covered is not None and identity is not None else "")
            return f'found in {esc(row["found_in"])}{detail}'
        return esc(row.get("recovered_by") or "")

    def row_html(row):
        length = row.get("recovered_length_aa")
        plddt = row.get("mean_plddt")
        confidence = (f'<span class="cf">confidence {plddt:.1f}</span>'
                      if isinstance(plddt, (int, float)) else "")
        unnamed = ('<span class="chip-x nv">unrecognised</span>'
                   if row.get("identified") is False else "")
        # Two different panels, decided by which side of the ceiling the chain falls on. A
        # chain over it opens onto the regions inside it; a chain under it opens onto its own
        # structure, which the recorded run folded and then dropped for every chain but one.
        target = row.get("target")
        length = row.get("recovered_length_aa") or 0
        if length and length > ceiling:
            panel = domain_block(regions.get(target) or {}, folds.get(target))
        elif length:
            panel = chain_fold_block(folds.get(target))
        else:
            panel = ""
        return f"""
    <div class="lr">
      <span class="who"><span class="t">{esc(row.get('target'))}{unnamed}</span>
        <span class="w">{where(row)}</span></span>
      {plot(row)}
      <span class="vl">{num(length) if length else '&mdash;'}{' aa' if length else ''}
        {confidence}</span>
    </div>{panel}"""

    # Three groups, and the middle one is the stage's answer to the case the project exists
    # for: a chain nothing recognises has no accession, no relative and no annotation, and is
    # folded anyway, because this is the one stage that needs none of those.
    inside = [r for r in rows
              if r.get("sequence_recovered") and r.get("identified") is not False
              and 0 < (r.get("recovered_length_aa") or 0) <= ceiling]
    novel = [r for r in rows if r.get("identified") is False]
    over = [r for r in rows
            if r.get("identified") is not False and r not in inside and r not in novel]
    groups = [
        (inside, f"Folded &middot; {len(inside)}" if recorded
         else f"Inside the ceiling &middot; {len(inside)}",
         f"at or under {ceiling} residues, so the folding endpoint will take them. Every one "
         f"is folded, not only the best-ranked, because a chain with no coordinates is a "
         f"chain the next stage can only measure if somebody has already solved it"),
        (novel, f"Unrecognised, folded anyway &middot; {len(novel)}",
         "no accession, no relative, no annotation. Folding asks for a sequence and nothing "
         "else, so it is the one stage a protein nobody can name still reaches"),
        (over, f"Too long to fold whole &middot; {len(over)}",
         f"longer than the {ceiling}-residue ceiling, or never located in this genome at "
         f"all. The length is a bar to folding the chain whole and not to reaching the "
         f"target: open a row to see the functionally important regions its relative&rsquo;s "
         f"entry annotates, and which of them are short enough. "
         + (f"{_rescued(inventory)} of these have one"
            if inventory else
            "no region inventory has been recorded for this run")),
    ]
    # The same scale strip the other two lanes carry, directly above the first row of bars.
    body, axis = [], _lane_axis(scale, ceiling, "the folding ceiling", unit="aa")
    for members, title, note in groups:
        if not members:
            continue
        body.append(f'<div class="lane-grp"><b>{title}</b><span>{note}</span></div>')
        body.append(axis)
        axis = ""
        body.extend(row_html(r) for r in sorted(
            members, key=lambda r: -(r.get("recovered_length_aa") or 0)))
    return f"""
  <div class="lane">{"".join(body)}
    <div class="lane-foot"><span>bar length is the chain&rsquo;s own residue count, from
      zero, on one scale &middot; longest {num(scale)} aa</span>
      <span>the upright line is the {num(ceiling)}-residue folding ceiling</span></div>
  </div>"""


def surface_lane(case):
    """Stage 4's rows: the patch each chain got, by which route, against the gate's threshold.

    One scale in square angstroms for every bar on the page, deposited and predicted
    together, because the comparison is the point: a predicted patch read beside a deposited
    one at the same scale is the only place this pipeline says how far the two sit apart.

    The footprint line is what makes the verdict column legible. Several of these patches
    clear 600 square angstroms comfortably and are still refused, and a reader looking at
    bars and chips side by side can see that the refusals are not about area -- the gate reads
    localisation first, and the footer counts how many it decides.
    """
    ceiling, rows = _lane_rows(case)
    if not rows:
        return ""
    recorded = _predicted_is_recorded(case)
    footprint = MACROCYCLE_FOOTPRINT_A2
    areas = [r["surface"].get("patch_range_a2") or [0, 0] for r in rows if r["measured"]]
    scale = max([a[1] or 0 for a in areas]
                + [(r["predicted"] or {}).get("patch_a2") or 0 for r in rows]
                + [footprint]) or 1

    def chip(row):
        """The gate's verdict, with the question it answers attached to it.

        Unlabelled, this column reads as a judgement on the target and is not one. It answers
        "could a macrocycle bind here at all", which on a viral replicase is usually settled
        by where the protein sits rather than by how large its patch is -- so a row can carry
        857 square angstroms against a 600 threshold and still say no, and a reader with no
        label has been shown a contradiction.
        """
        gate = row["surface"].get("gate")
        # A refusal gets no chip. On this genome the gate says no to every measured chain,
        # and a column of nine identical greyed words carried no information while reading
        # as a verdict on the targets -- the one thing it is not. The refusal itself is not
        # dropped: the row says in words what put the chain out of reach, and the sentence
        # under the lane counts them. What is dropped is the repetition.
        if not gate or gate == "no":
            return ""
        asks = {"yes": "a macrocycle could bind here",
                "borderline": "uncertain: the measurement does not decide"}.get(gate, gate)
        return (f'<span class="chip-x g-{esc(gate)}" title="Can a macrocycle bind here? '
                f'{esc(asks)}">{esc(gate)}</span>')

    def verdict_note(row):
        """Why this row's verdict reads the way it does, where localisation decided it.

        The gate reads three conditions in order and stops at the first that fails, so a
        refusal on the first of them never reaches the other two. That order is invisible in
        a one-word chip, and it is the whole explanation for every refusal in this run: the
        patch is irrelevant to a protein a macrocycle cannot get to.
        """
        surface = row["surface"]
        if surface.get("exposed") is not False:
            return ""
        seen, places = set(), []
        for place in (surface.get("locations") or []):
            if place.lower() not in seen:
                seen.add(place.lower())
                places.append(place.lower())
        where = ", ".join(places[:3]) or "inside the cell"
        # Named when the localisation is not the chain's own. A polyprotein's entry carries
        # the locations of every chain cut out of it, so a verdict read off the parent is a
        # weaker claim than one read off the chain, and the record says which this was.
        scope = surface.get("location_scope")
        borrowed = ("" if scope == "chain" else
                    " &middot; off the parent entry, not this chain")
        # The clause this used to end with -- "which puts it out of reach, so a macrocycle is
        # refused here whatever the patch measures" -- was identical on all nine measured rows
        # and three lines long in the mono face, so the sentence a reader had already read
        # eight times was the largest thing in the row and the chart was a strip beside it.
        # It is said once, in the group note and in the paragraph under the lane, and what
        # stays here is the only part that differs per chain: where this protein sits.
        #
        # "Out of reach" rather than "inside the cell", which is the commoner case but not
        # the only one: ORF9b is annotated `Virion`, which is inside the particle, and
        # UniProt distinguishes that from `Virion membrane` precisely because one is a
        # surface a molecule can arrive at and the other is not.
        return (f'<br><span class="oor">out of reach</span> annotated {esc(where)}'
                f'{borrowed}')

    def measured_row(row):
        surface = row["surface"]
        low, high = surface.get("patch_range_a2") or [0, 0]
        median = surface.get("patch_median_a2") or 0
        structures = surface.get("structures_measured")
        fraction = surface.get("patch_fraction_of_surface")
        share = (f" &middot; {fraction * 100:.1f} % of its surface"
                 if fraction is not None else "")
        # The spread under the value, and only where there is one. `522-522` on the single
        # structure of nsp2 reads as a measured agreement between two numbers when it is one
        # number printed twice.
        # How many structures is already the first thing the left column says, so repeating
        # it here cost the range its room: `287-1,186 across 5` overran a column sized for
        # `1,186 A2` and was clipped mid-word.
        spread = (f'<span class="cf">{low:,.0f}&ndash;{high:,.0f}</span>' if high > low else
                  '<span class="cf">one structure</span>')
        return f"""
    <div class="lr">
      <span class="who"><span class="t">{esc(row['target'])}{chip(row)}</span>
        <span class="w">{num(structures)} deposited structure{'' if structures == 1 else 's'}
          of the same domain{share}{verdict_note(row)}</span></span>
      <span class="plot">{_tick(footprint, scale)}
        {_bar(median / scale * 100)}{_whisker(low, high, scale)}</span>
      <span class="vl">{median:,.0f} Å²{spread}</span>
    </div>"""

    def predicted_row(row):
        predicted = row["predicted"]
        patch = predicted.get("patch_a2") or 0
        deposited = predicted.get("deposited_patch_median_a2")
        ratio = predicted.get("ratio_to_deposited_median")
        plddt = predicted.get("mean_plddt")
        note = ("the only surface this chain has" if predicted.get("only_route_to_a_surface")
                else f"a second surface beside its deposited {num(deposited)} Å²"
                     + (f", {ratio}&times; larger" if ratio else ""))
        confidence = (f" &middot; fold confidence {plddt:.1f}"
                      if isinstance(plddt, (int, float)) else "")
        return f"""
    <div class="lr">
      <span class="who"><span class="t">{esc(row['target'])}</span>
        <span class="w">{note}{confidence}</span></span>
      <span class="plot">{_tick(footprint, scale)}
        {_bar(patch / scale * 100, cls="b pred")}</span>
      <span class="vl">{patch:,.0f} Å²<span class="cf">predicted</span></span>
    </div>"""

    def bare_row(row):
        reason = row["surface"].get("unmeasured_because") or ""
        length = row["length"]
        # Separated, because the recorded sentence does not always end in a full stop and
        # "none covers 1-180 180 aa" is two numbers reading as one.
        tail = (f' &middot; <i>{num(length)} aa, inside the {num(ceiling)}-residue ceiling</i>'
                if row["inside"] else
                f' &middot; <i>{num(length)} aa, over the ceiling</i>' if length else "")
        return f"""
    <div class="lr none">
      <span class="who"><span class="t">{esc(row['target'])}{chip(row)}</span>
        <span class="w">{esc(reason)}{tail}</span></span>
      <span class="plot"></span>
      <span class="vl">&mdash;</span>
    </div>"""

    measured = [r for r in rows if r["measured"]]
    predicted = [r for r in rows if r["predicted"] and not r["measured"]]
    # Why the measured group reads the way it does, derived here because the group's note
    # asserts it. On the held-out SARS-CoV-2 record every measured chain is refused on where
    # the protein sits, which is worth saying once at the top of the group instead of in a
    # sentence repeated down every row -- but it is a fact about this record, and the same
    # lane is drawn for four other genomes. A note that stated it unconditionally would be a
    # claim about their gates that nothing had checked.
    turned_away = [r for r in measured if r["surface"].get("gate") == "no"]
    by_location = [r for r in turned_away if r["surface"].get("exposed") is False]
    all_on_location = bool(measured) and len(by_location) == len(measured)
    # The rescue set, which is the half of the reorder a record can only state once it has
    # run under it. Where the measurements exist they are shown; where they do not, these are
    # the chains that get them on the next run, named rather than counted, and the group says
    # which of the two it is.
    waiting = [r for r in rows if not r["measured"] and not r["predicted"] and r["inside"]]
    none = [r for r in rows if not r["measured"] and not r["predicted"] and not r["inside"]]

    groups = [
        (measured, measured_row,
         f"Measured on a structure someone solved &middot; {len(measured)}",
         "the coordinates came out of a laboratory, so the patch is measured on an "
         "observation of the protein. "
         + ("Every one of these is refused anyway, on where the protein sits rather than on "
            "its patch &mdash; the row names the place"
            if all_on_location else
            "A chip beside a name is the gate&rsquo;s separate verdict on whether a "
            "macrocycle could bind there at all, and it appears only where that answer is "
            "yes or uncertain. Where it is no, the row says what put the chain out of "
            "reach")),
        (predicted, predicted_row,
         f"Measured on a shape a model predicted &middot; {len(predicted)}",
         "nobody has ever solved these, so the shape being measured was predicted from the "
         "sequence by stage 3. The arithmetic is identical and the evidence is not, so this "
         "group stays in its own column, outside the count above and outside the ranking"),
        (waiting, bare_row,
         f"A shape could be predicted for these, and this run did not &middot; {len(waiting)}",
         f"nobody has solved these either, but each is short enough for stage 3 to fold, so "
         f"coordinates can exist for them and the current pipeline measures those. They are "
         f"blank here for a dull reason. This record was written while the surface stage "
         f"still ran before the fold, so when it asked for coordinates there were none yet. "
         f"Re-running the chain fills this group in"),
        (none, bare_row,
         f"No surface by either route &middot; {len(none)}",
         f"nothing is deposited for these and they are over the {num(ceiling)}-residue "
         f"ceiling, so this run folded none of them and has no surface to offer. It says so, "
         f"and the chain keeps its row. What it is not is a statement that none of them can "
         f"be reached: stage 3&rsquo;s lane above lists the functionally important regions "
         f"their relatives annotate, and they are short enough"),
    ]
    # The scale strip goes immediately above the first row of bars, not at the top of the
    # lane: a group's note runs to five or six lines, and an axis separated from the marks it
    # labels by that much stops being read as their axis at all.
    body, axis = [], _lane_axis(scale, footprint, "a macrocycle&rsquo;s footprint")
    for members, render, title, note in groups:
        if not members:
            continue
        body.append(f'<div class="lane-grp"><b>{title}</b><span>{note}</span></div>')
        body.append(axis)
        axis = ""
        order = (sorted(members, key=lambda r: -(r["surface"].get("patch_median_a2") or 0))
                 if render is measured_row else
                 sorted(members, key=lambda r: r.get("rank") or 99))
        body.extend(render(r) for r in order)

    # Why the verdicts read the way they do, counted rather than asserted. Every refusal in
    # this record is decided before area is read, and a column of "no" beside bars that clear
    # the threshold is unreadable without that sentence.
    refused, on_location = turned_away, by_location
    verdicts = ""
    if refused:
        verdicts = (f"All {len(refused)} of those verdicts are decided on where the protein "
                    f"sits, which the gate reads before it reads area. Every one of these is "
                    f"annotated somewhere a macrocycle cannot reach, either inside the cell "
                    f"or inside the virus particle. Several of their patches clear the "
                    f"footprint line comfortably and are refused anyway, which makes the gate "
                    f"a statement about this class of molecule rather than about the "
                    f"target."
                    if len(on_location) == len(refused) else
                    f"{len(on_location)} of {len(refused)} refusals are decided on where "
                    f"the protein sits. The rest fail on area.")
    return f"""
  <div class="lane">
    {"".join(body)}
    <div class="lane-foot"><span>every bar starts at zero, on one scale, deposited and
      predicted together</span>
      <span>the line across a bar is how far that domain&rsquo;s structures
      disagree</span></div>
  </div>
  {f'<p class="after">{verdicts}</p>' if verdicts else ''}"""


# The patch at the functional site rather than at the chain's best address. Written by
# `python -m hpc_pilots.site_patches`.
SITE_PATCH_RECORD = "site_patches.json"


def site_patch_record(case):
    """The site-patch record for this case, or None.

    Guarded on the same two keys the inventory, the folds and the literature are, and for the
    same reason: these rows are keyed by target and unit label, so a record of a different
    genome -- or of the same genome identified through its own curated entries rather than
    held out of them -- would join cleanly and silently.

    A refused record is returned rather than treated as absent. `site_patches` writes one when
    the machine has no solvent-accessibility backend, with `status` `refused`, an empty `rows`
    and the reason under `answer`, precisely so a consumer needs no second code path; dropping
    it here would make the page's behaviour depend on which machine built it, which is the
    failure that record shape exists to avoid.
    """
    path = RUNS / SITE_PATCH_RECORD
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if (payload.get("genome_accession") != case.get("genome_accession")
            or payload.get("reference_excluded_organism")
            != case.get("reference_excluded_organism")):
        return None
    return payload


def _publishable(row):
    """Whether this row's figures may be printed as a measurement of the annotated site.

    **This is the one gate on this section and it is not the `measured` flag.** `measured`
    says anchors were found in the coordinates and a neighbourhood was measured around them.
    It does not say those anchors are the residues the annotation names, which is a separate
    question the record answers separately: `geometry` matches an anchor by number and reports
    back whatever residue stands there, so a window offset against the numbering the
    annotation arrived in produces a complete, confident, entirely wrong measurement. The
    deposited lane has the worked case -- nsp2 on `7fac:A`, a C-terminal construct where all
    12 anchors land 111 residues from the residues they name -- and a page reading `measured`
    alone would have published that as nsp2's active-site patch with nothing on the row to
    say otherwise.

    So a False verdict sends the row to the refusal group, which prints the record's own
    sentence and no figures. None is not False and is not treated as one: it means the
    comparison could not be made, the figures stand, and the row is marked. On the record as
    it stands all 23 measured units come back True, so this gate changes nothing visible
    today -- which is the point at which it is worth installing, rather than after a run where
    it would have been needed.
    """
    return (bool(row.get("measured"))
            and row.get("measured_at_the_annotated_residues") is not False)


def _frame_note(row):
    """Which reading frame this row's span is a span in, named rather than implied.

    `span_in_this_genome` is 1-based along one open reading frame and not along the genome, so
    the pair of integers means nothing without the frame -- and two rows in different frames
    can carry spans that overlap as integers while sharing no residue. The record counts those
    pairs over its own rows, so the page prints the frame on every row it draws and never
    sorts, groups or joins on the span.
    """
    frame = row.get("reading_frame") or {}
    span = row.get("span_in_this_genome") or []
    where = f"{span[0]:,}&ndash;{span[1]:,}" if len(span) == 2 else ""
    label = frame.get("reading_frame")
    if where and label:
        return f"{where} of reading frame {esc(label)}"
    return where or (f"reading frame {esc(label)}" if label else "")


def site_patch_block(case):
    """The patch at the annotated site, beside the patch stage 4 scored the unit on.

    Stage 4 reports the largest hydrophobic patch anywhere on a structure, which answers "is
    this protein amenable to a macrocycle at all". It does not answer where on the protein that
    surface is, and the two answers routinely disagree: the main protease's largest patch is
    2,012.6 square angstroms and 185.6 of it is at the catalytic dyad, so 91 % of the surface
    the chain was scored on is somewhere other than the chemistry. Both bars are therefore
    drawn on one scale per row -- the faint one is the unit's largest patch and the solid one
    is the part of it at the annotated residues -- because the gap between them is the finding
    and a single number cannot carry it.

    Three things gate a row before any figure is printed, and each has its own group rather
    than a footnote. `_publishable` decides whether the figures describe the annotated site at
    all. `anchors_are_processing_sites_only` says the annotated residues here are proteolytic
    cleavage boundaries, which sit at a mature chain's terminus by construction, so the
    neighbourhood is a neighbourhood of a dangling end -- eight of this run's units are in that
    state and publishing them beside a catalytic site would be the same error one level down.
    And `anchor_confidence` is printed on every row, because a site measured on residues the
    model placed at pLDDT 30 is an arithmetic on coordinates the model never committed to.

    Nothing here is recomputed. Every figure is read off `runs/site_patches.json`, including
    `site_share_of_the_largest_patch`, which the record derives; the page's job is to show the
    comparison, and a second computation of it would be a second opinion that could drift.
    """
    payload = site_patch_record(case)
    if not payload:
        return ""
    if payload.get("status") != "complete":
        return f"""
  <div class="lane">
    <div class="lane-grp"><b>Not measured on the machine that built this page</b>
      <span>{esc(payload.get('answer') or 'no reason recorded')}</span></div>
  </div>"""
    rows = payload.get("rows") or []
    if not rows:
        return ""

    def confidence(block):
        """The anchors' own confidence, which decides whether the rest of the row means much."""
        block = block or {}
        mean = block.get("mean_plddt")
        below = block.get("below_the_confident_threshold") or 0
        seen = block.get("anchors_with_a_confidence") or 0
        threshold = block.get("confident_plddt")
        if mean is None:
            return ""
        # The threshold is the folding tool's own, read off the fold store per unit rather
        # than restated, so it can be absent on a store that never wrote one. Named only
        # where it is known: "under the model's own None" is worse than no number.
        line = (f'{below} of {seen} under the model&rsquo;s own {threshold:.0f}'
                if isinstance(threshold, (int, float))
                else f'{below} of {seen} under the threshold')
        warn = f", {line}" if below else ", none under the threshold"
        return (f'<span class="cf{" lowc" if below else ""}">confidence '
                f'{mean:.1f}{warn}</span>')

    def chemistry_line(row):
        """The same measurement over the residues that do chemistry, where it differs.

        Shown only where the subset is not the whole set. Where it is, the record leaves the
        block's figures null on purpose -- the row's own are the answer -- and printing a copy
        would put two numbers on the page that a later edit could separate.
        """
        block = row.get("chemistry_only") or {}
        if not block.get("measured") or block.get("same_as_the_full_set"):
            if block.get("not_measured_because"):
                return (f'<br><i>chemistry residues only:</i> '
                        f'{esc(block["not_measured_because"])}')
            return ""
        patch = (block.get("site_patch") or {}).get("site_patch_area_a2")
        if patch is None:
            return ""
        reach = (block.get("reach") or {}).get("residues")
        dropped = block.get("processing_sites_dropped") or 0
        plural = "" if dropped == 1 else "s"
        return (f'<br>over the chemistry residues alone, {dropped} cleavage '
                f'site{plural} dropped: {num(reach)} residues within reach, patch '
                f'{patch:,.1f} Å² {confidence(block.get("anchor_confidence"))}')

    def whose(row):
        """The chain this unit was cut out of, or nothing when the unit IS the chain.

        A whole-chain unit is labelled with its chain's own name, and a chain-level refusal
        row carries the name in both fields, so printing both unconditionally gives
        "Protein 3 &middot; Protein 3".
        """
        label, target = row.get("label"), row.get("target")
        return "" if label == target else esc(target)

    def measured_row(row):
        patch = row.get("site_patch") or {}
        unit = row.get("unit_patch") or {}
        reach = row.get("reach") or {}
        area = patch.get("site_patch_area_a2") or 0.0
        whole = unit.get("hydrophobic_patch_area") or 0.0
        share = patch.get("site_share_of_the_largest_patch")
        positions = row.get("positions_in_the_fold") or []
        shown = ", ".join(str(p) for p in positions[:8])
        if len(positions) > 8:
            shown += f" and {len(positions) - 8} more"
        # None, not False: the record distinguishes "the anchors are not the annotated
        # residues" from "nobody could compare them", and only the first is a refusal.
        unchecked = ('<br><span class="warnline">the residue at one or more anchors could not '
                     'be compared against the annotation, so these figures are not confirmed '
                     'to be the annotated site&rsquo;s</span>'
                     if row.get("measured_at_the_annotated_residues") is None else "")
        # The share as the record derived it, never recomputed from the two areas beside it.
        # Dividing them here would put a second arithmetic on the page that a rounding change
        # in either number could separate from the record's.
        quoted = (f"{share * 100:.0f} % of {whole:,.0f}" if share is not None
                  else "of the unit&rsquo;s patch")
        return f"""
    <div class="lr">
      <span class="who"><span class="t">{esc(row.get('label'))}</span>
        <span class="w">{" &middot; ".join(p for p in (whose(row), _frame_note(row)) if p)}<br>
          fold positions {esc(shown)} &middot; {num(reach.get('residues'))} residues within
          reach, {num(reach.get('hydrophobic'))} of them water-repelling
          {confidence(row.get('anchor_confidence'))}{chemistry_line(row)}{unchecked}</span></span>
      <span class="plot meter"><i class="half"></i>
        {_bar((share or 0) * 100, cls="b site")}</span>
      <span class="vl">{area:,.0f} Å²<span class="cf">{quoted}</span></span>
    </div>"""

    def bare_row(row):
        reason = (row.get("not_the_annotated_residues_because")
                  or row.get("not_measured_because") or "")
        head = " &middot; ".join(p for p in (whose(row), _frame_note(row)) if p)
        return f"""
    <div class="lr none">
      <span class="who"><span class="t">{esc(row.get('label'))}</span>
        <span class="w">{head}{'<br>' if head else ''}
          {esc(reason)}</span></span>
      <span class="plot meter"></span>
      <span class="vl">&mdash;</span>
    </div>"""

    # Four groups partitioning every row, so the table's length is the record's and not a
    # consequence of which rows passed which gate. Membership is tested by identity and not by
    # equality: these are deep nested dicts and `in` on a list of them would compare them
    # field by field, 37 times over, to answer a question about which object this is.
    wrong = [r for r in rows if r.get("measured")
             and r.get("measured_at_the_annotated_residues") is False]
    publishable = [r for r in rows if _publishable(r)]
    processing = [r for r in publishable if r.get("anchors_are_processing_sites_only")]
    only = {id(r) for r in processing}
    chemistry = [r for r in publishable if id(r) not in only]
    siteless = [r for r in rows if not r.get("measured")]

    summary = payload.get("summary") or {}
    groups = [
        (chemistry, measured_row,
         f"Measured at the residues the curators marked &middot; {len(chemistry)}",
         "each bar is one unit&rsquo;s own largest water-repelling patch, drawn full width "
         "whatever its area; the fill is how much of it a macrocycle on the marked residues "
         "could reach, and the area itself is the figure on the right. An empty bar means "
         "the unit&rsquo;s LARGEST patch is elsewhere, never that the site has none"),
        (processing, measured_row,
         f"Measured, but only at cleavage boundaries &middot; {len(processing)}",
         "the only residues the curators marked inside these units are the points where a "
         "protease cuts the polyprotein. A cleavage site sits at the end of a mature chain by "
         "construction, so these figures describe a dangling terminus rather than a place to "
         "put a binder, and the confidence beside them is low for the same reason: a predicted "
         "chain end is disordered. They are shown rather than dropped because the alternative "
         "is a shorter table whose length nobody can account for"),
        (wrong, bare_row,
         f"Not the annotated residues &middot; {len(wrong)}",
         "the measurement completed and the residues it landed on are not the ones the "
         "annotation names, so the figures describe some other location and are withheld here "
         "rather than printed. The record keeps them, with this sentence beside them, because "
         "a number reported with what is wrong with it can be checked and a number silently "
         "dropped cannot"),
        (siteless, bare_row,
         f"No functional site to measure &middot; {len(siteless)}",
         "four different absences, and the row says which: the relative&rsquo;s entry "
         "annotates no site residue for this chain at all, or every one it annotates falls "
         "outside the window that was folded, or the position has no counterpart in this "
         "genome because the alignment found an indel there, or the unit never got "
         "coordinates. None of them is a statement that nothing happens on these proteins"),
    ]
    # The scale strip sits immediately above the first row of bars, for the reason
    # `surface_lane` gives: a group note runs to several lines, and an axis that far from the
    # marks it labels stops being read as their axis.
    body, axis = [], _share_axis()
    for members, render, title, note in groups:
        if not members:
            continue
        body.append(f'<div class="lane-grp"><b>{title}</b><span>{note}</span></div>')
        body.append(axis)
        axis = ""
        order = (sorted(members,
                        key=lambda r: -((r.get("site_patch") or {}).get(
                            "site_share_of_the_largest_patch") or 0))
                 if render is measured_row else
                 sorted(members, key=lambda r: (r.get("rank") or 99, r.get("label") or "")))
        body.extend(render(r) for r in order)

    frames = summary.get("reading_frames_the_rows_are_numbered_along")
    overlapping = summary.get("unit_pairs_overlapping_in_different_reading_frames")
    pairs = ("" if not overlapping else
             f" Each row names the reading frame its span is numbered along, because "
             f"{overlapping} pairs of these rows carry spans that overlap as numbers while "
             f"sharing no residue.")
    return f"""
  <div class="lane">
    {"".join(body)}
    <div class="lane-foot"><span>each bar is one unit&rsquo;s whole patch; the fill is the
      part of it the marked residues reach</span>
      <span>areas are in the figures, not the bars, because the units are not one
      size</span></div>
  </div>
  <p class="after">An annotated position is curated on the relative&rsquo;s entry, the
  residues here are numbered along one reading frame of this genome, and the coordinates are
  numbered from one, so a figure given at the active site is three mappings deep. Each row was
  checked against the residue the relative&rsquo;s entry carries at the annotated position:
  {num(summary.get('units_measured_at_the_annotated_residues'))} of
  {num(summary.get('units_measured'))} measured units land on it,
  {num(summary.get('units_not_measured_at_the_annotated_residues'))} do not and are withheld
  above.{pairs} The rows run along {num(frames)} frames in all.</p>"""


def _site_examples(payload):
    """The three worked examples the prose quotes, each picked by a stated rule.

    Picked by rule and not by name, for the reason `_richest_facts` gives: a sentence written
    around whichever row read well on the day it was written goes stale silently, and a reader
    cannot tell a chosen example from a representative one. The rules here are the claims the
    paragraph is making.

    `low` and `high` are the two ends of the share among units that carry a residue doing
    chemistry and whose largest patch reaches the site at all. Both of those are
    `site_shares_the_patch` true -- that key is the yes-or-no, and the pair is the measurement
    showing the yes is too coarse to act on.

    `drop` is the unit where dropping the proteolytic cleavage boundaries moves the share
    furthest. That is the claim about processing sites made as a number rather than as a
    caution: a cleavage boundary sits at a mature chain's terminus by construction, so a
    neighbourhood drawn around one is a neighbourhood of a dangling end, and this is the unit
    where including them costs the most.
    """
    def share(row, block=None):
        patch = ((block or row).get("site_patch") or {})
        return patch.get("site_share_of_the_largest_patch")

    rows = [r for r in payload.get("rows") or ()
            if r.get("measured") and not r.get("anchors_are_processing_sites_only")
            and (r.get("site_patch") or {}).get("site_shares_the_patch")]
    if not rows:
        return {}
    low = min(rows, key=lambda r: share(r) or 0)
    high = max(rows, key=lambda r: share(r) or 0)

    moved = [r for r in payload.get("rows") or ()
             if (r.get("chemistry_only") or {}).get("measured")
             and not r["chemistry_only"].get("same_as_the_full_set")
             and share(r) is not None and share(r, r["chemistry_only"]) is not None]
    def where(row):
        """`a domain on its chain`, or the chain alone where the two names are one.

        A whole-chain unit is labelled with its chain's own name, so naming both gives
        "Protein 3 on Protein 3". Built here rather than in the prose because `fill` escapes a
        token and cannot choose between two of them.
        """
        label, target = row.get("label"), row.get("target")
        return label if label == target else f"{label} on {target}"

    facts = {
        "site_low_unit": where(low),
        "site_low_share": f"{(share(low) or 0) * 100:.0f}",
        "site_high_unit": where(high),
        "site_high_share": f"{(share(high) or 0) * 100:.0f}",
    }
    if not moved:
        return facts
    drop = max(moved, key=lambda r: abs(share(r) - share(r, r["chemistry_only"])))
    block = drop["chemistry_only"]
    dropped = block.get("processing_sites_dropped") or 0
    return facts | {
        "site_drop_unit": drop.get("label"),
        "site_drop_whole": f"{(drop['unit_patch'] or {}).get('hydrophobic_patch_area') or 0:,.1f}",
        "site_drop_full": f"{drop['site_patch'].get('site_patch_area_a2') or 0:,.1f}",
        "site_drop_full_share": f"{share(drop) * 100:.0f}",
        "site_drop_chem": f"{block['site_patch'].get('site_patch_area_a2') or 0:,.1f}",
        "site_drop_chem_share": f"{share(drop, block) * 100:.0f}",
        "site_drop_dropped": dropped,
        "site_drop_plural": "" if dropped == 1 else "s",
    }


def site_patch_facts(case):
    """The counts the site-patch prose states, or empty where no record was written.

    Empty and not zero, for the reason `_domain_facts` states: `fill` refuses to ship a
    `{{token}}` with no value, so a page written against these cannot be built from a run that
    has no site-patch record -- which is the correct failure. Zeros would have the page assert
    that no unit has a measurable functional site, on the strength of a file that was never
    written.
    """
    payload = site_patch_record(case)
    if not payload or payload.get("status") != "complete":
        return {}
    summary = payload.get("summary") or {}
    return _site_examples(payload) | {
        "site_units": summary.get("units"),
        "site_measured": summary.get("units_measured"),
        "site_at_annotated": summary.get("units_measured_at_the_annotated_residues"),
        "site_patch_at_site": summary.get("units_whose_largest_patch_is_at_the_site"),
        "site_processing_only": summary.get("units_whose_anchors_are_processing_sites_only"),
        "site_anchors": summary.get("anchors_in_the_folds"),
        "site_anchors_low": summary.get("anchors_below_the_confident_threshold"),
        "site_frames": summary.get("reading_frames_the_rows_are_numbered_along"),
        "site_overlaps": summary.get("unit_pairs_overlapping_in_different_reading_frames"),
        "site_reach": f"{payload.get('reach_a') or 0:.1f}",
    }


def lane_facts(case):
    """The counts the two stages' prose states, joined from the rows the lanes draw.

    These were the last hand-typed numbers on the page, and they were the ones most likely to
    go stale: each is a join across two stages, so re-running the chain could change any of
    them without touching a sentence. `surface_predicted` and `surface_any` follow the
    record's own measurements where it has them and the ceiling where it does not, which is
    the same distinction the lane's group headings make.
    """
    ceiling, rows = _lane_rows(case)
    if not rows:
        return {}
    recorded = _predicted_is_recorded(case)
    located = [r for r in rows if r["located"]]
    inside = [r for r in rows if r["inside"]]
    measured = [r for r in rows if r["measured"]]
    rescued = [r for r in rows if not r["measured"]
               and (r["predicted"] if recorded else r["inside"])]
    return {
        "fold_ceiling": ceiling,
        # The gate's threshold, for the prose beside the lane that draws it. Imported with
        # the rest of this block rather than typed, for the reason the ceiling is.
        "footprint": f"{MACROCYCLE_FOOTPRINT_A2:,.0f}",
        "fold_located": len(located),
        "fold_inside": len(inside),
        "fold_over": len(located) - len(inside),
        "surface_predicted": len(rescued),
        "surface_any": len(measured) + len(rescued),
        "surface_none": len(rows) - len(measured) - len(rescued),
        # The no-dropped-rows invariant as a count rather than as a literal zero. The page
        # asserted "0 quietly dropped" in a hand-typed `<b>0</b>`, which is the one number on
        # a page about not dropping rows that could not report a dropped row.
        "surface_dropped": len(case.get("scores") or []) - len(rows),
    } | _domain_facts(case)


def _domain_facts(case):
    """The region-inventory counts the stage-3 prose quotes, or empty where none is recorded.

    Empty and not zero. `fill` refuses to ship a `{{token}}` with no value, so a page written
    against these tokens cannot be built from a record that has no inventory -- which is the
    correct failure. Returning zeros instead would let the page state that no chain has a
    foldable region, on the strength of a file that was never written.
    """
    inventory = domain_inventory(case)
    if not inventory:
        return {}
    summary = inventory.get("summary") or {}
    stranded = summary.get("chains_over_the_ceiling_with_none")
    return {
        "domain_over": summary.get("chains_over_the_ceiling"),
        "domain_rescued": summary.get("chains_over_the_ceiling_with_a_foldable_region"),
        "domain_stranded": stranded,
        "domain_regions_offered": summary.get("regions_offered"),
        "domain_regions_foldable": summary.get("regions_foldable"),
        "domain_siteless": summary.get("chains_with_no_annotated_site_residues"),
        "domain_minimum_aa": inventory.get("minimum_fold_unit_aa"),
        # The shortest foldable region on the whole shortlist, with the chain it sits in.
        # Quoted in the prose because one concrete pair makes the claim legible in a way a
        # count cannot: a 1,945-residue chain carrying a 91-residue region.
    } | _gap_facts(inventory) | _richest_facts(inventory)


def _richest_facts(inventory):
    """The over-ceiling chain whose relative annotates the most foldable regions.

    The prose wants one worked example, and this picks it by a stated rule rather than by
    whichever chain read well on the day the sentence was written. Most foldable regions is
    the right rule for the point being made -- that a chain far over the ceiling is not a
    chain out of reach -- because it is the chain where the gap between the two is widest.
    """
    best = None
    for chain in inventory.get("chains") or ():
        if chain.get("whole_chain_inside_the_ceiling") or not chain.get("regions_foldable"):
            continue
        if best is None or chain["regions_foldable"] > best["regions_foldable"]:
            best = chain
    if not best:
        return {}
    offered = best.get("best_region") or {}
    return {
        "domain_richest_chain": best.get("target"),
        # Pre-formatted, for the reason `_gap_facts` states: `fill` escapes and does not
        # format, so a four-figure residue count reaches the page without its separator.
        "domain_richest_whole": f"{best.get('length_aa') or 0:,}",
        "domain_richest_count": best.get("regions_foldable"),
        "domain_richest_best": offered.get("description") or offered.get("type"),
        "domain_richest_best_aa": f"{offered.get('length_aa') or 0:,}",
    }


def _gap_facts(inventory):
    """The widest gap between a chain's length and its shortest reachable region.

    On the RATIO and not on the shortest region outright, which was the first version and
    picked the wrong example. Measured on the held-out run: the globally shortest foldable
    region is a 40-residue heptad repeat inside a 458-residue spike fragment, an eleven-fold
    gap, and it beat the case the sentence is actually for -- nsp3 at 1,945 residues carrying
    a 56-residue `Ubiquitin-like 2` domain, thirty-five fold. The claim being made is about
    how far the two can diverge, so the quantity picked has to be the divergence rather than
    either number alone.

    Lengths are pre-formatted here rather than at the token, because `fill` escapes a value
    and does not format it, so an int arrives on the page as `1945`.
    """
    best = None
    for chain in inventory.get("chains") or ():
        whole = chain.get("length_aa") or 0
        if chain.get("whole_chain_inside_the_ceiling") or not whole:
            continue
        foldable = [r["length_aa"] for r in (chain.get("regions") or ())
                    if r.get("foldable") and r.get("length_aa")]
        if not foldable:
            continue
        shortest = min(foldable)
        ratio = whole / shortest
        if best is None or ratio > best[0]:
            best = (ratio, shortest, chain.get("target"), whole)
    if not best:
        return {}
    _, shortest, target, whole = best
    return {"domain_shortest": f"{shortest:,}",
            "domain_shortest_chain": target,
            "domain_shortest_whole": f"{whole:,}",
            "domain_shortest_fold": f"{whole / shortest:.0f}"}


PREDICTED_RECORD = "predicted_models.json"


def build_predicted():
    """What the endpoint label "AlphaFold DB" on the rows above is actually counting.

    The evidence section of a per-genome page prints `structures.predicted` as scalars, and
    `predicted_model_count 50` beside the endpoint "AlphaFold DB, 3D-Beacons" reads as "AlphaFold modelled this
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
            host = ("the AlphaFold host was asked and holds nothing for it, an absence "
                    "the tool measured")

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
        f'      <span class="plain">where the counts above came from</span>\n'
        f'      <span class="db">{num(total)} models · {num(authored)} from '
        f'AlphaFold</span></div>\n'
        f'    <div class="tc-body">\n'
        f'      <p class="cap" style="margin:0 0 18px">Every <span class="mono">'
        f'structures.predicted</span> row above is labelled <span class="mono">AlphaFold DB '
        f'· 3D-Beacons</span> and reports a bare <span class="mono">'
        f'predicted_model_count</span>, which reads as though AlphaFold modelled these '
        f'proteins. It did not. The tool reports the host that <i>served</i> each model '
        f'separately from the tool that <i>computed</i> it, and for these accessions the two '
        f'disagree. Of {num(total)} predicted models, {num(by_host)} were served by the '
        f'AlphaFold host and <b>{num(authored)} were computed by AlphaFold</b>. The models '
        f'it served are ColabFold v1.5.2, which runs the same weights behind a different '
        f'alignment pipeline. The chain record cannot carry this, because its summariser '
        f'reduces the model list to a count on purpose. The figures here are read from '
        f'<span class="mono">runs/predicted_models.json</span>.</p>\n'
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
        # Same number again, short enough for a stat tile sized for "29,903". The tile sits
        # under a caption promising every figure in that row comes from the run, so it is
        # read from the record like the rest rather than typed to match it.
        "diversity_tile": (f"{diversity / 1e9:.0f} B" if diversity else None),
        "seconds": round(case.get("seconds") or 0),
        # `seconds` is the whole case, and the case also exercises the five off-path tools on
        # a grid of their own -- 222 s of the 363 s on the SARS-CoV-2 record. The landing page
        # is a walkthrough of the five steps, so billing the reader for work the same page
        # calls "not on this path" overstates them by more than twice. Summed from the five
        # stages by tool, not by slicing the stage list, for the same reason the lookup above
        # is keyed that way: the pipeline's order has already changed once.
        "chain_seconds": round(sum(
            (s.get("seconds") or 0) for s in (st1, st2, surface, folded, library))),
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
    #
    # The wall clock is written in here rather than left as a {{seconds}} token for `fill`.
    # This block is rendered into two different pages and only one of them is filled: the
    # landing page substitutes tokens after every generated block is in place, while a result
    # page assembles its HTML and ships it, so the token survived to the published Zika page
    # as the literal text `{{seconds}}`. The value is the one `facts` would have supplied.
    held = case.get("reference_excluded_organism")
    whose = (f' Each accession below is the relative the chain identified that target '
             f'through, not {esc(held)}, which was deleted from the reference before the '
             f'run. Every annotation here is a record of that relative.') if held else ""
    return f"""
<section id="evidence">
  <div class="sec-head"><h2>Evidence on the top targets</h2>
    <span class="tag">{num(cov.get('evidence_calls'))} calls</span></div>
  <p class="cap">Five tools the chain itself never calls, run on the highest-ranked targets
  after the ranking was already final. Nothing refuses them. The cost gate admits all five,
  because it turns a tool away only when its declared worst case is greater than the budget,
  and the default budget is 600 seconds, so
  <span class="mono">bioactivity.prior_binders</span>, declaring exactly 600, is admitted. What
  keeps them off the path is the cost they actually run up, about 210 seconds for one
  target across the five, against a chain that finishes in {num(round(case.get('seconds') or 0))} seconds.
  Nothing they return changes a rank.{whose}</p>
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
  <p class="cap" style="margin-top:26px">That is the one chain the record reports in full. The
  lane below is every candidate this stage was handed, with how its own sequence was found in
  this genome and where it fell against the residue ceiling, which decides whether the stage
  after it has coordinates to measure.</p>
  {fold_lane(case)}
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
  <div class="sec-head"><h2>The shortlist</h2>
    <span class="tag">{_cut_summary(case)}</span></div>
  <p class="cap">A different list from the stage-1 candidates above, not a shorter one. Stage
  1 ranks whole ORFs. Stage 2 cuts the long proteins among them into the shorter chains a
  virus folds, which is what a drug is aimed at. Each row names the ORF it was cut out of, and
  carries its score split into what each of the five fixed weights paid it.</p>
  {chain_matrix(case, link_orfs=False)}
</section>

{fold_html if (fold_step or 99) < (surface_step or 0) else ""}
<section>
  <div class="sec-head"><h2>Surfaces</h2><span class="tag">one row per ranked chain</span></div>
  <div class="good"><p><b>{num(cov.get('surface_rows'))} answers for
  {num(cov.get('chains_ranked'))} chains.</b> A chain can be refused, but it cannot
  disappear.</p></div>
  {surface_block(case)}
  {surface_lane(case)}
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
                'first, and the form is on the front page.</p></div>')
    return (head(f"Runs · {SITE}", "Every pathogen genome this system has been run on.",
                 depth=1, current="runs") + f"""
<main class="wrap" id="main">
<header class="top">
  <div class="eyebrow">generated · one page per record</div>
  <h1>Every genome this has been run on</h1>
  <p class="standfirst">Each page below is generated from a chain record. Submit an
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


# Every fold on the landing page -- the committed one in step 3 and each fold that opens
# under a stage-3 row -- is drawn so a reader can follow one residue through it: point at a
# letter of the sequence and the backbone and the confidence trace both mark that residue,
# and pointing at either of those finds the letter.
#
# **One renderer, in the browser, from compact data.** Thirty-eight folds and nearly six
# thousand residues drawn as server-side SVG would add well over a megabyte to a page that
# most readers never open a row of. So Python does the part that must match the PNGs -- the
# PDB columns, the principal axes with their pinned signs, the confidence bands, all through
# `panels` -- and ships each fold as ~25 bytes a residue. The script draws a fold the first
# time its row is opened. With scripts off, each fold's PNG is shown instead, from the same
# `panels.structure_figure` geometry, so the two media are one drawing.
#
# **The sequence is read back out of the PDB, never retyped,** and is checked against the
# fold's own record (`sequence_sha256`) before it is printed. A fold whose file has drifted
# from its record shows its PNG and no sequence, rather than letters it was not run on.
_AA_NAME = {"A": "Ala", "R": "Arg", "N": "Asn", "D": "Asp", "C": "Cys", "Q": "Gln",
            "E": "Glu", "G": "Gly", "H": "His", "I": "Ile", "L": "Leu", "K": "Lys",
            "M": "Met", "F": "Phe", "P": "Pro", "S": "Ser", "T": "Thr", "W": "Trp",
            "Y": "Tyr", "V": "Val", "X": "unknown"}
# Class suffix per band, in `panels.PLDDT_BANDS` order: very high, confident, low, very low.
_BAND_KEYS = ("vh", "c", "l", "vl")
# Viewbox units to the angstrom. Every backbone is drawn at this scale and capped at that
# many CSS pixels wide, so a 45-residue coiled coil and a 369-residue domain sit on one
# physical scale and the 20 Å bar under each is the same length on screen.
_TRACE_K = 6.0

_FOLD_IMG = """<img src="fold.png" alt="Left: the folded backbone of the main protease, 306 alpha-carbons drawn on its two principal axes and coloured by per-residue confidence. One compact well-predicted domain in blue sits beside a sprawling orange one. Right: confidence against residue position, with the stretches that never clear the threshold marked along the bottom.">"""


def _panels():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-site")
    os.environ.setdefault("MPLBACKEND", "Agg")
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "notebooks"))
    import panels                                                      # noqa: E402
    return panels


def _trace_data(pdb, rec, name, offset=None):
    """One fold's per-residue data for the browser, or None if it cannot be trusted.

    `offset` is where residue 1 of the fold sits in this genome's own numbering, given only
    when the fold is a contiguous slice of known length; the readout then prints both.
    """
    import hashlib
    import numpy as np
    if not (pdb.exists() and rec.exists()):
        return None
    panels = _panels()
    payload = json.loads(rec.read_text())
    text = pdb.read_text()
    resnum, xyz, bfac = panels.ca_trace(text)
    seq = panels.folded_sequence(text)
    if (len(resnum) < 2 or len(seq) != len(resnum)
            or hashlib.sha256(seq.encode()).hexdigest() != payload.get("sequence_sha256")):
        print(f"  {pdb.name}: sequence does not match its record; drawing the PNG instead")
        return None
    plddt, _ = panels.plddt_on_0_100(bfac)

    def band(v):
        for key, (lo, hi, _, _) in zip(_BAND_KEYS, panels.PLDDT_BANDS):
            if lo <= v < hi:
                return key
        return _BAND_KEYS[0]

    # `panels.structure_figure`'s projection, line for line. SVG's y runs down, so the second
    # axis is negated to keep the PNG's orientation.
    centred = xyz - xyz.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    for i in range(2):
        if vt[i, i] < 0:
            vt[i] *= -1
    proj = centred @ vt.T
    r1 = lambda a: [round(float(v), 1) for v in a]
    return {
        "name": name, "seq": seq, "num": [int(n) for n in resnum],
        "at": offset, "pl": r1(plddt), "band": "".join(
            {"vh": "a", "c": "b", "l": "c", "vl": "d"}[band(v)] for v in plddt),
        "x": r1(proj[:, 0] * _TRACE_K), "y": r1(-proj[:, 1] * _TRACE_K), "z": r1(proj[:, 2]),
        "runs": [list(r) for r in payload.get("low_confidence_runs", [])],
        "thr": float(payload.get("confident_plddt") or 70.0),
        "legend": [[k, label] for k, (_, _, _, label) in zip(_BAND_KEYS, panels.PLDDT_BANDS)],
    }


def _trace_block(data, fallback):
    """The placeholder the script draws a fold into, with its PNG for a reader without it."""
    if not data:
        return fallback
    # `</` is escaped so a label can never close the script element it is carried in.
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    return f"""
    <div class="ftrace">
      <noscript>{fallback}</noscript>
      <script type="application/json" class="ft-data">{blob}</script>
    </div>"""


def fold_trace():
    """The committed fold in step 3, traced by residue."""
    data = _trace_data(RUNS / "folded_structure/mpro_nsp5.pdb",
                       RUNS / "folded_structure/mpro_nsp5.json", "Mpro (nsp5)")
    return _trace_block(data, _FOLD_IMG)


def unit_trace(unit, fallback):
    """One fold under a stage-3 row, traced by residue; `fallback` is its PNG."""
    slug = unit.get("slug")
    span = unit.get("span_in_this_genome") or []
    length = unit.get("length_aa")
    offset = (span[0] if len(span) == 2 and length and span[1] - span[0] + 1 == length
              else None)
    data = _trace_data(RUNS / FOLD_STORE / f"{slug}.pdb", RUNS / FOLD_STORE / f"{slug}.json",
                       unit.get("label") or slug, offset)
    if data and offset is not None and len(data["seq"]) != length:
        data["at"] = None
    return _trace_block(data, fallback)


# The renderer and the shared cursor, once per page. Each fold's whole state is one index
# `i` (0-based into its sequence); every view draws only `i`, so the three cannot disagree.
FOLD_TRACE_JS = """
<script>
(function () {
  var NAMES = __NAMES__;
  var BAND = { a: "vh", b: "c", c: "l", d: "vl" };
  var BANDNAME = { vh: "very high", c: "confident", l: "low", vl: "very low" };
  var K = __K__;
  function esc(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c];
    });
  }
  function f(v) { return Math.round(v * 10) / 10; }
  function min(a) { return Math.min.apply(null, a); }
  function max(a) { return Math.max.apply(null, a); }
  function comma(n) { return Number(n).toLocaleString("en-US"); }

  function draw(root) {
    var D = JSON.parse(root.querySelector(".ft-data").textContent);
    var n = D.seq.length, b = [], i, s;
    for (i = 0; i < n; i++) b.push(BAND[D.band[i]]);

    // Backbone: all halos under all colour, as the PNG draws them -- a halo laid between
    // segments would bite a notch out of the chain at every joint.
    // The scale bar gets its own strip under the fold rather than a corner of it: a long thin
    // fold, a coiled coil, leaves no empty corner, and the bar would sit on the chain. The
    // frame is also never narrower than the bar.
    var pad = 4 * K, x0 = min(D.x) - pad, x1 = max(D.x) + pad, y0 = min(D.y) - pad, y1 = max(D.y) + pad + 4 * K;
    var widen = Math.max(0, 26 * K - (x1 - x0)) / 2; x0 -= widen; x1 += widen;
    var sd = [], order = [];
    for (s = 0; s < n - 1; s++) { sd.push((D.z[s] + D.z[s + 1]) / 2); order.push(s); }
    order.sort(function (p, q) { return sd[p] - sd[q]; });
    var dmin = min(sd), dspan = Math.max(max(sd) - dmin, 1e-9);
    var halo = [], lines = [];
    order.forEach(function (s) {
      var seg = D.x[s] + " " + D.y[s] + "L" + D.x[s + 1] + " " + D.y[s + 1];
      halo.push("M" + seg);
      lines.push('<line data-k="' + s + '" class="b-' + b[s] + '" x1="' + D.x[s] + '" y1="' + D.y[s] +
        '" x2="' + D.x[s + 1] + '" y2="' + D.y[s + 1] + '" stroke-width="' +
        (2.2 + 2.6 * (sd[s] - dmin) / dspan).toFixed(2) + '"/>');
    });
    var ends = [[0, "N"], [n - 1, "C"]].map(function (e) {
      return '<circle class="ft-end" cx="' + D.x[e[0]] + '" cy="' + D.y[e[0]] + '" r="9"/>' +
        '<text class="ft-endl" x="' + D.x[e[0]] + '" y="' + D.y[e[0]] + '">' + e[1] + "</text>";
    }).join("");
    var by = f(y1 - 1.5 * K), vbw = f(x1 - x0);
    var bb = '<figure class="ft-panel"><figcaption>backbone, ' + n +
      ' C&alpha;, orthographic on its two principal axes</figcaption>' +
      '<svg class="ft-bb" style="max-width:' + vbw + 'px" viewBox="' + f(x0) + " " + f(y0) + " " + vbw + " " + f(y1 - y0) +
      '" role="img" aria-label="The folded backbone of ' + esc(D.name) + ", " + n +
      ' alpha-carbons coloured by per-residue confidence. Point at it to find a residue in the sequence below.">' +
      '<path class="ft-halo" d="' + halo.join(" ") + '"/><g class="ft-chain">' + lines.join("") + "</g>" + ends +
      '<line class="ft-scale" x1="' + f(x1 - 22 * K) + '" y1="' + by + '" x2="' + f(x1 - 2 * K) + '" y2="' + by + '"/>' +
      '<text class="ft-scalel" x="' + f(x1 - 12 * K) + '" y="' + f(by - 1.4 * K) + '">20 &Aring;</text>' +
      '<g class="ft-mark" visibility="hidden"><circle r="11"/><circle class="ft-dot" r="3.5"/>' +
      '<text class="ft-tag" x="14" y="-12"></text></g></svg></figure>';

    // Confidence against position, on the fold's own residue numbering.
    var W = 560, H = 300, L = 40, R = 10, T = 12, B = 34, n0 = D.num[0], n1 = D.num[n - 1];
    function sx(v) { return f(L + (v - n0) / Math.max(n1 - n0, 1) * (W - L - R)); }
    function sy(v) { return f(T + (1 - v / 100) * (H - T - B)); }
    var PX = D.num.map(sx), PY = D.pl.map(sy);
    var shades = [[90, 100, "vh"], [70, 90, "c"], [50, 70, "l"], [0, 50, "vl"]].map(function (r) {
      return '<rect class="s-' + r[2] + '" x="' + L + '" y="' + sy(r[1]) + '" width="' + (W - L - R) +
        '" height="' + f(sy(r[0]) - sy(r[1])) + '"/>';
    }).join("");
    var grid = [0, 25, 50, 75, 100].map(function (v) {
      return '<line class="ft-grid" x1="' + L + '" x2="' + (W - R) + '" y1="' + sy(v) + '" y2="' + sy(v) + '"/>' +
        '<text class="ft-ax" x="' + (L - 6) + '" y="' + f(sy(v) + 4) + '" text-anchor="end">' + v + "</text>";
    }).join("");
    // Round ticks at the coarsest step that still gives a few, and both ends unless a round
    // tick already sits within a label's width of one.
    var span = n1 - n0, step = [10, 20, 25, 50, 100, 200].filter(function (t) { return span / t <= 7; })[0] || 500;
    var room = span * 0.05, ticks = [n0];
    for (var t = Math.ceil((n0 + 1) / step) * step; t < n1; t += step)
      if (t - n0 > room && n1 - t > room) ticks.push(t);
    ticks.push(n1);
    var xt = ticks.map(function (v) {
      return '<text class="ft-ax" x="' + sx(v) + '" y="' + (H - B + 15) + '" text-anchor="middle">' + v + "</text>";
    }).join("");
    var trace = "M" + PX.map(function (x, j) { return x + " " + PY[j]; }).join("L");
    var dots = PX.map(function (x, j) {
      return '<circle class="b-' + b[j] + '" cx="' + x + '" cy="' + PY[j] + '" r="2.2"/>';
    }).join("");
    var mean = D.pl.reduce(function (a, v) { return a + v; }, 0) / n;
    var runs = D.runs.map(function (r) {
      return '<line class="ft-run" x1="' + sx(r[0]) + '" x2="' + sx(r[1]) + '" y1="' + sy(4) + '" y2="' + sy(4) + '"/>' +
        (r[1] - r[0] > span * 0.04 && r[1] - r[0] > 3 ? '<text class="ft-runl" x="' + f((sx(r[0]) + sx(r[1])) / 2) +
          '" y="' + sy(7.5) + '">' + r[0] + "&ndash;" + r[1] + "</text>" : "");
    }).join("");
    var pl = '<figure class="ft-panel"><figcaption>per-residue confidence, and the stretches that never clear the threshold</figcaption>' +
      '<svg class="ft-pl" viewBox="0 0 ' + W + " " + H + '" role="img" aria-label="Confidence against residue position for ' +
      esc(D.name) + ", mean " + mean.toFixed(1) + ' of 100. Point at it to find a residue in the sequence below.">' +
      shades + grid + xt +
      '<text class="ft-ax" x="' + f((L + W - R) / 2) + '" y="' + (H - 4) + '" text-anchor="middle">residue position in the folded sequence</text>' +
      '<path class="ft-trace" d="' + trace + '"/>' + dots +
      '<line class="ft-thr" x1="' + L + '" x2="' + (W - R) + '" y1="' + sy(D.thr) + '" y2="' + sy(D.thr) + '"/>' +
      '<text class="ft-ax ft-lab" x="' + (W - R - 3) + '" y="' + f(sy(D.thr) - 5) + '" text-anchor="end">threshold ' + D.thr.toFixed(0) + "</text>" +
      '<line class="ft-mean" x1="' + L + '" x2="' + (W - R) + '" y1="' + sy(mean) + '" y2="' + sy(mean) + '"/>' +
      '<text class="ft-ax ft-lab ft-meanl" x="' + (L + 6) + '" y="' + f(sy(mean) + 13) + '">mean ' + mean.toFixed(1) + "</text>" +
      runs + '<g class="ft-cur" visibility="hidden"><line y1="' + T + '" y2="' + (H - B) + '"/><circle r="4.5"/></g></svg></figure>';

    // The sequence, ten to a block, each block numbered by its first residue. The numbers are
    // CSS-generated and the gaps are margins, so selecting the strip copies the bare sequence.
    var blocks = [];
    for (s = 0; s < n; s += 10) {
      var letters = "";
      for (i = s; i < Math.min(s + 10, n); i++)
        letters += '<span class="b-' + b[i] + '" data-i="' + i + '">' + D.seq[i] + "</span>";
      blocks.push('<span class="ft-blk" data-n="' + D.num[s] + '">' + letters + "</span>");
    }
    var legend = D.legend.map(function (l) {
      return '<span><i class="b-' + l[0] + '"></i>' + esc(l[1]) + "</span>";
    }).join("");
    var idle = "Point at a residue here, on the backbone or on the trace; click to hold it, then step with the arrow keys.";
    root.innerHTML = '<div class="ft-figs">' + bb + pl + '</div><div class="ft-legend">' + legend + "</div>" +
      '<div class="ft-seqhead"><span class="ft-h">The sequence that was folded</span>' +
      '<span class="ft-read" aria-live="polite">' + idle + "</span>" +
      '<button class="ft-copy" type="button">Copy FASTA</button></div>' +
      '<div class="ft-seq" tabindex="0" aria-label="The folded sequence of ' + esc(D.name) + ", " + n +
      ' residues. With a residue held, the arrow keys step through it.">' + blocks.join("") + "</div>";
    wire(root, D, b, PX, PY, idle);
  }

  function wire(root, D, b, PX, PY, idle) {
    var bb = root.querySelector(".ft-bb"), pl = root.querySelector(".ft-pl");
    var mark = bb.querySelector(".ft-mark"), tag = bb.querySelector(".ft-tag");
    var cur = pl.querySelector(".ft-cur"), curDot = cur.querySelector("circle"), curLine = cur.querySelector("line");
    var seqBox = root.querySelector(".ft-seq"), read = root.querySelector(".ft-read");
    var letters = seqBox.querySelectorAll("[data-i]"), segs = [];
    bb.querySelectorAll(".ft-chain line").forEach(function (el) { segs[+el.dataset.k] = el; });
    var shown = -1, held = -1, lit = [];

    function runOf(v) {
      for (var r = 0; r < D.runs.length; r++) if (v >= D.runs[r][0] && v <= D.runs[r][1]) return D.runs[r];
      return null;
    }
    function show(i, force) {
      if (i === shown && !force) return;
      if (shown >= 0) letters[shown].classList.remove("on");
      lit.forEach(function (el) { el.classList.remove("on"); });
      lit = [];
      shown = i;
      root.classList.toggle("tracing", i >= 0);
      if (i < 0) {
        mark.setAttribute("visibility", "hidden");
        cur.setAttribute("visibility", "hidden");
        read.textContent = idle;
        return;
      }
      var v = D.num[i], aa = D.seq[i];
      mark.setAttribute("transform", "translate(" + D.x[i] + " " + D.y[i] + ")");
      tag.textContent = aa + v;
      mark.setAttribute("visibility", "visible");
      [i - 1, i].forEach(function (k) { if (segs[k]) { segs[k].classList.add("on"); lit.push(segs[k]); } });
      curLine.setAttribute("x1", PX[i]); curLine.setAttribute("x2", PX[i]);
      curDot.setAttribute("cx", PX[i]); curDot.setAttribute("cy", PY[i]);
      cur.setAttribute("visibility", "visible");
      letters[i].classList.add("on");
      var run = runOf(v);
      read.textContent = "Residue " + v +
        (D.at != null && D.at + i !== v ? " (" + comma(D.at + i) + " in this genome)" : "") +
        " \\u00b7 " + (NAMES[aa] || aa) + " (" + aa + ")" +
        " \\u00b7 confidence " + D.pl[i] + ", " + BANDNAME[b[i]] +
        (run ? " \\u00b7 inside " + run[0] + "\\u2013" + run[1] + ", which never clears " + D.thr : "") +
        (i === held ? " \\u00b7 held" : "");
    }
    function rest() { show(held); }
    function hold(i) { held = (i === held) ? -1 : i; show(held >= 0 ? held : i, true); }
    function toLocal(svg, e) {
      var p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
      var m = svg.getScreenCTM(); return m ? p.matrixTransform(m.inverse()) : null;
    }
    // Nearest alpha-carbon in the projection. Where two pass close in the picture, the one
    // nearer the reader wins, since that is the one drawn on top -- but only as a tie-break:
    // twenty angstroms of depth is worth five viewBox units, under one angstrom on screen.
    function nearestCA(e) {
      var q = toLocal(bb, e); if (!q) return -1;
      var best = -1, bestScore = Infinity, reach = 36 * 36;
      for (var i = 0; i < D.x.length; i++) {
        var dx = D.x[i] - q.x, dy = D.y[i] - q.y, d2 = dx * dx + dy * dy;
        if (d2 > reach) continue;
        var score = Math.sqrt(d2) - D.z[i] * 0.25;
        if (score < bestScore) { bestScore = score; best = i; }
      }
      return best;
    }
    function nearestPos(e) {
      var q = toLocal(pl, e); if (!q) return -1;
      var best = -1, bd = Infinity;
      for (var i = 0; i < PX.length; i++) { var d = Math.abs(PX[i] - q.x); if (d < bd) { bd = d; best = i; } }
      return bd < 12 ? best : -1;
    }
    function hook(el, pick) {
      el.addEventListener("pointermove", function (e) { var i = pick(e); show(i >= 0 ? i : held); });
      el.addEventListener("pointerleave", rest);
      el.addEventListener("click", function (e) { var i = pick(e); if (i >= 0) hold(i); });
    }
    hook(bb, nearestCA);
    hook(pl, nearestPos);
    function letterOf(e) { var t = e.target.closest && e.target.closest("[data-i]"); return t ? +t.dataset.i : -1; }
    seqBox.addEventListener("pointerover", function (e) { var i = letterOf(e); if (i >= 0) show(i); });
    seqBox.addEventListener("pointerleave", rest);
    seqBox.addEventListener("click", function (e) {
      var i = letterOf(e);
      // A drag that selected text is a copy, not a pick.
      var sel = window.getSelection && String(window.getSelection());
      if (i >= 0 && !sel) hold(i);
    });
    seqBox.addEventListener("keydown", function (e) {
      var step = { ArrowRight: 1, ArrowLeft: -1, ArrowDown: 10, ArrowUp: -10 }[e.key];
      if (e.key === "Home") step = -Infinity;
      if (e.key === "End") step = Infinity;
      if (step === undefined) return;
      e.preventDefault();
      var from = held >= 0 ? held : (step > 0 ? -1 : D.seq.length);
      held = Math.max(0, Math.min(D.seq.length - 1, from + step));
      show(held, true);
    });
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && held >= 0) { held = -1; show(-1); }
    });
    var copy = root.querySelector(".ft-copy");
    copy.addEventListener("click", function () {
      var text = ">" + D.name + " | ESMFold v1 input | " + D.seq.length + " aa\\n" +
        D.seq.match(/.{1,60}/g).join("\\n") + "\\n";
      function done(ok) {
        copy.textContent = ok ? "Copied" : "Copy failed";
        setTimeout(function () { copy.textContent = "Copy FASTA"; }, 1600);
      }
      try { navigator.clipboard.writeText(text).then(function () { done(true); }, function () { done(false); }); }
      catch (err) { done(false); }
    });
  }

  // A fold is drawn the first time it can be seen: at once if no closed row hides it, else
  // when the last closed row around it opens. Thirty-seven of them sit in closed rows.
  function visible(el) {
    for (var d = el.parentElement; d; d = d.parentElement)
      if (d.tagName === "DETAILS" && !d.open) return false;
    return true;
  }
  function sweep(scope) {
    scope.querySelectorAll(".ftrace:not(.drawn)").forEach(function (root) {
      if (!visible(root)) return;
      root.classList.add("drawn");
      try { draw(root); } catch (err) { root.classList.remove("drawn"); }
    });
  }
  document.addEventListener("toggle", function (e) {
    if (e.target.open) sweep(e.target);
  }, true);
  sweep(document);
})();
</script>
"""
FOLD_TRACE_JS = (FOLD_TRACE_JS.replace("__NAMES__", json.dumps(_AA_NAME))
                 .replace("__K__", repr(_TRACE_K)))


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
    # Order-independent: `build_index` writes `<img>` tags by path and reads no pixels, so
    # these are written beside the committed fold's figure rather than before the page.
    build_domain_fold_images()
    build_results()
    if args.notebook:
        build_notebook()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    print(f"site built {stamp}")


if __name__ == "__main__":
    main()
