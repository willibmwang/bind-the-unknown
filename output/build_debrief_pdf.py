"""Render a project Markdown document to a presentation-quality PDF.

The original PDF was produced by headless Chrome (Skia/PDF), and this reproduces that
pipeline deterministically and offline: Markdown to styled HTML here, then Chrome's
--print-to-pdf. No network, no webfonts and no Markdown dependency, so the output is stable.

The converter handles exactly the constructs the debrief uses -- ATX headings, pipe tables
with alignment, blockquotes, bullet and ordered lists, thematic breaks, fenced code, and
inline bold/italic/code/strikethrough. Each top-level section becomes a page-breaking
<section> so the document reads as a deck of sections rather than a continuous wall.

    python3 output/build_debrief_pdf.py [source.md]   # default: the Week 4 debrief
"""
import html
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# The running head, by source stem. Each document says which review it belongs to, so a
# printed page is identifiable once it has been separated from the others.
EYEBROWS = {
    "W4_DEBRIEF_AND_TARGET_SELECTION": "SynX Therapeutics · MATS 11.0 · Week 4 review",
    "PROJECT_OVERVIEW_AND_PLAN": "SynX Therapeutics · MATS 11.0 · Project overview and plan",
    "ARCHITECTURE": "SynX Therapeutics · MATS 11.0 · System architecture",
    "PROTOTYPE_AND_PRIORITY_PATHOGENS": "SynX Therapeutics · MATS 11.0 · Background note",
}
DEFAULT_EYEBROW = "SynX Therapeutics · MATS 11.0"

CSS = """
@page { size: Letter; margin: 19mm 18mm 20mm; }
:root {
  --ink: #16191d; --muted: #5a6472; --rule: #d4dae2; --accent: #1f4e79;
  --panel: #f4f7fa; --code-bg: #eef1f5;
}
* { box-sizing: border-box; }
html { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body {
  margin: 0; color: var(--ink);
  font: 10.5pt/1.62 "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  text-rendering: optimizeLegibility;
}
h1, h2, h3, h4, .eyebrow, th { font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; }

.title-block { padding-bottom: 14pt; border-bottom: 2.5pt solid var(--accent); margin-bottom: 20pt; }
.eyebrow {
  font-size: 7.8pt; letter-spacing: .14em; text-transform: uppercase;
  color: var(--accent); font-weight: 600; margin-bottom: 10pt;
}
h1 { font-size: 21pt; line-height: 1.22; margin: 0 0 11pt; font-weight: 600; letter-spacing: -.012em; }
.title-block p { margin: 0 0 7pt; font-size: 9.4pt; line-height: 1.55; color: var(--muted); }
.title-block p:last-child { margin-bottom: 0; }
.title-block strong { color: var(--ink); }

section { break-before: page; }
section:first-of-type { break-before: auto; }
h2 {
  font-size: 15pt; font-weight: 600; margin: 0 0 13pt; padding-bottom: 6pt;
  border-bottom: .8pt solid var(--rule); letter-spacing: -.008em; break-after: avoid;
}
h3 { font-size: 11.4pt; font-weight: 600; margin: 19pt 0 7pt; color: var(--accent); break-after: avoid; }
h4 { font-size: 10.2pt; font-weight: 600; margin: 15pt 0 5pt; break-after: avoid; }
p { margin: 0 0 9pt; orphans: 3; widows: 3; }
/* A sentence introducing a table or quote must not be stranded at the foot of a page
   while the thing it introduces starts the next one. */
p:has(+ table), p:has(+ blockquote), p:has(+ pre) { break-after: avoid; }
hr { border: 0; border-top: .8pt solid var(--rule); margin: 16pt 0; }

/* The summary reads as a briefing panel rather than as body copy. */
section.summary-for-review > h2 { border-bottom-color: var(--accent); }
section.summary-for-review > p {
  background: var(--panel); border-left: 2.5pt solid var(--accent);
  padding: 9pt 12pt; margin: 0 0 7pt; break-inside: avoid;
}

table { width: 100%; border-collapse: collapse; margin: 11pt 0 13pt; font-size: 9.1pt; break-inside: avoid; }
th {
  text-align: left; font-size: 7.9pt; font-weight: 600; letter-spacing: .055em;
  text-transform: uppercase; color: var(--accent);
  border-bottom: 1.1pt solid var(--accent); padding: 0 7pt 5pt; vertical-align: bottom;
}
td { padding: 5pt 7pt; border-bottom: .5pt solid var(--rule); vertical-align: top; line-height: 1.46; }
tbody tr:nth-child(even) { background: #f8fafc; }
td.r, th.r { text-align: right; }
td.c, th.c { text-align: center; }

blockquote {
  margin: 11pt 0; padding: 2pt 0 2pt 13pt;
  border-left: 2.5pt solid var(--accent); color: #2b3240; break-inside: avoid;
}
blockquote p { margin: 0 0 5pt; }
blockquote p:last-child { margin-bottom: 0; }
ul, ol { margin: 0 0 10pt; padding-left: 20pt; }
li { margin-bottom: 4.5pt; line-height: 1.55; }
li:last-child { margin-bottom: 0; }
code {
  font-family: "SF Mono", Menlo, Consolas, monospace; font-size: .855em;
  background: var(--code-bg); padding: .1em .32em; border-radius: 2.5px;
}
pre {
  background: var(--code-bg); border-left: 2.5pt solid var(--rule);
  padding: 9pt 11pt; margin: 11pt 0; white-space: pre-wrap; overflow-wrap: break-word;
  font-size: 8.6pt; line-height: 1.5; break-inside: avoid;
}
pre code { background: none; padding: 0; font-size: 1em; }
del { color: var(--muted); }
strong { font-weight: 600; }
a { color: var(--accent); text-decoration: none; border-bottom: .5pt solid #a8bdd2; }
/* A reference to a companion document in the repository: named, but not pretending to be
   a working link for someone reading the printed page. */
.xref { font-style: italic; color: var(--muted); }
"""

BULLET = re.compile(r"^(\s*)[-*] (.*)$")
NUMBER = re.compile(r"^(\s*)\d+\. (.*)$")
RULE = re.compile(r"^-{3,}\s*$")
HEADING = re.compile(r"^(#{1,6}) (.*)$")
SEPARATOR = re.compile(r"^:?-{2,}:?$")


def link(match):
    """One markdown link. A web address becomes an anchor; a link to a companion file in the
    repository becomes plain marked-up text, because that target does not exist for a reader
    holding only the PDF and a dead blue link is worse than none."""
    label, target = match.group(1), match.group(2)
    if target.startswith(("http://", "https://", "mailto:")):
        # Ampersands are already escaped by the caller; only quotes still need handling.
        return f'<a href="{target.replace(chr(34), "&quot;")}">{label}</a>'
    return f'<span class="xref">{label}</span>'


def inline(text):
    """Inline markup. Code spans are stashed first so their contents stay literal."""
    spans = []

    def stash(match):
        spans.append(match.group(1))
        return f"\x00{len(spans) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash, text)
    text = html.escape(text, quote=False)
    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text, flags=re.S)
    text = re.sub(r"~~(.+?)~~", r"<del>\1</del>", text, flags=re.S)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text, flags=re.S)
    return re.sub(r"\x00(\d+)\x00",
                  lambda m: "<code>" + html.escape(spans[int(m.group(1))], quote=False) + "</code>",
                  text)


def render_table(rows):
    grid = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    if len(grid) < 2 or not all(SEPARATOR.match(c) for c in grid[1]):
        raise ValueError(f"Malformed table near: {rows[0][:70]!r}")
    classes = []
    for spec in grid[1]:
        if spec.startswith(":") and spec.endswith(":"):
            classes.append(' class="c"')
        elif spec.endswith(":"):
            classes.append(' class="r"')
        else:
            classes.append("")

    def cell(tag, index, value):
        return f"<{tag}{classes[index] if index < len(classes) else ''}>{inline(value)}</{tag}>"

    head = "".join(cell("th", j, c) for j, c in enumerate(grid[0]))
    body = "".join("<tr>" + "".join(cell("td", j, c) for j, c in enumerate(row)) + "</tr>"
                   for row in grid[2:])
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def convert(markdown):
    lines, out, buffer, i = markdown.split("\n"), [], [], 0

    def flush():
        if buffer:
            out.append("<p>" + inline(" ".join(buffer).strip()) + "</p>")
            buffer.clear()

    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            flush()
            i += 1
            code = []
            while i < len(lines) and not lines[i].startswith("```"):
                code.append(lines[i])
                i += 1
            i += 1
            out.append("<pre><code>" + html.escape("\n".join(code), quote=False) + "</code></pre>")
            continue
        heading = HEADING.match(line)
        if heading:
            flush()
            level = len(heading.group(1))
            out.append(f"<h{level}>{inline(heading.group(2).strip())}</h{level}>")
            i += 1
            continue
        if RULE.match(line):
            flush()
            out.append("<hr>")
            i += 1
            continue
        if line.lstrip().startswith("|"):
            flush()
            table = []
            while i < len(lines) and lines[i].lstrip().startswith("|"):
                table.append(lines[i])
                i += 1
            out.append(render_table(table))
            continue
        if line.startswith(">"):
            flush()
            quoted = []
            while i < len(lines) and lines[i].startswith(">"):
                quoted.append(lines[i][2:] if len(lines[i]) > 1 else "")
                i += 1
            out.append("<blockquote>" + convert("\n".join(quoted)) + "</blockquote>")
            continue
        if BULLET.match(line) or NUMBER.match(line):
            flush()
            ordered = NUMBER.match(line) is not None
            pattern = NUMBER if ordered else BULLET
            items, current = [], None
            while i < len(lines):
                match = pattern.match(lines[i])
                if match:
                    if current is not None:
                        items.append(current)
                    current = [match.group(2)]
                    i += 1
                elif lines[i].strip() and lines[i].startswith(" ") and current is not None:
                    current.append(lines[i].strip())  # wrapped continuation of the same item
                    i += 1
                else:
                    break
            if current is not None:
                items.append(current)
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>" + "".join(f"<li>{inline(' '.join(it))}</li>" for it in items)
                       + f"</{tag}>")
            continue
        if not line.strip():
            flush()
            i += 1
            continue
        buffer.append(line.strip())
        i += 1
    flush()
    return "\n".join(out)


def build(source):
    markdown = source.read_text()
    chunks = re.split(r"^## ", markdown, flags=re.M)
    title_md, section_mds = chunks[0], chunks[1:]
    if not section_mds:
        raise ValueError("No '## ' sections found; nothing to lay out")

    title = re.search(r"^# (.+)$", title_md, re.M)
    if not title:
        raise ValueError("No '# ' title found")
    intro = re.sub(r"^# .+$", "", title_md, count=1, flags=re.M)
    intro = re.sub(r"^-{3,}\s*$", "", intro, flags=re.M).strip()
    eyebrow = EYEBROWS.get(source.stem, DEFAULT_EYEBROW)
    parts = [f'<div class="title-block"><div class="eyebrow">{html.escape(eyebrow)}</div>'
             f"<h1>{inline(title.group(1).strip())}</h1>{convert(intro)}</div>"]

    for chunk in section_mds:
        heading, _, rest = chunk.partition("\n")
        slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
        # Section breaks replace the manual rules that separated sections in the source.
        rest = re.sub(r"^-{3,}\s*$", "", rest, flags=re.M)
        parts.append(f'<section class="{slug}"><h2>{inline(heading.strip())}</h2>'
                     f"{convert(rest)}</section>")

    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>{html.escape(title.group(1).strip())}</title>"
            f"<style>{CSS}</style></head><body>{''.join(parts)}</body></html>")

    pdf = source.with_suffix(".pdf")
    staged = ROOT / "output" / (source.stem + ".build.html")
    staged.write_text(page)
    # A throwaway profile, so the build works whether or not the user has Chrome open;
    # sharing the default profile makes headless fail on the singleton lock.
    started = time.time()
    try:
        with tempfile.TemporaryDirectory(prefix="debrief-chrome-") as profile:
            result = subprocess.run(
                [CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                 f"--user-data-dir={profile}", "--no-first-run", "--no-default-browser-check",
                 "--virtual-time-budget=4000", f"--print-to-pdf={pdf}", staged.as_uri()],
                capture_output=True, text=True)
    finally:
        staged.unlink(missing_ok=True)

    # Chrome on macOS often writes the PDF correctly and then fails during teardown, so the
    # artifact is the authority on success, not the exit code. A complete PDF ends with %%EOF.
    fresh = pdf.exists() and pdf.stat().st_mtime >= started and pdf.stat().st_size > 0
    if not (fresh and pdf.read_bytes().rstrip().endswith(b"%%EOF")):
        raise SystemExit(f"Chrome exited {result.returncode} without a complete PDF\n"
                         f"{result.stderr.strip()[-2000:]}")
    if result.returncode != 0:
        print(f"note: chrome exited {result.returncode} after writing the PDF (teardown only)")
    print(f"{pdf.relative_to(ROOT)}  {pdf.stat().st_size / 1024:.0f} KB  "
          f"from {len(section_mds)} sections, {len(markdown.splitlines())} source lines")


if __name__ == "__main__":
    # Resolved, so a relative argument still reports a path relative to the project root.
    build(Path(sys.argv[1]).resolve() if len(sys.argv) > 1
          else ROOT / "W4_DEBRIEF_AND_TARGET_SELECTION.md")
