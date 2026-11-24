"""Measure every text label on every residue track, in a real browser, with the real fonts.

**Estimating glyph widths is not good enough to place a label by, and it fails worst where it
matters.** A previous figure on this site had two labels moved to avoid a collision that did
not exist: the estimate said 242 units wide, the measured width was 211, and the error grows
with the string, so the longest labels are the ones a heuristic is worst at. A preview without
the Google Fonts link is its own trap, separate from the estimating one, because the monospace
falls back to Menlo and measures short.

So this does the only thing that settles it. It loads a built page in headless Chrome, waits for
`document.fonts.ready`, calls `getBBox()` on every `<text>` inside every `.ftrack svg`, and
reports two things: any pair of boxes on the same line that overlap, and any box that leaves the
viewBox. Nothing is estimated and no result is taken from a preview that did not load the fonts.

Run it after `output/build_site.py`:

    python3 output/measure_track_labels.py                # the landing page and five results
    python3 output/measure_track_labels.py --all          # every page that carries a track

It exits non-zero on a collision or an overflow, so it can gate a build.
"""
import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

# Two labels count as colliding only when their boxes overlap both horizontally and vertically.
# A label directly under another is not a collision, which is the whole layout of the left
# gutter: a lane's name and its small print sit in one column, thirteen units apart.
PROBE = r"""
(async () => {
  await document.fonts.ready;
  const out = [];
  document.querySelectorAll('.ftrack svg').forEach((svg, figure) => {
    const box = svg.viewBox.baseVal;
    const texts = [...svg.querySelectorAll('text')].map(node => {
      const b = node.getBBox();
      return {text: node.textContent.trim(),
              x: b.x, y: b.y, w: b.width, h: b.height,
              anchor: node.getAttribute('text-anchor') || 'start',
              cls: node.getAttribute('class') || ''};
    });
    out.push({figure, label: svg.getAttribute('aria-label') || '',
              viewBox: {w: box.width, h: box.height}, texts});
  });
  return JSON.stringify(out);
})()
"""


def measure(path):
    """Every text box on every track of one built page, measured in the browser."""
    harness = (
        "<!doctype html><meta charset=utf-8>"
        f"<base href=\"{path.as_uri()}\">"
        f"{path.read_text()}"
        "<script>"
        "window.addEventListener('load', async () => {"
        f"  const payload = await ({PROBE});"
        "  const pre = document.createElement('pre');"
        "  pre.id = 'measurement'; pre.textContent = payload;"
        "  document.body.appendChild(pre);"
        "});"
        "</script>")
    with tempfile.TemporaryDirectory(prefix="track-probe-") as scratch:
        staged = Path(scratch) / "probe.html"
        staged.write_text(harness)
        profile = Path(scratch) / "profile"
        result = subprocess.run(
            [CHROME, "--headless=new", "--disable-gpu", f"--user-data-dir={profile}",
             "--no-first-run", "--no-default-browser-check", "--virtual-time-budget=9000",
             "--dump-dom", staged.as_uri()],
            capture_output=True, text=True)
    found = re.search(r'<pre id="measurement">(.*?)</pre>', result.stdout, re.S)
    if not found:
        raise SystemExit(
            f"Chrome returned no measurement for {path.name}. It exited {result.returncode}.\n"
            f"A page with no .ftrack svg on it produces an empty list, not a missing one, so "
            f"this is a probe failure rather than a page without a track.\n"
            f"{result.stderr.strip()[-1500:]}")
    import html as html_module

    return json.loads(html_module.unescape(found.group(1)))


def _left(box):
    """A box's left edge, which getBBox already resolves for every anchor."""
    return box["x"]


def check(figures, page):
    """Collisions and overflows, as a list of complaints. Empty is the pass."""
    problems = []
    for figure in figures:
        width = figure["viewBox"]["w"]
        height = figure["viewBox"]["h"]
        texts = figure["texts"]
        for box in texts:
            if _left(box) < -0.5 or _left(box) + box["w"] > width + 0.5:
                problems.append(
                    f"{page}: figure {figure['figure']}: {box['text']!r} runs from "
                    f"{_left(box):.1f} to {_left(box) + box['w']:.1f} and the viewBox is "
                    f"{width:.0f} wide")
            if box["y"] < -0.5 or box["y"] + box["h"] > height + 0.5:
                problems.append(
                    f"{page}: figure {figure['figure']}: {box['text']!r} sits from "
                    f"{box['y']:.1f} to {box['y'] + box['h']:.1f} and the viewBox is "
                    f"{height:.0f} tall")
        for i, one in enumerate(texts):
            for other in texts[i + 1:]:
                vertical = (one["y"] < other["y"] + other["h"] - 0.5
                            and other["y"] < one["y"] + one["h"] - 0.5)
                horizontal = (_left(one) < _left(other) + other["w"] - 0.5
                              and _left(other) < _left(one) + one["w"] - 0.5)
                if vertical and horizontal:
                    problems.append(
                        f"{page}: figure {figure['figure']}: {one['text']!r} and "
                        f"{other['text']!r} overlap by "
                        f"{min(_left(one) + one['w'], _left(other) + other['w']) - max(_left(one), _left(other)):.1f} "
                        f"units horizontally")
    return problems


def pages(every=False):
    """Pages to probe. Five results by default; every one that carries a track with --all."""
    carrying = [p for p in sorted((DOCS / "results").glob("*.html"))
                if "ftrack" in p.read_text()]
    chosen = carrying if every else carrying[:5]
    return [DOCS / "index.html"] + chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="Probe every result page that carries a track, not five of them")
    args = parser.parse_args(argv)

    if not Path(CHROME).exists():
        raise SystemExit(f"No Chrome at {CHROME}. The measurement needs a real browser with "
                         f"the real fonts; estimating the widths instead is the mistake this "
                         f"script exists to prevent.")

    problems, measured, figures_seen = [], 0, 0
    for path in pages(args.all):
        figures = measure(path)
        figures_seen += len(figures)
        measured += sum(len(f["texts"]) for f in figures)
        problems.extend(check(figures, path.relative_to(ROOT)))

    print(f"measured {measured} label(s) across {figures_seen} track(s) "
          f"in {len(pages(args.all))} page(s), with the page's own fonts loaded")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print("no label overlaps another and none leaves its viewBox")
    return 0


if __name__ == "__main__":
    sys.exit(main())
