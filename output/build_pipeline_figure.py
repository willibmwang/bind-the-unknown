"""Render the two-arm pipeline flowchart to a file, from the chain record.

Section 0.1 of `notebooks/double_arm_mvp.ipynb` draws this figure inline through
`panels.workflow_figure`. The same picture is wanted outside the notebook -- in a talk, in a
document, in the README for a reader who is not on GitHub and so never sees the Mermaid block
render. This script is the entry point for that, and it calls the identical function rather
than redrawing it, so the standalone image and the notebook page cannot drift apart.

**The committed image is the holdout**, the genome with SARS-CoV-2 deleted from the reference,
because that is the setting every validation run uses: identification has to work through a
relative, and the surface stage keeps a measurable surface for 9 of 20 chains rather than 20 of
21. The other reference is a different claim about coverage and is not interchangeable with it,
so the subtitle names which one is drawn and `--case 0` draws the other on demand rather than
leaving two unlabelled images to be told apart by opening them.

The two gain ranges are arm 2's and are not in the chain record. They are quoted here from
the same source the notebook cell quotes, and passed in rather than typed inside the figure.

    python3 output/build_pipeline_figure.py              # the holdout, PNG
    python3 output/build_pipeline_figure.py --case 0     # the own-reference case instead
    python3 output/build_pipeline_figure.py --pdf --svg  # add vector formats
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "output"
CHAIN = RUNS / "onset_chain.json"

# The user's matplotlibrc declares an `inline` backend that only exists inside IPython, and
# reading it aborts the import. Point the config elsewhere before matplotlib is imported.
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "mpl-pipeline"))
os.environ.setdefault("MPLBACKEND", "Agg")

# `panels` lives beside the notebook it was lifted out of, and imports `hpc_pilots`, so both
# the repository root and the notebook directory have to be importable.
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "notebooks"))

import matplotlib.pyplot as plt  # noqa: E402
import panels  # noqa: E402

# Arm 2's two gain ranges, from the per-batch summaries section 2.1 reads. Quoted in one
# place so the figure, the notebook cell and the README table cannot disagree silently.
GAIN_SHARED = "+0.173 to +0.321"
GAIN_UNSHARED = "+0.010 to +0.033"

# What each case is for, in the filename, so neither image has to be identified by opening it.
STEMS = {
    0: "fig_pipeline_own_reference",
    1: "fig_pipeline_holdout",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", type=int, choices=(0, 1), action="append",
                    help="which reference to draw; repeatable, default the holdout (1)")
    ap.add_argument("--pdf", action="store_true", help="write PDF beside the PNG")
    ap.add_argument("--svg", action="store_true", help="write SVG beside the PNG")
    args = ap.parse_args()

    if not CHAIN.exists():
        sys.exit(f"missing {CHAIN.relative_to(ROOT)} -- rebuild with "
                 f"`python -m hpc_pilots chain`")
    chain = json.loads(CHAIN.read_text())

    cases = args.case if args.case else [1]
    for case in cases:
        if case >= len(chain["cases"]):
            sys.exit(f"the record holds {len(chain['cases'])} case(s); --case {case} is "
                     f"out of range")
        run = chain["cases"][case]
        cov = run["coverage"]
        print(f"case {case}: {run['case']}")
        print(f"  {run['genome_accession']}  ·  {len(run['stages'])} stages  ·  "
              f"{run['seconds']:.0f} s")
        print(f"  surface covers every chain: {cov['surface_covers_every_chain']}   "
              f"fold covers every chain: {cov['fold_covers_every_chain']}")

        fig = panels.workflow_figure(chain, case=case, gain_shared=GAIN_SHARED,
                                     gain_unshared=GAIN_UNSHARED)
        stem = STEMS[case]
        formats = ("png",) + (("pdf",) if args.pdf else ()) + (("svg",) if args.svg else ())
        for ext in formats:
            path = OUT / f"{stem}.{ext}"
            fig.savefig(path, facecolor=fig.get_facecolor(), bbox_inches="tight")
            print(f"  wrote {path.relative_to(ROOT)}")
        plt.close(fig)


if __name__ == "__main__":
    main()
