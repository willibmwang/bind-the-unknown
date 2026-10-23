"""Plot what the identity orbit removes from each policy's score.

The figure answers one question: of the attainment a policy reports, how much does the
target's *name* supply? Each row is a policy on a family packet, drawn as the nominal score
and the score the same policy reaches once every declared identifying token is replaced by
an opaque code. The gap between the two dots is `attributable_to_identity`.

**The comparison basis is `null` + `identity`, and that is a deliberate restriction.** The
orbit runs were not all executed over the same set of transformations -- the rubric runs on
this host carry `null,identity` only, the agent runs carry `null,presentation,identity`, and
the memoriser runs carry all four -- so `earned_mean` as recorded is a mean over different
kinds per file and is not comparable across policies. This script recomputes from
`identity_orbit.mean`, which every plotted file has, and reads the spread off `null_orbit`,
which is the same instance re-run and so is the policy's own variance. Mixing hosts would
introduce the same incomparability by a second route, so every plotted cell comes from one
host and the script refuses to run if that stops being true.

    python3 output/build_orbit_figure.py            # light; writes PNG and PDF
    python3 output/build_orbit_figure.py --dark
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORBIT = ROOT / "runs" / "orbit"
OUT = ROOT / "output"

# The user's matplotlibrc declares an `inline` backend that only exists inside IPython, and
# reading it aborts the import. Point the config elsewhere before matplotlib is imported.
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "mpl-orbit"))
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette slots 1 and 2, light and dark steps, taken unchanged. The pair is
# documented as clearing the all-pairs gates in both modes, so it is not re-stepped here.
THEME = {
    "light": dict(
        surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e", muted="#8a897f",
        rule="#dedcd4", nominal="#2a78d6", earned="#eb6834",
    ),
    "dark": dict(
        surface="#1a1a19", primary="#f0efec", secondary="#c3c2b7", muted="#8a897f",
        rule="#3a3936", nominal="#3987e5", earned="#d95926",
    ),
}

# Row order is the argument: the planted memoriser first because it is the demonstration,
# the agent second because it is the subject, the fixed rubric last because it is the
# control that makes the other two readable.
POLICIES = [
    ("memorised", "Planted memoriser"),
    ("agent", "LLM agent"),
    ("rubric", "Fixed rubric"),
]
PACKETS = [
    ("coronaviridae", "Coronaviridae"),
    ("henipavirus", "Henipavirus"),
    ("orthomyxoviridae", "Orthomyxoviridae"),
]

# Where the attributable-to-identity column sits, in axes fractions past the right spine.
COLUMN_X = 1.125


def load_cells():
    """Return one record per policy/packet, all from a single host, on the null+identity basis."""
    cells, hosts = [], set()
    for policy, policy_label in POLICIES:
        for packet, packet_label in PACKETS:
            # The unsuffixed file is the local host; `_modal` is the same run on Modal and
            # reaches different magnitudes, so it is not a substitute.
            path = ORBIT / f"{packet}_{policy}.json"
            if not path.exists():
                sys.exit(f"missing {path.relative_to(ROOT)} -- the plotted set must be complete")
            doc = json.loads(path.read_text())
            block = doc["earned"]["challenger_best"]
            if "identity" not in doc["kinds"]:
                sys.exit(f"{path.name} ran {doc['kinds']} and cannot sit on the identity basis")
            hosts.add(doc["provenance"]["host"])
            nominal = block["nominal"]
            earned = block["identity_orbit"]["mean"]
            null = block["null_orbit"]
            cells.append(dict(
                policy=policy, policy_label=policy_label,
                packet=packet, packet_label=packet_label,
                nominal=nominal, earned=earned, attributable=round(nominal - earned, 4),
                null_sd=null["sd"], null_n=null["n"], identity_n=block["identity_orbit"]["n"],
            ))
    if len(hosts) != 1:
        sys.exit(f"cells span hosts {sorted(hosts)} -- magnitudes are not comparable across them")
    return cells, hosts.pop()


def render(cells, host, mode, stem):
    c = THEME[mode]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "savefig.facecolor": c["surface"], "figure.facecolor": c["surface"],
    })
    fig, ax = plt.subplots(figsize=(11, 6.9), dpi=200)
    ax.set_facecolor(c["surface"])

    # Lay the rows out top to bottom with a gap between policy blocks, so the three blocks
    # read as three claims rather than nine unrelated bars.
    ys, labels, block_spans = [], [], []
    y = 0.0
    for i, (policy, policy_label) in enumerate(POLICIES):
        start = y
        for packet, packet_label in PACKETS:
            ys.append(y)
            labels.append(packet_label)
            y += 1.0
        block_spans.append((policy_label, start, y - 1.0))
        y += 0.75
    ordered = {(d["policy"], d["packet"]): d for d in cells}
    rows = [ordered[(p, k)] for p, _ in POLICIES for k, _ in PACKETS]

    # The two anchors the score is defined between. Without them "0.059" means nothing to
    # anyone who has not read the grader, and the whole figure turns on the reader seeing
    # that the memoriser's scrubbed score lands below the random one.
    ax.axvspan(-0.42, 0.0, color=c["earned"], alpha=0.05, lw=0, zorder=0)
    ax.axvline(0.0, color=c["rule"], lw=1.4, zorder=1)
    ax.axvline(1.0, color=c["rule"], lw=1.4, ls=(0, (4, 3)), zorder=1)

    for row, yy in zip(rows, ys):
        lo, hi = sorted((row["nominal"], row["earned"]))
        # The connector is the quantity: its length is what scrubbing the names cost.
        ax.plot([lo, hi], [yy, yy], color=c["muted"], lw=2.0, solid_capstyle="round", zorder=2)
        # A surface ring keeps both dots visible where they coincide exactly, which is the
        # whole point of the rubric block.
        ax.scatter([row["nominal"]], [yy], s=118, color=c["nominal"],
                   edgecolors=c["surface"], linewidths=2, zorder=3)
        ax.scatter([row["earned"]], [yy], s=118, color=c["earned"],
                   edgecolors=c["surface"], linewidths=2, zorder=4)

        if abs(row["nominal"] - row["earned"]) < 1e-9:
            ax.annotate(f"{row['nominal']:.3f}", (row["nominal"], yy),
                        textcoords="offset points", xytext=(15, 0),
                        va="center", ha="left", fontsize=9.5, color=c["secondary"])
        else:
            ax.annotate(f"{lo:.3f}", (lo, yy), textcoords="offset points", xytext=(-11, 0),
                        va="center", ha="right", fontsize=9.5, color=c["secondary"])
            ax.annotate(f"{hi:.3f}", (hi, yy), textcoords="offset points", xytext=(11, 0),
                        va="center", ha="left", fontsize=9.5, color=c["secondary"])

        # The gap gets its own column rather than a callout, so every row is quantified in
        # the same place and nothing has to be pointed at with an arrow.
        emphatic = row["attributable"] > 0.5
        share = "0.000" if row["attributable"] == 0 else f"{row['attributable']:+.3f}"
        ax.text(COLUMN_X, yy, share,
                transform=ax.get_yaxis_transform(), va="center", ha="right",
                fontsize=10.5 if emphatic else 9.5,
                fontweight="bold" if emphatic else "normal",
                color=c["earned"] if emphatic else c["muted"])

    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=10.5, color=c["secondary"])
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0, pad=6)
    ax.tick_params(axis="x", colors=c["muted"], labelsize=10)

    # Clear of the packet tick labels, which occupy the strip immediately left of the axes.
    for policy_label, start, end in block_spans:
        ax.text(-0.175, (start + end) / 2, policy_label, transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=12, fontweight="bold",
                color=c["primary"], clip_on=False)

    ax.text(COLUMN_X, ys[0] - 0.85, "supplied by\nthe name", transform=ax.get_yaxis_transform(),
            va="center", ha="right", fontsize=9, color=c["muted"],
            linespacing=1.4, clip_on=False)

    ax.set_xlim(-0.42, 1.16)
    anchors = [
        (0.0, "no better than\nbuying at random"),
        (1.0, "the best evidence\non offer"),
    ]
    for x, text in anchors:
        ax.annotate(text, xy=(x, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -34), textcoords="offset points",
                    ha="center", va="top", fontsize=9, color=c["muted"], linespacing=1.4)
    # Inside the band and low, where the rows leave the plot area empty — the legend owns
    # the top edge and the footnote owns everything below the axis.
    ax.annotate("worse than random", xy=(-0.21, 0.0), xycoords=("data", "axes fraction"),
                xytext=(0, 10), textcoords="offset points",
                ha="center", va="bottom", fontsize=9, color=c["earned"], style="italic")
    ax.grid(axis="x", color=c["rule"], lw=0.8, alpha=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=10, markerfacecolor=c["nominal"],
                   markeredgecolor=c["surface"], markeredgewidth=2, label="Nominal score"),
        plt.Line2D([], [], marker="o", ls="", ms=10, markerfacecolor=c["earned"],
                   markeredgecolor=c["surface"], markeredgewidth=2,
                   label="Earned — every target name replaced by an opaque code"),
    ]
    leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.005),
                    frameon=False, fontsize=10.5, handletextpad=0.5,
                    ncol=2, columnspacing=1.8)
    for text in leg.get_texts():
        text.set_color(c["secondary"])

    fig.text(0.012, 0.972, "A perfect score the orbit removes entirely",
             ha="left", va="top", fontsize=17, fontweight="bold", color=c["primary"])
    fig.text(0.012, 0.912,
             "Each turn the co-scientist spends one action buying a piece of evidence, and the "
             "score is how good that purchase was.\nOnly the hard cases count: states where the "
             "most useful evidence concerns a target that is not currently in the lead, so "
             "digging into the favourite cannot score.\nThe planted memoriser buys perfectly on "
             "every packet — until the targets are renamed, when it drops to no better than "
             "random, or worse.",
             ha="left", va="top", fontsize=11, color=c["secondary"], linespacing=1.5)

    def span(key):
        lo, hi = min(d[key] for d in cells), max(d[key] for d in cells)
        return str(lo) if lo == hi else f"{lo}–{hi}"

    fig.text(0.012, 0.022,
             f"Identity orbit, {span('identity_n')} elements per cell, read against a null orbit "
             f"of {span('null_n')} re-runs; basis is null + identity throughout, because the "
             f"recorded runs do not share a transform set. Single host ({host}).\n"
             f"The bound is lexical: a claim scrubbed of every name can still identify its subject, "
             f"so this measures what the name spells out and not what the biology gives away.",
             ha="left", fontsize=8.5, color=c["muted"], linespacing=1.5)

    fig.subplots_adjust(left=0.285, right=0.885, top=0.715, bottom=0.215)
    for ext in ("png", "pdf"):
        path = OUT / f"{stem}.{ext}"
        fig.savefig(path, facecolor=c["surface"])
        print(f"wrote {path.relative_to(ROOT)}")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true", help="render the dark-surface variant too")
    args = ap.parse_args()

    cells, host = load_cells()
    print(f"host: {host}    basis: null + identity")
    print(f"{'policy':20} {'packet':18} {'nominal':>9} {'earned':>9} {'attributable':>13} {'null sd':>8}")
    for d in cells:
        print(f"{d['policy_label']:20} {d['packet_label']:18} {d['nominal']:>9.4f} "
              f"{d['earned']:>9.4f} {d['attributable']:>13.4f} {d['null_sd']:>8.4f}")

    render(cells, host, "light", "fig_orbit_earned")
    if args.dark:
        render(cells, host, "dark", "fig_orbit_earned_dark")


if __name__ == "__main__":
    main()
