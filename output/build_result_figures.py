"""Draw the four results of 23-24 November, one claim a figure.

Each figure answers one question and is drawn from a record rather than from a number typed
here. Three read `runs/rubric_fit.json`, which carries its own provenance and content hash;
the fourth reads the two monomer sweeps. The per-row panel of figure 1 is the one exception
that needs the panel records themselves, because `rubric_fit.json` keeps the counts of each
conservation scale but not the values, and it goes through `rubric_fit.as_measured_genomes`
so a record written after the repair reports what its instrument read rather than what the
withholding substituted. That is also what makes it safe to draw while a panel is part-way
through a re-run.

    1  conservation_two_scales   the defect: one term, two rulers, and what the mixing costs
    2  conservation_repair       the repair: recall at k before and after, and one worked case
    3  rubric_arms              the weighting is not what carries the ranking
    4  monomer_transfer          the monomer-transfer null does not hold on a second landscape

Every figure prints the numbers it drew so they can be read against the record without
opening the image, and figure 1 prints the scale counts it found: if a re-run moves them,
the print says so rather than the picture drifting quietly.

    python3 output/build_result_figures.py                  # light, PNG and PDF
    python3 output/build_result_figures.py --dark           # the dark-surface variant too
    python3 output/build_result_figures.py --only 4         # one figure; repeatable
"""
import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
OUT = ROOT / "output"
FIT = RUNS / "rubric_fit.json"
MONOMERS = RUNS / "monomers"

# The user's matplotlibrc declares an `inline` backend that only exists inside IPython, and
# reading it aborts the import. Point the config elsewhere before matplotlib is imported.
os.environ.setdefault("MPLCONFIGDIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "mpl-results"))
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette slots 1 and 2, light and dark steps, taken unchanged from
# `output/build_orbit_figure.py` so the figures in this repository read as one set. The pair
# clears the all-pairs gates in both modes, so it is not re-stepped here.
THEME = {
    "light": dict(
        surface="#fcfcfb", primary="#0b0b0b", secondary="#52514e", muted="#8a897f",
        rule="#dedcd4", one="#2a78d6", two="#eb6834", control="#a8a69b",
    ),
    "dark": dict(
        surface="#1a1a19", primary="#f0efec", secondary="#c3c2b7", muted="#8a897f",
        rule="#3a3936", one="#3987e5", two="#d95926", control="#6f6e66",
    ),
}

# The alphabet each monomer sweep holds out of, from `hpc_pilots/configs/display.json`: 30
# variable building blocks on the thrombin library, 118 distinct monomers on CycPeptMPDB.
# Held-out counts are only comparable across the two as a fraction of these.
ALPHABET = {"thrombin_cyclic": 30, "cycpept_len6_pampa": 118}

# Sizes the CycPeptMPDB sweep ran but which it records as uninterpretable: five draws and
# draw-to-draw deviations up to 0.59. Drawn hollow rather than dropped, because the sweep
# wrote them and a reader should see what was excluded.
UNDERPOWERED = {"cycpept_len6_pampa": (2, 4)}

LABEL = {"thrombin_cyclic": "Does it stick to its target?\nthrombin, 30 blocks, one laboratory",
         "cycpept_len6_pampa": "Does it get through a membrane?\nCycPeptMPDB, 118 blocks, 22 laboratories"}

# Where each line carries its own name: the point to anchor on, how far off the line, and
# which side. Direct labels rather than a legend box, so identity is never a colour lookup.
ANCHOR = {"thrombin_cyclic": (3, 0.105, "bottom"),
          "cycpept_len6_pampa": (6, 0.115, "bottom")}


def style(mode):
    c = THEME[mode]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "savefig.facecolor": c["surface"], "figure.facecolor": c["surface"],
    })
    return c


def bare(ax, c, keep=("left", "bottom")):
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(side in keep)
    for side in keep:
        ax.spines[side].set_color(c["rule"])
    ax.set_facecolor(c["surface"])
    ax.tick_params(colors=c["muted"], labelsize=9.5, length=0, pad=5)
    ax.grid(False)


def heading(fig, c, title, subtitle, footer):
    """Title, standfirst and provenance note, wrapped to the figure's own width.

    Wrapped rather than typed with line breaks because `bbox_inches="tight"` grows the saved
    image to whatever the longest text runs to: one unwrapped footer stretched an 11-inch
    figure to 34 inches and left the plot in a corner of it.
    """
    width = fig.get_size_inches()[0]
    fig.text(0.012, 0.975, textwrap.fill(title, int(width * 8.2)), ha="left", va="top",
             fontsize=16.5, fontweight="bold", color=c["primary"])
    fig.text(0.012, 0.915, textwrap.fill(subtitle, int(width * 12.4)), ha="left", va="top",
             fontsize=10.5, color=c["secondary"], linespacing=1.5)
    fig.text(0.012, 0.018, textwrap.fill(footer, int(width * 15.4)), ha="left", va="bottom",
             fontsize=8.5, color=c["muted"], linespacing=1.45)


def save(fig, stem, c, formats=("png", "pdf")):
    for ext in formats:
        path = OUT / f"fig_{stem}.{ext}"
        fig.savefig(path, facecolor=c["surface"], bbox_inches="tight")
        print(f"  wrote {path.relative_to(ROOT)}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. The defect
# ---------------------------------------------------------------------------
def conservation_rows():
    """Every candidate row's conservation value and the instrument that read it.

    Through `as_measured_genomes`, so a record carrying the shipped withholding reports the
    measurement the withholding displaced. Returns the values per scale and the counts, and
    the counts are printed by the caller against the ones `rubric_fit.json` recorded.
    """
    from hpc_pilots import rubric_fit as rf

    genomes, _ = rf.panel_matrix(tier="all")
    index = rf.FEATURES.index("conservation")
    out = {}
    for genome in rf.as_measured_genomes(genomes):
        for values, row in zip(genome["x"], genome["rows"]):
            scale = (row.get("features") or {}).get("conservation_scale") or "unrecorded"
            out.setdefault(scale, []).append(float(values[index]))
    return out


def figure_two_scales(fit, c, mode):
    scales = conservation_rows()
    exact = np.asarray(scales.get("shared_kmer_fraction_of_query", []))
    remote = np.asarray(scales.get("bits_over_200", []))
    unrecorded = len(scales.get("unrecorded", []))
    recorded = fit["conservation_scales"]
    print(f"  scales drawn: exact {len(exact)}, remote {len(remote)}, "
          f"unrecorded {unrecorded}")
    print(f"  scales in runs/rubric_fit.json: {recorded}")
    if {"shared_kmer_fraction_of_query": len(exact), "bits_over_200": len(remote),
            "unrecorded": unrecorded} != recorded:
        print("  NOTE: the panel on disk no longer matches the fit record. The figure draws "
              "the panel; re-run `python -m hpc_pilots rubric-fit` to move the record to it.")

    split = fit["conservation_scale_split"]
    fig = plt.figure(figsize=(12.4, 6.4), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.45, 1.0), left=0.055, right=0.985,
                          top=0.77, bottom=0.16, wspace=0.26)

    # (a) the two rulers, every row drawn
    ax = fig.add_subplot(gs[0, 0])
    bare(ax, c, keep=("bottom",))
    rng = np.random.default_rng(20261124)
    for y, values, colour, name in ((1.0, exact, c["one"], "exact"),
                                    (0.0, remote, c["two"], "remote")):
        if not len(values):
            continue
        jitter = y + rng.uniform(-0.17, 0.17, len(values))
        ax.scatter(values, jitter, s=14, color=colour, alpha=0.5, linewidths=0,
                   zorder=3, label=None)
        median = float(np.median(values))
        ax.plot([median, median], [y - 0.28, y + 0.28], color=c["primary"],
                linewidth=2.0, zorder=4, solid_capstyle="butt")
        ax.annotate(f"median {median:.2f}", (median, y + 0.33), ha="center", va="bottom",
                    fontsize=9, color=c["secondary"])
    saturated = int(np.sum(remote >= 0.999)) if len(remote) else 0
    ax.annotate(f"{saturated} of {len(remote)} at 1.0",
                (1.0, -0.42), ha="right", va="top", fontsize=9, color=c["two"])
    ax.set_yticks([1.0, 0.0])
    ax.set_yticklabels([f"Shared 5-mer fraction\nexact tier, {len(exact)} rows",
                        f"min(1, bits / 200)\nremote tier, {len(remote)} rows"],
                       fontsize=10, color=c["secondary"], linespacing=1.35)
    ax.tick_params(axis="y", pad=8)
    ax.set_xlim(-0.03, 1.06)
    ax.set_ylim(-0.75, 1.6)
    ax.set_xlabel("conservation, as the rubric reads it", fontsize=10, color=c["secondary"])
    ax.annotate("one term, weight 3.0, read as one quantity",
                (0.5, 1.47), xycoords=("axes fraction", "data"), ha="center", va="bottom",
                fontsize=10, color=c["primary"], fontweight="bold")

    # (b) what the mixing costs, and what dropping the term recovers
    ax = fig.add_subplot(gs[0, 1])
    bare(ax, c, keep=("left",))
    arms = ("stated", "fitted_without_conservation")
    arm_labels = ("Term as shipped", "Term dropped")
    groups = (("mixed", c["two"], f"{split['counts']['mixed']} genomes, both scales"),
              ("single", c["one"], f"{split['counts']['single']} genomes, one scale"))
    width = 0.34
    for offset, (group, colour, label) in zip((-0.19, 0.19), groups):
        heights = [split["arms"][arm][group]["recall_at_k"]["top1"] for arm in arms]
        xs = np.arange(len(arms)) + offset
        ax.bar(xs, heights, width=width, color=colour, linewidth=0, label=label, zorder=3)
        for x, h in zip(xs, heights):
            ax.annotate(f"{h:.1f}", (x, h + 1.4), ha="center", va="bottom", fontsize=9.5,
                        color=c["secondary"])
    gap = split["shipped_rubric_costs_on_mixed_genomes"]
    low, high = (split["arms"]["stated"]["mixed"]["recall_at_k"]["top1"],
                 split["arms"]["stated"]["single"]["recall_at_k"]["top1"])
    ax.annotate("", xy=(-0.52, low), xytext=(-0.52, high),
                arrowprops=dict(arrowstyle="<->", color=c["primary"], linewidth=1.2))
    ax.annotate(f"{gap:.1f}\npoints", (-0.52, high + 1.5), ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=c["primary"], linespacing=1.3)
    ax.annotate("indistinguishable", (1.0, 71.5), ha="center", va="bottom",
                fontsize=9.5, color=c["secondary"], style="italic")
    ax.set_xlim(-0.8, 1.45)
    ax.set_xticks(np.arange(len(arms)))
    ax.set_xticklabels(arm_labels, fontsize=10, color=c["secondary"])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_ylim(0, 84)
    ax.set_ylabel("labels recovered at rank 1, %", fontsize=10, color=c["secondary"])
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, handlelength=0.9,
              handletextpad=0.6, labelcolor=c["secondary"])

    heading(
        fig, c,
        "The heaviest ranking term arrives on two incomparable scales",
        "A protein identified through a distant relative is scored on a bit score that "
        "saturates; one identified exactly is scored on a k-mer fraction that does not. "
        "The genomes carrying both rank worse, and dropping the term recovers it.",
        f"{unrecorded} further rows were never measured and are not drawn: a 0.0 there is "
        f"an absence of evidence, and the repair leaves it alone. "
        f"runs/rubric_fit.json, {fit['genomes_scored']} scoreable genomes, "
        f"{fit['candidate_rows']} candidate rows, {fit['labels_matched']} matched labels. "
        f"Recall at rank 1 against a chance baseline of "
        f"{fit['comparison']['chance']['top1']:.1f}%. One genome of the mixed group is worth "
        f"5.9 points and one of the single group 3.7, so neither panel carries an interval.")
    save(fig, f"conservation_two_scales{'_dark' if mode == 'dark' else ''}", c)


# ---------------------------------------------------------------------------
# 2. The repair
# ---------------------------------------------------------------------------
# The record names a chain as its source does. Two of them are wider than the panel, and a
# shortening is a display choice, so it is made here and not in the record.
SHORT = {"RNA-directed RNA polymerase nsp12": "RNA polymerase nsp12",
         "3C-like proteinase nsp5": "Main protease nsp5"}


def figure_repair(fit, c, mode):
    rep = fit["conservation_repair"]
    depths = (1, 3, 5)
    before = [rep["recall_before"][f"top{k}"] for k in depths]
    after = [rep["recall_after"][f"top{k}"] for k in depths]
    chance = [rep["chance"][f"top{k}"] for k in depths]
    print(f"  recall@1 {before[0]} -> {after[0]}, recall@3 {before[1]} -> {after[1]}, "
          f"recall@5 {before[2]} -> {after[2]}")
    print(f"  {rep['rows_imputed']} rows withheld, {rep['paired']['genomes_better']} genomes "
          f"better, {rep['paired']['genomes_worse']} worse, sign test p = "
          f"{rep['paired']['sign_test_p']}")

    fig = plt.figure(figsize=(12.4, 6.2), dpi=200)
    gs = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.22), left=0.055, right=0.985,
                          top=0.76, bottom=0.22, wspace=0.30)

    # (a) recall at k, before and after, with chance drawn as a reference and not a series
    ax = fig.add_subplot(gs[0, 0])
    bare(ax, c, keep=("left",))
    width = 0.33
    xs = np.arange(len(depths))
    for offset, heights, colour, label in ((-0.18, before, c["control"], "Term as shipped"),
                                           (0.18, after, c["one"], "Remote value withheld")):
        ax.bar(xs + offset, heights, width=width, color=colour, linewidth=0, label=label,
               zorder=3)
        for x, h in zip(xs + offset, heights):
            ax.annotate(f"{h:.1f}", (x, h + 1.3), ha="center", va="bottom", fontsize=9.5,
                        color=c["secondary"])
    for x, value in zip(xs, chance):
        ax.plot([x - 0.36, x + 0.36], [value, value], color=c["primary"], linewidth=1.3,
                linestyle=(0, (4, 3)), zorder=4)
    ax.annotate("chance", (xs[0] - 0.36, chance[0] + 1.5), ha="left", va="bottom",
                fontsize=9, color=c["primary"])
    for x, (b, a) in zip(xs, zip(before, after)):
        ax.annotate(f"+{a - b:.1f}", (x, max(a, b) + 7.5), ha="center", va="bottom",
                    fontsize=10, fontweight="bold", color=c["one"])
    ax.set_xticks(xs)
    ax.set_xticklabels([f"rank {k}" for k in depths], fontsize=10, color=c["secondary"])
    ax.tick_params(axis="x", length=0, pad=6)
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("labels recovered, %", fontsize=10, color=c["secondary"])
    ax.legend(loc="upper left", frameon=False, fontsize=9.5, handlelength=0.9,
              handletextpad=0.6, labelcolor=c["secondary"], bbox_to_anchor=(0, 1.02))

    # (b) the worked case, which is what the aggregate is made of
    ax = fig.add_subplot(gs[0, 1])
    bare(ax, c, keep=("bottom",))
    examples = [e for e in fit["cross_scale_inversions"]["examples"]
                if e["pathogen"] == "HCoV-229E"]
    winner = examples[0]["candidate_that_won"]
    rows = [(winner, examples[0]["candidate_conservation"], "remote", None)]
    for e in sorted(examples, key=lambda e: -e["points_between_them"]):
        rows.append((e["target_outranked"], e["target_conservation"], "exact",
                     e["points_between_them"]))
    ys = np.arange(len(rows))[::-1]
    for y, (name, value, scale, behind) in zip(ys, rows):
        colour = c["two"] if scale == "remote" else c["one"]
        ax.plot([0, value], [y, y], color=c["rule"], linewidth=1.4, zorder=2)
        ax.scatter([value], [y], s=96, color=colour, linewidths=0, zorder=3)
        ax.annotate(f"{value:.2f}", (value, y + 0.21), ha="center", va="bottom",
                    fontsize=9.5, color=c["secondary"])
        label = name.split(" ", 1)[-1] if name.startswith(("SPIKE_", "VME1_")) else name
        label = SHORT.get(label, label)
        ax.annotate(label, (-0.035, y), ha="right", va="center", fontsize=10,
                    color=c["primary"] if behind is None else c["secondary"],
                    fontweight="bold" if behind is None else "normal")
        ax.annotate("—" if behind is None else f"{behind:.2f}", (1.32, y), ha="center",
                    va="center", fontsize=10, color=c["secondary"])
    ax.annotate("points\nbehind", (1.32, len(rows) - 0.62), ha="center", va="bottom",
                fontsize=9, color=c["muted"], linespacing=1.3)
    ax.annotate("read on the saturating scale", (examples[0]["candidate_conservation"], ys[0] - 0.3),
                ha="right", va="top", fontsize=9, color=c["two"])
    ax.annotate("nsp5 and nsp12 both carry an approved antiviral.\nThe membrane protein does "
                "not, and outranked them.", (-0.5, -0.55), ha="left", va="top", fontsize=9,
                color=c["secondary"], linespacing=1.4)
    ax.set_xlim(-0.52, 1.46)
    ax.set_ylim(-1.15, len(rows) - 0.1)
    ax.set_yticks([])
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.spines["bottom"].set_bounds(0, 1.0)
    ax.set_xlabel("conservation as read, HCoV-229E", fontsize=10, color=c["secondary"])

    heading(
        fig, c,
        "Withholding the value the weight was not calibrated for",
        "A remote-tier row is given the typical exact-tier value of its own genome, so the "
        "term abstains there. Nothing is converted, so no divisor is chosen by hand. On "
        "HCoV-229E the membrane protein had been outranking two proteins that are already drugged.",
        f"The five shipped weights are unchanged and {rep['rows_imputed']} rows are "
        f"withheld: {rep['paired']['genomes_better']} genomes better, "
        f"{rep['paired']['genomes_worse']} worse, {rep['paired']['genomes_tied']} unmoved, "
        f"sign test p = {rep['paired']['sign_test_p']}, which is its floor for "
        f"{rep['paired']['genomes_that_moved']} moves. "
        "runs/rubric_fit.json, conservation_repair and cross_scale_inversions. Scored "
        "through `intake.withhold_remote_conservation` as shipped, not through the module "
        "that measured it. Five genomes is as much as a panel this size can say: this is a "
        "direction, not a significance claim.")
    save(fig, f"conservation_repair{'_dark' if mode == 'dark' else ''}", c)


# ---------------------------------------------------------------------------
# 3. The weighting
# ---------------------------------------------------------------------------
def figure_arms(fit, c, mode):
    comp = fit["comparison"]
    arms = [
        ("stated", "Weights the pipeline ships", c["one"]),
        ("uniform", "All five weighted equally", c["one"]),
        ("fitted", "Fitted, held out by family", c["one"]),
        ("fitted_without_conservation", "Fitted, conservation dropped", c["one"]),
        ("permuted", "Fitted on shuffled labels", c["control"]),
    ]
    values = [comp["recall_by_arm"][key]["top1"] for key, _, _ in arms]
    chance = comp["chance"]["top1"]
    print(f"  recall@1 by arm: " + ", ".join(
        f"{key} {v:.1f}" for (key, _, _), v in zip(arms, values)))
    print(f"  chance {chance:.1f}, spread {comp['spread_across_arms']['top1']:.1f} points, "
          f"fitted minus stated {comp['fitted_minus_stated']['top1']:+.1f}")

    fig, ax = plt.subplots(figsize=(11.2, 5.5), dpi=200)
    fig.subplots_adjust(left=0.30, right=0.965, top=0.70, bottom=0.17)
    bare(ax, c, keep=("bottom",))
    ys = np.arange(len(arms))[::-1]
    for y, (key, label, colour), value in zip(ys, arms, values):
        ax.barh(y, value, height=0.52, color=colour, linewidth=0, zorder=3)
        ax.annotate(f"{value:.1f}", (value + 0.7, y), ha="left", va="center", fontsize=10.5,
                    fontweight="bold" if key == "stated" else "normal",
                    color=c["secondary"])
        # `annotate` clips a label whose anchor lies outside the axes, and a row label by
        # definition does. `text` with clip_on off is the one that draws.
        ax.text(-1.2, y, label, ha="right", va="center", fontsize=10.5, clip_on=False,
                color=c["primary"] if key in ("stated", "permuted") else c["secondary"],
                fontweight="bold" if key == "stated" else "normal")
    ax.axvline(chance, color=c["primary"], linewidth=1.3, linestyle=(0, (4, 3)), zorder=4)
    ax.annotate(f"chance {chance:.1f}", (chance - 1.2, ys[0] + 0.55), ha="right", va="bottom",
                fontsize=9.5, color=c["primary"])
    low, high = min(values), max(values)
    ax.annotate("", xy=(low, -0.95), xytext=(high, -0.95),
                arrowprops=dict(arrowstyle="<->", color=c["muted"], linewidth=1.1))
    ax.annotate(f"every weighting lands inside {comp['spread_across_arms']['top1']:.1f} points",
                ((low + high) / 2, -1.28), ha="center", va="top", fontsize=9.5,
                color=c["muted"])
    ax.set_yticks([])
    ax.set_ylim(-1.75, len(arms) - 0.35)
    ax.set_xlim(0, 72)
    ax.set_xlabel("labels recovered at rank 1, %", fontsize=10, color=c["secondary"])

    heading(
        fig, c,
        "The five terms carry the ranking; their weighting does not",
        "Fitting the weights on every virus family but one gains "
        f"{comp['fitted_minus_stated']['top1']:+.1f} points at rank 1 over the weights "
        "the pipeline ships. The same fit on shuffled labels scores "
        f"{comp['fitted_minus_permuted']['top1']:.1f} points below the fitted arm rather "
        "than far below it, so a margin the control can match is not learning.",
        f"runs/rubric_fit.json, {fit['genomes_scored']} scoreable genomes over "
        f"{fit['families']} families, {fit['training_pairs']} training pairs, "
        f"leave-one-family-out. `intake.RUBRIC` is unchanged by the run. The labels are "
        "proteins that already carry an approved antiviral, so they are the "
        "best-characterised proteins of the best-characterised viruses; holding out a family "
        "cannot control for that.")
    save(fig, f"rubric_arms{'_dark' if mode == 'dark' else ''}", c)


# ---------------------------------------------------------------------------
# 4. The second landscape
# ---------------------------------------------------------------------------
def transfer_series(landscape):
    """Per held-out size: the real-descriptor advantage over shuffled descriptors.

    The control is the same ten numbers per position permuted across monomers, so
    dimensionality, scale and model class are untouched and only the correspondence between
    a block's identity and its structure is destroyed. An advantage that survives it is not
    chemistry. Forest arm, because both landscapes ran it; the ridge arm is printed by the
    caller and tells the same story.
    """
    record = json.loads((MONOMERS / f"{landscape}_sweep.json").read_text())
    alphabet = ALPHABET[landscape]
    out = []
    for block in record["summary"]:
        # A sweep writes its permuted-outcome control as one more summary block at one size.
        # It belongs to `permuted_band`, not to the series, and plotting it here would put a
        # control point on the same line as the measurement.
        if block.get("control") is not None:
            continue
        real = block["by_encoder"]["descriptor/forest"]["held_out_spearman"]
        shuffled = block["by_encoder"]["descriptor_shuffled/forest"]["held_out_spearman"]
        vocabulary = block["by_encoder"]["vocabulary/forest"].get("tied_rate") or {}
        out.append({
            "held_out": block["held_out_count"],
            "fraction": 100.0 * block["held_out_count"] / alphabet,
            "gap": real["mean"] - shuffled["mean"],
            "sd": real["sd"] or 0.0,
            "draws": block["draws"],
            "tied_rate": vocabulary.get("mean"),
        })
    return out


def permuted_band(landscape):
    """The range every arm of the permuted-outcome control reaches on this landscape."""
    paths = sorted(MONOMERS.glob(f"{landscape}_held*_permuted.json"))
    values = []
    for path in paths:
        for arm in json.loads(path.read_text())["results"]:
            value = arm.get("held_out_spearman")
            if value is not None:
                values.append(float(value))
    return (min(values), max(values), [p.name for p in paths]) if values else None


def figure_transfer(c, mode):
    series = {name: transfer_series(name) for name in ALPHABET}
    fig, ax = plt.subplots(figsize=(11.6, 6.1), dpi=200)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.74, bottom=0.235)
    bare(ax, c, keep=("left", "bottom"))

    bands = {name: permuted_band(name) for name in ALPHABET}
    low = min(b[0] for b in bands.values() if b)
    high = max(b[1] for b in bands.values() if b)
    ax.axhspan(low, high, color=c["control"], alpha=0.28, linewidth=0, zorder=1)
    ax.annotate(f"permuted outcomes, every arm: {low:+.2f} to {high:+.2f}",
                (25.5, low - 0.015), ha="left", va="top", fontsize=9, color=c["secondary"])
    ax.axhline(0, color=c["rule"], linewidth=1.1, zorder=2)

    for name, colour in (("thrombin_cyclic", c["one"]), ("cycpept_len6_pampa", c["two"])):
        points = series[name]
        weak = UNDERPOWERED.get(name, ())
        xs = [p["fraction"] for p in points]
        ys = [p["gap"] for p in points]
        ax.plot(xs, ys, color=colour, linewidth=2.0, zorder=4)
        for p in points:
            solid = p["held_out"] not in weak
            ax.errorbar(p["fraction"], p["gap"], yerr=p["sd"], fmt="none",
                        ecolor=colour, elinewidth=1.1, capsize=3, alpha=0.5, zorder=3)
            ax.scatter([p["fraction"]], [p["gap"]], s=74, zorder=5,
                       color=colour if solid else c["surface"],
                       edgecolors=colour, linewidths=0 if solid else 1.8)
        anchor, dy, va = ANCHOR[name]
        at = points[anchor]
        ax.annotate(LABEL[name], (at["fraction"], at["gap"] + dy), ha="center", va=va,
                    fontsize=10.5, color=colour, linespacing=1.35)
        print(f"  {name}: " + ", ".join(
            f"{p['held_out']}/{ALPHABET[name]} ({p['fraction']:.1f}%) {p['gap']:+.3f} "
            f"±{p['sd']:.2f}" for p in points))

    weak_points = [p for p in series["cycpept_len6_pampa"]
                   if p["held_out"] in UNDERPOWERED["cycpept_len6_pampa"]]
    ax.annotate("hollow: five draws and deviations to 0.59,\nwhich the sweep records as "
                "uninterpretable.\nDrawn because it ran them.",
                (25.5, 0.66), ha="left", va="top", fontsize=8.5, color=c["muted"],
                linespacing=1.45)
    ax.set_xlabel("share of that library's own building blocks hidden from the model, %",
                  fontsize=10, color=c["secondary"])
    # The y quantity is the whole argument and reads as an abstraction without the direction
    # said out loud. It goes in the axis label rather than in a margin arrow, which collided
    # with the label and the tick numbers at every position that was also legible.
    ax.set_ylabel("real chemistry minus scrambled chemistry\n"
                  "held-out Spearman · higher means the structures helped",
                  fontsize=10, color=c["secondary"], linespacing=1.6)
    ax.annotate("0 = the real structures\nwere worth nothing", (0.995, 0.055),
                xycoords="axes fraction", ha="right", va="bottom", fontsize=8.5,
                color=c["secondary"], linespacing=1.4)
    ax.set_xlim(0, 43)
    ax.set_ylim(-0.17, 0.72)

    heading(
        fig, c,
        "The monomer-transfer null does not hold on a second landscape",
        "On the thrombin library, grounding a held-out monomer in measured structure adds no "
        "ranking skill over a vocabulary encoding: the real structures are worth no more "
        "than structures shuffled between blocks. That was one assay, one target, one "
        "laboratory. On membrane permeability over four times the alphabet they are worth a "
        "great deal, at every size.",
        "runs/monomers/thrombin_cyclic_sweep.json and cycpept_len6_pampa_sweep.json, random "
        "forest arm, five draws a size, bars are the draw-to-draw deviation of the "
        "real-descriptor arm and not an interval. Endpoint is confounded with library "
        "design, alphabet size and source heterogeneity: thrombin is one laboratory's "
        "designed library over 30 blocks, CycPeptMPDB is 22 laboratories' literature "
        "compounds over 118. A hexamer's bulk logP and polar surface area are close to the "
        "sum of its monomers', so a per-monomer descriptor transfers nearly by construction "
        "for a physicochemical endpoint in a way it need not for molecular recognition.")
    save(fig, f"monomer_transfer{'_dark' if mode == 'dark' else ''}", c)


FIGURES = {1: figure_two_scales, 2: figure_repair, 3: figure_arms, 4: figure_transfer}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dark", action="store_true", help="render the dark-surface variant too")
    ap.add_argument("--only", type=int, choices=sorted(FIGURES), action="append",
                    help="draw only this figure; repeatable")
    args = ap.parse_args()

    if not FIT.exists():
        sys.exit(f"missing {FIT.relative_to(ROOT)} -- rebuild with "
                 f"`python -m hpc_pilots rubric-fit`")
    fit = json.loads(FIT.read_text())
    print(f"runs/rubric_fit.json  {fit['provenance']['recorded_at_utc']}  "
          f"commit {fit['provenance']['git_commit'][:9]}")

    wanted = sorted(set(args.only)) if args.only else sorted(FIGURES)
    for mode in ("light",) + (("dark",) if args.dark else ()):
        c = style(mode)
        for number in wanted:
            print(f"figure {number} ({mode})")
            draw = FIGURES[number]
            draw(c, mode) if draw is figure_transfer else draw(fit, c, mode)


if __name__ == "__main__":
    main()
